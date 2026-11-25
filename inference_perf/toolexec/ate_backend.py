# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SubstrateBackend for a real GKE Agent Substrate installation.

Control plane: gRPC ``substrate.ateapi.v1.Control`` (CreateActor, ResumeActor,
SuspendActor, DeleteActor), authenticated with a bearer JWT and/or a pod
certificate. Data plane: HTTP to ``atenet-router`` with the
``ate-target-actor: <atespace>/<actor>`` header; the router resumes a suspended
actor before forwarding (request parking). The actor is expected to serve the
agent-sandbox runtime API (``POST /execute {"command": ...}``) on its default
ingress port.

Only the handful of ateapi messages used here are declared, by field number,
in a private descriptor pool (wire-compatible with ateapi v1); this avoids
vendoring generated stubs for the whole API.
"""

import logging
import re
import ssl
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import aiohttp
from google.protobuf.message import Message

from inference_perf.config.datagen.toolexec import ActorConfig, SubstrateConfig
from inference_perf.toolexec.substrate_executor import ExecResult, SubstrateBackend

if TYPE_CHECKING:
    import grpc.aio

logger = logging.getLogger(__name__)

_PACKAGE = "substrate.ateapi.v1"
_SERVICE = f"/{_PACKAGE}.Control"

_PEM_BLOCK = re.compile(rb"-----BEGIN (?P<kind>[A-Z ]+)-----.*?-----END (?P=kind)-----\n?", re.DOTALL)

# (name, [(field_name, number, type, type_name)]) for the ateapi v1 subset used here.
# type is a FieldDescriptorProto.Type name; type_name is set for message/enum fields.
_MESSAGES: List[Tuple[str, List[Tuple[str, int, str, Optional[str]]]]] = [
    ("ObjectRef", [("atespace", 1, "TYPE_STRING", None), ("name", 2, "TYPE_STRING", None)]),
    (
        "ResourceMetadata",
        [
            ("atespace", 1, "TYPE_STRING", None),
            ("name", 2, "TYPE_STRING", None),
            ("uid", 3, "TYPE_STRING", None),
            ("version", 4, "TYPE_INT64", None),
        ],
    ),
    ("ActorStatus", [("state", 1, "TYPE_ENUM", "ActorState")]),
    (
        "Actor",
        [
            ("metadata", 1, "TYPE_MESSAGE", "ResourceMetadata"),
            ("actor_template", 2, "TYPE_MESSAGE", "ObjectRef"),
            ("status", 5, "TYPE_MESSAGE", "ActorStatus"),
        ],
    ),
    ("CreateActorRequest", [("actor", 1, "TYPE_MESSAGE", "Actor")]),
    ("ResumeActorRequest", [("actor", 1, "TYPE_MESSAGE", "ObjectRef")]),
    ("ResumeActorResponse", [("actor", 1, "TYPE_MESSAGE", "Actor"), ("resumed", 2, "TYPE_BOOL", None)]),
    ("SuspendOptions", [("mode", 1, "TYPE_ENUM", "DurabilityMode")]),
    (
        "SuspendActorRequest",
        [("actor", 1, "TYPE_MESSAGE", "ObjectRef"), ("options", 2, "TYPE_MESSAGE", "SuspendOptions")],
    ),
    ("SuspendActorResponse", [("actor", 1, "TYPE_MESSAGE", "Actor")]),
    ("DeleteActorRequest", [("actor", 1, "TYPE_MESSAGE", "ObjectRef")]),
]
_ENUMS: List[Tuple[str, List[Tuple[str, int]]]] = [
    (
        "ActorState",
        [
            ("ACTOR_STATE_UNSPECIFIED", 0),
            ("ACTOR_STATE_RESUMING", 1),
            ("ACTOR_STATE_RUNNING", 2),
            ("ACTOR_STATE_SUSPENDING", 3),
            ("ACTOR_STATE_SUSPENDED", 4),
            ("ACTOR_STATE_CRASHED", 5),
            ("ACTOR_STATE_DELETING", 6),
            ("ACTOR_STATE_REVERTING", 7),
        ],
    ),
    (
        "DurabilityMode",
        [("DURABILITY_MODE_UNSPECIFIED", 0), ("DURABILITY_MODE_SYNCHRONOUS", 1), ("DURABILITY_MODE_ASYNCHRONOUS", 2)],
    ),
]


def build_ateapi_messages() -> Dict[str, type[Message]]:
    """Build message classes for the ateapi v1 subset, keyed by short name."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    fdp = descriptor_pb2.FileDescriptorProto(
        name="inference_perf/toolexec/ateapi_v1_subset.proto", package=_PACKAGE, syntax="proto3"
    )
    for enum_name, values in _ENUMS:
        enum = fdp.enum_type.add(name=enum_name)
        for value_name, number in values:
            enum.value.add(name=value_name, number=number)
    for message_name, fields in _MESSAGES:
        message = fdp.message_type.add(name=message_name)
        for field_name, number, type_name, ref in fields:
            f = message.field.add(
                name=field_name,
                number=number,
                label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
                type=descriptor_pb2.FieldDescriptorProto.Type.Value(type_name),
            )
            if ref is not None:
                f.type_name = f".{_PACKAGE}.{ref}"
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return {name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{_PACKAGE}.{name}")) for name, _ in _MESSAGES}


def split_credential_bundle(bundle: bytes) -> Tuple[bytes, bytes]:
    """Split a Kubernetes pod-certificate bundle (PRIVATE KEY + CERTIFICATE blocks) into (key, chain)."""
    key: Optional[bytes] = None
    chain: List[bytes] = []
    for m in _PEM_BLOCK.finditer(bundle):
        if m.group("kind").endswith(b"PRIVATE KEY"):
            key = m.group(0)
        elif m.group("kind") == b"CERTIFICATE":
            chain.append(m.group(0))
    if key is None or not chain:
        raise ValueError("credential bundle must hold a PRIVATE KEY block and at least one CERTIFICATE block")
    return key, b"".join(chain)


def _serialize(message: Message) -> bytes:
    # Bound per instance: Message.SerializeToString on the base class is abstract.
    data: bytes = message.SerializeToString()
    return data


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class AteSubstrateBackend(SubstrateBackend):
    def __init__(self, config: SubstrateConfig, actor: ActorConfig) -> None:
        if not actor.atespace or not actor.template:
            raise ValueError("actor.atespace and actor.template are required for the Substrate backend")
        self.config = config
        self.atespace: str = actor.atespace
        self.template: str = actor.template
        self.execute_path = actor.tool_call.execute_path
        self._msgs = build_ateapi_messages()
        self._channel: Optional["grpc.aio.Channel"] = None
        self._http: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------ connections

    def _get_channel(self) -> "grpc.aio.Channel":
        if self._channel is None:
            import grpc

            root = _read(self.config.api.ca_path) if self.config.api.ca_path else None
            key: Optional[bytes] = None
            chain: Optional[bytes] = None
            if self.config.api.cert_bundle_path:
                key, chain = split_credential_bundle(_read(self.config.api.cert_bundle_path))
            creds = grpc.ssl_channel_credentials(root_certificates=root, private_key=key, certificate_chain=chain)
            options = [("grpc.ssl_target_name_override", self.config.api.server_name)]
            self._channel = grpc.aio.secure_channel(self.config.api.endpoint, creds, options=options)
        return self._channel

    def _metadata(self) -> List[Tuple[str, str]]:
        # Projected tokens rotate, so read the file on every call.
        if self.config.api.token_path:
            token = _read(self.config.api.token_path).decode("utf-8").strip()
            return [("authorization", f"Bearer {token}")]
        return []

    async def _call(self, method: str, request: Message, response_type: str) -> Message:
        rpc = self._get_channel().unary_unary(
            f"{_SERVICE}/{method}",
            request_serializer=_serialize,
            response_deserializer=self._msgs[response_type].FromString,
        )
        return await rpc(request, metadata=self._metadata(), timeout=self.config.request_timeout_sec)

    def _get_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            ctx = ssl.create_default_context(cafile=self.config.router.ca_path)
            if self.config.router.cert_bundle_path:
                # Python's ssl reads the key from the same PEM as the chain.
                ctx.load_cert_chain(self.config.router.cert_bundle_path)
            timeout = aiohttp.ClientTimeout(total=None)
            self._http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx, limit=0), timeout=timeout)
        return self._http

    def _ref(self, name: str) -> Message:
        return self._msgs["ObjectRef"](atespace=self.atespace, name=name)

    # ------------------------------------------------------------------ SubstrateBackend

    async def create_actor(self, name: str) -> None:
        actor = self._msgs["Actor"](
            metadata=self._msgs["ResourceMetadata"](atespace=self.atespace, name=name),
            actor_template=self._msgs["ObjectRef"](atespace=self.atespace, name=self.template),
        )
        await self._call("CreateActor", self._msgs["CreateActorRequest"](actor=actor), "Actor")

    async def resume_actor(self, name: str) -> bool:
        resp = await self._call("ResumeActor", self._msgs["ResumeActorRequest"](actor=self._ref(name)), "ResumeActorResponse")
        return bool(resp.resumed)

    async def suspend_actor(self, name: str, synchronous: bool) -> None:
        mode = 1 if synchronous else 2  # DURABILITY_MODE_SYNCHRONOUS / _ASYNCHRONOUS
        request = self._msgs["SuspendActorRequest"](actor=self._ref(name), options=self._msgs["SuspendOptions"](mode=mode))
        await self._call("SuspendActor", request, "SuspendActorResponse")

    async def delete_actor(self, name: str) -> None:
        await self._call("DeleteActor", self._msgs["DeleteActorRequest"](actor=self._ref(name)), "Actor")

    async def execute(self, name: str, command: str, timeout_sec: float) -> ExecResult:
        url = self.config.router.url.rstrip("/") + self.execute_path
        headers = {"ate-target-actor": f"{self.atespace}/{name}"}
        async with self._get_http().post(
            url, json={"command": command}, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_sec)
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"execute in actor {name} returned HTTP {resp.status}: {body[:300]}")
            data = await resp.json(content_type=None)
        exit_code = data.get("exit_code", 0) if isinstance(data, dict) else 0
        return ExecResult(exit_code=int(exit_code) if exit_code is not None else 0)

    async def aclose(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None
        if self._channel is not None:
            await self._channel.close()
        self._channel = None
