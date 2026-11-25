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
"""AteSubstrateBackend wire format and calls, against in-process gRPC and HTTP fakes."""

from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Tuple

import aiohttp
import grpc
import pytest
from aiohttp import web
from google.protobuf.message import Message

from inference_perf.config.sandbox import SubstrateConfig
from inference_perf.sandbox.ate_backend import AteSubstrateBackend, build_ateapi_messages, split_credential_bundle


class TestWireFormat:
    """Field numbers and types must match substrate.ateapi.v1 (api/ateapi/v1/ateapi.proto)."""

    def test_create_actor_request_encoding(self) -> None:
        m = build_ateapi_messages()
        req = m["CreateActorRequest"](
            actor=m["Actor"](
                metadata=m["ResourceMetadata"](atespace="a", name="n"),
                actor_template=m["ObjectRef"](atespace="a", name="t"),
            )
        )
        md = bytes.fromhex("0a016112016e")  # atespace=1, name=2
        tpl = bytes.fromhex("0a0161120174")
        actor = b"\x0a\x06" + md + b"\x12\x06" + tpl  # metadata=1, actor_template=2
        assert req.SerializeToString() == b"\x0a\x10" + actor  # CreateActorRequest.actor=1

    def test_suspend_request_encoding(self) -> None:
        m = build_ateapi_messages()
        req = m["SuspendActorRequest"](actor=m["ObjectRef"](atespace="a", name="n"), options=m["SuspendOptions"](mode=1))
        # actor=1 ObjectRef, options=2 {mode=1: DURABILITY_MODE_SYNCHRONOUS}
        assert req.SerializeToString() == bytes.fromhex("0a060a016112016e12020801")

    def test_resume_response_decoding_ignores_unknown_fields(self) -> None:
        m = build_ateapi_messages()
        # resumed=2 true, plus an unknown field 9 the subset does not declare.
        resp = m["ResumeActorResponse"].FromString(bytes.fromhex("10014a0178"))
        assert resp.resumed is True

    def test_split_credential_bundle(self) -> None:
        key = b"-----BEGIN PRIVATE KEY-----\nAAA\n-----END PRIVATE KEY-----\n"
        leaf = b"-----BEGIN CERTIFICATE-----\nBBB\n-----END CERTIFICATE-----\n"
        root = b"-----BEGIN CERTIFICATE-----\nCCC\n-----END CERTIFICATE-----\n"
        got_key, chain = split_credential_bundle(key + leaf + root)
        assert got_key == key and chain == leaf + root
        with pytest.raises(ValueError):
            split_credential_bundle(leaf)


class _FakeAteApi:
    """Generic gRPC handler for substrate.ateapi.v1.Control that records calls."""

    def __init__(self) -> None:
        self.msgs = build_ateapi_messages()
        self.calls: List[Tuple[str, Message, Dict[str, str]]] = []

    def handler(self) -> grpc.GenericRpcHandler:
        m = self.msgs
        methods: Dict[str, Tuple[str, Callable[[Message], Message]]] = {
            "CreateActor": ("CreateActorRequest", lambda req: req.actor),
            "ResumeActor": ("ResumeActorRequest", lambda req: m["ResumeActorResponse"](resumed=True)),
            "SuspendActor": ("SuspendActorRequest", lambda req: m["SuspendActorResponse"]()),
            "DeleteActor": ("DeleteActorRequest", lambda req: m["Actor"]()),
        }
        rpc_handlers = {}
        for name, (req_type, respond) in methods.items():

            def make(
                name: str = name, respond: Callable[[Message], Message] = respond
            ) -> Callable[[Message, grpc.aio.ServicerContext], Awaitable[Message]]:
                async def unary(request: Message, context: grpc.aio.ServicerContext) -> Message:
                    self.calls.append((name, request, dict(context.invocation_metadata() or ())))
                    return respond(request)

                return unary

            rpc_handlers[name] = grpc.unary_unary_rpc_method_handler(
                make(),
                request_deserializer=m[req_type].FromString,
                response_serializer=lambda r: r.SerializeToString(),
            )
        return grpc.method_handlers_generic_handler("substrate.ateapi.v1.Control", rpc_handlers)


async def test_backend_calls_control_plane_and_router(tmp_path: Path) -> None:
    api = _FakeAteApi()
    server = grpc.aio.server()
    server.add_generic_rpc_handlers((api.handler(),))
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    executed: List[Tuple[str, Dict[str, object]]] = []

    async def execute(request: web.Request) -> web.Response:
        executed.append((request.headers["ate-target-actor"], await request.json()))
        return web.json_response({"stdout": "", "stderr": "", "exit_code": 3})

    app = web.Application()
    app.router.add_post("/execute", execute)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    http_port = int(runner.addresses[0][1])

    token = tmp_path / "token"
    token.write_text("jwt-123\n")
    backend = AteSubstrateBackend(
        SubstrateConfig.model_validate(
            {
                "atespace": "bench",
                "actor_template": "python-runtime",
                "api": {"token_path": str(token)},
                "router": {"url": f"http://127.0.0.1:{http_port}"},
            }
        )
    )
    # The real backend dials TLS; point it at the plaintext fakes instead.
    backend._channel = grpc.aio.insecure_channel(f"127.0.0.1:{grpc_port}")
    backend._http = aiohttp.ClientSession()
    try:
        await backend.create_actor("ip-1")
        assert await backend.resume_actor("ip-1") is True
        result = await backend.execute("ip-1", "sleep 0.1", timeout_sec=5)
        await backend.suspend_actor("ip-1", synchronous=False)
        await backend.delete_actor("ip-1")
    finally:
        await backend.aclose()
        await runner.cleanup()
        await server.stop(None)

    assert [name for name, _, _ in api.calls] == ["CreateActor", "ResumeActor", "SuspendActor", "DeleteActor"]
    create_req = api.calls[0][1]
    assert (create_req.actor.metadata.atespace, create_req.actor.metadata.name) == ("bench", "ip-1")
    assert (create_req.actor.actor_template.atespace, create_req.actor.actor_template.name) == ("bench", "python-runtime")
    assert api.calls[2][1].options.mode == 2  # asynchronous
    assert all(md.get("authorization") == "Bearer jwt-123" for _, _, md in api.calls)
    assert executed == [("bench/ip-1", {"command": "sleep 0.1"})]
    assert result.exit_code == 3
