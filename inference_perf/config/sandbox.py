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
"""Configuration of the sandbox that executes replayed tool calls.

``server`` is the LLM under test; ``sandbox`` is the tool-execution side under
test. When set, TraceLab replay runs each recorded tool call in a sandbox actor
(GKE Agent Substrate) instead of sleeping through the gap, and reports turn
latency plus the sandbox's own overheads (actor wake, suspend, tool dispatch).
"""

import fnmatch
from enum import Enum
from typing import Dict, List, Mapping, Optional

from pydantic import Field, model_validator

from inference_perf.config.common import StrictBaseModel


class SandboxType(str, Enum):
    # Real GKE Agent Substrate: ateapi gRPC control plane + atenet-router data plane.
    SUBSTRATE = "substrate"
    # In-process stand-in with configurable latencies; no cluster required.
    FAKE_SUBSTRATE = "fake_substrate"


# Call attributes that identify the agent issuing an LLM call (OTel GenAI semantic
# conventions). Actor assignment rules give a matching agent a dedicated actor.
AGENT_ID_ATTRIBUTE = "gen_ai.agent.id"
AGENT_NAME_ATTRIBUTE = "gen_ai.agent.name"

# Call attributes the trace replay sets for the sandbox, describing the gap before
# the call: its kind ("tool" or "idle") and the tool calls to execute in it, as a
# list of {"name", "start_sec", "duration_sec"} relative to the gap start (the
# previous LLM call's end).
GAP_KIND_ATTRIBUTE = "agent.gap.kind"
GAP_TOOLS_ATTRIBUTE = "agent.gap.tools"


class ActorAssignment(str, Enum):
    # Run the gap on the session's shared actor (subagents share their parent's machine).
    SESSION = "session"
    # Run the gap on an actor of its own, one per agent instance (gen_ai.agent.id,
    # else gen_ai.agent.name). Calls with neither fall back to the session actor.
    DEDICATED = "dedicated"


class ActorAssignmentRule(StrictBaseModel):
    match: Dict[str, str] = Field(
        ...,
        min_length=1,
        description=(
            "Call attributes to match, as glob patterns (fnmatch). All must match; a missing attribute never "
            "matches. E.g. {'gen_ai.agent.name': 'Explore'}."
        ),
    )
    assign: ActorAssignment = Field(..., description="Where gaps of matching calls run.")


class ActorAssignmentConfig(StrictBaseModel):
    """Which actor runs each gap, chosen per LLM call from its attributes."""

    default: ActorAssignment = Field(ActorAssignment.SESSION, description="Assignment when no rule matches.")
    rules: List[ActorAssignmentRule] = Field(default_factory=list, description="Ordered rules; the first match wins.")

    def resolve(self, attributes: Mapping[str, object]) -> ActorAssignment:
        for rule in self.rules:
            if all(
                key in attributes and fnmatch.fnmatchcase(str(attributes[key]), pattern) for key, pattern in rule.match.items()
            ):
                return rule.assign
        return self.default


class SnapshotDurability(str, Enum):
    # SuspendActor returns once the snapshot is on the node; upload to GCS continues in the background.
    ASYNC = "async"
    # SuspendActor returns only after the snapshot is in durable storage.
    SYNC = "sync"


class ActorSuspendConfig(StrictBaseModel):
    """When the actor is suspended and how. The two moments are independent."""

    between_turns: bool = Field(
        True, description="Suspend during user idle time between turns; the next turn wakes the actor."
    )
    after_each_tool_call: bool = Field(
        False,
        description="Suspend after every tool gap, while the next LLM call runs; the next tool gap wakes the actor.",
    )
    snapshot_durability: SnapshotDurability = Field(
        SnapshotDurability.ASYNC, description="When SuspendActor returns relative to its snapshot reaching durable storage."
    )


class ActorToolCallConfig(StrictBaseModel):
    """How a replayed tool call runs in the actor (always ``sleep <recorded duration>``)."""

    timeout_sec: float = Field(300.0, gt=0, description="Timeout for one tool execution.")


class ActorConfig(StrictBaseModel):
    """Backend-independent actor behavior: naming, who uses which actor, suspend, tool calls, cleanup."""

    name_prefix: str = Field("ip", description="Prefix for generated actor names.")
    assignment: ActorAssignmentConfig = Field(
        default_factory=ActorAssignmentConfig,
        description=(
            "Which actor runs each gap. By default every call uses the session's shared actor; rules give "
            "selected agents a dedicated actor."
        ),
    )
    suspend: ActorSuspendConfig = Field(default_factory=ActorSuspendConfig, description="When and how actors are suspended.")
    tool_call: ActorToolCallConfig = Field(
        default_factory=ActorToolCallConfig, description="How a replayed tool call runs in the actor."
    )
    delete_on_session_end: bool = Field(True, description="Delete a session's actors once it drains.")


class FakeSubstrateConfig(StrictBaseModel):
    """Latency model for the in-process fake Substrate backend.

    Defaults are rounded from a real Substrate v1.0.0 run (gVisor on c3-standard-4 workers).
    """

    create_latency_ms: float = Field(25.0, ge=0, description="Latency of CreateActor.")
    cold_resume_latency_ms: float = Field(
        200.0, ge=0, description="Latency of the first resume of an actor, restored from the template's golden snapshot."
    )
    warm_resume_latency_ms: float = Field(140.0, ge=0, description="Latency of later resumes from the actor's own snapshot.")
    snapshot_capture_latency_ms: float = Field(
        85.0, ge=0, description="SuspendActor time to capture the snapshot on the node; all an async suspend waits for."
    )
    snapshot_upload_latency_ms: float = Field(
        350.0, ge=0, description="Extra SuspendActor time to upload the snapshot to durable storage; paid only by sync."
    )
    delete_latency_ms: float = Field(10.0, ge=0, description="Latency of DeleteActor.")
    exec_overhead_ms: float = Field(25.0, ge=0, description="Router and sandbox overhead added to every tool execution.")
    jitter_fraction: float = Field(0.2, ge=0, le=1, description="Uniform +/- jitter applied to every simulated latency.")


class SubstrateApiConfig(StrictBaseModel):
    """ateapi gRPC control plane."""

    endpoint: str = Field("api.ate-system.svc:443", description="ateapi gRPC endpoint (host:port).")
    server_name: str = Field("api.ate-system.svc", description="TLS server name ateapi presents.")
    ca_path: Optional[str] = Field(
        None, description="CA bundle that verifies ateapi (servicedns.podcert.ate.dev/identity ClusterTrustBundle)."
    )
    token_path: Optional[str] = Field(
        None, description="File holding a bearer JWT for ateapi, e.g. a projected SA token with audience api.ate-system.svc."
    )
    cert_bundle_path: Optional[str] = Field(
        None, description="Pod certificate credential bundle (key + chain PEM) for mTLS to ateapi."
    )


class SubstrateRouterConfig(StrictBaseModel):
    """atenet-router data plane (traffic to actors)."""

    url: str = Field("https://atenet-router.ate-system.svc", description="atenet-router base URL.")
    execute_path: str = Field(
        "/execute", description="Path of the tool runtime's execute endpoint inside the actor (agent-sandbox runtime API)."
    )
    ca_path: Optional[str] = Field(None, description="CA bundle that verifies atenet-router.")
    cert_bundle_path: Optional[str] = Field(
        None, description="Pod certificate credential bundle (key + chain PEM) for mTLS to atenet-router."
    )


class SubstrateConfig(StrictBaseModel):
    """GKE Agent Substrate specifics: where actors live and how to reach the installation."""

    atespace: str = Field(..., description="Atespace that holds the actors.")
    actor_template: str = Field(..., description="ActorTemplate (in the same atespace) every actor is created from.")
    api: SubstrateApiConfig = Field(default_factory=SubstrateApiConfig, description="Control plane connection.")
    router: SubstrateRouterConfig = Field(default_factory=SubstrateRouterConfig, description="Data plane connection.")
    request_timeout_sec: float = Field(120.0, gt=0, description="Timeout for a single control or data plane call.")


class SandboxConfig(StrictBaseModel):
    """The sandbox that executes replayed tool calls (supported with tracelab_trace_replay)."""

    type: SandboxType = Field(SandboxType.FAKE_SUBSTRATE, description="Sandbox backend.")
    actor: ActorConfig = Field(default_factory=ActorConfig, description="Backend-independent actor behavior.")
    substrate: Optional[SubstrateConfig] = Field(None, description="Substrate settings (required for type=substrate).")
    fake: FakeSubstrateConfig = Field(default_factory=FakeSubstrateConfig, description="Fake backend latency model.")

    @model_validator(mode="after")
    def validate_backend(self) -> "SandboxConfig":
        if self.type == SandboxType.SUBSTRATE and self.substrate is None:
            raise ValueError("sandbox.substrate is required when sandbox.type is 'substrate'")
        return self
