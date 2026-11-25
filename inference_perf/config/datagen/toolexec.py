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
"""Configuration for executing replayed tool gaps on an agent sandbox runtime.

During session replay, the gap between a predecessor's completion and an event's
dispatch stands for tool execution or user think time. By default that gap is a
plain sleep. A ``tool_executor`` config replaces the sleep with real work on an
agent runtime (GKE Agent Substrate actors), so the benchmark also measures actor
wake, tool execution and wake-to-first-token latency.
"""

import fnmatch
from enum import Enum
from typing import Dict, List, Mapping, Optional

from pydantic import Field, model_validator

from inference_perf.config.common import StrictBaseModel


class ToolExecutorType(str, Enum):
    # Real GKE Agent Substrate: ateapi gRPC control plane + atenet-router data plane.
    SUBSTRATE = "substrate"
    # In-process stand-in with configurable latencies; no cluster required.
    FAKE_SUBSTRATE = "fake_substrate"


# Call attributes that identify the agent issuing an LLM call (OTel GenAI semantic
# conventions). OTel traces carry them on their spans; weka trace replay sets them
# on subagent calls (name = subagent_type, id = agent_id).
AGENT_ID_ATTRIBUTE = "gen_ai.agent.id"
AGENT_NAME_ATTRIBUTE = "gen_ai.agent.name"

# Call attributes that describe the gap before an LLM call, when the trace records
# it (e.g. TraceLab timing_events). GAP_KIND_ATTRIBUTE is "tool" or "idle" and takes
# precedence over gap-length heuristics; the tool attributes are reported.
GAP_KIND_ATTRIBUTE = "agent.gap.kind"
TOOL_NAME_ATTRIBUTE = "gen_ai.tool.name"
TOOL_LATENCY_ATTRIBUTE = "agent.tool.latency_ms"


class ActorAssignment(str, Enum):
    # Run the gap on the session's shared actor (Claude Code style: subagents share
    # their parent's machine).
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
    actor: ActorAssignment = Field(..., description="Where gaps of matching calls run.")


class ActorAssignmentConfig(StrictBaseModel):
    """Which actor runs each gap, chosen per LLM call from its attributes."""

    default: ActorAssignment = Field(ActorAssignment.SESSION, description="Assignment when no rule matches.")
    rules: List[ActorAssignmentRule] = Field(default_factory=list, description="Ordered rules; the first match wins.")

    def resolve(self, attributes: Mapping[str, object]) -> ActorAssignment:
        for rule in self.rules:
            if all(
                key in attributes and fnmatch.fnmatchcase(str(attributes[key]), pattern) for key, pattern in rule.match.items()
            ):
                return rule.actor
        return self.default


class SnapshotDurability(str, Enum):
    # SuspendActor returns once the snapshot is on the node; upload to GCS continues in the background.
    ASYNC = "async"
    # SuspendActor returns only after the snapshot is in durable storage.
    SYNC = "sync"


class SuspendActorConfig(StrictBaseModel):
    """When the actor is suspended. The two moments are independent."""

    between_turns: bool = Field(
        True, description="Suspend during user idle time between turns; the next turn wakes the actor."
    )
    after_each_tool_call: bool = Field(
        False,
        description="Suspend after every tool call, while the next LLM call runs; the next tool call wakes the actor.",
    )


class FakeSubstrateConfig(StrictBaseModel):
    """Latency model for the in-process fake Substrate backend."""

    create_latency_ms: float = Field(20.0, ge=0, description="Latency of CreateActor.")
    cold_resume_latency_ms: float = Field(
        1000.0, ge=0, description="Latency of the first resume of an actor, restored from the template's golden snapshot."
    )
    warm_resume_latency_ms: float = Field(400.0, ge=0, description="Latency of later resumes from the actor's own snapshot.")
    suspend_latency_ms: float = Field(200.0, ge=0, description="Latency of SuspendActor.")
    delete_latency_ms: float = Field(10.0, ge=0, description="Latency of DeleteActor.")
    exec_overhead_ms: float = Field(5.0, ge=0, description="Router and sandbox overhead added to every tool execution.")
    jitter_fraction: float = Field(0.2, ge=0, le=1, description="Uniform +/- jitter applied to every simulated latency.")


class SubstrateConfig(StrictBaseModel):
    """Connection settings for a GKE Agent Substrate installation."""

    atespace: str = Field(..., description="Atespace that holds the actors.")
    actor_template: str = Field(..., description="ActorTemplate (in the same atespace) every actor is created from.")
    actor_name_prefix: str = Field("ip", description="Prefix for generated actor names.")
    api_endpoint: str = Field("api.ate-system.svc:443", description="ateapi gRPC endpoint (host:port).")
    api_server_name: str = Field("api.ate-system.svc", description="TLS server name ateapi presents.")
    api_ca_path: Optional[str] = Field(
        None, description="CA bundle that verifies ateapi (servicedns.podcert.ate.dev/identity ClusterTrustBundle)."
    )
    api_token_path: Optional[str] = Field(
        None, description="File holding a bearer JWT for ateapi, e.g. a projected SA token with audience api.ate-system.svc."
    )
    api_cert_bundle_path: Optional[str] = Field(
        None, description="Pod certificate credential bundle (key + chain PEM) for mTLS to ateapi."
    )
    router_url: str = Field(
        "https://atenet-router.ate-system.svc", description="atenet-router base URL for actor data-plane traffic."
    )
    router_ca_path: Optional[str] = Field(None, description="CA bundle that verifies atenet-router.")
    router_cert_bundle_path: Optional[str] = Field(
        None, description="Pod certificate credential bundle (key + chain PEM) for mTLS to atenet-router."
    )
    execute_path: str = Field(
        "/execute", description="Path of the tool runtime's execute endpoint inside the actor (agent-sandbox runtime API)."
    )
    request_timeout_sec: float = Field(120.0, gt=0, description="Timeout for a single control or data plane call.")


class ToolExecutorConfig(StrictBaseModel):
    """Run replayed tool gaps on agent sandbox actors instead of sleeping."""

    type: ToolExecutorType = Field(ToolExecutorType.FAKE_SUBSTRATE, description="Tool executor backend.")
    idle_gap_threshold_ms: int = Field(
        5000,
        ge=0,
        description=(
            "A gap at least this long, with no role:tool message to classify it, is treated as user idle time: the "
            "actor is suspended during the gap and woken at its end (a new turn). Shorter non-zero gaps are tool calls."
        ),
    )
    actor_assignment: ActorAssignmentConfig = Field(
        default_factory=ActorAssignmentConfig,
        description=(
            "Which actor runs each gap. By default every call, subagents included, uses the session's shared "
            "actor; rules give selected agents a dedicated actor."
        ),
    )
    suspend_actor: SuspendActorConfig = Field(
        default_factory=SuspendActorConfig, description="When the actor is suspended: between turns, after each tool call."
    )
    snapshot_durability: SnapshotDurability = Field(
        SnapshotDurability.ASYNC, description="When SuspendActor returns relative to its snapshot reaching durable storage."
    )
    tool_command_template: str = Field(
        "sleep {duration_sec:.3f}",
        description="Command run in the actor for a tool gap. {duration_sec} is the recorded (capped) gap length.",
    )
    tool_timeout_sec: float = Field(300.0, gt=0, description="Timeout for one tool execution.")
    delete_actors_on_session_end: bool = Field(True, description="Delete a session's actors once it drains.")
    fake: FakeSubstrateConfig = Field(default_factory=FakeSubstrateConfig, description="Fake backend latency model.")
    substrate: Optional[SubstrateConfig] = Field(None, description="Substrate connection settings (type=substrate).")

    @model_validator(mode="after")
    def validate_backend(self) -> "ToolExecutorConfig":
        if self.type == ToolExecutorType.SUBSTRATE and self.substrate is None:
            raise ValueError("tool_executor.substrate is required when tool_executor.type is 'substrate'")
        try:
            self.tool_command_template.format(duration_sec=0.0)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"invalid tool_command_template: {e}") from e
        return self
