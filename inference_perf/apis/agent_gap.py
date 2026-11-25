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
"""Records of the sandbox's work during session replay.

``AgentGapMetrics`` describes the gap that preceded one request, ``AgentTurnMetric``
one agent turn and ``SessionSandboxMetrics`` one session's sandbox activity.

All timestamps are ``time.perf_counter()`` values, the same monotonic clock as
``RequestLifecycleMetric.start_time`` and streamed token times, so they can be
subtracted from each other across worker processes.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class GapKind(str, Enum):
    # No gap work (an unclassified, zero-length gap between dependent calls).
    NONE = "none"
    # Tool calls: each recorded tool runs in the actor.
    TOOL = "tool"
    # User idle time: the actor may be suspended, then woken. Starts a turn.
    IDLE = "idle"


class ToolCallMetrics(BaseModel):
    """One replayed tool call, executed in the actor as ``sleep duration_sec``."""

    name: str
    # Planned start, relative to the gap start (the previous LLM call's end).
    start_sec: float
    # Part of the recorded tool window that falls after the previous LLM call ended.
    duration_sec: float
    # Round trip of the execution through the router, including the in-actor sleep.
    tool_exec_sec: Optional[float] = None
    # tool_exec_sec - duration_sec: router and sandbox overhead.
    tool_exec_overhead_sec: Optional[float] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None


class AgentGapMetrics(BaseModel):
    # Turn of the session this request belongs to (0-based); an idle gap starts the next one.
    turn_index: int = 0
    gap_kind: GapKind
    executor: str
    # Recorded gap length from the trace (after the replay cap, if any).
    trace_wait_sec: float
    # When the gap began (the previous LLM call ended) and when the request became ready to send.
    gap_start: float
    gap_end: Optional[float] = None
    actor: Optional[str] = None
    # CreateActor latency, set on the gap that created the actor.
    actor_create_sec: Optional[float] = None
    # SuspendActor latency, set when this gap suspended the actor.
    suspend_sec: Optional[float] = None
    # When the gap needed the actor (end of the idle time, or start of the tools).
    actor_ready_start: Optional[float] = None
    # From actor_ready_start until the actor serves: waiting out a background suspend
    # still in flight plus ResumeActor. None when the actor was already running.
    actor_ready_sec: Optional[float] = None
    # Whether the wake restored from the template's golden snapshot (first resume). None without a wake.
    cold: Optional[bool] = None
    tools: List[ToolCallMetrics] = []
    error: Optional[str] = None


class AgentTurnMetric(BaseModel):
    """One agent turn: from the moment it needs the actor to the end of its last LLM call.

    A turn starts at the session's first request or at an idle gap and runs until
    the next idle gap.
    """

    turn_index: int
    # The session's first turn: the actor wakes from the template's golden snapshot.
    cold: bool
    turn_latency_sec: float
    success: bool


class ActorReadySummary(BaseModel):
    cold: Optional[float] = None
    warm_mean: Optional[float] = None
    warm_max: Optional[float] = None


class SessionSandboxMetrics(BaseModel):
    """One session's sandbox activity, as counts and simple aggregates."""

    actors: List[str]
    actor_create_sec: Optional[float] = None
    num_wakes: int
    actor_ready_sec: ActorReadySummary
    num_suspends: int
    suspend_sec_mean: Optional[float] = None
    num_tool_calls: int
    tool_exec_overhead_sec_mean: Optional[float] = None
    tool_exec_overhead_sec_max: Optional[float] = None
    num_tool_errors: int
