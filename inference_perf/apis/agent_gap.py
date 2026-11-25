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
"""Per-request record of the replay gap that preceded a request.

All timestamps are ``time.perf_counter()`` values, the same monotonic clock as
``RequestLifecycleMetric.start_time`` and streamed token times, so they can be
subtracted from each other across worker processes.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class GapKind(str, Enum):
    # No gap work (zero-length gap between dependent calls).
    NONE = "none"
    # A tool call: the gap is executed in the actor.
    TOOL = "tool"
    # User idle / think time: the actor may be suspended, then woken. Starts a turn.
    IDLE = "idle"


class AgentGapMetrics(BaseModel):
    gap_kind: GapKind
    executor: str
    # Recorded (capped) gap length from the trace.
    trace_wait_sec: float
    # When the gap began (predecessors done) and when the request became ready to send.
    gap_start: float
    gap_end: Optional[float] = None
    actor: Optional[str] = None
    # CreateActor latency, set on the gap that created the actor.
    actor_create_sec: Optional[float] = None
    # SuspendActor latency, set when this gap suspended the actor.
    suspend_sec: Optional[float] = None
    # Wake: when it was triggered and how long until the actor served.
    wake_start: Optional[float] = None
    wake_sec: Optional[float] = None
    # True when wake_sec is derived (implicit resume) rather than timed directly.
    wake_estimated: bool = False
    # True if the actor was actually restored; False if it was already running.
    resumed: Optional[bool] = None
    # Whether this wake restored from the template's golden snapshot (first resume).
    cold: Optional[bool] = None
    tool_exec_sec: Optional[float] = None
    tool_exit_code: Optional[int] = None
    error: Optional[str] = None


class AgentTurnMetric(BaseModel):
    """One agent turn: from the wake that starts it to the end of its last LLM call.

    A turn starts at the session's first request or at an idle gap and runs until
    the next idle gap. Latencies follow "Measuring Agent Sessions":
    Lturn = Lwake + sum(LTTTC_i + Ltool_exec_i), alpha_inf = sum(LTTTC_i) / Lturn.
    """

    turn_index: int
    # First turn of the session: the actor wakes from the template's golden snapshot.
    cold: bool
    first_event_id: Optional[str] = None
    # Turn start: when the wake was triggered (or the first request's gap began).
    start_time: float
    end_time: float
    turn_latency_sec: float
    wake_sec: Optional[float] = None
    resumed: Optional[bool] = None
    # Lwake-TTFT, measured: first output token of the turn's first LLM call minus start_time.
    wake_to_first_token_sec: Optional[float] = None
    # Client-side time between the actor serving and the LLM request being sent.
    harness_sec: Optional[float] = None
    # TTFT of the turn's first LLM call alone (engine + gateway).
    first_call_ttft_sec: Optional[float] = None
    num_llm_calls: int
    num_tool_calls: int
    # Serial sums; with parallel subagents they can exceed turn_latency_sec.
    sum_llm_latency_sec: float
    sum_tool_exec_sec: float
    sum_tool_wake_sec: float
    alpha_inf: Optional[float] = None
    tool_exec_sec: List[float] = []
    tool_wake_sec: List[float] = []
    success: bool
