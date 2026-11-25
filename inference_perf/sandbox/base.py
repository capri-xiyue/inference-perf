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
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Set

from inference_perf.apis.agent_gap import AgentGapMetrics, GapKind
from inference_perf.config.sandbox import (
    AGENT_ID_ATTRIBUTE,
    AGENT_NAME_ATTRIBUTE,
    ActorAssignment,
    ActorAssignmentConfig,
)


class ToolExecutionError(Exception):
    """A gap could not be executed; carries the partial metrics recorded so far."""

    def __init__(self, message: str, metrics: Optional[AgentGapMetrics] = None) -> None:
        super().__init__(message)
        self.metrics = metrics


@dataclass(frozen=True)
class PlannedTool:
    """A recorded tool call to replay, timed relative to the gap start."""

    name: str
    start_sec: float
    duration_sec: float


@dataclass(frozen=True)
class GapContext:
    # Enclosing replay session (scheduling identity).
    session_id: str
    # Which actor runs this gap (see resolve_actor_key).
    actor_key: str
    event_id: str
    # Recorded gap length after the replay cap (max_wait_ms).
    wait_sec: float
    gap_kind: GapKind
    # Tool calls to execute in a tool gap.
    tools: Sequence[PlannedTool] = ()


def parse_planned_tools(value: object) -> List[PlannedTool]:
    """Read the gap's tool list from its call attribute (see config.sandbox.GAP_TOOLS_ATTRIBUTE)."""
    tools: List[PlannedTool] = []
    if not isinstance(value, list):
        return tools
    for item in value:
        if not isinstance(item, dict):
            continue
        start = item.get("start_sec", 0.0)
        duration = item.get("duration_sec", 0.0)
        tools.append(
            PlannedTool(
                name=str(item.get("name", "unknown")),
                start_sec=max(float(start), 0.0) if isinstance(start, (int, float)) else 0.0,
                duration_sec=max(float(duration), 0.0) if isinstance(duration, (int, float)) else 0.0,
            )
        )
    return tools


def resolve_actor_key(assignment: ActorAssignmentConfig, session_id: str, attributes: Mapping[str, object]) -> str:
    """Pick the actor for a call's gap from its attributes.

    The session's shared actor unless a rule assigns a dedicated one, keyed by the
    agent instance (gen_ai.agent.id, else gen_ai.agent.name) within the session, so
    every stream of that agent shares it.
    """
    if assignment.resolve(attributes) == ActorAssignment.DEDICATED:
        agent = attributes.get(AGENT_ID_ATTRIBUTE) or attributes.get(AGENT_NAME_ATTRIBUTE)
        if agent:
            return f"{session_id}::agent:{agent}"
    return session_id


def classify_gap(*, has_predecessors: bool, recorded_kind: Optional[str]) -> GapKind:
    """What the gap before an event stands for.

    The session's first request starts a turn (the actor is created and woken);
    later gaps use the kind the trace replay recorded (agent.gap.kind). An
    unclassified gap is replayed as a plain wait.
    """
    if not has_predecessors:
        return GapKind.IDLE
    if recorded_kind == GapKind.TOOL.value:
        return GapKind.TOOL
    if recorded_kind == GapKind.IDLE.value:
        return GapKind.IDLE
    return GapKind.NONE


class ToolExecutor(ABC):
    """Runs replay gaps. One instance per worker process; not picklable."""

    name: str = "base"

    def __init__(self) -> None:
        self._background: Set["asyncio.Task[None]"] = set()

    def schedule_end_session(self, session_id: str) -> None:
        """Release the session's actors in the background (callable from sync code on the loop)."""
        task = asyncio.get_running_loop().create_task(self.end_session(session_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def drain_background(self) -> None:
        if self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    @abstractmethod
    async def run_gap(self, ctx: GapContext) -> AgentGapMetrics:
        """Execute the gap and return its metrics. Raises ToolExecutionError on failure."""

    @abstractmethod
    async def end_session(self, session_id: str) -> None:
        """Release every actor the session used."""

    @abstractmethod
    async def aclose(self) -> None:
        """Release every remaining actor and close connections."""
