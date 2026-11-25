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
"""Agent-session metrics from sandbox replay runs.

Built from the per-request ``agent_gap`` records:

- turns: a turn starts at a session's first request or at an idle gap and runs
  until the next idle gap. Its latency runs from the moment it needs the actor
  (end of the user's idle time) to the end of its last LLM call.
- per-session sandbox activity: counts and simple aggregates.
- the run summary: turn latency split into cold (first turn of a session) and
  warm turns, and the distributions of the sandbox's own overheads.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, TypedDict

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.apis.agent_gap import (
    ActorReadySummary,
    AgentGapMetrics,
    AgentTurnMetric,
    SessionSandboxMetrics,
    ToolCallMetrics,
)


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _tool_failed(tool: ToolCallMetrics) -> bool:
    return tool.error is not None or (tool.exit_code is not None and tool.exit_code != 0)


def _gaps(requests: Sequence[RequestLifecycleMetric]) -> List[AgentGapMetrics]:
    return [r.info.agent_gap for r in requests if r.info.agent_gap is not None]


def compute_agent_turns(requests: Sequence[RequestLifecycleMetric]) -> Optional[List[AgentTurnMetric]]:
    """Group one session's requests into turns. None when no request carries an agent_gap."""
    by_turn: Dict[int, List[RequestLifecycleMetric]] = defaultdict(list)
    for r in requests:
        if r.info.agent_gap is not None:
            by_turn[r.info.agent_gap.turn_index].append(r)
    if not by_turn:
        return None
    turns: List[AgentTurnMetric] = []
    for turn_index in sorted(by_turn):
        group = by_turn[turn_index]
        group.sort(key=lambda r: (r.info.agent_gap.gap_start if r.info.agent_gap else r.start_time, r.start_time))
        first = group[0]
        first_gap = first.info.agent_gap
        assert first_gap is not None
        if first_gap.actor_ready_start is not None:
            start = first_gap.actor_ready_start
        elif first_gap.gap_end is not None:
            start = first_gap.gap_end
        else:
            start = first.start_time
        end = max(r.end_time for r in group)
        turns.append(
            AgentTurnMetric(
                turn_index=turn_index,
                cold=bool(first_gap.cold) if first_gap.cold is not None else turn_index == 0,
                turn_latency_sec=max(end - start, 0.0),
                success=all(r.error is None for r in group),
            )
        )
    return turns


def compute_session_sandbox(requests: Sequence[RequestLifecycleMetric]) -> Optional[SessionSandboxMetrics]:
    """One session's sandbox activity. None when no request carries an agent_gap."""
    gaps = _gaps(requests)
    if not gaps:
        return None
    actors: List[str] = []
    for g in gaps:
        if g.actor and g.actor not in actors:
            actors.append(g.actor)
    creates = [g.actor_create_sec for g in gaps if g.actor_create_sec is not None]
    wakes = [g for g in gaps if g.actor_ready_sec is not None]
    cold = [g.actor_ready_sec for g in wakes if g.cold and g.actor_ready_sec is not None]
    warm = [g.actor_ready_sec for g in wakes if not g.cold and g.actor_ready_sec is not None]
    suspends = [g.suspend_sec for g in gaps if g.suspend_sec is not None]
    tools = [t for g in gaps for t in g.tools]
    overheads = [t.tool_exec_overhead_sec for t in tools if t.error is None and t.tool_exec_overhead_sec is not None]
    return SessionSandboxMetrics(
        actors=actors,
        actor_create_sec=_mean(creates),
        num_wakes=len(wakes),
        actor_ready_sec=ActorReadySummary(cold=_mean(cold), warm_mean=_mean(warm), warm_max=max(warm) if warm else None),
        num_suspends=len(suspends),
        suspend_sec_mean=_mean(suspends),
        num_tool_calls=len(tools),
        tool_exec_overhead_sec_mean=_mean(overheads),
        tool_exec_overhead_sec_max=max(overheads) if overheads else None,
        num_tool_errors=sum(1 for t in tools if _tool_failed(t)),
    )


Stats = Optional[Dict[str, float]]


class SplitStats(TypedDict):
    """A statistic over all samples, and over the cold and warm ones."""

    all: Stats
    cold: Stats
    warm: Stats


class SandboxSummary(TypedDict):
    actor_ready_sec: SplitStats
    suspend_sec: Stats
    actor_create_sec: Stats
    tool_exec_overhead_sec: Stats
    tool_error_rate: Optional[float]


class AgentSessionSummary(TypedDict):
    num_turns: int
    num_turns_succeeded: int
    turn_latency_sec: SplitStats
    sandbox: SandboxSummary


def summarize_agent_session(
    turns: Sequence[AgentTurnMetric],
    gaps: Sequence[AgentGapMetrics],
    percentiles: List[float],
    summarize: Callable[[List[float], List[float]], Stats],
) -> AgentSessionSummary:
    """Aggregate turns and sandbox overheads across sessions."""
    ok = [t for t in turns if t.success]
    wakes = [g for g in gaps if g.actor_ready_sec is not None]

    def ready(selected: Sequence[AgentGapMetrics]) -> List[float]:
        return [g.actor_ready_sec for g in selected if g.actor_ready_sec is not None]

    tools = [t for g in gaps for t in g.tools]
    overheads = [t.tool_exec_overhead_sec for t in tools if t.error is None and t.tool_exec_overhead_sec is not None]
    return {
        "num_turns": len(turns),
        "num_turns_succeeded": len(ok),
        "turn_latency_sec": {
            "all": summarize([t.turn_latency_sec for t in ok], percentiles),
            "cold": summarize([t.turn_latency_sec for t in ok if t.cold], percentiles),
            "warm": summarize([t.turn_latency_sec for t in ok if not t.cold], percentiles),
        },
        "sandbox": {
            "actor_ready_sec": {
                "all": summarize(ready(wakes), percentiles),
                "cold": summarize(ready([g for g in wakes if g.cold]), percentiles),
                "warm": summarize(ready([g for g in wakes if not g.cold]), percentiles),
            },
            "suspend_sec": summarize([g.suspend_sec for g in gaps if g.suspend_sec is not None], percentiles),
            "actor_create_sec": summarize([g.actor_create_sec for g in gaps if g.actor_create_sec is not None], percentiles),
            "tool_exec_overhead_sec": summarize(overheads, percentiles),
            "tool_error_rate": (sum(1 for t in tools if _tool_failed(t)) / len(tools)) if tools else None,
        },
    }
