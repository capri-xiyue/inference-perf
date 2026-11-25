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
"""Agent-session metrics from tool-executor replay runs.

Turns are rebuilt from the per-request ``agent_gap`` records: a turn starts at a
session's first request or at an idle gap (the actor is woken) and runs until
the next idle gap. Definitions follow "Measuring Agent Sessions":

- Lwake-TTFT: first output token of the turn's first LLM call minus the wake trigger.
- Lturn: last LLM call end minus the wake trigger (= Lwake + sum of LTTTC and tool time
  on the critical path).
- alpha_inf: sum of LLM call latencies over Lturn.
"""

from typing import Callable, Dict, List, Optional, Sequence, TypedDict

from inference_perf.apis import RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.apis.agent_gap import AgentGapMetrics, AgentTurnMetric, GapKind


def _first_output_token_time(metric: RequestLifecycleMetric) -> Optional[float]:
    response_metrics = metric.info.response_metrics
    if isinstance(response_metrics, StreamedResponseMetrics) and response_metrics.output_token_times:
        return response_metrics.output_token_times[0]
    return None


def _build_turn(index: int, requests: List[RequestLifecycleMetric]) -> AgentTurnMetric:
    first = requests[0]
    first_gap = first.info.agent_gap
    assert first_gap is not None
    if first_gap.wake_start is not None:
        start = first_gap.wake_start
    elif first_gap.gap_end is not None:
        start = first_gap.gap_end
    else:
        start = first.start_time
    end = max(r.end_time for r in requests)

    wake_to_first_token: Optional[float] = None
    first_call_ttft: Optional[float] = None
    first_token = _first_output_token_time(first) if first.error is None else None
    if first_token is not None:
        wake_to_first_token = first_token - start
        first_call_ttft = first_token - first.start_time

    harness: Optional[float] = None
    if first_gap.wake_start is not None and first_gap.wake_sec is not None:
        harness = max(0.0, first.start_time - (first_gap.wake_start + first_gap.wake_sec))

    gaps: List[AgentGapMetrics] = [r.info.agent_gap for r in requests if r.info.agent_gap is not None]
    tool_gaps = [g for g in gaps if g.gap_kind == GapKind.TOOL]
    tool_exec = [g.tool_exec_sec for g in tool_gaps if g.tool_exec_sec is not None]
    tool_wake = [g.wake_sec for g in tool_gaps if g.wake_sec is not None and g.resumed]
    sum_llm = sum(r.end_time - r.start_time for r in requests)
    turn_latency = end - start

    return AgentTurnMetric(
        turn_index=index,
        cold=bool(first_gap.cold) if first_gap.cold is not None else index == 0,
        first_event_id=first.info.graph_event_id,
        start_time=start,
        end_time=end,
        turn_latency_sec=turn_latency,
        wake_sec=first_gap.wake_sec,
        resumed=first_gap.resumed,
        wake_to_first_token_sec=wake_to_first_token,
        harness_sec=harness,
        first_call_ttft_sec=first_call_ttft,
        num_llm_calls=len(requests),
        num_tool_calls=len(tool_gaps),
        sum_llm_latency_sec=sum_llm,
        sum_tool_exec_sec=sum(tool_exec),
        sum_tool_wake_sec=sum(tool_wake),
        alpha_inf=(sum_llm / turn_latency) if turn_latency > 0 else None,
        tool_exec_sec=tool_exec,
        tool_wake_sec=tool_wake,
        success=all(r.error is None for r in requests),
    )


def compute_agent_turns(requests: Sequence[RequestLifecycleMetric]) -> Optional[List[AgentTurnMetric]]:
    """Group one session's requests into turns. None when no request carries an agent_gap."""
    with_gap = [r for r in requests if r.info.agent_gap is not None]
    if not with_gap:
        return None
    with_gap.sort(key=lambda r: (r.info.agent_gap.gap_start if r.info.agent_gap else r.start_time, r.start_time))
    groups: List[List[RequestLifecycleMetric]] = []
    for r in with_gap:
        gap = r.info.agent_gap
        assert gap is not None
        if not groups or gap.gap_kind == GapKind.IDLE:
            groups.append([])
        groups[-1].append(r)
    return [_build_turn(i, g) for i, g in enumerate(groups)]


Stats = Optional[Dict[str, float]]


class SplitStats(TypedDict):
    """A statistic over all succeeded turns, and over the cold and warm ones."""

    all: Stats
    cold: Stats
    warm: Stats


class AgentSessionSummary(TypedDict):
    num_turns: int
    num_turns_succeeded: int
    num_cold_turns: int
    num_warm_turns: int
    turn_latency_sec: SplitStats
    wake_to_first_token_sec: SplitStats
    wake_sec: SplitStats
    harness_sec: Stats
    first_call_ttft_sec: Stats
    alpha_inf: Stats
    llm_calls_per_turn: Stats
    tool_calls_per_turn: Stats
    num_tool_calls: int
    tool_exec_sec: Stats
    tool_wake_sec: Stats


def _values(turns: Sequence[AgentTurnMetric], getter: Callable[[AgentTurnMetric], Optional[float]]) -> List[float]:
    out: List[float] = []
    for t in turns:
        v = getter(t)
        if v is not None:
            out.append(float(v))
    return out


def summarize_agent_turns(
    turns: Sequence[AgentTurnMetric],
    percentiles: List[float],
    summarize: Callable[[List[float], List[float]], Stats],
) -> AgentSessionSummary:
    """Aggregate turns across sessions, split into cold (first turn) and warm turns."""
    ok = [t for t in turns if t.success]
    cold = [t for t in ok if t.cold]
    warm = [t for t in ok if not t.cold]

    def split(getter: Callable[[AgentTurnMetric], Optional[float]]) -> SplitStats:
        return {
            "all": summarize(_values(ok, getter), percentiles),
            "cold": summarize(_values(cold, getter), percentiles),
            "warm": summarize(_values(warm, getter), percentiles),
        }

    tool_exec = [v for t in ok for v in t.tool_exec_sec]
    tool_wake = [v for t in ok for v in t.tool_wake_sec]
    return {
        "num_turns": len(turns),
        "num_turns_succeeded": len(ok),
        "num_cold_turns": len(cold),
        "num_warm_turns": len(warm),
        "turn_latency_sec": split(lambda t: t.turn_latency_sec),
        "wake_to_first_token_sec": split(lambda t: t.wake_to_first_token_sec),
        "wake_sec": split(lambda t: t.wake_sec if t.resumed else None),
        "harness_sec": summarize(_values(ok, lambda t: t.harness_sec), percentiles),
        "first_call_ttft_sec": summarize(_values(ok, lambda t: t.first_call_ttft_sec), percentiles),
        "alpha_inf": summarize(_values(ok, lambda t: t.alpha_inf), percentiles),
        "llm_calls_per_turn": summarize([float(t.num_llm_calls) for t in ok], percentiles),
        "tool_calls_per_turn": summarize([float(t.num_tool_calls) for t in ok], percentiles),
        "num_tool_calls": len(tool_exec),
        "tool_exec_sec": summarize(tool_exec, percentiles),
        "tool_wake_sec": summarize(tool_wake, percentiles),
    }
