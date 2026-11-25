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
"""Agent turns, per-session sandbox activity and the agent_session summary."""

from typing import List, Optional

import pytest

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.apis.agent_gap import AgentGapMetrics, GapKind, ToolCallMetrics
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.agent_session import compute_agent_turns, compute_session_sandbox, summarize_agent_session
from inference_perf.reportgen.base import summarize


def _req(
    event: str,
    gap: Optional[AgentGapMetrics],
    start: float,
    end: float,
    error: Optional[ErrorResponseInfo] = None,
) -> RequestLifecycleMetric:
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=10)),
        response_metrics=StreamedResponseMetrics(output_tokens=2, output_token_times=[start + 0.1, end]),
        graph_event_id=event,
        agent_gap=gap,
    )
    return RequestLifecycleMetric(
        session_id="s", scheduled_time=start, start_time=start, end_time=end, request_data="", info=info, error=error
    )


def _idle(turn: int, gap_start: float, ready_start: float, ready_sec: float, cold: bool, **kw: object) -> AgentGapMetrics:
    return AgentGapMetrics.model_validate(
        {
            "turn_index": turn,
            "gap_kind": GapKind.IDLE,
            "executor": "fake",
            "trace_wait_sec": ready_start - gap_start,
            "gap_start": gap_start,
            "gap_end": ready_start + ready_sec,
            "actor": "ip-a",
            "actor_ready_start": ready_start,
            "actor_ready_sec": ready_sec,
            "cold": cold,
            **kw,
        }
    )


def _tool(turn: int, gap_start: float, tools: List[ToolCallMetrics]) -> AgentGapMetrics:
    return AgentGapMetrics(
        turn_index=turn,
        gap_kind=GapKind.TOOL,
        executor="fake",
        trace_wait_sec=0.5,
        gap_start=gap_start,
        gap_end=gap_start + 0.5,
        actor="ip-a",
        actor_ready_start=gap_start,
        tools=tools,
    )


def _t(name: str, duration: float, overhead: float, exit_code: int = 0, error: Optional[str] = None) -> ToolCallMetrics:
    return ToolCallMetrics(
        name=name,
        start_sec=0.0,
        duration_sec=duration,
        tool_exec_sec=duration + overhead,
        tool_exec_overhead_sec=overhead,
        exit_code=exit_code,
        error=error,
    )


def _session() -> list[RequestLifecycleMetric]:
    # Turn 0 (cold): actor needed at 100.0, ready 101.0, call 101.1-102.0,
    #   tools 102.0-102.5, call 102.5-103.0.
    # Turn 1 (warm): idle from 103.0 (suspend), needed at 110.0, ready 110.4, call 110.5-111.0.
    return [
        _req("e0", _idle(0, 100.0, 100.0, 1.0, cold=True, actor_create_sec=0.02), 101.1, 102.0),
        _req("e1", _tool(0, 102.0, [_t("Read", 0.0, 0.02), _t("Bash", 0.45, 0.03)]), 102.5, 103.0),
        _req("e2", _idle(1, 103.0, 110.0, 0.4, cold=False, suspend_sec=0.09), 110.5, 111.0),
    ]


class TestComputeAgentTurns:
    def test_turns_run_from_needing_the_actor_to_the_last_call(self) -> None:
        turns = compute_agent_turns(_session())
        assert turns is not None and [t.turn_index for t in turns] == [0, 1]
        cold, warm = turns
        assert cold.cold is True and cold.turn_latency_sec == pytest.approx(3.0)  # 100.0 -> 103.0
        assert warm.cold is False and warm.turn_latency_sec == pytest.approx(1.0)  # 110.0 -> 111.0
        assert cold.success and warm.success

    def test_failed_request_fails_its_turn(self) -> None:
        reqs = _session()
        reqs[1] = _req("e1", reqs[1].info.agent_gap, 102.5, 103.0, error=ErrorResponseInfo(error_type="x", error_msg="boom"))
        turns = compute_agent_turns(reqs)
        assert turns is not None and turns[0].success is False and turns[1].success is True

    def test_without_agent_gaps_there_are_no_turns(self) -> None:
        assert compute_agent_turns([_req("e0", None, 1.0, 2.0)]) is None

    def test_serialized_turn_has_only_the_agreed_fields(self) -> None:
        turns = compute_agent_turns(_session())
        assert turns is not None
        assert set(turns[0].model_dump()) == {"turn_index", "cold", "turn_latency_sec", "success"}


class TestComputeSessionSandbox:
    def test_counts_and_simple_aggregates(self) -> None:
        s = compute_session_sandbox(_session())
        assert s is not None
        assert s.actors == ["ip-a"]
        assert s.actor_create_sec == pytest.approx(0.02)
        assert s.num_wakes == 2
        assert s.actor_ready_sec.cold == pytest.approx(1.0)
        assert s.actor_ready_sec.warm_mean == pytest.approx(0.4) and s.actor_ready_sec.warm_max == pytest.approx(0.4)
        assert s.num_suspends == 1 and s.suspend_sec_mean == pytest.approx(0.09)
        assert s.num_tool_calls == 2
        assert s.tool_exec_overhead_sec_mean == pytest.approx(0.025)
        assert s.tool_exec_overhead_sec_max == pytest.approx(0.03)
        assert s.num_tool_errors == 0

    def test_errors_and_nonzero_exits_count_as_tool_errors(self) -> None:
        reqs = [
            _req("e0", _idle(0, 0.0, 0.0, 0.2, cold=True), 0.3, 1.0),
            _req("e1", _tool(0, 1.0, [_t("A", 0.1, 0.02, exit_code=1), _t("B", 0.1, 0.02, error="timeout")]), 1.5, 2.0),
        ]
        s = compute_session_sandbox(reqs)
        assert s is not None and s.num_tool_errors == 2
        # Errored tools are left out of the overhead.
        assert s.tool_exec_overhead_sec_mean == pytest.approx(0.02)


class TestSummarizeAgentSession:
    def test_summary_splits_cold_and_warm_and_reports_sandbox_overheads(self) -> None:
        reqs = _session()
        turns = compute_agent_turns(reqs)
        assert turns is not None
        gaps = [r.info.agent_gap for r in reqs if r.info.agent_gap is not None]
        s = summarize_agent_session(turns, gaps, [50, 90], summarize)
        assert s["num_turns"] == 2 and s["num_turns_succeeded"] == 2
        lat = s["turn_latency_sec"]
        assert lat["cold"] is not None and lat["cold"]["mean"] == pytest.approx(3.0)
        assert lat["warm"] is not None and lat["warm"]["mean"] == pytest.approx(1.0)
        sb = s["sandbox"]
        ready = sb["actor_ready_sec"]
        assert ready["cold"] is not None and ready["cold"]["mean"] == pytest.approx(1.0)
        assert ready["warm"] is not None and ready["warm"]["mean"] == pytest.approx(0.4)
        assert sb["suspend_sec"] is not None and sb["suspend_sec"]["mean"] == pytest.approx(0.09)
        assert sb["actor_create_sec"] is not None and sb["actor_create_sec"]["mean"] == pytest.approx(0.02)
        assert sb["tool_exec_overhead_sec"] is not None and sb["tool_exec_overhead_sec"]["max"] == pytest.approx(0.03)
        assert sb["tool_error_rate"] == 0.0
