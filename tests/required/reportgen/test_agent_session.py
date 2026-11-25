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
"""Agent turn reconstruction and summary from per-request agent_gap records."""

from typing import Optional

import pytest

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.apis.agent_gap import AgentGapMetrics, GapKind
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.agent_session import compute_agent_turns, summarize_agent_turns
from inference_perf.reportgen.base import summarize


def _req(
    event: str,
    gap: Optional[AgentGapMetrics],
    start: float,
    first_token: float,
    end: float,
    error: Optional[ErrorResponseInfo] = None,
) -> RequestLifecycleMetric:
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=10)),
        response_metrics=StreamedResponseMetrics(output_tokens=2, output_token_times=[first_token, end]),
        graph_event_id=event,
        agent_gap=gap,
    )
    return RequestLifecycleMetric(
        session_id="s", scheduled_time=start, start_time=start, end_time=end, request_data="", info=info, error=error
    )


def _idle(gap_start: float, wake_start: float, wake_sec: float, cold: bool) -> AgentGapMetrics:
    return AgentGapMetrics(
        gap_kind=GapKind.IDLE,
        executor="fake",
        trace_wait_sec=wake_start - gap_start,
        gap_start=gap_start,
        gap_end=wake_start + wake_sec,
        wake_start=wake_start,
        wake_sec=wake_sec,
        resumed=True,
        cold=cold,
    )


def _tool(gap_start: float, exec_sec: float) -> AgentGapMetrics:
    return AgentGapMetrics(
        gap_kind=GapKind.TOOL,
        executor="fake",
        trace_wait_sec=exec_sec,
        gap_start=gap_start,
        gap_end=gap_start + exec_sec,
        tool_exec_sec=exec_sec,
        tool_exit_code=0,
    )


def _session() -> list[RequestLifecycleMetric]:
    # Turn 0 (cold): wake 100.0-101.0, call 101.1-102.0 (first token 101.5),
    #   tool 102.0-102.5, call 102.5-103.0.
    # Turn 1 (warm): idle from 103.0, wake 110.0-110.4, call 110.5-111.0 (first token 110.8).
    return [
        _req("e0", _idle(100.0, 100.0, 1.0, cold=True), 101.1, 101.5, 102.0),
        _req("e1", _tool(102.0, 0.5), 102.5, 102.7, 103.0),
        _req("e2", _idle(103.0, 110.0, 0.4, cold=False), 110.5, 110.8, 111.0),
    ]


def test_turns_split_at_idle_gaps() -> None:
    turns = compute_agent_turns(list(reversed(_session())))  # input order must not matter
    assert turns is not None and len(turns) == 2
    t0, t1 = turns
    assert (t0.cold, t1.cold) == (True, False)
    assert (t0.first_event_id, t1.first_event_id) == ("e0", "e2")
    assert (t0.num_llm_calls, t0.num_tool_calls, t1.num_llm_calls) == (2, 1, 1)


def test_turn_latencies() -> None:
    turns = compute_agent_turns(_session())
    assert turns is not None
    t0, t1 = turns
    # Lturn: wake trigger to the end of the turn's last call.
    assert t0.turn_latency_sec == pytest.approx(3.0)
    assert t1.turn_latency_sec == pytest.approx(1.0)
    # Lwake-TTFT: wake trigger to the first output token of the turn's first call.
    assert t0.wake_to_first_token_sec == pytest.approx(1.5)
    assert t1.wake_to_first_token_sec == pytest.approx(0.8)
    # Harness: actor served (wake_start + wake_sec) until the request went out.
    assert t0.harness_sec == pytest.approx(0.1)
    assert t1.harness_sec == pytest.approx(0.1)
    assert t0.first_call_ttft_sec == pytest.approx(0.4)
    assert t0.sum_llm_latency_sec == pytest.approx(0.9 + 0.5)
    assert t0.sum_tool_exec_sec == pytest.approx(0.5)
    assert t0.alpha_inf == pytest.approx(1.4 / 3.0)


def test_failed_request_fails_turn_and_is_excluded_from_summary() -> None:
    reqs = _session()
    reqs[2] = _req(
        "e2",
        _idle(103.0, 110.0, 0.4, cold=False),
        110.5,
        110.8,
        111.0,
        error=ErrorResponseInfo(error_type="HTTP", error_msg="500"),
    )
    turns = compute_agent_turns(reqs)
    assert turns is not None
    assert turns[1].success is False and turns[1].wake_to_first_token_sec is None
    summary = summarize_agent_turns(turns, [50.0], summarize)
    assert summary["num_turns"] == 2 and summary["num_turns_succeeded"] == 1
    assert summary["num_cold_turns"] == 1 and summary["num_warm_turns"] == 0
    cold_w2ft = summary["wake_to_first_token_sec"]["cold"]
    assert cold_w2ft is not None and cold_w2ft["mean"] == pytest.approx(1.5)
    assert summary["wake_to_first_token_sec"]["warm"] is None
    tool_exec = summary["tool_exec_sec"]
    assert tool_exec is not None and tool_exec["mean"] == pytest.approx(0.5)


def test_no_agent_gap_means_no_turns() -> None:
    assert compute_agent_turns([_req("e0", None, 1.0, 1.1, 1.2)]) is None


def test_agent_fields_omitted_from_serialized_records_without_tool_executor() -> None:
    plain = _req("e0", None, 1.0, 1.1, 1.2)
    assert "agent_gap" not in plain.info.model_dump()
    with_gap = _session()[0]
    assert with_gap.info.model_dump()["agent_gap"]["gap_kind"] == "idle"
