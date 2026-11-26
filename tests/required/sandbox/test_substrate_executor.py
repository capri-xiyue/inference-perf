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
"""SubstrateToolExecutor lifecycle and timing against the fake Substrate backend."""

from typing import List, Sequence, Tuple

import pytest

from inference_perf.apis.agent_gap import GapKind
from inference_perf.config.sandbox import (
    ActorAssignment,
    ActorAssignmentConfig,
    ActorAssignmentRule,
    FakeSubstrateConfig,
    SandboxConfig,
    SandboxType,
)
from inference_perf.sandbox import (
    GapContext,
    PlannedTool,
    ToolExecutionError,
    build_tool_executor,
    classify_gap,
    parse_planned_tools,
    resolve_actor_key,
)
from inference_perf.sandbox.fake_backend import FakeSubstrateBackend
from inference_perf.sandbox.substrate_executor import ExecResult, SubstrateToolExecutor

# Latencies large enough to dominate scheduling noise, small enough to keep tests fast.
COLD_MS = 120.0
WARM_MS = 60.0
SUSPEND_MS = 20.0
UPLOAD_MS = 80.0
EXEC_MS = 10.0


FAKE = FakeSubstrateConfig(
    create_latency_ms=0,
    cold_resume_latency_ms=COLD_MS,
    warm_resume_latency_ms=WARM_MS,
    snapshot_capture_latency_ms=SUSPEND_MS,
    snapshot_upload_latency_ms=UPLOAD_MS,
    delete_latency_ms=0,
    exec_overhead_ms=EXEC_MS,
    jitter_fraction=0,
)

ONE_TOOL = (PlannedTool(name="Bash", start_sec=0.0, duration_sec=0.0),)


def _config(**overrides: object) -> SandboxConfig:
    return SandboxConfig.model_validate({"type": SandboxType.FAKE_SUBSTRATE, "fake": FAKE, **overrides})


def _executor(**overrides: object) -> Tuple[SubstrateToolExecutor, FakeSubstrateBackend]:
    executor = build_tool_executor(_config(**overrides), seed=1)
    assert isinstance(executor, SubstrateToolExecutor)
    assert isinstance(executor.backend, FakeSubstrateBackend)
    return executor, executor.backend


def _ctx(
    kind: GapKind,
    wait_sec: float = 0.0,
    session: str = "s0",
    actor_key: str = "s0",
    event: str = "e",
    tools: Sequence[PlannedTool] = (),
) -> GapContext:
    return GapContext(session_id=session, actor_key=actor_key, event_id=event, wait_sec=wait_sec, gap_kind=kind, tools=tools)


def _ops(backend: FakeSubstrateBackend) -> List[str]:
    return [op for op, _ in backend.calls]


class TestClassifyGap:
    @pytest.mark.parametrize(
        "has_pred,recorded,expected",
        [
            (False, None, GapKind.IDLE),  # session start: create and cold-wake before the first call
            (False, "tool", GapKind.IDLE),
            (True, "tool", GapKind.TOOL),
            (True, "idle", GapKind.IDLE),
            (True, None, GapKind.NONE),  # unclassified: replayed as a plain wait
            (True, "bogus", GapKind.NONE),
        ],
    )
    def test_classify(self, has_pred: bool, recorded: str, expected: GapKind) -> None:
        assert classify_gap(has_predecessors=has_pred, recorded_kind=recorded) == expected


class TestParsePlannedTools:
    def test_reads_the_attribute_and_clamps_negatives(self) -> None:
        tools = parse_planned_tools(
            [
                {"name": "Read", "start_sec": 0.0, "duration_sec": 0.0},
                {"name": "Bash", "start_sec": -1.0, "duration_sec": 1.5},
                "junk",
            ]
        )
        assert tools == [PlannedTool("Read", 0.0, 0.0), PlannedTool("Bash", 0.0, 1.5)]

    def test_missing_attribute_is_empty(self) -> None:
        assert parse_planned_tools(None) == []


class TestActorAssignment:
    EXPLORE = {"gen_ai.agent.id": "a1", "gen_ai.agent.name": "Explore"}
    EXPLORE_2 = {"gen_ai.agent.id": "a3", "gen_ai.agent.name": "Explore"}
    PLANNER = {"gen_ai.agent.id": "a2", "gen_ai.agent.name": "Plan"}
    MAIN: dict[str, object] = {}

    def test_default_puts_every_call_on_the_session_actor(self) -> None:
        default = ActorAssignmentConfig()
        assert resolve_actor_key(default, "s", self.EXPLORE) == "s"
        assert resolve_actor_key(default, "s", self.MAIN) == "s"

    def test_only_named_agents_get_a_dedicated_actor(self) -> None:
        a = ActorAssignmentConfig(
            rules=[ActorAssignmentRule(match={"gen_ai.agent.name": "Expl*"}, assign=ActorAssignment.DEDICATED)]
        )
        assert resolve_actor_key(a, "s", self.EXPLORE) == "s::agent:a1"  # glob match
        assert resolve_actor_key(a, "s", self.EXPLORE_2) == "s::agent:a3"  # one actor per agent instance
        assert resolve_actor_key(a, "s", self.PLANNER) == "s"
        assert resolve_actor_key(a, "s", self.MAIN) == "s"  # missing attribute never matches

    def test_dedicated_without_agent_identity_falls_back_to_session(self) -> None:
        a = ActorAssignmentConfig(default=ActorAssignment.DEDICATED)
        assert resolve_actor_key(a, "s", self.MAIN) == "s"

    def test_rule_needs_a_match(self) -> None:
        with pytest.raises(ValueError):
            ActorAssignmentRule(match={}, assign=ActorAssignment.DEDICATED)


class TestSubstrateExecutor:
    async def test_first_idle_gap_creates_actor_and_cold_wakes(self) -> None:
        executor, backend = _executor()
        m = await executor.run_gap(_ctx(GapKind.IDLE))
        assert _ops(backend) == ["create", "resume"]
        assert m.actor is not None and m.actor.startswith("ip-")
        assert m.turn_index == 0 and m.cold is True
        assert m.actor_create_sec is not None
        assert m.actor_ready_sec is not None and m.actor_ready_sec >= COLD_MS / 1000 * 0.9
        # Freshly created actors are already suspended: nothing to suspend.
        assert m.suspend_sec is None
        assert m.actor_ready_start is not None and m.gap_end is not None and m.gap_end >= m.actor_ready_start

    async def test_tool_gap_runs_each_tool_at_its_offset_concurrently(self) -> None:
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE))
        tools = (PlannedTool("Read", 0.0, 0.0), PlannedTool("Bash", 0.05, 0.1), PlannedTool("Grep", 0.0, 0.1))
        m = await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.15, tools=tools))
        assert _ops(backend)[-3:] == ["execute", "execute", "execute"]
        # Already running: no wake for the tools.
        assert m.actor_ready_sec is None and m.cold is None
        assert [t.name for t in m.tools] == ["Read", "Bash", "Grep"]
        for t in m.tools:
            assert t.exit_code == 0 and t.error is None
            assert t.tool_exec_sec is not None and t.tool_exec_sec >= t.duration_sec
            assert t.tool_exec_overhead_sec is not None
            assert t.tool_exec_overhead_sec == pytest.approx(t.tool_exec_sec - t.duration_sec)
            assert t.tool_exec_overhead_sec >= EXEC_MS / 1000 * 0.9
        # Concurrent, not serial: Bash ends at 0.05 + 0.1 (+overhead), well under the 0.2s serial sum.
        assert m.gap_end is not None and m.gap_end - m.gap_start < 0.2 + 3 * EXEC_MS / 1000

    async def test_gap_waits_out_the_recorded_length_after_the_tools(self) -> None:
        executor, _ = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.2, tools=ONE_TOOL))
        assert m.gap_end is not None and m.gap_end - m.gap_start >= 0.2 * 0.95

    async def test_tool_gap_without_tools_is_a_plain_wait(self) -> None:
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE))
        calls = len(backend.calls)
        m = await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.02))
        assert len(backend.calls) == calls and m.tools == []

    async def test_later_idle_gap_suspends_then_warm_wakes_and_starts_a_turn(self) -> None:
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.IDLE, wait_sec=0.1))
        assert _ops(backend)[-2:] == ["suspend", "resume"]
        assert m.turn_index == 1 and m.cold is False
        assert m.suspend_sec is not None and m.suspend_sec >= SUSPEND_MS / 1000 * 0.9
        assert m.actor_ready_sec is not None and WARM_MS / 1000 * 0.9 <= m.actor_ready_sec < COLD_MS / 1000
        # The wake starts once the recorded idle time is over; suspend time is part of it.
        assert m.actor_ready_start is not None and m.actor_ready_start - m.gap_start >= 0.1 * 0.95

    async def test_turn_index_counts_idle_gaps_per_session(self) -> None:
        executor, _ = _executor()
        seen = []
        for kind in (GapKind.IDLE, GapKind.TOOL, GapKind.NONE, GapKind.IDLE, GapKind.TOOL):
            m = await executor.run_gap(_ctx(kind, tools=ONE_TOOL))
            seen.append(m.turn_index)
        other = await executor.run_gap(_ctx(GapKind.IDLE, session="s1", actor_key="s1"))
        assert seen == [0, 0, 0, 1, 1] and other.turn_index == 0

    async def test_no_suspend_between_turns_keeps_actor_running(self) -> None:
        executor, backend = _executor(actor={"suspend": {"between_turns": False}})
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.IDLE, wait_sec=0.01))
        assert "suspend" not in _ops(backend)
        assert m.actor_ready_sec is None and m.actor_ready_start is not None

    async def test_suspend_after_each_tool_call_makes_next_tool_gap_wake(self) -> None:
        executor, backend = _executor(actor={"suspend": {"after_each_tool_call": True}})
        await executor.run_gap(_ctx(GapKind.IDLE))
        await executor.run_gap(_ctx(GapKind.TOOL, tools=ONE_TOOL))
        m = await executor.run_gap(_ctx(GapKind.TOOL, tools=ONE_TOOL))
        assert _ops(backend)[-4:] == ["execute", "suspend", "resume", "execute"]
        assert m.cold is False
        assert m.actor_ready_sec is not None and m.actor_ready_sec >= WARM_MS / 1000 * 0.9

    @pytest.mark.parametrize("durability,expected_ms", [("async", SUSPEND_MS), ("sync", SUSPEND_MS + UPLOAD_MS)])
    async def test_snapshot_durability_sets_suspend_time(self, durability: str, expected_ms: float) -> None:
        executor, _ = _executor(actor={"suspend": {"snapshot_durability": durability}})
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.IDLE, wait_sec=0.0))
        assert m.suspend_sec is not None
        assert expected_ms / 1000 * 0.9 <= m.suspend_sec < (expected_ms + SUSPEND_MS) / 1000

    async def test_actor_ready_includes_waiting_for_background_sync_suspend(self) -> None:
        # Back-to-back tool gaps: the second finds the sync suspend still uploading
        # and waits for it before resuming.
        executor, _ = _executor(actor={"suspend": {"after_each_tool_call": True, "snapshot_durability": "sync"}})
        await executor.run_gap(_ctx(GapKind.IDLE))
        await executor.run_gap(_ctx(GapKind.TOOL, tools=ONE_TOOL))
        m = await executor.run_gap(_ctx(GapKind.TOOL, tools=ONE_TOOL))
        assert m.actor_ready_sec is not None
        assert m.actor_ready_sec >= (SUSPEND_MS + UPLOAD_MS + WARM_MS) / 1000 * 0.9

    async def test_none_gap_touches_no_actor(self) -> None:
        executor, backend = _executor()
        m = await executor.run_gap(_ctx(GapKind.NONE))
        assert backend.calls == [] and m.actor is None

    async def test_actor_per_key_and_end_session_deletes_only_that_session(self) -> None:
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE, session="a", actor_key="a"))
        await executor.run_gap(_ctx(GapKind.IDLE, session="a", actor_key="a::agent:x"))
        await executor.run_gap(_ctx(GapKind.IDLE, session="b", actor_key="b"))
        names = {name for op, name in backend.calls if op == "create"}
        assert len(names) == 3
        await executor.end_session("a")
        deleted = [name for op, name in backend.calls if op == "delete"]
        assert len(deleted) == 2 and executor.actor_name("b") not in deleted
        await executor.aclose()
        assert [name for op, name in backend.calls if op == "delete"][-1] == executor.actor_name("b")

    async def test_tool_failure_raises_with_per_tool_metrics(self) -> None:
        class BrokenRouter(FakeSubstrateBackend):
            async def execute(self, name: str, command: str, timeout_sec: float) -> ExecResult:
                raise RuntimeError("router 503")

        executor = SubstrateToolExecutor(_config(), BrokenRouter(FAKE), name="fake")
        await executor.run_gap(_ctx(GapKind.IDLE))
        with pytest.raises(ToolExecutionError) as err:
            await executor.run_gap(_ctx(GapKind.TOOL, tools=ONE_TOOL))
        metrics = err.value.metrics
        assert metrics is not None and metrics.error is not None and "router 503" in metrics.error
        assert metrics.tools[0].error is not None and "router 503" in metrics.tools[0].error


class TestSleepChunks:
    def test_long_tools_are_split_under_the_router_timeout(self) -> None:
        from inference_perf.sandbox.substrate_executor import MAX_EXEC_CHUNK_SEC, _sleep_chunks

        assert _sleep_chunks(0.0) == [0.0]
        assert _sleep_chunks(1.5) == [1.5]
        chunks = _sleep_chunks(2 * MAX_EXEC_CHUNK_SEC + 7.0)
        assert chunks == [MAX_EXEC_CHUNK_SEC, MAX_EXEC_CHUNK_SEC, pytest.approx(7.0)]
        assert all(c <= MAX_EXEC_CHUNK_SEC for c in chunks)

    async def test_executor_issues_one_call_per_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import inference_perf.sandbox.substrate_executor as mod

        monkeypatch.setattr(mod, "MAX_EXEC_CHUNK_SEC", 0.02)
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.TOOL, tools=(PlannedTool("Bash", 0.0, 0.05),)))
        assert _ops(backend).count("execute") == 3  # 0.02 + 0.02 + 0.01
        assert m.tools[0].tool_exec_sec is not None and m.tools[0].tool_exec_sec >= 0.05


class TestConfig:
    def test_substrate_type_requires_substrate_block(self) -> None:
        with pytest.raises(ValueError, match="sandbox.substrate is required"):
            SandboxConfig(type=SandboxType.SUBSTRATE)
        cfg = SandboxConfig.model_validate(
            {"type": "substrate", "substrate": {"atespace": "bench", "actor_template": "python-runtime"}}
        )
        assert cfg.substrate is not None and cfg.substrate.api.endpoint == "api.ate-system.svc:443"

    def test_nested_actor_block(self) -> None:
        cfg = SandboxConfig.model_validate(
            {
                "actor": {
                    "name_prefix": "bench",
                    "assignment": {"rules": [{"match": {"gen_ai.agent.name": "Explore"}, "assign": "dedicated"}]},
                    "suspend": {"between_turns": False, "after_each_tool_call": True, "snapshot_durability": "sync"},
                    "tool_call": {"timeout_sec": 5},
                    "delete_on_session_end": False,
                }
            }
        )
        assert cfg.actor.assignment.resolve({"gen_ai.agent.name": "Explore"}) == ActorAssignment.DEDICATED
        assert cfg.actor.suspend.after_each_tool_call and not cfg.actor.suspend.between_turns
        assert cfg.actor.suspend.snapshot_durability.value == "sync"

    def test_command_template_is_gone(self) -> None:
        with pytest.raises(ValueError):
            SandboxConfig.model_validate({"actor": {"tool_call": {"command_template": "true"}}})
