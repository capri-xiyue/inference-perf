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

from typing import List, Tuple

import pytest

from inference_perf.apis.agent_gap import GapKind
from inference_perf.config.datagen.toolexec import (
    ActorAssignment,
    ActorAssignmentConfig,
    ActorAssignmentRule,
    FakeSubstrateConfig,
    ResumeMode,
    ToolExecutorConfig,
    ToolExecutorType,
)
from inference_perf.toolexec import GapContext, ToolExecutionError, build_tool_executor, classify_gap, resolve_actor_key
from inference_perf.toolexec.fake_backend import FakeSubstrateBackend
from inference_perf.toolexec.substrate_executor import ExecResult, SubstrateToolExecutor

# Latencies large enough to dominate scheduling noise, small enough to keep tests fast.
COLD_MS = 120.0
WARM_MS = 60.0
SUSPEND_MS = 20.0


FAKE = FakeSubstrateConfig(
    create_latency_ms=0,
    cold_resume_latency_ms=COLD_MS,
    warm_resume_latency_ms=WARM_MS,
    suspend_latency_ms=SUSPEND_MS,
    delete_latency_ms=0,
    exec_overhead_ms=0,
    jitter_fraction=0,
)


def _config(**overrides: object) -> ToolExecutorConfig:
    return ToolExecutorConfig.model_validate({"type": ToolExecutorType.FAKE_SUBSTRATE, "fake": FAKE, **overrides})


def _executor(**overrides: object) -> Tuple[SubstrateToolExecutor, FakeSubstrateBackend]:
    executor = build_tool_executor(_config(**overrides), seed=1)
    assert isinstance(executor, SubstrateToolExecutor)
    assert isinstance(executor.backend, FakeSubstrateBackend)
    return executor, executor.backend


def _ctx(kind: GapKind, wait_sec: float = 0.0, session: str = "s0", actor_key: str = "s0", event: str = "e") -> GapContext:
    return GapContext(session_id=session, actor_key=actor_key, event_id=event, wait_sec=wait_sec, gap_kind=kind)


def _ops(backend: FakeSubstrateBackend) -> List[str]:
    return [op for op, _ in backend.calls]


class TestClassifyGap:
    @pytest.mark.parametrize(
        "has_pred,wait_ms,role,expected",
        [
            (False, 0, "user", GapKind.IDLE),  # session start: cold wake before the first call
            (True, 0, "tool", GapKind.TOOL),  # role:tool result follows a tool call, however short
            (True, 9000, "tool", GapKind.TOOL),
            (True, 9000, "user", GapKind.IDLE),  # long gap without tool role: user idle
            (True, 5000, "user", GapKind.IDLE),  # threshold is inclusive
            (True, 300, "user", GapKind.TOOL),  # short gap: tool call (weka has no tool roles)
            (True, 0, "user", GapKind.NONE),
            (True, 0, None, GapKind.NONE),
        ],
    )
    def test_classify(self, has_pred: bool, wait_ms: int, role: str, expected: GapKind) -> None:
        got = classify_gap(has_predecessors=has_pred, wait_ms=wait_ms, last_message_role=role, idle_gap_threshold_ms=5000)
        assert got == expected

    @pytest.mark.parametrize(
        "recorded,wait_ms,role,expected",
        [
            ("tool", 30000, "user", GapKind.TOOL),  # a 30s build is still a tool, whatever its length
            ("idle", 300, "tool", GapKind.IDLE),  # recorded user input beats heuristics
            ("bogus", 300, "user", GapKind.TOOL),  # unknown values fall through to heuristics
            (None, 9000, "user", GapKind.IDLE),
        ],
    )
    def test_recorded_kind_takes_precedence(self, recorded: str, wait_ms: int, role: str, expected: GapKind) -> None:
        got = classify_gap(
            has_predecessors=True,
            wait_ms=wait_ms,
            last_message_role=role,
            idle_gap_threshold_ms=5000,
            recorded_kind=recorded,
        )
        assert got == expected

    def test_session_start_is_always_idle(self) -> None:
        got = classify_gap(
            has_predecessors=False, wait_ms=0, last_message_role=None, idle_gap_threshold_ms=5000, recorded_kind="tool"
        )
        assert got == GapKind.IDLE


class TestActorAssignment:
    EXPLORE = {"gen_ai.agent.id": "a1", "gen_ai.agent.name": "Explore"}
    EXPLORE_2 = {"gen_ai.agent.id": "a3", "gen_ai.agent.name": "Explore"}
    PLANNER = {"gen_ai.agent.id": "a2", "gen_ai.agent.name": "Plan"}
    MAIN: dict[str, object] = {}

    def test_default_puts_every_call_subagents_included_on_the_session_actor(self) -> None:
        default = ActorAssignmentConfig()
        assert resolve_actor_key(default, "s", self.EXPLORE) == "s"
        assert resolve_actor_key(default, "s", self.PLANNER) == "s"
        assert resolve_actor_key(default, "s", self.MAIN) == "s"

    def test_only_named_agents_get_a_dedicated_actor(self) -> None:
        a = ActorAssignmentConfig(
            rules=[ActorAssignmentRule(match={"gen_ai.agent.name": "Expl*"}, actor=ActorAssignment.DEDICATED)]
        )
        assert resolve_actor_key(a, "s", self.EXPLORE) == "s::agent:a1"  # glob match
        assert resolve_actor_key(a, "s", self.EXPLORE_2) == "s::agent:a3"  # one actor per agent instance
        assert resolve_actor_key(a, "s", self.PLANNER) == "s"  # not named: shares the session actor
        assert resolve_actor_key(a, "s", self.MAIN) == "s"  # missing attribute never matches

    def test_first_matching_rule_wins_and_all_keys_must_match(self) -> None:
        a = ActorAssignmentConfig(
            rules=[
                ActorAssignmentRule(match={"gen_ai.agent.name": "Explore", "tier": "cpu"}, actor=ActorAssignment.SESSION),
                ActorAssignmentRule(match={"gen_ai.agent.name": "*"}, actor=ActorAssignment.DEDICATED),
            ]
        )
        assert resolve_actor_key(a, "s", self.EXPLORE) == "s::agent:a1"
        assert resolve_actor_key(a, "s", {**self.EXPLORE, "tier": "cpu"}) == "s"

    def test_dedicated_without_agent_identity_falls_back_to_session(self) -> None:
        a = ActorAssignmentConfig(default=ActorAssignment.DEDICATED)
        assert resolve_actor_key(a, "s", self.MAIN) == "s"
        assert resolve_actor_key(a, "s", {"gen_ai.agent.name": "Explore"}) == "s::agent:Explore"

    def test_rule_needs_a_match(self) -> None:
        with pytest.raises(ValueError):
            ActorAssignmentRule(match={}, actor=ActorAssignment.DEDICATED)


class TestSubstrateExecutor:
    async def test_first_idle_gap_creates_actor_and_cold_wakes(self) -> None:
        executor, backend = _executor()
        m = await executor.run_gap(_ctx(GapKind.IDLE))
        assert _ops(backend) == ["create", "resume"]
        assert m.actor is not None and m.actor.startswith("fake-")
        assert m.cold is True and m.resumed is True
        assert m.wake_sec is not None and m.wake_sec >= COLD_MS / 1000 * 0.9
        # Freshly created actors are already suspended: nothing to suspend.
        assert m.suspend_sec is None
        assert m.wake_start is not None and m.gap_end is not None and m.gap_end >= m.wake_start

    async def test_tool_gap_on_running_actor_runs_command_without_wake(self) -> None:
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.05))
        assert _ops(backend)[-1] == "execute"
        assert m.wake_sec is None and m.wake_start is None
        assert m.tool_exit_code == 0
        assert m.tool_exec_sec is not None and m.tool_exec_sec >= 0.05

    async def test_later_idle_gap_suspends_then_warm_wakes_within_the_gap(self) -> None:
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.IDLE, wait_sec=0.1))
        assert _ops(backend)[-2:] == ["suspend", "resume"]
        assert m.cold is False and m.resumed is True
        assert m.suspend_sec is not None and m.suspend_sec >= SUSPEND_MS / 1000 * 0.9
        assert m.wake_sec is not None and WARM_MS / 1000 * 0.9 <= m.wake_sec < COLD_MS / 1000
        # The wake starts once the recorded idle time is over; suspend time is part of it.
        assert m.wake_start is not None and m.wake_start - m.gap_start >= 0.1 * 0.95

    async def test_suspend_on_idle_disabled_keeps_actor_running(self) -> None:
        executor, backend = _executor(suspend_on_idle=False)
        await executor.run_gap(_ctx(GapKind.IDLE))
        m = await executor.run_gap(_ctx(GapKind.IDLE, wait_sec=0.01))
        assert "suspend" not in _ops(backend)
        assert m.resumed is False and m.wake_sec == 0.0

    async def test_suspend_during_inference_makes_next_tool_call_wake(self) -> None:
        executor, backend = _executor(suspend_during_inference=True)
        await executor.run_gap(_ctx(GapKind.IDLE))
        await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.0))
        m = await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.0))
        assert _ops(backend)[-4:] == ["execute", "suspend", "resume", "execute"]
        assert m.resumed is True and m.cold is False
        assert m.wake_sec is not None and m.wake_sec >= WARM_MS / 1000 * 0.9

    async def test_implicit_resume_attributes_excess_exec_time_to_wake(self) -> None:
        executor, backend = _executor(resume_mode=ResumeMode.IMPLICIT, suspend_during_inference=True)
        await executor.run_gap(_ctx(GapKind.IDLE))  # implicit wake: a no-op request
        await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.0))
        m = await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.05))
        assert "resume" in _ops(backend)  # the fake router resumes inside execute
        assert m.wake_estimated is True and m.resumed is True
        assert m.wake_sec is not None and m.wake_sec >= WARM_MS / 1000 * 0.9
        assert m.tool_exec_sec is not None and m.tool_exec_sec >= 0.05 + WARM_MS / 1000 * 0.9

    async def test_none_gap_touches_no_actor(self) -> None:
        executor, backend = _executor()
        m = await executor.run_gap(_ctx(GapKind.NONE))
        assert backend.calls == [] and m.actor is None

    async def test_actor_per_key_and_end_session_deletes_only_that_session(self) -> None:
        executor, backend = _executor()
        await executor.run_gap(_ctx(GapKind.IDLE, session="a", actor_key="a"))
        await executor.run_gap(_ctx(GapKind.IDLE, session="a", actor_key="a::sa:x:s0"))
        await executor.run_gap(_ctx(GapKind.IDLE, session="b", actor_key="b"))
        names = {name for op, name in backend.calls if op == "create"}
        assert len(names) == 3
        await executor.end_session("a")
        deleted = [name for op, name in backend.calls if op == "delete"]
        assert len(deleted) == 2 and executor.actor_name("b") not in deleted
        await executor.aclose()
        assert [name for op, name in backend.calls if op == "delete"][-1] == executor.actor_name("b")

    async def test_backend_failure_raises_with_partial_metrics(self) -> None:
        class BrokenRouter(FakeSubstrateBackend):
            async def execute(self, name: str, command: str, timeout_sec: float) -> ExecResult:
                raise RuntimeError("router 503")

        executor = SubstrateToolExecutor(_config(), BrokenRouter(FAKE), name="fake", actor_name_prefix="fake")
        await executor.run_gap(_ctx(GapKind.IDLE))
        with pytest.raises(ToolExecutionError) as err:
            await executor.run_gap(_ctx(GapKind.TOOL, wait_sec=0.0))
        assert err.value.metrics is not None
        assert err.value.metrics.error is not None and "router 503" in err.value.metrics.error


class TestConfig:
    def test_substrate_type_requires_connection_settings(self) -> None:
        with pytest.raises(ValueError, match="tool_executor.substrate is required"):
            ToolExecutorConfig(type=ToolExecutorType.SUBSTRATE)

    def test_rejects_bad_command_template(self) -> None:
        with pytest.raises(ValueError, match="invalid tool_command_template"):
            ToolExecutorConfig(tool_command_template="sleep {seconds}")
