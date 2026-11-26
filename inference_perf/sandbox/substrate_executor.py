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
"""Run replay gaps on GKE Agent Substrate actors.

The executor owns the actor lifecycle policy (create, suspend on idle, wake,
execute, delete) and the timing; a SubstrateBackend performs the calls, either
against a real installation (AteSubstrateBackend) or an in-process fake.
"""

import asyncio
import hashlib
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from inference_perf.apis.agent_gap import AgentGapMetrics, GapKind, ToolCallMetrics
from inference_perf.config.sandbox import SandboxConfig, SnapshotDurability
from inference_perf.sandbox.base import GapContext, PlannedTool, ToolExecutionError, ToolExecutor

logger = logging.getLogger(__name__)

_NAME_UNSAFE = re.compile(r"[^a-z0-9-]+")

# Longest single execute call. atenet-router answers 504 to requests that run
# past ~300s, so a longer tool is replayed as consecutive sleeps of at most this.
MAX_EXEC_CHUNK_SEC = 240.0


def _sleep_chunks(duration_sec: float) -> List[float]:
    """Split a replayed tool duration into router-safe sleeps (at least one, possibly zero-length)."""
    chunks: List[float] = []
    remaining = max(duration_sec, 0.0)
    while remaining > MAX_EXEC_CHUNK_SEC:
        chunks.append(MAX_EXEC_CHUNK_SEC)
        remaining -= MAX_EXEC_CHUNK_SEC
    chunks.append(remaining)
    return chunks


@dataclass(frozen=True)
class ExecResult:
    exit_code: int


class SubstrateBackend(ABC):
    """The Substrate operations the executor needs. Implementations must be async-safe."""

    @abstractmethod
    async def create_actor(self, name: str) -> None: ...

    @abstractmethod
    async def resume_actor(self, name: str) -> bool:
        """Resume the actor; return True if it was restored, False if it was already running."""

    @abstractmethod
    async def suspend_actor(self, name: str, synchronous: bool) -> None: ...

    @abstractmethod
    async def delete_actor(self, name: str) -> None: ...

    @abstractmethod
    async def execute(self, name: str, command: str, timeout_sec: float) -> ExecResult:
        """Run a command in the actor through the data plane; a suspended actor is resumed by the router."""

    @abstractmethod
    async def aclose(self) -> None: ...


@dataclass
class _ActorState:
    name: str
    session_id: str
    created: bool = False
    # CreateActor leaves an actor SUSPENDED, holding the template's golden snapshot.
    suspended: bool = True
    ever_resumed: bool = False
    inflight_tools: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending_suspend: Optional["asyncio.Task[Optional[float]]"] = None


def _now() -> float:
    return time.perf_counter()


class SubstrateToolExecutor(ToolExecutor):
    def __init__(self, config: SandboxConfig, backend: SubstrateBackend, name: str) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.name = name
        self._prefix = _NAME_UNSAFE.sub("-", config.actor.name_prefix.lower()).strip("-") or "ip"
        # Actor names must not collide across worker processes or runs.
        self._run_id = uuid.uuid4().hex[:6]
        self._actors: Dict[str, _ActorState] = {}
        # Current turn of each session; an idle gap starts the next one.
        self._turns: Dict[str, int] = {}

    def actor_name(self, actor_key: str) -> str:
        digest = hashlib.sha1(actor_key.encode("utf-8")).hexdigest()[:12]
        return f"{self._prefix}-{self._run_id}-{digest}"

    # ------------------------------------------------------------------ lifecycle

    async def _get_actor(self, ctx: GapContext, metrics: AgentGapMetrics) -> _ActorState:
        key = ctx.actor_key
        actor = self._actors.get(key)
        if actor is None:
            actor = _ActorState(name=self.actor_name(key), session_id=ctx.session_id)
            self._actors[key] = actor
        metrics.actor = actor.name
        async with actor.lock:
            if not actor.created:
                t0 = _now()
                await self.backend.create_actor(actor.name)
                metrics.actor_create_sec = _now() - t0
                actor.created = True
        return actor

    async def _suspend(self, actor: _ActorState) -> Optional[float]:
        async with actor.lock:
            if actor.suspended or actor.inflight_tools > 0 or not actor.created:
                return None
            t0 = _now()
            await self.backend.suspend_actor(
                actor.name, synchronous=self.config.actor.suspend.snapshot_durability == SnapshotDurability.SYNC
            )
            actor.suspended = True
            return _now() - t0

    async def _await_pending_suspend(self, actor: _ActorState) -> None:
        task = actor.pending_suspend
        if task is not None:
            actor.pending_suspend = None
            try:
                await task
            except Exception as e:
                logger.warning("Background suspend of actor %s failed: %s", actor.name, e)

    async def _wake(self, actor: _ActorState, metrics: AgentGapMetrics) -> None:
        """Make the actor serve, recording the wake on metrics.

        The wake starts when the actor is needed: waiting out a background suspend
        still in flight (after_each_tool_call) is part of it. Leaves actor_ready_sec
        unset when the actor was already running.
        """
        ready_start = _now()
        metrics.actor_ready_start = ready_start
        await self._await_pending_suspend(actor)
        async with actor.lock:
            if not actor.suspended:
                return
            metrics.cold = not actor.ever_resumed
            await self.backend.resume_actor(actor.name)
            metrics.actor_ready_sec = _now() - ready_start
            actor.suspended = False
            actor.ever_resumed = True

    # ------------------------------------------------------------------ gaps

    async def run_gap(self, ctx: GapContext) -> AgentGapMetrics:
        if ctx.gap_kind == GapKind.IDLE:
            self._turns[ctx.session_id] = self._turns.get(ctx.session_id, -1) + 1
        metrics = AgentGapMetrics(
            turn_index=max(self._turns.get(ctx.session_id, 0), 0),
            gap_kind=ctx.gap_kind,
            executor=self.name,
            trace_wait_sec=ctx.wait_sec,
            gap_start=_now(),
        )
        try:
            if ctx.gap_kind == GapKind.IDLE:
                await self._run_idle_gap(ctx, metrics)
            elif ctx.gap_kind == GapKind.TOOL and ctx.tools:
                await self._run_tool_gap(ctx, metrics)
            # Unclassified gaps, and tool gaps without recorded tools, keep the plain wait.
            await self._wait_rest_of_gap(ctx, metrics)
        except asyncio.CancelledError:
            raise
        except ToolExecutionError:
            raise
        except Exception as e:
            metrics.error = f"{type(e).__name__}: {e}"
            metrics.gap_end = _now()
            raise ToolExecutionError(metrics.error, metrics) from e
        metrics.gap_end = _now()
        if any(t.error is not None for t in metrics.tools):
            failed = next(t for t in metrics.tools if t.error is not None)
            metrics.error = f"tool {failed.name}: {failed.error}"
            raise ToolExecutionError(metrics.error, metrics)
        return metrics

    @staticmethod
    async def _wait_rest_of_gap(ctx: GapContext, metrics: AgentGapMetrics) -> None:
        remaining = ctx.wait_sec - (_now() - metrics.gap_start)
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def _run_idle_gap(self, ctx: GapContext, metrics: AgentGapMetrics) -> None:
        actor = await self._get_actor(ctx, metrics)
        if self.config.actor.suspend.between_turns:
            await self._await_pending_suspend(actor)
            metrics.suspend_sec = await self._suspend(actor)
        # The user is idle for the rest of the recorded gap; suspend time counts toward it.
        await self._wait_rest_of_gap(ctx, metrics)
        await self._wake(actor, metrics)

    async def _run_tool_gap(self, ctx: GapContext, metrics: AgentGapMetrics) -> None:
        """Wake the actor once, then run every recorded tool at its recorded offset, concurrently."""
        actor = await self._get_actor(ctx, metrics)
        await self._wake(actor, metrics)
        metrics.tools = [ToolCallMetrics(name=t.name, start_sec=t.start_sec, duration_sec=t.duration_sec) for t in ctx.tools]
        actor.inflight_tools += 1
        try:
            await asyncio.gather(
                *(
                    self._run_tool(actor, ctx, metrics.gap_start, planned, m)
                    for planned, m in zip(ctx.tools, metrics.tools, strict=True)
                )
            )
        finally:
            actor.inflight_tools -= 1
        if self.config.actor.suspend.after_each_tool_call and actor.pending_suspend is None:
            # Suspend while the next LLM call runs, off the request's critical path.
            actor.pending_suspend = asyncio.get_running_loop().create_task(self._suspend(actor))

    async def _run_tool(
        self, actor: _ActorState, ctx: GapContext, gap_start: float, planned: PlannedTool, m: ToolCallMetrics
    ) -> None:
        delay = gap_start + planned.start_sec - _now()
        if delay > 0:
            await asyncio.sleep(delay)
        t0 = _now()
        result = ExecResult(exit_code=0)
        try:
            for chunk in _sleep_chunks(planned.duration_sec):
                result = await self.backend.execute(actor.name, f"sleep {chunk:.3f}", self.config.actor.tool_call.timeout_sec)
                if result.exit_code != 0:
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            m.error = f"{type(e).__name__}: {e}"
            return
        finally:
            m.tool_exec_sec = _now() - t0
            m.tool_exec_overhead_sec = m.tool_exec_sec - planned.duration_sec
        m.exit_code = result.exit_code
        if result.exit_code != 0:
            logger.debug("Tool %s in actor %s exited %d (event %s)", planned.name, actor.name, result.exit_code, ctx.event_id)

    # ------------------------------------------------------------------ teardown

    async def _delete(self, actor: _ActorState) -> None:
        await self._await_pending_suspend(actor)
        if not actor.created:
            return
        try:
            await self.backend.delete_actor(actor.name)
        except Exception as e:
            logger.warning("Failed to delete actor %s: %s", actor.name, e)

    async def end_session(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
        if not self.config.actor.delete_on_session_end:
            return
        keys = [k for k, a in self._actors.items() if a.session_id == session_id]
        actors: List[_ActorState] = [self._actors.pop(k) for k in keys]
        await asyncio.gather(*(self._delete(a) for a in actors))

    async def aclose(self) -> None:
        await self.drain_background()
        actors = list(self._actors.values())
        self._actors.clear()
        if self.config.actor.delete_on_session_end:
            await asyncio.gather(*(self._delete(a) for a in actors))
        await self.backend.aclose()
