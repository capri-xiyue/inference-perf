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

from inference_perf.apis.agent_gap import AgentGapMetrics, GapKind
from inference_perf.config.datagen.toolexec import ResumeMode, SuspendMode, ToolExecutorConfig
from inference_perf.toolexec.base import GapContext, ToolExecutionError, ToolExecutor

logger = logging.getLogger(__name__)

_NAME_UNSAFE = re.compile(r"[^a-z0-9-]+")


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
    def __init__(self, config: ToolExecutorConfig, backend: SubstrateBackend, name: str, actor_name_prefix: str) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.name = name
        self._prefix = _NAME_UNSAFE.sub("-", actor_name_prefix.lower()).strip("-") or "ip"
        # Actor names must not collide across worker processes or runs.
        self._run_id = uuid.uuid4().hex[:6]
        self._actors: Dict[str, _ActorState] = {}

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
            await self.backend.suspend_actor(actor.name, synchronous=self.config.suspend_mode == SuspendMode.SYNC)
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
        """Wake the actor and record the wake on metrics (explicit, or a no-op request when implicit)."""
        await self._await_pending_suspend(actor)
        async with actor.lock:
            metrics.wake_start = _now()
            if not actor.suspended:
                metrics.wake_sec = 0.0
                metrics.resumed = False
                return
            metrics.cold = not actor.ever_resumed
            if self.config.resume_mode == ResumeMode.EXPLICIT:
                metrics.resumed = await self.backend.resume_actor(actor.name)
            else:
                # The router parks the request and resumes the actor; time the round trip.
                await self.backend.execute(actor.name, "true", self.config.tool_timeout_sec)
                metrics.resumed = True
                metrics.wake_estimated = True
            metrics.wake_sec = _now() - metrics.wake_start
            actor.suspended = False
            actor.ever_resumed = True

    # ------------------------------------------------------------------ gaps

    async def run_gap(self, ctx: GapContext) -> AgentGapMetrics:
        metrics = AgentGapMetrics(
            gap_kind=ctx.gap_kind,
            executor=self.name,
            trace_wait_sec=ctx.wait_sec,
            gap_start=_now(),
        )
        try:
            if ctx.gap_kind == GapKind.NONE:
                if ctx.wait_sec > 0:
                    await asyncio.sleep(ctx.wait_sec)
            elif ctx.gap_kind == GapKind.IDLE:
                await self._run_idle_gap(ctx, metrics)
            else:
                await self._run_tool_gap(ctx, metrics)
        except asyncio.CancelledError:
            raise
        except ToolExecutionError:
            raise
        except Exception as e:
            metrics.error = f"{type(e).__name__}: {e}"
            metrics.gap_end = _now()
            raise ToolExecutionError(metrics.error, metrics) from e
        metrics.gap_end = _now()
        return metrics

    async def _run_idle_gap(self, ctx: GapContext, metrics: AgentGapMetrics) -> None:
        actor = await self._get_actor(ctx, metrics)
        if self.config.suspend_on_idle:
            await self._await_pending_suspend(actor)
            metrics.suspend_sec = await self._suspend(actor)
        # The user is idle for the rest of the recorded gap; suspend time counts toward it.
        remaining = ctx.wait_sec - (_now() - metrics.gap_start)
        if remaining > 0:
            await asyncio.sleep(remaining)
        await self._wake(actor, metrics)

    async def _run_tool_gap(self, ctx: GapContext, metrics: AgentGapMetrics) -> None:
        actor = await self._get_actor(ctx, metrics)
        if self.config.resume_mode == ResumeMode.EXPLICIT:
            await self._wake(actor, metrics)
            if metrics.resumed is False:
                # Already running: no wake happened for this tool call.
                metrics.wake_start = None
                metrics.wake_sec = None
        else:
            await self._await_pending_suspend(actor)

        implicit_wake = self.config.resume_mode == ResumeMode.IMPLICIT and actor.suspended
        if implicit_wake:
            metrics.cold = not actor.ever_resumed
        command = self.config.tool_command_template.format(duration_sec=ctx.wait_sec)
        actor.inflight_tools += 1
        try:
            t0 = _now()
            result = await self.backend.execute(actor.name, command, self.config.tool_timeout_sec)
            metrics.tool_exec_sec = _now() - t0
        finally:
            actor.inflight_tools -= 1
        metrics.tool_exit_code = result.exit_code
        if implicit_wake:
            # The router resumed the actor inside the call; whatever exceeds the tool's own
            # duration is attributed to the wake.
            metrics.wake_start = t0
            metrics.wake_sec = max(0.0, metrics.tool_exec_sec - ctx.wait_sec)
            metrics.wake_estimated = True
            metrics.resumed = True
            actor.suspended = False
            actor.ever_resumed = True
        if result.exit_code != 0:
            logger.debug("Tool in actor %s exited %d (event %s)", actor.name, result.exit_code, ctx.event_id)

        if self.config.suspend_during_inference and actor.pending_suspend is None:
            # Suspend while the next LLM call runs, off the request's critical path.
            actor.pending_suspend = asyncio.get_running_loop().create_task(self._suspend(actor))

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
        if not self.config.delete_actors_on_session_end:
            return
        keys = [k for k, a in self._actors.items() if a.session_id == session_id]
        actors: List[_ActorState] = [self._actors.pop(k) for k in keys]
        await asyncio.gather(*(self._delete(a) for a in actors))

    async def aclose(self) -> None:
        await self.drain_background()
        actors = list(self._actors.values())
        self._actors.clear()
        if self.config.delete_actors_on_session_end:
            await asyncio.gather(*(self._delete(a) for a in actors))
        await self.backend.aclose()
