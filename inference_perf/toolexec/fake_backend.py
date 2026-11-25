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
"""In-process stand-in for GKE Agent Substrate with a configurable latency model.

Lets the tool-execution path, metrics and reports run end to end without a
cluster. ``sleep <seconds>`` commands sleep for that long; any other command
costs only the configured exec overhead.
"""

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from inference_perf.config.datagen.toolexec import FakeSubstrateConfig
from inference_perf.toolexec.substrate_executor import ExecResult, SubstrateBackend

_SLEEP_COMMAND = re.compile(r"^\s*sleep\s+([0-9]*\.?[0-9]+)\s*$")


@dataclass
class _FakeActor:
    suspended: bool = True
    ever_resumed: bool = False


class FakeSubstrateBackend(SubstrateBackend):
    def __init__(self, config: FakeSubstrateConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self._actors: Dict[str, _FakeActor] = {}
        # (operation, actor) in call order, for tests and debugging.
        self.calls: List[Tuple[str, str]] = []

    async def _delay(self, ms: float) -> None:
        if ms <= 0:
            return
        j = self.config.jitter_fraction
        factor = 1.0 + self._rng.uniform(-j, j) if j > 0 else 1.0
        await asyncio.sleep(ms * factor / 1000.0)

    def _get(self, name: str) -> _FakeActor:
        actor = self._actors.get(name)
        if actor is None:
            raise RuntimeError(f"actor {name!r} not found")
        return actor

    async def create_actor(self, name: str) -> None:
        self.calls.append(("create", name))
        if name in self._actors:
            raise RuntimeError(f"actor {name!r} already exists")
        await self._delay(self.config.create_latency_ms)
        self._actors[name] = _FakeActor()

    async def resume_actor(self, name: str) -> bool:
        self.calls.append(("resume", name))
        actor = self._get(name)
        if not actor.suspended:
            return False
        await self._delay(self.config.warm_resume_latency_ms if actor.ever_resumed else self.config.cold_resume_latency_ms)
        actor.suspended = False
        actor.ever_resumed = True
        return True

    async def suspend_actor(self, name: str, synchronous: bool) -> None:
        self.calls.append(("suspend", name))
        actor = self._get(name)
        if actor.suspended:
            return
        await self._delay(self.config.suspend_latency_ms)
        actor.suspended = True

    async def delete_actor(self, name: str) -> None:
        self.calls.append(("delete", name))
        self._get(name)
        await self._delay(self.config.delete_latency_ms)
        del self._actors[name]

    async def execute(self, name: str, command: str, timeout_sec: float) -> ExecResult:
        self.calls.append(("execute", name))
        actor = self._get(name)
        if actor.suspended:
            # Request parking: the router resumes the actor before forwarding.
            await self.resume_actor(name)
        await self._delay(self.config.exec_overhead_ms)
        match = _SLEEP_COMMAND.match(command)
        if match:
            await asyncio.sleep(min(float(match.group(1)), timeout_sec))
        return ExecResult(exit_code=0)

    async def aclose(self) -> None:
        self._actors.clear()
