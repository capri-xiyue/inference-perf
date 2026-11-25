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
from typing import Optional

from inference_perf.config.sandbox import SandboxConfig, SandboxType
from inference_perf.sandbox.base import ToolExecutor
from inference_perf.sandbox.substrate_executor import SubstrateToolExecutor


def build_tool_executor(config: SandboxConfig, seed: Optional[int] = None) -> ToolExecutor:
    if config.type == SandboxType.FAKE_SUBSTRATE:
        from inference_perf.sandbox.fake_backend import FakeSubstrateBackend

        return SubstrateToolExecutor(config, FakeSubstrateBackend(config.fake, seed=seed), name="fake_substrate")
    if config.type == SandboxType.SUBSTRATE:
        from inference_perf.sandbox.ate_backend import AteSubstrateBackend

        assert config.substrate is not None  # enforced by SandboxConfig
        return SubstrateToolExecutor(config, AteSubstrateBackend(config.substrate), name="substrate")
    raise ValueError(f"Unsupported sandbox type: {config.type}")
