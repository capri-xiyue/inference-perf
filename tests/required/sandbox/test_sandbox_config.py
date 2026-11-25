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
"""The top-level sandbox config and where it is accepted."""

import pytest
from pydantic import ValidationError

from inference_perf.config.config import Config
from inference_perf.config.sandbox import SandboxType


def test_sandbox_is_rejected_for_data_types_without_per_tool_timing() -> None:
    with pytest.raises(ValidationError, match="sandbox requires data.type 'tracelab_trace_replay'"):
        Config.model_validate({"data": {"type": "mock"}, "sandbox": {"type": "fake_substrate"}})


def test_sandbox_is_accepted_with_tracelab_replay() -> None:
    cfg = Config.model_validate(
        {
            "load": {"type": "trace_session_replay", "stages": [{"concurrent_sessions": 1, "num_sessions": 1}]},
            "data": {"type": "tracelab_trace_replay", "tracelab_trace_replay": {"trace_files": ["t.jsonl"]}},
            "sandbox": {"type": "fake_substrate"},
        }
    )
    assert cfg.sandbox is not None and cfg.sandbox.type == SandboxType.FAKE_SUBSTRATE


def test_no_sandbox_by_default() -> None:
    assert Config().sandbox is None
