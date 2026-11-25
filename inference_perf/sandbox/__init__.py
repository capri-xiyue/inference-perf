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
"""Sandbox: run session-replay gaps on an agent sandbox runtime.

inference-perf acts as the agent harness. LLM calls go to the model server as
usual; the gap before each call (recorded tool calls or user idle time) is
handed to a ToolExecutor, which runs it on a sandbox actor and reports actor
wake, tool execution and suspend timings as an AgentGapMetrics record.
"""

from inference_perf.sandbox.base import (
    GapContext,
    PlannedTool,
    ToolExecutionError,
    ToolExecutor,
    classify_gap,
    parse_planned_tools,
    resolve_actor_key,
)
from inference_perf.sandbox.factory import build_tool_executor

__all__ = [
    "GapContext",
    "PlannedTool",
    "ToolExecutionError",
    "ToolExecutor",
    "build_tool_executor",
    "classify_gap",
    "parse_planned_tools",
    "resolve_actor_key",
]
