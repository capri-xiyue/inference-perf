# Tool Execution on GKE Agent Substrate (PoC)

> Status: proof of concept on branch `capri-xiyue/agentic-poc`.

Session replay (Weka, OTel, TraceLab and synthetic agentic traces) models
everything between two LLM calls — tool execution and user think time — as a
plain sleep of the recorded gap. With `tool_executor` set, that gap runs on a
[GKE Agent Substrate](https://github.com/agent-substrate/substrate) **actor**
instead. inference-perf is still the agent harness and LLM calls still go to the
model server (for example llm-d through the Inference Gateway), but each replayed
session now also exercises a real sandbox: the actor is created, suspended while
the user is idle, woken for the next turn, and runs every tool call.

The run then reports agent-session metrics from "Measuring Agent Sessions":
turn latency, **Wake-to-First-Token latency (Lwake-TTFT)**, actor wake, tool
execution time and `alpha_inf`.

## How a gap is executed

Each event's gap is classified before the event is dispatched, from the best evidence the trace has:

1. The session's first request is `idle` (the user starts the session).
2. A kind the trace recorded on the call (`agent.gap.kind` = `tool` / `idle`). TraceLab replay sets it from `timing_events`: a round fed by `tool_result` follows a tool gap, one fed by `user_message` follows user idle time. It also records the tools that ran (`gen_ai.tool.name`, from the previous round's `tool_calls`) and their wall time (`agent.tool.latency_ms`).
3. A request whose last message is `role: tool` (OTel traces) follows a tool gap.
4. Otherwise the gap length decides (Weka / AgentX traces record neither): at least `idle_gap_threshold_ms` (default 5000) is `idle`, shorter non-zero gaps are `tool`, zero is `none`.

| Gap | What the executor does |
|---|---|
| `idle` | Suspend the actor (`actor.suspend.between_turns`), wait out the rest of the recorded gap, then **wake** it. Starts a new turn. |
| `tool` | Wake the actor if needed, then run `actor.tool_call.command_template` in it (default `sleep {duration_sec}`, so the in-actor time matches the trace). The recorded tool names are reported in `agent_gap.tool_names`. |
| `none` | Nothing. |

Actors:

- By default every call — main agent and subagents alike — runs its gaps on the session's shared actor, as Claude Code subagents share their parent's machine. `actor.assignment.rules` give selected agents a **dedicated** actor: one per agent instance (`gen_ai.agent.id`, else `gen_ai.agent.name`), shared by that agent's parallel streams. Rules match the call's attributes with glob patterns; the first match wins, and `default` (`session`) applies otherwise. A dedicated rule on a call with no agent attributes falls back to the session actor.

  Agent identity comes from the OTel GenAI attributes `gen_ai.agent.name` / `gen_ai.agent.id`. OTel traces carry them on their spans (any other span attribute can be matched too); Weka replay sets them on subagent calls (`name` = `subagent_type`, `id` = `agent_id`). Main-agent calls in Weka carry neither, so they follow `default`.

  ```yaml
  actor:
    assignment:
      rules:                                         # everything else shares the session actor
        - match: {gen_ai.agent.name: "Explore"}      # code-search subagents get their own sandbox
          assign: dedicated
  ```

- Actors never talk to each other — inference-perf passes outputs between them, as it already does between graph events.
- Actors are deleted when their session drains (`actor.delete_on_session_end`), and any left at the end of a stage are deleted at teardown.

Waking:

Suspend and wake (the two suspend points are independent). A suspended actor is always woken by inference-perf calling `ResumeActor`, timed directly as `wake_sec`:

| Setting | Values | Meaning |
|---|---|---|
| `actor.suspend.between_turns` | `true` (default) / `false` | Suspend during user idle time between turns; the next turn wakes the actor |
| `actor.suspend.after_each_tool_call` | `false` (default) / `true` | Suspend after every tool call, off the request path while the next LLM call runs; the next tool call wakes the actor |
| `actor.suspend.snapshot_durability` | `async` (default) / `sync` | `SuspendActor` returns once the snapshot is on the node, or only after it reaches durable storage (Substrate `DurabilityMode`) |

## Metrics

Every request whose gap ran on the executor carries `info.agent_gap` in
`per_request_lifecycle_metrics.json`:

```json
"agent_gap": {
  "gap_kind": "idle", "executor": "substrate", "trace_wait_sec": 6.7,
  "actor": "ip-3f2a1c-6b0e...", "suspend_sec": 0.17,
  "wake_start": 812.31, "wake_sec": 0.37, "resumed": true, "cold": false,
  "tool_exec_sec": null, "gap_start": 805.43, "gap_end": 812.68
}
```

Sessions get `agent_turns` in `per_session_lifecycle_metrics.json`, and
`summary_session_lifecycle_metrics.json` gets an `agent_session` block, split
into `cold` (first turn: restore from the template's golden snapshot) and
`warm` turns:

| Field | Definition |
|---|---|
| `turn_latency_sec` | Lturn: wake trigger → end of the turn's last LLM call |
| `wake_to_first_token_sec` | **Lwake-TTFT**: wake trigger → first output token of the turn's first LLM call (= Lwake + Lharness + TTFT, measured) |
| `wake_sec` | Lwake (Lsandbox_restore): actor resume time |
| `harness_sec` | Lharness: actor ready → LLM request sent (inference-perf overhead) |
| `first_call_ttft_sec` | TTFT of the turn's first call alone |
| `alpha_inf` | Σ LLM call latency / Lturn (serial sum; can exceed 1 with parallel subagents) |
| `tool_exec_sec` | Ltool_exec per tool call (round trip through the router, incl. in-actor time) |
| `tool_wake_sec` | Wakes paid by tool calls (with `actor.suspend.after_each_tool_call`) |

Omitting `tool_executor` leaves every report byte-identical to a run without it.

## Configuration

```yaml
data:
  type: weka_trace_replay
  weka_trace_replay:
    hf_dataset_path: semianalysisai/cc-traces-weka-with-subagents-060826-256k
    tool_executor:
      type: substrate            # or fake_substrate (no cluster)
      idle_gap_threshold_ms: 5000
      actor:                     # backend-independent actor behavior
        name_prefix: ip
        assignment:              # which actor runs each gap
          default: session
          rules:
            - match: {gen_ai.agent.name: "Explore"}
              assign: dedicated
        suspend:                 # when and how actors are suspended
          between_turns: true
          after_each_tool_call: false
          snapshot_durability: async
        tool_call:               # what runs in the actor for a tool gap
          command_template: "sleep {duration_sec:.3f}"
          timeout_sec: 300
        delete_on_session_end: true
      substrate:                 # Substrate specifics: where actors live, how to reach it
        atespace: inference-perf
        actor_template: python-runtime
        api:
          endpoint: api.ate-system.svc:443
          ca_path: /run/servicedns-ca/ca.crt
          token_path: /run/ate-token/token
        router:
          url: https://atenet-router.ate-system.svc
          execute_path: /execute
          ca_path: /run/router-ca/ca.crt
          cert_bundle_path: /run/podidentity.podcert.ate.dev/credential-bundle.pem
```

`type: fake_substrate` runs the same lifecycle against an in-process stand-in
whose latencies come from `tool_executor.fake` (cold/warm resume, suspend,
create, exec overhead, jitter). Use it to try the metrics without a cluster.

## Running against a real Substrate install

Step-by-step manifests (worker pool, actor template, access policy, an optional
vLLM backend and the benchmark Job) are in [deploy/substrate](../deploy/substrate/README.md).

The backend talks to ateapi over gRPC (`substrate.ateapi.v1.Control`) and to
actors through `atenet-router`, so inference-perf runs as a Pod in the cluster:

1. Label the benchmark namespace `ate.dev/api-access=true` (ate-system's NetworkPolicies admit API clients only from labeled namespaces).
2. Authenticate to ateapi with a projected ServiceAccount token (audience `api.ate-system.svc`, `substrate.api.token_path`) or a pod certificate (`substrate.api.cert_bundle_path`); verify it with the `servicedns.podcert.ate.dev/identity` ClusterTrustBundle (`substrate.api.ca_path`).
3. `atenet-router` accepts mTLS only: project a pod certificate (`substrate.router.cert_bundle_path`) and add the Pod's SPIFFE ID to the router's `--ingress-allowed-spiffe-ids`.
4. Create an ActorTemplate whose container serves the agent-sandbox runtime API (`POST /execute {"command": ...}` → `{stdout, stderr, exit_code}`) on the actor's **default ingress port (80)**, plus a WorkerPool sized for the concurrency you replay.

## PoC limitations

- Tool output is not fed back into the next prompt; the recorded tool result is kept so prompt lengths and KV-cache behavior match the trace.
- The data plane uses the actor's default port; reaching a non-default port needs an HTTP `CONNECT` tunnel through the router, which is not implemented.
- Wake and the next LLM call are sequential. Overlapping them (KV prefetch / "Wake Hint" to the llm-d EPP) is not modeled yet.
- The LLM call does not traverse the actor's egress gateway (inference-perf is the harness, outside the actor), so Lgateway is not measured.
- `grpcio` and `protobuf` must be importable for `type: substrate` (they are installed with the `otel` extra).
