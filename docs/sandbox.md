# Sandbox: replaying tool calls on GKE Agent Substrate

During TraceLab session replay, the gap between two LLM calls is either tool
execution or user idle time. By default inference-perf sleeps through it. With a
top-level `sandbox` config, the gap runs on a **sandbox actor** instead
([GKE Agent Substrate](https://github.com/agent-substrate/substrate)), so the
benchmark also measures what the sandbox adds to an agent session: actor wake,
suspend, and tool dispatch through the router.

`server` is the LLM under test; `sandbox` is the tool-execution side under test.
inference-perf acts as the agent harness: LLM calls go to the model server as
usual, and each recorded tool call runs in the session's actor.

> [!NOTE]
> The sandbox is supported with `tracelab_trace_replay` only. TraceLab records
> every tool call with its own timestamps and which input triggered each round;
> other traces (Weka, OTel without tool spans) only record one undifferentiated
> gap. Configuring `sandbox` with another `data.type` is rejected at startup.

## What happens in a session

The replay stays closed-loop: each LLM call is sent after its predecessor
completes plus the recorded gap. Token counts, prefix reuse and gaps come from
the trace; LLM latency is measured live.

1. **Session start**: create an actor from the ActorTemplate and wake it (cold:
   restore from the template's golden snapshot). This starts turn 0.
2. **LLM call**: as in plain TraceLab replay.
3. **Gap before the next round**, classified from the trace:

   | The next round was triggered by | Gap | What the sandbox does |
   |---|---|---|
   | `tool_result` (last input event) | tool | Wake the actor if suspended, then run each tool the previous round emitted, concurrently, at its recorded offset |
   | `user_message` (last input event) | idle | Suspend the actor (`actor.suspend.between_turns`), wait out the recorded idle time, wake it. Starts the next turn |
   | no input events, previous round emitted tools | tool | As above |
   | no input events, no tools | none | Plain wait (the recorded gap is zero) |

4. **Session end**: delete the actor (`actor.delete_on_session_end`).

### Replaying each tool call

A tool gap starts when the previous LLM call ends (`prev_end`: the round's last
model-output event). Tools only run after the LLM call has finished, so each tool
covers the part of its recorded window that falls after that point:

- start `max(emitted_at − prev_end, 0)`, end `result_at − prev_end`
- executed in the actor as `sleep <end − start>`
- a tool that finished while the response was still being generated runs
  `sleep 0` (one router round trip)

Every tool result therefore lands when the trace recorded it, and the gap keeps
its recorded length. After the tools finish, any remainder of the recorded gap is
waited out client-side. Tools without timestamps start at the gap start and run
for their recorded latency.

`tool_wall_latency_ms` is measured from the moment the tool call appears in the
response (`result_at − emitted_at`), so the trace cannot tell whether a tool
started while the response was streaming or only after it. Replaying from
`prev_end` keeps the gap faithful either way; it does not model actor occupancy
during the LLM output, and with `after_each_tool_call` the wake sits on the
critical path instead of overlapping the output (pessimistic for the sandbox).

### Actors

- By default every call runs its gaps on the session's shared actor.
  `actor.assignment.rules` give selected agents a **dedicated** actor, one per
  agent instance (`gen_ai.agent.id`, else `gen_ai.agent.name`). Rules match call
  attributes with glob patterns; the first match wins. TraceLab calls carry no
  agent identity, so with TraceLab every session uses one actor.
- Suspend points are independent: `actor.suspend.between_turns` (default on) and
  `actor.suspend.after_each_tool_call` (default off; the next tool gap then wakes
  the actor). `actor.suspend.snapshot_durability` picks Substrate's
  `DurabilityMode`: `async` returns once the snapshot is on the node, `sync` once
  it is in durable storage.
- A suspended actor is always woken by inference-perf calling `ResumeActor`.

## Metrics

Three report files gain sandbox fields; runs without `sandbox` are unchanged.

**`per_request_lifecycle_metrics.json`**: `info.agent_gap` describes the gap
before the request:

```json
"agent_gap": {
  "turn_index": 0, "gap_kind": "tool", "executor": "substrate",
  "trace_wait_sec": 1.55, "gap_start": 812.31, "gap_end": 813.89,
  "actor": "ip-3f2a1c-6b0e…", "actor_create_sec": null, "suspend_sec": null,
  "actor_ready_start": 812.31, "actor_ready_sec": null, "cold": null,
  "tools": [
    {"name": "Read", "start_sec": 0.0, "duration_sec": 0.0,  "tool_exec_sec": 0.026, "tool_exec_overhead_sec": 0.026, "exit_code": 0, "error": null},
    {"name": "Bash", "start_sec": 0.0, "duration_sec": 1.55, "tool_exec_sec": 1.578, "tool_exec_overhead_sec": 0.028, "exit_code": 0, "error": null}
  ],
  "error": null
}
```

- `actor_ready_sec`: from the moment the gap needs the actor until it serves,
  including waiting out a background suspend still in flight. `null` when the
  actor was already running.
- `tool_exec_overhead_sec` = `tool_exec_sec − duration_sec`: router and sandbox
  overhead per tool call.

**`per_session_lifecycle_metrics.json`**: per session, `agent_turns` (one entry
per turn: `turn_index`, `cold`, `turn_latency_sec`, `success`) and `sandbox`
(counts and simple aggregates: actors, create time, wakes and their cold / warm
latency, suspends, tool calls and their overhead, tool errors).

`turn_latency_sec` runs from the moment the turn needs the actor (session start,
or the end of the user's idle time) to the end of the turn's last LLM call.

**`summary_session_lifecycle_metrics.json`**: `agent_session`, over all turns of
all sessions:

| Field | Definition |
|---|---|
| `turn_latency_sec` | `all` / `cold` (first turn of each session) / `warm` |
| `sandbox.actor_ready_sec` | Wake latency, `all` / `cold` / `warm` |
| `sandbox.suspend_sec` | SuspendActor latency |
| `sandbox.actor_create_sec` | CreateActor latency |
| `sandbox.tool_exec_overhead_sec` | Router + sandbox overhead per tool call |
| `sandbox.tool_error_rate` | Tool calls that errored or exited non-zero |

Turns are pooled, so sessions with many turns weigh more; cold samples number
one per session.

## Configuration

```yaml
sandbox:
  type: substrate                  # substrate | fake_substrate
  actor:                           # backend-independent actor behavior
    name_prefix: ip
    assignment:
      default: session             # session | dedicated
      rules: []                    # e.g. - match: {gen_ai.agent.name: "Explore"}
                                   #        assign: dedicated
    suspend:
      between_turns: true
      after_each_tool_call: false
      snapshot_durability: async   # async | sync
    tool_call:
      timeout_sec: 300
    delete_on_session_end: true
  substrate:                       # Substrate specifics (type: substrate)
    atespace: ip-bench
    actor_template: python-runtime
    api:
      endpoint: api.ate-system.svc:443
      server_name: api.ate-system.svc
      ca_path: /run/ate/ca.crt
      cert_bundle_path: /run/ate/credential-bundle.pem   # or token_path
    router:
      url: https://atenet-router.ate-system.svc
      execute_path: /execute
      ca_path: /run/ate/ca.crt
      cert_bundle_path: /run/ate/credential-bundle.pem
    request_timeout_sec: 120

data:
  type: tracelab_trace_replay
  tracelab_trace_replay:
    trace_files: [/data/tracelab.jsonl]
    filter: "lambda x: x['max_tokens'] < 262144"   # skip sessions that do not fit the model
    trace_idle_gap_cap_seconds: null               # replay every gap as recorded
```

`type: fake_substrate` runs the same lifecycle against an in-process stand-in
whose latencies come from `sandbox.fake` (create, cold / warm resume, snapshot
capture, snapshot upload, exec overhead, jitter); defaults are rounded from a
real Substrate run. An `async` suspend waits for the capture only, a `sync` one
for the upload too. The fake has no nodes, so durability shows up as latency
only. With every latency at zero it is an "ideal sandbox" baseline to compare a
real run against.

## Running against a real Substrate install

Step-by-step manifests (worker pool, actor template, access policy, an llm-d
backend and the benchmark Job) are in [deploy/substrate](../deploy/substrate/README.md).

The backend talks to ateapi over gRPC (`substrate.ateapi.v1.Control`) and to
actors through `atenet-router`, so inference-perf runs as a Pod in the cluster:

1. Label the benchmark namespace `ate.dev/api-access=true` (ate-system's NetworkPolicies admit API clients only from labeled namespaces).
2. Authenticate to ateapi with a projected ServiceAccount token (audience `api.ate-system.svc`, `substrate.api.token_path`) or a pod certificate (`substrate.api.cert_bundle_path`); verify it with the `servicedns.podcert.ate.dev/identity` ClusterTrustBundle (`substrate.api.ca_path`).
3. `atenet-router` accepts mTLS only: project a pod certificate (`substrate.router.cert_bundle_path`) and add the Pod's SPIFFE ID to the router's `--ingress-allowed-spiffe-ids`.
4. Create an ActorTemplate whose container serves the agent-sandbox runtime API (`POST /execute {"command": ...}` → `{stdout, stderr, exit_code}`) on the actor's **default ingress port (80)**, plus a WorkerPool sized for the concurrency you replay.

## Limitations

- Tools are replayed as `sleep`: no CPU or I/O load, so real command cost under gVisor and node contention are not measured.
- The recorded tool latency already includes the original machine's dispatch overhead; the sandbox overhead is added on top, so turn latency is an upper bound.
- Tool latency may include human approval time (permission prompts), which the trace does not separate.
- Tools start only after the LLM call ends (see above).
- The LLM call does not traverse the actor's egress gateway (inference-perf is the harness, outside the actor).
- `grpcio` and `protobuf` must be importable for `type: substrate` (they are installed with the `otel` extra).
