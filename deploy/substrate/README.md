# Running inference-perf against GKE Agent Substrate

Manifests for running the tool-executor PoC on a GKE cluster with Agent
Substrate installed. See [docs/substrate_tool_execution.md](../../docs/substrate_tool_execution.md)
for what the tool executor does and the metrics it reports.

| File | What it creates |
|---|---|
| `worker-pool.yaml` | Namespace `ip-bench` (labeled `ate.dev/api-access=true`), ServiceAccount `inference-perf`, and a gVisor `WorkerPool` |
| `actor-template.yaml` | ActorTemplate `ip-bench/python-runtime`: the agent-sandbox runtime serving `POST /execute` on port 80 |
| `access-policy.yaml` | Editor on the `ip-bench` atespace for the inference-perf pod's SPIFFE identity |
| `vllm-qwen3-l4.yaml` | Optional LLM backend: vLLM serving Qwen3-0.6B (128K context via YaRN) on one spot L4 |
| `inference-perf-job.yaml` | The benchmark: config in a ConfigMap, a Job with the pod certificate and Substrate CA projected |

## Steps

1. **Install Substrate** (GKE Standard 1.37+), with the published images and the
   inference-perf pod allowed on the ingress router. With
   [substrate-gke](https://github.com/ai-on-gke/substrate-gke) (`docs/manual-gke-setup.md`)
   and a checkout of `agent-substrate/substrate` at `v1.0.0`:

   ```bash
   go -C tools/setup-gcp run . bootstrap            # substrate-gke checkout, .ate-dev-env.sh sourced
   go run ./cmd/ate-setup deploy ate-system \       # substrate checkout
     --image-repo ghcr.io/agent-substrate/substrate --image-tag v1.0.0 \
     --sandbox-classes=gvisor --cordon-system-infra-components \
     --credential-provider='{"name":"k8s.io"}' \
     --ingress-allowed-spiffe-ids=spiffe://cluster.local/ns/ip-bench/sa/inference-perf
   ```

2. **Worker pool, atespace, template, access** (`kubectl ate` from the same checkout):

   ```bash
   kubectl apply -f worker-pool.yaml
   kubectl create token -n ate-system ate-client --audience api.ate-system.svc --duration=24h > ate.token
   kubectl ate --token-file ate.token create atespace ip-bench
   kubectl ate --token-file ate.token create actor-template -f actor-template.yaml
   kubectl ate --token-file ate.token create access-policy -a ip-bench -f access-policy.yaml
   # Wait until the template has a GOLDEN TAG before running.
   kubectl ate --token-file ate.token get actor-template -a ip-bench
   ```

3. **LLM backend.** Point `server.base_url` at your llm-d gateway, or apply
   `vllm-qwen3-l4.yaml` after adding a spot L4 node pool. The Deployment sets
   `enableServiceLinks: false`: a Service named `vllm` otherwise injects
   `VLLM_PORT=tcp://...`, which vLLM refuses to start with.

4. **Trace and benchmark.** Build the image with the `otel` dependency group
   (it brings `grpcio`), put the trace in a ConfigMap, and run the Job:

   ```bash
   kubectl -n ip-bench create configmap tracelab-trace --from-file=tracelab_small.jsonl=<trace.jsonl>
   sed "s#image: IMAGE#image: <your image>#" inference-perf-job.yaml | kubectl apply -f -
   ```

   The Job keeps its pod after the run so the reports can be copied out:
   `kubectl -n ip-bench exec <pod> -- tar cf - -C /reports/run . | tar xf -`.
   Deleting the pod mid-run skips the tool executor's teardown; delete any
   leftover `ip-*` actors with `kubectl ate delete actor`.
