# Running inference-perf against GKE Agent Substrate

Manifests for the sandbox benchmark: TraceLab session replay with every recorded
tool call executed in a GKE Agent Substrate actor and LLM calls served by llm-d.
See [docs/sandbox.md](../../docs/sandbox.md) for what the sandbox does and the
metrics it reports.

| File | What it creates |
|---|---|
| `worker-pool.yaml` | Namespace `ip-bench` (labeled `ate.dev/api-access=true`), ServiceAccount `inference-perf`, and a gVisor `WorkerPool` |
| `actor-template.yaml` | ActorTemplate `ip-bench/python-runtime`: the agent-sandbox runtime serving `POST /execute` on port 80 |
| `access-policy.yaml` | Editor on the `ip-bench` atespace for the inference-perf pod's SPIFFE identity |
| `llm-d/` | Kustomize overlay of the llm-d optimized-baseline model server: Qwen3-30B-A3B-Instruct-2507-FP8, TP1, 256K context, one replica per H100 |
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

2. **Worker pool, atespace, template, access** (`kubectl ate` from the same checkout).
   Size the WorkerPool for the number of concurrent sessions (one actor each):

   ```bash
   kubectl apply -f worker-pool.yaml
   kubectl create token -n ate-system ate-client --audience api.ate-system.svc --duration=24h > ate.token
   kubectl ate --token-file ate.token create atespace ip-bench
   kubectl ate --token-file ate.token create actor-template -f actor-template.yaml
   kubectl ate --token-file ate.token create access-policy -a ip-bench -f access-policy.yaml
   # Wait until the template has a GOLDEN TAG before running.
   kubectl ate --token-file ate.token get actor-template -a ip-bench
   ```

3. **LLM backend: llm-d in Gateway mode.** Enable the GKE Gateway API, create a
   `gke-l7-rilb` Gateway, install the llm-d router from the
   [optimized-baseline guide](https://github.com/llm-d/llm-d/tree/main/guides/optimized-baseline)
   with `provider.name=gke,httpRoute.create=true`, then the model servers on an
   H100 node pool (a3-highgpu-4g: 4 replicas; a3-megagpu-8g: set `replicas: 8`):

   ```bash
   kubectl create namespace llm-d-bench
   kubectl -n llm-d-bench create secret generic llm-d-hf-token --from-literal=HF_TOKEN=<token or empty>
   kubectl -n llm-d-bench apply -k llm-d/
   ```

   Put the Gateway address in `server.base_url` of `inference-perf-job.yaml`.

4. **Trace and benchmark.** Export whole TraceLab sessions (do not cut sessions
   short; the config's `filter` skips sessions that do not fit the model), put
   them in a ConfigMap, build the image with the `otel` dependency group (it
   brings `grpcio`), and run the Job:

   ```bash
   kubectl -n ip-bench create configmap tracelab-trace --from-file=tracelab.jsonl=<trace.jsonl>
   sed "s#image: IMAGE#image: <your image>#" inference-perf-job.yaml | kubectl apply -f -
   ```

   The Job keeps its pod after the run so the reports can be copied out:
   `kubectl -n ip-bench exec <pod> -- tar cf - -C /reports/run . | tar xf -`.
   Deleting the pod mid-run skips the sandbox teardown; delete any leftover
   `ip-*` actors with `kubectl ate delete actor`.
