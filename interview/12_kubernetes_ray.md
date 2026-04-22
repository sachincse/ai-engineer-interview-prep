# Chapter 12 — Kubernetes, Ray, Docker
## Container orchestration for AI workloads — a bench-gap area for you, study hard

> Your resume lists Kubernetes, OpenShift, Docker, Ray in skills. JD explicitly calls out **Kubernetes + Ray + LLMOps practices**. This chapter closes the depth gap.

---

## 12.1 Docker — the 5-minute refresher

```
Dockerfile → docker build → Image → docker push → Registry (ECR/ACR)
                                                         │
                                                         ▼
                                              docker pull → Container runs
```

### Key concepts
- **Image** — read-only template; layers stacked for cacheability
- **Container** — running instance of an image with a read-write layer
- **Registry** — storage for images (ECR, ACR, Docker Hub, Harbor)
- **Multi-stage builds** — `FROM X as builder` → `FROM Y` copies artifacts only. Smaller prod images.

### ML Dockerfile best practices
```dockerfile
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS base
WORKDIR /app

# Install deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code later (invalidation-aware)
COPY src/ src/

# Non-root user
RUN useradd -m runner
USER runner

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Image size matters
- ML images are 5-15 GB typical → slow pulls
- Use **slim base** (`python:3.11-slim` for CPU, `nvidia/cuda:X-runtime` for GPU)
- Prune dev deps, delete pip cache
- Multi-stage builds for build vs runtime separation

---

## 12.2 Kubernetes — the mental model

```
                      ┌─────────── Kubernetes Cluster ────────────┐
                      │                                            │
                      │  ┌──── Control Plane ────┐                │
                      │  │ API Server            │                │
                      │  │ etcd                  │                │
                      │  │ Scheduler             │                │
                      │  │ Controller Manager    │                │
                      │  └──────────────────────┘                 │
                      │                                            │
                      │  ┌──── Worker Nodes ────────────────────┐  │
                      │  │                                       │  │
                      │  │  ┌── Node 1 ──┐   ┌── Node 2 ──┐    │  │
                      │  │  │  kubelet   │   │  kubelet    │    │  │
                      │  │  │  kube-proxy│   │  kube-proxy │    │  │
                      │  │  │            │   │             │    │  │
                      │  │  │  Pod A     │   │  Pod C      │    │  │
                      │  │  │  Pod B     │   │  Pod D      │    │  │
                      │  │  └────────────┘   └─────────────┘    │  │
                      │  └───────────────────────────────────────┘  │
                      └──────────────────────────────────────────────┘
```

---

## 12.3 Kubernetes primitives (the must-knows)

### Pod
Smallest deployable unit. One or more containers sharing net/storage namespaces. **Ephemeral — if Pod dies, replacement has a new IP.**

### Deployment
Manages replica count of stateless Pods. Rolling updates, rollbacks.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: rag-api }
spec:
  replicas: 3
  selector: { matchLabels: { app: rag-api } }
  template:
    metadata: { labels: { app: rag-api } }
    spec:
      containers:
      - name: rag-api
        image: acme/rag-api:abc123
        resources:
          requests: { cpu: 500m, memory: 1Gi }
          limits:   { cpu: 2,    memory: 4Gi }
        readinessProbe: { httpGet: { path: /health, port: 8080 } }
        livenessProbe:  { httpGet: { path: /health, port: 8080 } }
```

### Service
Stable virtual IP + DNS name pointing at Pods matching a selector.

- **ClusterIP** — internal only
- **NodePort** — exposed on every node (rare in production)
- **LoadBalancer** — cloud LB provisioned (AWS NLB/ALB)
- **Headless** — direct Pod IPs, used for stateful services

### Ingress
HTTP routing layer in front of Services. Path/host-based routing, TLS termination.

### ConfigMap / Secret
Inject config / sensitive data into Pods.

### Namespace
Logical separation (multi-tenant, multi-env within a cluster).

### StatefulSet
Stable Pod identities (for DBs, distributed systems). Use sparingly — most ML isn't stateful.

### DaemonSet
Run a Pod on every node (e.g., node-exporter, log shipper).

### Job / CronJob
One-shot or scheduled batch workloads.

---

## 12.4 Networking

```
Service (ClusterIP 10.0.1.5:80)
    │
    ▼
Pod 1 (10.1.1.1:8080)
Pod 2 (10.1.1.2:8080)
Pod 3 (10.1.1.3:8080)
```

- **kube-proxy** updates iptables / IPVS rules to route Service IPs to Pods
- **CoreDNS** resolves `rag-api.default.svc.cluster.local`
- **NetworkPolicy** — firewall rules for Pod-to-Pod (multi-tenant hygiene)

---

## 12.5 Resource management

### Requests vs Limits
- **Requests** — guaranteed minimum; scheduler uses for placement
- **Limits** — hard maximum; container throttled (CPU) or OOMKilled (memory)

### QoS classes
- **Guaranteed** — requests == limits
- **Burstable** — requests < limits
- **BestEffort** — none set (risky)

### Autoscaling
- **HPA** (Horizontal Pod Autoscaler) — scale replicas by CPU/memory/custom metric
- **VPA** (Vertical Pod Autoscaler) — adjust requests/limits of a single Pod
- **Cluster Autoscaler** — add/remove nodes
- **Karpenter** (AWS) — faster, smarter node autoscaler
- **KEDA** (Kubernetes Event-Driven Autoscaling) — scale on external metrics (Kafka lag, SQS depth, Prometheus queries, GPU util). **Essential for ML.**

---

## 12.6 GPU on Kubernetes

### The stack
- **NVIDIA device plugin** — advertises GPUs as resources
- **Node labels/taints** — isolate GPU nodes
- **Resource requests:** `nvidia.com/gpu: 1`
- **MIG (Multi-Instance GPU)** on A100/H100 — partition into up to 7 instances per GPU

### GPU scheduling patterns
```yaml
spec:
  nodeSelector: { nvidia.com/gpu.product: NVIDIA-A100-PCIe-40GB }
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  containers:
  - name: model
    resources:
      limits: { nvidia.com/gpu: 1, memory: 40Gi }
```

### Multi-model per GPU
Running multiple models on one GPU — options:
1. **Triton Inference Server** (NVIDIA) — batches across models
2. **vLLM multi-LoRA** — one base model, many adapters
3. **Ray Serve multiplex** — replicas load different models
4. **MIG partition** — hardware split (A100/H100)

### GPU cold start problem
Images with CUDA base + libs = 5-15 GB → pulling on a new GPU node takes minutes. Solutions:
- **DaemonSet image pre-puller** (Kubestash, Spegel)
- **Warm pool** — keep 1 GPU node alive always
- **Karpenter with pre-warmed AMIs**

---

## 12.7 KServe vs Kubeflow vs plain Deployments for ML serving

### Plain Deployments
- Fine for <10 models
- Hand-roll autoscaling, canary, payload logging

### KServe
- **CRDs:** `InferenceService` → standard predictor/transformer/explainer pattern
- Scale-to-zero via Knative
- Payload logging to async sink
- **Winner for pure model serving in 2026**

### Kubeflow
- Broader platform — Pipelines, Notebooks, Training Operators
- Overkill if you just want serving
- Use when you need end-to-end in one place

### Recommended for Avrioc
KServe for serving, custom Helm charts for app (FastAPI + LangGraph), Argo Workflows for training orchestration.

---

## 12.8 KEDA — the ML autoscaler

HPA scales only on CPU/memory. KEDA scales on **anything**:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: { name: rag-worker-scaler }
spec:
  scaleTargetRef: { name: rag-worker }
  minReplicaCount: 0     # scale to zero
  maxReplicaCount: 50
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.me-central-1.amazonaws.com/...
      queueLength: '10'
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      query: sum(rate(llm_requests_total[1m]))
      threshold: '100'
```

KEDA supports scale-to-zero, critical for LLM workloads where GPU hours are expensive.

---

## 12.9 Helm — package manager for K8s

```
chart/
├── Chart.yaml          # metadata
├── values.yaml         # defaults
├── values-prod.yaml    # env override
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── hpa.yaml
    ├── configmap.yaml
    └── secret.yaml
```

Install: `helm install rag ./chart -f values-prod.yaml`

For ML:
- Template KServe `InferenceService`
- Parameterize model URI, resource requests, min/max replicas
- Include `ServiceMonitor` for Prometheus
- Use `helm dependency update` to pull in ecosystem charts

---

## 12.10 GitOps with Argo CD

Declarative, pull-based deployment:

```
Git repo = desired state
    │
    ▼
Argo CD watches
    │
    ▼
Reconciles cluster = desired state
```

Benefits:
- Deploy via PR
- Full audit (git history = deploy history)
- Drift detection (cluster ≠ git alerts)
- Self-healing (Argo fixes manual changes)

Standard in MLOps 2026.

---

## 12.11 Service mesh — Istio / Linkerd / Cilium

Runtime observability + security + traffic management:
- **mTLS** between services
- **Traffic splitting** for canary / A/B
- **Retries, timeouts, circuit breakers**
- **Distributed tracing**

For ML: useful for A/B testing two model versions via traffic split.

---

## 12.12 Ray — the ML-native distributed framework

### Core abstractions
- **Task** — stateless remote function (`@ray.remote`)
- **Actor** — stateful remote class (persists state on a worker)
- **Object store** — shared memory for zero-copy data passing

### Ray Core
Base layer for distributed Python. Like Dask, but stateful actors + GPU support.

### Ray Serve
- Deploy models as scalable HTTP services
- Built-in batching (`@serve.batch`)
- Autoscaling (replicas based on queue depth)
- Multi-model routing (`@serve.deployment`)

### Ray Train
- Distributed training (PyTorch DDP, FSDP, DeepSpeed)
- Fault tolerance, checkpointing, elastic scaling
- Integrates with Ray Tune for HPO

### Ray Tune
- Distributed hyperparameter optimization
- ASHA, BOHB, PBT algorithms
- Checkpoint/resume
- Beats Optuna for multi-node GPU HPO

### Ray Data
- Distributed data loading / preprocessing
- Streaming from S3, Parquet, etc.
- Used with Ray Train for large datasets

### When Ray wins
- Heterogeneous Python workloads (CPU + GPU)
- Need stateful actors (parameter servers, model serving state)
- LLM training + serving in one cluster
- Alternative to Spark when your workload is Python-first

### When Spark wins
- Structured data ETL at PB scale
- SQL-heavy workloads
- Strong data partitioning

### KubeRay — Ray on Kubernetes
- `RayCluster` CRD = declarative Ray cluster on K8s
- `RayJob` CRD = one-shot Ray job on K8s
- `RayService` CRD = Ray Serve on K8s with zero-downtime deploy

---

## 12.13 Ray Serve architecture

```
┌──────────────────────────────────────────────────────────┐
│  Ray Serve Application                                    │
│                                                           │
│  HTTP/gRPC ingress (uvicorn replicas)                     │
│          │                                                │
│          ▼                                                │
│  Router (consistent hashing / round-robin)                │
│          │                                                │
│    ┌─────┴─────┬─────────┬──────────┐                    │
│    ▼           ▼         ▼          ▼                    │
│  Deployment  Deploy.   Deploy.   Deploy.                 │
│  "summarize" "embed"   "rerank"  "llm"                   │
│    replicas   replicas  replicas  replicas               │
│                                                           │
│  Each deployment is a Ray actor; replicas autoscale       │
└──────────────────────────────────────────────────────────┘
```

### Batching
```python
@serve.deployment
class Embedder:
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()
```

Ray Serve aggregates requests across replicas; vLLM batches tokens within a replica. Both layers matter.

---

## 12.14 vLLM on Ray Serve

The common production pattern:

```
User requests → FastAPI gateway → LangGraph orchestration
                                        │
                              ┌─────────┼─────────┐
                              ▼         ▼         ▼
                        Embed API   Rerank API   LLM API
                        (Ray Serve) (Ray Serve) (vLLM on Ray)
```

vLLM has a native `vllm.entrypoints.openai.api_server`; Ray Serve can host it as a deployment for unified autoscaling and multi-model serving.

---

## 12.15 Operational patterns

### Canary deployments
1. Deploy new version at 5% traffic
2. Watch error rate + latency
3. Ramp to 50% if healthy
4. Full cutover
5. Keep old version for 24h rollback

### Blue-green
1. Deploy green alongside blue (full capacity)
2. Switch traffic at router
3. Keep blue for fast rollback

### Shadow traffic
1. Mirror prod traffic to new model (don't use outputs)
2. Compare predictions offline
3. Promote if agreement is high

---

## 12.16 Interview Q&A — Kubernetes & Ray

**Q1. Pod vs Container vs Deployment?**
> Container: a running instance of an image. Pod: smallest K8s unit — one or more containers sharing net/storage namespaces. Deployment: manages replica set of Pods with rolling updates.

**Q2. HPA vs KEDA — when each?**
> HPA scales on CPU/memory or custom metrics from metrics-server. KEDA scales on external signals (Kafka lag, SQS, Prometheus, GPU util) and supports scale-to-zero. For ML, KEDA is essential.

**Q3. Requests vs Limits?**
> Requests = scheduler's guaranteed minimum. Limits = hard maximum (CPU throttled, memory OOMKilled). QoS class Guaranteed (req=lim) gets highest priority on eviction.

**Q4. GPU scheduling — how?**
> NVIDIA device plugin advertises GPU resource. Taint GPU nodes (nvidia.com/gpu: NoSchedule); pods add matching toleration. Request `nvidia.com/gpu: 1`. Use MIG for sharing on A100/H100, time-slicing for dev.

**Q5. How do you share one GPU across multiple models?**
> Triton Inference Server (NVIDIA batches across models), vLLM multi-LoRA (one base + many adapters), Ray Serve multiplex, or MIG hardware partition. Choice depends on model sizes and isolation needs.

**Q6. KServe vs plain Deployments for model serving?**
> Plain Deployments: fine for <10 models, hand-roll autoscaling and canaries. KServe: InferenceService CRD, scale-to-zero, predictor/transformer/explainer pattern — the winner for pure model serving.

**Q7. [Gotcha] Your GPU node takes 5 min to start serving. Why?**
> Image pull. ML images are 5-15 GB. Solutions: image pre-puller DaemonSet (Spegel, Kraken), warm pool of 1 GPU node, Karpenter with pre-warmed AMIs. Also check CUDA driver init and model weight download.

**Q8. [Gotcha] Inference pod OOMKills despite GPU memory being fine. What do you check?**
> CPU memory (host RAM), not VRAM. Tokenizers, request queues, Python object overhead live in host RAM. Verify memory requests/limits. Check for leaking tensors in long-lived FastAPI workers. Enable /dev/shm volume for DataLoader workers.

**Q9. Argo CD — what problem does it solve?**
> GitOps: cluster state = git state. Deploy = PR merge. Full audit trail, drift detection, self-healing. Standard in MLOps 2026.

**Q10. Ray vs Spark — when each?**
> Ray: heterogeneous Python workloads, stateful actors, GPU-first, ML training + serving in one cluster. Spark: structured ETL at PB scale, SQL-heavy. For LLM training + serving, Ray.

**Q11. Ray task vs actor?**
> Task: stateless remote function — embarrassingly parallel work. Actor: stateful remote class that lives on a specific worker and serializes method calls — model serving, parameter servers.

**Q12. How does Ray Serve batching work?**
> `@serve.batch(max_batch_size, batch_wait_timeout_s)` — Ray Serve aggregates incoming requests into a batch until either size or timeout. For LLMs, combine with vLLM's continuous batching inside replicas.

**Q13. How do you autoscale Ray Serve?**
> `autoscaling_config` with min_replicas, max_replicas, target_ongoing_requests. Ray Serve scales replicas. Pair with Ray Autoscaler on KubeRay for node-level scaling. Scale-to-zero works but cold-starts are 30-90s for LLMs.

**Q14. Ray Train vs torchrun?**
> Ray Train wraps PyTorch DDP / FSDP / DeepSpeed with fault tolerance, checkpointing, elastic scaling. torchrun is lower-level. Ray Train integrates with Ray Tune for HPO and Ray Data for streaming input.

**Q15. Ray Tune vs Optuna?**
> Ray Tune: distributed trials across a Ray cluster, ASHA/PBT/BOHB built-in, checkpoint/resume. Optuna: simpler single-node. For GPU-heavy HPO, Ray Tune.

**Q16. [Gotcha] Your Ray cluster shows idle GPUs but tasks pending. Why?**
> Resource fragmentation or placement group issues. Tasks request `num_gpus=1` but a specific node type, or placement groups reserve GPUs without using them. Check `ray status`, inspect placement groups, verify custom resources match pod labels.

**Q17. What is KubeRay?**
> Kubernetes operator for Ray: RayCluster CRD (declarative Ray on K8s), RayJob (one-shot), RayService (zero-downtime Ray Serve deploys).

**Q18. Service mesh for ML — why?**
> mTLS between services, traffic splitting for canary / A/B (serve model v2 to 10% of traffic), retries/timeouts/circuit breakers, distributed tracing. Istio or Linkerd. For simple cases, KServe's built-in canary suffices.

**Q19. Helm chart structure?**
> Chart.yaml (metadata), values.yaml (defaults), templates/ (Kubernetes manifests). Parameterize model URI, resource requests, replica counts. Chart dependencies pull in ecosystem charts (Prometheus, cert-manager).

**Q20. Canary vs blue-green?**
> Canary: gradual traffic shift (5→50→100%); cost-efficient, detects real-world issues. Blue-green: full deploy in parallel, instant switch, fast rollback, doubles capacity temporarily. Pick canary for cost, blue-green for zero-risk cutovers.

**Q21. NetworkPolicy — why and how?**
> Kubernetes firewall for Pod-to-Pod traffic. Default deny, allow only needed (e.g., rag-api → vector-db on 6333). Essential for multi-tenant or regulated deployments.

**Q22. [Gotcha] Your vLLM pod works locally but crashes in K8s. Common causes?**
> (1) Shared memory — need `/dev/shm` volume emptyDir medium=Memory. (2) NCCL issues with multi-GPU — need hostNetwork or IB/EFA setup. (3) CUDA mismatch — pod's CUDA version ≠ node's driver. (4) OOM — GPU memory limits may not reflect actual usage.

**Q23. PDB (PodDisruptionBudget) — why?**
> Prevents voluntary evictions from taking down too many replicas. For critical services (LLM endpoint), `minAvailable: 2` during node upgrades.

**Q24. [Gotcha] Rolling update drops traffic briefly. Fix?**
> Correct readiness + liveness probes. `terminationGracePeriodSeconds` + `preStop` hook that sleeps long enough for ingress to drain. RollingUpdate strategy with maxUnavailable: 0.

**Q25. For Avrioc: Kubernetes cluster recommendation?**
> EKS in me-central-1 or self-managed K8s on EC2. GPU node groups (g5.xlarge for embed/rerank, p5 for LLM), CPU node group for gateway. KServe + KEDA + Argo CD + Prometheus + Grafana + Loki. Karpenter for fast GPU scaling. Istio if traffic-shaping is needed.

---

## 12.17 Resume tie-ins

- **"Containerization & Orchestration: Docker, Kubernetes, OpenShift, Docker Compose, NVIDIA Container Toolkit, Ray"** — have a specific story even if light. E.g., "At ResMed, the IHS platform ran on EKS with KServe InferenceServices per model. I contributed the FastAPI transformer containers and Helm chart parameterization."
- **"AI-powered ML workspace with on-demand GPU/CPU EC2 provisioning"** — analogous to K8s dynamic provisioning. Explain how you chose EC2+EFS instead of K8s (probably: simpler for the team, no cluster operator overhead, faster launch).

---

Continue to **[Chapter 13 — Frameworks](13_frameworks.md)**.
