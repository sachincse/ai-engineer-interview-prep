# Chapter 12 — Kubernetes, Ray, Docker

## Container orchestration for AI workloads — the most critical chapter for Avrioc

> Avrioc JD names **Kubernetes + Ray + LLMOps practices** explicitly. This chapter goes deep — every primitive, every diagram, every gotcha that a Senior ML Engineer should be able to whiteboard cold.

---

## 12.0 The big picture — why containers + orchestration?

Before diving into Kubernetes specifics, let's anchor *why* this stack exists.

When I started ML in 2017, deploying a model meant SSH'ing into an EC2 instance, installing dependencies, copying a Pickle file, running a Flask server inside `screen`, and praying. If the box died, the model died. If two models needed different Python versions, you had two boxes. If traffic spiked, you manually launched another instance. **The pain is real**, and containers + Kubernetes solved it in three layers:

1. **Docker** packages the model, code, and runtime into one immutable artifact that runs the same on your laptop, CI, and prod.
2. **Kubernetes** schedules containers across a fleet of nodes, restarts them on failure, scales them on load, and abstracts the underlying machines.
3. **Operators** (KServe, KEDA, KubeRay) layer ML-specific concerns — autoscaling on queue depth, GPU sharing, distributed training coordination — on top of Kubernetes primitives.

```
   ┌───────────────── The stack you're explaining ─────────────────┐
   │                                                                │
   │   ML application code (FastAPI + LangGraph + vLLM)             │
   │      ── Layer 1: code                                          │
   │                                                                │
   │   Docker image                                                 │
   │      ── Layer 2: package once, run anywhere                    │
   │                                                                │
   │   Kubernetes Deployments / StatefulSets / Services             │
   │      ── Layer 3: orchestration primitives                      │
   │                                                                │
   │   Operators: KServe, KEDA, KubeRay, GPU Operator               │
   │      ── Layer 4: ML-specific extensions                        │
   │                                                                │
   │   Cluster: EKS / AKS / OpenShift / GKE                         │
   │      ── Layer 5: managed control plane                         │
   │                                                                │
   │   Cloud infrastructure: VPC + IAM + EBS + load balancers       │
   │      ── Layer 6: substrate                                     │
   └────────────────────────────────────────────────────────────────┘
```

> **How to say this in an interview:** "Docker gives me reproducible artifacts. Kubernetes gives me orchestration — scheduling, healing, scaling. Operators give me ML-aware behavior on top of Kubernetes. The managed cluster (EKS) lets me skip running etcd. Each layer adds value above the last; you can't skip layers without paying somewhere else."

---

## 12.1 Docker — the foundation

### Why Docker exists

Before Docker, deploying software meant either fat VMs (slow, heavy) or hoping your dependencies worked on the target system (they didn't). Docker gave us **OS-level virtualization**: a container shares the host kernel but has its own filesystem, processes, network namespace, and resource limits via Linux cgroups. The result is small (~MB), fast-starting (<1s), reproducible artifacts.

### Mental model with analogy

Docker is like **shipping containers in maritime logistics**. Before standardized containers, every cargo type had bespoke handling. Standardization meant any ship could carry any container, any port could load any container. Software shipped in Docker images is the same — any cluster, any cloud, any laptop runs the same artifact.

### Anatomy of an image

```
   ┌──────────────── Docker Image Layers (read-only) ─────────────┐
   │                                                                │
   │   Layer 5:  COPY src/ src/                  (your code)        │
   │   Layer 4:  RUN pip install -r reqs.txt     (Python deps)      │
   │   Layer 3:  COPY requirements.txt .         (deps file)        │
   │   Layer 2:  FROM nvidia/cuda:12.4-runtime  (base image)       │
   │   Layer 1:  Linux base (Ubuntu 22.04)                          │
   │                                                                │
   │   Each layer is content-addressed; cached separately.          │
   │   Change a file in upper layer → only that layer rebuilds.     │
   └────────────────────────────────────────────────────────────────┘

   ┌──────────────── Container (read-write on top) ───────────────┐
   │   Layer 6:  Process state, /tmp, /var/log (ephemeral)         │
   └────────────────────────────────────────────────────────────────┘
```

The implication: **put rarely-changing things at the bottom** (base image, system deps) and **frequently-changing things at the top** (your code). This maximizes layer cache hits during rebuild.

### Multi-stage build — the must-know pattern

```dockerfile
# ============ Stage 1: builder ============
FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder

WORKDIR /build

# Install build deps (won't ship to prod)
RUN apt-get update && apt-get install -y \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Python deps with build-time optimization
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Compile any C extensions
COPY src/ ./src/
RUN cd src && python setup.py build_ext --inplace

# ============ Stage 2: runtime ============
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime

WORKDIR /app

# Copy ONLY the artifacts from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/src /app/src

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user (security)
RUN useradd -m -u 1000 mluser \
    && chown -R mluser:mluser /app
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Walkthrough of the load-bearing lines:**

- `FROM nvidia/cuda:12.4-devel-ubuntu22.04 AS builder` — the dev image has compilers, headers, ~5 GB.
- `FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS runtime` — the runtime image is ~1.5 GB without dev tools.
- `COPY --from=builder` — pulls only what's needed, leaving build cruft behind.
- `USER mluser` — never run as root in production; least-privilege for security.
- `HEALTHCHECK` — Kubernetes readiness/liveness probes complement this.

### BuildKit cache mounts (the modern speedup)

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

`--mount=type=cache` lets the pip cache persist across builds inside the build context, even when layers invalidate. CI builds drop from 5 min to 30 s on cache hit.

### .dockerignore for ML

```
.git
.venv
.pytest_cache
__pycache__
*.pyc
*.pyo
data/
notebooks/
*.ipynb_checkpoints
docs/
tests/
.env
*.log
.DS_Store
node_modules/
```

Without this, your model artifact directory or Jupyter checkpoints (potentially gigabytes) get COPY'd into every layer. ML teams forgetting `.dockerignore` is the #2 cause of huge images.

### Worked example — image size matters

Bad Dockerfile (real example I've seen):

```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3 python3-pip
COPY . .
RUN pip3 install -r requirements.txt
CMD ["python3", "main.py"]
```

This image is ~12 GB. Why?

- `apt-get` cached at /var/lib/apt/lists/ (~200 MB)
- No `.dockerignore` so `data/` and `.git` are inside (~5 GB)
- `pip` cache at /root/.cache/pip (~2 GB)
- Single-stage build keeps build artifacts (~3 GB)

Same workload, optimized:

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ src/
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "src.main"]
```

Result: ~400 MB. **Pull time on a fresh K8s node drops from 4 min to 20 s.**

### Common Docker mistakes (senior-level)

1. **Building on architecture mismatch.** You build on M1 Mac, deploy on x86 Linux — segfaults. Always `--platform linux/amd64` in CI or use `buildx` for multi-arch.
2. **Putting secrets in `ENV`.** They land in image layers. Use BuildKit secrets (`--mount=type=secret`) or pass at runtime via Kubernetes Secrets.
3. **Running as root.** Container escapes are real. Always create a non-root user.
4. **No health check.** Kubernetes won't know when your app is actually ready.

### Docker Q&A

**Q1. Multi-stage builds — why?**
> Build dependencies and runtime dependencies are different. Compilers, headers, build caches don't need to ship to production. Multi-stage lets you have one stage with everything needed to build, then a second stage with only the artifacts that need to run. Result is dramatically smaller, faster-pulling, more secure images. For ML, the difference is often 5x — `nvidia/cuda:devel` is 5 GB, `nvidia/cuda:runtime` is 1.5 GB.

**Q2. Why do you order Dockerfile layers the way you do?**
> Cache locality. Each layer's cache is invalidated when its inputs change, plus everything below it. So I put the most stable inputs first — base image, OS packages — and the most-changing inputs last — application code. Means a code change rebuilds only the top one or two layers, not the whole image. CI cycle drops from 8 minutes to under one.

**Q3. How do you handle CUDA in a Docker image?**
> Use NVIDIA's official base images: `nvidia/cuda:12.4-runtime-ubuntu22.04` for runtime, `nvidia/cuda:12.4-devel-ubuntu22.04` for builds that need nvcc. The host needs the matching NVIDIA driver — image bundles CUDA libraries, not the driver. The Docker daemon needs the NVIDIA Container Toolkit installed. At runtime, `--gpus all` (Docker) or `nvidia.com/gpu: 1` (Kubernetes) exposes GPUs to the container.

---

## 12.2 Kubernetes — the mental model

### Why Kubernetes exists

Imagine you have 50 EC2 instances and 200 containers to run. Manual placement is hell — which container goes where, what happens when a container crashes, how do you scale up, how do you roll out a new version without downtime, how do you handle network routing? Kubernetes is the **distributed scheduler** that solves all of this declaratively. You describe the desired state ("I want 5 replicas of this Deployment with this image"); Kubernetes makes it so and keeps it so.

### Mental model with analogy

Kubernetes is like a **city's logistics planning department**. Nodes are warehouses; pods are shipments; the scheduler decides which warehouse fits each shipment based on capacity, special requirements (like refrigeration = GPU), and traffic patterns. The control plane is the planning office; kubelet on each node is the local manager. When a warehouse burns down, the planning office reassigns shipments automatically.

### Architecture

```
   ┌───────────────────────── Kubernetes Cluster ─────────────────────────────┐
   │                                                                            │
   │   ┌─────────── Control Plane (master) ────────────────────────────┐       │
   │   │                                                                │       │
   │   │   API Server  ── REST entry point; everything goes through it │       │
   │   │       │                                                         │       │
   │   │       ▼                                                         │       │
   │   │   etcd  ── distributed key-value store; cluster state          │       │
   │   │                                                                │       │
   │   │   Scheduler  ── watches Pending pods, picks a Node             │       │
   │   │   Controller Manager  ── runs reconciliation loops             │       │
   │   │   Cloud Controller Manager  ── EBS, ALB integration            │       │
   │   └────────────────────────────────────────────────────────────────┘      │
   │                              │                                              │
   │                              │ (kubectl, kubelet talk to API)              │
   │                              │                                              │
   │   ┌─── Worker Node 1 ────┐  │  ┌─── Worker Node 2 ────┐                   │
   │   │  kubelet              │  │  │  kubelet              │                   │
   │   │  kube-proxy           │  │  │  kube-proxy           │                   │
   │   │  Container runtime    │  │  │  Container runtime    │                   │
   │   │  (containerd)         │  │  │  (containerd)         │                   │
   │   │                       │  │  │                       │                   │
   │   │  ┌─Pod──┐  ┌─Pod──┐  │  │  │  ┌─Pod──┐  ┌─Pod──┐  │                   │
   │   │  │vllm  │  │redis │  │  │  │  │embed │  │worker│  │                   │
   │   │  │GPU=1 │  │      │  │  │  │  │GPU=0 │  │      │  │                   │
   │   │  └──────┘  └──────┘  │  │  │  └──────┘  └──────┘  │                   │
   │   └──────────────────────┘  │  └──────────────────────┘                   │
   │                              │                                              │
   │   Networking: CNI plugin (Calico, Cilium) gives every Pod an IP            │
   │   DNS: CoreDNS resolves Service names to ClusterIPs                        │
   └────────────────────────────────────────────────────────────────────────────┘
```

### Control plane components — what each does

- **API Server** — the only component that talks to etcd; everything else (kubectl, kubelet, controllers) goes through API. Authoritative API.
- **etcd** — distributed KV store; raft consensus; cluster state lives here. Backup etcd or lose everything.
- **Scheduler** — watches `nodeName=""` pods, picks a node based on resource requests, taints/tolerations, affinity, and runs the binding.
- **Controller Manager** — runs the control loops (Deployment, ReplicaSet, Node lifecycle, etc.). The reconciliation engine.
- **Cloud Controller Manager** — talks to cloud APIs (creates ALBs for LoadBalancer Services, attaches EBS for PVCs).

### Worker node components

- **kubelet** — the agent on each node; talks to API server, manages pods locally, runs probes.
- **kube-proxy** — programs iptables / IPVS rules to route Service IPs to backing pods.
- **Container runtime** — containerd (or CRI-O); pulls images and runs containers.
- **CNI plugin** — Calico, Cilium, AWS VPC CNI; provides pod networking.

### Worked example — what happens when you `kubectl apply -f deploy.yaml`?

```
   1. kubectl validates YAML, sends POST to API Server
   2. API Server authenticates, validates schema, writes to etcd
   3. Deployment controller sees new Deployment in etcd via watch
      → creates a ReplicaSet (one per pod template)
   4. ReplicaSet controller sees new RS, sees desired replicas=3
      → creates 3 Pod objects (no node assigned yet)
   5. Scheduler sees 3 Pending pods (nodeName="")
      → for each: filter nodes (capacity, taints, affinity), score, bind
   6. kubelet on each chosen node sees a pod assigned to it
      → pulls image (via containerd), creates container, starts it
   7. kube-proxy programs iptables for the Service's ClusterIP
   8. Readiness probe passes → endpoint added to Service's endpoints
   9. Traffic flows: Service IP → kube-proxy → Pod IP
```

This is the choreography you should be able to walk through cold.

> **How to say this in an interview:** "When I do kubectl apply, the YAML hits the API server which writes desired state to etcd. The Deployment controller notices the new deployment, creates a ReplicaSet, which creates Pods with no node assigned. The Scheduler picks nodes based on resource requests, affinity, and taints, and binds each pod to a node. Kubelet on each node pulls the image and starts the container. Kube-proxy programs iptables so the Service's ClusterIP routes to the Pod IPs. Readiness probe passing adds the pod to the Service's endpoint set. Every step is observable — that's why kubectl describe and kubectl events are my go-to tools when something doesn't work."

---

## 12.3 Pods, Deployments, ReplicaSets

### Pod — the atom

A Pod is one or more containers sharing a network namespace (same IP, can talk via localhost) and storage. It's the smallest deployable unit. **Pods are ephemeral** — if the pod dies, the replacement has a new IP. Don't write code that assumes pod IPs are stable; use Services.

### Pod lifecycle

```
   Pending  ── created, waiting for scheduling or image pull
      │
      ├─→ FailedScheduling  ── no node fits requests
      ├─→ ImagePullBackOff  ── image not found, ECR auth fail
      └─→ ContainerCreating
              │
              ▼
   Running  ── containers started, at least one alive
              │
              ├─→ CrashLoopBackOff  ── container keeps crashing
              ├─→ OOMKilled         ── memory limit exceeded
              └─→ healthy
              │
              ▼
   Succeeded (Job) or Failed or Terminating (deleted)
```

When a probe fails, kubelet restarts the container in the pod. When a pod dies entirely, the ReplicaSet creates a new pod (different name, different IP).

### Deployment — managed pod sets

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3
  labels: { app: vllm-llama3 }
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # one extra pod during rollout
      maxUnavailable: 0    # never drop below desired count
  selector:
    matchLabels: { app: vllm-llama3 }
  template:
    metadata:
      labels: { app: vllm-llama3 }
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.3
        args:
        - --model=meta-llama/Llama-3.3-70B-Instruct
        - --tensor-parallel-size=2
        - --gpu-memory-utilization=0.9
        - --enable-prefix-caching
        ports: [{ containerPort: 8000 }]
        resources:
          requests:
            cpu: "8"
            memory: 64Gi
            nvidia.com/gpu: 2
          limits:
            cpu: "16"
            memory: 96Gi
            nvidia.com/gpu: 2
        readinessProbe:
          httpGet: { path: /health, port: 8000 }
          initialDelaySeconds: 120  # model loads slowly
          periodSeconds: 10
        livenessProbe:
          httpGet: { path: /health, port: 8000 }
          initialDelaySeconds: 300
          periodSeconds: 30
        volumeMounts:
        - { name: dshm, mountPath: /dev/shm }
        - { name: model-cache, mountPath: /root/.cache/huggingface }
      volumes:
      - name: dshm
        emptyDir: { medium: Memory, sizeLimit: 16Gi }
      - name: model-cache
        persistentVolumeClaim:
          claimName: hf-cache-pvc
```

**Walkthrough of load-bearing fields:**

- `nodeSelector` + `tolerations` — pin to GPU nodes; the GPU node taint keeps non-GPU workloads off these expensive boxes.
- `resources.requests` — what the scheduler reserves; ensures placement on a node that has enough free.
- `resources.limits` — hard cap; CPU throttles, memory triggers OOMKill.
- `nvidia.com/gpu: 2` — tensor parallelism across 2 GPUs.
- `initialDelaySeconds: 120` — the model takes ~2 min to load; without this, kubelet kills the pod thinking it failed.
- `/dev/shm` as `medium: Memory` — vLLM uses shared memory for inter-process; default `/dev/shm` in K8s is only 64 MB.
- `model-cache` PVC — HuggingFace cache shared across pod restarts so we don't re-download a 140 GB model.

### ReplicaSet vs Deployment

A Deployment manages a ReplicaSet, which manages Pods. **You almost never create ReplicaSet directly.** Deployment adds rolling-update logic on top — when you update the pod template, Deployment creates a new ReplicaSet, scales it up while scaling the old one down.

### Update strategies

- **RollingUpdate** (default) — gradual replacement, zero downtime if probes are correct.
- **Recreate** — kill all old pods, then create new (downtime). Use only for stateful migrations that can't run two versions concurrently.
- **Blue-Green** (custom; requires two Deployments + Service swap) — full new fleet, atomic switch.
- **Canary** (custom; two Deployments behind one Service with weighted endpoints, or service mesh) — gradual traffic shift.

### Pod / Deployment Q&A

**Q1. What's the difference between a Pod and a container?**
> A container is a single process tree with isolated filesystem and network. A Pod is one or more containers sharing a network namespace and optionally storage. They share an IP and can talk via localhost. The pattern is one main container plus optional sidecars — for instance, a vLLM container plus a Prometheus exporter sidecar that scrapes vLLM's /metrics endpoint and exposes them in a different format. Both die and restart together as one unit.

**Q2. What happens during a rolling update?**
> Deployment notices a pod template change, creates a new ReplicaSet at zero replicas. It then scales the new RS up by `maxSurge` and the old RS down by `maxUnavailable` in alternation, waiting for new pods' readiness probes between steps. If a new pod fails to become ready within `progressDeadlineSeconds`, the rollout stalls. You can `kubectl rollout undo` to revert to the previous RS. The whole thing is observable via `kubectl rollout status`.

**Q3. Pod stuck in Pending — how do you debug?**
> First `kubectl describe pod` and look at Events. Most common: `0/N nodes are available: insufficient cpu/memory/nvidia.com/gpu`. That means resource requests can't be satisfied — either bump cluster size, reduce requests, or check if a node group is at quota. Second cause: taints without matching toleration. Third: PVC binding stuck because StorageClass mismatch. Fourth, rare but real: image pull issues that present as Pending until kubelet starts pulling. The Events section narrows it down in 30 seconds.

**Q4. Pod stuck in CrashLoopBackOff — playbook?**
> `kubectl logs <pod> --previous` to see the last crash output. Common causes: bad config (missing env var, unreachable DB), OOMKilled (check `kubectl describe` for last state OOMKilled), readiness/liveness probe killing it before it starts (initialDelaySeconds too low), bug in code on startup. Fix is data-driven from logs. Backoff time grows exponentially, capped at 5 min, so reproducing fast requires editing the Deployment.

---

## 12.4 Services, Ingress, networking

### Why Services exist

Pods come and go — IPs change. You need a stable address. A **Service** is a virtual IP plus DNS name that load-balances to a set of pods selected by labels.

```
   ┌──────────────── Service abstraction ────────────────┐
   │                                                       │
   │   Service "vllm" (ClusterIP 10.0.5.10:8000)          │
   │       selector: app=vllm                              │
   │             │                                         │
   │             │ Endpoints object updated by kubelet     │
   │             │ as pods become ready/unready            │
   │             ▼                                         │
   │     ┌───────┴───────┬───────┐                        │
   │     ▼               ▼       ▼                        │
   │   Pod 10.1.1.1   10.1.1.2  10.1.1.3                  │
   │                                                       │
   │   kube-proxy on each node programs iptables to        │
   │   route 10.0.5.10 → one of the endpoints              │
   └───────────────────────────────────────────────────────┘
```

### Service types

- **ClusterIP** (default) — internal only; reachable from within the cluster. The bread and butter.
- **NodePort** — exposes on every node's IP at a high port (30000-32767). Rarely used; older pattern.
- **LoadBalancer** — provisions a cloud LB (AWS NLB or ALB, Azure LB). Public-facing services.
- **Headless** (`clusterIP: None`) — no virtual IP; DNS returns pod IPs directly. Used for stateful services like databases.
- **ExternalName** — DNS alias to an external service (CNAME).

### Ingress — HTTP routing

A Service typically lives at L4 (TCP). For HTTP-level routing — multiple hostnames, paths, TLS — you use an **Ingress** controller. Common ones: NGINX Ingress, AWS ALB Ingress Controller, Traefik, Istio Gateway.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-platform
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
spec:
  rules:
  - host: api.avrioc.ai
    http:
      paths:
      - path: /v1/chat
        pathType: Prefix
        backend:
          service: { name: vllm-gateway, port: { number: 8000 } }
      - path: /v1/embed
        pathType: Prefix
        backend:
          service: { name: embedding-svc, port: { number: 8000 } }
```

### Network diagram — full request path

```
   ┌───────────────────────── Full Request Path ────────────────────────┐
   │                                                                     │
   │   User browser                                                      │
   │       │ HTTPS                                                       │
   │       ▼                                                             │
   │   AWS ALB (provisioned by Ingress)                                  │
   │       │ host=api.avrioc.ai, path=/v1/chat                           │
   │       ▼                                                             │
   │   Ingress controller pod                                            │
   │       │ (path match → backend service)                              │
   │       ▼                                                             │
   │   Service "vllm-gateway" (ClusterIP)                                │
   │       │ kube-proxy iptables DNAT                                    │
   │       ▼                                                             │
   │   Pod IP (one of 3 vllm pods)                                       │
   │       │ container port 8000                                         │
   │       ▼                                                             │
   │   FastAPI app inside container                                      │
   │       │                                                             │
   │       ▼                                                             │
   │   Response retraces the path                                        │
   └─────────────────────────────────────────────────────────────────────┘
```

### Network policy — pod-to-pod firewall

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: vllm-ingress-only-from-gateway
spec:
  podSelector:
    matchLabels: { app: vllm }
  policyTypes: [Ingress]
  ingress:
  - from:
    - podSelector: { matchLabels: { app: gateway } }
    ports: [{ protocol: TCP, port: 8000 }]
```

This says: vLLM pods only accept ingress from gateway pods on port 8000. Multi-tenant hygiene.

### Networking Q&A

**Q1. Walk me through how a Service's ClusterIP routing works.**
> When you create a Service, kube-proxy on every node sees it via watch and programs iptables (or IPVS) rules. Rule: "packets destined to ClusterIP:port DNAT to one of the backing pod IPs at random." The Endpoints controller updates the backing pod list as pods become ready or unready. So a request to the Service IP from any pod in the cluster is randomly load-balanced across healthy backends, all without DNS or any centralized load balancer. CoreDNS resolves the Service name to the ClusterIP for cluster-internal callers.

**Q2. ClusterIP vs Ingress — when each?**
> ClusterIP for internal traffic between services within the cluster — service-to-service calls, your gateway calling your model. Ingress for external HTTP traffic from outside the cluster, with hostname and path routing, TLS termination, and integration with cloud load balancers. They compose: Ingress routes external traffic to a ClusterIP Service, which load-balances to pods. Most production clusters have one Ingress controller per environment fronting many services.

**Q3. NetworkPolicy — when do you actually use it?**
> Multi-tenant clusters where one tenant's pods shouldn't reach another's. Compliance scenarios where data-tier pods should only be reachable from app-tier pods. Defense-in-depth — limit blast radius of a compromised pod. Default is "allow all" for backward compatibility, which is bad for production. My pattern: namespace-level default-deny, then explicit allow rules for known traffic patterns. Calico or Cilium implement this; the AWS VPC CNI requires the network policy controller add-on.

---

## 12.5 ConfigMap, Secret, Volume

### ConfigMap — non-secret config

```yaml
apiVersion: v1
kind: ConfigMap
metadata: { name: vllm-config }
data:
  model_name: "meta-llama/Llama-3.3-70B-Instruct"
  max_model_len: "8192"
  log_level: "INFO"
  prompt_templates.yaml: |
    rag:
      system: "You are a helpful assistant grounded in the provided context."
    chat:
      system: "You are a helpful assistant."
```

Inject into pod via env vars or as a mounted file.

### Secret — sensitive data

Secrets are base64-encoded by default (NOT encrypted at rest unless you enable etcd encryption). For real secrets use:
- **External Secrets Operator** + AWS Secrets Manager / Azure Key Vault
- **Sealed Secrets** (Bitnami) for GitOps
- **AWS Pod Identity** or IRSA for runtime credentials

```yaml
apiVersion: v1
kind: Secret
metadata: { name: hf-token }
type: Opaque
stringData:
  HUGGING_FACE_HUB_TOKEN: hf_xxx...
```

### Volume types for ML

- **emptyDir** — ephemeral; lost on pod restart. Use for `/dev/shm` (vLLM, PyTorch DataLoader).
- **emptyDir.medium=Memory** — RAM-backed; faster but counts against memory limits.
- **PersistentVolumeClaim (PVC)** — durable; backed by EBS/EFS/Azure Disk. Use for HF model cache.
- **hostPath** — mounts host directory; avoid in production (couples pod to node).
- **ConfigMap/Secret** — mount as files into the pod.

### Worked example — vLLM with EFS-backed model cache

```yaml
apiVersion: v1
kind: PersistentVolume
metadata: { name: hf-cache-efs }
spec:
  capacity: { storage: 500Gi }
  accessModes: [ReadWriteMany]
  csi:
    driver: efs.csi.aws.com
    volumeHandle: fs-0123456789abcdef
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: { name: hf-cache-pvc }
spec:
  accessModes: [ReadWriteMany]
  storageClassName: ""
  resources: { requests: { storage: 500Gi } }
  volumeName: hf-cache-efs
```

Now any pod that mounts `hf-cache-pvc` shares the HF cache. First pod downloads the 140 GB Llama-3 70B; second pod starts in 30 seconds because the model is cached.

---

## 12.6 Resource requests vs limits — the make-or-break

### Why this matters

Get this wrong and your cluster either over-commits and OOMKills randomly, or under-commits and burns money on idle capacity.

### The semantics

```
   resources:
     requests:    ← scheduler reserves this; placement decision
       cpu: "1"
       memory: "4Gi"
     limits:      ← hard cap at runtime
       cpu: "2"
       memory: "8Gi"
```

- **requests.cpu / memory** — what the scheduler considers "consumed" on a node. If a node has 8 CPU and 4 pods request 2 CPU each, the node is full (from scheduler's view).
- **limits.cpu** — CPU is throttled at the limit (process slowed, not killed).
- **limits.memory** — exceeding triggers **OOMKill** by the kernel. Container restarts.

### QoS classes

Kubernetes assigns each pod a QoS class based on requests/limits:
- **Guaranteed** — requests == limits for both CPU and memory across all containers. Last to be evicted under node pressure.
- **Burstable** — at least one resource has request < limit. Evicted before Guaranteed.
- **BestEffort** — no requests or limits. First evicted. Avoid for production.

### Worked example — sizing a vLLM pod

For Llama-3 70B on 2× A100-80GB with TP=2:

```
   GPU memory  = 140 GB weights + 30 GB KV cache + 10 GB overhead
              = ~180 GB across 2 GPUs (90 GB per GPU; tight on 80 GB)
              → fall back to AWQ quant or 4× A100

   CPU memory  = ~60 GB (tokenizer, request queues, Python heap)
              → request 64 GB, limit 96 GB

   CPU         = ~8 cores (concurrent tokenization, async loop)
              → request 8 cores, limit 16 cores

   GPU         = nvidia.com/gpu: 2

   /dev/shm    = 16 GB (NCCL communication for TP)
```

```yaml
resources:
  requests:
    cpu: "8"
    memory: 64Gi
    nvidia.com/gpu: 2
  limits:
    cpu: "16"
    memory: 96Gi
    nvidia.com/gpu: 2
```

### Common mistakes

1. **Setting requests = limits for CPU.** This makes pod Guaranteed but disables CPU bursting — your idle pod can't use spare cycles when needed. For most ML workloads, set requests at sustained load and limits 2x for headroom.
2. **Memory limit too tight.** OOMKilled is invisible to your app — it just dies. Better to over-provision memory and underrun than to OOMKill mid-inference.
3. **Forgetting `/dev/shm`.** Default shared memory in K8s is 64 MB. PyTorch DataLoader, NCCL, vLLM all need more. Symptom: torch crashes with shmem errors; NCCL hangs.

### Resource Q&A

**Q1. Requests vs limits — explain like I'm five.**
> Requests are the room I reserve at the hotel — guaranteed mine even if I don't use it. Limits are how loud I'm allowed to be — exceed them and security shows up. For CPU, "showing up" means throttling; for memory, "showing up" means OOMKill — eviction with extreme prejudice. The scheduler only looks at requests when deciding where to place pods, so requests determine cluster capacity. Limits determine runtime behavior.

**Q2. Why don't you just set requests = limits everywhere?**
> Two reasons. First, CPU bursting is useful — if you've requested 2 cores but the node has 4 idle, your latency-sensitive workload should be allowed to use them temporarily. Setting requests = limits gives you Guaranteed QoS but disables that benefit. Second, fitting more pods per node is cost-efficient — over-commitment via burstable QoS is how clusters achieve >70% utilization. The trade-off is that under contention, burstable pods are evicted first. So: requests = limits for production-critical pods (Guaranteed), burstable for less-critical.

---

## 12.7 Autoscaling — HPA, KEDA, Cluster Autoscaler, Karpenter

### Three layers of autoscaling

```
   ┌──────────────── Three Layers of Autoscaling ────────────────┐
   │                                                                │
   │   Pod-level:                                                   │
   │     HPA (Horizontal Pod Autoscaler)  ── replicas based on     │
   │     metric (CPU, memory, custom). Default for stateless.      │
   │                                                                │
   │     KEDA (Kubernetes Event-Driven Autoscaler)  ── replicas    │
   │     based on external signals (Kafka lag, SQS depth,          │
   │     Prometheus query, GPU util). ESSENTIAL for ML.            │
   │                                                                │
   │     VPA (Vertical Pod Autoscaler)  ── adjust requests/limits  │
   │     of single pod. Rarely used; conflicts with HPA.           │
   │                                                                │
   │   Node-level:                                                  │
   │     Cluster Autoscaler  ── traditional; adds/removes nodes    │
   │     when pods can't be scheduled. 1-2 min latency.            │
   │                                                                │
   │     Karpenter (AWS)  ── modern; faster, smarter, picks         │
   │     instance types dynamically. 30-60 sec latency.            │
   └────────────────────────────────────────────────────────────────┘
```

### HPA — the default

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: gateway-hpa }
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gateway
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target: { type: Utilization, averageUtilization: 70 }
```

HPA watches the metric and adjusts replica count to keep utilization at the target. **The problem for ML:** CPU and memory are bad signals for inference workloads. A vLLM pod can hold 100 concurrent requests at 50% GPU util but only 10 at 90% — CPU usage doesn't track.

### KEDA — the ML autoscaler

KEDA scales on **external signals** via **scalers** for 60+ sources. Critically: scale-to-zero supported.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: { name: vllm-scaler }
spec:
  scaleTargetRef: { name: vllm-deployment }
  minReplicaCount: 1
  maxReplicaCount: 10
  pollingInterval: 15
  cooldownPeriod: 300
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: |
        sum(vllm:num_requests_waiting) /
        sum(kube_deployment_status_replicas{deployment="vllm-deployment"})
      threshold: '5'
      activationThreshold: '0.5'
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: 'avg(vllm:gpu_cache_usage_perc)'
      threshold: '0.85'
```

**Walkthrough:** Two triggers. First scales when waiting requests per replica exceed 5 (queue building up). Second scales when KV cache usage exceeds 85% (memory pressure). KEDA picks the higher of the two recommendations. `activationThreshold: 0.5` is the "wake up from zero" threshold.

### Cluster Autoscaler vs Karpenter

```
   Cluster Autoscaler                Karpenter
   ──────────────────                ─────────
   Works with node groups            Provisions arbitrary instance types
   Slower (1-2 min)                  Faster (30-60 sec)
   Uses ASG / MIG underneath          Direct EC2 API (Fleet)
   Simple                            More flexible
   Mature                            AWS-native (and now broader)
```

For GPU workloads at Avrioc, Karpenter is the modern choice. It can pick `g5.2xlarge` for embedding pods and `p5.48xlarge` for big LLM pods within the same NodePool, optimizing cost.

### Worked example — KEDA + Karpenter for vLLM scale-from-zero

```
   ┌──────────────── Scale-from-zero flow ──────────────────┐
   │                                                          │
   │   T+0s:   No vLLM pods. No GPU nodes.                   │
   │                                                          │
   │   T+1s:   Request lands at gateway, queued.             │
   │                                                          │
   │   T+5s:   KEDA polls Prometheus, sees queue > 0.5,      │
   │           scales Deployment from 0 → 1 replica.         │
   │                                                          │
   │   T+10s:  Pod stuck in Pending (no GPU node).            │
   │           Karpenter sees unschedulable pod.              │
   │                                                          │
   │   T+30s:  Karpenter provisions p5.48xlarge.              │
   │                                                          │
   │   T+60s:  Node Ready. Pod scheduled. Image pulling.      │
   │                                                          │
   │   T+180s: Image pulled (15 GB), container starts.       │
   │                                                          │
   │   T+300s: vLLM loads model from cache, ready.           │
   │                                                          │
   │   T+302s: Request served.                               │
   │                                                          │
   │   Cold start: ~5 minutes. Mitigations:                   │
   │     - Pre-warmed image (DaemonSet image puller)          │
   │     - PVC-cached model weights                           │
   │     - Karpenter pre-warmed AMIs                          │
   │     - Keep min=1 replica always                          │
   └──────────────────────────────────────────────────────────┘
```

For latency-sensitive workloads, the 5-min cold start is unacceptable. Solution: `minReplicaCount: 1` keeps one pod always warm; KEDA scales up under load. Trade-off: pay for one GPU continuously.

### Autoscaling Q&A

**Q1. HPA vs KEDA — when each?**
> HPA for CPU/memory-bound workloads where utilization tracks load — gateway services, simple APIs. KEDA when the meaningful signal is external — queue depth, GPU memory pressure, token throughput. For LLM serving, KEDA on `vllm:num_requests_waiting` plus `vllm:gpu_cache_usage_perc` is the standard pattern. HPA's CPU metric just doesn't capture how loaded a vLLM pod actually is.

**Q2. Why does KEDA support scale-to-zero?**
> Cost. A GPU node costs $30+/hour idle. For workloads with bursty traffic (chat agents, ad-hoc batch jobs), scaling to zero between bursts saves real money. HPA can't do this — it requires `minReplicas >= 1`. KEDA introduces an `activationThreshold` separate from the scaling threshold, letting it activate from zero when a single request appears. Trade-off is cold-start latency, which we mitigate with image pre-pulling and PVC-cached models.

**Q3. Karpenter vs Cluster Autoscaler?**
> Cluster Autoscaler scales node groups — preconfigured AWS Auto Scaling Groups. Slower (1-2 min) and constrained to instance types you've predefined. Karpenter provisions nodes directly via EC2 Fleet, picks instance types dynamically based on pod requirements, and is faster (30-60 sec). For ML where you might need a g5 for embedding and a p5 for LLM in the same cluster, Karpenter's flexibility is a big win. AWS officially recommends Karpenter for new clusters.

**Q4. [Gotcha] Your HPA scales up but new pods OOMKill immediately. Why?**
> The Deployment was sized assuming a single pod handles X load. As HPA scales to N pods, total memory required is N × pod memory, but you may have set `requests.memory` too tight. Each new pod needs its full request to be scheduled, and at runtime each pod's actual memory usage matches what one pod uses (since they each handle 1/N of traffic). The fix is sizing pods at the per-replica steady-state load, not at peak. Sometimes the memory growth comes from per-replica caches (KV cache for vLLM), in which case you may need fewer larger replicas instead of more smaller ones.

---

## 12.8 GPU on Kubernetes

### Why this is hard

GPUs aren't just another resource. They have device drivers (host-side), runtime requirements (CUDA, cuDNN), library compatibility (PyTorch built against CUDA 12.4 needs driver >= 525), and they're expensive enough that fragmentation is painful.

### The stack

```
   ┌──────────────── GPU on Kubernetes Stack ──────────────────┐
   │                                                              │
   │   Application: PyTorch / vLLM / TensorRT                    │
   │                       │                                       │
   │                       ▼                                       │
   │   CUDA libraries (in container image)                        │
   │                       │                                       │
   │                       ▼                                       │
   │   NVIDIA Container Toolkit (on host) — exposes /dev/nvidia*  │
   │                       │                                       │
   │                       ▼                                       │
   │   NVIDIA driver (on host kernel)                             │
   │                       │                                       │
   │                       ▼                                       │
   │   GPU hardware                                                │
   │                                                                │
   │   K8s plumbing:                                                │
   │     NVIDIA k8s device plugin (DaemonSet) — advertises GPUs    │
   │     GPU Operator (umbrella) — installs all of the above       │
   │     Node feature discovery — labels nodes with GPU model      │
   └──────────────────────────────────────────────────────────────┘
```

### NVIDIA GPU Operator

The recommended way to manage all the above. It's a Helm chart that installs:
- The NVIDIA driver via container (no host install)
- NVIDIA Container Toolkit
- The k8s device plugin
- DCGM exporter (Prometheus metrics)
- Optional: MIG manager, node feature discovery

`helm install gpu-operator nvidia/gpu-operator`. Done.

### Requesting GPUs

```yaml
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
    resources:
      limits:
        nvidia.com/gpu: 4
```

Note `nvidia.com/gpu` is on **limits** (not requests). The device plugin allocates whole GPUs by default — you cannot request 0.5 GPU through the standard API.

### Sharing GPUs — three techniques

**1. MIG (Multi-Instance GPU)** — A100/H100 hardware partition. One GPU becomes up to 7 isolated instances with their own memory and SMs. Use for predictable, isolated multi-tenancy.

```yaml
resources:
  limits:
    nvidia.com/mig-3g.20gb: 1   # 3-slice MIG instance with 20GB
```

**2. Time-slicing** — software round-robin; multiple pods share one GPU but no memory isolation. Use for dev clusters.

**3. MPS (Multi-Process Service)** — NVIDIA's CUDA context sharing. Concurrent kernels from different processes share SMs. Higher throughput than time-slicing, less isolated than MIG.

### GPU node taints

```yaml
# applied to GPU nodes via Karpenter NodePool or eksctl
taints:
- key: nvidia.com/gpu
  effect: NoSchedule
```

```yaml
# pod toleration
spec:
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

Why? GPU nodes are expensive ($30/hr+). Tainting prevents accidental scheduling of CPU pods on them. Pods that need GPU add a toleration.

### Node affinity for specific GPU models

```yaml
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values: [NVIDIA-A100-SXM4-80GB, NVIDIA-H100-80GB-HBM3]
```

For workloads that need 80 GB VRAM (Llama-3 70B), pin to A100-80GB or H100-80GB; reject A100-40GB.

### GPU cold start problem

ML images are 5-15 GB. Pulling on a fresh node takes 2-5 minutes. Add ~30 seconds for CUDA driver init, model load, etc. **Total cold start: 5-10 minutes.** This is the biggest practical problem in K8s ML.

Mitigations:
1. **Image pre-puller DaemonSet** (Spegel, Kraken). Runs on every GPU node and pulls common images at startup.
2. **Karpenter pre-warmed AMI** — bake the image into the node AMI itself.
3. **PVC-cached model weights** — model artifact on EFS so multiple pods share.
4. **Min replicas = 1** — keep one always warm; KEDA scales up.

### GPU Q&A

**Q1. How do you share one GPU across multiple models?**
> Three options. MIG hardware partitioning on A100/H100 splits one GPU into up to 7 isolated instances with their own memory and SMs — best for predictable multi-tenancy with isolation guarantees. Time-slicing is software round-robin: multiple pods share one GPU with no memory isolation; OK for dev, dangerous for prod. MPS (Multi-Process Service) is NVIDIA's CUDA context sharing — better throughput than time-slicing, less isolated than MIG. Application-level: vLLM multi-LoRA serves many adapters from one base model, or Triton Inference Server batches across models within one container. For Avrioc-scale, MIG on H100 plus vLLM multi-LoRA is the high-end pattern.

**Q2. Your GPU pod takes 8 minutes to start serving. Where's the time?**
> Probable breakdown: 2-3 min image pull (5-15 GB image), 30-60 sec container start + CUDA init, 1-2 min model download from S3 if not cached, 30-60 sec model load into GPU memory, 30 sec readiness probe ramp. Total 5-8 min. Mitigations in order of impact: (1) pre-pull image via DaemonSet, (2) PVC-cache model on EFS, (3) Karpenter pre-warmed AMI, (4) min-replicas=1 always-warm, (5) speculative pre-warming based on traffic forecasts.

**Q3. Why do GPU pods need /dev/shm bumped?**
> Default `/dev/shm` in Docker/K8s is 64 MB. PyTorch DataLoader workers, NCCL inter-GPU communication for tensor parallelism, and vLLM all use shared memory for high-throughput IPC. 64 MB is laughable. Mount an `emptyDir` with `medium: Memory, sizeLimit: 16Gi` at /dev/shm. Symptom of getting it wrong: PyTorch crashes with shmem errors during data loading; NCCL hangs during multi-GPU init.

**Q4. [Gotcha] Your vLLM pod works locally with `docker run --gpus all` but crashes in K8s with CUDA errors. Why?**
> Most common: CUDA runtime version in your container doesn't match the host driver version. Local Docker is forgiving; K8s with the GPU Operator pins specific versions. Check `nvidia-smi` on the node (driver version), check `nvcc --version` in the container (CUDA version). PyTorch-bundled CUDA must be compatible with the host driver — driver >= CUDA-toolkit version. Second cause: missing `nvidia-container-runtime` config on a self-managed node. Third: forgot `nvidia.com/gpu: N` in resources, so kubelet didn't expose any GPU to the container.

---

## 12.9 StatefulSet, DaemonSet, Job, CronJob

### StatefulSet — when pod identity matters

```
   StatefulSet "milvus" with 3 replicas
     → Pods named milvus-0, milvus-1, milvus-2 (stable)
     → Each gets a stable PVC: data-milvus-0, data-milvus-1, etc.
     → DNS: milvus-0.milvus-headless.default.svc.cluster.local
```

Use for: distributed databases (etcd, Cassandra, Milvus), Kafka, anything where pod-X must always be pod-X with persistent data. **Most ML workloads are stateless and use Deployment instead.**

### DaemonSet — one pod per node

Use for things that should run everywhere:
- Logging agents (Fluent Bit shipping to CloudWatch / Loki)
- Node-exporter for Prometheus
- NVIDIA device plugin
- Image pre-puller (Spegel, Kraken)
- Calico / Cilium agent

### Job — one-shot batch

```yaml
apiVersion: batch/v1
kind: Job
metadata: { name: train-llama }
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: trainer
        image: my-trainer:abc
        command: [python, train.py]
        resources:
          limits: { nvidia.com/gpu: 8 }
```

Used for training jobs, batch inference, data preprocessing.

### CronJob — scheduled Job

```yaml
apiVersion: batch/v1
kind: CronJob
metadata: { name: nightly-retrain }
spec:
  schedule: "0 2 * * *"  # 2am daily
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: retrain
            image: my-retrain:latest
            command: [python, retrain.py]
```

For drift-triggered retraining, an EventBridge rule + KEDA `cron` scaler may be cleaner than CronJob.

### Init container — model weight downloader

```yaml
spec:
  initContainers:
  - name: model-downloader
    image: amazon/aws-cli
    command:
    - sh
    - -c
    - aws s3 cp s3://prod-ml-models/llama3-70b-awq/ /models/llama3-70b/ --recursive
    volumeMounts:
    - { name: models, mountPath: /models }
  containers:
  - name: vllm
    image: vllm/vllm-openai:v0.6.3
    args: [--model, /models/llama3-70b]
    volumeMounts:
    - { name: models, mountPath: /models }
  volumes:
  - name: models
    emptyDir: { sizeLimit: 200Gi }
```

Init containers run to completion **before** main containers start. Pattern: download artifacts, set permissions, run migrations.

---

## 12.10 Probes — readiness, liveness, startup

### Three probes you must understand

```
   Liveness probe:    "is the container alive?"
                      Failure → kubelet kills container, restarts.
                      Use for: deadlocked process, stuck event loop.

   Readiness probe:   "is the container ready to serve?"
                      Failure → removed from Service endpoints (no traffic).
                      Use for: model still loading, DB connection failed.

   Startup probe:     "has the container started yet?"
                      During startup phase, disables liveness checks.
                      Use for: slow-starting apps (model load 2 min).
```

### ML-specific probe pattern

```yaml
containers:
- name: vllm
  startupProbe:                  # 5 min budget for slow model load
    httpGet: { path: /health, port: 8000 }
    failureThreshold: 30
    periodSeconds: 10
  livenessProbe:                 # checks every 30s once started
    httpGet: { path: /health, port: 8000 }
    periodSeconds: 30
    failureThreshold: 3
  readinessProbe:                # remove from LB on transient issues
    httpGet: { path: /health, port: 8000 }
    periodSeconds: 5
    failureThreshold: 2
```

**Walkthrough:** During startup (model loading), `startupProbe` runs every 10s with 30 attempts = 5 min budget. While startupProbe is failing, liveness is suppressed. Once startup passes, liveness checks every 30s — three failures kills the pod. Readiness checks every 5s and pulls from LB on two failures — quick traffic drain on transient issues without killing.

### Probe Q&A

**Q1. Why have both liveness and readiness?**
> They answer different questions. Liveness asks "should this pod be killed and restarted?" Readiness asks "should this pod receive traffic right now?" A pod can be alive but not ready — for example, during a brief upstream DB outage where the app is fine but can't serve. You don't want to kill and restart the pod; you want to remove it from the load balancer until the DB recovers. Liveness restart for permanent failures; readiness traffic-drain for transient ones.

**Q2. Liveness probe killed your pod mid-inference. How do you fix?**
> Three causes. First, probe timeout too short — long inferences block the event loop. Make the health check completely independent of inference (separate uvicorn worker, separate thread). Second, probe interval too short — failureThreshold × periodSeconds must exceed normal pause durations. Third, no startupProbe so liveness fires during slow startup. Fix all three: separate /health endpoint that checks process liveness without doing real work; reasonable thresholds (failureThreshold: 3, periodSeconds: 30); startupProbe with generous failureThreshold for slow boots.

---

## 12.11 KServe — model-serving CRD

### Why KServe

Plain Deployment + Service works for one or two models. With dozens of models, you want a higher-level abstraction: "I want to serve this model with these resources, with autoscaling and canaries built in." That's **KServe** — InferenceService CRD on top of Knative serving.

### Mental model

KServe is to model serving what Deployment is to stateless apps. You declare what (model URI, framework), and KServe handles the how (Pod, Service, autoscaler, scale-to-zero, canary, payload logging).

### Sample InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata: { name: llama3-chat }
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 80           # request concurrency
    containers:
    - name: kserve-container
      image: vllm/vllm-openai:v0.6.3
      args: [--model, /mnt/models, --tensor-parallel-size=2]
      resources:
        limits: { nvidia.com/gpu: 2, memory: 96Gi }
    storageUri: s3://prod-ml-models/llama3-70b-awq/
  transformer:                # optional pre/post-process
    containers:
    - name: kserve-container
      image: my-org/llama-transformer:v1
```

### KServe components

- **predictor** — runs the model.
- **transformer** — optional preprocessing / postprocessing before/after predictor.
- **explainer** — optional model interpretability (SHAP, LIME).

### Canary built-in

```yaml
spec:
  predictor:
    canaryTrafficPercent: 10
    storageUri: s3://prod-ml-models/llama3-70b-v2/
```

KServe routes 10% of traffic to the new model, 90% to the previous.

### KServe vs alternatives

| | KServe | Plain Deployment | SageMaker |
|--|--------|------------------|-----------|
| Scale-to-zero | Yes (Knative) | No (HPA min=1) | Async/Serverless yes |
| Canary | Built-in | Custom (Argo Rollouts, Istio) | Built-in |
| Multi-framework | Yes | Manual | Yes |
| Payload logging | Async sink | Custom | Built-in |
| Best for | Many models on one cluster | One-off apps | AWS-managed |

---

## 12.12 GitOps — Argo CD vs Flux

### Why GitOps

Kubectl apply is imperative. GitOps is **declarative** — git is the source of truth, a controller continuously reconciles cluster state to match git. Benefits:
- Deploy via PR (auditable, reviewable)
- Drift detection (cluster ≠ git → alert)
- Self-healing (kubectl-edit by hand → reverted)
- Disaster recovery is just `git apply` to a fresh cluster

### Argo CD

```
   Git repo (k8s/prod/) ──┐
                           │
                           ▼
                      Argo CD controller
                           │
                           │ (continuous reconcile)
                           ▼
                      Kubernetes cluster
                           │
                           │ (status fed back)
                           ▼
                      Argo CD UI shows: Synced / OutOfSync / Healthy
```

Argo CD has a UI showing every Application's sync status. Self-heal toggle reverts manual changes. Sync waves order multi-step deploys.

### Flux

Pull-based like Argo CD but more CLI-driven, less UI. Better integration with Helm-only or kustomize-only workflows. Lighter-weight.

### Recommendation

For new clusters in 2026, Argo CD is the more popular choice; the UI matters for operators. Flux for highly-automated GitOps shops with no-UI culture.

---

## 12.13 Helm — package manager

### Why Helm

Templating Kubernetes YAML manually is painful. Helm bundles your manifests + values + templates into a **Chart**. Render once with values; install many times.

### Chart structure

```
   chart/
   ├── Chart.yaml          # metadata: name, version, dependencies
   ├── values.yaml         # default values
   ├── values-dev.yaml     # dev overrides
   ├── values-prod.yaml    # prod overrides
   ├── templates/
   │   ├── deployment.yaml      # uses {{ .Values.replicaCount }}
   │   ├── service.yaml
   │   ├── ingress.yaml
   │   ├── hpa.yaml
   │   ├── configmap.yaml
   │   ├── _helpers.tpl         # named templates
   │   └── NOTES.txt            # post-install message
   └── crds/                    # optional, for CRDs
```

### Sample template

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "myapp.fullname" . }}
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
      - name: app
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        resources: {{- toYaml .Values.resources | nindent 10 }}
```

```yaml
# values-prod.yaml
replicaCount: 3
image:
  repository: 123.dkr.ecr.me-central-1.amazonaws.com/myapp
  tag: v1.2.3-prod
resources:
  requests: { cpu: 1, memory: 4Gi }
  limits:   { cpu: 2, memory: 8Gi }
```

`helm install myapp ./chart -f values-prod.yaml --namespace prod`

### Helm vs Kustomize

- **Helm** — templating + package versioning. Use for distributed software (vendor charts).
- **Kustomize** — overlay-based, no templating. Use for own apps with simple env diffs.

Many teams use both: Helm for vendor charts (Prometheus, Argo CD), Kustomize for own apps. Argo CD supports both natively.

---

## 12.14 Common K8s ML pitfalls (senior-level)

1. **Mutable image tags.** `:latest` rebuilt mid-deploy means pods in same Deployment running different bits. Always use immutable tags + git SHA.
2. **No PodDisruptionBudget on critical Deployments.** Voluntary disruptions (node drain) can take down all replicas. Set `minAvailable: 2` for any user-facing service.
3. **Probes that block on dependencies.** Health endpoint that checks DB returns unhealthy when DB blips → all pods die at once. Health checks should be **process-level** liveness; readiness can include downstream checks.
4. **Forgotten `/dev/shm`.** vLLM and PyTorch DataLoader silently fail. Always mount memory-backed emptyDir.
5. **Cluster Autoscaler + StatefulSet with EBS.** EBS is AZ-bound; if Cluster Autoscaler removes the only node in an AZ where a StatefulSet pod's PV lives, the pod can never reschedule. Use EFS or set `volumeBindingMode: WaitForFirstConsumer`.

---

## 12.15 Ray — the ML-native distributed framework

### Why Ray exists alongside Kubernetes

Kubernetes orchestrates **containers** at the **node** level. But within an ML application, you often want to orchestrate **Python tasks** across **threads, processes, and machines** with first-class support for stateful actors, GPU scheduling, and shared memory. That's a different abstraction layer — and Ray fills it.

You can have **Ray on Kubernetes** (KubeRay) — Ray gives you in-application distribution, K8s gives you cluster-level resource management.

### Mental model with analogy

If Kubernetes is the highway system (moves vehicles between cities), Ray is the public transit network within a city (moves people between buildings). You need both. The highway gets you to a city; the transit network distributes you within it.

### Ray's surface area

```
   ┌──────────────────────── Ray Stack ───────────────────────────┐
   │                                                                │
   │   Ray Serve     ── HTTP/gRPC model serving                    │
   │   Ray Train     ── distributed training (PyTorch DDP/FSDP)    │
   │   Ray Tune      ── HP optimization                            │
   │   Ray RLlib     ── reinforcement learning                     │
   │   Ray Data      ── distributed dataset processing             │
   │   ─────────────────────────────────────────────────────────   │
   │   Ray Core      ── tasks, actors, object store                │
   │   ─────────────────────────────────────────────────────────   │
   │   Ray Cluster   ── head + workers (autoscaling)               │
   │   ─────────────────────────────────────────────────────────   │
   │   Substrate     ── bare metal / VMs / Kubernetes (KubeRay)    │
   └────────────────────────────────────────────────────────────────┘
```

### Ray Core — tasks vs actors

```python
import ray
ray.init()

# Task (stateless function)
@ray.remote
def square(x):
    return x ** 2

futures = [square.remote(i) for i in range(10)]
results = ray.get(futures)  # [0, 1, 4, 9, ..., 81]

# Actor (stateful class)
@ray.remote
class Counter:
    def __init__(self):
        self.count = 0
    def inc(self):
        self.count += 1
        return self.count

counter = Counter.remote()
ray.get(counter.inc.remote())  # 1
ray.get(counter.inc.remote())  # 2 — state persists between calls
```

**Walkthrough:** `@ray.remote` wraps a function or class for distributed execution. `.remote()` returns a future (`ObjectRef`). `ray.get()` blocks until the future resolves. Tasks are scheduled on any worker; actors are pinned to a specific worker process.

### Object store — zero-copy data sharing

Ray has a shared-memory **Plasma** object store on each node. Large objects (numpy arrays, tensors) live in shared memory; tasks/actors get zero-copy references. This is huge for ML — passing a 10 GB feature matrix between tasks doesn't serialize/deserialize.

### Ray Cluster topology

```
   ┌──────────────────────── Ray Cluster ─────────────────────────┐
   │                                                                │
   │   ┌── Head Node ────────────────────────────────────────────┐ │
   │   │   GCS (Global Control Service) — cluster state         │ │
   │   │   Driver process — your main script                     │ │
   │   │   Plasma object store                                   │ │
   │   │   Raylet — local scheduler                              │ │
   │   │   Dashboard (port 8265)                                 │ │
   │   └────────────────────────────────────────────────────────┘ │
   │                          │                                     │
   │   ┌── Worker Node 1 ──┐  │  ┌── Worker Node 2 ──┐             │
   │   │  Raylet            │  │  │  Raylet            │            │
   │   │  Plasma            │  │  │  Plasma            │            │
   │   │  Worker processes  │  │  │  Worker processes  │            │
   │   │  GPUs: 8           │  │  │  GPUs: 8           │            │
   │   └───────────────────┘  │  └────────────────────┘            │
   │                                                                 │
   │   On Kubernetes: head pod + N worker pods, managed by KubeRay  │
   └────────────────────────────────────────────────────────────────┘
```

### Ray Serve — model serving

```python
from ray import serve

@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 1})
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-large-en")

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

serve.run(Embedder.bind(), route_prefix="/embed")
```

**Walkthrough:** `@serve.deployment` registers the class as a deployment. `num_replicas` is initial count; autoscaling can adjust. `num_gpus: 1` reserves one GPU per replica. `@serve.batch` aggregates incoming requests into batches up to 32 or 50 ms wait — perfect for embedding throughput. `serve.run` deploys and exposes HTTP at `/embed`.

### Ray Serve composition (the killer feature)

```python
@serve.deployment
class Retriever:
    async def __call__(self, query: str) -> list[str]:
        return await self.search(query)

@serve.deployment
class Reranker:
    async def __call__(self, query, chunks):
        return self.rerank(query, chunks)

@serve.deployment
class LLM:
    async def __call__(self, prompt: str) -> str:
        return await self.generate(prompt)

@serve.deployment
class RAGPipeline:
    def __init__(self, retriever, reranker, llm):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm

    async def __call__(self, query: str) -> str:
        chunks = await self.retriever.remote(query)
        ranked = await self.reranker.remote(query, chunks)
        prompt = build_prompt(query, ranked[:5])
        return await self.llm.remote(prompt)

# Wire them up — each independent autoscaler
app = RAGPipeline.bind(
    retriever=Retriever.bind(),
    reranker=Reranker.bind(),
    llm=LLM.bind(),
)
serve.run(app, route_prefix="/rag")
```

The DAG: each component is independently autoscaled, GPU-allocated, replicated. The pipeline is just Python composition — no YAML, no deployment manifests.

### Fractional GPUs

```python
@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class SmallModel: ...
```

Two replicas on one GPU. No isolation (just sharing the device); fine for small models that don't saturate VRAM.

### Ray Train — distributed training

```python
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_func(config):
    model = ...
    optimizer = ...
    train_loader = ray.train.torch.prepare_data_loader(loader)
    model = ray.train.torch.prepare_model(model)
    for epoch in range(config["epochs"]):
        for batch in train_loader:
            ...
        ray.train.report({"loss": loss})

trainer = TorchTrainer(
    train_func,
    train_loop_config={"epochs": 10, "lr": 1e-4},
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
)
result = trainer.fit()
```

**Walkthrough:** `TorchTrainer` wraps PyTorch DDP. `prepare_data_loader` shards data; `prepare_model` wraps in DDP. `report` logs metrics. Ray handles fault tolerance — if a worker dies, the job resumes from the last checkpoint with a new worker.

### Ray Tune — hyperparameter search

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def trainable(config):
    model = build_model(config["lr"], config["dropout"])
    for step in range(100):
        loss = train_step(model)
        tune.report(loss=loss)

tuner = tune.Tuner(
    trainable,
    param_space={
        "lr": tune.loguniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.0, 0.5),
    },
    tune_config=tune.TuneConfig(
        num_samples=50,
        scheduler=ASHAScheduler(metric="loss", mode="min", grace_period=10),
    ),
)
results = tuner.fit()
```

**Walkthrough:** ASHA (Asynchronous Successive Halving Algorithm) early-stops bad trials, focuses compute on promising ones. Ray Tune runs trials in parallel across the cluster. Far more efficient than naive grid search.

### Ray Data — streaming dataset

```python
import ray.data

ds = (
    ray.data.read_parquet("s3://my-bucket/data/")
    .map_batches(preprocess, batch_size=1024)
    .filter(lambda r: r["valid"])
    .random_shuffle()
)

trainer = TorchTrainer(
    train_func,
    datasets={"train": ds},
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
)
```

Streams from S3 in parallel, shuffles globally, hands batches to training workers. Replaces PyTorch DataLoader for big-data training.

### When Ray vs vanilla K8s? Decision tree

```
   Are you doing distributed training across nodes?
     YES → Ray Train (or PyTorch Lightning + DDP launched by K8s Job)
     NO ↓
   Are you doing parallel hyperparameter search?
     YES → Ray Tune
     NO ↓
   Are you composing multiple models into a pipeline with independent autoscaling?
     YES → Ray Serve
     NO ↓
   Is your workload heterogeneous Python (mix of CPU/GPU tasks)?
     YES → Ray Core
     NO ↓
   Plain stateless inference at scale?
     → Vanilla K8s + KServe (Ray adds complexity without payoff)
```

### Ray Q&A

**Q1. Ray vs Spark — when each?**
> Ray for Python-first heterogeneous workloads — mix of CPU and GPU tasks, stateful actors, ML training plus serving in one cluster. Spark for structured ETL at PB scale, SQL-heavy work, strong partitioning. The dividing line: are your transforms pure dataframe operations? Spark. Are they arbitrary Python with state and GPU? Ray. For LLM training plus serving in one cluster, Ray. For massive Parquet shuffles, Spark.

**Q2. Task vs actor — give an ML example of each.**
> Task: parallel feature extraction over 10,000 images. Each call is independent and pure — `@ray.remote def extract_features(image): ...`. Tasks scale horizontally and don't carry state. Actor: a model server holding loaded weights. `@ray.remote class ModelActor: def __init__(self): self.model = load(); def predict(self, x): ...`. The model loads once when the actor starts; subsequent predicts reuse it. Actors keep state and are pinned to a worker.

**Q3. Ray Serve batching — how does it work?**
> `@serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)` decorates an async method. Ray Serve queues incoming requests; when either 32 are queued or 50 ms have passed, it calls the method with the batch. The method returns a list of results, Ray Serve fans them back to the original callers. This dramatically improves throughput on GPU workloads — embedding 32 texts together is far cheaper than 32 separate calls. For LLMs, layer with vLLM's continuous batching inside the replica.

**Q4. How do you autoscale Ray Serve?**
> `@serve.deployment(autoscaling_config={"min_replicas": 1, "max_replicas": 10, "target_ongoing_requests": 5})`. Ray Serve scales replicas based on queued + ongoing requests per replica. Pair with KubeRay's autoscaler at the cluster level: when Ray Serve needs more replicas than nodes can host, the cluster autoscaler provisions more pods. Scale-to-zero works but cold starts for LLMs are 30-90 seconds.

**Q5. Ray Train vs torchrun?**
> Torchrun is the lower-level PyTorch launcher — multi-process, multi-node coordination via env vars. It's fine for a fixed cluster. Ray Train wraps torchrun with fault tolerance (worker dies → resume from checkpoint), elastic scaling (add workers mid-training), integration with Ray Tune for HPO and Ray Data for streaming input. For a research workflow on a fixed cluster, torchrun is simpler. For a production training pipeline that needs reliability, Ray Train.

**Q6. Ray Tune vs Optuna?**
> Optuna is single-node — runs trials in sequence in one Python process. Great for laptop experimentation. Ray Tune distributes trials across a Ray cluster, supports advanced schedulers (ASHA, PBT, BOHB) out of the box, integrates with checkpoint/resume. For GPU-heavy HPO where you want 50 trials in parallel across 8 GPUs, Ray Tune. For a quick sweep on a single laptop, Optuna.

**Q7. [Gotcha] Your Ray cluster shows idle GPUs but tasks pending. Why?**
> Resource fragmentation. The pending task requests `num_gpus=1` plus `num_cpus=4`, and while there are idle GPUs, they're on a node where 4 CPUs aren't free. Ray needs both at once on the same worker. Or: the task requires a custom resource (`resources={"high_mem": 1}`) that no idle worker has. Diagnosis: `ray status` shows pending demand. Fix: rebalance resource requests, add a homogeneous node type, or split the workload.

**Q8. KubeRay vs running Ray on bare EC2?**
> KubeRay gives you the operator-pattern conveniences — declarative `RayCluster` and `RayJob` and `RayService` CRDs, integration with K8s autoscaling, observability via Prometheus, secrets via K8s Secrets. Bare EC2 with `ray up` is simpler for a single team experimenting. KubeRay wins for production multi-team clusters where you need RBAC, GitOps, and integration with the rest of your K8s ecosystem.

**Q9. Ray Serve LLM with vLLM — the modern pattern?**
> Ray Serve has a built-in vLLM integration in `ray.serve.llm`. You declare an LLMConfig with the model and quantization, Ray Serve manages the vLLM engine inside a deployment with continuous batching enabled. The Ray Serve layer handles HTTP, autoscaling, and pipeline composition; vLLM inside handles token-level scheduling. For prefill-decode disaggregation in the latest Ray versions, you can deploy prefill as one deployment and decode as another, splitting the GPU memory pressure across them.

---

## 12.16 Worked example — Avrioc reference architecture

Pulling it all together for the whiteboard:

```
   ┌─────────────────────── Avrioc K8s ML Platform ──────────────────────────┐
   │                                                                           │
   │   ┌── Ingress Layer ─────────────────────────────────────┐               │
   │   │  AWS ALB (Ingress) → NGINX Ingress / Istio Gateway  │               │
   │   │  TLS, WAF, rate limiting (per-tenant)                 │               │
   │   └──────────────────────┬──────────────────────────────┘               │
   │                          │                                                │
   │   ┌── Gateway Layer ─────▼──────────────────────────────┐               │
   │   │  FastAPI gateway (HPA on CPU, min=2, max=20)        │               │
   │   │  Auth (JWT), routing, observability                  │               │
   │   └──────────────────────┬──────────────────────────────┘               │
   │                          │                                                │
   │   ┌── Orchestration Layer ▼─────────────────────────────┐               │
   │   │  LangGraph agent in pod, Postgres-backed state      │               │
   │   │  Calls: vLLM, embed, retrieve, tools                 │               │
   │   └─┬──────────────────┬────────────────┬──────────────┘                │
   │     │                  │                │                                 │
   │     ▼                  ▼                ▼                                 │
   │   ┌──────────┐   ┌────────────┐   ┌──────────────┐                       │
   │   │ vLLM     │   │ Embedding  │   │ Vector DB    │                       │
   │   │ (KServe) │   │ (Ray Serve)│   │ (Milvus      │                       │
   │   │ KEDA on  │   │ batched    │   │  StatefulSet)│                       │
   │   │ queue+VRAM   │            │   │              │                       │
   │   │ p5 GPUs  │   │ g5 GPUs    │   │ EBS backed   │                       │
   │   └──────────┘   └────────────┘   └──────────────┘                       │
   │                                                                           │
   │   ┌── Cross-cutting ────────────────────────────────────┐                │
   │   │  Argo CD (GitOps)        Prometheus + Grafana       │                │
   │   │  cert-manager + Vault    Loki (logs)                │                │
   │   │  External Secrets        Tempo (traces)             │                │
   │   │  Karpenter (nodes)       Datadog (alt)              │                │
   │   │  GPU Operator            DCGM exporter              │                │
   │   └──────────────────────────────────────────────────────┘               │
   └──────────────────────────────────────────────────────────────────────────┘
```

> **How to say this in an interview:** "I'd architect Avrioc on EKS in me-central-1 with three node pools — CPU for the gateway and orchestration, mid-range GPU like g5 for embedding and rerank, and high-end like p5 for the LLM. KServe manages the LLM as an InferenceService backed by vLLM with KEDA autoscaling on `vllm:num_requests_waiting` and GPU cache utilization. Ray Serve runs the embedding and rerank pipeline because we want independent autoscaling and batching. LangGraph in a separate pod handles agent orchestration, calling those services and tools. Argo CD manages GitOps deploys; Prometheus, Grafana, Loki, Tempo for observability; Karpenter for fast GPU node provisioning. PVC-cached model weights on EFS so multiple GPU pods share the 140 GB Llama download."

---

## 12.17 Comprehensive Q&A — Kubernetes & Ray

**Q1. Pod vs Container vs Deployment?**
> A container is a single isolated process tree. A Pod is a wrapper around one or more containers that share networking and optional storage — they're the smallest unit Kubernetes schedules. A Deployment is a higher-level controller that manages a set of identical Pods via a ReplicaSet, providing rolling updates and rollbacks. You'd run a sidecar container in the same Pod as your main app; you'd run multiple replicas as a Deployment.

**Q2. Walk me through what happens when you `kubectl apply`.**
> Kubectl validates and POSTs to the API server. API server authenticates, validates, and writes desired state to etcd. Controllers running on the control plane watch etcd via informers — the Deployment controller sees the new spec, creates a ReplicaSet, which creates Pod objects. The Scheduler watches for unscheduled pods and binds them to nodes based on resource requests, taints, and affinity. Kubelet on each chosen node pulls the image and starts the container. Kube-proxy programs iptables for the Service. Once readiness probe passes, the Pod is added to the Service's endpoint set. Every step is observable via kubectl events.

**Q3. HPA vs KEDA — when each?**
> HPA scales replicas based on metrics-server data — CPU and memory primarily. Good for stateless services where load tracks CPU. KEDA scales on external signals — Kafka lag, SQS queue depth, Prometheus queries, GPU memory pressure — and supports scale-to-zero. For ML workloads where the meaningful signal isn't CPU but request queue or GPU cache utilization, KEDA is the right choice. For our vLLM serving, I'd use KEDA on `vllm:num_requests_waiting` plus `vllm:gpu_cache_usage_perc`.

**Q4. Requests vs Limits — and what's QoS class?**
> Requests are what the scheduler reserves on the node — placement decisions are based on requests. Limits are the runtime cap — CPU is throttled, memory triggers OOMKill. QoS class is derived: Guaranteed when requests equal limits for both, Burstable when one differs, BestEffort when neither set. Under node pressure, BestEffort is evicted first, then Burstable, then Guaranteed. For production-critical pods I set Guaranteed; for mostly-idle services I set Burstable to use spare capacity.

**Q5. GPU scheduling — walk me through it.**
> The NVIDIA GPU Operator deploys the device plugin as a DaemonSet, which advertises `nvidia.com/gpu` resources to kubelet. Pods request `nvidia.com/gpu: N` in resource limits — only limits because GPUs are integer-allocated whole. Tainted GPU nodes prevent accidental scheduling of CPU pods. Pods add a matching toleration. Node affinity pins to specific GPU models — A100-80GB versus A100-40GB matters. For sharing, MIG hardware-partitions an A100/H100 into up to 7 instances, each requested as `nvidia.com/mig-3g.20gb: 1`. Time-slicing or MPS share at the software level.

**Q6. How do you share one GPU across multiple models?**
> MIG hardware partitioning is the cleanest — A100/H100 splits into up to 7 isolated instances with their own memory and SMs. Time-slicing and MPS share at software with less isolation. At application level, vLLM multi-LoRA serves many adapters from one base model, or Triton Inference Server batches across models in one container. Ray Serve's multiplexing also works. Pick MIG for predictable multi-tenancy with isolation; vLLM multi-LoRA for many fine-tuned variants of the same base.

**Q7. KServe vs plain Deployment for model serving?**
> KServe abstracts away the boilerplate. It's an `InferenceService` CRD that gives you predictor + transformer + explainer pattern, scale-to-zero via Knative, canary built-in via `canaryTrafficPercent`, async payload logging to a sink for offline analysis. For one or two models, plain Deployment is simpler. For dozens of models, KServe pays for itself in the consistency it brings — every model gets the same observability, scaling, and rollout pattern.

**Q8. [Gotcha] Your GPU node takes 5 minutes to start serving. Where's the time?**
> Image pull dominates — ML images are 5-15 GB, takes 2-3 min on first pull. CUDA driver init another 30 seconds. Model download from S3 if not cached, 1-2 min for a 70B model. Model load into GPU memory, 30-60 sec. Readiness probe ramp, 30 sec. Total 5-10 min. Mitigations in priority: image pre-puller DaemonSet keeps common images warm on every GPU node; PVC on EFS caches model weights across pod restarts; Karpenter pre-warmed AMIs bake the image into the node image; min-replicas=1 keeps one pod always warm.

**Q9. [Gotcha] Inference pod OOMKills despite GPU memory being fine. What do you check?**
> CPU memory, not VRAM. The pod's host RAM holds tokenizers, request queues, Python heap, audio buffers. Check `kubectl describe pod` for last termination reason — OOMKilled is reported. Increase `requests.memory` and `limits.memory`. If it's a memory leak (long-lived FastAPI workers accumulating tensors), profile with memray and look for unbounded caches. Make sure `/dev/shm` is mounted as memory-backed emptyDir with sufficient size — undersized shmem causes obscure failures that look like OOM.

**Q10. Argo CD — what problem does it solve?**
> GitOps. Cluster state lives in git; a controller continuously reconciles the cluster to match. Deploy via PR — auditable, reviewable, revertible by `git revert`. Drift detection — manual kubectl-edit gets reverted by Argo or alerts. Self-healing — accidental delete restored. Disaster recovery — point Argo at a fresh cluster and it brings everything back. For multi-environment promotion, App-of-Apps pattern manages a dev/staging/prod hierarchy declaratively.

**Q11. Ray vs Spark — when each?**
> Ray for Python-first, heterogeneous workloads — mix of CPU and GPU tasks, stateful actors, ML training plus serving in one cluster. Spark for structured ETL at PB scale and SQL-heavy work. For LLM training and serving in one cluster, Ray. For massive Parquet shuffles, Spark. Many shops run both — Spark for the data lakehouse, Ray for ML.

**Q12. Ray task vs actor — when each?**
> Tasks for stateless, embarrassingly-parallel work — feature extraction over images, parallel inference over batches. Actors for stateful work — a model server holding loaded weights, a parameter server in distributed training, a counter or rate limiter. Tasks scale horizontally without coordination; actors are pinned to a worker and serialize calls per actor. Use the right tool: don't use an actor if state isn't needed; the actor adds scheduling overhead.

**Q13. How does Ray Serve's @serve.batch work with vLLM's continuous batching?**
> Two layers of batching that complement. Ray Serve aggregates incoming HTTP requests across replicas — `@serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)` collects up to 32 requests or 50 ms, hands the batch to one replica. vLLM inside that replica then continuously batches at the token level — every iteration's KV cache is re-allocated based on which sequences are still generating, allowing new requests to join mid-flight. Combined: throughput is maximized by aggregating at HTTP layer and continuous-batching at GPU layer.

**Q14. Ray Train vs PyTorch DDP launched by K8s Job?**
> Both work. Pure PyTorch DDP via `torchrun` in a K8s Job is lower-level — multi-pod coordination via env vars and a head pod, no fault tolerance for worker death. Ray Train wraps DDP/FSDP/DeepSpeed with checkpointing, fault tolerance (worker dies → resume), elastic scaling, and integration with Ray Tune and Ray Data. For a research workflow on a fixed cluster, raw torchrun is fine. For production training where reliability matters, Ray Train.

**Q15. KubeRay — what does it give you?**
> Kubernetes operator for Ray. CRDs: `RayCluster` (declarative Ray cluster on K8s), `RayJob` (one-shot batch run), `RayService` (zero-downtime Ray Serve). Integration with K8s autoscaling — when Ray needs more workers, KubeRay creates more pods, which Karpenter or Cluster Autoscaler turns into nodes. Standard observability via Prometheus and Grafana. Secrets via K8s Secrets. The operator pattern is the cleanest way to run Ray in production multi-team environments.

**Q16. Service mesh for ML — Istio or Linkerd?**
> Both give mTLS, traffic splitting, retries, distributed tracing. Istio is more featureful and complex; Linkerd is simpler with a smaller resource footprint. For ML, the win is canary deployments via traffic split — serve model v2 to 10% of traffic, observe metrics, ramp up. KServe's built-in canary suffices for simple cases; mesh shines when you have many services and want consistent policies. For Avrioc, I'd default to no mesh, add it when traffic-shaping or zero-trust policies become real requirements.

**Q17. Helm chart for ML — what would it template?**
> Image repository and tag (so the same chart deploys dev/staging/prod with different images). Resource requests and limits per environment. Replica counts and HPA min/max. Model URI (S3 path) injected as env var. Vault or Secrets Manager refs for credentials. ServiceMonitor for Prometheus. Optional: KServe InferenceService template, KEDA ScaledObject template, Istio VirtualService. For a vLLM chart specifically, vLLM args (model, tensor-parallel-size, quant, max-model-len) parameterized.

**Q18. Canary vs blue-green — pick one.**
> Canary for cost-conscious gradual rollout. New version gets 5% of traffic, observe error rate and latency for 30-60 min, ramp to 50%, then 100%. Blue-green for atomic switchover where you can afford double capacity. Both versions deployed at full size; flip traffic at the router instantly. Blue-green has zero rollout duration but doubles infrastructure cost during the window. For LLM serving where GPU is expensive, I prefer canary — Argo Rollouts or Istio handles the traffic split.

**Q19. NetworkPolicy — when do you use it?**
> Multi-tenant clusters and compliance scenarios. Default-deny at the namespace level, then explicit allow rules. Example: data-tier pods reachable only from app-tier pods on specific ports. Limits blast radius of a compromised pod. Default in K8s is "allow all" for backward compatibility; for any production cluster handling regulated data, NetworkPolicy is non-negotiable. Calico, Cilium, or AWS VPC CNI's network policy add-on enforces them.

**Q20. [Gotcha] Your vLLM pod works locally but crashes in K8s. Common causes?**
> First, `/dev/shm` size — default is 64 MB, vLLM needs 16 GB+. Mount memory-backed emptyDir. Second, NCCL with multi-GPU — needs `IPC_LOCK` capability and proper interconnect (EFA on AWS); without, it hangs. Third, CUDA mismatch — image's CUDA version and host driver must be compatible. Fourth, GPU memory limits not reflecting actual VRAM — Kubernetes doesn't enforce VRAM limits, vLLM gets "all available" and competes with other pods. Pin to dedicated GPU via `nvidia.com/gpu: N`.

**Q21. PodDisruptionBudget — why?**
> Prevents voluntary disruptions (node drain, cluster upgrade) from taking down too many replicas at once. `minAvailable: 2` means at most one replica can be evicted simultaneously, even during a node drain. Critical for user-facing services. Without PDB, a cluster admin draining a node where two of your three replicas live takes you to one replica during the drain — unsafe.

**Q22. [Gotcha] Rolling update drops traffic briefly. Fix?**
> Multi-step. First, readiness probes — old pods must report not-ready before being terminated, but probes need a brief window to actually fail. Add a `preStop` hook that sleeps 5-10 seconds, letting in-flight requests drain. Second, set `terminationGracePeriodSeconds: 30` so the pod has time after preStop. Third, ensure your app actually drains: stop accepting new requests at SIGTERM, wait for in-flight to complete. Fourth, RollingUpdate strategy `maxUnavailable: 0` so we never go below desired count.

**Q23. EKS vs self-managed K8s — when each?**
> EKS for almost everything. AWS manages the control plane, etcd backups, version upgrades. You manage worker nodes (via Karpenter) and workloads. Self-managed only for unusual requirements: OpenShift compliance, on-prem hybrid, deep customization. For Avrioc, EKS in me-central-1 is the answer.

**Q24. Helm vs Kustomize?**
> Helm for templated packages with versioning — vendor charts (Prometheus, Argo CD, cert-manager). Kustomize for overlay-based config of your own apps with simple env diffs. Many teams use both: Helm for vendor, Kustomize for own. Argo CD supports both natively. For ML, vLLM Helm chart for serving, Kustomize overlays for the FastAPI gateway across environments.

**Q25. Avrioc K8s recommendation in one paragraph?**
> EKS in me-central-1 with three node pools managed by Karpenter — CPU for gateway and orchestration, g5 GPU for embedding and rerank, p5 GPU for LLM. NVIDIA GPU Operator on day one. KServe for LLM serving with KEDA autoscaling on `vllm:num_requests_waiting`. Ray Serve for the embedding and rerank pipeline with batched deployments. LangGraph in a separate pod for agent orchestration. Argo CD for GitOps deploys; Prometheus, Grafana, Loki, Tempo for observability; cert-manager + External Secrets + AWS Secrets Manager for credentials. PVC-cached model weights on EFS so multiple GPU pods share the 140 GB Llama download. Three-environment promotion via App-of-Apps.

---

## 12.18 Resume tie-in narratives

> **How to say this in an interview about TrueBalance:**
> "At TrueBalance my Lambda was containerized via a multi-stage Dockerfile — `nvidia/cuda:12.4-devel` for the build stage with compiler-heavy XGBoost + Python deps, then a `runtime` stage with only the artifacts. Final image around 800 MB, pulled in 15 seconds. Even though Lambda was the runtime not Kubernetes, the same image conceptually runs anywhere — and as we considered scaling, the migration target was an EKS cluster with KServe for the predictor, KEDA for autoscaling, and the same image deployed there."

> **How to say it about the ML workspace assistant:**
> "I architected the ML workspace as a LangGraph agent in an EKS pod, with sidecar pods for tool execution against Jira, GitHub, Athena, and Jenkins. Postgres-backed checkpointer for state. Ray Serve fronted the embedding and reranking layer for the RAG component. The whole thing was Helm-charted with values per environment, Argo CD-deployed via GitOps, observed in Grafana with Prometheus scraping per-tool latency."

> **How to say it about ResMed IHS:**
> "At ResMed the IHS platform ran on EKS with KServe InferenceServices per model. Each clinical model was a multi-container endpoint pattern — preprocess sidecar, main predictor, postprocess sidecar. Argo CD for GitOps, Argo Workflows for retraining DAGs, MLflow for experiment tracking. I contributed the FastAPI transformer containers and Helm chart parameterization."

---

Continue to **[Chapter 13 — Frameworks](13_frameworks.md)**.
