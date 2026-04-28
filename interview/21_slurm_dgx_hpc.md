# Chapter 21 — Slurm, NVIDIA DGX, and the HPC stack

> **Why this chapter exists:** The Avrioc JD explicitly lists **Slurm** and **NVIDIA DGX**, but the rest of the prep pack focuses on Kubernetes/Ray (which assume cloud-native serving). This chapter closes that gap. Avrioc almost certainly runs an on-premises DGX cluster in Abu Dhabi for training, with Kubernetes for inference — and they want you to understand both halves.

---

## 1. The mental model: Slurm for training, K8s for inference

If you remember nothing else, remember **this picture**:

```
┌─────────────────────────────────────────────────────────────┐
│           Avrioc on-prem (likely DGX BasePOD/SuperPOD)      │
├──────────────────────────────────┬──────────────────────────┤
│      TRAINING SIDE (Slurm)       │   SERVING SIDE (K8s)     │
│                                  │                          │
│ • Multi-node DDP/FSDP jobs       │ • Online inference (vLLM)│
│ • Hyperparam sweeps              │ • REST/gRPC APIs         │
│ • Long-running batch             │ • Autoscale to traffic   │
│ • Fair-share queues              │ • Rolling deploys        │
│ • Gang scheduling, MPI           │ • Triton/Ray Serve       │
│ • srun / sbatch                  │ • kubectl / Helm         │
│                                  │                          │
│  Filesystem: Lustre / WekaFS / NFS shared between both       │
└──────────────────────────────────┴──────────────────────────┘
```

**The interview-winning sentence:**
> "Slurm and Kubernetes solve different problems. Slurm is gang-scheduling for tightly-coupled HPC jobs — multi-node training, MPI, fair-share queues. Kubernetes is the autoscaling layer for stateless serving — inference, APIs, rolling deploys. On a DGX cluster I'd run training under Slurm with PyTorch DDP via torchrun, then ship the trained weights to S3 or a shared filesystem and deploy with vLLM on K8s for inference. They share the same DGX hardware via partitions or node taints."

---

## 2. Slurm — the bare minimum you need to know

Slurm (Simple Linux Utility for Resource Management) is the de-facto HPC scheduler. Key concepts:

| Concept | What it is | When mentioned |
|---------|-----------|---------------|
| **Node** | A physical machine (e.g., one DGX A100 = one node with 8 GPUs) | All over |
| **Partition** | A queue / pool of nodes (e.g., `gpu-train`, `gpu-debug`) | `--partition=gpu` |
| **Job** | A unit of work submitted with `sbatch` or `srun` | Job ID, `squeue` |
| **Step** | A sub-process inside a job, launched via `srun` | DDP launches one step per node |
| **Account / QoS** | Billing/priority unit, used for fair-share | `--account=ai_team` |
| **Gres (Generic Resources)** | Non-CPU resources, primarily GPUs | `--gres=gpu:8` |

### 2.1 The five Slurm commands you must know

```bash
sbatch train.sh                  # submit a batch job; returns a job ID
srun --pty bash                  # interactive session on a compute node
squeue -u $USER                  # see your queued/running jobs
scancel <job_id>                 # kill a job
sinfo                            # show partitions and node states
```

### 2.2 An sbatch script for multi-node DDP training

This is the canonical script — be ready to write it on a whiteboard.

```bash
#!/bin/bash
#SBATCH --job-name=llama-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=4                       # 4 DGX nodes
#SBATCH --ntasks-per-node=8             # 8 ranks per node (one per GPU)
#SBATCH --gres=gpu:8                    # 8 GPUs per node
#SBATCH --cpus-per-task=12              # 12 CPUs per rank for dataloader
#SBATCH --mem=0                         # use all node memory
#SBATCH --time=12:00:00                 # 12-hour time limit
#SBATCH --output=logs/%x-%j.out         # job_name-job_id.out
#SBATCH --error=logs/%x-%j.err

# Networking for NCCL across InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch one torchrun per node; SLURM gives us 4 nodes and we want 8 procs/node
srun --ntasks-per-node=1 --gres=gpu:8 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --config configs/llama_8b_lora.yaml
```

**Three details a senior reviewer notices:**

1. `--ntasks-per-node=1` on the `srun` (not 8) — torchrun spawns the 8 processes itself, so srun launches one *torchrun* per node.
2. `MASTER_ADDR` derived from `$SLURM_JOB_NODELIST` so rendezvous works across nodes.
3. NCCL env vars set explicitly so InfiniBand is used (DGX nodes have IB; without it, NCCL silently falls back to TCP and your training is 10x slower).

### 2.3 Common Slurm interview questions

**Q: "What does gang scheduling mean and why does it matter for ML?"**
A: All-or-nothing scheduling — a 4-node job runs only when 4 nodes are simultaneously free. ML training is tightly coupled (every gradient step requires all-reduce across all ranks), so partial allocation = wasted GPUs sitting idle waiting for the rest. Slurm guarantees gang scheduling; Kubernetes by default does not (you need Volcano or Kueue for it).

**Q: "How do you do hyperparameter sweeps on Slurm?"**
A: Job arrays — `#SBATCH --array=0-31` runs 32 jobs in parallel, each gets a `$SLURM_ARRAY_TASK_ID` (0..31) which you map to a hyperparam tuple. Combined with a sweep config in YAML, this is the bread-and-butter HPC HP search. Or use Ray Tune as the orchestrator and Slurm as the backend.

**Q: "Fair share — what is it?"**
A: Slurm tracks how many resources each account has consumed historically. The next job's priority is reduced if the account has over-consumed recently. Multi-team clusters use this to prevent one team from monopolizing. As a platform engineer you'd tune `PriorityWeightFairshare` in `slurm.conf`.

**Q: "Can you mix Slurm and Kubernetes on the same cluster?"**
A: Yes — common patterns: (1) use Slurm for training, K8s for serving on disjoint node pools; (2) project `slurm-operator` lets you submit Slurm jobs from K8s; (3) NVIDIA's BCM (Base Command Manager) integrates both natively on DGX BasePOD.

---

## 3. NVIDIA DGX — what to know

A **DGX** is NVIDIA's pre-engineered AI server — 8 H100 (or A100/B200) GPUs in one chassis, all interconnected by **NVLink/NVSwitch** (intra-node), with **InfiniBand** uplinks (inter-node). They come in three rough sizes:

| Form factor | What it is |
|------------|------------|
| **DGX H100/A100** | One server, 8 GPUs |
| **DGX BasePOD** | Reference architecture for ~4–32 DGX servers (small cluster) |
| **DGX SuperPOD** | Reference architecture for 32+ DGX servers, full HPC fabric |

### 3.1 The two interconnect tiers (memorize this)

```
WITHIN A DGX NODE:
  GPU0 ─NVLink/NVSwitch─ GPU1 ─NVLink/NVSwitch─ ... ─ GPU7
  Speed: ~900 GB/s aggregate (H100), ~600 GB/s (A100)
  Used for: tensor-parallel within node

BETWEEN DGX NODES:
  Node0 ─InfiniBand HDR/NDR─ Node1 ─...
  Speed: 200-400 Gb/s per port, multiple ports (800 Gb/s+ aggregate)
  Used for: data-parallel all-reduce, pipeline parallel, tensor-parallel across nodes (lower-bandwidth, avoid)
```

**Key implication for model parallelism:** put your highest-traffic dimension (tensor parallel) **within** a node where NVLink is fast; put data parallel **across** nodes where IB is slower but bandwidth requirement is lower (only gradients, once per step).

### 3.2 DGX-aware interview questions

**Q: "If you have 4 DGX H100 nodes (32 GPUs total) and want to train a 70B model, how do you partition?"**
A:
- TP = 8 (within each node, leveraging NVLink)
- PP or DP across the 4 nodes (over IB)
- For Llama-70B: TP=8 + PP=4 is one valid split; another is TP=8 + DP=4 with FSDP if memory permits.
- Quote concrete numbers: 70B FP16 = 140 GB weights, plus optimizer states (Adam: 8x weights = 1.1 TB) → must shard. With FSDP + ZeRO-3, optimizer states distribute across DP group.

**Q: "Why do you care about NCCL_TOPO_FILE?"**
A: NCCL collective operations (all-reduce, all-gather) plan their communication ring based on the network topology. The default auto-detection sometimes picks suboptimal paths. On a DGX system, NVIDIA ships a topology file that pins the optimal ring; setting `NCCL_TOPO_FILE` ensures NCCL uses it. Forgetting this can cost 20-30% of training throughput.

**Q: "How does GPU Direct RDMA help?"**
A: It lets a GPU on node A read/write directly to a GPU on node B without going through the host CPU/memory. On a DGX cluster with IB, GPUDirect RDMA halves all-reduce latency for cross-node communication. As a platform engineer, you ensure the right NIC drivers (MOFED) are installed and `nv_peer_mem` is loaded.

**Q: "DGX comes with what software stack?"**
A: DGX OS (Ubuntu-based), CUDA, cuDNN, NCCL, NVIDIA drivers, Docker + NVIDIA Container Toolkit, Base Command Manager (BCM) for cluster ops, optionally NGC (NVIDIA's container catalog) for pre-built ML images.

---

## 4. The full HPC AI stack on Avrioc's likely setup

```
┌──────────────────────────────────────────────────────────────┐
│ User-facing: research notebooks, Slurm CLI, Kubeflow UI      │
├──────────────────────────────────────────────────────────────┤
│ Schedulers: Slurm (training) + Kubernetes (serving)          │
├──────────────────────────────────────────────────────────────┤
│ Frameworks: PyTorch + DeepSpeed/FSDP, vLLM, Ray              │
├──────────────────────────────────────────────────────────────┤
│ Containers: Docker + NVIDIA Container Toolkit + Pyxis/Enroot │
├──────────────────────────────────────────────────────────────┤
│ Compute: NVIDIA DGX H100 / A100 nodes                        │
├──────────────────────────────────────────────────────────────┤
│ Network: InfiniBand HDR/NDR + NVLink/NVSwitch                │
├──────────────────────────────────────────────────────────────┤
│ Storage: Lustre / WekaFS / DDN — high-throughput parallel FS │
└──────────────────────────────────────────────────────────────┘
```

You don't need to be an expert in every layer — you need to be able to **point at each layer and say what it does**.

### 4.1 Container runtime on Slurm: Pyxis + Enroot

Slurm's native container runner. Lets you `srun --container-image=docker://nvcr.io/nvidia/pytorch:24.04-py3 ...` and your job runs inside that container, with GPUs exposed.

**Why it matters:** on a shared DGX cluster you don't want every team's Python deps polluting the host. Pyxis/Enroot give you per-job containers without the K8s overhead.

### 4.2 Storage: parallel filesystems

DGX clusters need **shared, fast** storage because:
- Datasets are massive (1-100 TB)
- Multiple nodes read the same shards in parallel
- Checkpoints (multi-GB) write from many ranks simultaneously

Common choices: **Lustre, WekaFS, DDN ExaScaler, NVIDIA's BCM-bundled storage**. As a candidate you mention them by name and explain you've used "shared parallel storage" — don't claim deep expertise unless you have it.

---

## 5. Bridge questions: Slurm ↔ Kubernetes

These are the most likely "platform thinking" questions.

**Q: "I have a fine-tune job that takes 3 days on 4 nodes. Where does it run, and how does the trained model get to inference?"**
A:
1. **Train on Slurm** (`sbatch train.sh`) — gang scheduling, fair-share, IB networking via NCCL.
2. **Checkpoint to shared FS** every N steps; final weights to S3/MinIO with a model registry entry (MLflow / W&B).
3. **CI** (GitHub Actions or Jenkins) picks up the new model, packages a vLLM container with that weight, pushes to ECR.
4. **Argo CD or kubectl rollout** updates the K8s `Deployment` with the new image; rolling update with readiness probes; old pods drained.
5. **Canary 1% → 5% → 50% → 100%** via service mesh (Istio) or feature flag.

**Q: "How do you autoscale K8s inference pods on GPU usage, not CPU?"**
A: Default HPA scales on CPU/memory which is wrong for inference. Use **KEDA** (Kubernetes Event-Driven Autoscaler) with a Prometheus scaler reading `vllm:num_requests_waiting` or queue depth. Scale on the metric that matters: queue length (latency proxy) or request rate, not GPU util (already saturated under load).

---

## 6. Cheatsheet — sentences to drop in the interview

- *"On a DGX node I keep tensor parallelism within the chassis to leverage NVLink, and use data parallelism across nodes over InfiniBand — that's the bandwidth-aware partitioning."*
- *"Slurm gives me gang scheduling and fair-share, which Kubernetes by default doesn't — I'd use Slurm for training and K8s for serving on shared hardware."*
- *"For multi-node DDP I'd derive `MASTER_ADDR` from `$SLURM_JOB_NODELIST`, set `NCCL_IB_DISABLE=0`, and launch one torchrun per node via srun."*
- *"NCCL_TOPO_FILE matters on DGX because NCCL plans its communication ring around the topology — getting it wrong costs 20-30% throughput."*

---

Continue to **[Chapter 22 — 2-Day Intensive Study Plan](22_2day_study_plan.md)**.
