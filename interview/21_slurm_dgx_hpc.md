# Chapter 21 — Slurm, NVIDIA DGX, and the HPC stack

> **Why this chapter exists:** The Avrioc JD explicitly lists **Slurm** and **NVIDIA DGX**, but most candidates' AI/ML preparation focuses on Kubernetes and Ray, which assume cloud-native serving infrastructure. That's a gap. Avrioc almost certainly runs an on-premises DGX cluster in Abu Dhabi for training large models, with Kubernetes layered on for inference, and they want to hear that you understand both halves and can talk about how the trained-model artifact flows from the HPC side to the serving side. This chapter is your bridge into that conversation.

---

## 1. The big mental model — Slurm for training, K8s for inference

If you only remember one thing from this chapter, remember this picture and the sentence underneath it. They are the highest-leverage seventy seconds of preparation you can do for the Avrioc interview's HPC-flavored questions.

### Why these are different schedulers

The fundamental observation is that **training jobs and inference jobs have completely different scheduling requirements**, and any single scheduler that tried to do both well would end up doing both badly. Training is gang-scheduled and tightly-coupled — a multi-node DDP job either gets all its nodes simultaneously or it sits idle, because every gradient step requires an all-reduce across all ranks. Inference is the opposite — each request is independent, latency-sensitive, and the system needs to autoscale up and down based on traffic. Slurm was designed for HPC, where gang scheduling and fair-share queues are non-negotiable. Kubernetes was designed for stateless web services, where rolling deployments and horizontal autoscaling are the priority. Use each for what it's best at.

### The picture

```
┌─────────────────────────────────────────────────────────────┐
│           Avrioc on-prem (likely DGX BasePOD/SuperPOD)      │
├──────────────────────────────────┬──────────────────────────┤
│      TRAINING SIDE (Slurm)       │   SERVING SIDE (K8s)     │
│                                  │                          │
│  • Multi-node DDP/FSDP jobs      │  • Online inference (vLLM│
│  • Hyperparam sweeps (job arrays)│    Triton, Ray Serve)    │
│  • Long-running batch (days)     │  • REST/gRPC APIs        │
│  • Fair-share queues             │  • Autoscale to traffic  │
│  • Gang scheduling, MPI          │  • Rolling deploys       │
│  • srun / sbatch                 │  • kubectl / Helm        │
│                                  │                          │
│  Filesystem: Lustre / WekaFS / NFS shared between both       │
└──────────────────────────────────┴──────────────────────────┘
              │                                  │
              └─────► trained checkpoint ────────┘
                      → S3 / model registry
                      → vLLM container build
                      → K8s rolling deploy
```

### The interview-winning sentence

> "Slurm and Kubernetes solve different problems. Slurm is gang-scheduling for tightly-coupled HPC jobs — multi-node training, MPI, fair-share queues. Kubernetes is the autoscaling layer for stateless serving — inference, APIs, rolling deploys. On a DGX cluster I'd run training under Slurm with PyTorch DDP via torchrun, then ship the trained weights to S3 or a shared filesystem and deploy with vLLM on Kubernetes for inference. They share the same DGX hardware via partitions or node taints, so I'm not running two physical clusters — just two scheduling layers on top of one."

---

## 2. Slurm fundamentals — the bare minimum you need to know

Slurm (Simple Linux Utility for Resource Management) is the de-facto HPC job scheduler. It's open-source, originally developed at Lawrence Livermore National Laboratory, now maintained by SchedMD. It runs on most of the world's top supercomputers and on virtually every NVIDIA DGX deployment.

### 2.1 The vocabulary

| Concept | What it is | Where it shows up |
|---------|-----------|---------------------|
| **Node** | A physical machine (one DGX A100 = one node with 8 GPUs) | Everywhere |
| **Partition** | A queue / pool of nodes (e.g., `gpu-train`, `gpu-debug`) | `--partition=gpu` |
| **Job** | A unit of work submitted with `sbatch` or `srun` | Job ID, `squeue` |
| **Step** | A sub-process inside a job, launched via `srun` | DDP launches one step per node |
| **Account / QoS** | Billing/priority unit, used for fair-share | `--account=ai_team` |
| **GRES (Generic Resources)** | Non-CPU resources, primarily GPUs | `--gres=gpu:8` |

The mental analogy that helps me: Slurm is to HPC clusters what `kubectl` is to Kubernetes — the user-facing scheduler interface — but with very different design priorities. Where kubectl is event-driven and reconciliation-based (declare desired state, controllers reconcile), Slurm is request-based (submit a job, the controller makes one scheduling decision and runs it).

### 2.2 The five Slurm commands you must know cold

```bash
sbatch train.sh                  # submit a batch job; returns a job ID
srun --pty bash                  # interactive session on a compute node
squeue -u $USER                  # see your queued/running jobs
scancel <job_id>                 # kill a job
sinfo                            # show partitions and node states
```

Examples of typical output you should recognize:

```
$ squeue -u sachin
JOBID    PARTITION  NAME              USER     ST  TIME       NODES NODELIST
12345    gpu        llama-finetune    sachin   R   2:14:33    4     dgx-[01-04]
12346    gpu        eval-sweep        sachin   PD  0:00       8     (Resources)

$ sinfo
PARTITION  AVAIL  TIMELIMIT  NODES  STATE     NODELIST
gpu*       up     7-00:00    16     idle      dgx-[05-20]
gpu*       up     7-00:00    4      mixed     dgx-[01-04]
debug      up     2:00:00    2      idle      dgx-[21-22]
```

`R` means running, `PD` means pending, `(Resources)` in the NODELIST column means the job is queued waiting for GPUs to free up.

### 2.3 The canonical multi-node DDP sbatch script

This is the script you should be able to write on a whiteboard. Read every line and understand why it's there — the interviewer may point at any line and ask "why this?"

```bash
#!/bin/bash
#SBATCH --job-name=llama-finetune
#SBATCH --partition=gpu
#SBATCH --nodes=4                       # 4 DGX nodes
#SBATCH --ntasks-per-node=8             # 8 ranks per node (one per GPU) — see note below
#SBATCH --gres=gpu:8                    # 8 GPUs per node (DGX has 8)
#SBATCH --cpus-per-task=12              # 12 CPUs per rank for dataloader workers
#SBATCH --mem=0                         # use all node memory
#SBATCH --time=12:00:00                 # 12-hour wall time limit
#SBATCH --output=logs/%x-%j.out         # job_name-job_id.out
#SBATCH --error=logs/%x-%j.err

# Networking for NCCL across InfiniBand
export NCCL_IB_DISABLE=0                # ensure InfiniBand is used
export NCCL_DEBUG=INFO                  # log NCCL operations for debug
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

### 2.4 The three details a senior reviewer notices

**One.** `--ntasks-per-node=1` on the `srun` line, not 8. This is counterintuitive and almost everyone gets it wrong the first time. The reason: torchrun spawns the eight worker processes itself, one per GPU on the node. If you set `srun --ntasks-per-node=8`, srun would launch eight torchruns per node, each of which would then try to spawn eight more processes — you'd have sixty-four processes per node fighting over eight GPUs. The correct mental model is: srun launches one torchrun per node (so four total across four nodes); torchrun is the one that fans out per-node.

**Two.** `MASTER_ADDR` is derived from `$SLURM_JOB_NODELIST` so the rendezvous works across nodes. `scontrol show hostnames` expands the bracket-notation nodelist (`dgx-[01-04]`) into actual hostnames; we take the first one as the rendezvous master. Hardcoding a hostname would break the moment the scheduler picked different nodes.

**Three.** `NCCL_IB_DISABLE=0` is set explicitly. Without it, NCCL sometimes silently falls back from InfiniBand to TCP if it can't auto-detect the IB fabric, and your training is suddenly ten times slower because all-reduce traffic is going over the management ethernet instead of the high-speed IB. `NCCL_DEBUG=INFO` writes a startup log that tells you exactly which transport it picked — read that log on every first run to confirm it says "IB" not "Socket."

### 2.5 Common Slurm interview questions

**Q: "What does gang scheduling mean and why does it matter for ML training?"**

Gang scheduling is all-or-nothing scheduling. A four-node job runs only when four nodes are simultaneously free; it never gets two nodes now and two later. This matters for ML training because the workload is tightly coupled — every gradient step requires all-reduce across every rank, so partial allocation produces idle GPUs sitting and waiting for the rest. Slurm guarantees gang scheduling natively. Kubernetes by default does not — it's designed for independent web pods, where partial scheduling is fine. To get gang scheduling on Kubernetes you have to layer in Volcano or Kueue, which is why a lot of teams prefer to keep training on Slurm and inference on K8s, rather than fight that mismatch.

**Q: "How do you do hyperparameter sweeps on Slurm?"**

Two patterns. The first is job arrays — `#SBATCH --array=0-31` runs thirty-two jobs in parallel, each receiving a `$SLURM_ARRAY_TASK_ID` from 0 to 31, which you then map to a hyperparameter tuple via a YAML config or a Python preprocessing step. This is the bread-and-butter HPC pattern. The second is using Ray Tune (or Optuna, or W&B Sweeps) as the orchestrator and Slurm as the backend — Tune launches Slurm jobs via `submitit` or a similar bridge, and you get adaptive search algorithms like ASHA or BOHB instead of just grid search. I'd default to job arrays for simple sweeps and Ray Tune for serious HP search.

**Q: "Fair share — what is it and how do you tune it?"**

Slurm tracks how many resources each account has consumed historically, weighted by recency. The next job's priority is reduced if the account has over-consumed recently, increased if it's been under-consuming. Multi-team clusters use this to prevent one team from monopolizing a shared resource — the team that ran a thousand-GPU-hour sweep yesterday gets lower priority today. As a platform engineer you tune `PriorityWeightFairshare` in `slurm.conf` along with `PriorityDecayHalfLife` (how fast historical usage "decays" out of the calculation). Typical values: half-life of a week or two, fairshare weight as the dominant factor versus job size and age.

**Q: "Can you mix Slurm and Kubernetes on the same cluster?"**

Yes, and it's increasingly common. Three patterns: First, use Slurm for training and K8s for serving on disjoint node pools — the simplest and what I'd default to. Second, use the `slurm-operator` project, which lets you submit Slurm jobs from a Kubernetes CRD — useful if you want a single user-facing interface. Third, NVIDIA's Base Command Manager (BCM) integrates both natively on DGX BasePOD — that's the reference architecture NVIDIA ships, and if Avrioc bought a BasePOD, that's what they probably run.

**Q: "What's the difference between sbatch and srun?"**

`sbatch` submits a script to be run later, asynchronously — you get a job ID back immediately and your terminal is free. The script is queued, runs when resources free up, and writes its output to the configured log files. `srun` runs interactively or as a sub-step inside an sbatch script. Use sbatch for production pipelines, srun for interactive debugging or for launching the actual training step inside an sbatch script.

---

## 3. NVIDIA DGX — what to know

A **DGX** is NVIDIA's pre-engineered AI server — a single rack-mounted appliance that contains eight H100 (or A100, or B200) GPUs in one chassis, all interconnected by **NVLink/NVSwitch** for fast intra-node communication, with **InfiniBand** uplinks for fast inter-node communication. They come in three rough sizes:

| Form factor | What it is |
|------------|------------|
| **DGX H100/A100** | One server, 8 GPUs |
| **DGX BasePOD** | Reference architecture for ~4–32 DGX servers (small cluster) |
| **DGX SuperPOD** | Reference architecture for 32+ DGX servers, full HPC fabric, often >256 GPUs |

### 3.1 The two interconnect tiers (memorize this)

The single most important DGX concept for an AI engineer is the bandwidth hierarchy. The interview-friendly mental model:

```
     ┌───────────────────────────────────────────────────────────┐
     │                  ONE DGX H100 NODE (8 GPUs)               │
     │                                                            │
     │   GPU0 ─NVLink/NVSwitch─ GPU1 ─NVLink/NVSwitch─ ... GPU7   │
     │                                                            │
     │   Aggregate intra-node bandwidth: ~900 GB/s (H100)         │
     │                                  ~600 GB/s (A100)          │
     │   Use for: tensor-parallel within node                     │
     └───────────────────────────────────────────────────────────┘
                        │       InfiniBand HDR/NDR
                        │       200-400 Gb/s per port
                        ▼       multiple ports per node
     ┌───────────────────────────────────────────────────────────┐
     │                 ANOTHER DGX H100 NODE                     │
     │                       (same as above)                     │
     └───────────────────────────────────────────────────────────┘

         INTER-NODE BANDWIDTH: ~100-400 GB/s aggregate (much lower)
         Use for: data-parallel all-reduce (gradients only)
                  pipeline-parallel (activation handoff)
                  tensor-parallel only as last resort (lower-bw, slow)
```

### 3.2 The two-tier analogy

The way I explain this to engineers who haven't worked with DGX before: think of the GPUs in a single node as people working at the same table — they can hand papers back and forth in milliseconds. The GPUs across nodes are like people in different buildings — they have a fast network between them (InfiniBand is the fastest in the data-center world), but it's still much slower than handing papers across the table. So when you partition a model across GPUs, you put the most communication-heavy operation (tensor parallelism — which exchanges activations every layer) on the fast link (NVLink, within node) and the lightest communication (data parallelism — which only exchanges gradients once per training step) on the slower link (InfiniBand, between nodes). Pipeline parallelism sits in the middle: handing activations between pipeline stages happens at layer boundaries, not every layer, so it tolerates IB.

### 3.3 DGX-aware interview questions

**Q: "If you have 4 DGX H100 nodes (32 GPUs total) and want to train a 70B model, how do you partition?"**

Start with sizing. A 70B-parameter model in FP16 occupies 140 gigabytes of weights alone, plus optimizer states (Adam keeps two moments per parameter, so eight times the parameter count in bytes for FP32 Adam, which is roughly 1.1 terabytes), plus activations during forward pass. So you need to shard across multiple GPUs no matter what. The partitioning I'd choose: tensor-parallel of size eight within each node (using NVLink for the heavy traffic), and then either pipeline-parallel of four across the nodes, or data-parallel of four across the nodes with FSDP/ZeRO-3 sharding the optimizer states across the DP group. The tradeoff between PP and DP depends on the activation memory budget — if activations fit, DP+FSDP is simpler and gets you the highest hardware efficiency. If activations don't fit, pipeline-parallel chunks them into micro-batches and keeps memory bounded.

**Q: "Why do you care about NCCL_TOPO_FILE on a DGX?"**

NCCL is the NVIDIA Collective Communications Library — the thing that does all-reduce and all-gather for distributed training. NCCL plans its communication ring based on the network topology. The default auto-detection is okay but sometimes picks suboptimal paths, especially on heterogeneous setups. NVIDIA ships a topology file with each DGX system that pins the optimal ring; setting `NCCL_TOPO_FILE=/path/to/dgx_h100_topo.xml` ensures NCCL uses it. Forgetting this setting can cost twenty to thirty percent of training throughput, which on a 4-node H100 job translates to many thousands of dollars per training run. As a platform engineer this is the kind of detail you bake into the cluster's environment-modules so users get it for free.

**Q: "How does GPUDirect RDMA help, and what does it require?"**

GPUDirect RDMA lets a GPU on node A read or write directly to the memory of a GPU on node B without going through the host CPU's main memory. The traditional path was GPU-A → CPU-A memory → network adapter → CPU-B memory → GPU-B, with multiple memory copies. GPUDirect RDMA collapses that to GPU-A → network adapter → GPU-B, eliminating the host-side hops. On a DGX cluster with InfiniBand, GPUDirect RDMA halves all-reduce latency for cross-node communication, which is a big deal for tightly-coupled training. The requirements: the right NIC drivers (Mellanox OFED), the `nv_peer_mem` kernel module loaded, and the IB adapter and GPU on the same PCIe root complex (DGX systems are wired this way by design).

**Q: "What software stack ships with DGX?"**

DGX OS is NVIDIA's customized Ubuntu-based distribution, pre-loaded with CUDA, cuDNN, NCCL, NVIDIA drivers, Docker plus the NVIDIA Container Toolkit, and Base Command Manager (BCM) for cluster operations. NGC (NVIDIA's container catalog) ships pre-built ML images — PyTorch, TensorFlow, Triton, RAPIDS — that are tuned for DGX hardware and known to "just work." For someone who doesn't want to manage Python environments themselves, `srun --container-image=nvcr.io/nvidia/pytorch:24.04-py3 ...` (via Pyxis/Enroot) is a pragmatic answer.

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

You don't need to be an expert in every layer — you need to be able to point at each layer and say what it does and what the alternative would be.

### 4.1 Container runtime on Slurm: Pyxis + Enroot

Pyxis is NVIDIA's Slurm plugin for container support, and Enroot is the underlying lightweight container runtime. Together they let you `srun --container-image=docker://nvcr.io/nvidia/pytorch:24.04-py3 ...` and your job runs inside that container, with GPUs exposed via the NVIDIA Container Toolkit. The mental analogy: it's `docker run --gpus all` but integrated into Slurm's resource accounting and node allocation. You don't need a Kubernetes cluster sitting on the side just to run containers.

Why it matters: on a shared DGX cluster, you don't want every team's Python dependencies polluting the host operating system. Pyxis/Enroot give you per-job containers without the Kubernetes overhead. From a platform engineer's perspective, this is what makes a multi-tenant Slurm cluster usable for ML — without containers, you'd be stuck with conda environments and version conflicts.

### 4.2 Storage: parallel filesystems

DGX clusters need shared, high-throughput storage because:

- **Datasets are massive** — modern LLM pretraining datasets are 1 to 100 terabytes
- **Multiple nodes read the same shards in parallel** — at peak you might have 32 GPUs across 4 nodes all hitting the same filesystem
- **Checkpoints write from many ranks simultaneously** — a 70B-model checkpoint is hundreds of GB, written every N steps, often from every rank in parallel

Common choices and where each fits:

- **Lustre** — the workhorse of HPC. POSIX filesystem, scales to thousands of clients, hundreds of GB/s aggregate throughput. Open source. Operationally complex (multiple metadata servers, OSTs, careful tuning). Default choice for serious HPC sites.
- **WekaFS** — commercial parallel filesystem, popular at NVIDIA's reference architectures. Easier to operate than Lustre, often higher single-client throughput.
- **DDN ExaScaler** — DDN's enterprise Lustre offering, common at large NVIDIA SuperPOD deployments.
- **NFS over RDMA** — for smaller clusters or for the homedir filesystem (where users keep code and small files).
- **NVIDIA's BCM-bundled storage** — typically a turnkey answer for BasePOD customers.

As a candidate, mention them by name and say you understand the trade-offs without claiming deep expertise unless you have it. "I've used shared parallel storage at the team-tooling level — I understand the latency-versus-throughput trade-offs and why metadata-server design matters for many-small-file workloads, but I haven't operated Lustre myself."

---

## 5. Bridge questions: Slurm ↔ Kubernetes

These are the most likely "platform thinking" questions Avrioc will ask, because they probe whether you can think holistically about the training-to-serving pipeline rather than treating each piece as a silo.

**Q: "I have a fine-tune job that takes 3 days on 4 nodes. Walk me through where it runs and how the trained model gets to inference."**

The full pipeline:

1. **Train on Slurm** via `sbatch train.sh`. Gang scheduling on four DGX nodes, NCCL all-reduce over InfiniBand, FSDP for the optimizer state sharding. Wall time three days; checkpoints every thousand steps to the shared parallel filesystem.

2. **Final checkpoint saved** to the shared FS at the end of training. A separate post-training script copies the final weights to S3 (or MinIO, if on-prem) and registers the artifact in a model registry — MLflow, Weights & Biases, or a homegrown registry. This step matters because Slurm and K8s share the parallel FS but inference shouldn't read directly from it (operational coupling).

3. **CI pipeline** — GitHub Actions or Jenkins — picks up the new model registry entry, builds a vLLM container with that weight, and pushes the image to ECR (or whatever container registry).

4. **Argo CD or kubectl rollout** updates the K8s `Deployment` with the new image. Rolling update with readiness probes ensures old pods drain gracefully.

5. **Canary** — 1% of traffic to the new model, monitor for guardrail breaches (refusal rate, latency, output length), then 5%, 50%, 100%. Use Istio or a feature flag service for the traffic split.

This pipeline answer is gold because it shows you understand the full lifecycle, not just one piece.

**Q: "How do you autoscale K8s inference pods on GPU usage, not CPU?"**

The default Horizontal Pod Autoscaler scales on CPU and memory, which is wrong for GPU inference because the GPU is already pegged at full utilization under any non-trivial load. You need KEDA (Kubernetes Event-Driven Autoscaler) with a Prometheus scaler reading the metric that actually correlates with user-experienced latency — typically `vllm:num_requests_waiting` (queue depth) or queue time. Scaling on queue depth is a leading indicator of latency degradation, whereas GPU utilization is a lagging indicator (it saturates before the queue grows, so it doesn't tell you when to scale up). On the scale-down side, set a conservative cooldown (often 5 minutes) so you don't flap pods when traffic spikes briefly.

**Q: "Could you run the whole training stack on Kubernetes instead of Slurm?"**

Technically yes, with caveats. You'd use Kubeflow's TrainingOperator (PyTorchJob, TFJob), which knows how to launch multi-node DDP. For gang scheduling you'd add Volcano or Kueue, since stock K8s won't gang-schedule. For fair share across teams, you'd configure Kueue's ClusterQueue with weighted shares. For the InfiniBand fabric, you'd install the SR-IOV device plugin so pods can use IB adapters. By the time you've stitched all that together, you've essentially rebuilt Slurm in Kubernetes — and Slurm has thirty years of HPC experience baked in. So: technically possible, operationally usually not worth it. Use Slurm where it's good and Kubernetes where it's good.

---

## 6. NVIDIA-specific networking deep-dive

This section goes one level deeper for two reasons: first, the JD names DGX explicitly so this is the kind of question that distinguishes a "studied for the interview" candidate from one who just knows generic ML; second, the answers are short and easy to remember once you understand the structure.

### 6.1 NVLink and NVSwitch — the intra-node fabric

NVLink is NVIDIA's GPU-to-GPU interconnect. On an H100 each GPU has eighteen NVLink lanes, each lane running at 50 GB/s, for a total of 900 GB/s per GPU of bidirectional bandwidth to the other GPUs in the node. NVSwitch is the on-chassis switch fabric that connects all eight GPUs to each other in a fully-connected pattern — every GPU talks to every other GPU at full NVLink speed simultaneously. Compare that to PCIe Gen5, which is only 128 GB/s per direction — NVLink is roughly seven times faster.

The implication for ML: tensor-parallel operations (which exchange activations every transformer layer) are bottlenecked on bandwidth. Putting them on NVLink within a node makes them feasible; putting them on InfiniBand across nodes makes them slow.

### 6.2 InfiniBand HDR/NDR — the inter-node fabric

InfiniBand is the high-bandwidth, low-latency network used between DGX nodes. HDR is 200 Gb/s per port; NDR is 400 Gb/s per port (newer). DGX nodes typically have eight to ten IB adapters, so the aggregate is 1.6 to 4 Tb/s per node of inter-node bandwidth. Latency is sub-microsecond, way better than ethernet.

The two key features for ML:

- **RDMA (Remote Direct Memory Access)** — one node can read another's memory without involving the remote CPU. NCCL all-reduce uses this heavily.
- **GPUDirect RDMA** — extension of RDMA where the source/destination is GPU memory, bypassing the host CPU entirely. Halves all-reduce latency. Required modules: NVIDIA driver, Mellanox OFED, `nv_peer_mem`.

### 6.3 The bandwidth hierarchy summary

For a 4-node DGX H100 cluster:

| Tier | Where | Bandwidth (aggregate per GPU) | What goes here |
|------|-------|-------------------------------|-----------------|
| Tier 1 | NVLink/NVSwitch (within node) | ~900 GB/s | Tensor parallel, intra-node all-reduce |
| Tier 2 | InfiniBand NDR (between nodes) | ~50-400 GB/s aggregate per node | Data parallel gradient all-reduce, pipeline parallel activation handoff |
| Tier 3 | Ethernet management network | 10-100 Gb/s | Slurm control, log shipping, monitoring |

The art of distributed training is putting the right communication on the right tier.

---

## 7. Cheatsheet — sentences to drop in the interview

Practice saying these out loud. Each is a complete, fluent sentence you can drop into a system-design or "what would you do" answer.

- *"On a DGX node I keep tensor parallelism within the chassis to leverage NVLink, and use data parallelism across nodes over InfiniBand — that's the bandwidth-aware partitioning."*
- *"Slurm gives me gang scheduling and fair-share, which Kubernetes by default doesn't — I'd use Slurm for training and K8s for serving on shared hardware."*
- *"For multi-node DDP I'd derive `MASTER_ADDR` from `$SLURM_JOB_NODELIST`, set `NCCL_IB_DISABLE=0`, and launch one torchrun per node via srun."*
- *"NCCL_TOPO_FILE matters on DGX because NCCL plans its communication ring around the topology — getting it wrong costs 20-30% throughput."*
- *"Pyxis plus Enroot is the per-job container layer for Slurm — it gives me Kubernetes-style isolation without the K8s overhead."*
- *"For 70B fine-tuning on 4 DGX H100 nodes I'd use TP=8 within node and either PP=4 or DP=4-with-FSDP across nodes, depending on activation memory budget."*
- *"The trained checkpoint goes from the shared parallel FS to S3, then a CI pipeline builds a vLLM image, and Argo CD does the rolling deploy on K8s."*

---

## 8. How to say this in an interview

When the topic comes up — and it will, because the JD names both Slurm and DGX — say something like:

> "My honest position on the HPC stack is that I haven't operated a DGX cluster myself end-to-end, but I've thought hard about how I'd run one. The model I'd start from is Slurm-for-training and Kubernetes-for-serving sharing the same DGX hardware via partitions. Within Slurm, the canonical pattern is sbatch with one srun-launched torchrun per node, with NCCL configured to use InfiniBand and the topology file. The bandwidth hierarchy — NVLink within a node at 900 GB/s, InfiniBand between nodes at lower aggregate — drives every model-parallelism decision. For a 70B model on four H100 nodes I'd go tensor-parallel of eight within node and data-parallel of four across nodes with FSDP. I'd love to learn the operational reality of the cluster you have in Abu Dhabi — that's the kind of work I want to do."

That answer is honest (acknowledges what you haven't done), demonstrates depth (specific numbers, specific commands, specific design decisions), and ends with curiosity (which is what senior engineers want to hire).

---

Continue to **[Chapter 22 — 2-Day Intensive Study Plan](22_2day_study_plan.md)**.
