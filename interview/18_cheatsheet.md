# Chapter 18 — Cheatsheet

> **Why this chapter exists:** This is the last thing you read on Thursday morning before the interview. Its job is **refresh, not teach**. Every section opens with one or two sentences that name the concept and remind you why it matters — so you can scan in 10 minutes and have the patterns warm in your head. If a section feels foreign now, don't try to learn it from this chapter; jump back to the deeper chapter and come back here later.

---

## 18.1 Self-attention — the formula and why each piece is there

Self-attention is the operation that lets every token attend to every other token in a sequence and decide how much each one matters for its own representation. The formula is the heart of every transformer, every LLM, every embedding model.

```
   Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

Why each piece:

- **Q · Kᵀ** — every query token compares against every key token; the result is a (T × T) matrix of compatibility scores.
- **÷ √d_k** — without scaling, dot products grow with d_k and softmax saturates, killing gradients. Scaling keeps variance ≈ 1.
- **softmax** — converts arbitrary scores into a probability distribution over tokens that sums to 1.
- **· V** — weighted sum of value vectors. The token gets a representation that is a blend of the values it attended to most.

**Multi-head attention**: do attention `h` times in parallel with different learned projections, each on a slice of dimension `d_k = d_model / h`. Concatenate. Project back to `d_model`.

**Causal masking** (decoder): prevent each position from attending to positions ahead by setting the upper triangle of the score matrix to `-∞` before softmax.

---

## 18.2 Transformer dimensions — the variables you'll see at every whiteboard

```
   B          batch size                    e.g. 32
   T          sequence length / context     e.g. 8192
   d_model    model hidden dimension        e.g. 4096 (Llama-7B), 8192 (70B)
   h          number of attention heads     e.g. 32, 64
   d_k        per-head dimension            d_model / h    e.g. 128
   d_ff       feedforward inner dim         typically 4 * d_model
   n_layers   number of transformer blocks  e.g. 32 (7B), 80 (70B)
   V          vocabulary size               e.g. 32K, 128K, 256K
```

Quick Q on a whiteboard: "What's the shape of the attention weight matrix for one head?" → (B, h, T, T). Per head it's (B, T, T).

---

## 18.3 Common LLM sizes — memory and GPUs needed

The mental model: parameters × 2 bytes (FP16) ≈ memory for weights alone. Add KV cache (~10-20% more for typical context) and optimizer states for training (~6-8x more for Adam).

```
   Model size       FP16 weights     INT8 weights    INT4 weights    GPUs (FP16 inference)
   ──────────────────────────────────────────────────────────────────────────────────────
   1B   (Phi, Llama-1B)     2 GB         1 GB          0.5 GB          1× consumer
   7B   (Llama-7B, Mistral) 14 GB        7 GB          3.5 GB          1× A100/H100 (80GB)
   13B  (Llama-13B)         26 GB        13 GB         6.5 GB          1× A100/H100
   70B  (Llama-70B)         140 GB       70 GB         35 GB           4-8× A100/H100
   405B (Llama-405B)        810 GB       405 GB        ~200 GB         16+× H100
```

For training, multiply by ~6-8x to cover Adam optimizer states (m, v, plus master FP32 weights with mixed precision).

---

## 18.4 Distributed training — when each strategy applies

Distributed training is about splitting work across multiple GPUs. There are three orthogonal axes you can split on, and most real systems combine them.

```
   ┌──────────────────────────────────────────────────────────────────┐
   │ DDP (Data Parallel)        Each GPU gets full model + diff data  │
   │                            All-reduce gradients each step        │
   │                            Best when: model fits on one GPU      │
   │                                                                  │
   │ FSDP (Fully Sharded DP)    Same as DDP but parameters, gradients │
   │                            and optimizer states are also sharded │
   │                            Best when: model too big for one GPU  │
   │                                                                  │
   │ TP (Tensor Parallel)       Split attention/FFN matrices across   │
   │                            GPUs. High-bandwidth NVLink required  │
   │                            Best when: keeping within node        │
   │                                                                  │
   │ PP (Pipeline Parallel)     Split layers across GPUs. Process     │
   │                            micro-batches in pipelined fashion    │
   │                            Best when: too many layers for TP     │
   └──────────────────────────────────────────────────────────────────┘
```

DeepSpeed ZeRO stages are FSDP variants:
- **ZeRO-1**: shard optimizer states only
- **ZeRO-2**: shard optimizer states + gradients
- **ZeRO-3**: shard optimizer states + gradients + parameters (= FSDP)

---

## 18.5 vLLM CLI flags — the ones you'll mention by name

vLLM is the JD-named LLM serving framework. Knowing the flags signals you've actually deployed it, not just heard of it.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8           # shard across 8 GPUs (NVLink intra-node)
  --quantization awq                 # AWQ INT4 weights, ~2x throughput
  --gpu-memory-utilization 0.9       # use 90% of GPU memory for weights+KV
  --max-model-len 8192               # cap context window — bounds KV size
  --enable-chunked-prefill           # smooth latency for chat workloads
  --max-num-batched-tokens 8192      # cap batch by token count
  --max-num-seqs 256                 # cap batch by sequence count
```

The two flags that win interviews if you mention them: `--enable-chunked-prefill` (interleaves prefill with decode for chat-friendly latency) and `--quantization awq` (or `gptq`, `fp8`, `bitsandbytes` — there are many; AWQ and GPTQ are the standard for INT4).

---

## 18.6 Kubernetes commands — the ones you'll use in the interview

```bash
kubectl get pods -n <ns>                    # list pods in namespace
kubectl describe pod <name>                 # see events, status, resource limits
kubectl logs -f <pod>                       # follow logs
kubectl logs <pod> --previous               # logs from previous container if crashed
kubectl exec -it <pod> -- /bin/bash         # shell into pod
kubectl get hpa                             # see autoscaler state (or KEDA scaledobjects)
kubectl rollout status deployment/<name>    # watch rolling deploy progress
kubectl rollout undo deployment/<name>      # roll back to previous version
kubectl top pods                            # CPU/memory per pod (needs metrics-server)
kubectl cp <pod>:/path/to/file ./file       # copy out files for debugging
```

For GPU workloads specifically:

```bash
kubectl describe node <gpu-node>            # see nvidia.com/gpu allocatable
kubectl get pods --field-selector=spec.nodeName=<node>
```

---

## 18.7 Slurm commands — the ones you'll use on a DGX cluster

```bash
sbatch train.sh                  # submit batch job, returns job ID
srun --pty --gres=gpu:1 bash    # interactive session, 1 GPU
squeue -u $USER                 # see your queued/running jobs
scancel <job_id>                # kill a job
sinfo                           # show partitions and node states
scontrol show job <job_id>      # detailed job info
sacct -j <job_id>               # accounting info (cpu hours, end status)
```

The interview-winning sentence: "Slurm gives me gang scheduling and fair-share, which Kubernetes by default doesn't — I'd use Slurm for training and K8s for serving on shared hardware."

---

## 18.8 Drift detection — the formulas and thresholds

Drift detection is monitoring whether your production data has shifted from your training baseline. Without it, you only learn about model degradation through accuracy drops, which is far too late.

**Population Stability Index (PSI):**
```
   PSI = Σᵢ (pᵢ - qᵢ) · ln(pᵢ / qᵢ)
```
- p = production proportion in bin i, q = baseline proportion
- Bin count: typically 10
- Add small epsilon to avoid log(0)
- Thresholds: < 0.1 stable, 0.1-0.25 minor drift, > 0.25 significant drift

**Kolmogorov-Smirnov test**: max difference between two empirical CDFs. Use `scipy.stats.ks_2samp`. Robust for continuous features, sensitive to scale-mismatched bins.

**Embedding drift**: cosine distance from production embedding to training-data centroid. Useful for monitoring text/image input distributions where features are not interpretable.

**The pipeline:**
```
   Production logger → Snowflake/S3 → drift script (daily)
                                  → Datadog/Prometheus metrics
                                  → Alert if metric > threshold
                                  → Trigger retraining if persistent
```

(Sachin's signature ResMed story.)

---

## 18.9 RAG evaluation metrics — what each measures

RAG fails in distinct ways and you need different metrics to catch each. The four RAGAS metrics decompose RAG failure modes cleanly.

| Metric | What it asks | Failure it catches |
|--------|--------------|---------------------|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Hallucination — model invented facts |
| **Answer Relevance** | Does the answer actually address the question? | Off-topic — model answered a different question |
| **Context Precision** | Do the retrieved chunks contain the answer? | Reranker / embedding model is bad |
| **Context Recall** | Is all relevant info in the retrieved chunks? | Top-k too small or hybrid search missing |

If faithfulness is low → fix prompting or guardrails. If precision low → reranker. If recall low → hybrid search or query rewriting.

---

## 18.10 AWS instance types for ML

The right instance shape depends on whether you're doing training, real-time inference, or batch inference, and on whether your model fits in CPU or needs GPU.

```
   GPU instances (training and large-model inference):
   g4dn.xlarge      1× T4 (16GB)     small inference, fast
   g5.xlarge        1× A10G (24GB)   small/mid inference
   p3.16xlarge      8× V100 (16GB)   older training
   p4d.24xlarge     8× A100 (40GB)   training + 70B inference
   p4de.24xlarge    8× A100 (80GB)   training + 70B inference (more KV)
   p5.48xlarge      8× H100 (80GB)   modern training + serving
   inf2.xlarge      1× Inferentia2   custom NPU, cheap inference

   CPU instances (small models, FastAPI servers):
   c6i.xlarge       4 vCPU, 8 GB     XGBoost/sklearn endpoints
   c6i.4xlarge      16 vCPU, 32 GB   bigger CPU inference
   r6i.xlarge       4 vCPU, 32 GB    memory-heavy workloads
```

Pricing rule of thumb: GPU instances are 5-50× CPU pricing. Use spot instances for training (50-70% discount, tolerable interruptions). Use on-demand for production inference.

---

## 18.11 Common ML formulas

These are the formulas you might be asked to write or interpret on a whiteboard. Each is preceded by a one-line reminder of when it's used.

**Cross-entropy loss** (classification, language modeling):
```
   L = -Σᵢ yᵢ · log(pᵢ)
```
where y is the one-hot true label and p is the predicted probability.

**Softmax** (turn logits into probabilities):
```
   softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Precision / Recall / F1** (binary classification eval):
```
   Precision = TP / (TP + FP)
   Recall    = TP / (TP + FN)
   F1        = 2·P·R / (P + R)
```

**ROC AUC** (probabilistic classifier ranking quality): area under (FPR, TPR) curve. AUC = 0.5 is random; AUC = 1.0 is perfect.

**KL Divergence** (distance between distributions, used in RLHF/DPO):
```
   KL(P || Q) = Σᵢ pᵢ · log(pᵢ / qᵢ)
```

**Cosine similarity** (between embeddings):
```
   cos(A, B) = (A · B) / (||A|| · ||B||)
```
For unit vectors, just `A · B`.

---

## 18.12 LLM serving latency budgets

Different LLM products have very different latency requirements. Knowing the rules of thumb means you can answer "what's a reasonable latency target for X?" without thinking.

```
   Chat product (customer support, coding assistant):
       TTFT < 500ms p95          (else feels broken)
       ITL < 50ms                (humans read at ~3 words/sec)
       Total: depends on output length, target < 5s for short answer

   Voice product (real-time conversation):
       TTFT < 300ms p95          (else conversation feels laggy)
       ITL < 30ms                (audio synthesis must keep up)

   Batch product (document summarization, embedding):
       TTFT: irrelevant
       Total throughput is king: tokens/sec/GPU

   API for downstream services (RAG retrieval step):
       Total p99 < 200ms         (else upstream chat TTFT breaks)
```

---

## 18.13 Common production gotchas — the senior signals

These are things juniors get wrong and seniors catch. Mentioning any of these unprompted in an interview signals depth.

1. **Loading the model inside the request handler.** Should be loaded once at startup; otherwise cold load on every request.
2. **Sync I/O library inside an async FastAPI handler.** Blocks the event loop, serializes all requests on that worker. Use `httpx.AsyncClient` with async, or just use sync handlers.
3. **Conflating /health and /ready in Kubernetes.** Kubernetes uses readiness to gate traffic and liveness to restart; collapsing them causes either premature traffic routing or unnecessary restarts.
4. **Autoscaling on CPU for GPU inference.** GPU is busy while CPU is idle; HPA doesn't fire. Use KEDA on queue depth or request rate instead.
5. **Stored embeddings without versioning.** Upgrading the embedding model invalidates every stored vector silently. Version them.
6. **Forgetting embedding model query/document prefixes.** E5 and BGE need `"query: "` and `"passage: "` prefixes; missing them silently degrades recall by 10-20%.
7. **PSI without epsilon for empty bins.** Log(0) = -infinity, breaks computation. Always add a small epsilon or use Laplace smoothing.
8. **Reading `request_id` after middleware overwrites it.** Make sure your structured logger pulls `request_id` from a single source — usually a contextvar set by middleware.
9. **Static batching for chat workloads.** Wastes GPU on the batch's slowest sequence. Use vLLM continuous batching.
10. **Lambda layers for ML deps with current size near 250MB limit.** Will hit limit on next upgrade. Use container images for any nontrivial ML.

---

## 18.14 The 60-second "Tell me about yourself" — for the morning of

Read this once before the call. Don't try to memorize it word-for-word; memorize the rhythm.

> *I'm Sachin, a senior ML engineer with eight years of experience productionizing ML and LLM systems across fintech, healthcare, and enterprise. At TrueBalance today I own a real-time XGBoost Lambda for loan-withdrawal prediction with p99 under five hundred milliseconds and three-environment VPC isolation, plus a Claude-powered developer platform that integrates Jira, GitHub, AWS Athena, and Jenkins for our ML team. Before that I spent two and a half years at ResMed building their IHS MLOps platform — eight models to production in six months, a RAG-based clinical chatbot, and the Datadog drift dashboard utility that became the team standard. My sweet spot is the bridge from research to production: LLMOps with vLLM and Kubernetes, real-time inference, model optimization, and the observability that keeps models healthy at scale. That bridge is exactly what Avrioc's JD describes, which is why this onsite Abu Dhabi role excites me.*

Should land at 55-65 seconds. Memorize the first ten and last ten words; improvise the middle.

---

## 18.15 The three closing questions — written on a sticky note

When the interviewer says "do you have any questions for us?", read these:

1. *"What does the AI infrastructure look like today — is it primarily Kubernetes plus Ray, or does Slurm own training while Kubernetes handles inference?"*
2. *"Of MyWhoosh, Comera, Labaiik, and Hyre, which one's AI roadmap is the team's biggest current focus?"*
3. *"What does success look like for this role over the first six months?"*

(Save salary, visa, vacation for the HR round.)

---

## 18.16 Read order on Thursday morning

10 minutes total. Strict order:

1. **§18.14** — your "tell me about yourself"
2. **§18.13** — the senior gotchas (so they're warm if a question lets you mention one)
3. **§18.5** — vLLM flags (most likely deep dive)
4. **§18.1, §18.2** — attention math and dimensions (in case they ask you to write attention)
5. **§18.15** — your three closing questions

Stop. Don't read more. Take a deep breath. Walk in.

---

End of pack. You're ready. Walk in calm.
