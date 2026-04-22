# Chapter 09 — Model Optimization
## Quantization, Pruning, Knowledge Distillation

> The JD explicitly calls out "Optimizing models using techniques such as quantization, pruning, and distillation." This chapter is the full technical spread — with opinionated production picks.

---

## 9.1 Why optimize?

A 70B FP16 model is 140 GB. Doesn't fit on one H100 (80 GB). Inference is memory-bandwidth-bound, so weight size directly caps throughput.

Optimization trades **capability** for **cost, latency, or fit**.

Three main techniques:

| Technique | Typical effect | Typical quality cost |
|-----------|---------------|---------------------|
| **Quantization** | 2-8× memory reduction | 0-3% quality |
| **Pruning** | 1.5-3× speedup if structured | 1-5% quality |
| **Distillation** | 2-10× size reduction | 3-15% quality |

Often combined: **distill → quantize → serve**.

---

## 9.2 Quantization — the landscape

### 9.2.1 The idea
Represent FP16/BF16 weights in fewer bits:

| Precision | Bits | Memory | Quality |
|-----------|------|--------|---------|
| FP32 | 32 | 4× | Baseline |
| FP16 / BF16 | 16 | 1× | ~baseline |
| FP8 | 8 | 0.5× | ~0 loss (with calib) |
| INT8 | 8 | 0.5× | <1% loss (with calib) |
| INT4 | 4 | 0.25× | 0-3% loss (sensitive) |
| INT3, INT2 | 3-2 | 0.1-0.2× | significant loss |

### 9.2.2 PTQ vs QAT

| | **Post-Training Quantization** | **Quantization-Aware Training** |
|--|-------------------------------|------------------------------|
| Process | Convert after training, small calibration set | Simulate quantization during training |
| Cost | Hours, no retraining | Days, full retraining |
| Quality | 1-5% loss | Near-lossless |
| When | LLMs (>7B) — too expensive to retrain | Small production models where accuracy is critical |

**Reality:** For LLMs in 2026, everything is PTQ. QAT is for small classifiers.

---

## 9.3 GPTQ — second-order aware PTQ

**Idea (Frantar & Alistarh, 2022):** quantize weights one layer at a time; at each layer, use the Hessian of a calibration-set reconstruction loss to decide which columns to quantize next, minimizing the error.

- 4-bit, group-wise (group_size=128 typical)
- **Calibration:** ~128 samples from target domain
- **Hardware:** GPU kernels via ExLlama, AutoGPTQ
- **Trade-off:** small quality drop, strong tooling

**Production pattern:** solid baseline. Slightly better perplexity than naive round-to-nearest.

---

## 9.4 AWQ — Activation-aware Weight Quantization

**Idea (Lin et al., 2023):** not all weights are equally important. Identify **salient weights** (those aligned with high-magnitude activation channels) and protect them via per-channel scaling before quantization.

- 4-bit, usually beats GPTQ on instruction-tuned models
- Calibration faster than GPTQ
- **Current production default for vLLM deployments**

```
Weights × activation_scale  → quantize → dequantize(× inv_scale) at inference
```

---

## 9.5 GGUF — llama.cpp's format

Not a quantization algorithm per se, but a **container format** optimized for CPU / Apple Silicon / small-GPU inference.

### K-quant variants
- **Q4_K_M** — 4-bit with per-block scales, ~best quality/size tradeoff ("the sweet spot")
- **Q5_K_M** — 5-bit, higher quality, bigger
- **Q8_0** — 8-bit baseline
- **Q2_K**, **Q3_K_S** — aggressive (use only if necessary)

Target: local/edge deployments (consumer laptops, Macs, Jetson).

---

## 9.6 bitsandbytes NF4 — for QLoRA

**NormalFloat-4 (NF4):** 4-bit datatype optimized for normally-distributed weights (which NN weights approximately are).

- Double quantization (quantize the quantization constants) saves ~0.4 bits/param
- Paged optimizers prevent OOM
- **Use case:** QLoRA fine-tuning specifically. Base model stays in NF4; LoRA adapters in BF16.

Not the best choice for pure inference (use AWQ / GPTQ); use NF4 when fine-tuning.

---

## 9.7 SmoothQuant — unlock W8A8

### The problem
Activation outliers in LLMs (especially FFN) make W8A8 (int8 weights AND int8 activations) crash — you get large dynamic range in activations that small-bit can't represent.

### The fix
Shift difficulty from activations to weights via per-channel scaling:

```
y = (x / s) · (W · s)
    ─────────  ─────────
    activation weights
    easier     harder
    (calmer)   (range absorbed here)
```

Now W8A8 becomes tractable. Unlocks TensorRT-LLM's fastest kernels (INT8 matmul tensor cores).

---

## 9.8 FP8 on Hopper / Blackwell — the 2025-2026 default

H100 and newer have **native FP8 tensor cores** with two formats:
- **E4M3** for forward pass (better precision)
- **E5M2** for gradients (better range)

### Why FP8 wins over INT8 for LLMs
- Exponent bits preserve dynamic range → tolerates outliers without calibration
- Native hardware support → same throughput as INT8
- Training AND inference can use it
- Minimal quality loss

### Adoption
- vLLM supports FP8 weights + activations
- TensorRT-LLM goes further with fused kernels
- DeepSeek-V3 trained in FP8

**Practical pick (H100+):** FP8 for weights AND KV-cache. Near-lossless, 2× memory savings.

---

## 9.9 KV-cache quantization

Not about weights — about the **KV-cache** during inference. As context grows, KV dominates memory.

| KV dtype | Quality | Memory | Throughput impact |
|----------|---------|--------|------------------|
| FP16 | baseline | 1× | baseline |
| **FP8** | ~0 loss | 2× | 2× concurrent requests |
| INT8 | <1% on short | 2× | 2× concurrent requests |
| INT4 | noticeable | 4× | 4× concurrent requests |

vLLM: `--kv-cache-dtype fp8`. This is often the **single biggest throughput lever** for long-context workloads.

---

## 9.10 Pruning — removing weights

### 9.10.1 Unstructured (magnitude pruning)
Zero out individual weights below a threshold. Produces **sparse matrices**. To get speedup, need:
- Specialized kernels (cuSPARSE)
- Or **2:4 structured sparsity** on Ampere/Hopper (2 zeros in every 4 values)

Without these, unstructured pruning saves memory but NOT latency.

### 9.10.2 Structured pruning
Remove **entire heads, channels, or layers**.
- Immediate latency win on commodity hardware
- Quality loss is bigger (coarser granularity)
- Usually followed by LoRA recovery fine-tuning

### 9.10.3 Wanda (2023)
Modern LLM pruning: score weights by `|W_ij| · ||X_j||` (weight magnitude × activation norm). No gradients, no retraining — single forward pass on calibration data. Matches or beats SparseGPT at fraction of the cost.

### 9.10.4 LLM-Pruner
Structured: removes coupled groups (attention heads, FFN neurons) using gradient-based importance; followed by LoRA recovery. Real speedup on standard hardware, ~5% quality loss at 50% sparsity.

### 9.10.5 SparseGPT
Layer-wise one-shot pruning using second-order information. Similar territory to GPTQ but for sparsity. Heavier compute than Wanda.

### Pruning decision

| Goal | Pick |
|------|------|
| Minimum memory (with sparse kernels) | Wanda (unstructured) |
| Latency on commodity hardware | LLM-Pruner (structured) + LoRA recovery |
| Combine with quantization | Prune first, then AWQ / GPTQ |

---

## 9.11 Knowledge Distillation

### 9.11.1 The DistilBERT recipe
Teacher: BERT-base (110M). Student: DistilBERT (66M).

Loss = α · L_soft + β · L_hard + γ · L_cosine

- **L_soft** — cross-entropy between student and teacher soft probabilities at temperature T=2
- **L_hard** — cross-entropy between student and ground-truth labels (MLM)
- **L_cosine** — cosine similarity between student and teacher hidden states

Student initialized from teacher (every other layer). Result: 97% of BERT performance at 60% size, 60% speedup.

### 9.11.2 Why temperature in distillation?
T in `softmax(z/T)` spreads the teacher's probability distribution, exposing more signal about which classes are "close" (dark knowledge). T=2-4 typical.

### 9.11.3 MiniLM — distill only the attention
Distill only the self-attention distributions of the last transformer layer. Simpler; works across architectures. Used in all-MiniLM-L6-v2 (a popular small embedding model).

### 9.11.4 Task-specific distillation
For a specific task (e.g., classification), distill the teacher's output logits into a smaller architecture. Can change architecture entirely (LSTM student from BERT teacher).

### 9.11.5 Model-level distillation for LLMs
- Generate teacher outputs (soft / hard) on a large corpus
- Train student from scratch or SFT on teacher outputs
- Used in: Alpaca (distill from GPT), Vicuna (from ShareGPT), Orca (distill with reasoning traces)

### 9.11.6 When distillation beats quantization
- You want to change architecture (shrink 70B to 7B)
- Teacher's *behavior* matters more than exact weights (instruction following)
- You have unlabeled domain data
- **Best production recipe: distill first, then AWQ-quantize the student**

---

## 9.12 Speculative decoding — optimization at inference, not model

Not weight optimization but latency optimization. A small "draft" model proposes K tokens; the target verifies them in one forward pass.

- 2-3× speedup
- Identical output distribution (no quality loss)
- Variants: Medusa (heads on target), EAGLE-2 (lookahead), self-speculative
- Key knob: draft acceptance rate (>60% → speedup; <40% → net loss)

---

## 9.13 Optimization pipeline — the production stack

```
FP16 model (70B)
      │
      │ 1. Distillation (optional)
      ▼
FP16 smaller model (e.g., 14B)
      │
      │ 2. Pruning (optional, structured + LoRA recovery)
      ▼
Pruned FP16 model
      │
      │ 3. Quantization: AWQ 4-bit weights
      ▼
AWQ-4 model
      │
      │ 4. KV-cache quantization: FP8
      ▼
Final served model
      │
      │ 5. Speculative decoding (runtime-only)
      ▼
2-3× further throughput
```

Each step ~2× memory or latency. Compounded: 10-20× improvement total.

---

## 9.14 Interview Q&A — Model Optimization

**Q1. PTQ vs QAT — when each?**
> PTQ: fast, no retraining, 1-5% quality loss. For LLMs (too expensive to retrain). QAT: simulates quantization during training, near-lossless, for small production models where every accuracy point matters.

**Q2. GPTQ vs AWQ vs GGUF — which do you ship?**
> AWQ for GPU serving (vLLM). GGUF Q4_K_M for CPU/edge. GPTQ only when AWQ kernels not available. AWQ usually beats GPTQ on instruction-tuned models.

**Q3. SmoothQuant — what does it solve?**
> Activation outliers make W8A8 crash. SmoothQuant per-channel-scales activations down and weights up — making both tractable for INT8. Unlocks TensorRT-LLM's fastest INT8 kernels.

**Q4. FP8 vs INT8 on H100 — pick?**
> FP8. Preserves dynamic range (exponent bits) → tolerates outliers without calibration. Native tensor cores match INT8 throughput. INT8 only on Ampere/L4 where FP8 isn't supported.

**Q5. [Gotcha] INT4-quantized 13B has 5% accuracy drop vs FP16. Debug?**
> (1) Per-layer sensitivity — quantize one layer at a time, find the sensitive ones (LM head, first/last blocks). (2) Calibration data — 128 domain samples minimum. (3) Switch GPTQ → AWQ. (4) group_size 128 → 64. (5) Mixed-precision — keep outlier layers FP16. 5% usually means over-quantizing attention projections.

**Q6. QLoRA NF4 — why "normal float"?**
> NF4 is info-theoretically optimal for weights that are normally distributed (which NN weights approximately are). Quantile-based levels match actual weight distribution better than uniform INT4. Combined with double quant + paged optimizers, it enables 65B QLoRA on a single 48GB GPU.

**Q7. Structured vs unstructured pruning — which for latency?**
> Structured (entire heads, channels, layers) gives immediate speedup on commodity hardware. Unstructured (magnitude pruning) produces sparse matrices needing specialized kernels (cuSPARSE, 2:4 sparsity on Ampere/Hopper). Default: structured + LoRA recovery for latency; Wanda for memory.

**Q8. What is Wanda?**
> Pruning by Weights and Activations (2023). Score = |W_ij| · ||X_j||. No gradients, single forward pass on calibration data. Matches/beats SparseGPT at a fraction of cost. Standard answer for modern LLM pruning.

**Q9. LLM-Pruner vs Wanda?**
> LLM-Pruner: structured pruning of coupled groups (heads, FFN neurons), gradient-based importance, LoRA recovery fine-tune. Wanda: unstructured, training-free. LLM-Pruner when you need real latency wins; Wanda for smallest memory footprint with sparse kernels.

**Q10. Explain DistilBERT's three losses.**
> Soft-target CE on teacher logits at T=2 (dark knowledge). Hard-target MLM. Cosine embedding loss between student and teacher hidden states. Student initialized from every other layer of teacher.

**Q11. When distillation over quantization?**
> Need architectural change (70B→7B for edge). Teacher's behavior matters more than exact weights. You have unlabeled domain data. Distillation preserves *how the model thinks*, not just weight values. Production stack often does both: distill first, then quantize.

**Q12. Temperature in distillation — why?**
> Softmax(z/T) with T>1 spreads the teacher's distribution, exposing "dark knowledge" — relationships between classes (e.g., "dog" and "wolf" are closer than "dog" and "table"). T=2-4 typical.

**Q13. MiniLM approach?**
> Distill only the self-attention distributions of the last transformer layer. Simpler than DistilBERT's triple loss; works across architectures. Used in all-MiniLM-L6-v2.

**Q14. [Gotcha] Your quantized model is the right size but inference is slower than FP16. Why?**
> Quantization library without optimized kernels on your hardware. Or computing dequant on every op. Fix: use AWQ/GPTQ with proper vLLM / TRT-LLM kernels. FP16 on A100 has great kernels; naive INT4 without proper kernels can be slower.

**Q15. Speculative decoding — what and why?**
> Small draft model proposes K tokens; large model verifies them in one forward pass. 2-3× speedup, identical output distribution. Works because verification is parallel but generation is sequential.

**Q16. KV-cache quantization — what's safe?**
> FP8 on H100: ~zero loss, 2× memory → 2× concurrent requests. Single biggest throughput lever for long-context workloads. INT8 has minor degradation; INT4 has noticeable. Enable via vLLM `--kv-cache-dtype fp8`.

**Q17. How do you validate a quantized model before deployment?**
> (1) Perplexity on held-out corpus vs FP16 baseline. (2) Task-specific eval on domain benchmarks. (3) Human / LLM-judge on realistic prompts. (4) Latency / throughput benchmarks. (5) Regression on MMLU, HellaSwag.

**Q18. Block size / group size in quantization?**
> The granularity of quantization constants. group_size=128 means one scale + zero_point per 128 weights. Smaller group = finer precision = slightly better quality but more metadata. Default 128 is a good tradeoff; 64 helps for sensitive models.

**Q19. 2:4 structured sparsity — what is it?**
> Nvidia Ampere and newer: every 4 consecutive weights have at most 2 non-zeros. Hardware-accelerated → real 2× speedup on SpMM kernels. Enabled via NVIDIA ASP (Automatic Sparsity).

**Q20. [Gotcha] Your SmoothQuant INT8 model has good per-sample accuracy but breaks on batch-1 inference. Why?**
> Likely activation statistics calibrated for batch size N don't hold at batch 1. Fix: calibrate with expected production batch sizes, or use per-token dynamic quantization of activations.

**Q21. Production quantization workflow?**
> (1) Pick base model. (2) Calibrate with 128-512 domain samples. (3) AWQ or GPTQ 4-bit weights. (4) FP8 KV-cache. (5) Evaluate vs FP16 baseline. (6) Deploy with vLLM. (7) A/B vs previous version in production, monitor quality drift.

**Q22. How do you handle mixed-precision (keeping certain layers in FP16 while quantizing others)?**
> Identified by per-layer sensitivity analysis (per-layer perplexity delta). Keep LM head + first/last blocks in FP16; quantize the rest. vLLM supports this via quant config overrides.

**Q23. Model merging vs distillation?**
> Merging combines fine-tuned checkpoints (TIES, DARE, SLERP) into one — zero training cost, preserves all params. Distillation creates a smaller student from a teacher — training cost, architecture change. Different tools for different goals.

**Q24. [Gotcha] LoRA adapters + quantized base — what changes in serving?**
> Base quantized (e.g., AWQ-4), adapters in BF16. vLLM supports this (LoRA + AWQ). Inference: dequantize weights on-the-fly + add BA. Small compute overhead; saves massive memory.

**Q25. When is it safe to go INT2/INT3?**
> Almost never for production. Research models only. Quality loss is catastrophic for reasoning / long-context tasks. 4-bit (AWQ/GPTQ) is the practical floor.

---

## 9.15 Resume tie-ins

- **"Machine Learning & LLMs: ... Quantization, Pruning, Distillation"** — have ONE concrete hands-on story. Example: "I quantized a DistilBERT classifier from FP32 to INT8 with ONNX Runtime for edge deployment; accuracy dropped 0.4% but inference latency on Lambda improved from 220 ms to 65 ms — well within our 500 ms p99 budget."
- Better: "I evaluated AWQ-4 Llama-3-8B vs BF16 on a RAG-answer eval — perplexity delta was 0.08, MMLU delta was -0.6 points, inference throughput on A10 went from 35 tok/s to 85 tok/s. We shipped AWQ."
- **"Multi-container SageMaker endpoints for cost efficiency"** — cost efficiency is what optimization is *for*. Tie the concepts together.

---

Continue to **[Chapter 10 — MLOps & LLMOps](10_mlops_llmops.md)**.
