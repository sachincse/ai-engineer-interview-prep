# Chapter 09 — Model Optimization
## Quantization, Pruning, Knowledge Distillation, Speculative Decoding, FlashAttention

> The Avrioc JD explicitly calls out "Optimizing models using techniques such as quantization, pruning, and distillation." This chapter is your whiteboard companion — narrative-first, math-second, with worked examples you can verbalize without notes.

---

## 9.1 Why optimization exists — the cost story you must be able to tell

### 9.1.1 The problem in plain English

A 70B parameter model in FP16 weighs **140 GB**. Two bytes per weight, seventy billion weights — do the multiplication. That doesn't fit on a single H100 (80 GB), it doesn't fit on an A100 (80 GB), it doesn't even fit on two L40S cards stitched together cleanly. You need an 8×H100 box, which is $30-40/hour on AWS, just to load the weights — before you've served a single token.

Now consider what happens when a real product team comes asking. Their support chatbot needs to answer 500 queries per minute. p99 latency budget is 800 ms. They want to use the 70B model because the smaller ones hallucinate on their domain. The CFO sees the bill and panics. This is the conversation that drives every optimization decision in the field — and it's the conversation an interviewer is testing whether you've actually had.

### 9.1.2 The numbers you should have memorized

| Precision | Bits per weight | 70B model size | Notes |
|-----------|-----------------|----------------|-------|
| FP32 | 32 | 280 GB | Training, rarely used for inference |
| FP16 / BF16 | 16 | 140 GB | Standard "baseline" for serving |
| FP8 | 8 | 70 GB | H100/B200 native, near-lossless |
| INT8 | 8 | 70 GB | Calibrated, 0.5-1% loss |
| INT4 (AWQ/GPTQ) | 4 | 35 GB | 1-3% loss, sweet spot for self-host |
| INT3 | 3 | 26 GB | Research only |
| INT2 | 2 | 18 GB | Research only |

**The mental model:** every halving of bit width roughly halves memory and roughly doubles throughput on memory-bandwidth-bound hardware (which inference always is). That's why 4-bit weights are the default in production self-hosted serving — they let you cram a 70B model into a single 80 GB GPU with room left for KV cache.

### 9.1.3 Three techniques, three different levers

```
                ┌─────────────────────────────────────────┐
                │       MODEL OPTIMIZATION LEVERS         │
                └─────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
 ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
 │QUANTIZATION │          │   PRUNING   │          │DISTILLATION │
 │             │          │             │          │             │
 │ Fewer bits  │          │ Remove      │          │ Train smaller│
 │ per weight  │          │ weights or  │          │ student from │
 │             │          │ structure   │          │ teacher      │
 │ 2-8x smaller│          │ 1.5-3x faster│         │ 2-10x smaller│
 │ 0-3% qual   │          │ 1-5% qual   │          │ 3-15% qual   │
 └─────────────┘          └─────────────┘          └─────────────┘
        │                          │                          │
        └──────────────────────────┴──────────────────────────┘
                                   ▼
                        ┌──────────────────────┐
                        │ Often combined:      │
                        │ Distill -> Prune ->  │
                        │ Quantize             │
                        └──────────────────────┘
```

### 9.1.4 How to say this in an interview

> "When I think about model optimization, I think about three orthogonal levers — quantization changes the precision of each weight, pruning removes weights or structures entirely, and distillation trains a smaller architecture to mimic a larger one. They compose. The production recipe I've used most often is distill first to get the architecture you actually want, then quantize the student to four bits with AWQ, then on top of that quantize the KV cache to FP8. That stack typically gets me ten to twenty times the throughput of the FP16 teacher with under three percent quality drop."

---

## 9.2 Quantization — the foundational technique

### 9.2.1 Why this exists

Neural network weights are real numbers. Computers store real numbers as floats. A float-32 takes 4 bytes; a float-16 takes 2 bytes; an int-8 takes 1 byte; an int-4 takes half a byte. The whole game of quantization is "can I represent this matrix of real numbers with a smaller datatype without breaking the model?"

The intuition for why this works: neural network weights are massively over-parameterized. The model never needed 32 bits of precision to encode "this attention head pays attention to verbs." Most of those bits are noise. Throw them away and the model still works.

### 9.2.2 The mental model — analogy

Think of FP16 weights as a high-resolution photograph at 16-bit color depth. Quantization is reducing it to 8-bit color or even GIF's 256-color palette. For most images you can barely tell the difference because the human eye doesn't need that much precision. For a few images with subtle gradients (sunsets, skin tones) you'll see banding. Quantization research is the science of "for which weights does precision actually matter?"

### 9.2.3 The math you must be able to draw on a whiteboard

Symmetric INT8 quantization in one formula:

```
scale = max(|w|) / 127
q     = round(w / scale)              # int8 in [-127, 127]
w'    = q * scale                     # dequantized back to float
```

Asymmetric INT8 quantization (used when distribution isn't centered on zero):

```
scale = (max(w) - min(w)) / 255
zp    = round(-min(w) / scale)        # zero point
q     = round(w / scale + zp)         # uint8 in [0, 255]
w'    = (q - zp) * scale              # dequantized
```

**Symbol meanings:** `w` is the original FP weight, `q` is the quantized integer, `scale` maps integer steps to float steps, `zp` (zero-point) shifts the distribution so zero in float lands on a representable integer.

### 9.2.4 Worked example you can verbalize

Suppose a single weight tensor has values in the range `[-2.0, 3.0]`. Walk this through asymmetric INT8.

```
min(w) = -2.0
max(w) = 3.0
scale  = (3.0 - (-2.0)) / 255  = 5.0 / 255  = 0.0196
zp     = round(-(-2.0) / 0.0196)
       = round(102.04)         = 102
```

Now quantize a specific weight, say `w = 1.5`:

```
q = round(1.5 / 0.0196 + 102)
  = round(76.5 + 102)
  = round(178.5)
  = 179             # uint8
```

Dequantize back:

```
w' = (179 - 102) * 0.0196
   = 77 * 0.0196
   = 1.5092         # error of 0.0092
```

That tiny error per weight, multiplied across billions of weights, is the quality cost of quantization. The job of quantization research is to make sure the errors **cancel** rather than **compound**.

### 9.2.5 PTQ vs QAT — block diagram

```
   POST-TRAINING QUANTIZATION (PTQ)
   ════════════════════════════════
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │ Trained   │──▶│Calibration│──▶│ Compute   │──▶│ Quantized │
   │ FP16 model│   │ data      │   │ scales/zp │   │ INT8 model│
   │           │   │ (~128 ex) │   │ per layer │   │           │
   └───────────┘   └───────────┘   └───────────┘   └───────────┘
   Cost: hours. No retraining. 0.5-2% quality loss.

   QUANTIZATION-AWARE TRAINING (QAT)
   ═════════════════════════════════
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐
   │ Pre-train │──▶│Insert fake│──▶│ Fine-tune │──▶│ Quantized │
   │ FP16 model│   │ quant ops │   │ with q in │   │ INT8 model│
   │           │   │ (forward) │   │ the loop  │   │           │
   └───────────┘   └───────────┘   └───────────┘   └───────────┘
   Cost: days. Full retraining. Near-lossless.
```

**Reality of LLMs in 2026:** PTQ for everything. QAT is too expensive when models are 70B+; nobody retrains a Llama-70B for quantization. QAT remains relevant for tiny edge classifiers (mobile vision, on-device NER) where every accuracy point matters.

### 9.2.6 Common mistakes

1. **Calibrating with random or generic data.** Quantization scales are derived from the calibration set's activations. If your prod traffic is medical Q&A but you calibrated on Wikipedia, your scales are wrong and accuracy drops 5%. Always calibrate with 128-512 samples drawn from production-like prompts.

2. **Quantizing the LM head and embedding layers.** These are sensitivity hotspots. The LM head projects hidden states to vocabulary logits — a small error here distorts the entire generation. Standard practice: keep LM head and embedding in FP16, quantize the rest.

3. **Believing the size = the speed.** A naive INT4 model without optimized kernels can be **slower** than FP16 because dequantization happens on every forward pass. You need vLLM's AWQ kernels, TensorRT-LLM's INT8 kernels, or llama.cpp's GGUF kernels to actually cash in.

### 9.2.7 Interview Q&A — basics of quantization

**Q1. What does quantization actually do to a neural network?**
> Quantization replaces high-precision floats — usually FP16 or BF16 — with lower-precision integers, typically INT8 or INT4. You compute a scale factor and optionally a zero-point per layer or per group, then store the integer codes plus that small amount of metadata. At inference time you dequantize on the fly back to float to do matmuls, or you use specialized integer matmul kernels. The whole point is that neural network weights are massively over-parameterized — they don't need 16 bits of precision to encode useful patterns, so you throw away the noise and keep the signal.

**Q2. Walk me through PTQ versus QAT.**
> Post-training quantization happens after the model is fully trained. You take a calibration set of about 128 to 512 representative samples, push them through the model to observe the activation ranges, compute scales and zero-points per layer, and emit the quantized weights. It takes hours and gives you typically half a percent to two percent quality loss. Quantization-aware training inserts fake quantization operations during training so the model learns to be robust to the quantization noise. It's near-lossless but takes days of GPU time. For LLMs in 2026, everyone uses PTQ because retraining a 70B model just to quantize it isn't economically rational. QAT is still common for small mobile classifiers where you need every accuracy point.

**Q3. What's the difference between symmetric and asymmetric quantization?**
> Symmetric quantization centers the integer range on zero — so INT8 maps to negative-127 through positive-127, and you only need a scale factor. It works well when the weight distribution is roughly centered on zero, which is usually the case for transformer weights after training. Asymmetric quantization adds a zero-point offset, mapping the float range to the full unsigned integer range zero through 255. It's preferred for activations after ReLU, which are non-negative, because symmetric quantization would waste half the integer range. In practice, weights go symmetric, post-ReLU activations go asymmetric.

**Q4. What's a calibration set and why does it matter?**
> A calibration set is a small batch of representative inputs — typically 128 to 512 samples — that you push through the model to observe what range the activations actually take in production-like usage. From those observations you compute the scale factors. If you calibrate with the wrong distribution, your scales clip too aggressively or waste range, and accuracy collapses. The classic mistake is calibrating a domain-specific model with generic Wikipedia text. I always pull calibration data from real production logs — sanitized for PII — because the activation statistics on real traffic are what matters at serving time.

---

## 9.3 GPTQ — second-order aware quantization

### 9.3.1 Why GPTQ exists

Naive round-to-nearest quantization treats every weight independently, which is suboptimal. The Frantar & Alistarh 2022 paper realized that the **error from quantizing one weight propagates** through the layer — so you should quantize weights one at a time and use what you learn from each rounding decision to adjust the unquantized weights, minimizing the total reconstruction error of the layer's output.

### 9.3.2 The mental model

Imagine you're packing a suitcase and replacing each item with a slightly-different-size copy. Naive quantization picks the closest copy for each item independently. GPTQ packs items in order, and after each placement looks at the leftover space and re-chooses smaller items to compensate. The tool it uses to compute "how does this rounding decision affect the rest" is the **Hessian matrix** of the reconstruction loss.

### 9.3.3 The pipeline

```
┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ Layer L weights│──▶│Compute Hessian │──▶│Order columns by│
│   (FP16)       │   │of L2 recon loss│   │sensitivity     │
└────────────────┘   └────────────────┘   └────────┬───────┘
                                                    ▼
                          ┌─────────────────────────────────┐
                          │ For each column in order:       │
                          │   1. Quantize column to INT4    │
                          │   2. Compute residual error     │
                          │   3. Update remaining FP cols   │
                          │      to compensate via Hessian  │
                          └─────────────┬───────────────────┘
                                        ▼
                          ┌─────────────────────────────────┐
                          │ Output: INT4 weights + scales   │
                          │  (group_size 128 typical)       │
                          └─────────────────────────────────┘
```

### 9.3.4 In practice

- 4-bit weights, group size 128 by default
- Calibration: 128 samples is enough
- Tooling: AutoGPTQ, ExLlamaV2 kernels
- Quality: typically 0.3-1.0 perplexity points worse than FP16

---

## 9.4 AWQ — activation-aware weight quantization

### 9.4.1 Why AWQ exists

The Lin et al. 2023 paper observed something important: **not all weights matter equally**. About 1% of weights — those whose corresponding activation channels carry large magnitudes — are responsible for most of the model's behavior. Quantize those at INT4 and you crush the model. Protect them and you can crush everything else into INT4 with almost no damage.

### 9.4.2 The mental model

Think of a transformer's weights as a corporation. Most employees do generic work; a small number of "salient" employees do critical work that the whole company depends on. AWQ identifies those salient employees by looking at how loud their inputs are (activation magnitudes), then scales them up before quantization so they land in the high-precision part of the integer range.

### 9.4.3 The trick — per-channel scaling

Mathematically, you can multiply weights by a scale `s` and divide activations by `s` and the layer output is unchanged:

```
y = x · W = (x/s) · (W·s)
```

AWQ chooses `s` per channel so that salient weights get bigger (and so quantize better) while activations stay tractable. The whole point: the multiplication is mathematically identity, but the quantization error becomes much smaller.

### 9.4.4 GPTQ vs AWQ — practical comparison

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Approach | Hessian-based optimal rounding | Per-channel salience scaling |
| Calibration time | Slow (Hessian inversions) | Fast (forward passes only) |
| Instruction-tuned models | OK | **Better** |
| vLLM kernel quality | Good | **Excellent** |
| Default in 2026 | When AWQ kernels unavailable | **Production default** |

### 9.4.5 How to say this in an interview

> "When someone asks me to ship a quantized 70B model on a single H100, my default is AWQ four-bit. The reason is that AWQ identifies the small fraction of weights that actually carry the model's behavior — the salient ones — and protects them by per-channel rescaling. GPTQ is also good and uses Hessian information for optimal rounding, but on instruction-tuned models AWQ tends to score better and the vLLM kernels for AWQ are battle-tested. I'd reach for GPTQ only if AWQ kernels weren't available for my hardware."

---

## 9.5 GGUF — the format, not an algorithm

GGUF is the file format used by llama.cpp's ecosystem. It's not a quantization algorithm itself; it's a container that stores quantized weights with metadata so the same file runs on a Mac, an Intel laptop, and a Jetson Orin.

### 9.5.1 The K-quant family

| Variant | Bits/weight | Quality | When to use |
|---------|-------------|---------|-------------|
| Q2_K | ~2.6 | Bad, research only | Last resort on tiny VRAM |
| Q3_K_S | ~3.4 | Noticeable degradation | Aggressive compression |
| Q4_K_M | ~4.8 | Sweet spot | **Default for local/edge** |
| Q5_K_M | ~5.7 | Better than Q4 | When you have headroom |
| Q6_K | ~6.6 | Near-lossless | Almost always overkill |
| Q8_0 | 8.0 | Lossless effectively | Reference quality |

**The "K" in K-quants** refers to the per-block structure — small blocks (typically 32 or 64 weights) get their own scale, plus a coarser super-block scale on top. This two-level structure preserves quality cheaply.

### 9.5.2 Resume tie-in

> "When I prototype models on my MacBook Pro M3, I pull the Q4_K_M GGUF from HuggingFace and run it via llama.cpp — it gives me about 30-40 tokens per second on Llama-3-8B for free, and the quality is indistinguishable from FP16 for my testing purposes. For real production serving on H100s, I switch to AWQ via vLLM."

---

## 9.6 SmoothQuant — unlocking W8A8

### 9.6.1 The problem it solves

LLMs have **activation outliers** — a small number of channels in the FFN produce activations 100× larger than the median. These outliers crash naive INT8 because you can't represent both the typical scale and the outlier scale in 8 bits. So most LLMs only quantize **weights** to INT8, leaving activations in FP16. That gives you W8A16, which is fine, but the matmul is still FP16 — no INT8 tensor core speedup.

### 9.6.2 The trick

```
   Activation X has outliers (hard to quantize)
   Weight    W is well-behaved (easy to quantize)

   Apply per-channel scale s:
   Y = X · W = (X/s) · (s·W)
              ────────  ────────
              "calmed"  "absorbs"
              activation outlier
                        range
```

Mathematically identity. Practically, both X' = X/s and W' = s·W are now within INT8 range, so you can do **W8A8** matmul on TensorRT-LLM's INT8 tensor cores — roughly 2× the FP16 throughput.

### 9.6.3 The block diagram

```
   ┌────────────┐                                ┌────────────┐
   │ FP16 input │                                │FP16 output │
   │     X      │                                │     Y      │
   └─────┬──────┘                                └─────▲──────┘
         │                                             │
         │  divide by s                                │ scale
         ▼                                             │ back
   ┌────────────┐    ┌──────────────────┐    ┌────────┴─────┐
   │ X/s in     │───▶│ INT8 matmul on   │───▶│ INT32 accum  │
   │ INT8 range │    │ tensor core      │    │              │
   └────────────┘    │ X'·W' = X·W      │    └──────────────┘
                     └──────────────────┘            ▲
                          ▲                          │
                          │                          │
                     ┌────┴──────┐                   │
                     │ s·W in    │───────────────────┘
                     │ INT8 range│
                     └───────────┘
```

---

## 9.7 FP8 — the 2025-2026 default on Hopper and Blackwell

### 9.7.1 Why FP8 wins over INT8 for LLMs

H100 and B200 have native FP8 tensor cores, with two formats:

- **E4M3** — 4 exponent bits, 3 mantissa bits. Better precision, used for forward pass weights/activations.
- **E5M2** — 5 exponent bits, 2 mantissa bits. Better range, used for gradients during training.

Why FP8 is preferred over INT8:

1. **Exponent bits preserve dynamic range.** That nasty activation-outlier problem that plagues INT8? FP8 handles it natively because exponents scale to the data.
2. **No calibration needed.** INT8 requires a calibration set; FP8 quantization can use simple scaling factors that work out of the box.
3. **Native hardware throughput.** H100's FP8 tensor cores match INT8 throughput, so there's no speed penalty.
4. **Same recipe trains and serves.** DeepSeek-V3 was trained natively in FP8.

### 9.7.2 Numbers worth remembering

```
   Format    Bits   Range         Precision   Use
   ──────    ────   ─────         ─────────   ───
   FP32      32     ±3.4×10^38   ~7 dec       Training (legacy)
   BF16      16     ±3.4×10^38   ~3 dec       Training (current)
   FP16      16     ±6.5×10^4    ~3 dec       Training/inference
   FP8 E4M3  8      ±448         ~2 dec       Inference forward
   FP8 E5M2  8      ±5.7×10^4    ~1 dec       Gradients
   INT8      8      ±127         exact int    PTQ inference
```

---

## 9.8 KV-cache quantization — the throughput multiplier

This isn't about weights — it's about the activation cache stored during autoregressive generation. As context grows long, the KV cache **dominates GPU memory**. A 70B model on H100 might have weights at 35 GB (AWQ-4) but a 32k-token KV cache eating 30+ GB.

| KV dtype | Quality impact | Memory savings | Concurrent requests |
|----------|----------------|----------------|---------------------|
| FP16 | baseline | 1× | baseline |
| **FP8** | ~zero | 2× | **2×** |
| INT8 | <1% on short, more on long | 2× | 2× |
| INT4 | 1-3%, more on reasoning | 4× | 4× |

In vLLM, this is one flag: `--kv-cache-dtype fp8`. It's often the single biggest throughput lever for chatbot workloads where context is long.

---

## 9.9 Pruning — removing weights

### 9.9.1 Why pruning exists

If quantization is "use fewer bits per weight," pruning is "have fewer weights." Networks are over-parameterized; many weights are near-zero and contribute almost nothing. Set them to exactly zero and the model still works — and now your matrix is sparse, which (with the right kernels) means less compute and less memory.

### 9.9.2 Magnitude pruning — the simplest method

```
For each weight w in layer L:
    if |w| < threshold:
        w = 0
```

That's it. Pick a threshold so that, say, 30% of weights become zero. Result: a sparse matrix. Quality drop typically 1-3% for moderate sparsity.

### 9.9.3 The sparsity-vs-speedup gap

Here's the catch every junior engineer trips over: **unstructured sparsity rarely gives you a speedup**. GPU matmul kernels are dense. To turn sparsity into speed you need either:

- **Specialized sparse kernels** (cuSPARSE, BSR matmul) — often slower than dense for moderate sparsity
- **2:4 structured sparsity** — Ampere/Hopper hardware feature: in every group of 4 consecutive weights, exactly 2 must be zero. NVIDIA's tensor cores hardware-accelerate this for a real 2× speedup.

```
   Unstructured pruning (50% sparse):
   [0.3, 0, 0, 0.5, 0, 0.2, 0, 0.7]  ← random zero pattern
   Memory: 50% saved
   Speed: 0% saved on standard kernels

   2:4 structured pruning:
   [0.3, 0, 0.5, 0]  [0.2, 0.7, 0, 0]  ← exactly 2 zeros per 4
   Memory: 50% saved
   Speed: 2x via Ampere/Hopper sparse tensor cores
```

### 9.9.4 Structured pruning — the latency lever

Remove **entire heads, channels, or layers**. Coarser granularity means bigger quality hit but immediate latency win on commodity hardware.

```
   Original transformer layer:
   ┌────────────────────────────────────────┐
   │ Heads: H1 H2 H3 H4 H5 H6 H7 H8 H9 ... │
   └────────────────────────────────────────┘

   After structured pruning (remove unimportant heads):
   ┌────────────────────────────────────────┐
   │ Heads: H1    H3    H5 H6    H8         │
   └────────────────────────────────────────┘
   ↓ followed by ↓
   LoRA recovery fine-tuning to recoup 1-2% quality
```

### 9.9.5 Wanda (2023) — the modern LLM pruning recipe

Wanda scores each weight by `|W_ij| × ||X_j||` — weight magnitude times the L2 norm of the corresponding activation. No gradients, no retraining, single forward pass on calibration data. It matches or beats SparseGPT at a fraction of the compute cost. Ask any senior MLE about LLM pruning in 2026 and they'll say "Wanda."

### 9.9.6 Lottery ticket hypothesis (brief)

Frankle & Carbin 2018 showed that within a randomly initialized dense network, there exists a sparse sub-network ("the winning ticket") that, if you trained it from scratch with the same initialization, would reach the dense network's accuracy. It's mostly an academic result but informs the intuition that "most weights aren't necessary."

### 9.9.7 Common mistakes

1. **Pruning unstructured and expecting speedup on commodity GPUs.** You'll get memory savings only.
2. **Pruning too aggressively without recovery fine-tuning.** Always budget LoRA recovery after structured pruning.
3. **Pruning then quantizing in the wrong order.** Usually prune first (so quantization doesn't fight against zeros), then quantize.

### 9.9.8 Interview Q&A — pruning

**Q1. Magnitude pruning — what's the catch on speed?**
> Magnitude pruning zeros out individual weights below a threshold and gives you a sparse matrix. The catch is that GPU matmul kernels are dense by default — they multiply zeros and nonzeros indiscriminately. To turn sparsity into actual latency improvement you need specialized sparse kernels, which often underperform dense for moderate sparsity, or you need 2:4 structured sparsity on Ampere or Hopper, which the hardware accelerates natively for a real 2x speedup. The right answer to "I pruned my model 50% but it's not faster" is "yeah, you need 2:4 sparsity or you need to switch to structured pruning."

**Q2. When do you use structured vs unstructured pruning?**
> Structured pruning removes whole heads, channels, or layers — it gives you immediate latency wins on stock GPUs because the resulting matrices are smaller dense matrices. The cost is a bigger quality hit because the granularity is coarse, so you typically follow it with a short LoRA recovery fine-tune. Unstructured magnitude pruning gives you memory savings and works well in research, but for production latency on commodity hardware you almost always pick structured. The exception is if you have access to 2:4 sparsity hardware features, in which case unstructured at exactly 50% sparsity becomes very attractive.

**Q3. Walk me through Wanda.**
> Wanda is the standard modern recipe for pruning large language models. Instead of using gradients or retraining, it scores each weight by the product of its absolute value and the L2 norm of its corresponding input activation. The intuition is that a weight matters if it's both large and seeing large activations. You compute these scores on a small calibration set in a single forward pass, then prune the lowest-scoring weights. It matches or beats SparseGPT, which uses second-order Hessian information, while costing a fraction of the compute. For LLMs in 2026 it's the default.

---

## 9.10 Knowledge Distillation — the architecture-changing lever

### 9.10.1 Why distillation exists

Quantization and pruning shrink an existing model. Distillation lets you **change the architecture entirely** — train a small model from scratch (or close to it) to mimic a large model's behavior. You can go from a 70B teacher to a 7B student, from a transformer teacher to an LSTM student, from a closed-source GPT-4 teacher to an open-source Llama student.

### 9.10.2 The mental model — analogy

A grandmaster chess teacher doesn't just tell you "this move is correct, this move is wrong." She tells you "this move is best, but those two are also reasonable, and these are bad for these reasons." The richness of her **soft probabilities** over the move space is what makes you a strong player — much more than just labeled win/loss data. Distillation captures the same thing: the teacher's soft outputs encode "dark knowledge" about how concepts relate, which the student learns from much faster than from hard labels alone.

### 9.10.3 The loss function — the three-component DistilBERT recipe

```
L_total = α · L_soft + β · L_hard + γ · L_cosine

where:
  L_soft   = KL_divergence( student_logits/T , teacher_logits/T ) · T²
  L_hard   = cross_entropy( student_logits , ground_truth )
  L_cosine = 1 - cos( student_hidden , teacher_hidden )
```

**Symbol meanings:**
- `T` is temperature, typically 2-4. Higher T softens the distribution and exposes "dark knowledge" about which classes are close.
- `α, β, γ` are weights, typically `α=5, β=2, γ=1` in DistilBERT.
- The `T²` factor compensates for the gradient scaling when temperature is non-1.

### 9.10.4 The block diagram

```
                  ┌────────────────────────┐
                  │  Training data batch    │
                  └──────────┬─────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
     ┌─────────────────┐            ┌─────────────────┐
     │   TEACHER       │            │    STUDENT      │
     │ (BERT-base 110M)│            │ (DistilBERT 66M)│
     │  frozen         │            │   trained       │
     └────────┬────────┘            └────────┬────────┘
              │ logits, hidden states         │ logits, hidden states
              ▼                                ▼
     ┌─────────────────────────────────────────────┐
     │  L_soft   = KL(softmax(s/T), softmax(t/T))  │
     │  L_hard   = CE(softmax(s), y_true)          │
     │  L_cosine = 1 - cos(h_s, h_t)               │
     └──────────────────┬──────────────────────────┘
                        ▼
               ┌────────────────┐
               │ Backprop into  │
               │ student only   │
               └────────────────┘
```

### 9.10.5 Worked example — DistilBERT numbers

| Metric | BERT-base (teacher) | DistilBERT (student) |
|--------|---------------------|----------------------|
| Parameters | 110 M | 66 M (60%) |
| GLUE average | 79.5 | 77.0 (97% of teacher) |
| Inference speed | 1× | 1.6× |
| Initialization | random | every other teacher layer |

### 9.10.6 Distillation for LLMs — the modern story

For LLMs, full-loss distillation is rarely practical because logits over a 128k vocab are too large. Modern recipes:

- **Synthetic-data distillation** — generate teacher outputs on a large corpus, then SFT the student on `(prompt, teacher_response)` pairs. Used in Alpaca, Vicuna, Phi-3.
- **Reasoning-trace distillation** — teacher emits chain-of-thought, student learns to reproduce reasoning. Used in Orca, DeepSeek-R1 distillations.
- **Behavior cloning + DPO** — student trained on pairs `(teacher_preferred, teacher_dispreferred)` to match the teacher's preferences.

### 9.10.7 Common mistakes

1. **Skipping initialization.** A randomly-initialized 66M student learns much slower than one initialized from every other teacher layer. DistilBERT only works because of this trick.
2. **Wrong temperature.** T=1 ignores dark knowledge; T=10 makes the distribution too flat. T=2-4 is the sweet spot.
3. **Distilling on the wrong data distribution.** The student matches the teacher only where you train them on; if your distillation corpus doesn't cover production traffic, the student fails out of distribution.

### 9.10.8 Interview Q&A — distillation

**Q1. Why does temperature matter in distillation?**
> Temperature controls how peaked or flat the teacher's softmax distribution is. With T equal to one you get the standard softmax, where the teacher might assign 99% probability to the correct class — that's almost the same signal as a hard label. With T equal to 2 or 3, the distribution spreads out and you can see that the teacher thinks "dog" is most likely but "wolf" and "fox" are also plausible while "table" is essentially impossible. That richer probability distribution is what's called dark knowledge — it teaches the student about the conceptual structure of the label space, not just the correct answer. Empirically T equals 2 to 4 is the sweet spot.

**Q2. Walk me through DistilBERT's three losses.**
> DistilBERT trains the student against three objectives jointly. The soft-target loss is the KL divergence between student and teacher logits, both passed through softmax at temperature 2 — this is where the dark knowledge transfer happens. The hard-target loss is the standard masked language modeling cross-entropy against the ground-truth tokens — this anchors the student to the actual task. The cosine embedding loss matches the student's hidden states to the teacher's hidden states layer by layer — this enforces representational alignment. The student is initialized from every other layer of the teacher, which is essential — random init makes the recipe much slower to converge.

**Q3. When would you choose distillation over quantization?**
> Distillation when I need to change the architecture — for example, I want a 7B student from a 70B teacher because 7B fits on the consumer GPUs my customer has. Quantization preserves the architecture; distillation can shrink dimensionality, reduce layer count, even change the family of model. I also reach for distillation when the teacher's *behavior* matters more than its exact weights — instruction-following, chain-of-thought reasoning, brand voice — because distillation transfers behavior, not bits. The production stack often does both: distill first to get the architecture I want, then AWQ-quantize the student for serving.

**Q4. What's MiniLM doing differently?**
> MiniLM simplifies distillation down to a single objective: match the self-attention probability distributions of the last transformer layer. No need to align hidden state dimensions, no triple loss — just attention transfer. Because attention distributions are architecture-agnostic, you can distill across layer counts and even slightly different architectures. The famous all-MiniLM-L6-v2 sentence-transformer was distilled this way. It's elegant and works well; the trade-off is you lose the explicit hidden-state alignment that DistilBERT gets from the cosine loss.

---

## 9.11 Speculative decoding — runtime-only latency optimization

### 9.11.1 Why this exists

Autoregressive generation is sequential — to produce token 100 you must have produced token 99. That sequential dependency is the latency killer. But **verifying** a candidate sequence is parallel: the LLM can score 8 candidate tokens in one forward pass for almost the same cost as scoring 1.

Speculative decoding exploits this asymmetry. A small **draft model** proposes the next K tokens (cheaply, autoregressively); the large **target model** verifies them all in a single parallel forward pass. If the draft was right, you got K tokens for the price of 1 target call. If wrong, you fall back to standard generation from the first divergence.

### 9.11.2 The block diagram

```
   STEP 1 — Draft model proposes K=4 tokens
   ─────────────────────────────────────────
   ┌───────────────┐
   │  Draft model  │ "the cat sat on" → "the mat at home"
   │  (1B params)  │                    └────────┬───────┘
   └───────────────┘                             │ proposed
                                                 ▼
   STEP 2 — Target model verifies in parallel
   ─────────────────────────────────────────
   ┌───────────────┐  Score "the mat at home" in ONE pass
   │ Target model  │  ┌─────┬─────┬─────┬─────┐
   │ (70B params)  │─▶│ ✓   │ ✓   │ ✗   │  -  │  reject from "at"
   └───────────────┘  └─────┴─────┴─────┴─────┘

   STEP 3 — Accept prefix, sample next from target
   ───────────────────────────────────────────────
   Output: "the mat" + new token from target distribution
```

### 9.11.3 The math intuition

The trick that makes speculative decoding **lossless** — same output distribution as the target — is rejection sampling. If the draft proposes token x with probability `q(x)` and the target's true probability is `p(x)`:

- Accept with probability `min(1, p(x) / q(x))`
- On rejection, sample from the residual distribution `max(0, p(x) - q(x))`

The math guarantees the final samples are distributed exactly as if you'd sampled from `p` directly. **No quality loss**, just speed.

### 9.11.4 Numbers from production

- Typical speedup: 2-3×
- Acceptance rate: 60-80% on similar-quality drafter
- Variants: Medusa (extra prediction heads on the target), EAGLE-2 (lookahead trees), self-speculative (target uses early layers as drafter)

### 9.11.5 Common mistakes

1. **Draft too weak.** Acceptance rate below 40% and the overhead of running the drafter exceeds the savings. Net loss.
2. **Draft too strong.** A 13B drafter for a 70B target costs nearly as much as the target itself. Net wash.
3. **Sampling temperature mismatch.** If draft and target sample at different temperatures, acceptance rates collapse.

---

## 9.12 FlashAttention recap — the kernel everyone uses

FlashAttention deserves its own callout because it's the optimization that **everything else depends on**. Detailed treatment is in Chapter 02; here's the optimization-stack-relevant summary.

### 9.12.1 The problem

Standard attention reads and writes the `N × N` attention matrix to GPU HBM (high-bandwidth memory) — which is large but slow compared to on-chip SRAM. For long sequences, the entire forward and backward pass is bottlenecked by HBM bandwidth, not compute.

### 9.12.2 The trick — tiling and recomputation

```
   Standard attention (HBM-bound):
   ┌─────────────────────────────────┐
   │  Materialize full N×N attn      │ ← writes O(N²) to HBM
   │  matrix in HBM                  │
   └─────────────────────────────────┘

   FlashAttention (SRAM-bound):
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ Tile of Q,K│─▶│Compute attn│─▶│Stream out  │
   │ in SRAM    │  │ block-wise │  │softmax acc │
   │ (~64×64)   │  │ in SRAM    │  │ to HBM     │
   └────────────┘  └────────────┘  └────────────┘
   Never materializes the full N×N matrix in HBM.
```

### 9.12.3 The numbers

- Memory: from `O(N²)` down to `O(N)` in HBM
- Speed: 2-4× faster forward, 5-9× faster backward
- Drop-in replacement: `pip install flash-attn` and you're done

This is why long-context training (32k, 128k, 1M tokens) became feasible. Without FlashAttention, attention dominates HBM and you can't fit the activations.

---

## 9.13 The production optimization stack

```
   ┌─────────────────────────────────────────────────────┐
   │  Step 1: Pick a teacher (Llama-3-70B, e.g.)         │
   └────────────────────┬────────────────────────────────┘
                        │ Distill (synthetic data SFT)
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 2: 8B student model in BF16                   │
   └────────────────────┬────────────────────────────────┘
                        │ Wanda pruning (50% with LoRA recovery)
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 3: 8B sparse model in BF16                    │
   └────────────────────┬────────────────────────────────┘
                        │ AWQ 4-bit quantization
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 4: 8B AWQ-INT4 weights, ~4 GB                 │
   └────────────────────┬────────────────────────────────┘
                        │ FP8 KV cache, vLLM PagedAttention
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 5: Production serving on a single L40S        │
   │  with FlashAttention-2 kernels                      │
   └────────────────────┬────────────────────────────────┘
                        │ Speculative decoding (1.5B drafter)
                        ▼
   ┌─────────────────────────────────────────────────────┐
   │  Step 6: 2-3x further latency reduction             │
   │  Total: 10-20x cheaper than FP16 70B teacher        │
   └─────────────────────────────────────────────────────┘
```

---

## 9.14 Comparison table — when to use which technique

| Goal | Technique |
|------|-----------|
| Reduce memory footprint, fit in smaller GPU | AWQ-4 weights + FP8 KV cache |
| Reduce latency on commodity GPU | Structured pruning + LoRA recovery |
| Reduce latency without quality loss | Speculative decoding |
| Change architecture (70B → 7B) | Distillation |
| Local/Mac inference | GGUF Q4_K_M |
| QLoRA fine-tuning | bitsandbytes NF4 |
| Edge / mobile | INT8 PTQ + ONNX Runtime |
| Long context throughput | FlashAttention-2 + FP8 KV cache |

---

## 9.15 Resume tie-in — Sachin's quantization story

The resume mentions: **"Quantization, Pruning, Distillation"** in skills. The interviewer will ask. Have ONE concrete story ready.

**The story to tell:**

> "At TrueBalance I had a BERT-base NER model classifying customer support intents. We were running it on a SageMaker real-time endpoint and our p99 latency was around 80 ms — good but expensive because we were paying for ml.g4dn.xlarge GPUs at scale. I quantized it to INT8 using ONNX Runtime's PTQ flow with about 256 calibration samples drawn from production logs. The F1 dropped from 0.91 to 0.905 — under 1% — and we moved the endpoint to ml.c5.large CPU instances. p99 dropped to 35 ms because the CPU INT8 path actually beat the GPU FP16 path for our small batch sizes, and the cost dropped about 70%. That was my first real lesson that quantization isn't just memory — for the right workload it can win on latency too."

**Variation for LLM-flavored questions:**

> "On the IHS RAG chatbot at ResMed I evaluated AWQ-4 Llama-3-8B against the BF16 baseline. Calibration was 256 samples drawn from real clinical questions, group size 128. Perplexity delta on our held-out clinical eval was 0.08, MMLU delta was -0.6 points, and inference throughput on an A10 went from about 35 tokens per second to 85. We shipped AWQ. The KV cache was also FP8 which doubled the concurrent-request capacity of each replica."

---

## 9.16 Master Q&A — the full interview spread

**Q1. Why optimize models at all?**
> Cost, latency, and fit. A 70B FP16 model is 140 GB and needs an 8xH100 box just to load. Most production teams can't justify that — they need the model on a single GPU, with sub-second latency, and ideally fitting their existing infrastructure. Optimization lets you trade marginal capability for substantial cost and speed. The standard production recipe is distill plus quantize plus speculative decode, which compounds to ten to twenty times throughput improvement at single-digit quality cost.

**Q2. PTQ versus QAT — when do you choose each?**
> PTQ for anything large. It's hours of work, no retraining, calibration-set based. For LLMs above seven billion parameters it's the only economically rational choice because you're not going to retrain the model just to quantize it. QAT for small production models where every accuracy point matters — mobile vision classifiers, edge NER models, things where you have a fixed accuracy budget and you can afford the days of GPU time. In LLM land, QAT is essentially extinct.

**Q3. GPTQ versus AWQ versus GGUF — pick one for each scenario.**
> For GPU serving via vLLM I default to AWQ because it's activation-aware, which means it identifies the small fraction of weights that actually drive model behavior and protects them. For local Mac or laptop inference I use GGUF Q4_K_M via llama.cpp because it has a portable format and great CPU-and-Apple-Silicon kernels. For GPTQ I reach for it only when AWQ kernels aren't available for my exact hardware. Practically, on an H100 box serving instruction-tuned models, AWQ is the answer ninety percent of the time.

**Q4. SmoothQuant — what problem does it solve?**
> Activation outliers. LLMs have a small set of FFN channels that produce activations a hundred times larger than typical, which makes naive INT8 of activations clip horribly. So most LLMs only quantize weights to INT8, leaving activations in FP16 — which means the matmul is still FP16 and you don't get INT8 tensor core speedup. SmoothQuant migrates the difficulty from activations to weights with per-channel scaling: divide activations by s, multiply weights by s, the math is identity but now both fit comfortably in INT8. This unlocks W8A8 INT8 matmul on TensorRT-LLM kernels for roughly two times throughput.

**Q5. FP8 versus INT8 on H100 — which wins?**
> FP8 wins for LLMs. The exponent bits in FP8 — particularly E4M3 — preserve dynamic range, which means activation outliers get handled natively without calibration. INT8 needs SmoothQuant-style tricks to be tractable. FP8 has dedicated tensor cores on H100 with the same throughput as INT8, and the same FP8 recipe trains and serves — DeepSeek-V3 was trained natively in FP8. INT8 is still relevant on Ampere or L4 hardware where FP8 isn't available, but on H100 and B200, FP8 is the default.

**Q6. Why is KV-cache quantization so impactful?**
> Because for long-context workloads, the KV cache dominates GPU memory more than the weights do. A 70B AWQ-4 model is 35 GB, but a 32k-token KV cache for that model can eat 30+ GB. Halving the KV cache by switching from FP16 to FP8 doubles the number of concurrent requests you can fit on a GPU, which directly doubles your throughput in tokens-per-second per dollar. In vLLM it's literally one flag: dash-dash-kv-cache-dtype fp8. For chatbot or RAG workloads with long contexts, it's the single biggest throughput lever.

**Q7. Structured versus unstructured pruning — which gives latency?**
> Structured. Removing entire heads, channels, or layers makes the resulting matrices smaller dense matrices, which gives you immediate speedup on stock GPU kernels. The cost is a bigger quality hit because the granularity is coarse, so you usually follow with a short LoRA recovery fine-tune. Unstructured magnitude pruning gives you sparse matrices that need specialized kernels to be fast — and those kernels usually underperform dense matmul for moderate sparsity. The exception is 2:4 structured sparsity on Ampere or Hopper, where the hardware accelerates the pattern natively for a real two-times speedup.

**Q8. What's Wanda and why is it the modern default?**
> Wanda — Pruning by Weights and Activations, 2023 — scores each weight by its absolute magnitude times the L2 norm of the corresponding input activation. The intuition is that a weight is important if it's both large and seeing large activations. You compute these scores in a single forward pass on a small calibration set, then prune the lowest-scoring weights. There are no gradients, no retraining. It matches or beats SparseGPT, which uses second-order Hessian information, at a fraction of the compute cost. For pruning large language models in 2026, Wanda is what you reach for first.

**Q9. Walk me through DistilBERT's three-loss training.**
> The teacher BERT-base is frozen; the student DistilBERT is half the layers, initialized from every other teacher layer. Three losses jointly: KL divergence between student and teacher logits at temperature 2 — that's the soft-target loss, where dark knowledge transfers. Standard MLM cross-entropy against ground-truth tokens — that's the hard-target loss, which anchors the student to the real task. Cosine similarity loss between student and teacher hidden states — that's representational alignment. The result is a student with sixty percent of the parameters and ninety-seven percent of GLUE performance, running about sixty percent faster.

**Q10. Why does distillation temperature matter?**
> Temperature controls the peakedness of the teacher's softmax. At T equals one, the teacher's distribution is sharp — assigning ninety-nine percent to the correct class is almost the same signal as a hard label. At T equals two or three, the distribution flattens and you can see the teacher's relative confidences across classes — "this is most likely dog, but wolf and fox are plausible while table is essentially impossible." That richness is the dark knowledge — it teaches the student the conceptual structure of the label space, not just the right answer. Two to four is the empirical sweet spot.

**Q11. When do you pick distillation over quantization?**
> Three cases. One — when I need to change the architecture, like going from a 70B teacher to a 7B student because the customer's GPUs can't hold the bigger one. Two — when the teacher's behavior matters more than its exact weights, like instruction-following or chain-of-thought reasoning. Three — when I have lots of unlabeled domain data, because distillation can use teacher outputs as labels. For most production stacks I do both: distill first to get the architecture I want, then AWQ-quantize the student. They compound.

**Q12. What is speculative decoding and why is it lossless?**
> A small draft model proposes K tokens autoregressively; the large target model verifies them all in one parallel forward pass. If the draft was right, you got K tokens for the price of one target call — a two to three times speedup. The losslessness comes from rejection sampling: each draft token is accepted with probability min of one and the target's probability divided by the draft's probability, and on rejection you sample from the residual distribution. The math guarantees the final samples are distributed exactly as if you'd sampled from the target directly — same output distribution, just faster.

**Q13. When does speculative decoding stop helping?**
> When the draft acceptance rate drops below about forty percent. The draft model itself costs compute, so if the target is rejecting most proposals, you're paying for the draft and still doing standard target generation. The sweet spot is acceptance around sixty to eighty percent, which usually means a draft model about ten to twenty percent of the target's parameter count, fine-tuned on similar data. Mismatched sampling temperatures between draft and target also collapse acceptance rates.

**Q14. Explain FlashAttention in plain terms.**
> Standard attention computes a quadratic-sized N-by-N attention matrix and writes it to GPU high-bandwidth memory before the softmax. For long sequences this is bottlenecked not by compute but by memory bandwidth. FlashAttention tiles the computation: it loads small blocks of Q and K into the GPU's on-chip SRAM, computes the attention for that block locally, and streams the running softmax accumulator out to HBM. The full N-by-N matrix is never materialized in HBM. Memory drops from O(N squared) to O(N), and the kernel runs two to four times faster forward and five to nine times faster backward. It's the optimization that made long-context training feasible.

**Q15. INT4-quantized 13B has 5% accuracy drop versus FP16. Debug this for me.**
> First, per-layer sensitivity analysis — quantize one layer at a time and find the sensitive ones. Usually it's the LM head and the first or last few transformer blocks. Second, calibration data quality — am I using one hundred and twenty-eight or more samples drawn from production-like prompts? Third, switch GPTQ to AWQ if I'm using GPTQ — AWQ does better on instruction-tuned models. Fourth, reduce group size from one twenty-eight to sixty-four for tighter scales. Fifth, mixed precision — keep the LM head in FP16 and quantize the rest. Five percent drop usually means I'm over-quantizing the attention output projection.

**Q16. Quantized model is the right size but slower than FP16. Diagnose.**
> Almost certainly a kernel issue — the quantization library doesn't have optimized kernels for my hardware, so it's dequantizing every weight on every forward pass into a slow path. Fix is to use AWQ via vLLM or GPTQ via ExLlamaV2 — frameworks with battle-tested CUDA kernels. FP16 on A100 has phenomenal kernels; naive INT4 without proper kernels can absolutely be slower. Always benchmark end-to-end after quantization, never trust theoretical FLOPs.

**Q17. Block size in quantization — what does it mean?**
> Block size, or group size, is the granularity at which you store quantization scales. Group size of one twenty-eight means one scale and one zero-point per one twenty-eight contiguous weights. Smaller group means finer precision but more metadata overhead. Default of one twenty-eight is the standard tradeoff; sixty-four helps on sensitive models at the cost of slightly more memory; thirty-two is rare and expensive. The metadata overhead matters less than you'd think because it's tiny relative to the weights themselves.

**Q18. 2:4 structured sparsity — explain the hardware trick.**
> NVIDIA Ampere and newer GPUs have a tensor core feature where, if every group of four consecutive weights has at most two non-zeros, the hardware can skip the zero multiplies and process the matmul roughly twice as fast. The fifty percent sparsity is fixed by the hardware — it's not arbitrary. NVIDIA provides ASP, Automatic Sparsity, which fine-tunes a dense model to land in the 2:4 pattern with minimal accuracy loss. It's a real two-times speedup on the right kernels, but it's the only structured sparsity pattern with hardware support, so it's not as flexible as research-grade unstructured sparsity.

**Q19. When is INT3 or INT2 acceptable?**
> Almost never for production. Quality loss is catastrophic for reasoning tasks, long-context tasks, and instruction following. INT4 with AWQ or GPTQ is the practical floor. INT3 and INT2 show up in research papers exploring extreme compression, and occasionally in absolute-edge scenarios — embedded devices with kilobyte-scale memory — but I haven't shipped them. If I needed extreme compression I'd reach for distillation to a smaller architecture before going below INT4.

**Q20. NF4 in QLoRA — why "normal float"?**
> NF4 is information-theoretically optimal for normally-distributed weights, which neural network weights approximately are after training. Instead of uniformly-spaced quantization levels like INT4, NF4 uses quantile-based levels — more codes near zero where weights are densest, fewer codes in the tails. Combined with double quantization — quantizing the quantization scale itself — and paged optimizers to prevent OOM, it enables sixty-five-billion-parameter QLoRA on a single forty-eight-gigabyte GPU. NF4 is specifically a fine-tuning datatype; for pure inference I'd still pick AWQ or GPTQ.

**Q21. Validating a quantized model before shipping it — what's the checklist?**
> Five things. Perplexity on a held-out corpus versus the FP16 baseline — a fast smoke test. Task-specific eval on whatever benchmarks matter for the use case — MMLU for reasoning, HumanEval for code, RAGAS for retrieval. Human or LLM-judge eval on realistic production prompts — automated metrics miss subtle quality drops. Latency and throughput benchmarks at production batch sizes, not just batch one. And a regression check — ten or twenty handpicked golden prompts that shouldn't change behavior. If all five clear, I ship.

**Q22. SmoothQuant model has good per-sample accuracy but breaks at batch size one. Why?**
> Activation statistics calibrated for the production batch size don't hold at batch one. SmoothQuant's per-channel scales are computed assuming a certain distribution of activation magnitudes; at batch one, you're sampling a single point from that distribution and the scales might not fit. Fix is either to calibrate with the actual production batch sizes you'll see, or use per-token dynamic quantization of activations rather than static. Static quantization is faster but more sensitive to batch-size mismatch.

**Q23. Production quantization workflow you'd run.**
> Pick the base model. Pull two hundred fifty-six to five hundred twelve calibration samples from sanitized production logs. Run AWQ four-bit weights with group size one twenty-eight. Switch the KV cache to FP8. Evaluate against the FP16 baseline on perplexity, task benchmarks, and a small human-eval set. If the deltas are within budget — typically under one percent on benchmarks, under three percent on human eval — deploy with vLLM. Then A/B test against the previous version in production for a week, monitor quality drift via daily golden-set runs, and roll out fully if metrics hold.

**Q24. LoRA adapters plus quantized base — what changes in serving?**
> The base model stays quantized — for example AWQ-4 — and the LoRA adapters stay in BF16. vLLM and LoRAX both support this combination. At inference, the base weights are dequantized on the fly during the matmul, and the LoRA delta — B times A — is added in BF16. There's a small compute overhead because you're doing two matmuls instead of one, but the memory savings from keeping the base quantized are massive. This is how you serve fifty fine-tuned versions of a base model on one GPU with hot-swappable adapters.

**Q25. Model merging versus distillation — when each?**
> Model merging combines fine-tuned checkpoints — TIES, DARE, SLERP — into a single model with no training. It's free, preserves all the parameters, and works when you have multiple specialists you want to combine into one generalist. Distillation creates a smaller student from a teacher, which costs training compute but gives you architectural change. Different tools for different goals. Merging when you have many fine-tunes and want one combined model; distillation when you need a smaller architecture or to teach behavior to a fresh model.

---

Continue to **[Chapter 10 — MLOps & LLMOps](10_mlops_llmops.md)**.
