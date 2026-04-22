# Chapter 02 — Transformer Architecture, Deep Dive
## From scaled dot-product attention to FlashAttention-3

> The single most important topic for any modern AI Engineer interview. You **will** be asked to explain self-attention. You will **probably** be asked about RoPE, GQA/MQA, FlashAttention, or KV-cache. Be ready.

---

## 2.1 The "Attention is All You Need" Moment (Vaswani et al., 2017)

Before 2017, sequence modelling used RNNs (LSTM, GRU) which processed tokens one at a time — slow, limited context, vanishing gradients. Transformers replaced recurrence with **attention**, enabling:

- **Full parallelization** across the sequence during training
- **Direct long-range dependencies** (any token can attend to any other in 1 hop)
- **Scalability** to hundreds of billions of parameters

### The full diagram

```
           ┌───────────────────────── Inputs ─────────────────────────┐
           │                                                           │
     ┌─────▼─────┐                                             ┌──────▼───────┐
     │  Token    │                                             │   Target     │
     │  Embed    │                                             │   Token      │
     └─────┬─────┘                                             │   Embed      │
           │                                                    └──────┬───────┘
     ┌─────▼─────┐                                             ┌──────▼───────┐
     │ + Pos Enc │                                             │   + Pos Enc  │
     └─────┬─────┘                                             └──────┬───────┘
           │                                                           │
     ┌─────▼──────────────── Encoder ─────────┐            ┌───────────▼────────── Decoder ──────────────────┐
     │                                         │            │                                                 │
     │  ┌──────────────────────────────────┐   │            │  ┌──────────────────────────────────────┐      │
     │  │  Multi-Head Self-Attention       │   │            │  │  Masked Multi-Head Self-Attention    │      │
     │  │  (Q,K,V from same sequence)      │   │            │  │  (causal mask)                       │      │
     │  └─────────────┬────────────────────┘   │            │  └──────────────┬───────────────────────┘      │
     │          Residual + LayerNorm           │            │            Residual + LayerNorm                │
     │                 │                        │            │                 │                              │
     │                 │                        │            │  ┌──────────────▼───────────────────────┐      │
     │                 │                        │            │  │  Multi-Head Cross-Attention          │      │
     │                 │                        │            │  │  (Q from decoder, K,V from encoder)  │      │
     │                 │                        │   ────────▶│  └──────────────┬───────────────────────┘      │
     │                 │                        │            │            Residual + LayerNorm                │
     │  ┌──────────────▼──────────────────┐    │            │                 │                              │
     │  │  Feed-Forward Network (FFN)     │    │            │  ┌──────────────▼───────────────────────┐      │
     │  │  FFN(x) = max(0, xW1+b1)W2+b2   │    │            │  │  Feed-Forward Network                │      │
     │  └─────────────┬───────────────────┘    │            │  └──────────────┬───────────────────────┘      │
     │          Residual + LayerNorm           │            │            Residual + LayerNorm                │
     │                 │                        │            │                 │                              │
     │           (× N = 6 layers)               │            │          (× N = 6 layers)                      │
     └─────────────────┼────────────────────────┘            └─────────────────┼──────────────────────────────┘
                       │                                                        │
                       └────────────────────────────────────────────────────────┤
                                                                               │
                                                                     ┌─────────▼────────┐
                                                                     │   Linear + Softmax │
                                                                     └─────────┬────────┘
                                                                               │
                                                                     ┌─────────▼────────┐
                                                                     │ Output P(x_t+1)  │
                                                                     └──────────────────┘
```

---

## 2.2 Scaled Dot-Product Attention — the math, slowly

### Step-by-step

Given token embeddings X ∈ ℝ^(n × d) (n tokens, d dim):

```
Q = X · W_Q     (n × d_k)   ← what am I looking for?
K = X · W_K     (n × d_k)   ← what do I contain?
V = X · W_V     (n × d_v)   ← if you match me, what do I tell you?
```

Compute attention:

```
          Q · Kᵀ
scores = ────────       (n × n)
           √d_k

weights = softmax(scores + mask)   (row-wise softmax)

output = weights · V              (n × d_v)
```

### Why each piece is there

| Component | Purpose |
|-----------|---------|
| **Three separate projections (Q, K, V)** | Decouples what a token "asks" from what it "offers" and what it "returns" |
| **Dot product Q·K** | Cosine-like similarity between queries and keys |
| **√d_k scaling** | Prevents softmax saturation at large d_k (variance stays ~1) |
| **Softmax** | Converts scores into a probability distribution over tokens |
| **Mask** | -∞ on disallowed positions (padding, future tokens) |
| **Weighted sum of V** | Each output = attention-weighted mixture of all values |

**Geometric intuition:** Attention is a learned soft-lookup. K is the key, V is the value, Q is the query — just like a dict, but continuous and differentiable.

---

## 2.3 Multi-Head Attention (MHA)

Instead of one big attention, use h "heads" in parallel, each with smaller d_k = d_model/h:

```
     X (n × d_model)
      │
      ├─ Head 1: Q₁K₁V₁ → attention → head₁_out (n × d_v)
      ├─ Head 2: Q₂K₂V₂ → attention → head₂_out
      ├─ ...
      └─ Head h: QₕKₕVₕ → attention → headₕ_out
                    │
                    ▼
        concat(head₁, ..., headₕ)   (n × d_model)
                    │
                    × W_O           ← final projection
                    │
                    ▼
                 Output
```

**Why multiple heads?** Each head learns different patterns:
- One head tracks syntactic dependencies (subject-verb)
- Another tracks semantic relations (coreference)
- Another handles positional structure

Ablations show each head *does* specialize, though interpretability is an active research area.

### Typical sizes (GPT-2 small vs LLaMA-2 70B)

| Model | d_model | n_heads | d_k = d_model/n_heads | Layers |
|-------|---------|---------|----------------------|--------|
| GPT-2 small | 768 | 12 | 64 | 12 |
| GPT-3 175B | 12,288 | 96 | 128 | 96 |
| LLaMA-2 7B | 4,096 | 32 | 128 | 32 |
| LLaMA-2 70B | 8,192 | 64 | 128 | 80 |
| LLaMA-3 405B | 16,384 | 128 | 128 | 126 |

---

## 2.4 MHA → MQA → GQA → MLA (the KV-cache optimization chain)

### The problem

At inference, each layer caches K and V for every token generated so far. For a 70B model at 8K context:

```
KV-cache = 2 (K,V) × layers × n_heads × d_k × seq_len × dtype_bytes
         = 2 × 80 × 64 × 128 × 8192 × 2 bytes
         ≈ 21.5 GB per request
```

That's per user. Serve 10 concurrent users and you're out of VRAM on an 80GB H100.

### The fix: reduce the number of K/V heads

| Variant | Query heads | KV heads | KV-cache size | Quality |
|---------|------------|----------|---------------|---------|
| **MHA** (classic) | N | N | 1× baseline | Best |
| **MQA** (PaLM) | N | 1 | 1/N | -2-3% perplexity |
| **GQA** (LLaMA-2) | N | G (e.g., 8) | 1/(N/G) = G/N | ~MHA |
| **MLA** (DeepSeek-V2/V3) | N | low-rank latent | ~1/16 × MHA | ~MHA |

### Visual

```
MHA:  32 Q heads → 32 KV pairs   (1:1)
MQA:  32 Q heads → 1  KV pair    (32:1)
GQA:  32 Q heads → 8 KV groups   (4:1)   ← LLaMA-2, LLaMA-3, Mistral
MLA:  K,V compressed to a shared latent vector, decompressed on the fly
```

**Production default in 2026:** GQA with 4-8 groups, or MLA for frontier models.

---

## 2.5 Positional Encoding

Attention is permutation-invariant — {token A, token B} gives the same output as {B, A} without positional info. Solutions:

### 2.5.1 Sinusoidal (original, Vaswani 2017)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Added to embeddings. Fixed, deterministic, extrapolates slightly.

### 2.5.2 Learned absolute PE (GPT-2, BERT)

One learnable vector per position. Simpler, but can't extrapolate beyond training length.

### 2.5.3 RoPE — Rotary Position Embedding (RoFormer, 2021)

Used in LLaMA, Mistral, Qwen, DeepSeek, GPT-NeoX, Gemma.

**Idea:** Instead of adding position to the embedding, **rotate Q and K in 2D subspaces** by an angle proportional to position:

```
For each pair of dimensions (2i, 2i+1):
    θ = 10000^(-2i/d)
    R_pos = [cos(pos·θ)  -sin(pos·θ)]
            [sin(pos·θ)   cos(pos·θ)]
    Q_rotated = R_pos · Q_slice
    K_rotated = R_pos · K_slice
```

**Why it's brilliant:** The inner product `Q_rot_i · K_rot_j` depends only on the **relative position** `i - j` — not on absolute positions. So the model gets relative-position bias for free, and the math is a rotation (cheap, preserves norms).

**Why it dominates 2026 LLMs:**
- Relative position is what matters for language
- Extrapolates better than learned PE (with YaRN scaling)
- Composes cleanly with FlashAttention
- No extra parameters

### 2.5.4 RoPE scaling — extending context (NTK, YaRN)

Problem: train with 4K context, want 32K at inference. Vanilla RoPE breaks because high-frequency dimensions wrap around too fast at unseen positions.

Fixes:

- **NTK-aware scaling** — interpolate RoPE base (stretches more at low freq than high freq). Cheap.
- **YaRN** — adds temperature correction and ramp function; extends 4-32× context with minimal fine-tuning.
- **Dynamic NTK** — applied only when sequence exceeds training length.

### 2.5.5 ALiBi — Attention with Linear Biases

```
attention_score(i, j) += -m · |i - j|
```
where m is a head-specific slope. No positional embeddings at all — just a linear penalty for distant tokens.

- **Pros:** Zero params, extrapolates to any length without fine-tuning
- **Cons:** Weaker on strict relative-reasoning tasks than RoPE
- **Used in:** BLOOM, MPT

---

## 2.6 FlashAttention — same math, 5-20× less memory

Standard attention materializes the n×n attention matrix in HBM (GPU memory), which is expensive and memory-hungry:

```
Standard:  Q·K^T → [n, n] → softmax → [n, n] → · V → output
           (writes and re-reads the full matrix)
```

**FlashAttention (Dao et al., 2022):** tile Q, K, V into blocks that fit in SRAM, fuse the softmax and matmul, and never materialize the full matrix. Outputs are mathematically identical.

```
for each block of Q:
    for each block of K, V:
        load into SRAM
        compute partial attention
        update running softmax statistics
    write final output to HBM
```

**Results:** 2-4× faster, 5-20× less memory at long context. Enabled 8K→128K contexts on commodity hardware.

Evolution:
- **FlashAttention-1** (2022) — the original
- **FlashAttention-2** (2023) — better work partitioning across warps
- **FlashAttention-3** (2024) — Hopper asynchrony, FP8 support, another ~2× speedup

**Practical:** Use via PyTorch 2+ `torch.nn.functional.scaled_dot_product_attention`, or vLLM / TensorRT-LLM (both integrate it).

---

## 2.7 The Feed-Forward Network (FFN)

After attention, each token passes through a 2-layer MLP:

```
FFN(x) = σ(x W_1 + b_1) W_2 + b_2
         ├──────┬──────┘      └──┬──┘
         expand to 4d            project back to d
```

**Why 4×?** Empirically found to work well. This is where most of the model's parameters live (~2/3 of weights in a transformer block).

### Modern upgrade: SwiGLU / GeGLU (used in LLaMA, Mistral, PaLM)

```
SwiGLU(x) = (Swish(x · W_gate)) ⊙ (x · W_up) · W_down
```

Gating lets the model dynamically scale information. To keep parameter count ~same, hidden size is usually ~(2/3) × 4d = 8/3·d.

---

## 2.8 Normalization — Pre-LN vs Post-LN, LayerNorm vs RMSNorm

### Pre-LN vs Post-LN

```
Post-LN (original):      Pre-LN (modern):
x → Attn → +x → LN       x → LN → Attn → +x
   ↓                        ↓
x → FFN → +x → LN        x → LN → FFN → +x
```

**Pre-LN** is more stable for deep stacks (easier to train 100+ layers without learning-rate warmup tricks). **Post-LN** performs slightly better when stable. Modern LLMs universally use Pre-LN.

### LayerNorm vs RMSNorm

| | LayerNorm | RMSNorm |
|--|-----------|---------|
| Formula | (x - μ) / σ · γ + β | x / RMS(x) · γ |
| Mean-centering | Yes | **No** |
| Learnable bias β | Yes | No |
| Speed | baseline | ~7-10% faster |
| Quality | ~identical | ~identical |

RMSNorm is standard in LLaMA family and most modern LLMs.

---

## 2.9 Encoder-only vs Decoder-only vs Encoder-Decoder

| | **Encoder-only (BERT, RoBERTa)** | **Decoder-only (GPT, LLaMA, Claude)** | **Encoder-Decoder (T5, BART)** |
|--|----------------------------------|---------------------------------------|-------------------------------|
| Attention | Bidirectional (all positions) | Causal/masked (can only see past) | Enc: bidirectional; Dec: causal + cross-attn |
| Pretraining | MLM (masked token) + NSP | Next-token prediction | Denoising / span corruption |
| Output | Hidden states for each token | Next-token probability | Autoregressive target |
| Best for | Classification, NER, embeddings | Generation, chat, reasoning | Translation, summarization |

### Why decoder-only "won" for LLMs

- **Training signal density**: every token predicts the next one → N signals per sequence
- **Simpler**: one stack, not two
- **Unifies tasks**: everything is "generate the answer"
- **Scales beautifully**: GPT-3 showed pure decoder-only matches or beats enc-dec at scale

---

## 2.10 The Math You Should Know for a Whiteboard

Given a single attention head:

```
Inputs:     X ∈ ℝ^(n × d_model)
Weights:    W_Q, W_K ∈ ℝ^(d_model × d_k),  W_V ∈ ℝ^(d_model × d_v)

Q = X W_Q               [n × d_k]
K = X W_K               [n × d_k]
V = X W_V               [n × d_v]

A = softmax( (Q K^T) / √d_k + M )    [n × n]
O = A V                              [n × d_v]
```

With multi-head (h heads), concat and project:

```
MHA(X) = Concat(head_1, ..., head_h) W_O    where head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
```

Total params per attention layer ≈ 4 · d_model² (Q, K, V, O projections).

---

## 2.11 Compute & memory — back-of-envelope

### Training
- Forward FLOPs ≈ 2 · params · tokens
- Forward + backward ≈ 6 · params · tokens
- Example: train 70B on 2T tokens → 6 × 70e9 × 2e12 = **8.4 × 10²² FLOPs** ≈ 10,000 H100-days

### Inference
- **Prefill** (processing prompt): parallel, ~O(n·params) per layer, compute-bound
- **Decode** (generating tokens): sequential, ~O(params) per token, **memory-bandwidth-bound** (you're streaming model weights from HBM to compute units)

**This is why vLLM's PagedAttention and continuous batching matter so much** — decode is bandwidth-starved, so max-batching requests is the only lever.

---

## 2.12 Interview Q&A — Transformer

> *Synthesis: 50+ high-probability questions, with the "tricky" ones flagged.*

**Q1. Why scale by √d_k in attention?**
> Dot products of d_k-dim random vectors have variance ~d_k. Without scaling, softmax saturates at large d_k and gradients vanish. √d_k keeps variance at 1.

**Q2. What's the computational complexity of self-attention?**
> O(n² · d) for the n×n attention matrix and O(n·d²) for projections. The quadratic-in-n term is the bottleneck at long contexts — motivating FlashAttention, sliding window, and linear-attention variants.

**Q3. MHA vs MQA vs GQA — which do production LLMs use?**
> MHA = full N/N; MQA = N query heads, 1 KV head (aggressive cache savings, quality loss); GQA = groups (compromise). LLaMA-2/3, Mistral, Qwen all use GQA. DeepSeek-V2/V3 use MLA (compressed latent KV), the current SOTA.

**Q4. What is the KV-cache and why do we need it?**
> During autoregressive decode, we'd recompute K, V for all prior tokens at every step — O(n²). Caching K, V means new-token decode is O(n) (just compute one new Q and attend to cached K, V).

**Q5. Why doesn't the KV-cache exist during training?**
> Training processes the full sequence in parallel (teacher forcing). There's no autoregressive loop; all Q,K,V are computed in one shot per layer.

**Q6. Why RoPE instead of sinusoidal?**
> RoPE encodes *relative* position through rotation of Q and K, so the inner product depends only on i-j. Extrapolates better with NTK/YaRN scaling and composes with FlashAttention. Sinusoidal is additive and absolute; models have to learn relative-position from scratch.

**Q7. What is YaRN?**
> Yet another RoPE extensioN — scales RoPE frequencies non-uniformly (high-freq dims stretched less than low-freq) with a temperature correction and ramp function. Enables 4-32× context extension with minimal fine-tuning.

**Q8. ALiBi vs RoPE?**
> ALiBi: no positional embedding, adds linear bias proportional to distance. Zero-shot extrapolates to any length but quality is slightly lower. RoPE: rotation-based, needs YaRN for long context, higher quality. Production LLMs in 2026 use RoPE.

**Q9. Why LayerNorm / RMSNorm instead of BatchNorm in transformers?**
> Sequences are variable length and batch=1 at inference is common. BatchNorm's batch statistics are unreliable. LayerNorm normalizes per-token across features; RMSNorm skips mean-centering (~7-10% faster). Modern LLMs use RMSNorm.

**Q10. Pre-LN vs Post-LN?**
> Pre-LN places the norm inside the residual stream; stable for deep nets. Post-LN places it after the residual; slightly higher ceiling when stable but needs LR warmup. Modern LLMs use Pre-LN.

**Q11. What does the Feed-Forward (FFN) block actually do?**
> Two-layer MLP applied independently per position. Empirically stores factual knowledge (probing research shows FFN acts as a key-value memory). Expands to 4d, projects back. Modern LLMs use SwiGLU gating instead of ReLU.

**Q12. How does FlashAttention work and why is it faster?**
> IO-aware: tiles Q, K, V into blocks that fit in SRAM, fuses softmax + matmul, never materializes the full n×n attention matrix in HBM. Same math, 2-4× faster, 5-20× less memory. FA-2 improves work-partitioning; FA-3 adds Hopper async + FP8.

**Q13. Why causal masking in GPT-style models?**
> To enforce autoregressive factorization P(x_t | x_<t). Position t is masked from attending to t+1…n via -∞ in attention scores. Remove the mask and you get a bidirectional (encoder-like) model — can't generate autoregressively.

**Q14. Encoder-only vs decoder-only — which to use for what?**
> Encoder-only (BERT) — classification, NER, embeddings. Decoder-only (GPT, LLaMA) — generation, chat, reasoning. Encoder-decoder (T5) — translation, summarization when strict input→output is needed. Modern generalists are decoder-only.

**Q15. [Gotcha] In GQA with 32 Q-heads and 8 KV-heads, does training FLOPs change?**
> Almost no. Attention matmul dominates over K/V projections. The savings are purely at inference — KV-cache shrinks 4×, memory bandwidth stays in budget when serving long-context requests.

**Q16. [Gotcha] Your transformer's loss plateaus after a few thousand steps. What do you check first?**
> (1) Learning rate — too high blows up; too low stalls. Try LR warmup. (2) Gradient clipping — grads exploding. (3) LayerNorm placement — Post-LN without warmup is unstable. (4) Bad data — duplicate-heavy or low-quality corpus. (5) Precision — FP16 without loss-scaling can underflow; use BF16 if available.

**Q17. Why do transformers use residual connections?**
> Enables arbitrarily deep stacks. Without residuals, gradients through N layers of nonlinearities vanish or explode. A skip path with gradient 1 preserves signal flow through 100+ layers.

**Q18. [Gotcha] Two models have the same num_params — same quality?**
> Not necessarily. Architecture choices (GQA vs MHA, SwiGLU vs ReLU, pre-LN vs post-LN, RoPE vs learned PE), training-token budget, data quality, and hyperparameters matter enormously. A 7B model trained on 15T tokens can beat a 30B trained on 300B.

**Q19. What's the difference between attention weights and attention *scores*?**
> Scores = pre-softmax (raw dot products / √d_k). Weights = post-softmax (non-negative, sum to 1). People sometimes use them interchangeably in ML papers — be precise in interviews.

**Q20. Speculative decoding — explain it to a non-expert.**
> A tiny "draft" model guesses the next k tokens; the big model checks all k in one forward pass and accepts the longest valid prefix. Since verification is parallel and generation is sequential, you get 2-3× speedup with identical output distribution.

**Q21. What is Multi-Head Latent Attention (MLA) in DeepSeek?**
> MLA compresses K and V into a shared low-rank latent vector cached instead of full K/V tensors. Achieves ~93% KV-cache reduction vs MHA with MHA-level quality. Combined with decoupled-RoPE for position info. Used in DeepSeek-V2/V3.

**Q22. Why is decode memory-bandwidth-bound?**
> Each decode step processes 1 token but needs to stream ALL model weights (for the Q projection) and ALL KV-cache (for attention) from HBM. Compute per token is tiny; memory transfer dominates. This is why max-batching (serving many requests concurrently) is essential — amortizes the weight transfer over many tokens.

**Q23. FlashAttention-3 specific advantages?**
> Hopper (H100/H200) async pipelining — overlaps memory copy and compute via CUDA MMAs. Native FP8 tensor cores. Better TMA (Tensor Memory Accelerator) usage. ~2× over FA-2 on Hopper.

**Q24. What happens if you remove positional encoding entirely?**
> Attention becomes permutation-invariant — the model can't tell word order apart. "Dog bites man" and "Man bites dog" get identical representations. For causal decoder-only models, some position info leaks through the mask itself, but this is weak — you'd lose accuracy without explicit PE.

**Q25. [Gotcha] You train a model with 4K context and deploy with 16K. What breaks?**
> With vanilla RoPE/PE, positions 4K-16K are OOD — the model produces garbage. Fixes: (1) RoPE scaling (NTK/YaRN), (2) train with longer context from the start, (3) sliding-window attention (Mistral-style) that caps effective context at a fixed window. Always test before deploying past training context length.

**Q26. What are sparse attention patterns?**
> Attention variants that skip most pairs: local/windowed (Longformer), strided (Sparse Transformer), global-plus-local (BigBird), random (Reformer). Reduce O(n²) to O(n·w) or O(n log n). Trade-off: quality loss on long-range dependencies. Mostly replaced by FlashAttention + RoPE + YaRN in 2026.

**Q27. Why does scaled dot-product attention use dot product and not cosine?**
> Cosine similarity would ignore magnitude; dot product preserves it. Also, cosine requires extra normalization (compute cost). The √d_k scaling recovers the stability benefit of normalization.

**Q28. How many parameters in one transformer block?**
> Attention: 4·d² (Q, K, V, O). FFN: 2 · d · 4d = 8d² (or ~2·8/3·d² = 5.33d² for SwiGLU-matched sizes). LayerNorm params negligible. Total ≈ 12d² per block for classic, ~9.33d² with SwiGLU.

**Q29. [Gotcha] You increase heads from 8 to 16 with fixed d_model. What happens?**
> Each head gets d_k = d_model/16 instead of d_model/8, so heads are "thinner." Total compute stays same. Sometimes helps (more specialization), sometimes hurts (heads too narrow to capture patterns). Typical sweet spot: d_k = 64 or 128.

**Q30. What is "absolute" vs "relative" position encoding — why does relative win?**
> Absolute: each position gets a unique vector (works for train-length, breaks OOD). Relative: encode distance i-j, not absolute i. Relative generalizes naturally to longer sequences — key insight behind T5-style bias, Shaw-style embeddings, and RoPE.

---

## 2.13 Resume tie-ins

- **"Architected AI-powered ML workspace assistant on Claude"** — Claude is decoder-only, RoPE-likely, long-context-capable. Be ready to discuss why Claude Sonnet was a good choice for agentic tool-calling (strong instruction-following, JSON mode, multi-step reasoning).
- **"RAG knowledge-base chatbot at ResMed"** — embedding models are encoder-only transformers; generation uses decoder-only. Be ready to distinguish.
- **"Real-time XGBoost Lambda"** — not a transformer, but you can contrast: XGBoost for tabular (trees capture feature interactions well), transformers for sequential (attention captures long-range deps). Pick the right tool.

---

Continue to **[Chapter 03 — How LLMs Work End-to-End](03_llms.md)**.
