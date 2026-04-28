# Chapter 02 — Transformer Architecture, Deep Dive
## From scaled dot-product attention to FlashAttention-3

> The single most important topic in any modern AI Engineer interview. You **will** be asked to explain self-attention. You will **probably** be asked about RoPE, GQA/MQA, FlashAttention, KV-cache, or causal masking. This chapter is the one to memorize cold — but more importantly, to internalize so deeply that you can draw the diagrams from scratch.

---

## 2.1 Why transformers exist — the story before "Attention Is All You Need"

Picture the world of NLP in 2016. The state of the art was sequence-to-sequence with LSTMs and GRUs, plus an attention mechanism bolted on top (Bahdanau 2014). To translate English to French, you'd run an LSTM encoder over the source sentence, summarize it into a vector, and decode word-by-word with a second LSTM. It worked, but barely — long sentences degraded quality because the LSTM hidden state forgot early tokens, and training was sequential, so you couldn't parallelize across the sequence.

Three problems were strangling the field:

1. **Sequential bottleneck**: An LSTM processes tokens one at a time. To compute the hidden state at position 100, you need positions 1-99 already done. You cannot parallelize across the time dimension. With 1000-token sequences, GPUs sat idle most of the time.
2. **Long-range dependencies**: Information from token 1 has to flow through 99 LSTM steps to influence token 100. Each step multiplies by a weight matrix and applies a nonlinearity, and the signal degrades exponentially. LSTMs partially fixed this with gating, but not enough.
3. **Vanishing gradients**: Same problem in reverse — the gradient from position 100 has to flow back through 99 layers, and it dies on the way.

Vaswani et al.'s 2017 paper "Attention Is All You Need" did something audacious: throw out the recurrence entirely. Replace it with a single mechanism — attention — that lets every token directly look at every other token in one step. Now you can process the whole sequence in parallel during training, and any two tokens are exactly one hop apart in the computational graph.

That single architectural decision unlocked everything: GPT-1 (2018), BERT (2018), GPT-2 (2019), GPT-3 (2020), ChatGPT (2022), Claude (2023), and every modern LLM. The transformer is the most consequential ML architecture of the last decade.

### Mental model

Think of a transformer layer as a "gossip round." Every token in the sequence simultaneously asks every other token: "Hey, do you have something relevant for me?" Tokens that match exchange information, weighted by relevance. Then each token updates its own state with the weighted gossip. That's one layer. Stack 24-126 of these and you have a modern LLM.

### The full encoder-decoder block diagram

```
                    ┌──────────────────── Inputs (source) ─────────────────────┐
                    │                                                          │
              ┌─────▼─────┐                                            ┌──────▼───────┐
              │ Token     │                                            │   Target     │
              │ Embedding │                                            │   Token      │
              └─────┬─────┘                                            │   Embedding  │
                    │                                                  └──────┬───────┘
              ┌─────▼─────┐                                            ┌──────▼───────┐
              │ + Pos Enc │                                            │  + Pos Enc   │
              └─────┬─────┘                                            └──────┬───────┘
                    │                                                         │
              ┌─────▼─────────── ENCODER (×N) ───────┐         ┌──────────────▼─── DECODER (×N) ────────┐
              │                                       │         │                                        │
              │  ┌─────────────────────────────────┐  │         │  ┌──────────────────────────────────┐  │
              │  │ Multi-Head Self-Attention       │  │         │  │ Masked Multi-Head Self-Attention │  │
              │  │ (Q,K,V from same sequence)      │  │         │  │ (causal mask)                    │  │
              │  └────────────┬────────────────────┘  │         │  └────────────┬─────────────────────┘  │
              │       Residual + LayerNorm            │         │       Residual + LayerNorm             │
              │                │                       │         │                │                       │
              │                │                       │         │  ┌─────────────▼─────────────────────┐ │
              │                │                       │         │  │ Multi-Head Cross-Attention        │ │
              │                │                       │  ──────▶│  │ (Q from decoder, K,V from encoder)│ │
              │                │                       │         │  └─────────────┬─────────────────────┘ │
              │                │                       │         │       Residual + LayerNorm             │
              │  ┌─────────────▼──────────────────┐    │         │                │                       │
              │  │ Feed-Forward Network (FFN)     │    │         │  ┌─────────────▼─────────────────────┐ │
              │  │ FFN(x) = SwiGLU(x)             │    │         │  │ Feed-Forward Network              │ │
              │  └─────────────┬──────────────────┘    │         │  └─────────────┬─────────────────────┘ │
              │       Residual + LayerNorm            │         │       Residual + LayerNorm             │
              └────────────────┼──────────────────────┘         └────────────────┼───────────────────────┘
                               │                                                 │
                               └─────────────────────────────────────────────────┤
                                                                                 │
                                                                       ┌─────────▼────────┐
                                                                       │ Linear + Softmax │
                                                                       └─────────┬────────┘
                                                                                 │
                                                                       ┌─────────▼────────┐
                                                                       │  P(next token)   │
                                                                       └──────────────────┘
```

This is the original Vaswani 2017 picture. Modern decoder-only LLMs (GPT, LLaMA, Claude) drop the encoder and the cross-attention block, keeping only the right side.

---

## 2.2 Self-attention — the math, slowly, with shapes

This is the single most important concept in the chapter. Take it slow.

### The plain English mental model

Think of attention as a soft, differentiable dictionary lookup. In a Python dict, you have keys and values, and a query exact-matches one key. In attention, every input token produces three vectors — Query, Key, Value — and the lookup is *soft*: the query is compared to every key, similarities are normalized into a probability distribution (the attention weights), and the output is a weighted sum of the values.

Or think of it like Google search. The query is your search string. The keys are the index entries — what the documents claim to be about. The values are the actual document contents. You match your query against keys to get relevance scores, normalize, then return a relevance-weighted blend of contents.

### The math, step by step

Given a sequence of token embeddings X with shape `(n, d_model)` — n tokens, each of dimension d_model:

```
Step 1: project X into Q, K, V
        Q = X · W_Q     (n × d_k)        ← what am I looking for?
        K = X · W_K     (n × d_k)        ← what do I contain?
        V = X · W_V     (n × d_v)        ← if you match me, here's what I tell you

Step 2: compute scores
                   Q · Kᵀ
        scores = ──────────       (n × n)
                    √d_k

Step 3: apply mask (for causal / padding)
        scores = scores + M       (M has -∞ at masked positions)

Step 4: softmax row-wise
        weights = softmax(scores)  (n × n, each row sums to 1)

Step 5: weighted sum of values
        output = weights · V      (n × d_v)
```

### Why each piece is there

- **Three separate projections (Q, K, V)**: A token's role as a "looker" (query) is different from its role as something to be looked at (key) and what it actually contributes (value). Decoupling these gives the model more expressive power. With d_model=512 and d_k=64, each projection is a 512-to-64 linear map.
- **Dot product Q·K**: A direct measure of similarity between query and key in the same vector space.
- **`sqrt(d_k)` scaling**: When d_k is large, the dot product of two random vectors has variance ~d_k. Without scaling, the softmax saturates and gradients vanish. Dividing by sqrt(d_k) keeps variance ~1.
- **Softmax**: Turns scores into a probability distribution over tokens. Differentiable, and ensures the weights sum to 1 so the output is a convex combination of values.
- **Mask**: For causal attention (decoder), positions can't attend to the future, so we add `-inf` to those scores. After softmax those positions get weight 0.
- **Weighted sum of V**: The output for each query position is the attention-weighted blend of all value vectors.

### Worked example with numbers

Imagine d_model=512, batch=8, seq_len=128, single head with d_k=d_v=64.

- X has shape `(8, 128, 512)`.
- W_Q, W_K, W_V each have shape `(512, 64)`.
- Q = X @ W_Q has shape `(8, 128, 64)`.
- K transpose has shape `(8, 64, 128)`.
- `Q @ K.T` has shape `(8, 128, 128)` — this is the attention matrix. For batch=8 and seq=128, that's 8 * 128 * 128 = 131,072 attention scores.
- After softmax over the last dim, each row of each attention matrix sums to 1.
- `weights @ V` has shape `(8, 128, 64)` — same shape as Q, ready to project back up to d_model.

For seq_len=8K, that attention matrix is 8 * 8192 * 8192 = ~537M entries per head per batch. **This is the quadratic-in-n bottleneck** that motivates FlashAttention and sliding-window attention.

### Causal mask — the 4×4 example

For autoregressive generation, position t cannot see positions > t. The mask matrix M has -inf above the diagonal:

```
                 K position
         k=0   k=1   k=2   k=3
q=0   [   0   -inf  -inf  -inf  ]
q=1   [   0     0   -inf  -inf  ]
q=2   [   0     0     0   -inf  ]
q=3   [   0     0     0     0   ]
```

After adding to scores and applying softmax row-wise, position 0 attends only to itself, position 1 to {0, 1}, position 2 to {0, 1, 2}, etc. This enforces P(x_t given x_<t) — exactly what next-token prediction needs.

### Why scale by `sqrt(d_k)` — derivation

Suppose Q and K entries are independent with mean 0 and variance 1. Then `Q·K = sum over d_k of q_i * k_i`. The variance of a sum of d_k zero-mean unit-variance products is d_k. So scores have standard deviation sqrt(d_k), and at d_k=64 that's 8. After softmax with std=8, the distribution is essentially one-hot — gradients almost zero. Dividing by sqrt(d_k) keeps std=1, where softmax has healthy gradients.

### Common mistakes / gotchas

1. **Forgetting to scale**: Without `1/sqrt(d_k)`, training is unstable at any reasonable d_k.
2. **Confusing the mask shape**: The mask is added to scores *before* softmax, not multiplied after. Adding -inf, then softmax, gives 0 weight; multiplying by 0 leaves an unrenormalized output.
3. **Mixing up Q@K.T vs K@Q.T**: The convention is `Q @ K.T` so that scores are `(n_query, n_key)`. The opposite gives `(n_key, n_query)` and breaks the rest of the math.

### How to say this in an interview

> "Self-attention is a learned soft-lookup. Each token produces three vectors — query, key, value — by linear projection. The query of every token is dot-producted against every other token's key, scaled by `1 over sqrt(d_k)` to keep variance bounded, then softmaxed to give a probability distribution over the sequence. The output for each token is the weighted sum of values under that distribution. Geometrically, a token gathers information from other tokens proportional to how similar their key is to its query. The reason this works so well is that any two tokens are exactly one hop apart in the computational graph, so long-range dependencies are easy. The cost is quadratic in sequence length, which is why FlashAttention and sliding window matter."

### Interview Q&A

**Q1. Why is attention scaled by `1/sqrt(d_k)`?**
> "Because the dot product of two d_k-dimensional vectors with unit-variance components has variance d_k. Without scaling, scores at large d_k have huge variance, and softmax saturates — meaning the maximum score becomes essentially 1 and the rest become 0, making gradients vanish. Dividing by `sqrt(d_k)` keeps the variance at 1, which keeps softmax in a regime with non-zero gradients. With d_k=64 you'd have std=8 unscaled — already enough to ruin training."

**Q2. Walk me through self-attention with shapes.**
> "Take input X with shape `batch by seq_len by d_model`, say `(8, 128, 512)`. We project to Q, K, V with three linear maps `W_Q`, `W_K`, `W_V`, each `512 by 64` for one head. So Q, K, V are each `(8, 128, 64)`. We compute `Q matmul K transpose` to get an attention matrix of shape `(8, 128, 128)` — that's the score for every query against every key. We divide by `sqrt(64) = 8`, optionally add a causal mask, softmax over the last dim, then matmul with V to get output `(8, 128, 64)`. Concatenate across heads and project back to `d_model=512`."

**Q3. What's the computational complexity of self-attention?**
> "It's `O(n^2 d)` for the attention matrix and `O(n d^2)` for the projections. The quadratic-in-n term is the dominant cost at long context — for a 32K context, that attention matrix is over a billion entries per layer per head. This is exactly why FlashAttention exists, why sliding-window attention exists, why GQA reduces KV-cache, and why linear-attention research is still active. Anything that touches long context fundamentally has to attack this n-squared term."

**Q4. Why three separate projections for Q, K, V?**
> "Because a token's query — what it's looking for — is conceptually different from what it offers as a key or what it returns as a value. Decoupling them gives the model the expressive freedom to act as both a 'looker' and a 'thing to be looked at' with different representations. If you tied Q=K=V, you'd basically be doing similarity-weighted averaging of the input — which is much weaker than what attention actually learns. Empirically, untying these is worth a few perplexity points."

**Q5. What does the softmax in attention do, intuitively?**
> "It converts raw similarity scores into a probability distribution over the keys, so the output is a convex combination of value vectors. This guarantees the output stays bounded and interpretable, and it forces the model to make 'choices' — putting most weight on a few keys rather than averaging everything. The temperature is implicit in `sqrt(d_k)` scaling: smaller scaling factor means peakier distribution, larger means flatter. The choice of softmax over linear blending is empirical — softmax-attention beats linear-attention variants on most benchmarks."

**Q6. What happens if you remove the causal mask in a decoder?**
> "You break the autoregressive factorization. Each position becomes able to attend to future tokens, which means at training time the model can cheat by looking at the answer. The loss collapses to near zero on training data and the model is useless for generation. The mask is what enforces `P(x_t given x_<t)` and makes next-token prediction a meaningful task. Without it you'd have an encoder, not a decoder."

---

## 2.3 Multi-head attention — running attention many ways at once

Single-head attention sees the sequence through one lens. **Multi-head attention** runs attention h times in parallel, each with smaller dimension d_k = d_model / h, then concatenates and projects.

### The block diagram

```
                    X (batch, n, d_model)
                       │
                  ┌────┼────┬────────┬────────┐
                  ▼    ▼    ▼        ▼        ▼
              Head 1  Head 2  Head 3 ... Head h    (each computes its own Q, K, V)
                  │    │    │        │        │
                  ▼    ▼    ▼        ▼        ▼
              attn_1 attn_2 attn_3  ...     attn_h  (each shape: batch × n × d_v)
                  │    │    │        │        │
                  └────┴────┴────────┴────────┘
                                │
                          ┌─────▼─────┐
                          │  Concat   │  (batch × n × d_model)
                          └─────┬─────┘
                                │
                          ┌─────▼─────┐
                          │  W_O      │  (final output projection)
                          └─────┬─────┘
                                │
                                ▼
                          Output (batch × n × d_model)
```

### Why multiple heads?

Each head learns a different "attention pattern." Empirical studies show:
- Some heads track syntax (subject-verb agreement, dependency arcs).
- Some track semantics (coreference: "the dog... it...").
- Some track positional structure (next-token, previous-token).
- Some are interpretable as induction heads (copy-paste patterns).

It's like having a committee of 32 specialist linguists, each looking at the sentence through a different lens, and combining their perspectives.

### Sizes for real models

| Model | d_model | n_heads | d_k | Layers | Total params |
|-------|---------|---------|-----|--------|--------------|
| GPT-2 small | 768 | 12 | 64 | 12 | 124M |
| GPT-3 175B | 12,288 | 96 | 128 | 96 | 175B |
| LLaMA-2 7B | 4,096 | 32 | 128 | 32 | 7B |
| LLaMA-2 70B | 8,192 | 64 | 128 | 80 | 70B |
| LLaMA-3 405B | 16,384 | 128 | 128 | 126 | 405B |

Notice d_k stays around 64 or 128 across all model sizes — that's the empirical sweet spot. Larger d_model is allocated to *more* heads, not bigger heads.

### The math compactly

```
MHA(X) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(X · W_Q^i, X · W_K^i, X · W_V^i)
```

Total params per attention layer ≈ 4 · d_model² (Q, K, V, O all are d_model × d_model when d_v = d_k = d_model / h).

### Common mistakes / gotchas

1. **Treating heads as independent models**: They share the same residual stream — head outputs are concatenated and projected back to d_model so they all influence the same downstream layer.
2. **Forgetting the output projection W_O**: Concat alone isn't enough; W_O mixes information across heads. Without it, each head would only contribute to its own slice of d_model.
3. **Picking too many heads**: With fixed d_model, more heads means each is thinner. d_k=8 is too narrow; d_k=64 or 128 is the empirical sweet spot.

### Interview Q&A

**Q1. Why do we split into multiple heads instead of one big attention?**
> "Because each head can specialize on different relationships. With one big d_model-dimensional attention, you have one set of Q, K, V mappings that has to capture everything — syntax, semantics, position, coreference. With multiple heads, each gets its own Q, K, V projection, and empirically heads specialize: some attend to previous-token, some to subject-verb, some to coreferents. You get richer representations at no extra parameter cost, since the per-head dimension shrinks proportionally."

**Q2. What's the role of the output projection W_O?**
> "After the heads compute their independent attention outputs, they're concatenated back into a d_model vector. W_O is the final linear map that mixes information across heads — it lets the model decide how much weight to give each head's output. Without W_O, each head would only contribute to its own slice of d_model, and downstream layers couldn't combine head outputs flexibly. W_O is also where most of the attention block's parameters live alongside Q, K, V."

**Q3. If I increase n_heads from 8 to 16 with fixed d_model, what changes?**
> "Each head's d_k drops from `d_model/8` to `d_model/16`, so each head sees a thinner subspace. Total compute and parameters stay basically the same. Whether quality improves depends on the task — sometimes more heads help (more specialization), sometimes hurt (heads too narrow to capture useful patterns). The empirical sweet spot is `d_k = 64` or `128`. So with `d_model=4096`, you'd want 32 heads of 128 or 64 heads of 64."

---

## 2.4 MHA → MQA → GQA → MLA — the KV-cache optimization saga

This is one of the highest-yield interview topics for senior roles. Let me walk through the entire arc.

### The problem MQA was created to solve

At inference, every layer caches K and V for every previously-generated token. For a 70B model at 8K context with full MHA:

```
KV cache size = 2 (K and V) × n_layers × n_heads × d_k × seq_len × dtype_bytes
              = 2 × 80 × 64 × 128 × 8192 × 2 bytes
              ≈ 21.5 GB per request
```

That's per single user request. Serve 10 concurrent users at 8K context and you've blown past 215 GB — well over an 80GB H100's capacity. Memory bandwidth becomes the binding constraint on throughput.

### The fix: reduce the number of K and V heads

```
                   Q heads        K/V heads      KV-cache size
   MHA   (classic)   N               N              1×           Best quality
   MQA   (PaLM)      N               1              1/N          -2-3% perplexity
   GQA   (LLaMA-2)   N               G (e.g., 8)    G/N          ~MHA quality
   MLA   (DeepSeek)  N               low-rank R     ~1/16        ~MHA quality
```

### Visual

```
MHA: 32 Q heads each pair with their own KV head     (1:1)
       Q1  Q2  ... Q32
       K1  K2  ... K32
       V1  V2  ... V32

MQA: 32 Q heads share 1 KV pair                       (32:1)
       Q1  Q2  ... Q32
                K (shared)
                V (shared)

GQA: 32 Q heads in 8 groups of 4 share 1 KV per group (4:1)
       Q1 Q2 Q3 Q4   Q5 Q6 Q7 Q8 ... 
        K1, V1         K2, V2     ... K8, V8

MLA: K and V are compressed into a shared low-rank latent vector,
     decompressed on the fly via learned up-projections
```

### The tradeoff curve

- MHA: 100% quality, 100% KV-cache size — old default.
- MQA: ~97% quality, ~3% KV-cache size — too aggressive for most production.
- GQA: ~99-100% quality, 12-25% KV-cache size — sweet spot.
- MLA: 100% quality, ~7% KV-cache size — current SOTA, used by DeepSeek-V2/V3.

LLaMA-2, LLaMA-3, Mistral, Qwen all use GQA with 4-8 KV groups. DeepSeek-V3 uses MLA with decoupled RoPE.

### Worked example with numbers

LLaMA-3 70B: 64 query heads, 8 KV groups (so GQA ratio 8:1), d_k=128, 80 layers.

- KV-cache per token = `2 × 80 × 8 × 128 × 2 bytes = 327 KB`.
- At 8K context: `327 KB × 8192 tokens = 2.7 GB per user`.
- vs MHA equivalent: `2 × 80 × 64 × 128 × 2 bytes = 2.6 MB per token = 21 GB at 8K`.

So GQA cuts KV-cache by 8x, letting you serve 8x more concurrent users on the same GPU.

### Why MLA is the current frontier

MLA learns a low-rank latent representation that K and V are reconstructed from. The cache stores just the latent (say rank 512), not the full K and V (typically 8192-dim total). At attention time, K and V are decompressed via learned up-projections. Combined with a "decoupled RoPE" trick to handle position, it achieves MHA quality at ~1/16 the KV cache. DeepSeek-V2 paper has the cleanest description.

### Common mistakes / gotchas

1. **Confusing MQA and GQA**: MQA has *one* shared K/V across all Q heads. GQA has *groups* — multiple shared K/Vs.
2. **Thinking GQA changes training cost much**: It doesn't — savings are at inference, where KV-cache size is the bottleneck.
3. **Overestimating quality loss of MQA**: 2-3% perplexity sounds bad but for most tasks it's invisible. The reason GQA won is that 8x cache savings with no quality loss is just better.

### Interview Q&A

**Q1. Why do production LLMs use GQA instead of MHA?**
> "KV-cache size. At inference each layer caches K and V for every prior token, and the cache scales with `2 × n_layers × n_heads × d_k × seq_len`. For a 70B model at 8K context, MHA gives ~21 GB of KV-cache per user — you can serve maybe 3 users on an 80GB H100. GQA reduces this 4-8x by having Q heads share K/V groups. LLaMA-2 70B uses 8:1 (64 Q heads, 8 KV groups), giving ~2.7 GB per user — you can serve 25+ users. The quality loss is essentially zero on standard benchmarks. So GQA is just strictly better economically."

**Q2. What's the difference between MQA and GQA?**
> "MQA — multi-query attention — has one shared K and V across all Q heads. Maximum compression, but quality drops 2-3% perplexity. GQA — grouped-query attention — has multiple shared K/Vs in groups. With 64 Q heads and 8 KV groups, every group of 8 Q heads shares one K and V. Quality is essentially MHA but cache is 8x smaller. GQA is the production sweet spot; MQA was an interesting waypoint."

**Q3. What is MLA in DeepSeek-V3?**
> "Multi-head Latent Attention. Instead of storing K and V at full dimension, MLA learns a low-rank shared latent vector — say 512 dimensions — and stores just that in the cache. At attention time, learned up-projections decompress to the full K and V tensors. There's a clever decoupling of the position embedding (RoPE part vs content part) so that the latent doesn't have to encode position. The result is ~93% KV-cache reduction vs MHA with no quality loss. DeepSeek-V2 and V3 both use MLA, and several other frontier labs are adopting it."

**Q4. Does GQA save training compute?**
> "Almost no. Training computes the full attention matrix per layer per head, and that's dominated by `Q matmul K transpose` which scales with the number of *query* heads, not KV heads. The K/V projection is a tiny fraction of total FLOPs. Savings are entirely at inference, where the bottleneck is streaming weights and KV-cache from HBM to compute units. GQA shrinks KV-cache, which directly increases the number of users you can serve in parallel. So pick GQA for inference economics, accept zero training cost."

---

## 2.5 Positional encoding — telling the transformer where things are

Pure attention is permutation-invariant. Without positional information, "the dog bit the man" and "the man bit the dog" produce identical token-level outputs (just with rearranged positions). We have to inject position somewhere.

### The four big ideas, in chronological order

1. **Sinusoidal (Vaswani 2017)**: Add fixed sin/cos signals to embeddings.
2. **Learned absolute (BERT, GPT-2)**: One learnable vector per position.
3. **RoPE (RoFormer 2021)**: Rotate Q and K in 2D subspaces by an angle proportional to position.
4. **ALiBi (2022)**: Don't add anything to embeddings; bias attention scores by distance.

### 2.5.1 Sinusoidal positional encoding

The original. For position `pos` and embedding dimension index `i`:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Plain English: each pair of dimensions oscillates at a different frequency, ranging from very-fast at the lowest dimension index to very-slow at the highest. Together they form a unique, deterministic "fingerprint" for each position. Added directly to the token embedding before the first attention block.

The clever property: `PE(pos+k)` is a fixed linear function of `PE(pos)`, which means the model can in principle learn relative positions from these absolute signals. In practice, BERT and GPT-2 found learned absolute PE worked just as well, and the field switched.

### 2.5.2 Learned absolute PE

Just initialize a `(max_seq_len, d_model)` table and learn it. Simpler. The catch: at inference you can't go beyond `max_seq_len` because there's no learned vector for those positions. Sinusoidal can extrapolate; learned absolute cannot.

### 2.5.3 RoPE — Rotary Position Embedding

This is the modern default. Used in LLaMA, Mistral, Qwen, Gemma, DeepSeek, Claude. It's the most important PE method to understand.

### Mental model

Imagine each pair of dimensions (2i, 2i+1) of Q and K as a 2D vector. We rotate that 2D vector by an angle proportional to the token's position. Because rotations are orthogonal transformations, they preserve vector magnitudes. And — this is the magic — the inner product of two rotated vectors depends only on the *difference* of rotation angles, i.e., the relative position.

### The math

For each pair of dims (2i, 2i+1), define rotation angle:
```
θ_i = 10000^(-2i/d)   (a frequency, like in sinusoidal)
```

For a token at position `pos`, the rotation matrix for that pair is:
```
R(pos, i) = [ cos(pos · θ_i)  -sin(pos · θ_i) ]
            [ sin(pos · θ_i)   cos(pos · θ_i) ]
```

Apply this rotation to both Q and K:
```
Q'_i = R(pos_q, i) · Q_i
K'_i = R(pos_k, i) · K_i
```

The dot product `Q'_i · K'_i` simplifies (after some trig) to a function of `(pos_q - pos_k)` only. So attention scores depend on relative position.

### Why RoPE wins

1. **Relative position natively**: The model gets relative-position bias for free, without learning it.
2. **No extra parameters**: Rotation matrices are determined by frequency formulas, not learned.
3. **Extrapolates well**: With YaRN/NTK scaling (below), you can extend a 4K-trained model to 32K-128K with minimal fine-tuning.
4. **Plays nicely with FlashAttention**: Apply rotation to Q and K before the attention kernel.
5. **Norms are preserved**: Rotations don't change vector magnitudes, so other parts of the architecture don't need to change.

### Block diagram of RoPE

```
                              Standard attention with RoPE
        X
        │
   ┌────▼────┐
   │ W_Q, W_K│
   └────┬────┘
        │
   Q, K = X·W_Q, X·W_K
        │
   ┌────▼─────────────────┐
   │ Rotate Q, K by RoPE  │   ← Q'_i = R(pos, i)·Q_i
   │ at each position     │     K'_i = R(pos, i)·K_i
   └────┬─────────────────┘
        │
   ┌────▼────┐
   │ Q' · K'^T / √d_k  │   ← inner product depends only on (pos_q - pos_k)
   └────┬────┘
        │
   softmax → attention weights → · V
```

### 2.5.4 RoPE scaling — YaRN, NTK, dynamic NTK

Vanilla RoPE breaks if you train at 4K and infer at 32K — the high-frequency rotations wrap around too fast at unseen positions, and the model produces garbage.

Three fixes, in order of sophistication:

- **Position interpolation (Meta)**: Compress positions by `train_len / target_len`. Cheap, requires light fine-tuning.
- **NTK-aware scaling**: Stretch low frequencies more than high frequencies (low frequencies need to keep their precision; high ones can absorb the extension). Better quality with no fine-tuning.
- **YaRN** (Peng 2023): NTK + temperature correction + ramp function. The current SOTA, extends 4K to 32K-128K with minimal fine-tuning. Default in many open-source long-context models.
- **Dynamic NTK**: Apply scaling only beyond the training length. Smooth handoff.

### 2.5.5 ALiBi — bias instead of embedding

Instead of injecting position into Q and K, ALiBi adds a fixed bias to the attention scores:

```
attention_score(i, j) += -m · |i - j|
```

where m is a head-specific slope. Tokens far apart get a more negative bias; the softmax pushes their weights down. No positional embedding at all — just a linear distance penalty.

- Pros: Zero extra params, extrapolates to arbitrary length without fine-tuning.
- Cons: Slightly weaker on tasks requiring fine-grained relative-position reasoning.
- Used in: BLOOM, MPT.

### How to say this in an interview

> "Pure attention is permutation-invariant — without positional information, 'the dog bit the man' would have the same representation as 'the man bit the dog.' The original transformer used additive sinusoidal encodings. BERT/GPT-2 switched to learned absolute. The modern dominant choice is RoPE — Rotary Position Embedding. It rotates each pair of dimensions in Q and K by an angle proportional to the token's position. The key property is that the inner product of two rotated vectors depends only on the relative position, so the model gets relative-position bias for free, with no extra parameters. RoPE also extrapolates better than learned PE — with YaRN scaling, you can train on 4K context and extend to 128K. LLaMA, Mistral, Qwen, Gemma all use it."

### Interview Q&A

**Q1. Why use RoPE instead of sinusoidal?**
> "Sinusoidal PE is added to the embedding before any layer, so the model has to learn from scratch how to extract relative-position information from those absolute signals. RoPE rotates Q and K directly, and the rotation algebra makes the inner product of two rotated vectors depend only on the *difference* of positions. So relative position is built into attention scores by construction. RoPE also has zero extra parameters, extrapolates better with YaRN scaling, and integrates cleanly with FlashAttention because you can rotate Q and K before the attention kernel."

**Q2. Explain RoPE in plain English.**
> "Imagine each pair of dimensions in Q and K as a 2D vector. RoPE rotates that 2D vector by an angle proportional to the token's position — at higher frequencies for low-index dimensions, slower frequencies for high-index ones, like a mix of clocks ticking at different rates. Because rotation preserves vector magnitudes, nothing else in the model has to compensate. And because the dot product of two rotated vectors depends on the difference of angles, attention naturally captures relative position. The whole thing is parameter-free — frequencies come from the same `10000^(-2i/d)` formula as sinusoidal."

**Q3. What is YaRN?**
> "Yet another RoPE extensioN. The problem it solves: vanilla RoPE breaks if you train at 4K context and infer at 32K, because the high-frequency rotation dimensions wrap around too fast at unseen positions. YaRN fixes this with three pieces: NTK-aware scaling that stretches low frequencies more than high (low frequencies don't need fine-grained precision; they can absorb the extension), a temperature correction to keep softmax in the right regime, and a ramp function for smooth handoff. With minimal fine-tuning — sometimes none — YaRN extends models to 4-32x their training context length."

**Q4. ALiBi vs RoPE.**
> "ALiBi adds a fixed linear bias proportional to distance directly to attention scores — no positional embedding, no rotation. Tokens far apart get a more negative bias and lower attention weights. Zero parameters, extrapolates trivially to any length. Quality is slightly worse than RoPE on benchmarks that require precise relative-position reasoning. Used in BLOOM and MPT but RoPE is the production default in 2026 — better quality plus YaRN scaling handles the long-context extrapolation."

**Q5. What happens if you remove positional encoding entirely?**
> "Attention becomes permutation-invariant. The model can't tell word order, so 'dog bites man' and 'man bites dog' produce the same token-level representations, just permuted. For decoder-only models with causal masks, some position information leaks through the mask itself — position 5 can see positions 0 to 4, position 6 can see 0 to 5, so the model can in principle infer position from 'who I'm allowed to attend to' — but this is a weak signal and quality degrades significantly. Always include explicit PE in production."

---

## 2.6 FlashAttention — same math, dramatically less memory

Standard attention materializes the full `n × n` attention matrix in HBM (high-bandwidth GPU memory). For a 32K context, that's a billion entries per layer per head — a big chunk of even an H100's 80 GB. Tri Dao's FlashAttention (2022) showed you don't need to materialize it.

### The core insight — memory is the bottleneck, not compute

GPUs have two memory tiers: HBM (large, slow, ~3 TB/s on H100) and SRAM (small, fast, ~20 TB/s, but only ~200 KB per SM). Standard attention computes `Q @ K.T`, writes the n×n attention matrix to HBM, reads it back to compute softmax, writes again, reads to multiply by V. Each round-trip to HBM is expensive.

### What FlashAttention does

```
for each block of Q (small enough to fit in SRAM):
    initialize block output to zero
    initialize running softmax stats (max, sum)
    for each block of K, V:
        load K_block, V_block into SRAM
        compute partial attention: scores = Q_block @ K_block.T / sqrt(d_k)
        update block output using running softmax (online algorithm)
    write final block output to HBM
```

The key trick is the **online softmax algorithm**, which lets you compute softmax incrementally over chunks while keeping running max and running sum statistics. The math is the same — bit-for-bit identical to standard attention — but the matrix is never fully materialized.

### Block diagram

```
HBM:                                        SRAM (fast, small):
┌──────────────────┐                        ┌─────────────────┐
│ Q (n × d)        │ ─── load Q block ────▶ │ Q_block (b × d) │
│ K (n × d)        │ ─── load K block ────▶ │ K_block (b × d) │
│ V (n × d)        │ ─── load V block ────▶ │ V_block (b × d) │
│                  │                        │                 │
│                  │                        │  online softmax │
│                  │                        │  + matmul       │
│                  │                        │                 │
│ Output (n × d)   │ ◀─── write block ───── │ O_block (b × d) │
└──────────────────┘                        └─────────────────┘

(no n × n matrix ever lives in HBM)
```

### Why this is so much faster

- 2-4x faster wall-clock on long context.
- 5-20x less memory at long sequence (you never allocate the n×n tensor).
- Enabled 8K → 128K contexts on commodity hardware (A100, H100).

### The evolution

- **FlashAttention-1** (2022): The original. Tiling + online softmax.
- **FlashAttention-2** (2023): Better work partitioning across warps and threadblocks. Roughly 2x speedup over FA-1.
- **FlashAttention-3** (2024): Hopper-architecture (H100) async pipelining, FP8 tensor core support, better TMA usage. Another ~2x on H100.

### Practical use

You almost never write FlashAttention from scratch. PyTorch 2+ ships `torch.nn.functional.scaled_dot_product_attention` which auto-selects the FA implementation. vLLM and TensorRT-LLM both integrate it. xFormers and FlashInfer have alternative implementations.

### Common mistakes / gotchas

1. **Thinking FA changes the math**: It doesn't — outputs are bit-identical to standard attention. It's a memory-IO optimization.
2. **Assuming all FA versions support all features**: FA-1 didn't have causal mask + dropout combined; FA-2 and FA-3 do. Check the version's matrix-of-supported-ops.
3. **Mixing FA with custom mask shapes**: FA assumes a regular causal or no-mask pattern. Custom masks (block-sparse, etc.) may need FlashInfer or a fallback.

### Interview Q&A

**Q1. Why is FlashAttention faster?**
> "It's IO-aware. Standard attention writes the n-by-n attention matrix to HBM, reads it back to softmax, writes again, then reads to multiply by V — many round-trips between HBM and SRAM. FlashAttention tiles Q, K, V into blocks that fit in SRAM, fuses the softmax and matmul into one operation, and uses an online softmax algorithm so it never has to materialize the full attention matrix. The math is identical — outputs are bit-for-bit the same — but you get 2-4x speedup and 5-20x less memory at long context. That's what enabled 8K-to-128K context windows on commodity GPUs."

**Q2. What does FlashAttention-3 add over FlashAttention-2?**
> "Hopper-specific optimizations. H100 has async pipelining via CUDA's TMA — Tensor Memory Accelerator — that lets you overlap memory copies with compute. FA-3 leverages this to hide more latency. It also adds native FP8 tensor core support, which roughly doubles throughput at small accuracy cost. End-to-end on H100, FA-3 is about 2x faster than FA-2. PyTorch 2.4+ and vLLM 0.6+ ship FA-3 by default on supported hardware."

**Q3. Walk me through the online softmax trick.**
> "Standard softmax is `exp(x_i - max(x)) / sum_j exp(x_j - max(x))` — you need the global max and global sum to normalize. In an online setting, you process one block at a time. Maintain a running max and running sum. When a new block arrives with its own block-max and block-sum, you adjust the running stats: rescale the running sum by `exp(old_max - new_max)` and add the block-sum. The output of attention is `weights @ V`, which can be similarly accumulated as a weighted sum across blocks. End result: same numbers as standard softmax, never materializing the full distribution. This is the heart of FlashAttention."

**Q4. When would you not use FlashAttention?**
> "If you have an exotic attention pattern not supported — block-sparse, varying mask per query position, or a custom kernel that fuses extra ops. Also for very short sequences, FlashAttention's overhead dominates and standard attention is competitive. But for any modern transformer with seq_len greater than 512 and a standard or causal mask, FlashAttention is the right choice and PyTorch's `scaled_dot_product_attention` will pick it automatically."

---

## 2.7 The feed-forward network — where the model actually stores knowledge

After attention mixes information across positions, each token passes through a position-wise MLP — the feed-forward network or FFN. This is where most of the model's parameters live (about 2/3 of the weights).

### The classic FFN

```
FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2
         ┌──────┬──────┐         └──┬──┘
         expand to 4d              project back to d
```

For d_model=512, the hidden dimension is typically 4 * 512 = 2048. Two linear layers with ReLU (or GELU) in between. Applied independently per token.

### Why 4×?

Empirical sweet spot. Smaller (2x) underfits; larger (8x) doesn't help much. The Vaswani 2017 paper used 4x and it stuck. Probing research (Geva et al. 2021) shows the FFN behaves like a key-value memory: each row of W_1 is a "pattern detector," and the corresponding column of W_2 is the "value" emitted when that pattern fires.

### Modern upgrade — SwiGLU

LLaMA, Mistral, PaLM, Gemma all use SwiGLU:

```
SwiGLU(x) = (Swish(x · W_gate)) ⊙ (x · W_up) · W_down
            ┌───────────┬──────┘ ┌────┬────┘   ┌────┬────┘
            gate branch          up branch     output projection
            (sigmoid-ish gate)   (linear)
```

Three weight matrices instead of two. The gate branch produces a per-position scalar gate (after Swish nonlinearity), elementwise-multiplied with the up branch. To keep parameter count constant, hidden size shrinks from `4d` to roughly `(2/3) * 4d ≈ 8/3 * d`.

### Why SwiGLU helps

The gate branch lets the model dynamically decide, per token, how much of the up-branch's information to let through. It's a learned per-position attention over the FFN's internal computations. Empirically worth 1-2 perplexity points over GELU.

### Block diagram

```
                Standard FFN                      SwiGLU FFN
                
    x  ──────▶ W_1 ──▶ ReLU/GELU         x  ──────┬──▶ W_gate ──▶ Swish ──┐
                          │                       │                          ⊙
                          ▼                       └──▶ W_up    ────────────┘
                         W_2                                   │
                          │                                    ▼
                          ▼                                   W_down
                       output                                  │
                                                               ▼
                                                            output
```

### Why FFN holds knowledge

A 70B model has roughly 50B parameters in FFN weights. Probing studies show that factual associations ("Paris is the capital of France") are stored in specific FFN neurons. Editing tools like ROME and MEMIT specifically target FFN weights to inject or update facts. Attention is the routing layer; FFN is the storage.

### Common mistakes / gotchas

1. **Confusing FFN expansion ratio**: 4x is for ReLU/GELU FFNs. SwiGLU uses ~8/3x to keep parameter count the same.
2. **Forgetting position-independence**: FFN is applied identically and independently per token. No cross-token interaction here — that's attention's job.
3. **Underestimating FFN parameter cost**: Per layer, FFN has roughly twice as many parameters as attention (8d² vs 4d²). It dominates the parameter count of the whole model.

### Interview Q&A

**Q1. What does the FFN do in a transformer?**
> "The FFN is a position-wise MLP applied independently to every token after attention. It expands the d_model representation to roughly 4 times d_model, applies a nonlinearity, then projects back. Empirically, the FFN is where most factual knowledge is stored — probing research shows it acts as a key-value memory, with each row of the first matrix being a pattern detector and the corresponding column of the second matrix being the value emitted. About two-thirds of a transformer's parameters live here."

**Q2. Why does SwiGLU beat plain GELU?**
> "SwiGLU adds a gating mechanism. Instead of just `GELU(xW_1) W_2`, you compute `Swish(xW_gate) elementwise* (xW_up)` and project with W_down. The gate branch lets the model dynamically scale information per token — it's a learned, per-position multiplicative attention over the FFN's internal channels. Empirically worth one to two perplexity points. The cost is a third weight matrix, so to keep parameter count the same we shrink the hidden size from 4d to about 8/3 d. PaLM, LLaMA, Mistral, and Gemma all use SwiGLU."

**Q3. Why is FFN 4 times d_model wide?**
> "Empirical. Vaswani 2017 picked 4x and it works well. Smaller widths underfit; larger widths don't help proportionally. With SwiGLU we shrink to about 8/3 d to keep parameter count constant given the extra gate matrix. There's no deep theoretical reason for exactly 4x — it's a hyperparameter that's been measured across many model scales and stuck. Some recent papers explore non-uniform widths per layer but production models still use ~4x or 8/3x uniformly."

---

## 2.8 Encoder-only vs decoder-only vs encoder-decoder — the three architectures

| | Encoder-only (BERT) | Decoder-only (GPT) | Encoder-Decoder (T5) |
|---|---------------------|--------------------|-----------------------|
| Attention | Bidirectional | Causal | Enc: bidir, Dec: causal + cross |
| Pretraining | MLM (mask 15%) | Next-token | Span corruption / denoising |
| Output | Hidden state per token | Next-token probs | Full target sequence |
| Best for | Classification, NER, embeddings | Generation, chat, reasoning | Translation, summarization |

### Why decoder-only won for LLMs

In 2018-2020 it wasn't obvious which would win. T5 had encoder-decoder elegance for seq-to-seq tasks. BERT was great for classification. But three things tipped the scales for decoder-only:

1. **Training signal density**: Every position predicts the next token, so a single sequence gives you hundreds or thousands of training examples. MLM masks only 15% of tokens, giving 6-7x less signal per sequence.
2. **Task unification**: Every NLP task can be cast as "generate the answer given the input." Classification, QA, summarization, translation — all fit. Encoder-only or encoder-decoder need task-specific heads.
3. **Scaling beauty**: GPT-3 (2020) showed pure decoder-only matches or beats encoder-decoder at scale, even on tasks like translation that "should" need encoders.

So today: GPT, LLaMA, Claude, Gemini, Mistral, Qwen, DeepSeek — all decoder-only. BERT-family lives on as embedding models and classification heads. T5/BART-family is niche.

### Block diagrams

```
ENCODER-ONLY (BERT):                  DECODER-ONLY (GPT, LLaMA):
                                      
   tokens                                tokens
     │                                     │
   embed                                 embed
     │                                     │
  ┌──▼──┐                              ┌──▼──┐
  │bidir│  attention                   │causal│ attention
  │attn │                              │attn  │
  └──┬──┘                              └──┬──┘
     │ (every pos sees all)               │ (pos sees only past)
     ▼                                     ▼
   FFN                                   FFN
     │                                     │
   (× N)                                 (× N)
     │                                     │
   hidden states                        next-token logits
```

```
ENCODER-DECODER (T5, BART):

   source tokens                   target tokens (shifted right)
     │                                  │
   embed + PE                        embed + PE
     │                                  │
   ENCODER (bidir)                   DECODER (causal)
     │  ┌─────────────────────────┐    │
     │  │ Cross-attention from    │    │
     │  │ decoder Q to encoder    │ ───┘
     │  │ K, V                    │
     │  └─────────────────────────┘
     ▼                                  ▼
   final encoder states           next-token logits
```

### Common mistakes / gotchas

1. **Calling Claude or GPT "encoder-decoder"**: They're decoder-only. The "encoder" in chat context is just the prefill phase, not a separate architecture component.
2. **Assuming MLM is dead**: It's not — embedding models still use MLM-pretrained encoders (BERT, RoBERTa) as backbones and contrastively fine-tune.
3. **Thinking encoder-only can generate**: It can't, autoregressively. You can do mask-filling or use a separate decoder, but BERT alone won't generate fluent text.

### Interview Q&A

**Q1. Encoder-only vs decoder-only — which to use for what?**
> "Encoder-only models like BERT have bidirectional attention — every position sees every other. They're trained with masked language modeling and produce a hidden state per token. Best for classification, NER, sentence embeddings, where you need to understand a fixed input but not generate. Decoder-only like GPT or LLaMA has causal attention — each position sees only past tokens. Trained with next-token prediction. Best for generation, chat, reasoning, agents. Today's general-purpose LLMs are all decoder-only because next-token prediction gives a denser training signal and unifies all tasks as generation."

**Q2. Why did decoder-only win over encoder-decoder for LLMs?**
> "Three reasons. First, training signal density: every position predicts the next token, giving you hundreds of supervisory signals per sequence, vs MLM's 15% or encoder-decoder's per-target-token. Second, task unification: classification, summarization, translation, chat all fit as 'generate the answer given input,' so no task-specific architecture changes. Third, scaling: GPT-3 showed pure decoder-only matches or beats encoder-decoder on every benchmark at scale, including translation. Simplicity wins, and you can always condition the decoder on input via the prompt."

**Q3. Where do encoder models live in 2026?**
> "As embedding models. BERT and RoBERTa backbones, contrastively fine-tuned, power retrieval and search — sentence-BERT, BGE, E5, all encoder-only. Also as classification heads in production NLP — sentiment, intent, NER. Encoders are still the right choice when the task is 'understand fixed input' and there's no generation. They're cheaper at inference (one forward pass, no autoregressive loop) and produce fixed-shape outputs that downstream pipelines can consume."

---

## 2.9 Compute and memory — back-of-envelope

### Training FLOPs

- Forward FLOPs ≈ `2 × params × tokens`
- Forward + backward (training) ≈ `6 × params × tokens`

Example: train 70B on 2T tokens → `6 × 70e9 × 2e12 = 8.4 × 10^22` FLOPs. At 1 PFLOP per H100 sustained, that's ~10,000 H100-days, or 416 H100-years. Cluster of 10K H100s: 17 days.

### Inference — prefill vs decode

```
Prompt        ────────▶  PREFILL (compute-bound)
"What is..."             - Process all prompt tokens in parallel
                         - GPU runs at peak utilization
                         - Latency: ~ms per K tokens
                                     │
                                     ▼
                         First output token

                                     ▼
"The answer  ────────▶  DECODE (memory-bandwidth-bound)
 is..."                  - One token at a time
                         - Stream weights from HBM per token
                         - Latency: ~tens of ms per token
```

This dichotomy is critical. Prefill is compute-bound — saturates tensor cores. Decode is memory-bandwidth-bound — every token requires streaming the entire model from HBM. Decode is why max-batching matters: amortizing the weight transfer over many concurrent requests is the only way to get decent throughput.

### KV-cache size — the formula to memorize

```
KV cache (bytes) = 2 (K and V) × n_layers × n_kv_heads × d_k × seq_len × dtype_bytes
```

LLaMA-3 70B at 8K context, GQA with 8 KV heads:
```
2 × 80 × 8 × 128 × 8192 × 2 = 2.7 GB per request
```

LLaMA-3 70B with full MHA at 8K context:
```
2 × 80 × 64 × 128 × 8192 × 2 = 21 GB per request  ← infeasible at scale
```

GQA gives ~8x cache savings. MLA (DeepSeek) gives ~16x.

### Resume tie-in

> When I built the XGBoost lambda for TrueBalance with p99 < 500ms, the bottleneck wasn't compute — it was cold-start of the lambda container. For LLM serving the bottleneck is fundamentally different: prefill is compute-bound and decode is memory-bandwidth-bound. The latency mindset is the same — measure p50, p95, p99 — but the optimization levers are completely different. KV-cache management, paged attention, continuous batching, prefix caching — that's the LLM serving toolkit, and it maps to the careful resource management we did at TrueBalance for cold-start mitigation.

### Interview Q&A

**Q1. Why is decode memory-bandwidth-bound?**
> "Each decode step processes one token but has to stream all model weights and all KV-cache from HBM to compute units. For a 70B model in BF16, that's 140 GB of weights plus several GB of KV-cache, every single token. Compute per token is tiny by comparison — a few hundred GFLOPs against 2-3 TB/s of memory bandwidth. So you're bandwidth-bound: wall time per token is roughly `weights_size / bandwidth`. The only way to amortize this is max-batching — serve N concurrent requests and you stream weights once for N tokens, multiplying throughput. That's why vLLM's continuous batching is so impactful."

**Q2. Why does prefill have a different performance profile?**
> "Prefill processes all prompt tokens in parallel — say 1000 prompt tokens in one forward pass per layer. That's a giant matmul that saturates tensor cores, so it's compute-bound. Wall time scales roughly with prompt length and model size, but the GPU runs near peak utilization. Decode, by contrast, is sequential — one token at a time — so each forward pass is a thin matmul that under-utilizes compute and just streams weights. The transition from prefill to decode is the moment latency characteristics flip from compute-bound to bandwidth-bound."

**Q3. Estimate KV-cache size for a 70B model at 32K context with GQA.**
> "Assume 80 layers, 8 KV heads, d_k=128, FP16. The cache size per token is 2 (K and V) times 80 layers times 8 heads times 128 dim times 2 bytes, which is roughly 327 KB per token. At 32K tokens, that's about 10.5 GB per request. With FP8 KV cache it's 5 GB. So on an 80 GB H100 you can fit weights — say 35 GB of model — plus 6-8 long-context users in parallel. That's the kind of capacity-planning math you need to do for any production deployment."

---

## 2.10 Interview Q&A — Transformer (synthesis)

> 30+ high-probability questions, drawn from real interview banks. The "Gotcha" tag means it's a trap — get it right and you sound senior.

**Q1. Why scale by `sqrt(d_k)` in attention?**
> "Because the dot product of two d_k-dimensional vectors with unit-variance entries has variance d_k, so unscaled scores have standard deviation `sqrt(d_k)`. Without scaling, softmax saturates at large d_k and gradients vanish. Dividing by `sqrt(d_k)` keeps the variance at 1, where softmax is well-behaved."

**Q2. Compute and parameter complexity of self-attention?**
> "Compute is `O(n^2 d)` from the attention matrix and `O(n d^2)` from the projections. Parameters per attention layer are about `4 d^2` — Q, K, V, and output projections each `d × d`. The quadratic-in-n cost is the bottleneck at long context, motivating FlashAttention, sliding window, and linear-attention variants."

**Q3. MHA vs MQA vs GQA — what does each save?**
> "MHA has `N` Q heads and `N` KV heads. MQA has `N` Q heads but only 1 shared KV pair, cutting KV-cache by N at a 2-3% perplexity cost. GQA has `N` Q heads in `G` groups, each sharing a KV pair, giving `N/G` cache savings with essentially no quality loss. LLaMA-2/3, Mistral, Qwen, DeepSeek-V1 use GQA with G between 4 and 8. Production default."

**Q4. What is the KV-cache and why does it exist?**
> "During autoregressive decoding, every new token attends to all previous tokens. Without caching, we'd recompute K and V for every prior token at every step — `O(n^2)` per layer per generated token. The KV-cache stores K and V for all prior positions, so the new token only needs to compute its own Q, K, V and attend to the cache. New-token decode becomes `O(n)` per layer. The cost is memory: cache size grows linearly with context length and number of concurrent users."

**Q5. Why doesn't the KV-cache exist during training?**
> "Training processes the entire sequence in parallel via teacher forcing. There's no autoregressive loop — all positions get their K, Q, V computed in one shot per layer per batch. Caching saves nothing in that regime. KV-cache is purely an inference-time optimization."

**Q6. Why RoPE instead of sinusoidal PE?**
> "Sinusoidal PE is added to the embedding before any attention block — the model has to learn from scratch that positions encode relative distance. RoPE rotates Q and K by an angle proportional to position, and the inner product of two rotated vectors depends only on the relative position by construction. So relative-position bias is built-in. RoPE also extrapolates better with YaRN scaling, has zero parameters, and integrates cleanly with FlashAttention."

**Q7. Explain YaRN.**
> "Yet another RoPE extensioN. Vanilla RoPE breaks if you train at 4K and infer at 32K because high-frequency rotation dimensions wrap around too fast. YaRN scales the RoPE frequencies non-uniformly — low frequencies stretched more than high — adds a temperature correction to keep softmax in the right regime, and uses a ramp function for smooth handoff. Result: 4-32x context extension with minimal fine-tuning."

**Q8. ALiBi vs RoPE.**
> "ALiBi adds a fixed linear bias proportional to distance directly to attention scores — no positional embedding. Zero parameters, extrapolates trivially. RoPE rotates Q and K, gets relative-position bias by construction, needs YaRN for very long context. RoPE has higher quality on benchmarks; ALiBi is simpler. Production LLMs in 2026 use RoPE — LLaMA, Mistral, Qwen, Gemma. ALiBi is in BLOOM and MPT."

**Q9. Why LayerNorm instead of BatchNorm in transformers?**
> "Sequences are variable length, padding skews batch statistics, and inference is often batch=1. LayerNorm normalizes per-token across features, sidestepping all of these. RMSNorm goes further — drops the mean-centering step for a 7-10% speedup with no quality loss. LLaMA, Mistral, Qwen, Gemma all use RMSNorm."

**Q10. Pre-LN vs Post-LN?**
> "Pre-LN places the norm inside the residual block, before attention or FFN — gradient has a clean residual path that doesn't go through the norm. Stable for deep stacks. Post-LN places the norm after the residual addition — slightly higher quality when stable, but needs LR warmup or careful init to avoid divergence. Modern LLMs universally use Pre-LN."

**Q11. What does the FFN actually do?**
> "Two-layer position-wise MLP — expand to ~4d, apply nonlinearity, project back to d. Applied independently per token. Empirically the FFN behaves as a key-value memory: each row of W_1 is a pattern detector, the corresponding column of W_2 is the value emitted when that pattern fires. Most factual knowledge in an LLM lives in FFN weights. Modern LLMs use SwiGLU instead of plain ReLU/GELU."

**Q12. How does FlashAttention work and why is it faster?**
> "It's IO-aware. Standard attention writes the n-by-n attention matrix to HBM, reads it back to softmax, writes again, multiplies by V. Many round-trips. FlashAttention tiles Q, K, V into blocks that fit in SRAM, fuses softmax and matmul, uses an online softmax algorithm to never materialize the full matrix. Same math, 2-4x faster, 5-20x less memory at long context. FA-1 was the original, FA-2 improved work partitioning, FA-3 adds Hopper async pipelining and FP8."

**Q13. What is causal masking and why?**
> "Add `-infinity` to attention scores at positions where the query shouldn't see the key — specifically all positions later than the query's position in a decoder. After softmax, those positions get zero weight, so each token only attends to itself and earlier tokens. This enforces the autoregressive factorization `P(x_t given x_<t)` and makes next-token prediction a meaningful training task. Remove it and you get a bidirectional encoder that can't generate."

**Q14. Encoder-only vs decoder-only — which for what?**
> "Encoder-only like BERT — bidirectional attention, MLM pretraining — best for classification, NER, embeddings. Decoder-only like GPT or LLaMA — causal attention, next-token pretraining — best for generation, chat, reasoning. Encoder-decoder like T5 — best for strict input-to-output tasks like translation. Modern general-purpose LLMs are decoder-only because next-token gives denser signal and unifies all tasks as generation."

**Q15. [Gotcha] In GQA with 32 Q-heads and 8 KV-heads, do training FLOPs change vs MHA?**
> "Almost not at all. Training FLOPs are dominated by `Q matmul K transpose` and `attention matmul V`, which scale with the number of *query* heads, not KV heads. The KV projection is a tiny fraction of total compute. GQA's savings are entirely at inference, where KV-cache size dominates memory and bandwidth. So you get 4-8x more concurrent users at inference for essentially the same training cost."

**Q16. [Gotcha] Your transformer's loss plateaus after a few thousand steps. What do you check?**
> "First, learning rate — too high blows the loss up; too low stalls. Try linear warmup over 500-2000 steps. Second, gradient clipping — clip norm at 1.0 to prevent exploding gradients. Third, LayerNorm placement — Post-LN without warmup is unstable. Fourth, data quality — duplicate-heavy or low-quality data corrupts training. Fifth, precision — FP16 without loss-scaling can underflow; switch to BF16. Sixth, batch size — too small means noisy gradients; gradient accumulation helps."

**Q17. Why residual connections in transformers?**
> "To enable arbitrarily deep stacks. Without residuals, gradients through N layers of nonlinearities vanish or explode. A skip connection adds a path with gradient 1, so signal and gradient flow cleanly through 100+ layers. Every transformer block has two residual additions — around attention and around FFN. Without them, training a 32-layer transformer would be nearly impossible."

**Q18. [Gotcha] Two models with the same num_params — same quality?**
> "Not necessarily. Architecture choices matter enormously: GQA vs MHA, SwiGLU vs ReLU, Pre-LN vs Post-LN, RoPE vs learned PE. Training-token budget matters even more — a 7B trained on 15T tokens beats a 30B trained on 300B. Data quality, hyperparameter tuning, fine-tuning recipe, all are big levers. Param count is a rough proxy for capacity but not a reliable predictor of quality."

**Q19. Difference between attention weights and attention scores.**
> "Scores are pre-softmax — raw `Q dot K transpose / sqrt(d_k)`. Weights are post-softmax — non-negative, sum to 1 per query. Be precise in interviews — papers and blog posts use these interchangeably, but a senior interviewer will appreciate the distinction. The mask is added to scores; the weighted sum uses weights."

**Q20. Speculative decoding — explain it.**
> "A small fast draft model proposes the next k tokens sequentially. The big main model verifies all k in one forward pass — since verification is parallel and each forward pass is bandwidth-bound regardless of how many tokens it processes, the cost of verifying k tokens is roughly the cost of generating one. Accept the longest valid prefix where draft and main agree on the argmax. Net result: 2-3x speedup with mathematically identical output distribution. Variants include Medusa heads, EAGLE, self-speculative."

**Q21. Multi-head Latent Attention (MLA)?**
> "DeepSeek-V2 introduced this. Compress K and V into a shared low-rank latent vector — say rank 512 — and cache only the latent. At attention time, learned up-projections decompress to full K and V. Combined with decoupled RoPE so the latent doesn't have to encode position. Achieves ~93% KV-cache reduction vs MHA at MHA-level quality. DeepSeek-V2 and V3 use it."

**Q22. Why is decode memory-bandwidth-bound?**
> "Each decode step processes one token but streams the entire model from HBM — 140 GB for a 70B BF16 model — plus KV-cache. Compute is tiny against 2-3 TB/s bandwidth. Wall time is dominated by `weights_size / bandwidth`. Max-batching is the only way to amortize the weight transfer across many concurrent users, multiplying throughput. That's why vLLM's continuous batching is so high-impact."

**Q23. FlashAttention-3 specific advantages?**
> "Hopper architecture (H100/H200) async pipelining via TMA — overlaps memory copy with compute, hiding latency. Native FP8 tensor core support, doubling throughput at small precision cost. Better warp scheduling. End-to-end ~2x over FA-2 on H100. PyTorch 2.4+ ships it; vLLM uses it by default."

**Q24. What if you remove positional encoding entirely?**
> "Attention becomes permutation-invariant — `dog bites man` and `man bites dog` get identical token-level representations, just permuted. For decoder-only models, some position info leaks through the causal mask itself — position 5 sees positions 0-4, position 6 sees 0-5 — but it's a weak signal. Quality degrades significantly without explicit PE. Always include it in production."

**Q25. [Gotcha] Train at 4K, deploy at 16K — what breaks?**
> "Vanilla RoPE or learned PE produce garbage at unseen positions. Fixes: (1) RoPE scaling — NTK-aware or YaRN — to extend without retraining; (2) train with longer context from the start; (3) sliding-window attention that caps effective context at a fixed window. Always test before deploying past training context."

**Q26. Sparse attention patterns?**
> "Variants that skip most pairs — local/windowed (Longformer), strided (Sparse Transformer), global+local (BigBird), random (Reformer). Reduce O(n²) to O(n·w) or O(n log n). Quality cost on long-range dependencies. Mostly displaced by FlashAttention + RoPE + YaRN, which handle long context efficiently without sparsifying."

**Q27. Why dot product, not cosine, in attention?**
> "Cosine similarity normalizes out magnitude — but vector magnitudes carry information about token importance. Also cosine requires extra L2 normalization, more compute. Dot product preserves magnitude info; the `sqrt(d_k)` scaling recovers numerical stability. Empirically dot-product attention works as well or better than cosine."

**Q28. Parameters per transformer block?**
> "Attention: 4d² for Q, K, V, O projections. FFN: 8d² for ReLU/GELU (W_1 of d×4d, W_2 of 4d×d), or about ~5.3d² for SwiGLU at matched parameter count. LayerNorm/RMSNorm params negligible. Total per block ~12d² classic, ~9.3d² with SwiGLU."

**Q29. [Gotcha] Increase heads from 8 to 16 with fixed d_model — what happens?**
> "Each head's d_k drops from `d_model/8` to `d_model/16` — heads get thinner. Total compute and parameters stay essentially the same. Quality is task-dependent — sometimes more specialization helps, sometimes thinner heads can't capture useful patterns. Empirical sweet spot is `d_k = 64` or `128`."

**Q30. Absolute vs relative position encoding — why does relative win?**
> "Absolute encoding gives each position a unique vector — works for trained lengths, breaks beyond. Relative encodes the *distance* between positions, not absolute index — generalizes naturally to longer sequences. Key insight behind T5's relative bias, Shaw-style relative embeddings, and RoPE. Modern LLMs use RoPE because it gives relative position by construction in attention's inner product."

---

## 2.11 Resume tie-ins

- **"Architected AI-powered ML workspace assistant on Claude"**: Claude is decoder-only, RoPE-based, long-context-capable. Good place to discuss why decoder-only LLMs dominated and how I leveraged Claude's tool-calling for our agent. The architecture decisions we made at TrueBalance — long context for code understanding, structured tool calls — all reflect transformer fundamentals.
- **"RAG knowledge-base chatbot at ResMed"**: Retrieval used encoder-only sentence-transformer embeddings; generation used decoder-only LLMs. The contrast between bidirectional and causal attention drove our design choices — why we couldn't just "use the LLM for everything" and needed a separate embedding model.
- **"Real-time XGBoost Lambda"**: Not a transformer, but the latency-engineering mindset transfers. KV-cache management for LLMs is conceptually similar to feature-cache management for tabular models — both are about amortizing memory cost across requests.

---

Continue to **[Chapter 03 — How LLMs Work End-to-End](03_llms.md)**.
