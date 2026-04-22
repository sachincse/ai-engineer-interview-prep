# Chapter 03 — How LLMs Work End-to-End
## Tokenization → Pretraining → SFT → Alignment → Inference

> The JD says "building, deploying, maintaining AI/ML models in production" and "productionizing LLM-based applications." This chapter is your mental map of the full LLM lifecycle.

---

## 3.1 The Four-Stage LLM Pipeline

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ 1. Pretraining  │   │ 2. SFT          │   │ 3. Alignment    │   │ 4. Deployment   │
│ (next-token on  │→  │ (instruction    │→  │ (RLHF / DPO     │→  │ (vLLM, batching,│
│  1-15T tokens)  │   │  following,     │   │  preference     │   │  KV-cache,      │
│                 │   │  100K-10M pairs)│   │  optimization)  │   │  monitoring)    │
└─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘
    weeks-months            hours-days           hours-days          ongoing
    $millions               $thousands            $thousands          $$ per token
```

Base model = foundation. SFT = shapes format. Alignment = shapes preferences. Deployment = scales it.

---

## 3.2 Stage 0 — Tokenization

### Why not characters or words?
- Chars: sequences too long, inefficient
- Words: 100K+ vocab, OOV explosion, poor multilingual

### Subword tokenization — the sweet spot

**BPE (Byte-Pair Encoding)** — used in GPT-2/3/4, LLaMA
1. Start with character-level vocab
2. Find most frequent adjacent pair, merge it
3. Repeat until target vocab size (e.g., 50K or 128K)

```
"loweroflow" → ['l','o','w','e','r','o','f','l','o','w']
After merges:
    'l','o' → 'lo'
    'lo','w' → 'low'
    'e','r' → 'er'
    → 'low','er','o','f','low'
```

**Byte-level BPE (GPT-2+, tiktoken)** — operates on raw UTF-8 bytes. Every text encodes; no OOV ever. Perfect for mixed-script / emoji / code.

**SentencePiece (Google)** — a *framework* (not algorithm). Treats input as a raw Unicode stream; no pre-tokenization splits that leak language assumptions. Uses BPE or Unigram under the hood. Standard for multilingual (mT5, NLLB, Gemma).

**WordPiece (BERT)** — similar to BPE but merges by likelihood (add the pair that maximizes training-corpus likelihood) instead of frequency.

### Why this matters in interviews

| Question | Answer pattern |
|----------|----------------|
| Why byte-level BPE? | No OOV, handles all scripts, no language-specific pre-tokenizer |
| Why SentencePiece? | Multilingual, no whitespace assumption |
| Why does Llama tokenizer have a 128K vocab in Llama-3? | Better multilingual efficiency, fewer tokens per non-English word |
| What is the "leading space" problem? | Most BPE tokenizers emit tokens like "▁hello" (▁ = space); forgetting this breaks string-match stops |

### Tokens != words
- English: ~1 token ≈ 0.75 words
- Code: ~1 token ≈ 0.5 words
- CJK (Chinese/Japanese/Korean): 1-3 tokens per character (varies by tokenizer)

### Tokenization cost implications
- Arabic with a tokenizer not optimized for it can be 3-5× more tokens than English for the same text. For a UAE deployment, pick a tokenizer (Qwen, Gemma, or explicitly multilingual variants) that handles Arabic efficiently.

---

## 3.3 Stage 1 — Pretraining (Next-Token Prediction)

### The objective
```
L = - Σ_t log P(x_t | x_<t)
```
"Given the first t-1 tokens, maximize probability of the true t-th token." Averaged over trillions of tokens.

### Data
Modern frontier LLMs: **1-15 trillion tokens** (LLaMA-3: 15T).
Mix: web (CommonCrawl refined), code (GitHub), books, arXiv, Wikipedia, synthetic data from smaller models.

### Compute
- Chinchilla scaling law (2022): optimal ≈ 20 tokens/param
- LLaMA-3 went well past this (210 tokens/param for 8B), showing over-training yields better inference-time quality
- Training: thousands of H100s for weeks

### Key training tricks
- **AdamW** with cosine LR schedule, warmup
- **Mixed precision** (BF16 for weights, FP32 for optimizer states)
- **Gradient clipping** (||g|| ≤ 1.0)
- **Z-loss** (encourages logit stability)
- **ZeRO / FSDP** sharding for memory
- **Activation checkpointing** trades compute for memory
- **Context-parallel** for long context
- **Tensor / pipeline parallelism** for multi-GPU

### Emergent abilities
Around certain compute thresholds, abilities "emerge" suddenly: in-context learning, chain-of-thought, instruction-following. Debate continues whether this is real discontinuity or a metric artifact.

---

## 3.4 Stage 2 — Supervised Fine-Tuning (SFT)

### What it does
Teach the model to follow instructions. Training data: (instruction, response) pairs.

```json
{
  "instruction": "Summarize the email below in one sentence.",
  "input": "Hi team, we need to move Monday's meeting to Tuesday at 3pm...",
  "output": "Monday's meeting is moved to Tuesday 3pm."
}
```

Loss: same next-token CE, but **only over the response tokens** (mask the instruction).

### Scale
- 100K to 10M examples typical
- Quality >> quantity (LIMA paper: 1K curated examples beats 50K messy)
- Hardest part is the data, not the training

### Chat templates
Every model family has a specific format; mismatches break performance:

```
Llama-3 chat template:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi!<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

```
ChatML (OpenAI / Qwen):
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
```

**Always use the model's official tokenizer `apply_chat_template()` — don't hand-build.**

---

## 3.5 Stage 3 — Alignment (RLHF, DPO, and friends)

### The goal
SFT teaches "what a good response looks like." Alignment teaches "what a *preferred* response looks like." Preferences > format.

### 3.5.1 RLHF pipeline (OpenAI InstructGPT)

```
  SFT model ──────────┐
      │                │
      │                ▼
 Generate N samples    │         Human labeler picks
 per prompt ──────────▶│─────▶   winner among pairs
                       │                │
                       │                ▼
                       │        Train Reward Model
                       │                │
                       │                ▼
                       └──────▶ PPO: optimize SFT
                                       model against RM,
                                       with KL penalty
                                       vs SFT reference
                                       │
                                       ▼
                              Aligned model
```

### 3.5.2 DPO — the modern default

**Direct Preference Optimization** (Rafailov et al., 2023): no explicit reward model, no PPO.

Derivation: given preference pair (chosen y_w, rejected y_l):
```
L_DPO = - log σ( β · log[π(y_w|x)/π_ref(y_w|x)] - β · log[π(y_l|x)/π_ref(y_l|x)] )
```

- π = current policy (being trained)
- π_ref = SFT reference (frozen)
- β = KL strength (typically 0.1-0.5)

**Why DPO dominates in 2025-2026:**
- No reward model to train (removes a noise source)
- Simpler, more stable — just a classification-style loss
- Single GPU, hours of training
- Close to or better than PPO in empirical results

### 3.5.3 Other variants worth knowing

| Method | One-line summary |
|--------|------------------|
| **RLAIF** | RLHF with an LLM judge instead of humans — Constitutional AI |
| **KTO** (Kahneman-Tversky) | Binary "good/bad" labels instead of pairs; cheaper labeling |
| **ORPO** | Combines SFT + DPO in one stage |
| **SimPO** | DPO without the reference model — cheaper compute |
| **IPO** (Identity PO) | Fixes DPO's over-optimization tendency |

### 3.5.4 KL regularization — why it matters

The KL term vs the SFT reference prevents the aligned model from drifting too far. Without it:
- "Reward hacking" — the policy exploits the RM instead of getting truly better
- Mode collapse — responses become generic / repetitive
- Loss of general knowledge

Too-high β: no improvement. Too-low β: hacking. Typical: 0.1-0.5.

---

## 3.6 Stage 4 — Inference

### The decoding loop (autoregressive)

```
input = tokenize(prompt)
kv_cache = empty

# PREFILL — process the prompt in parallel
logits = model(input, kv_cache=kv_cache)  # populates kv_cache for all prompt tokens
next_token = sample(logits[-1])

while not stop_condition:
    # DECODE — one token at a time
    logits = model(next_token, kv_cache=kv_cache)  # only new token goes in; attends to cache
    next_token = sample(logits[0])
    output.append(next_token)
```

Two distinct phases:
- **Prefill:** compute-bound (parallel over prompt length)
- **Decode:** memory-bandwidth-bound (sequential, one token at a time)

### Latency metrics

| Metric | What it measures | Typical target |
|--------|------------------|----------------|
| **TTFT** (Time to First Token) | prefill latency + first-token compute | <500 ms for chat |
| **TPOT** (Time Per Output Token) | steady-state decode speed | <50 ms for chat |
| **ITL** (Inter-Token Latency) | same as TPOT | streaming UX |
| **E2E latency** | total time for full response | depends on output length |

### Throughput optimizations

**Continuous batching (vLLM's secret sauce):**
- Naive batching waits for the slowest request in a batch → bad for variable lengths
- Continuous batching swaps finished requests out *every decode step* and admits new ones
- 5-10× throughput gain on heterogeneous workloads

**PagedAttention:**
- KV-cache as virtual memory pages (fixed-size blocks, not contiguous buffers)
- Eliminates internal fragmentation
- Enables prefix caching (share blocks across requests with same prompt prefix)

**Speculative decoding:**
- Small draft model proposes k tokens, large model verifies in one pass
- 2-3× latency reduction with identical output distribution
- Variants: Medusa heads, EAGLE-2, self-speculative

**Prefix caching:**
- Hash prompt prefixes; reuse KV blocks across requests sharing them
- RAG apps with common system prompts: TTFT drops 5-10×

**Chunked prefill:**
- Split long prefill into chunks that interleave with decode steps
- Smooths tail latency for mixed workloads

---

## 3.7 Context Window & Memory

For a 70B model at 8K context, FP16:
```
Weights:        70B × 2B    = 140 GB
KV-cache/req:   ~20 GB @ MHA (GQA cuts 4-8×)
Activations:    ~5-10 GB
Total/GPU:      fits on 2×H100-80GB or 4×A100-80GB with TP
```

Long context (128K+) pressures the KV-cache; this is why production deployments use GQA / MLA + KV-cache quantization (FP8/INT8/INT4).

---

## 3.8 Tool-Calling / Function-Calling

Modern chat LLMs can emit structured JSON to call tools:

```
User: What's the weather in Abu Dhabi?
Model: {"tool":"get_weather","args":{"city":"Abu Dhabi"}}
System: (executes, returns {"temp_c":34,"humidity":65})
Model: It's 34°C and 65% humidity in Abu Dhabi.
```

Under the hood:
- Model is SFT+DPO'd with tool-use examples
- Constrained decoding (Outlines, xgrammar, JSON mode) enforces schema validity
- Framework (LangChain, AgentSDK) handles the tool dispatch + result injection

This is exactly what powered your **Claude-based ML workspace** — tool-calling into Jira/GitHub/Athena/Jenkins.

---

## 3.9 Open vs Closed LLMs — the 2026 landscape

| Tier | Closed | Open |
|------|--------|------|
| Frontier | GPT-5, Claude Opus/Sonnet 4.x, Gemini 2.x | LLaMA-3.3/4, Qwen 2.5/3, DeepSeek-V3, Mistral Large |
| Mid | GPT-4o-mini, Claude Haiku 4.x | Qwen 72B, Llama-3.3 70B, Mixtral 8×22B |
| Small | GPT-4o-mini quantized, Claude Haiku | Phi-4, Gemma 2, Llama-3.2 1B/3B, Qwen 2.5-7B |

For Avrioc's likely **vLLM + self-hosted** stack, open models dominate: Llama-3.3-70B AWQ, Qwen-72B GPTQ, DeepSeek-V3 for frontier.

---

## 3.10 Interview Q&A — LLMs End-to-End

**Q1. Walk through the full LLM pipeline.**
> Pretraining (next-token on trillions of web tokens) → SFT (instruction-response pairs, often 100K-10M) → Alignment (RLHF with RM+PPO or DPO with preference pairs) → Deployment (quantization + vLLM + monitoring).

**Q2. Why DPO over RLHF?**
> DPO reparameterizes the RLHF objective so the reward model is implicit in the policy. One classification-style loss, no PPO instability, no reward-hacking from an OOD RM. Trains on a single GPU in hours instead of days. Production default in 2025-2026.

**Q3. What is KL divergence's role in RLHF?**
> A KL penalty vs the SFT reference prevents the policy from drifting into reward-hacked outputs that score high on RM but lose fluency or general knowledge. β controls the trade-off. Too low → mode collapse; too high → the policy never moves.

**Q4. BPE vs SentencePiece vs WordPiece?**
> BPE: frequency-based pair merges (GPT, Llama use byte-level BPE). WordPiece: likelihood-based merges (BERT). SentencePiece: a framework that treats input as raw Unicode, uses BPE or Unigram underneath — language-agnostic, great for multilingual.

**Q5. Why byte-level BPE?**
> Guarantees no OOV (every byte representable), handles emoji/code/rare scripts, no language-specific pre-tokenizer. Modern models (GPT-4, Llama-3) use byte-level BPE via tiktoken or sentencepiece-BPE.

**Q6. What is "exposure bias" in autoregressive training?**
> Training uses teacher forcing — the model always sees the ground-truth prefix. At inference it sees its own (potentially wrong) predictions. Error compounds. Techniques like scheduled sampling, contrastive search, and sampling strategies at inference mitigate it.

**Q7. Decoding strategies — pick one and explain.**
> (Pick nucleus.) Top-p (nucleus) samples from the smallest set whose cumulative probability ≥ p. Adapts to distribution shape (narrow at peaks, wide at uncertainty) — more robust than top-k which always picks k tokens regardless of peakedness.

**Q8. Speculative decoding — why does it work?**
> A small draft model generates candidate tokens sequentially (cheap). The big model verifies them in parallel (one forward pass). Since the big model's forward pass is dominated by weight transfer (memory-bandwidth-bound), verifying k tokens costs only slightly more than verifying 1. 2-3× speedup with identical distribution.

**Q9. [Gotcha] You ran SFT on 1M examples but eval got worse. Why?**
> Most likely: (1) data contamination / quality issues — garbage responses poison the model, (2) catastrophic forgetting of general capabilities — mix 10-20% general instruction data, (3) template mismatch — check you're using the model's exact chat template, (4) LR too high — try 1e-5 or 2e-5 for full FT, 2e-4 for LoRA.

**Q10. [Gotcha] Your SFT model has perfect eval loss but hallucinates more after DPO/RLHF. Why?**
> Preference optimization targets "what humans prefer," not "what's true." RMs learn proxy features (confident tone, length, politeness) that correlate with preference but diverge from truthfulness. Fixes: factuality-aware reward models, RAG grounding, lower β.

**Q11. What is continuous batching?**
> Naive batching waits for all requests in a batch to finish. Continuous batching swaps finished sequences out and admits new ones *every decoding step*. vLLM's core trick — 5-10× throughput gain vs static batching on variable-length workloads.

**Q12. What is PagedAttention?**
> vLLM's memory management: KV-cache stored in fixed-size blocks (pages) like OS virtual memory. Eliminates fragmentation; enables prefix sharing across requests. 2-4× more concurrent requests vs naive implementations.

**Q13. What is the prefill-decode distinction?**
> Prefill processes the prompt in parallel (compute-bound, uses the full parallelism of the GPU). Decode generates one token at a time (memory-bandwidth-bound — has to stream weights + KV-cache per step). Optimizations differ: prefill benefits from tensor-parallel compute; decode benefits from max-batching.

**Q14. TTFT vs TPOT — which matters more?**
> Depends on use case. Chat UX: TTFT matters most — users see the first word quickly. Batch analytics: E2E latency. Agent loops: TPOT (many short responses). Good SLO: TTFT < 500ms, TPOT < 50ms.

**Q15. Why is decode memory-bandwidth-bound?**
> One token's worth of compute is tiny. But we have to stream all model weights (70B params @ FP16 = 140 GB) and all KV-cache from HBM per step. Compute is instant; memory transfer dominates. Max-batching amortizes weight transfer across many simultaneous requests — the only way to get decode throughput up.

**Q16. What is chunked prefill?**
> Split a long prefill into chunks that interleave with in-flight decode steps (vLLM 0.6+). Smooths tail latency when a mix of short decodes and long prefills share the same engine. Important for chat apps with long RAG prompts.

**Q17. Constitutional AI vs RLAIF?**
> Constitutional AI (Anthropic, 2022): LLM-generated critiques based on a constitution; uses both SFT (critique-revise) and RL. RLAIF: broader term for replacing human feedback with LLM judges. CAI is a specific recipe; RLAIF is the category.

**Q18. What is PPO's "clip" in RLHF?**
> Clips the probability ratio π_new / π_old to [1-ε, 1+ε] (ε≈0.1-0.2). Prevents a single step from moving policy so far that subsequent updates diverge. Without it, RLHF is extremely unstable.

**Q19. Chain-of-Thought — decoding or training trick?**
> Both. As a *prompting* technique, "Let's think step by step" or few-shot CoT elicits intermediate reasoning at inference time. As *training*, you fine-tune on CoT traces (e.g., OpenAI o1-style) so the model reliably emits a reasoning trace before answers.

**Q20. Self-consistency vs Tree-of-Thoughts?**
> Self-consistency: sample N CoT traces, majority-vote final answer — cheap, effective for math. ToT: search (BFS/DFS) over branching reasoning states with a value function — more expensive, needed for harder planning. Order of quality: ToT > self-consistency > CoT > direct answer.

**Q21. [Gotcha] Temperature=0 in vLLM sometimes gives different outputs. Why?**
> Floating-point non-determinism in GPU reductions (atomic additions, tensor-core MAC order) means identical inputs can produce last-bit-different logits, causing argmax ties to break inconsistently. True determinism needs deterministic kernels + fixed seeds + fixed batch order.

**Q22. How does tool-calling work under the hood?**
> Model is SFT+DPO'd on tool-use traces in a standardized format. At inference, a system prompt describes available tools' JSON schemas. The model generates a JSON call (constrained decoding can enforce schema). Framework dispatches the tool, injects the result as a "tool" role turn, loops until the model stops calling tools.

**Q23. What does the chat template actually do?**
> Wraps user/assistant/system turns with model-specific control tokens (e.g., `<|eot_id|>`, `<|im_end|>`) so the model recognizes role boundaries. Mismatched templates = model confusion = bad outputs. Always use `tokenizer.apply_chat_template()`.

**Q24. Mixture of Experts (MoE) — quick explanation?**
> Replace the dense FFN with N "experts"; a learned router picks K (usually 2) experts per token. Total params large (e.g., Mixtral 8×7B = 47B total) but compute per token is only 2×7B = 14B. Training is tricky (load balancing); inference is cheaper for the quality level. Mixtral, DeepSeek-V3, GPT-4 (rumored) are MoE.

**Q25. What is grouped-query KV-cache quantization?**
> Quantizing the KV-cache itself (not just weights) to FP8/INT8/INT4. FP8 is nearly lossless and saves 2× memory → 2× concurrent requests. INT8 has a small quality drop; INT4 has noticeable degradation. vLLM supports `--kv-cache-dtype fp8`.

---

## 3.11 Resume tie-ins

- **"RAG-based knowledge-base chatbot at ResMed"** — the generation side used a decoder-only LLM. Be ready to walk through: tokenization → prompt construction (system + retrieved context + user query) → decode → citations.
- **"Claude-powered ML workspace integrating Jira, GitHub, Athena, Jenkins"** — pure tool-calling architecture. Discuss structured outputs, JSON mode, max-turn caps, and error handling.
- **"Real-time XGBoost Lambda p99 < 500ms"** — not an LLM, but the *latency mindset* transfers directly: TTFT/TPOT for LLMs is the analogue of p50/p99 for Lambda. Make the connection.

---

Continue to **[Chapter 04 — Embedding Models](04_embeddings.md)**.
