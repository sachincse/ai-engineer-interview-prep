# Chapter 05 — LLM Parameter Tuning (Inference)
## Every knob that changes LLM output — what it does, when to use it, how to set it

> The user asked: "what are all the parameters used in LLM for parameter tuning and how it impacts." This chapter is the full answer, sorted by importance.

---

## 5.1 The mental model — sampling pipeline

Every decoding step follows the same flow:

```
  Logits z (|V|,) from model forward pass
           │
   ┌───────▼───────┐
   │  Temperature  │   z_i ← z_i / T
   └───────┬───────┘
           │
   ┌───────▼───────────┐
   │ Repetition / freq │  z_i ← z_i - penalty(i)
   │ / presence / DRY  │
   └───────┬───────────┘
           │
   ┌───────▼───────────┐
   │ Top-k / Top-p     │  mask out unselected tokens
   │ / min-p / typical │
   └───────┬───────────┘
           │
   ┌───────▼───────────┐
   │    Softmax         │  p_i ← exp(z_i) / Σ exp(z_j)
   └───────┬───────────┘
           │
   ┌───────▼───────────┐
   │    Sample         │  pick next token
   └───────┬───────────┘
           │
       Token emitted
```

### Parameter hierarchy

```
Must understand deeply:    Temperature, top-p, top-k, max_tokens, stop, seed
Important:                  repetition/frequency/presence penalty, min-p, system_fingerprint
Niche but asked:            beam search (width, length penalty), mirostat, typical sampling
Specialty / guided:         JSON mode, grammar, tool-calling
```

---

## 5.2 Temperature — the master dial

```
z_i  ← z_i / T
p_i  = softmax(z / T)
```

| T | Effect | Use case |
|---|--------|----------|
| 0.0 | Greedy / argmax (deterministic modulo fp non-determinism) | Code, JSON, math, structured |
| 0.2 | Near-deterministic, high confidence | SQL gen, API extraction, factual RAG |
| 0.7 | Default for chat; balanced creative/factual | General chat, assistants |
| 1.0 | Raw distribution | Exploratory / diverse sampling |
| >1.0 | Flattens distribution → more randomness | Creative writing, brainstorming |
| >1.5 | Usually gibberish territory | Avoid without min-p |

**Interview gotcha:** T=0 is argmax, but vLLM/PyTorch don't *guarantee* determinism on GPU. Atomics, tensor-core MAC order, and batch position variability can produce last-bit-different logits; argmax ties break inconsistently. If you need true reproducibility: deterministic kernels + fixed seed + fixed batch.

---

## 5.3 Top-k — hard cap on candidates

```
Keep only the k highest-probability tokens; renormalize; sample.
```

- Typical: k=40 or k=50 for open-ended.
- k=1 equivalent to greedy.
- **Problem:** Doesn't adapt to distribution shape. If the distribution is peaked (one token is "obvious"), k=50 samples from noise. If it's flat (everything ~equal), k=50 is still too few.

**When to use:** Older stacks, or as a belt-and-suspenders cap (e.g., top-k=100 combined with top-p=0.95).

---

## 5.4 Top-p (nucleus) — the smart default

```
Sort tokens by probability descending.
Accumulate from the top until cumulative probability ≥ p.
Keep that "nucleus" set; renormalize; sample.
```

- Typical: p=0.9-0.95
- **Adapts to distribution shape** — few tokens when peaked, many when flat
- More robust default than top-k

**Combining top-k and top-p:** Common pattern: top-k=50, top-p=0.95. Top-k acts as a safety cap; top-p does the smart filtering.

---

## 5.5 Min-p — the modern replacement

```
Let p_max = max probability in the distribution.
Keep only tokens with p_i ≥ min_p · p_max.
```

- Typical: min_p = 0.05 - 0.1
- **Stable across temperature changes** — at high T, top-p collapses to noise but min-p still filters effectively
- Now default in llama.cpp, exllama, many local-LLM stacks

**When to prefer:** If you're running high temperature for creative generation, min-p keeps quality stable.

---

## 5.6 Typical sampling

```
Select tokens whose information content -log(p_i) is CLOSE TO the entropy of the distribution.
```

- Avoids both the boring (greedy — too low info) and the incoherent (high-T — too high info) tails
- Niche, mostly used for creative writing

---

## 5.7 Mirostat

Feedback-controlled decoding: dynamically adjusts top-k to maintain a target **perplexity** (surprisal) throughout the generation.

- **Mirostat-v2**: single τ parameter (target perplexity, e.g., 5.0)
- Prevents "drift into nonsense" that happens at long context with high temperature
- Popular in local-LLM creative writing

---

## 5.8 Repetition penalty — stop the loops

```
For each already-seen token t:
    z_t ← z_t / r     (if r > 1 and z_t > 0)
    z_t ← z_t · r     (if r > 1 and z_t < 0)
```

- Typical: r = 1.1-1.2
- **Gotcha:** Stacks multiplicatively on every prior token. After hundreds of tokens, common tokens (articles, punctuation) get zeroed out — model forced into low-probability garbage.
- **Fix:** Apply only over a rolling window (last 256 tokens), or use frequency penalty instead.

---

## 5.9 Frequency penalty (OpenAI)

```
z_i ← z_i - α · count(token_i in history)
```

- **Additive**, bounded — no stacking blowup
- α ≈ 0.5 - 1.0 is typical
- Scales with repetition count

---

## 5.10 Presence penalty (OpenAI)

```
z_i ← z_i - α · I(token_i seen at least once)
```

- Binary: token has appeared at all, or not
- α ≈ 0.5 - 1.0

**When to use what:**
- Frequency penalty for "don't let it loop on a word"
- Presence penalty for "try to cover diverse topics" (brainstorming)

---

## 5.11 DRY (Don't Repeat Yourself) sampler

Modern anti-repetition sampler that penalizes *sequences* of repeated tokens, not just individual tokens. Gaining adoption in local-LLM stacks for its elegance.

---

## 5.12 Beam search — only for structured tasks

```
Maintain top-B partial hypotheses; expand each by all next tokens; keep top-B overall.
```

- **Length penalty α** divides cumulative log-prob by length^α to offset the natural bias toward shorter sequences. Typical α = 0.6-1.0.
- Good for: machine translation, summarization, constrained output
- **Bad for open-ended generation** — beam search converges to "safe, boring, repetitive" outputs
- Not available in sampling-based inference (vLLM, most local stacks)

---

## 5.13 Max tokens and stop sequences

### Max tokens
- Caps output length. Also a safety/cost guard.
- Count tokens at the TOKENIZER level, not word level.
- For chat: set max_tokens to `context_length - prompt_tokens - some_buffer`.

### Stop sequences
- Strings that, when produced, terminate generation. E.g., `["\nUser:", "```end"]`.
- **Gotcha:** Stops are checked at the *decoded string* level, not tokens, because tokenizers may split a stop string across multiple tokens.
- vLLM, TGI handle this correctly with rolling string match.

---

## 5.14 Seed & Reproducibility

```
seed = 42
generator.manual_seed(seed)
# BUT: GPU non-determinism may still cause drift
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

For true bit-reproducibility across runs:
1. Fix seed
2. Deterministic kernels (slower)
3. Fixed batch order (sort by request arrival)
4. Same GPU + CUDA + framework version

---

## 5.15 System fingerprint

OpenAI and Anthropic return a `system_fingerprint` on each response. If it changes between calls, the underlying model/infra changed — your outputs may shift even with fixed seed/temperature. **Always log and alert on fingerprint changes.**

---

## 5.16 Logit bias

```
logit_bias = {" YES": 10.0, " NO": 10.0, " MAYBE": -10.0}
```

- Bump/suppress specific tokens' logits manually
- Use case: constrain classification to specific tokens
- Bias in the range -100 to +100 (OpenAI semantics — +100 forces, -100 bans)

---

## 5.17 Constrained / guided decoding (JSON, grammar, regex)

### Schema-constrained JSON
```json
{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
```

At each decoding step, mask out tokens that would violate the schema. Libraries: **Outlines**, **xgrammar**, **llama.cpp GBNF**, **LMFE**. OpenAI/Anthropic offer JSON mode + structured outputs.

### Tool-calling
- Function schema defines allowed tool names and argument types
- Constrained decoding guarantees 100% valid JSON
- Combined with temperature=0 and greedy → "reliable function-calling"

### Grammar-constrained
- Custom CFG (context-free grammar) — e.g., SQL, specific DSL
- llama.cpp supports GBNF

---

## 5.18 KV-cache quantization

A decoding-time memory trick (not a sampler, but changes output quality):

| KV dtype | Quality | Memory savings |
|----------|---------|---------------|
| FP16 / BF16 | baseline | 1× |
| **FP8** | ~0 loss | 2× → 2× more concurrent requests |
| INT8 | Small (<1% on short ctx, more on long) | 2× |
| INT4 | Noticeable, especially on reasoning | 4× |

vLLM: `--kv-cache-dtype fp8`. On H100, FP8 KV-cache is essentially free quality and doubles throughput.

---

## 5.19 Speculative decoding

Not a sampling parameter per se, but knobs:
- **draft_model** — which small model to use (e.g., Llama-3.2-1B for Llama-3-70B)
- **num_speculative_tokens (k)** — typically 4-7
- **acceptance rate** — track; if <50%, draft model is too different from target

vLLM: `--speculative-model ... --num-speculative-tokens 5`

---

## 5.20 Presets / recipes — what to set for common tasks

| Task | T | top-p | top-k | freq pen | rep pen | max_tokens | Notes |
|------|---|-------|-------|----------|---------|-----------|-------|
| **Code generation** | 0.0-0.2 | 1.0 | - | 0 | 1.0 | 512-2048 | Greedy is often best; constrained decoding for tool-calling |
| **Factual QA / RAG** | 0.0-0.2 | 0.9 | 50 | 0 | 1.0 | 512 | Deterministic, grounded |
| **Chat (default)** | 0.7 | 0.95 | 50 | 0 | 1.1 | 1024-2048 | Balanced |
| **Creative writing** | 0.9-1.0 | 0.95 | 50 | 0.3 | 1.1 | 1500+ | Anti-repetition matters |
| **Brainstorming** | 0.9 | 0.95 | 100 | 0 | 1.0 | 800 | Presence penalty for diversity |
| **JSON / tool calling** | 0.0-0.2 | 1.0 | - | 0 | 1.0 | 512 | Grammar-constrained decoding |
| **Summarization** | 0.3 | 0.9 | 50 | 0 | 1.1 | 512 | Low-T for faithfulness |
| **Classification (logit)** | 0.0 | - | 1 | 0 | 1.0 | 1-5 | Use logit_bias to force class tokens |

---

## 5.21 Parameter interactions — what NOT to combine naively

| Combination | Issue |
|-------------|-------|
| High T (≥1) + high top-p (0.99) | Pure noise; use min-p instead |
| Repetition penalty ≥1.3 + long output | Model pushed into low-prob garbage mid-generation |
| Beam search + open-ended task | Repetitive, safe, boring |
| Very low T + high rep penalty | Forces picking unlikely tokens just to avoid repetition — gibberish |
| Large top-k + low T | Irrelevant; top-k barely kicks in |
| Constrained JSON + high T | Possible valid-JSON outputs that are still semantically bad |

---

## 5.22 Interview Q&A — Parameter tuning

**Q1. Temperature — what exactly does it change?**
> Temperature T scales logits before softmax: p_i ∝ exp(z_i / T). T=0 → argmax. T<1 → sharper, more deterministic. T>1 → flatter, more random. Effective range 0.0-1.2 in production; above 1.5 you almost always need min-p.

**Q2. Top-p vs top-k — which do you prefer?**
> Top-p. Top-k is a hard cap regardless of distribution shape. Top-p adapts — few candidates when the model is confident, many when it's uncertain. Can combine (top-k=50, top-p=0.95) as a safety net.

**Q3. What's min-p and why is it gaining traction?**
> Min-p sets a floor as a fraction of the top token's probability. Unlike top-p, it's stable across temperature — at T=2, top-p=0.9 becomes noise but min-p=0.05 still filters garbage. Default in many local-LLM stacks.

**Q4. Frequency vs presence vs repetition penalty?**
> Repetition penalty (HF): divides logits of seen tokens by r (stacks multiplicatively — can blow up). Frequency penalty (OpenAI): additive, scales with count. Presence penalty: binary, scales with "seen at least once." Use frequency for loop prevention; presence for diversity.

**Q5. [Gotcha] Your model produces garbage mid-way through a long generation. What's happening?**
> Classic repetition-penalty stack. Rep penalty applied to every prior token pushes common tokens into low-prob regions. Fix: apply rep penalty only over a rolling window (e.g., last 256 tokens), switch to frequency penalty, or use DRY sampler.

**Q6. When is greedy actually the right choice?**
> Code, JSON, math-with-CoT, tool-calling, classification — anywhere correctness dominates diversity. Combined with low T and constrained decoding, greedy gives the most reliable structured outputs.

**Q7. [Gotcha] Temperature=0 in vLLM produces different outputs on different runs. Why?**
> GPU floating-point non-determinism — atomics in softmax, tensor-core MAC order, batch-dependent fused kernels — produce last-bit different logits, and argmax breaks ties inconsistently. True determinism needs deterministic kernels, fixed seed, fixed batch order.

**Q8. Beam search — why not for chat?**
> Beam search maximizes summed log-prob, which has a bias toward safe, repetitive text (the "the-the-the" problem). Great for MT/summarization where there's a single "right" answer; bad for open-ended generation. Sampling-based (top-p + T) produces more natural outputs.

**Q9. Max tokens — any subtleties?**
> Count at TOKENIZER level, not words. Set max_tokens = context_length - prompt_tokens - buffer. For streaming, large max_tokens is fine — you can stop emitting mid-stream. For batch pricing, tighten to real needs.

**Q10. How do stop sequences work if a stop string spans multiple tokens?**
> Must decode tokens to text and string-match; vLLM and TGI do rolling decoding. Naive implementations comparing token IDs miss stops.

**Q11. What is logit bias and when to use it?**
> Directly bump/suppress specific token logits (-100 bans, +100 forces). Use for constraining outputs to specific tokens (yes/no classification, controlled vocabulary).

**Q12. Constrained / guided decoding — how does it work?**
> At each step, compile a grammar / schema to an FSM; mask tokens that would violate it before sampling. Guarantees 100% valid JSON / function calls. Libraries: Outlines, xgrammar, GBNF.

**Q13. [Gotcha] With temperature=0.7, top-p=0.95, repetition_penalty=1.2, my 70B model produces gibberish around token 500. Fix?**
> Rolling rep penalty window (last 256 tokens), switch to frequency_penalty=0.6, or use DRY sampler. The issue is compounding rep penalty zeroing out common tokens.

**Q14. What does seed actually do in LLM inference?**
> Seeds the random number generator for sampling (noise when masking, tie-breaking). Doesn't guarantee bit-reproducibility on GPU without also enabling deterministic kernels — the "sampling sequence" is reproducible, but underlying logits can differ due to GPU non-determinism.

**Q15. System fingerprint — what is it and why monitor?**
> A server-returned ID indicating the model/infra version. If it changes, outputs may change even with identical sampling params. OpenAI, Anthropic, and others return it — log it for prompt-regression debugging.

**Q16. How does KV-cache quantization affect sampling quality?**
> FP8: ~zero loss, 2× concurrent requests — safe default on H100. INT8: minor degradation on long context. INT4: noticeable, especially on reasoning. Not a sampling parameter but materially affects output quality at scale.

**Q17. Speculative decoding — knobs?**
> `draft_model` (smaller sibling), `num_speculative_tokens` (usually 4-7). Acceptance rate is the health metric — <50% means the draft model diverges too much; try a smaller/more-aligned draft.

**Q18. Presets for structured output / tool calling?**
> T=0 or T=0.2, top-p=1.0, freq/pres=0, max_tokens=appropriate schema size, JSON mode or grammar-constrained decoding enabled. Goal: fully deterministic, schema-valid JSON every time.

**Q19. [Gotcha] Why isn't rep penalty sufficient for creative writing anti-loop?**
> Rep penalty is per-token; it doesn't understand *sequences* of repetition ("the cat sat on the mat. The cat sat on the mat."). DRY sampler specifically penalizes repeated sequences.

**Q20. What happens if you set top_k=1 AND top_p=0.9?**
> Top-k=1 wins — you're effectively greedy. Top-p has nothing to do because there's only one candidate after top-k. If you want smart filtering, don't set top-k=1.

**Q21. Recommended defaults for a production RAG chatbot?**
> T=0.3, top-p=0.9, freq_penalty=0.3, rep_penalty=1.0 (or 1.1 for longer answers), max_tokens=800. Low-T for faithfulness; modest frequency penalty to prevent loops; no top-k cap.

**Q22. What is "calibrated" output probability?**
> Whether the model's predicted probability matches empirical frequency. A well-calibrated model assigning p=0.8 should be right 80% of the time. Raw LLMs are often miscalibrated after RLHF; techniques like temperature scaling (post-hoc) can help for classification use cases.

**Q23. Difference between max_tokens and max_new_tokens?**
> Huggingface convention: `max_tokens` = total output length budget; `max_new_tokens` = only new tokens (excluding the prompt). OpenAI API uses `max_tokens` = new tokens only. Read the specific API's docs.

**Q24. What does "early_stopping=True" mean in beam search?**
> Stops as soon as B beams have produced an end-of-sequence token. Faster but may truncate longer answers. For MT / summarization, usually fine.

**Q25. Logprobs — why request them?**
> Get top-K per-token probabilities. Uses: scoring answers, detecting uncertainty (high entropy = model unsure — flag for review), classification via logit comparison, self-consistency evaluation. Small storage cost, big debuggability win.

---

## 5.23 Resume tie-ins

- **"Production-grade ML and LLM systems"** — the JD wants someone who has tuned inference params for real workloads. Have a specific story: "We had a RAG system with inconsistent answers. I dropped T from 0.7 to 0.2, added frequency_penalty=0.3, and faithfulness scores on the RAGAS eval set went from 0.72 to 0.86."
- **"Claude-based ML workspace"** — Claude's `max_tokens` and `stop_sequences` matter for tool-calling reliability. Mention you set `temperature=0, tool_choice=auto, max_tokens=1000` to get deterministic tool dispatch.
- **"XGBoost Lambda p99 < 500ms"** — not LLM, but the latency-vs-quality trade-off framing (temperature, speculative decoding, KV quant) transfers cleanly.

---

Continue to **[Chapter 06 — Fine-tuning](06_fine_tuning.md)**.
