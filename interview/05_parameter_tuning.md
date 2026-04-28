# Chapter 05 — LLM Parameter Tuning (Inference)
## A whiteboard tour of every knob that changes what an LLM says

> The interviewer at Avrioc will almost certainly ask: *"Walk me through how you'd tune an LLM for production. Temperature, top-p, the whole thing."* This chapter is the long-form version of that answer — with diagrams, worked numbers, and the kind of stories that make a senior engineer sound senior.

---

## 5.1 The mental model — what actually happens at each decoding step

Before we touch any parameter, let's get the picture right. An LLM doesn't "speak" — it predicts one token at a time. At each step, it produces a vector of raw scores (called *logits*) — one number per token in the vocabulary (typically 30,000 to 200,000 tokens). These logits then go through a series of transformations before a token is finally picked.

Think of it like a tournament bracket. Every word in the dictionary shows up to compete. The model assigns each one a raw score (logit). Then we apply rules — temperature reshapes scores, penalties dock points from words that already played, top-k/top-p eliminate weak contenders, and finally we draw a winner from those left standing. Every parameter you tune is a rule in that tournament.

### The decoding pipeline

```
                      ┌────────────────────────┐
                      │  Model Forward Pass    │
                      │  (transformer layers)  │
                      └───────────┬────────────┘
                                  │
                        Logits z ∈ ℝ^|V|
                                  │
                      ┌───────────▼────────────┐
                      │   Repetition / Freq /  │   z_i ← z_i − penalty(i)
                      │   Presence Penalties   │
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │      Temperature       │   z_i ← z_i / T
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │   Top-k / Top-p /      │   mask out unselected
                      │   min-p / typical      │
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │       Softmax          │   p_i ← exp(z_i) / Σ exp(z_j)
                      └───────────┬────────────┘
                                  │
                      ┌───────────▼────────────┐
                      │   Sample (or argmax)   │
                      └───────────┬────────────┘
                                  │
                                  ▼
                          Token emitted
```

Notice the order. Logits flow downward, each stage is a filter or a reshape. The final softmax converts surviving scores into a probability distribution that sums to 1, and then we draw one token.

> **How to say this in an interview:** "Sampling in an LLM is a pipeline. The model gives me raw logits, then penalties dock points from already-seen tokens, temperature flattens or sharpens the distribution, top-p or top-k truncates the long tail of unlikely tokens, softmax normalizes what's left, and finally we sample. Each parameter is a step in that pipeline, and you can usually trace a generation problem to one specific stage."

---

## 5.2 Our running example — the same prompt across every parameter

To make this concrete, let's pin a single prompt and watch how each parameter changes the outcome:

> Prompt: **"The capital of France is"**

A trained model gives us a probability distribution over the vocabulary at the next token. The top 10 tokens look roughly like this (real numbers from a Llama-style model):

| Rank | Token | Logit | Probability (T=1) |
|------|-------|-------|-------------------|
| 1 | " Paris" | 12.5 | 0.78 |
| 2 | " the" | 9.8 | 0.10 |
| 3 | " a" | 8.6 | 0.04 |
| 4 | " located" | 8.0 | 0.022 |
| 5 | " known" | 7.4 | 0.012 |
| 6 | " famous" | 7.0 | 0.008 |
| 7 | " Lyon" | 6.5 | 0.005 |
| 8 | " home" | 6.2 | 0.0036 |
| 9 | " situated" | 5.9 | 0.0027 |
| 10 | " Marseille" | 5.5 | 0.0018 |
| ... | (long tail of ~50K others) | ... | ~0.002 combined |

This is a *peaked* distribution — Paris dominates with 78% probability. We'll come back to this table throughout the chapter to show what each parameter does.

---

## 5.3 Temperature — the master dial

### Why it exists

The model's raw distribution is whatever the training data made it. Sometimes that's right (you want Paris), sometimes you wish it were sharper (code generation — give me the one right answer), sometimes you wish it were flatter (creative writing — surprise me). Temperature is the single knob that reshapes the entire distribution before any other filtering.

### The mental model

Imagine the probability distribution as a mountain range. Temperature is gravity. T=1 is normal gravity — peaks where they were. T<1 is heavy gravity — peaks become taller and narrower, valleys get deeper. T>1 is anti-gravity — everything flattens, peaks get squashed, the long tail of unlikely tokens floats up. T=0 is infinite gravity — everything except the very tallest peak collapses to zero.

### The math

```
z_i ← z_i / T
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

In plain English: divide each logit by T before softmax. Small T amplifies the gap between the top token and everything else; large T shrinks it.

### Worked example with our distribution

Apply T=0.5 (sharper) and T=1.5 (flatter) to the Paris example:

| Token | T=0.5 | T=1.0 | T=1.5 |
|-------|-------|-------|-------|
| " Paris" | 0.96 | 0.78 | 0.55 |
| " the" | 0.027 | 0.10 | 0.16 |
| " a" | 0.005 | 0.04 | 0.10 |
| " located" | 0.001 | 0.022 | 0.07 |
| " known" | <0.001 | 0.012 | 0.05 |
| ... | ~0 | ... | longer tail still alive |

At T=0.5, Paris is essentially guaranteed. At T=1.5, Paris is still the most likely but the model could legitimately say "the largest city in France" or "located in northern Europe" — variety is back.

### Side-by-side outputs at different temperatures

Same prompt: *"Write the opening line of a noir detective story."*

```
T = 0.0  (greedy, deterministic)
─────────────────────────────────
"The rain hadn't stopped for three days, and neither had the
 questions about the dead man in apartment 4B."

T = 0.7  (balanced, default chat)
─────────────────────────────────
"The neon sign above Murphy's Bar flickered like a dying heart,
 casting red shadows across the wet pavement where she lay."

T = 1.5  (high creativity, risky)
─────────────────────────────────
"Cigarette ashes danced upon the rim of yesterday's whiskey,
 while my reflection blinked at me from puddles that knew too much."
```

Notice T=0 is competent but predictable — the same story 100 times. T=0.7 is varied but coherent. T=1.5 is more poetic but starts to drift; at T=2.0+ you'd see grammatical breakdowns.

### When to use what

- **T=0** — code generation, JSON output, math, classification, tool calling. Anywhere correctness > variety.
- **T=0.2–0.3** — RAG answers, factual QA, SQL generation, structured extraction. You want determinism with a tiny bit of robustness to tokenizer quirks.
- **T=0.7** — default chat, customer support, general assistant. The "ChatGPT default."
- **T=0.9–1.0** — creative writing, brainstorming, story continuation.
- **T>1.2** — danger zone. Only use with min-p as a safety net.

### Common mistakes

1. **Setting T=0 and expecting bit-exact reproducibility on GPU.** Even greedy isn't fully deterministic on CUDA — atomic adds in attention, tensor-core MAC ordering, and batch-position fusion can produce last-bit-different logits. Argmax breaks ties inconsistently. If you need true reproducibility, you also need deterministic kernels and a fixed batch order.
2. **Cranking T to 1.5+ and being shocked at gibberish.** Without min-p or top-p truncation, high T floods the long tail with probability — model picks "absurd" tokens.
3. **Using T=0.7 for tool-calling and getting flaky JSON.** For anything structured, use T=0 plus constrained decoding (JSON mode, grammar). Don't gamble.

> **How to say this in an interview:** "Temperature is the master dial. It rescales logits before softmax — divide by T. Lower T sharpens the distribution toward greedy, higher T flattens it. I default to T=0 for anything structured — code, JSON, tool calls — and T=0.7 for chat. I never go above 1.2 without a min-p safety net, because high temperature without truncation just amplifies the long tail of garbage tokens."

### Interview Q&A — Temperature

**Q1. What does temperature actually do mathematically?**
> Temperature divides each logit by T before the softmax. The softmax is exp(z_i/T) divided by the sum of exp(z_j/T). When T is below 1, the gaps between logits get amplified — the top token's probability climbs, the rest collapse. When T is above 1, gaps shrink and probability mass spreads to lower-ranked tokens. At T=0 you're effectively taking the argmax. The magic is that temperature changes the *shape* of the distribution without changing the *ranking* of tokens.

**Q2. Why might T=0 still produce different outputs across runs?**
> Floating-point non-determinism on GPU. Atomic operations in attention kernels, tensor-core MAC ordering, and batched fused kernels can produce logits that differ in the last few bits between runs. When two top tokens have nearly equal logits, those last bits decide the argmax, so you get different outputs. To guarantee bit-exact reproducibility you need deterministic CUDA kernels (slower), a fixed seed, fixed batch order, and the same hardware/CUDA/framework versions. Most production teams accept this drift instead of paying the latency cost.

**Q3. When would you use T=0.2 instead of T=0?**
> When you want near-determinism but a small amount of resilience. T=0 picks the single highest logit, which can be brittle — sometimes the top two tokens are essentially tied and the model "feels" like the second one is better grammatically. T=0.2 lets that occasionally happen without introducing real variance. I use it for RAG answers and SQL generation where I want consistency but not rigid greediness.

**Q4. Walk me through how T=1.5 affects the Paris example.**
> Take the original Paris distribution. Paris is at 78% probability. After dividing logits by 1.5, the gap between Paris's logit and "the" or "a" shrinks. When you re-softmax, Paris drops to maybe 55%, "the" climbs to 16%, "a" to 10%, and even "located" or "known" become real candidates. The distribution is flatter, so sampling can actually pick a non-Paris token now — which is sometimes what you want for variety, but for a factual question this is wrong.

---

## 5.4 Top-k — the hard candidate cap

### Why it exists

Even at T=1, there are 50,000+ tokens in the vocabulary. Most have negligible probability, but they still contribute noise to sampling. Top-k says: only consider the k highest-probability tokens, throw away the rest, renormalize, and sample from those.

### The mental model

Top-k is like a guest list with a fixed size. "Only the top 50 most likely words get into the party. Everyone else, go home." The party then redistributes total probability among those 50.

### The math

```
1. Sort tokens by probability descending.
2. Keep only the top k.
3. Renormalize: p_i ← p_i / Σ(top-k probabilities).
4. Sample from the renormalized distribution.
```

### Worked example

For our Paris distribution with k=3, only Paris (0.78), "the" (0.10), and "a" (0.04) survive. We renormalize: Paris becomes 0.78 / 0.92 = 0.85, "the" becomes 0.11, "a" becomes 0.04. Everything else is set to zero.

### The problem with top-k

It doesn't adapt to distribution shape. Consider two cases:

```
Case A — peaked (Paris example):
    p1 = 0.78, p2 = 0.10, p3 = 0.04, ...
    Top-k=50 keeps 50 tokens, but everything past rank 5 has near-zero
    probability. Top-k is too generous — adds noise.

Case B — flat (next token after "Once upon a"):
    p1 = 0.05, p2 = 0.04, p3 = 0.04, ...
    Top-k=50 might still cut off legitimate candidates. Too restrictive.
```

This is why top-p (next section) is the smarter default.

### When to use top-k

- As a **belt-and-suspenders cap** on top of top-p (e.g., k=100 with p=0.95)
- In older stacks that don't support top-p well
- When you specifically need a hard cap on candidates regardless of shape

### Common mistakes

1. **Setting top-k=1 and wondering why temperature doesn't matter.** Top-k=1 is greedy. No matter what T is, only one token survives.
2. **Top-k=50 with low temperature.** The temperature already collapses to near-greedy, so top-k=50 has nothing to do.

> **How to say this in an interview:** "Top-k caps the candidate set at the k highest-probability tokens, renormalizes, then samples. The downside is it's distribution-shape-blind — k=50 might be wildly too generous on a peaked distribution and not enough on a flat one. I use top-p as my primary truncator and reserve top-k as a hard safety cap, like top-k=100 combined with top-p=0.95."

---

## 5.5 Top-p (nucleus sampling) — the smart default

### Why it exists

Top-k's blindness to distribution shape is fixable: instead of "keep k tokens," say "keep the smallest set of tokens whose cumulative probability is at least p." This automatically adapts. Peaked distribution? Just a few tokens. Flat distribution? Many tokens.

### The mental model

Think of it as a probability spotlight. You sweep from the highest-probability token down, accumulating probability. The moment you cross threshold p (say 0.9), you stop. Everything inside the spotlight is the *nucleus*. Everything outside is discarded.

### The math

In English first: sort tokens by probability descending. Walk down the sorted list, summing probabilities. Stop the first time the running sum reaches p. Keep all tokens included so far. Renormalize. Sample.

```
Keep token i iff Σ_{j ≤ rank(i)} p_j ≤ p
(plus the token that crosses the threshold)
```

### Worked example with p=0.9

Our Paris distribution, sorted:

```
Rank  Token       p_i     Cumulative
 1    Paris       0.78    0.78  ← under 0.9, keep
 2    the         0.10    0.88  ← under 0.9, keep
 3    a           0.04    0.92  ← crosses 0.9, keep this one too, STOP
```

Nucleus = {Paris, the, a}. Renormalize: Paris 0.78/0.92=0.85, the 0.11, a 0.04. Same as top-k=3 in this case — *coincidentally*. Now consider a flat distribution where the top 50 tokens each have probability ~0.02. Top-p=0.9 would keep ~45 tokens. Top-k=3 would keep 3. Top-p adapts; top-k doesn't.

### When to use top-p

- Default for any chat/creative use case (p=0.9–0.95)
- Combined with top-k as a safety cap
- Combined with temperature 0.7–1.0

### The classic combo

```
T = 0.7
top_p = 0.95
top_k = 50  (safety cap)
```

This is the de facto "good general chat" setting.

### Common mistakes

1. **Thinking top-p=1.0 disables sampling.** It doesn't — it just keeps the entire distribution. Sampling still happens at T>0.
2. **Top-p=0.99 with high T.** At T=2, the top-1 token might have probability 0.05 and you need to accumulate dozens of tokens to reach 0.99. You're sampling from noise.

> **How to say this in an interview:** "Top-p is nucleus sampling — you sort tokens by probability, walk down, accumulate, and cut off the moment cumulative probability reaches p like 0.9. Unlike top-k, it adapts to distribution shape. Peaked distribution gives a small nucleus. Flat distribution gives a wide one. My default for chat is T=0.7, top-p=0.95, with top-k=50 as a hard cap."

---

## 5.6 Min-p — the modern, temperature-stable filter

### Why it exists

Top-p has a hidden weakness — it's not stable across temperature. At T=2.0, even a "good" model produces a fairly flat distribution. Top-p=0.9 then includes hundreds of tokens, including absurd ones. Min-p fixes this by anchoring the threshold to the *top* token's probability, not absolute cumulative probability.

### The mental model

Min-p says: "Only keep tokens that are at least 5% as likely as the most likely token." If the top token is 0.6, the cutoff is 0.03. If the top token is 0.05 (flat dist), the cutoff is 0.0025. The threshold scales automatically.

### The math

```
p_max = max_i p_i
Keep token i iff p_i ≥ min_p × p_max
```

### Worked example

Paris distribution, min_p = 0.05:
- p_max = 0.78 (Paris)
- threshold = 0.05 × 0.78 = 0.039
- Keep: Paris (0.78), the (0.10), a (0.04). Drop everything else.

Now flat distribution with p_max = 0.05:
- threshold = 0.05 × 0.05 = 0.0025
- Keep: any token with p ≥ 0.0025 (could be hundreds, all roughly equal)

Notice min-p kept the right thing in both cases without us tuning per-prompt. This is why it's gaining traction for creative writing where temperature is high.

### When to use min-p

- High-temperature creative writing (T=1.0–1.5 with min-p=0.05)
- Any time top-p feels unstable
- Local LLM stacks where min-p is now the default (llama.cpp, exllama)

### Common mistakes

1. **Setting min-p too high (>0.2).** Now you're effectively greedy on peaked dists.
2. **Using both top-p and min-p without thinking about interaction.** They're both truncators — pick one as primary.

> **How to say this in an interview:** "Min-p sets a floor as a fraction of the top token's probability. If min-p is 0.05 and the top token is at 0.6, anything below 0.03 gets cut. The reason it's better than top-p at high temperature is that it's *relative* to the peak, not absolute. At T=2, top-p=0.9 ends up sampling from noise; min-p still filters effectively. It's now the default in llama.cpp and most local-LLM stacks for creative generation."

---

## 5.7 Visualizing top-k vs top-p vs min-p side-by-side

Same Paris distribution. Different truncators.

```
ORIGINAL DISTRIBUTION (top 10 shown):
  Paris      ████████████████████████████████  0.78
  the        ████                              0.10
  a          ██                                0.04
  located    █                                 0.022
  known      ▌                                 0.012
  famous     ▎                                 0.008
  Lyon       ▎                                 0.005
  home       ▎                                 0.0036
  situated   ▏                                 0.0027
  Marseille  ▏                                 0.0018
  ... ~50K more ...

TOP-K = 5:
  Keeps:  [Paris, the, a, located, known]
  Drops:  everything ranked 6+

TOP-P = 0.9:
  Cumulative: 0.78 → 0.88 → 0.92 (crosses 0.9)
  Keeps:  [Paris, the, a]
  Drops:  everything ranked 4+

MIN-P = 0.05:
  Threshold: 0.05 × 0.78 = 0.039
  Keeps:  [Paris, the]   (a at 0.04 just barely makes it depending on impl)
  Drops:  everything below 0.039
```

Notice how each method gives a slightly different cutoff. On peaked distributions, all three converge — they all keep ~3 tokens. On flat distributions, they diverge wildly.

---

## 5.8 Repetition penalty — the loop killer

### Why it exists

LLMs love loops. "The cat sat on the mat. The cat sat on the mat. The cat sat..." This happens because once a phrase appears, it strongly conditions the model to repeat it (the prefix becomes more likely). Repetition penalty fights back by reducing the logit of any token already in the context.

### The mental model

Imagine docking the popularity of any politician who's already given a speech. Each time they show up again, the docking compounds. After enough speeches, even the most popular politician has been docked into oblivion and the audience has to listen to someone else.

### The math (HuggingFace style)

```
For each token t already in context:
    if z_t > 0:  z_t ← z_t / r
    if z_t < 0:  z_t ← z_t × r

Typical r = 1.1 to 1.2
```

The asymmetry on positive vs negative logits is to keep the penalty pushing in one direction (down) regardless of sign.

### Worked example

Suppose the model has just generated *"The dog ran. The dog ran."* Without rep penalty, the next prediction strongly favors "The" again. With rep penalty r=1.1, the logit for "The" gets divided by 1.1. After 5 occurrences, the cumulative effect is /1.1^5 ≈ /1.61 — a hefty dock. The model now picks something else, say "It."

### Side-by-side: rep penalty on vs off

Prompt: *"List five things about cats."*

```
WITHOUT rep penalty (r=1.0):
─────────────────────────────
1. Cats are mammals.
2. Cats are carnivores.
3. Cats are predators.
4. Cats are nocturnal.
5. Cats are common pets.

(Model loops on "Cats are X" structure.)

WITH rep penalty r=1.15:
─────────────────────────────
1. They are obligate carnivores.
2. Felines purr at 25-50 Hz.
3. Whiskers sense air currents.
4. A group of cats is called a clowder.
5. They sleep 12-16 hours daily.

(Diversity kicks in — phrasing varies.)
```

### The big gotcha — compound stacking

Rep penalty applies to *every* prior occurrence multiplicatively. Generate 1000 tokens with r=1.2, and the word "the" — which appears maybe 50 times — gets divided by 1.2^50 ≈ 9100. The logit for "the" becomes vanishingly small. The model is now forced to never use "the" again, leading to weird ungrammatical output by token 500.

### Mitigation strategies

1. **Use a rolling window** — only penalize the last 256 tokens, not all of history
2. **Switch to frequency penalty** (additive, doesn't compound multiplicatively)
3. **Use DRY sampler** — penalizes repeated *sequences*, not single tokens
4. **Lower r** (1.05–1.1 is safer than 1.2 for long output)

> **How to say this in an interview:** "Repetition penalty divides the logit of any seen token by r, typically 1.1 to 1.2. The classic gotcha is multiplicative compounding — over a long generation, common words like 'the' get divided down so far that the model can't use them anymore, and you get gibberish around token 500. The fix is either a rolling window penalty, switching to additive frequency penalty, or using the DRY sampler which penalizes repeated *sequences* instead of individual tokens."

---

## 5.9 Frequency and presence penalties — OpenAI's additive cousins

### Frequency penalty

```
z_i ← z_i − α × count(token_i in history)
```

Additive (not multiplicative). Scales with how often the token has appeared. α typically 0.3–1.0.

Think of it as: "Each time you've used this word, lose α points." After 5 uses with α=0.5, you've lost 2.5 logit points — significant but bounded.

### Presence penalty

```
z_i ← z_i − α × I(token_i seen at least once)
```

Binary. The penalty kicks in once and doesn't grow with repetition. α typically 0.3–1.0.

Think of it as: "If you've used this word at all, lose α points. Period." Encourages topic diversity rather than just preventing loops.

### When to use which

- **Frequency penalty** — anti-loop. "Don't keep saying the same word." Use in summarization, long-form generation.
- **Presence penalty** — diversity. "Cover new topics." Use in brainstorming.
- **Repetition penalty (HF)** — legacy multiplicative version; use only with rolling window.

### Worked comparison

Generate a 200-word essay. With each penalty type:

```
NO PENALTY:
"The system is fast. The system is reliable. The system is..."
(Loops on "The system is")

FREQ PENALTY α=0.6:
"The system is fast. It is reliable. The platform also..."
(Pronouns and synonyms appear; loops broken)

PRESENCE PENALTY α=0.6:
"The system handles X. Notably, performance metrics indicate Y..."
(Each new sentence introduces fresh topic words)

REP PENALTY r=1.15 (multiplicative):
"The system is fast. A platform proves reliable. Our solution provides..."
(Similar to freq, but late-stage may break)
```

> **How to say this in an interview:** "Frequency penalty docks a token's logit linearly with how often it's appeared — additive, bounded, won't blow up over long generation. Presence penalty docks once and only once when a token first appears, so it pushes the model toward new topics. Repetition penalty is the old multiplicative version that compounds and can wreck long generations. For chat I default to frequency penalty around 0.3, presence at 0.0 unless I want brainstorming behavior."

---

## 5.10 Beam search — and why it's nearly extinct in production

### Why it exists

Sampling is good for diversity but can pick a token that paints the model into a corner. Beam search addresses this by maintaining multiple candidate sequences in parallel, exploring several paths simultaneously, and choosing the highest-scoring complete sequence at the end.

### The mental model

Imagine writing a sentence by always keeping the 5 best partial sentences alive at every step. At step 1 you have 5 candidate first words. At step 2 you expand each of those 5 to all possible second words (5 × |V|), then prune back down to the best 5 *complete* two-word sequences. Repeat until each beam terminates.

### The 2-step decoding tree

```
Start: "The capital of France is"

Step 1 (beam width B=3):
                    [start]
                       │
            ┌──────────┼──────────┐
         Paris       the         a               ← keep top-3
        (0.78)     (0.10)      (0.04)

Step 2 (expand each beam, keep top-3 OVERALL):
        Paris               the              a
          │                  │                │
    ┌─────┼─────┐      ┌─────┼─────┐    ┌─────┼─────┐
   .       ,    is     city  capital one  city  small location
  (0.5)  (0.3) (0.1)  (0.4)  (0.3)        (0.3)

  Score = log_p sum across both steps.
  Top-3 combined paths kept:
  1. "Paris."           score = log(0.78) + log(0.5)  = -0.94
  2. "Paris,"           score = log(0.78) + log(0.3)  = -1.45
  3. "the city"         score = log(0.10) + log(0.4)  = -3.22

Continue until each beam hits EOS.
Length penalty applied at end:
   final_score = sum_log_p / length^α    (α = 0.6 typical)
```

### Why beam search lacks diversity

All beams converge toward the highest-probability path. For open-ended generation, that path is "safe, common, repetitive" — the famous "the the the" pathology. Beam search optimizes for *summed log-probability*, which is biased toward boring text.

### When beam search is still used

- **Machine translation** — there's a single "correct" target sentence; beam search shines
- **Constrained summarization** — likewise, a single best target
- **Speech recognition** — finding the most likely transcript
- **Any task where there's a clear "right answer"**

### When beam search is wrong

- **Open-ended chat** — produces dull, repetitive responses
- **Creative writing** — actively destroys creativity
- **Code generation** — surprisingly, sampling often beats beam here because beam over-commits early

### Why production rarely uses it

1. **Compute cost** — B times the forward passes
2. **Latency** — can't easily stream tokens (you don't know which beam wins until the end)
3. **Engineering complexity** — most modern serving stacks (vLLM, TGI, SGLang) skip it or only support it grudgingly
4. **Quality** — for most LLM use cases, sampling produces better outputs

> **How to say this in an interview:** "Beam search keeps the top-B partial sequences alive at each step and expands them all in parallel, picking the single highest-scoring complete sequence at the end. It's great for tasks with a single right answer like translation or summarization, but it's terrible for open-ended generation because it converges to safe, repetitive text — the 'the-the-the' problem. Modern serving stacks like vLLM mostly skip it. For chat I always use sampling-based decoding."

---

## 5.11 Max tokens, stop sequences, and seeds

### Max tokens

The cap on output length. Two subtleties:

1. **Token, not word.** "unbelievable" might be 1 token or 5 depending on the tokenizer.
2. **API conventions differ.** OpenAI's `max_tokens` = output only. HuggingFace's `max_length` = total (prompt + output). HuggingFace's `max_new_tokens` = output only. *Always read the docs.*

### Stop sequences

Strings that, when emitted, terminate generation immediately. E.g., `["\nUser:", "</answer>"]`.

The gotcha: stops match against the *decoded string*, not the token stream. A stop sequence like "</answer>" might span multiple tokens, so naive token-ID matching misses it. Production stacks (vLLM, TGI) decode rolling and string-match.

### Seed

Seeds the random number generator for sampling. Same prompt + same seed = same output (modulo GPU non-determinism). Without seed, the random generator is initialized from time/entropy.

For true reproducibility:
1. Fix `seed`
2. Enable deterministic kernels (`torch.use_deterministic_algorithms(True)`)
3. Fix batch order
4. Same hardware/CUDA/framework versions

Even then, *system_fingerprint* changes (model patch, infra change) will produce different outputs.

---

## 5.12 Constrained / guided decoding — when format matters more than fluency

### Why it exists

Even at T=0, an LLM can produce malformed JSON ("almost valid, missing a comma"). Constrained decoding fixes this at the *sampling* level — at each step, mask out any token that would violate the desired format.

### How it works

You compile a schema (JSON, regex, grammar) into a finite-state automaton. At each decoding step, the FSM tells you which tokens are valid given what's been generated so far. You set the logits of all invalid tokens to -infinity, then sample from what's left.

```
    JSON schema
         │
         ▼
    ┌────────────┐
    │   FSM      │
    │  (states)  │
    └─────┬──────┘
          │ for each step:
          │   "what tokens are valid?"
          ▼
    ┌──────────────┐
    │ Token mask   │
    │ (set invalid │
    │  to -inf)    │
    └──────┬───────┘
           │
           ▼
        Sample
```

### When to use

- **JSON output** — guarantees valid JSON every time
- **Tool calling** — guarantees valid function arguments
- **SQL generation** — limits to valid SQL grammar
- **Custom DSLs** — any domain-specific format

Libraries: Outlines, xgrammar, llama.cpp GBNF, OpenAI structured outputs.

### Common mistake

Combining grammar constraint with high T. The grammar guarantees *syntactic* validity but not *semantic* correctness. T=0.8 + JSON constraint = valid JSON with potentially nonsense values.

---

## 5.13 The full preset table — what to set for which task

| Task | T | top-p | top-k | freq pen | rep pen | max_tokens | Notes |
|------|---|-------|-------|----------|---------|------------|-------|
| **Code generation** | 0.0–0.2 | 1.0 | – | 0 | 1.0 | 512–2048 | Greedy + grammar constraint for tool calls |
| **JSON / tool calling** | 0.0 | 1.0 | – | 0 | 1.0 | 512 | Constrained decoding mandatory |
| **Factual QA / RAG** | 0.0–0.2 | 0.9 | 50 | 0.3 | 1.0 | 512 | Low T for grounding |
| **Chat (default)** | 0.7 | 0.95 | 50 | 0.3 | 1.0 | 1024 | The "ChatGPT default" |
| **Summarization** | 0.3 | 0.9 | 50 | 0.4 | 1.0 | 512 | Low T for faithfulness |
| **Creative writing** | 0.9 | 0.95 | 50 | 0.0 | 1.05 | 1500+ | Min-p=0.05 if T>1.0 |
| **Brainstorming** | 0.9 | 0.95 | 100 | 0.0 | 1.0 | 800 | Presence penalty=0.6 for diversity |
| **Classification** | 0.0 | – | 1 | 0 | 1.0 | 1–5 | Use logit_bias to force class tokens |
| **Translation (open-ended)** | 0.3 | 0.9 | 50 | 0.2 | 1.0 | 1024 | Beam=4 if you want classical MT quality |

---

## 5.14 Common parameter interactions to avoid

| Combination | Issue |
|-------------|-------|
| High T (≥1) + top-p=0.99 | Pure noise — top-p includes too much mass; use min-p instead |
| Rep penalty ≥1.3 + long output | Model pushed into low-prob garbage mid-generation |
| Beam search + open-ended task | Repetitive, safe, boring outputs |
| Very low T + high rep penalty | Forces unlikely tokens just to avoid repetition — gibberish |
| Top-k=1 + non-zero temperature | Top-k=1 is greedy; T does nothing |
| Constrained JSON + high T | Valid JSON with semantically bad values |
| No max_tokens + agent loop | Runaway costs and runaway response time |

---

## 5.15 Resume tie-in — Sachin's stories

> **Resume tie-in (TrueBalance — Claude ML workspace):** "We had a workspace where users asked the assistant to generate dashboards from a SQL warehouse. We tried T=0.3 first thinking we wanted slight variety. We saw 4–5% of tool calls produce malformed JSON arguments. Dropping to T=0 plus enabling Claude's tool-use schema validation took that to zero. Lesson: structured output + low T + constrained decoding is non-negotiable for production tool calls."

> **Resume tie-in (ResMed — Clinical RAG):** "On the clinical chatbot, users complained answers were 'too creative' for medical content. We were running T=0.7 by default. I dropped to T=0.2, added frequency_penalty=0.3 to prevent the model looping on phrasings like 'It is recommended that...', and our RAGAS faithfulness score went from 0.72 to 0.86 on the eval set. Almost no other change — just inference parameters."

> **Resume tie-in (Real-time XGBoost p99 < 500ms):** "Not LLM, but the latency-quality tradeoff framing transfers cleanly. For LLM serving, the equivalent levers are KV-cache quantization (FP8 saves 2× memory at no quality loss on H100), speculative decoding (4–7 draft tokens, 2× throughput if draft model is well-aligned), and batching."

---

## 5.16 Master Interview Q&A — Parameter Tuning

**Q1. Walk me through every parameter you'd consider in a production LLM call.**
> Start with the sampling pipeline. Temperature reshapes the distribution. Top-p or min-p truncates the long tail. Top-k can act as a hard cap. Repetition, frequency, and presence penalties prevent loops or push diversity. Max_tokens caps cost and latency. Stop sequences terminate cleanly. Logit_bias forces or bans specific tokens. Constrained decoding for structured formats. Seed for reproducibility. Plus serving-time params like KV-cache dtype and speculative decoding. For each task I have defaults — chat is T=0.7 top-p=0.95, RAG is T=0.2 top-p=0.9, code is T=0.

**Q2. Top-p versus top-k — which do you prefer and why?**
> Top-p, almost always. Top-k is a hard cap regardless of distribution shape — it gives you the same number of candidates whether the model is highly confident or completely uncertain. Top-p is adaptive — keep just enough tokens to reach cumulative probability p, so a peaked distribution naturally yields a small nucleus and a flat distribution yields a wide one. I do combine them sometimes: top-k=50 as a safety cap, top-p=0.95 doing the smart filtering.

**Q3. What is min-p and when does it beat top-p?**
> Min-p sets a floor as a fraction of the top token's probability. If min-p is 0.05 and the top token is at 0.6, anything below 0.03 is cut. The reason it beats top-p at high temperature is that it scales with the peak — if the peak collapses to 0.05 because T is high, the threshold becomes 0.0025 automatically. Top-p doesn't know that the distribution has flattened, so top-p=0.9 at high T includes garbage. Min-p has become the default in llama.cpp and most local-LLM stacks for creative writing.

**Q4. Difference between repetition, frequency, and presence penalties?**
> Repetition penalty divides the logit of any seen token by r, typically 1.1, and stacks multiplicatively. That stacking is the gotcha — over a long generation, common words get divided into oblivion. Frequency penalty subtracts a constant times the count, so it scales linearly and stays bounded. Presence penalty subtracts a constant once if a token has appeared at all — it doesn't grow with frequency, so it pushes diversity rather than just preventing loops. For long-form chat I use frequency at 0.3. For brainstorming I add presence at 0.6.

**Q5. Your model produces gibberish around token 500 of a long generation. Diagnose.**
> Almost always compounding repetition penalty. With r=1.2, any token seen 30 times gets its logit divided by 1.2^30, which is over 200× — you've effectively banned the most common words. The model then has to pick from low-probability garbage. Three fixes: switch to frequency penalty which is additive and bounded, apply repetition penalty only over a rolling window like the last 256 tokens, or use the DRY sampler which penalizes repeated sequences instead of individual tokens. I'd also lower the rep penalty to 1.05 if I had to keep it.

**Q6. Greedy decoding versus low-temperature sampling — when each?**
> Greedy (T=0) for anything structured: code, JSON, tool calls, classification. The argmax is what you want — no randomness, no flakiness. Low-temperature sampling (T=0.2) when you want near-determinism but a small amount of robustness — RAG answers, SQL generation, where occasionally the second-ranked token is a slightly better grammatical fit. The difference matters most when the top two logits are nearly tied. Greedy makes a hard pick; low-T softens it.

**Q7. Why does T=0 in vLLM still produce different outputs across runs?**
> GPU floating-point non-determinism. Atomic operations in attention kernels, tensor-core MAC ordering, and batched fused kernels can produce logits that differ in their last few bits between runs. When two top tokens are nearly tied — which happens often — those last bits decide the argmax. To get true bit-exact reproducibility, you need deterministic CUDA kernels (slower), a fixed seed, fixed batch order, and the same hardware. Most teams accept the drift rather than pay the latency cost.

**Q8. Beam search — explain it and why it's not used for chat.**
> Beam search maintains the top-B partial sequences at each decoding step. You expand each beam to all possible next tokens, score each combined path by summed log-probability, and keep the top-B overall. At the end, you pick the highest-scoring complete sequence with a length penalty applied. It works well for translation or summarization where there's a single right answer. For open-ended chat it's terrible — it converges to safe, repetitive, "the-the-the" text because optimizing summed log-probability biases toward boring continuations. Modern serving stacks like vLLM mostly skip it.

**Q9. Stop sequences — any subtleties?**
> Two big ones. First, stops match against decoded strings, not token IDs. A stop sequence like "</answer>" can span multiple tokens, so naive token-level matching misses it. Production stacks like vLLM and TGI decode rolling and string-match. Second, partial-match handling — if the model has emitted "</ans" you don't yet know whether it's a stop. The serving layer needs to buffer and only emit confirmed-non-stop tokens to the user.

**Q10. Logit bias — what is it and when do you use it?**
> Logit bias lets you bump or suppress specific token logits manually. OpenAI's range is -100 to +100, where +100 is "force this token" and -100 is "ban this token." Classic uses: forcing a yes/no classification by biasing those tokens up, banning specific words for content moderation, or constraining a multiple-choice answer to A/B/C/D. It's a precise scalpel for cases where the prompt alone isn't enough.

**Q11. Constrained / guided decoding — how does it work?**
> You compile a schema or grammar into a finite-state automaton. At each decoding step, the FSM tells you which tokens are valid given what's been generated. You set the logits of all invalid tokens to negative infinity, then sample from what's left. The result is *guaranteed* to be syntactically valid JSON, regex match, or grammar-conformant. Libraries: Outlines, xgrammar, llama.cpp GBNF, plus OpenAI's structured outputs and Anthropic's tool-use schemas. The catch — it doesn't guarantee semantic correctness. Always pair with low temperature.

**Q12. KV-cache quantization — does it affect output quality?**
> Yes, materially. FP8 KV-cache is essentially zero quality loss on H100 and doubles concurrent requests — basically free. INT8 is fine for short context, slight degradation on long. INT4 is noticeable, especially on reasoning-heavy tasks. It's not a sampling parameter, but at scale it changes outputs as much as some sampling knobs. vLLM exposes it via `--kv-cache-dtype fp8`. I default to FP8 on H100.

**Q13. Speculative decoding — how does it work and what knobs?**
> A small "draft" model proposes the next k tokens (typically 4–7), and the large target model verifies them in a single forward pass. If accepted, you get k tokens for the cost of one. If rejected at position j, you keep tokens 1 through j-1 and fall back to greedy on token j. Knobs: which draft model (typically a same-family 1B for a 70B), and num_speculative_tokens. The health metric is acceptance rate — if it's below 50%, the draft model is too divergent. vLLM supports this via `--speculative-model`.

**Q14. Why isn't repetition penalty enough for creative writing?**
> Rep penalty is per-token. It doesn't understand *sequences* of repetition. The text "the cat sat on the mat. The cat sat on the mat." has each individual token repeated only twice — modest penalty — but the *phrase* is wholly repeated. The DRY sampler specifically penalizes repeated n-gram sequences, which is what you actually want for creative writing.

**Q15. What happens if you set top_k=1 AND top_p=0.9 simultaneously?**
> Top-k=1 wins. After top-k filtering, only one token survives, so top-p has nothing to filter — there's only one candidate. Effectively you're greedy. If you want smart filtering, don't set top-k=1; use top-k=50 or higher with top-p=0.9.

**Q16. Recommended defaults for a production RAG chatbot?**
> Temperature 0.2, top-p 0.9, top-k 50, frequency penalty 0.3 (to prevent the model looping on phrases like "Based on the context"), repetition penalty 1.0, max_tokens 800, plus prompt-engineering for citations. Low temperature is critical for grounding — high temperature lets the model drift away from the retrieved context and hallucinate. For tool-calling steps within the agent, drop to T=0 with constrained JSON.

**Q17. What is system_fingerprint and why does it matter?**
> A server-returned ID indicating the underlying model and infrastructure version. OpenAI and Anthropic both expose it. If it changes between calls, the model or infra was updated — your outputs may shift even with identical sampling parameters. Always log it for prompt-regression debugging. If you're running A/B tests on prompts, gate the analysis on fingerprint stability.

**Q18. Calibrated probabilities — what does that mean for an LLM?**
> A model is well-calibrated if its predicted probabilities match empirical frequencies — when it says "I'm 80% confident," it should be right 80% of the time. Raw LLMs are often miscalibrated after RLHF, which compresses confidence. For classification use cases, post-hoc temperature scaling (scaling logits by a learned T_calib) on a held-out set restores calibration. For generation, calibration is mostly about whether logprobs reflect true uncertainty — useful for confidence-thresholded answer / abstain decisions.

**Q19. Difference between max_tokens and max_new_tokens?**
> HuggingFace convention: max_length is total budget (prompt + output), max_new_tokens is output only. OpenAI's max_tokens is output only. Anthropic's max_tokens is also output only. The dangerous one is HuggingFace's max_length — set it without thinking and your 4000-token prompt leaves room for only 96 output tokens. Always read the specific API's docs.

**Q20. Logprobs — why request them?**
> The API returns the top-K log-probabilities at each position. Uses: scoring multiple-choice answers (compare logprobs of A vs B vs C vs D), detecting model uncertainty (high entropy at a position = "model unsure" — flag for human review), classification via direct logit comparison, and self-consistency evaluation. Small storage cost, big debuggability win. I always log them in dev, sample them in production.

---

Continue to **[Chapter 06 — Fine-tuning](06_fine_tuning.md)**.
