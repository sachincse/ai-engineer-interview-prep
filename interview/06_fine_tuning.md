# Chapter 06 — Fine-tuning LLMs
## A senior engineer's guide to SFT, RLHF, DPO, LoRA, QLoRA, and the math that makes them work

> "Have you fine-tuned LLMs?" is a coin-flip interview question — the wrong answer is yes/no without nuance. The right answer is a 5-minute story that walks through *when* you'd fine-tune, *what method* you'd pick, and *why*. This chapter gives you that story end-to-end.

---

## 6.1 Why fine-tuning exists at all

A pretrained LLM is a brilliant generalist. It's read most of the internet, knows grammar in 50 languages, can write Python, can explain quantum mechanics. But it has three problems for production use:

1. **It doesn't know your data.** Your customer support tickets, your legal templates, your medical reports — none of that was in pretraining.
2. **It doesn't know your style.** A pretrained model writes like the average internet. Your brand voice, your formatting conventions, your domain terminology — all generic.
3. **It doesn't know your task.** "Given this ticket, generate a structured triage decision in this exact JSON format" — the base model has no idea what good looks like for this.

You have three tools to fix these gaps:

- **Prompt engineering** — cheap, fast, brittle. Good for surface adaptation.
- **RAG** — gives the model your data at query time without training. Good for knowledge.
- **Fine-tuning** — bakes behavior into the weights. Good for style, format, and task patterns.

The art is knowing which tool fits the gap.

### The decision tree (memorize this for the interview)

```
                ┌────────────────────────────┐
                │  Do I need to adapt the    │
                │       model at all?        │
                └─────────────┬──────────────┘
                              │
              ┌───────────────┴────────────────┐
             yes                                no
              │                                  └──▶ ship base + prompt
              ▼
       ┌─────────────────┐
       │ Is the gap      │
       │ KNOWLEDGE-based?│ ──── yes ──▶ RAG (cheaper, auditable, updatable)
       └────────┬────────┘
                │ no (style/format/task gap)
                ▼
       ┌─────────────────┐
       │ Is prompt eng + │
       │ few-shot enough?│ ──── yes ──▶ ship prompts + version them
       └────────┬────────┘
                │ no
                ▼
       ┌─────────────────┐
       │ Have you got    │
       │ labeled data?   │
       └────────┬────────┘
                │ yes
                ▼
       ┌─────────────────┐
       │  How much data? │
       └────┬────────┬───┘
            │        │
        <50K      >100K  + deep behavior change
        pairs      pairs
            │        │
            ▼        ▼
        LoRA /    Full FT (or continued pretraining)
        QLoRA
```

**The 2026 default:** LoRA or QLoRA for 95% of fine-tuning. Full fine-tuning is mostly for frontier-model labs.

> **How to say this in an interview:** "Before fine-tuning, I always check if RAG or prompt engineering can close the gap. RAG handles knowledge, prompts handle simple style. Fine-tuning is for when you need a behavior baked into the weights — tool-call format consistency, brand voice, a complex task pattern. With less than 50K examples I default to LoRA. With 50K to a few hundred thousand and deeper behavior change, QLoRA on 4-bit base. Full fine-tune only if I'm a frontier lab."

---

## 6.2 The full LLM training pipeline — three stages

A modern instruction-tuned LLM (Llama-3-Instruct, Claude, GPT-4) goes through three training stages. Knowing this end-to-end matters because every fine-tuning method we discuss slots into one of these stages.

```
   ┌────────────────────────────────────────────────────────────────┐
   │                       Stage 1: PRETRAINING                     │
   │  Data: trillions of tokens of internet text + books + code     │
   │  Objective: next-token prediction (causal LM)                  │
   │  Output: a "base" model — knows language, no instruction skill │
   │  Cost: $10M–$100M+, 1000s of GPUs, weeks                       │
   └────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │           Stage 2: SUPERVISED FINE-TUNING (SFT)                │
   │  Data: 10K–1M (instruction, response) pairs                    │
   │  Objective: still next-token, but on instruction format        │
   │  Output: an "instruct" model — follows directions, knows JSON  │
   │  Cost: hundreds to thousands of $, days on a GPU cluster        │
   └────────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │     Stage 3: PREFERENCE OPTIMIZATION (RLHF / DPO / ORPO)       │
   │  Data: 5K–100K (chosen, rejected) preference pairs             │
   │  Objective: maximize "preferred" outputs, minimize "rejected"  │
   │  Output: aligned model — helpful, harmless, honest, polite     │
   │  Cost: hundreds of $, day or two on a GPU                       │
   └────────────────────────────────────────────────────────────────┘
```

When we say "fine-tuning" we usually mean either Stage 2 (SFT) or Stage 3 (preference optimization), or both in sequence. Almost no one does Stage 1 from scratch.

### Bonus: continued pretraining (CPT)

Sometimes inserted between Stage 1 and Stage 2 when you have a large unlabeled corpus in a niche domain (medical, legal, Arabic). You take the base model and continue Stage 1 on your domain text. Then SFT on top. Useful but expensive — only do CPT when domain knowledge is very different from the base model's training data.

---

## 6.3 Stage 2 deep-dive: Supervised Fine-Tuning (SFT)

### Why it exists

After pretraining, the model is great at *continuing* text but not at *following instructions*. SFT teaches it the chat format and how to map an instruction to a response.

### The data format

You need (instruction, response) pairs. Modern format uses a chat template:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize this article: ..."},
    {"role": "assistant", "content": "The article discusses..."}
  ]
}
```

The tokenizer applies a template that wraps each role with special tokens:
```
<|system|>You are a helpful assistant.<|end|>
<|user|>Summarize this article: ...<|end|>
<|assistant|>The article discusses...<|end|>
```

### The training objective

Standard next-token prediction (cross-entropy loss) — but only on the *assistant* tokens. We don't want to train the model to predict the user's messages; we want it to predict how to respond. Implementation: mask the loss on system/user tokens.

```
Loss = − Σ_{t ∈ assistant_tokens} log p(token_t | tokens_<t)
```

### Worked example — fine-tuning for a clinical triage assistant

Imagine Sachin's ResMed work. We want the chatbot to extract structured triage from a clinical note. We collect 5,000 examples like:

```json
{
  "messages": [
    {"role": "system", "content": "You are a clinical triage assistant. Output JSON only."},
    {"role": "user", "content": "Pt is 67yo M with SOB at rest, O2 sat 88%, hx of CHF..."},
    {"role": "assistant", "content": "{\"severity\": \"high\", \"category\": \"respiratory\", \"recommended_action\": \"escalate_to_physician\"}"}
  ]
}
```

We run SFT for 1–3 epochs at a learning rate of around 2e-5 (full FT) or 2e-4 (LoRA). After training, the model reliably produces JSON, in the right schema, with the right severity calls — without any prompt engineering at inference.

### SFT hyperparameter cheat sheet

| Hyper | Full FT | LoRA |
|-------|---------|------|
| Learning rate | 2e-5 to 5e-5 | 1e-4 to 5e-4 |
| Batch size (effective) | 32–128 | 16–64 |
| Epochs | 1–3 | 1–5 |
| LR schedule | cosine with 3% warmup | cosine with 3–10% warmup |
| Weight decay | 0.01–0.1 | 0.0–0.05 |
| Max grad norm | 1.0 | 1.0 |
| Sequence length | 4096 (or longer) | 4096 |

### Common SFT mistakes

1. **Computing loss on the user/system tokens too.** Inflates loss numbers and trains the model to imitate users — wrong.
2. **Inconsistent chat template between train and inference.** Use `tokenizer.apply_chat_template()` everywhere.
3. **Too many epochs.** SFT overfits fast. >3 epochs almost always memorizes.
4. **Not deduplicating data.** Near-duplicates inflate effective epochs invisibly.

> **How to say this in an interview:** "SFT is supervised next-token prediction on instruction-response pairs, with the loss masked to only count the assistant tokens. The data format is a chat template with system, user, and assistant roles. I run 1–3 epochs at LR 2e-5 for full FT or 2e-4 for LoRA, with a cosine schedule and 3% warmup. The two biggest gotchas are forgetting to mask the loss on user tokens — which makes the model imitate users — and inconsistent chat templates between training and inference."

---

## 6.4 Stage 3a: RLHF with PPO — the original alignment recipe

### Why it exists

SFT teaches the model the *format* and *task*. But there are subtle qualities that aren't easy to capture in (instruction, response) pairs — politeness, helpfulness, hedging style, refusing harmful requests gracefully. RLHF teaches the model these from *preference* data: which of two responses is better.

### The mental model

Imagine teaching someone to write good emails. SFT is showing them 1,000 example emails. RLHF is showing them 1,000 pairs of emails ("this one is better than that one") and asking them to internalize the *taste*. Preference data captures relative quality more naturally than absolute labels.

### The three-step RLHF pipeline

```
   STEP 1: Collect preference pairs
   ┌──────────────────────────────────────────────────┐
   │  prompt → SFT model samples N candidate answers  │
   │  Human (or LLM judge) picks chosen + rejected    │
   │  Output: dataset of (prompt, chosen, rejected)   │
   └─────────────────────────────────────────────────┬┘
                                                     │
                                                     ▼
   STEP 2: Train a Reward Model (RM)
   ┌──────────────────────────────────────────────────┐
   │  Init from SFT model, replace LM head with       │
   │  scalar regression head                           │
   │  Loss: log σ(r(chosen) − r(rejected))            │
   │  Output: RM that scores any (prompt, response)    │
   └─────────────────────────────────────────────────┬┘
                                                     │
                                                     ▼
   STEP 3: PPO loop — policy optimization
   ┌──────────────────────────────────────────────────┐
   │  policy = SFT model (initial)                    │
   │  for iteration in 1..N:                          │
   │     prompt → policy generates response           │
   │     reward = RM(response) − β·KL(policy ∥ SFT)   │
   │     update policy with PPO using reward          │
   │  Output: aligned policy model                     │
   └──────────────────────────────────────────────────┘
```

### Why each piece matters

- **Reward model** — turns "preference" into a scalar score the policy can optimize against
- **KL penalty** — prevents the policy from drifting too far from the SFT model and exploiting RM weaknesses (reward hacking)
- **PPO clipping** — keeps each policy update small, preventing instability

### The reward hacking problem

PPO + RM is famously unstable. The policy can find weird ways to fool the reward model. Example: if the RM rewards "long, polite-sounding answers," the policy learns to produce verbose, hedge-filled, content-light responses. This is *reward hacking* — the policy maximizes the proxy (RM score) without maximizing the actual goal (human preference).

The KL penalty is the main defense — it pulls the policy toward the SFT distribution. β=0.1 to 0.5 is typical.

### Why RLHF/PPO is hard in practice

1. **Three models in memory** — policy, RM, and a frozen reference (SFT) model. Memory hog.
2. **Unstable training** — small bugs in advantage normalization or learning rate cause runs to diverge silently.
3. **Reward hacking** — constant whack-a-mole.
4. **Hyperparameter-sensitive** — KL beta, PPO clip ratio, PPO epochs, learning rate all interact.

This is why DPO came along.

> **How to say this in an interview:** "RLHF with PPO has three stages. First you collect preference pairs — prompt with chosen and rejected response. Then you train a reward model that takes a prompt-response and outputs a scalar score, using a Bradley-Terry loss on the pairs. Finally you run PPO — the policy generates responses, the RM scores them, you compute a reward minus a KL penalty against the frozen SFT model, and you do PPO updates. The KL penalty is essential because without it the policy drifts and reward-hacks the RM."

---

## 6.5 Stage 3b: DPO — the simpler, more stable replacement

### Why it exists

PPO is a mess. Three models in memory, hyperparameter-sensitive, prone to reward hacking. The 2023 DPO paper (Rafailov et al.) had a beautiful insight: you can derive a closed-form loss that directly optimizes the *implicit* reward model, skipping PPO entirely.

### The math intuition (in plain English)

Start with the optimal RLHF policy under a KL penalty:
```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
```

Solve for the reward in terms of the policy:
```
r(x,y) = β · log[π*(y|x) / π_ref(y|x)] + const
```

Substitute this into the Bradley-Terry preference loss:
```
L_DPO = −E[log σ(β · log π(y_w|x)/π_ref(y_w|x) − β · log π(y_l|x)/π_ref(y_l|x))]
```

What this is doing in plain English: maximize the log-ratio between policy and reference for the *chosen* response, minimize it for the *rejected* response. No reward model. No PPO. Just supervised learning on preference pairs against a frozen reference.

### Why this is huge

```
RLHF with PPO needs:        DPO needs:
─────────────────────       ──────────────────
Policy model                Policy model
Reward model                Reference model (frozen, can be on CPU)
Reference model (frozen)
Critic / value head
PPO optimization loop       Plain backprop
Online sampling             Offline data
Days of training            Hours of training
Reward hacking risk         Direct optimization
```

DPO is now the default at most labs for the alignment stage.

### The β hyperparameter

β controls how much the policy can drift from the reference. 
- β=0.1 — relatively free, more aggressive optimization, faster but riskier
- β=0.3 — balanced
- β=0.5 — conservative, stays close to reference

Start with β=0.1. If you see catastrophic forgetting or weird outputs, increase it.

### Worked example

You SFT'd a 7B model on tickets. Now you want to align it to prefer concise, polite responses. You collect 10K preference pairs (sampling 4 responses per prompt, picking best/worst). You run DPO at β=0.1 for 1 epoch with LR 5e-7 (much lower than SFT). After training, the model produces preferred-style responses without any prompt engineering.

### Common DPO mistakes

1. **Using the same LR as SFT.** DPO needs a much lower LR — 5e-7 to 5e-6. SFT-level LR (2e-5) tanks the model.
2. **Skipping the reference model.** You need a frozen reference (usually the SFT model). Without it, DPO can't compute the KL ratio.
3. **Bad preference data.** If chosen and rejected are too similar, gradient is tiny and DPO learns nothing. If they're too different (one is just trash), you teach the model trivial discrimination.

> **How to say this in an interview:** "DPO derives a closed-form loss that's mathematically equivalent to RLHF under a KL penalty, but skips the reward model and PPO entirely. You just need a frozen reference model — usually your SFT checkpoint — and preference pairs. The loss maximizes the policy-to-reference log-ratio on chosen responses and minimizes it on rejected. β controls drift; I start at 0.1. Learning rate is much lower than SFT, around 5e-7. It's stable, simple, and now the default at most labs."

---

## 6.6 ORPO — combining SFT and preference optimization

### Why it exists

Even DPO is two stages: SFT, then DPO. ORPO (Odds Ratio Preference Optimization, 2024) merges them into a single training pass with a single dataset of preference pairs.

### The intuition

ORPO's loss = SFT loss on chosen response + λ × odds-ratio penalty pushing chosen above rejected. You train from scratch on preference data — no separate SFT phase needed.

```
L_ORPO = L_SFT(y_chosen) + λ · L_OR(y_chosen, y_rejected)

where L_OR = −log σ(log[odds(y_chosen) / odds(y_rejected)])
```

### When to use

- You have preference data but limited SFT data
- You want a one-stage pipeline
- You're optimizing for compute efficiency

ORPO closes most of the gap with SFT-then-DPO at half the training compute.

---

## 6.7 PEFT (Parameter-Efficient Fine-Tuning) — the why

Fine-tuning every parameter in a 70B model means storing gradients for every parameter (~140 GB at BF16) and optimizer states (~280 GB for AdamW). Plus the weights themselves (140 GB). Plus activations. You're at 600+ GB on a single forward-backward pass. That requires multiple H100s with FSDP.

PEFT says: most of fine-tuning is "small" updates. We don't need to update every parameter. We can freeze 99% of the model and inject small *trainable* modules, training only those.

### The hypothesis (Aghajanyan et al., 2020)

Fine-tuning updates lie in a *low-intrinsic-dimension* subspace. The "real" change between a pretrained model and its fine-tuned variant can be captured with far fewer parameters than the model has.

### PEFT methods at a glance

| Method | Trainable params | Quality vs full FT | Notes |
|--------|------------------|---------------------|-------|
| **Prompt tuning** | <0.01% | 80–90% | Soft prompts in input only; works at scale |
| **Prefix tuning** | 0.1% | 90–95% | Learnable prefix per attention layer |
| **Adapters (Houlsby)** | ~1% | 95–99% | Small MLP between transformer blocks |
| **LoRA** | 0.1–1% | 95–100% | Low-rank update on attention/MLP weights |
| **QLoRA** | 0.1–1% (on 4-bit base) | 95–100% | LoRA with 4-bit base — fits 65B on 1×48GB GPU |
| **DoRA** | LoRA + magnitude/direction split | 98–100% | Closes gap at low rank |
| **IA³** | <0.01% | 95% | Multiplicative scaling vectors |

LoRA dominates. QLoRA dominates when memory-constrained. The others are niche or research-stage.

---

## 6.8 LoRA — the workhorse explained from first principles

### The problem LoRA solves

You're fine-tuning a 7B model. Each attention projection (W_q, W_k, W_v, W_o) is a 4096×4096 matrix — 16.7M parameters per matrix. Across all the layers and projections, it's 100M+ parameters per attention block alone. Updating them all means storing a full gradient and optimizer state for each — gigabytes of memory.

But what if the *update* — the change from the pretrained W to the fine-tuned W — is itself low-rank? Then we can represent it cheaply.

### The math

Let W be a frozen pretrained weight matrix of shape d_out × d_in. Fine-tuning would replace W with W + ΔW. LoRA parameterizes ΔW as the product of two low-rank matrices:

```
ΔW = B · A
where:
  B ∈ ℝ^(d_out × r)   ← initialized to ZERO
  A ∈ ℝ^(r × d_in)    ← initialized from Gaussian
  r ≪ min(d_out, d_in), typically 8–64
```

The forward pass becomes:
```
y = W·x + (α/r) · B·A·x
        ↑                ↑
    frozen           trainable, scaled
```

Where α is a scaling hyperparameter (typically α=2r).

### Why initialize B to zero?

So that ΔW = BA = 0 at the start of training. This means at step zero, the LoRA-augmented model produces *exactly* the same outputs as the base model. Training begins from the unmodified base behavior. If both A and B were random, you'd start from a slightly corrupted base.

### Why does it work?

The key insight is that fine-tuning updates have *low intrinsic rank*. Empirically, the singular values of (W_finetuned − W_pretrained) decay rapidly — most of the update lives in a low-dimensional subspace. LoRA explicitly parameterizes this subspace.

### The picture

```
                 d_in
              ┌───────┐
              │       │
              │       │           ← Frozen W (d_out × d_in)
       d_out  │   W   │              16.7M params for 4096×4096
              │       │              NOT TRAINED
              │       │
              └───────┘
                  +
              ┌───────┐
              │       │
              │       │           ← LoRA update ΔW = BA
       d_out  │  BA   │              4096 × 16 (B)  +  16 × 4096 (A)
              │       │              = 131K params total
              │       │              TRAINED (with scaling α/r)
              └───────┘

   Decomposition of BA:
                  r
              ┌─────┐
              │  B  │  d_out × r
              │     │
              └─────┘
                       ┌──────────────┐
                       │       A      │  r × d_in
                       └──────────────┘
```

### Parameter savings (concrete)

Llama-3-8B has 32 layers, each with q/k/v/o projections of 4096×4096 plus MLP projections.

- Full fine-tuning: 8B parameters trained
- LoRA r=16 on q/v only: ~4M parameters trained (2000× fewer)
- LoRA r=16 on q/k/v/o + gate/up/down: ~30M parameters trained (270× fewer)

The training memory for LoRA is dominated by the *frozen* weights (which still need to be in memory for forward pass), but the gradient and optimizer state memory is tiny. Net: you can fine-tune a 7B model on a single 24 GB GPU.

### LoRA hyperparameter table

| Hyper | Typical | Effect |
|-------|---------|--------|
| **rank r** | 8, 16, 32, 64 | ↑ = closer to full FT, more params |
| **alpha α** | 2r | Scales ΔW; effective LR on update is α/r |
| **dropout** | 0.05 | Regularize the adapter |
| **target modules** | q,k,v,o + gate,up,down | More modules = more quality, more params |
| **learning rate** | 1e-4 to 5e-4 | Higher than full FT due to α/r scaling |
| **warmup** | 3–10% of steps | Standard cosine warmup |
| **batch size** | 16–64 effective | Via gradient accumulation |
| **epochs** | 1–3 | Watch for overfitting |

### The α/r ratio explained

The effective scale of the update is α/r. The original LoRA paper used α=r (ratio of 1). Common practice now is α=2r (ratio of 2) — slightly faster convergence. Rank-stabilized LoRA (rsLoRA) uses α/√r — better behavior at high ranks.

### Target module selection

Where in the model should you inject LoRA adapters? The answer matters for quality.

- **q_proj, v_proj only** — original LoRA paper default; cheapest; works but quality gap
- **q_proj, k_proj, v_proj, o_proj** — common best-practice; 2× params; better quality
- **+ gate_proj, up_proj, down_proj** — adds MLP projections; biggest quality lift; largest adapter

In 2026, the community default is "all attention projections + all MLP projections." The extra params are still tiny relative to the base.

### Common LoRA mistakes

1. **Wrong initialization** — if you accidentally init B from Gaussian, you start from a corrupted base. Symptom: loss spikes at step 0.
2. **Forgetting α/r scaling at inference.** When you merge LoRA into the base, you must scale ΔW by α/r. Some libraries do this automatically; some don't.
3. **Too low rank for the task.** r=4 works for narrow style adaptation but not for big behavior changes. Try r=16 first; bump to r=32 if quality lags.
4. **Targeting only q,v.** Leaves quality on the table for ~minimal extra cost.

> **How to say this in an interview:** "LoRA decomposes the fine-tuning update ΔW into a product BA where B is d_out by r and A is r by d_in, with r typically 8 to 64. The base W stays frozen. Only B and A are trained. B is initialized to zero so ΔW starts at zero and the model behaves identically to base at step zero. The effective scale of the update is alpha over r — common practice is alpha equals 2r. I default to r=16, alpha=32, targeting all attention and MLP projections, with LR 2e-4."

---

## 6.9 QLoRA — fine-tune a 65B on a single 48 GB GPU

### Why it exists

LoRA dramatically reduces the *trainable* parameter count, but the frozen base weights still have to live in memory for the forward and backward pass. For a 65B model in BF16, that's 130 GB — three H100-80GB at minimum.

QLoRA (Dettmers et al., 2023) attacks this with three innovations:

1. **4-bit NF4 quantization of the frozen base weights** — drops base from 130 GB to 33 GB
2. **Double quantization** — quantize the per-block scale factors too, saves another ~0.4 bits/param
3. **Paged optimizers** — when optimizer state spikes (gradient surges), page out to CPU via NVIDIA unified memory, preventing OOM crashes

### The picture

```
Standard LoRA:
   ┌────────────────────────────┐
   │  Frozen W (BF16, 130 GB)   │  ◄─── MEMORY HOG
   │  + LoRA adapters (BF16)    │
   └────────────────────────────┘

QLoRA:
   ┌────────────────────────────┐
   │  Frozen W (NF4, ~33 GB)    │  ◄─── 4× smaller
   │  + LoRA adapters (BF16)    │  ◄─── still BF16, no quality loss in adapter
   │  Compute: dequantize on    │
   │  the fly to BF16 for matmul│
   └────────────────────────────┘
```

### NF4 — what makes it special

NormalFloat-4 is a 4-bit datatype designed specifically for normally-distributed weights (which neural network weights approximately are). Unlike INT4 (uniform quantiles), NF4 places its 16 representable values at quantiles of a normal distribution — more precision where the data is dense, less where it's sparse. Information-theoretically optimal for Gaussian-like distributions.

### Double quantization

When you quantize a tensor block-by-block (typical block size 64), each block needs a scale factor (a float). For a 65B model with block size 64, that's a billion scale factors. Double quantization quantizes those scale factors to 8-bit, saving ~0.4 bits per original weight.

### Paged optimizers

Adam's first and second moment vectors (m and v) double the memory footprint of trainable parameters. Sometimes a gradient spike causes a momentary memory surge — without paging, OOM crash. With NVIDIA unified memory, the spike spills to CPU memory transparently. Slower for the spike but no crash.

### QLoRA configuration in practice

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 datatype
    bnb_4bit_use_double_quant=True,     # Double quantization on
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16, store in NF4
)
```

The base model loads in NF4. During the forward pass, weights are dequantized on-the-fly to BF16 for matmul. Slower than pure BF16 but only by ~10–20%, and the memory savings are 4×.

### Result

You can fine-tune a 65B model on a single 48 GB A6000. Quality is essentially the same as LoRA on BF16 base — the paper shows <1% degradation on standard benchmarks.

> **How to say this in an interview:** "QLoRA has three innovations. First, 4-bit NF4 quantization of the frozen base — NF4 is information-theoretically optimal for Gaussian-distributed weights. Second, double quantization, which quantizes the per-block scale factors to save another 0.4 bits per parameter. Third, paged optimizers using NVIDIA unified memory to spill to CPU on gradient spikes. Net result: fine-tune a 65B on one 48 GB GPU with effectively the same quality as BF16 LoRA."

---

## 6.10 DoRA, LoRA+, and other LoRA variants

### DoRA — Weight-Decomposed LoRA

Empirical observation: at low ranks (r=4 to 8), LoRA has a quality gap with full fine-tuning. DoRA closes it by decomposing the weight update into magnitude and direction:

```
W_new = m · (V / ||V||_c)
where:
   m = learnable magnitude (one scalar per output channel)
   V = direction (LoRA update applied here)
```

Only the direction gets the LoRA treatment; magnitude is a separate small parameter. DoRA closes the gap to full FT at r=4 with ~10% more parameters than LoRA.

### LoRA+

Empirical finding: B should update faster than A. LoRA+ applies a different LR multiplier (typically 16×) to B. Often gives 2× faster convergence on the same compute budget.

### rsLoRA

Rank-stabilized LoRA uses α/√r instead of α/r as the scaling factor. Better behavior at high ranks (r=128+) where α/r becomes too small.

---

## 6.11 IA³, prefix tuning, prompt tuning — the lightweight cousins

### IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Multiplies activations by learnable scaling vectors at three places (key, value, MLP intermediate). Tiny — ~0.01% of params. Quality is decent but less than LoRA. Useful when memory is critical.

### Prefix tuning

Prepends learnable "soft prefix" vectors to the keys and values of every attention layer. Around 0.1% of params. Works well at scale (>10B) but underperforms LoRA on smaller models.

### Prompt tuning

Like prefix tuning but only at the input layer — learnable soft tokens prepended to the input embeddings. The lightest of all (<0.01% params). Only works at very large scale (>40B model).

### When to use them

Honestly, in 2026 you almost always use LoRA or QLoRA. The lightweight cousins are mostly for research or extremely memory-constrained settings.

---

## 6.12 Catastrophic forgetting — the silent killer

### What it is

You fine-tune a 7B model on customer support tickets. Performance on tickets improves dramatically. You celebrate. Then someone runs MMLU and your model has lost 8 points. Math is worse. Coding is worse. The model is great at tickets and worse at *everything else*.

This is catastrophic forgetting — fine-tuning on a narrow distribution shifts the model away from its general capabilities.

### Why it happens

Gradient updates on a narrow distribution don't preserve performance on out-of-distribution tasks. The model's weights move toward a local minimum that's good for tickets but bad for everything else.

### Mitigations

1. **Mix in 10–20% general data.** Add Alpaca / OpenHermes / Tulu data to your domain SFT. Acts as a regularizer.
2. **Lower rank in LoRA.** r=8 instead of r=64 — fewer degrees of freedom for the model to drift.
3. **Lower learning rate.** Smaller updates = less drift.
4. **KL regularization.** DPO with β=0.1+ keeps the policy close to the SFT reference.
5. **Model merging.** TIES, DARE, or SLERP merge the fine-tuned model with the base — recover generality.
6. **Always evaluate.** Run MMLU, HellaSwag, GSM8K before and after fine-tuning. If you didn't measure it, you don't know if you broke it.

### Worked example

You fine-tune Llama-3-8B on 10K legal contracts with LoRA r=64, LR 5e-4, 5 epochs.
- Pre-FT MMLU: 65
- Post-FT MMLU: 51 (down 14 points — catastrophic)
- Mitigations applied: r=16, LR 2e-4, 2 epochs, mix in 20% Alpaca
- Re-evaluate: legal task quality 92% of original, MMLU = 64 (down 1 — acceptable)

---

## 6.13 Model merging — the free lunch

### Why it exists

You have two fine-tuned LoRAs: one for legal QA, one for medical QA. You want a single model that does both. Options: train a third model on combined data (expensive), serve both adapters and route (complex), or *merge* the two adapters.

### Methods

- **TIES** (Trim, Elect Sign, Merge): trim small parameter deltas, elect a sign per parameter, merge by averaging
- **DARE** (Drop And REscale): randomly drop parameter deltas, rescale survivors
- **SLERP** (Spherical Linear Interpolation): interpolate between two checkpoints along the unit sphere

### When to use

- Combining specialized adapters into one
- Recovering general capability (merge fine-tuned with base)
- Ensembling without inference-time cost

mergekit is the standard library.

---

## 6.14 The full fine-tuning recipe — step-by-step

```
1. Pick a base model
   ├── Smallest size that could plausibly work (start at 7B for cost)
   ├── Match license (commercial: Llama-3, Qwen, Mistral; non-commercial: Alpaca-licensed)
   └── Tokenizer fits your domain (multilingual? Arabic? code?)

2. Prepare data
   ├── Format as chat-template using tokenizer.apply_chat_template()
   ├── Deduplicate (exact + MinHash near-dup)
   ├── Quality filter (drop refusals, template artifacts, very-short responses)
   ├── Balance task types
   └── Split train/val/test (90/5/5)

3. Baseline evaluation
   ├── Run base model on your eval set FIRST
   ├── Record: task metric, MMLU, HellaSwag, perplexity on held-out
   └── Now you have a target to beat

4. Pick PEFT method
   ├── Default: LoRA r=16, α=32, all attn + MLP
   ├── If memory-constrained: QLoRA with NF4
   ├── If quality-critical at low rank: DoRA
   └── Full FT only with serious GPU budget

5. Train SFT
   ├── LR 2e-4 (LoRA) or 2e-5 (full FT)
   ├── Cosine schedule, 3% warmup
   ├── Effective batch 16–64 via gradient accumulation
   ├── 1–3 epochs (more = overfit)
   ├── Checkpoint every N steps
   └── Monitor: train loss ↓, val loss plateau, MMLU unchanged

6. (Optional) Stage 3 alignment
   ├── Collect 5K–50K preference pairs
   ├── DPO at LR 5e-7, β=0.1, 1 epoch
   └── Or ORPO if combining with SFT

7. Evaluate
   ├── Task metric on held-out test set
   ├── General-capability regression (MMLU, HellaSwag, GSM8K)
   ├── RAGAS / LLM-judge for qualitative dimensions
   └── Compare to baseline

8. Merge or keep adapter
   ├── Multi-tenant serving → keep adapter separate, swap per request
   ├── Single-tenant high-throughput → merge adapter into base, ship as one model
   └── Use mergekit for combining multiple adapters

9. Deploy
   ├── vLLM with --enable-lora for adapter swapping
   ├── Monitor drift / regressions post-launch
   └── Keep golden eval set for regression testing
```

---

## 6.15 Hyperparameter cheat sheet (memorize this)

| Hyper | SFT (full FT) | SFT (LoRA) | DPO |
|-------|---------------|------------|-----|
| LR | 2e-5 | 2e-4 | 5e-7 |
| Batch (effective) | 64 | 16–32 | 16 |
| Epochs | 1–3 | 1–3 | 1 |
| Schedule | cosine + 3% warmup | cosine + 3% warmup | cosine + 10% warmup |
| Weight decay | 0.01 | 0.0 | 0.0 |
| Max grad norm | 1.0 | 1.0 | 1.0 |
| LoRA rank | – | 16 | (inherits SFT rank) |
| LoRA alpha | – | 32 | – |
| LoRA dropout | – | 0.05 | – |
| Target modules | – | q,k,v,o,gate,up,down | – |
| β (DPO) | – | – | 0.1 |

---

## 6.16 Resume tie-ins — Sachin's stories

> **Resume tie-in (TrueBalance — Claude ML workspace):** "Claude is closed, so we couldn't fine-tune. We chose Claude over an open Qwen-72B + LoRA stack because tool-calling reliability mattered more than per-token cost — Claude's structured-output guarantees were better at the time. The trade-off was honest: we paid more per call but eliminated the engineering cost of running a fine-tune pipeline. We saved fine-tuning for narrow downstream tasks like SQL classification where we trained a 7B LoRA on 5K examples and beat zero-shot Claude-Haiku at 1/20 the serving cost."

> **Resume tie-in (ResMed — Clinical RAG):** "We considered fine-tuning the base LLM on clinical reports. Decided against it for two reasons. First, knowledge changes weekly — new clinical guidelines, new drug data — fine-tuning would mean retraining monthly. RAG handles that with reindexing. Second, audit and citation requirements meant we needed traceable answers. RAG gives you 'this answer came from doc 42, page 7' for free. Fine-tuning bakes the answer into weights with no audit trail. RAG was the right call."

> **Resume tie-in (concrete LoRA story):** "I LoRA fine-tuned a Mistral-7B on internal ticket triage data — 10K instruction-response pairs, r=16, α=32, target all attention and MLP projections, 2 epochs at LR 2e-4. Beat zero-shot Claude-Haiku on our eval set at one-twentieth the serving cost. Catastrophic-forgetting check: MMLU dropped from 62 to 60 — within tolerance. Key was mixing 15% Alpaca data into the SFT set."

---

## 6.17 Master Interview Q&A — Fine-tuning

**Q1. When would you fine-tune versus use RAG versus just prompt-engineer?**
> Three different problems. RAG is for knowledge gaps — your model doesn't know your data, RAG plugs it in at query time. Prompt engineering is for surface adaptation — small format and tone tweaks. Fine-tuning is for *baking behavior into weights* — when you need consistent tool-call format, brand voice, complex task patterns the model doesn't have. My decision tree: try prompts first, then RAG for knowledge, then fine-tune only when those aren't enough. Within fine-tuning, default to LoRA at <50K examples, QLoRA when memory-constrained, full FT only with serious hardware.

**Q2. Walk me through LoRA mathematically.**
> Take a frozen weight matrix W of shape d_out by d_in. Fine-tuning would replace it with W plus delta-W. LoRA parameterizes delta-W as the product B times A, where B is d_out by r, A is r by d_in, and r is small — typically 8 to 64. The forward pass becomes y = Wx plus alpha-over-r times BAx. B is initialized to zero so delta-W is zero at step zero, meaning the model starts at exact base behavior. A is Gaussian-initialized. Only A and B are trained. Alpha over r is the effective scaling on the update; common practice is alpha equals 2r.

**Q3. Why does LoRA work — what's the underlying assumption?**
> The intrinsic-rank hypothesis from Aghajanyan 2020. Fine-tuning updates have low intrinsic dimensionality — empirically, the singular values of (W_finetuned minus W_pretrained) decay rapidly, so most of the update lives in a low-dimensional subspace. LoRA explicitly parameterizes that subspace with the rank-r product BA. Training only that captures most of the meaningful change. The cost: about 0.1 to 1 percent of the parameters, with 95 to 100 percent of full-FT quality on most tasks.

**Q4. Walk me through QLoRA's three innovations.**
> First, 4-bit NF4 quantization of the frozen base weights. NF4 is information-theoretically optimal for Gaussian-distributed weights — it places its 16 quantization levels at quantiles of a normal distribution rather than uniformly. Second, double quantization: the per-block scale factors themselves get 8-bit quantized, saving another 0.4 bits per parameter. Third, paged optimizers using NVIDIA unified memory — when optimizer state spikes from gradient surges, it spills transparently to CPU memory instead of OOM-crashing. Net result: fine-tune a 65B model on a single 48GB GPU with essentially the same quality as BF16 LoRA.

**Q5. RLHF with PPO versus DPO — what's the difference?**
> RLHF with PPO has three stages: collect preference pairs, train a reward model on them with a Bradley-Terry loss, then run PPO where the policy generates responses, the reward model scores them, you compute reward minus a KL penalty against a frozen reference, and update the policy. DPO derives a closed-form loss that's mathematically equivalent under the KL constraint, so you skip the reward model and PPO entirely. You just need a frozen reference and preference pairs, train with plain backprop. DPO is more stable, simpler, faster, and avoids reward hacking. It's now the default at most labs.

**Q6. DPO loss — explain in plain English.**
> DPO maximizes the policy-to-reference log-ratio on chosen responses and minimizes it on rejected responses. Concretely, the loss is the negative log-sigmoid of beta times log-pi-policy-of-chosen-over-pi-ref-of-chosen minus beta times log-pi-policy-of-rejected-over-pi-ref-of-rejected. Beta controls how much the policy can drift from the reference — small beta is more aggressive, larger beta is more conservative. I start at beta=0.1. Learning rate is much lower than SFT, around 5e-7.

**Q7. What is catastrophic forgetting and how do you mitigate it?**
> Fine-tuning on a narrow distribution shifts the model away from its general capabilities. You fine-tune on tickets and MMLU drops eight points. Mitigations are layered. Mix in 10 to 20 percent general data — Alpaca, OpenHermes — as a regularizer. Lower rank in LoRA — fewer degrees of freedom for drift. Lower learning rate. Use KL regularization, like DPO with beta 0.1 or higher. Use model merging — TIES or DARE — to recover generality. And critically, always measure: run MMLU and HellaSwag before and after every fine-tune. If you didn't measure, you didn't notice.

**Q8. Why is the LoRA learning rate higher than full fine-tuning?**
> The effective scale on the update is alpha-over-r. With alpha=32 and r=16, the actual update applied to the weights is 2x the raw gradient. To produce a similar magnitude of weight change as full FT — which doesn't have that scaling — you need a higher raw learning rate. Hence 2e-4 for LoRA versus 2e-5 for full FT. Roughly the same effective learning rate on the underlying weights, just expressed differently.

**Q9. Target modules in LoRA — which layers do you adapt?**
> Original LoRA paper: q_proj and v_proj only. Cheapest, works, but quality gap. Common best practice now: q, k, v, o — all four attention projections. Roughly doubles parameters, meaningfully better quality. State-of-the-art: add the MLP projections too — gate, up, down on Llama-style models. That's the biggest quality lift. Default in 2026 is "all attention plus all MLP projections," because the extra parameters are still tiny relative to the base.

**Q10. Your LoRA fine-tuned model catastrophically forgot general capabilities. What do you do?**
> First, measure the damage — run MMLU, HellaSwag, GSM8K. Quantify it. Then mitigations in order of cheapness. Mix 15 to 20 percent general instruction data into your SFT set — Alpaca or OpenHermes. Lower the rank — try r=8 if you were at r=64. Lower the learning rate by half. Reduce epochs — most catastrophic forgetting happens in epochs 2-plus. Use KL-regularized training like DPO with beta=0.1 if you're past SFT. As a last resort, model-merge with the base using TIES or DARE — that almost always recovers some general capability.

**Q11. Loss plateaus during LoRA training. What do you check?**
> Six things in order. First, data quality — refusals, template artifacts, very-short responses can poison training. Second, chat-template consistency between train and inference. Third, learning rate too low — bump to 5e-4 for LoRA. Fourth, rank too low — try r=32. Fifth, base model mismatch — maybe you need continued pretraining first if your domain is far from base training data. Sixth, gradient clipping masking an exploding loss — log unclipped grad norm and check.

**Q12. How do you serve 50 LoRA adapters efficiently?**
> Multi-LoRA serving. Load one base model in VRAM, swap tiny adapter weights per request. Adapter swap is milliseconds — it's just a few MB of memcopy. vLLM supports this with the enable_lora flag. LoRAX and Punica also do it. The tricky part is concurrent requests with different adapters in the same batch — that needs batched LoRA kernels like SGMV. Naive implementations serialize, killing throughput. Net win: one base model in memory regardless of tenant count, marginal cost per tenant is just adapter training.

**Q13. SFT versus continued pretraining versus DPO — when each?**
> CPT, continued pretraining, is for raw domain knowledge — you have lots of unlabeled in-domain text and the base model doesn't know the domain. SFT is for instruction-following and format — you have instruction-response pairs and want the model to learn the task pattern. DPO is for preference and style — you have pairs of better-versus-worse responses and want to teach taste. The classic pipeline is CPT optional, then SFT, then DPO. Most production teams skip CPT.

**Q14. Why do small, high-quality datasets often beat large messy ones?**
> The LIMA paper showed 1K curated examples can match or beat 50K messy sets. Quality beats quantity at moderate scales. The reasons: messy data teaches the model bad patterns alongside good ones, near-duplicates inflate effective epochs invisibly, refusals and template artifacts are pure noise. A small, deduplicated, quality-filtered, format-consistent dataset gives the model clean signal. The hygiene pipeline is the highest-leverage thing in fine-tuning — dedup with MinHash, drop refusals, validate chat templates, balance task types.

**Q15. Preference data sources for DPO?**
> Three options. Human pairwise ranking is gold standard but expensive — 0.10 to 1 dollar per pair, depending on rater quality. LLM-as-judge with GPT-4 or Claude is much cheaper and scales — but inherits the judge's biases. Synthetic from best-of-N sampling of your own SFT model — generate four responses per prompt, take best and worst per a quality metric. Most production DPO uses a mix: synthetic for volume, LLM-judge for quality, human for the most sensitive dimensions like safety. Typical dataset size is 5K to 50K pairs.

**Q16. What happens if you don't initialize B to zero in LoRA?**
> You start training from a slightly corrupted base. B and A both random means delta-W = BA is non-zero at step zero, so the model's outputs differ from the base. The loss spikes at step zero, the optimizer has to first recover the base behavior before learning the adaptation, training is unstable. Always init B to zero (or A to zero, the other choice — but always one of them). The non-zero one is Gaussian. This guarantees identity behavior at start.

**Q17. ORPO — what does it do differently?**
> ORPO (Odds Ratio Preference Optimization, 2024) merges SFT and preference optimization into a single training pass. The loss is SFT cross-entropy on the chosen response plus an odds-ratio penalty pushing chosen above rejected. You train from scratch on preference data — no separate SFT phase. Closes most of the gap with SFT-then-DPO at half the training compute. Good when you have preference data but limited pure-SFT data.

**Q18. Model merging — TIES, DARE, SLERP — what are they?**
> Three methods to combine fine-tuned checkpoints without retraining. TIES — Trim, Elect Sign, Merge — trims small parameter deltas, elects a single sign per parameter across checkpoints, merges by averaging the survivors. DARE — Drop And REscale — randomly drops parameter deltas with probability p, rescales survivors to maintain magnitude. SLERP — Spherical Linear Interpolation — interpolates between two checkpoints along the unit sphere. Use case: combine a legal-domain LoRA with a medical-domain LoRA into one adapter that handles both. mergekit is the standard library.

**Q19. RLAIF — what is it?**
> RL from AI Feedback. Replace human labelers with an LLM judge for preference data collection. Cheaper, scales massively, but the judge's biases propagate into the policy — so you have to carefully calibrate the judge. Constitutional AI (Anthropic) is a specific RLAIF recipe with a written constitution as the judge's rubric, ensuring the bias is *explicit*. Most modern alignment uses RLAIF for volume, with human preferences for the highest-stakes dimensions.

**Q20. How do you evaluate a fine-tuned model end-to-end?**
> Five layers. First, task metric on held-out test set — accuracy, F1, BLEU, RAGAS faithfulness, whatever your task uses. Second, general-capability regression — MMLU, HellaSwag, GSM8K — to detect catastrophic forgetting. Third, safety — toxicity classifier, refusal rate on a curated set of harmful prompts. Fourth, calibration — Expected Calibration Error if you care about confidence. Fifth, human or LLM-judge qualitative eval on a representative sample. Always establish the pre-fine-tune baseline for all five so you know what changed.

---

Continue to **[Chapter 07 — RAG](07_rag.md)**.
