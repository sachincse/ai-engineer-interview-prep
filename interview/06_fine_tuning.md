# Chapter 06 — Fine-tuning LLMs
## Full FT, LoRA, QLoRA, DoRA, PEFT — when, why, how

> The JD says "productionizing LLM-based applications" — which increasingly means fine-tuning (or at least adapting) open-weight models. This chapter covers the math, the recipes, and the gotchas.

---

## 6.1 The fine-tuning decision tree

```
Do you need to adapt an LLM?
           │
           ▼
 ┌─────────────────────┐
 │ Is RAG enough?      │ → Yes → Use RAG. It's cheaper, auditable, updatable.
 └──────────┬──────────┘
            │ No
            ▼
 ┌─────────────────────┐
 │ Is prompt engineering│ → Yes → Ship prompts. Version them.
 │ + few-shot enough?   │
 └──────────┬──────────┘
            │ No
            ▼
 ┌─────────────────────┐
 │ How much data?      │
 └─────┬────────────┬──┘
       │            │
   <50k pairs    >100k pairs OR deep behavior change needed
       │            │
       ▼            ▼
   LoRA / QLoRA    Full FT or continued pretraining
```

**Default in 2026:** LoRA or QLoRA for 95% of use cases. Full FT only for frontier-model teams with multi-GPU clusters.

---

## 6.2 The four fine-tuning modes

| Mode | Data | What it teaches | When to use |
|------|------|-----------------|-------------|
| **Continued Pretraining (CPT)** | Raw domain text, unlabeled | Domain knowledge, vocabulary, style | Domain shift (medical, legal, financial, Arabic) |
| **Supervised Fine-tuning (SFT)** | (instruction, response) pairs | Task-following, format | General task adaptation |
| **Preference Optimization (DPO/RLHF)** | Preference pairs (chosen, rejected) | Style preferences, safety, subtle quality | Final polish, alignment |
| **Reinforcement Fine-Tuning (RFT)** | Task + reward function | Complex objective optimization | o1-style reasoning models |

**Order:** CPT → SFT → DPO (or RFT). Most production deployments skip CPT (the base already knows the domain).

---

## 6.3 Full fine-tuning — the expensive option

All parameters update. For Llama-3-70B at BF16:
- Weights: 140 GB
- Gradients: 140 GB
- AdamW optimizer states (m, v): 280 GB
- Activations (recomputed or checkpointed): ~30 GB
- **Total: ~600 GB**

Needs 8× H100-80GB with FSDP or DeepSpeed ZeRO-3. Training cost: thousands of dollars per epoch.

**When worth it:** Frontier model providers, teams with fleet hardware, deep behavioral changes needed.

---

## 6.4 Parameter-Efficient Fine-tuning (PEFT) — the overview

**Idea:** freeze 99%+ of the base model; inject small trainable modules.

### The hypothesis (Aghajanyan et al., 2020)
Fine-tuning updates have **low intrinsic rank** — you don't need to change every parameter. A tiny adapter captures most of the adaptation.

### PEFT methods at a glance

| Method | Params trained | Quality vs full FT | Notes |
|--------|---------------|---------------------|-------|
| **Prompt tuning** | <0.01% (soft prompts) | 80-90% | Only works at large scale |
| **Prefix tuning** | 0.1% | 90-95% | Adds learnable prefix to every attention layer |
| **Adapters** (Houlsby) | ~1% | 95-99% | Small MLP inserted between transformer layers |
| **LoRA** | 0.1-1% | 95-100% | Low-rank ΔW into attention projections |
| **QLoRA** | LoRA on 4-bit quantized base | 95-100% | Train 65B on 1×48GB GPU |
| **DoRA** | Like LoRA + magnitude/direction split | 98-100% | Slightly more params, closer to full FT |
| **IA³** | ~0.01% | 95% | Multiplicative scaling vectors |

---

## 6.5 LoRA — the workhorse

### Math

For a linear layer y = W·x (W ∈ ℝ^(d_out × d_in)):
```
y = W·x + B·A·x
         ↑
       learnable low-rank update
```
where:
- A ∈ ℝ^(r × d_in)   ← init from Gaussian
- B ∈ ℝ^(d_out × r)  ← init to ZERO (so ΔW = 0 at start)
- r ≪ min(d_in, d_out), typically 8-64

### Why the zero-init on B?
The update ΔW = BA must start at **zero** so training begins from the exact base-model behavior. If B or A were both random, you'd be training from a slightly-corrupted base.

### Parameter count savings

For a 4096×4096 attention projection:
- Full FT: 16.7M params
- LoRA r=16: 2 × 4096 × 16 = 131K params (**127× fewer**)

Across all attention projections in a 7B model, LoRA ≈ 0.1-1% of full params.

### Target modules — where to inject?

Options:
- **q_proj, v_proj** (Hu et al. default — cheapest, works)
- **q_proj, k_proj, v_proj, o_proj** (common best-practice, ~2× params)
- **+ gate_proj, up_proj, down_proj** (LLaMA-style MLP — biggest quality lift, largest adapter)

### Hyperparameters that matter

| Hyper | Typical | Effect |
|-------|---------|--------|
| rank r | 8-64 | ↑ = closer to full FT, ↑ params |
| alpha α | 2r (common heuristic) | Scale of ΔW (alpha/r is the effective LR on ΔW) |
| dropout | 0.05 | Regularize |
| LR | 1e-4 to 5e-4 | **Higher than full FT** |
| Warmup | 3-10% of steps | Standard |
| Target modules | attn + MLP | MLP modules add significant quality |

### alpha/r ratio
The effective scaling is α/r. LoRA paper uses α = r (ratio 1); common practice uses α = 2r (ratio 2) for slightly faster convergence. Rank-stabilized LoRA (rsLoRA) uses α / √r for better scaling at high r.

---

## 6.6 QLoRA — fine-tune a 65B on a consumer GPU

**Three innovations** (Dettmers et al., 2023):

### 1. 4-bit NF4 base weights
NormalFloat-4 is an information-theoretically optimal 4-bit datatype for normally-distributed weights (which NN weights approximately are). The base model stays frozen in 4-bit; only LoRA adapters are in BF16.

### 2. Double quantization
The quantization constants (per-block scale factors) themselves get quantized. Saves ~0.4 bits/param.

### 3. Paged optimizers
When optimizer state overflows GPU RAM (rare but happens on gradient spikes), offload pages to CPU via NVIDIA unified memory. Prevents OOM crashes.

### Result
Fine-tune a 65B model on a single 48GB GPU. ~Same quality as LoRA on BF16 base.

### QLoRA knobs
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # compute in BF16, store in NF4
)
```

---

## 6.7 DoRA — Weight-Decomposed LoRA

**Problem with LoRA:** At low ranks, quality gap with full FT is visible.

**Fix (DoRA, 2024):** Decompose weight update into **magnitude** and **direction**:
```
W = m · (V / ||V||_c)
```
- m: learnable magnitude (scalar per output channel)
- V: direction (apply LoRA only here)

At low ranks (r=4-8), DoRA closes the gap to full FT. ~10% more params than LoRA, now standard in newer PEFT releases.

---

## 6.8 LoRA+ — different LRs for A and B

Empirical finding: B should update faster than A. LoRA+ applies a different LR multiplier (typically 16×) to B. Often 2× faster convergence on the same compute budget.

---

## 6.9 Serving multiple LoRA adapters

Big cost win for multi-tenant scenarios:

- **Strategy:** Load ONE base model in GPU memory; swap tiny adapters per request.
- Adapter swap is ~milliseconds (memcopy of a few MB).
- **vLLM** (`enable_lora=True, max_loras=...`), **LoRAX**, **Punica** support this.
- Watch: concurrent requests with *different* adapters need batched LoRA kernels (SGMV). Naive implementations serialize.

Example: 50 tenant-specific adapters on one Llama-3-70B endpoint. Cost: 1× base model; variable cost per tenant = adapter training only.

---

## 6.10 Instruction tuning data — the 2026 playbook

Quality > quantity. A 1K well-curated set often beats a 100K messy set (LIMA, "Less is More for Alignment").

### Sources
- **Open**: Alpaca, OpenHermes, WildChat, Tulu-3, Orca
- **Synthetic**: Distill from GPT-4 / Claude Sonnet (filter aggressively)
- **Human-curated**: Expensive but highest quality

### Dataset hygiene
- Dedup (exact + MinHash)
- Filter out refusals, template artifacts, very short responses
- Verify chat template consistency
- Balance task types (don't over-index on math if deploying a chatbot)

---

## 6.11 Preference data for DPO

Format:
```json
{
  "prompt": "Explain quantum entanglement simply.",
  "chosen": "Quantum entanglement is when two particles...",
  "rejected": "In quantum mechanics, the wavefunction..."
}
```

### Preference sources
- Pairwise human ranking (~$0.1-1 per pair)
- LLM-judge preferences (GPT-4o / Claude as judge)
- Synthetic (best-of-N + worst-of-N sampling from the SFT model)

### Typical data size
5K-50K preference pairs for DPO.

---

## 6.12 Catastrophic forgetting — the silent killer

**Symptom:** Fine-tuned model gets great on target task but regresses on general capabilities (MMLU, HellaSwag, code).

### Mitigations
1. **Mix 10-20% general data** into domain SFT (Alpaca, OpenHermes)
2. **Lower rank, lower LR** (if using LoRA)
3. **KL regularization** (DPO with β=0.1)
4. **Model merging** (TIES, DARE) — merge base + fine-tuned to recover generality
5. **Measure it** — always eval on MMLU / HellaSwag before + after

---

## 6.13 Model merging — the free lunch

**TIES** (Trim, Elect Sign, Merge), **DARE** (Drop And REscale), **SLERP** — merge multiple fine-tuned checkpoints to combine capabilities without retraining.

Use case: you fine-tuned one LoRA for legal, one for medical. Merge their deltas → a single adapter good at both. mergekit is the standard library.

---

## 6.14 The fine-tuning recipe (step-by-step)

```
1. Pick base model
   - Size: smallest that could work
   - Match license (commercial — Llama, Qwen, Mistral)
   - Tokenizer: check Arabic/multilingual needs

2. Prepare data
   - Format as chat-template
   - Dedup (Datasets + hash)
   - Quality filter
   - Split train/val/test (90/5/5)

3. Baseline eval
   - Run the base model on your eval set FIRST
   - Record: task metric, MMLU (general), perplexity on held-out

4. Pick PEFT method
   - LoRA default
   - QLoRA if VRAM-constrained
   - DoRA for lowest rank
   - Full FT only with serious hardware

5. Train
   - r=16, alpha=32, target all attn + MLP projections
   - LR 2e-4 (LoRA) or 5e-5 (full FT)
   - batch 16-32 effective (via gradient accumulation)
   - 1-3 epochs (more = overfit)
   - Checkpoint every N steps
   - Monitor: train loss ↓, val loss plateau, MMLU unchanged

6. Evaluate
   - Task metric on test set
   - MMLU / HellaSwag on general
   - RAGAS / LLM-judge on qualitative dimensions

7. Merge / save
   - Merge LoRA to base (optional — for single deployment)
   - Or keep adapter separate for multi-tenant serving

8. Deploy
   - vLLM with --enable-lora for adapter swapping
   - Monitor drift / regressions post-launch
```

---

## 6.15 Interview Q&A — Fine-tuning

**Q1. LoRA intuition — why does it work?**
> Fine-tuning updates lie in a low-intrinsic-rank subspace — you don't need to move every parameter. LoRA parameterizes ΔW = BA with rank r ≪ dim. Train 0.1-1% of params; match full FT on most tasks.

**Q2. LoRA — explain the math.**
> y = Wx + BAx, where W is frozen, A is r×d_in, B is d_out×r. B is init to 0 so ΔW=0 at start (exact base behavior). A is init from Gaussian. Only A and B train. Scaling factor α/r is applied at inference.

**Q3. Why alpha = 2r and not alpha = r?**
> Empirical. alpha/r is the effective LR on ΔW. alpha=2r roughly doubles the effective LR, speeding convergence without obvious downsides. rsLoRA uses alpha/√r for better scaling at high r.

**Q4. QLoRA's three innovations?**
> (1) 4-bit NF4 quantization of frozen base — information-theoretically optimal for Gaussian-ish weights. (2) Double quantization — quantize the quantization constants. (3) Paged optimizers — CPU offload on OOM spikes. Net: fine-tune 65B on one 48GB GPU.

**Q5. DoRA — how does it differ from LoRA?**
> Weight-decomposed: splits ΔW into magnitude (learnable scalar per channel) and direction (LoRA applied here). Closes the gap to full FT at low ranks (r=4-8). ~10% more params than LoRA.

**Q6. Target modules — which layers to adapt?**
> Attention q/v is the minimum. Add k/o for better results. For Llama-style MLPs, adding gate/up/down gives the biggest lift. Default: all attention + MLP projections.

**Q7. Good LoRA hyperparameters?**
> r=16, alpha=32, dropout=0.05, LR 2e-4 with cosine schedule, effective batch size 16-32 via accumulation, 1-3 epochs, target q/k/v/o + gate/up/down.

**Q8. Full FT vs PEFT — decision?**
> PEFT (LoRA/QLoRA) for <50K examples, surface adaptation, multi-tenant serving, limited GPU. Full FT for deep behavioral change (new language), >100K examples, and serious hardware. PEFT is the default in 2026.

**Q9. [Gotcha] Your LoRA fine-tuned model catastrophically forgot general capabilities. Fix?**
> (1) Mix 10-20% general instruction data into your SFT set. (2) Lower rank, lower LR. (3) Shorter training (fewer epochs). (4) KL-regularized methods (DPO with β=0.1). (5) Model merging (TIES, DARE) of base + fine-tuned. Always eval on MMLU/HellaSwag before and after.

**Q10. [Gotcha] Training loss goes down fine but validation loss diverges early. What's happening?**
> Overfitting — train on train, but model memorizes. LR too high OR rank too high OR data too small. Mitigations: lower LR, lower rank, add weight decay, early stop, augment data.

**Q11. [Gotcha] Loss plateaus at some value. Won't budge. What do you check?**
> (1) Data quality (refusals/templates poisoning). (2) Chat template consistency. (3) LR too low — try 5e-4 for LoRA. (4) Rank too low — bump to 32. (5) Base-model mismatch — maybe you need continued pretraining first. (6) Grad clipping masking an exploding loss.

**Q12. How do you serve 50 LoRA adapters efficiently?**
> Multi-LoRA serving — one base model in VRAM, adapters swapped per request. vLLM with `enable_lora`, LoRAX, or Punica. Adapter swap is ~ms. For concurrent requests with different adapters, need batched LoRA kernels (SGMV).

**Q13. Instruction tuning vs continued pretraining vs DPO?**
> CPT: raw domain text, teaches knowledge. SFT: (instruction, response), teaches format/task. DPO: preference pairs, teaches preferences/safety. Pipeline: CPT (optional) → SFT → DPO.

**Q14. LIMA / data quality vs quantity?**
> LIMA showed 1K curated examples can match or beat 50K messy sets. Dataset hygiene (dedup, quality filter, format consistency) is the #1 lever. Quality > quantity, at least up to ~10K pairs.

**Q15. DPO vs PPO vs RLHF?**
> PPO-based RLHF: train reward model on pairs, then PPO against RM with KL penalty. Expensive, unstable. DPO: closed-form loss directly on preference pairs, no RM. Simpler, more stable, similar quality. DPO is the 2025-2026 default.

**Q16. KL β in DPO — what's the range?**
> 0.1-0.5 typical. Too low → model drifts, loses general capability. Too high → no improvement. 0.1 is a reasonable start; tune on eval.

**Q17. Preference data sources?**
> Human pairwise ranking (~$0.1-1/pair). LLM-judge (GPT-4o, Claude as judge). Synthetic from best-of-N sampling of SFT model. 5K-50K pairs typical for DPO.

**Q18. Model merging — TIES / DARE / SLERP?**
> Merge fine-tuned checkpoints without retraining. TIES: trim small deltas, elect sign, merge. DARE: random drop + rescale. SLERP: spherical interpolation. mergekit is the standard library. Combine domain-specific LoRAs into one adapter.

**Q19. [Gotcha] You fine-tuned with LoRA but the merged model is slower at inference. Why?**
> If you don't merge back (keep LoRA separate), every attention projection does an extra matmul for BA. Merging (adding ΔW into W) eliminates this overhead but loses the adapter-swap benefit. Trade-off: multi-tenant serving → keep separate; single-model serving → merge.

**Q20. Why is the LR for LoRA higher than full FT?**
> LoRA updates are scaled by α/r (e.g., 32/16 = 2). Full FT updates the underlying weights directly. To compensate for the scaling, LoRA uses higher raw LR (2e-4 vs 5e-5 for full FT).

**Q21. [Gotcha] How do you evaluate a fine-tuned model end-to-end?**
> (1) Task metric on held-out test (accuracy, F1, BLEU, whatever fits). (2) General-capability regression — MMLU, HellaSwag, GSM8K. (3) Safety — toxicity classifier, refusal rate. (4) Calibration — ECE. (5) Human / LLM-judge qualitative eval. Always establish baseline (pre-FT) for all.

**Q22. How do you design a preference data collection pipeline?**
> Sample N responses per prompt (best-of-N from SFT model). Human or LLM judge picks preferred / rejected. Inter-annotator agreement target ≥0.6. Balance topic coverage. For sensitive dimensions (safety, factuality), use specialized raters.

**Q23. What is RLAIF?**
> RL from AI Feedback — replace human labelers with an LLM judge. Cheaper, scales, but the judge's biases propagate into the policy. Constitutional AI (Anthropic) is a specific RLAIF recipe with a written constitution as the judge's rubric.

**Q24. When would you do continued pretraining?**
> Domain is very different from base (Arabic legal, specialized medical terminology). You have a lot of unlabeled in-domain text. You want the model to "know" the domain before you teach it formats via SFT.

**Q25. How does instruction-tuning data format affect inference?**
> Model learns the specific chat template used during SFT. Mismatch at inference = degraded performance. Always use `tokenizer.apply_chat_template()` matching the SFT format.

---

## 6.16 Resume tie-ins

- **"Architected AI-powered ML workspace assistant on Claude"** — Claude is closed, so you used prompting + tool-calling rather than fine-tuning. Be explicit: "we considered fine-tuning an open model (Qwen-72B + LoRA) but the tool-calling quality gap and latency made Claude via API the better choice for V1."
- **"RAG-based chatbot at ResMed"** — frame the RAG-vs-FT choice: RAG chosen because clinical knowledge is volatile (new docs weekly), fine-tuning would require monthly retraining. Also: audit/citation requirements favored RAG.
- **Your LoRA/QLoRA skills listing** — prepare one concrete story: "I LoRA fine-tuned a Mistral-7B on our internal ticketing data (10K pairs); r=16, 1 epoch, 2e-4 LR; it beat zero-shot Claude-Haiku on our eval set at 1/20 the serving cost."

---

Continue to **[Chapter 07 — RAG](07_rag.md)**.
