# Chapter 22 — 48-Hour Intensive Study Plan (Tuesday → Thursday)

> **Why this chapter exists:** Your interview is **Thursday 2026-04-30**. Today is **Tuesday 2026-04-28**. The original 14-day plan in [00_index.md](00_index.md) doesn't fit. This is the realistic, opinionated 48-hour version.

**Strategy:** Drop everything you already know cold. Use the time to (1) drill the high-probability gaps, (2) rehearse stories out loud, (3) sleep enough that you arrive sharp.

---

## Tuesday (today, 2026-04-28) — Diagnosis + Foundations

### Morning block (3 hours, ~9:00–12:00)

**Goal:** baseline yourself and patch the biggest holes.

| Time | Task | Doc |
|------|------|-----|
| 30 min | **Read [Chapter 19 — Avrioc Company Intel](19_avrioc_company_intel.md) carefully**. Memorize the four product names: MyWhoosh, Comera, Labaiik, Hyre. | 19 |
| 30 min | **Skim [Chapter 18 — Cheatsheet](18_cheatsheet.md)**. If something on the cheatsheet is unfamiliar, mark it for the afternoon. | 18 |
| 60 min | **Re-read [Chapter 02 — Transformers](02_transformers.md) and [Chapter 03 — LLMs](03_llms.md)** at speed, *only* to refresh attention math, KV-cache, and decoding. | 02, 03 |
| 60 min | **[Chapter 09 — Model Optimization](09_model_optimization.md)** — INT8/INT4 quantization (GPTQ, AWQ), pruning, distillation. Read the *interview Q&A* sections out loud. | 09 |

### Afternoon block (3 hours, ~13:30–16:30)

**Goal:** lock down the JD-named stack — Kubernetes, Ray, vLLM, FastAPI, LangChain.

| Time | Task | Doc |
|------|------|-----|
| 75 min | **[Chapter 12 — Kubernetes & Ray](12_kubernetes_ray.md)** — focus on: pods/deployments/services, HPA vs KEDA, GPU operator, Ray Core/Serve/Train/Tune, when to choose each. | 12 |
| 75 min | **[Chapter 13 — Frameworks](13_frameworks.md)** — focus on FastAPI async patterns, vLLM (PagedAttention, continuous batching), LangChain agents. | 13 |
| 30 min | **[Chapter 21 — Slurm + DGX](21_slurm_dgx_hpc.md)** — read once carefully. Memorize the sbatch script in §2.2. | 21 |

### Evening block (2.5 hours, ~18:00–20:30)

**Goal:** live coding warmup. **DO actual coding, don't just read.**

| Time | Task | Doc |
|------|------|-----|
| 45 min | **[Chapter 20 — Live Coding Bank §A](20_live_coding_bank.md)**. Code A.1 (rotate array), A.3 (sliding window), A.6 (binary search), A.8 (LRU) **from scratch on paper**. | 20 |
| 45 min | **§B.1 + B.2 — attention from scratch.** Write multi-head attention in PyTorch in a fresh notebook. No copying. | 20 |
| 30 min | **§B.3 — FastAPI streaming endpoint with cancellation.** Write it and run it locally. | 20 |
| 30 min | **§C — quick warm-ups, all 10**. Solve in your head, talk out loud. Tape yourself. | 20 |

### Night block (60 min before bed)

| Time | Task | Doc |
|------|------|-----|
| 30 min | **[Chapter 19 §4 — Reverse-chronology drill](19_avrioc_company_intel.md#4-the-reverse-chronology-project-walkthrough-drill-critical)** — practice your TrueBalance + ResMed walkthroughs **out loud, timed**. Aim 4-5 min for TrueBalance, 3-4 min for ResMed. | 19 |
| 30 min | **[Chapter 17 — Behavioral & UAE](17_behavioral_hr.md)** — read once. Lock in your "Why Abu Dhabi" answer. | 17 |

**Bedtime: by 22:30.** No screens after 22:00.

---

## Wednesday (2026-04-29) — Depth + System design + Mock

### Morning block (3 hours, ~9:00–12:00)

**Goal:** RAG, embeddings, fine-tuning depth — these are common deep-dive territory.

| Time | Task | Doc |
|------|------|-----|
| 60 min | **[Chapter 07 — RAG](07_rag.md)**. Naive → advanced RAG, hybrid search, reranking, RAGAS, GraphRAG. | 07 |
| 45 min | **[Chapter 04 — Embeddings](04_embeddings.md)** — read interview Q&A only if time-pressed. | 04 |
| 60 min | **[Chapter 06 — Fine-tuning](06_fine_tuning.md)** — LoRA/QLoRA math, DPO vs RLHF, SFT pitfalls. | 06 |
| 15 min | Break / coffee. |

### Afternoon block (3 hours, ~13:30–16:30)

**Goal:** system design and observability — Avrioc's likely "deeper" round.

| Time | Task | Doc |
|------|------|-----|
| 90 min | **[Chapter 16 — System Design](16_system_design.md)** — read all worked examples. Then **do one mock yourself**: pick "Design AI coaching for MyWhoosh" or "Design a customer support assistant for Comera." Sketch on paper, time yourself 45 min. | 16 |
| 60 min | **[Chapter 14 — Monitoring & Drift](14_monitoring_drift.md)** — PSI formula, KS test, Evidently, online vs offline eval. | 14 |
| 30 min | **[Chapter 11 — AWS & Azure](11_aws_azure.md)** — refresh SageMaker endpoints, Lambda, Databricks. Skim. | 11 |

### Evening block (3 hours, ~18:00–21:00)

**Goal:** resume rehearsal + final live coding + interview kit.

| Time | Task | Doc |
|------|------|-----|
| 75 min | **[Chapter 15 — Resume Deep Dive](15_resume_deep_dive.md)** — for **every** resume bullet, prepare a 30-second answer to: "tell me more about that." Rehearse out loud. | 15 |
| 60 min | **[Chapter 20 §B](20_live_coding_bank.md#b-ml--python-coding-human-led-round) — ML coding round 2.** Implement B.4 (Ray Serve deployment), B.7 (RAG retriever) from scratch. | 20 |
| 30 min | **Practice your "Tell me about yourself"** — Chapter 00 has the 60-second version. Time it. | 00 |
| 15 min | **Prep your 3 questions to ask them** — Chapter 19 §8. Write them on a sticky note for tomorrow. | 19 |

### Night block (60 min before bed)

| Time | Task | Doc |
|------|------|-----|
| 30 min | **Re-read [Chapter 18 — Cheatsheet](18_cheatsheet.md)**. Anything still fuzzy → 5 min refresh from the deep chapter. | 18 |
| 30 min | **Logistics check:** clothes ready, charger, water bottle, ID, second monitor / good camera angle if remote, screen-share dry run. | — |

**Bedtime: 22:00.** Sleep is performance.

---

## Thursday (2026-04-30) — Interview day

### Morning before interview (1.5–2 hours pre-call)

**Do NOT cram new material.** It hurts more than it helps.

| Task | Why |
|------|-----|
| Light breakfast — protein + slow carbs, avoid heavy/sugary. | Steady energy. |
| 20 min walk outside or 10 min cardio. | Blood flow → cognition. |
| **Re-read [Chapter 18 — Cheatsheet](18_cheatsheet.md) ONCE.** | Refresh, not learn. |
| **Re-read [Chapter 19 — Avrioc Company Intel](19_avrioc_company_intel.md) ONCE.** | Names, products, the 3 closing questions. |
| **Tell yourself your "tell me about yourself"** out loud, in front of mirror, ONCE. | Confidence prime. |
| **DO NOT** open new docs, new papers, or new YouTube videos. | Anxiety amplifier. |

### 30 minutes before

- Bathroom, water, sit down.
- Open: cheat-sheet PDF, your resume, the JD on a side monitor (or printed).
- Close: Slack, email, browser tabs unrelated to the interview.
- If remote: test camera/mic 10 min before, with a friend if possible.

### During the interview

**Behavioral checklist (read this before the call):**
- [ ] **Restate every question** in your own words before answering.
- [ ] **Lead with numbers** in every project answer (p99 < 500 ms, 29.7% → 68%, 8 models / 6 months).
- [ ] **Sketch a diagram** unsolicited for any system question.
- [ ] If you don't know something: **"I don't know that one specifically — here's how I'd find out: …"**.
- [ ] When stuck on coding: **think out loud**, name the blocker, propose alternatives.
- [ ] **Do NOT argue with a redirect.** Pivot fast.
- [ ] **Mention a product name** (MyWhoosh / Comera / Labaiik / Hyre) at least once.
- [ ] **Ask all 3 of your prepared questions** at the end.

### After the interview

| Task | Why |
|------|-----|
| Write a short thank-you email to the recruiter within 4 hours. | UAE business norm; cheap, signals follow-through. |
| Note what they asked you that you didn't have a great answer for. | For round 2 prep, if any. |
| **Don't replay the interview obsessively.** Reward yourself — go for a walk. | Cortisol management. |

---

## What to skip (deliberately)

You have 18 chapters. You **cannot read all of them deeply** in 48 hours. Skip:

- **Chapter 01 (Foundations)** — backprop and word2vec almost certainly not asked at this level.
- **Chapter 05 (Parameter tuning)** — quick scan only; you know temperature/top-p.
- **Chapter 08 (Vector databases)** — skim if Chapter 07 RAG covered it well; don't re-read.

If you finish early, **don't add more reading**. Add more reps of the **reverse-chronology drill** and live coding.

---

## A note on anxiety

Two days is enough. You have 8 years of real production experience. The job of these 48 hours is not to learn new material — it's to **make sure your existing knowledge is on the surface, not buried**.

You've shipped 8 models in 6 months. You've taken NER accuracy from 29.7% to 68%. You've built an XGBoost Lambda with p99 < 500 ms. You're already qualified. **The interview just needs to confirm it.** Walk in calm.

---

Continue to **[Chapter 23 — High-Probability Q&A Bank](23_high_probability_qa.md)**.
