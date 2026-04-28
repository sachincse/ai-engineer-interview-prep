# Chapter 22 — 48-Hour Intensive Study Plan (Tuesday → Thursday)

> **Why this chapter exists:** Your interview is on Thursday 2026-04-30. Today is Tuesday 2026-04-28. The original 14-day plan in [Chapter 00](00_index.md) doesn't fit your reality. This is the realistic, opinionated 48-hour plan.
>
> **Mindset:** the goal of these two days is **not to learn new material**. After eight years in production ML you already know enough to do this job. The goal is to **bring what you know to the surface** so it's available under interview pressure rather than buried under generic anxiety. Treat this plan like an athlete tapering before a race — light, focused work that loads the right patterns into muscle memory, then deliberate rest.

---

## 22.1 The high-level rhythm

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                          TUESDAY                                │
   │  Morning: Avrioc context + LLM/transformer foundations          │
   │  Afternoon: JD-named stack (K8s, Ray, vLLM, Slurm, FastAPI)     │
   │  Evening: Live coding hands-on (DS&A + ML)                      │
   │  Night: Reverse-chronology drill, behavioral, lights out 22:30  │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                         WEDNESDAY                               │
   │  Morning: RAG + embeddings + fine-tuning (depth)                │
   │  Afternoon: System design + monitoring/drift                    │
   │  Evening: Resume rehearsal + final live coding + interview kit  │
   │  Night: Cheatsheet skim, logistics check, lights out 22:00      │
   └─────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────────┐
   │                          THURSDAY                               │
   │  Pre-interview: light breakfast, walk, ONE re-read of cheatsheet│
   │  During: behavioral checklist, sketch diagrams unsolicited      │
   │  After: thank-you email, reward yourself, do not replay         │
   └─────────────────────────────────────────────────────────────────┘
```

The key principle: **front-load the technical density on Tuesday and Wednesday morning**. By Wednesday afternoon you should be doing rehearsal and consolidation, not new reading. Thursday morning is just refresh — no learning.

---

## 22.2 Tuesday in detail

### Morning block (3 hours, ~9:00–12:00) — diagnostic and Avrioc context

The morning's purpose is to lock in **what makes you valuable to Avrioc specifically** before drilling general technical material. If you don't know who they are by lunch, you'll talk in generic ML-engineer terms all day Wednesday and that pattern will follow you into the interview.

| Time | Task | Why |
|------|------|-----|
| 30 min | Read [Chapter 19 — Avrioc Company Intel](19_avrioc_company_intel.md) carefully. Memorize the four product names (MyWhoosh, Comera, Labaiik, Hyre). | If you can drop a product name unprompted in the interview, you instantly differentiate. Most candidates won't. |
| 30 min | Skim [Chapter 18 — Cheatsheet](18_cheatsheet.md). Flag anything unfamiliar for the afternoon. | The cheatsheet is the smallest unit of "everything you might need at a whiteboard." If something on it is foreign now, you have a gap to close today. |
| 60 min | Re-read [Chapter 02 — Transformers](02_transformers.md) and [Chapter 03 — LLMs](03_llms.md) at speed. Focus only on attention math, KV-cache, and decoding. | These are the most-asked LLM concepts at this seniority level. If the interviewer probes, you need to be fluent — not deep, fluent. |
| 60 min | Re-read [Chapter 09 — Model Optimization](09_model_optimization.md). Read the interview Q&A sections **out loud**. | Quantization, pruning, distillation are JD-named. Reading aloud surfaces the gaps in your fluency that silent reading hides. |

### Afternoon block (3 hours, ~13:30–16:30) — JD-named stack

The afternoon's purpose is to lock down the specific technologies named in the JD. Avrioc's keyword list is the de facto interview rubric.

| Time | Task | Why |
|------|------|-----|
| 75 min | [Chapter 12 — Kubernetes & Ray](12_kubernetes_ray.md). Focus on: pods/deployments/services, HPA vs KEDA, GPU operator, Ray Core/Serve/Train/Tune, the decision tree for when to choose each. | K8s and Ray are the JD's heaviest hitters and the area where most ML candidates are weakest. This block is your differentiator. |
| 75 min | [Chapter 13 — Frameworks](13_frameworks.md). Focus on FastAPI async patterns, vLLM CLI flags and PagedAttention, LangGraph for agents, Chainlit for chat UIs. | Every framework on the JD lives here. If the interviewer asks "what would you build a chatbot with?", you need to answer in combinations — Chainlit + LangGraph + vLLM + FastAPI — not just name a single tool. |
| 30 min | [Chapter 21 — Slurm + DGX](21_slurm_dgx_hpc.md). Memorize the multi-node DDP sbatch script. | The JD names Slurm and DGX. Almost no other AI candidate will be ready for this. Even superficial fluency wins points. |

### Evening block (2.5 hours, ~18:00–20:30) — live coding hands-on

This block is for **doing**, not reading. Read-only learning produces fluency in interpretation; coding produces fluency in production.

| Time | Task | Why |
|------|------|-----|
| 45 min | [Chapter 20 §A — DS&A patterns](20_live_coding_bank.md). Code A.1 (rotate array), A.3 (sliding window), A.6 (binary search), A.8 (LRU) **from scratch on paper**. Don't peek at solutions. | A reviewer mentioned "rotate an array" specifically appearing on Avrioc's coding test. The patterns you drill here cover 80% of timed-test problems. |
| 45 min | [Chapter 20 §B.1 + B.2 — attention from scratch](20_live_coding_bank.md). Write multi-head attention in PyTorch in a fresh notebook. **No copying.** | If they ask you to implement attention live (which is common at this seniority), the only way to do it under pressure is to have written it cold this week. |
| 30 min | §B.3 — FastAPI streaming endpoint with cancellation. Write it and run it locally. | The JD names FastAPI and vLLM. A streaming endpoint is the single most likely live-coding ask for an LLM platform role. |
| 30 min | §C — quick warm-ups, all 10. Solve in your head, talk out loud. Tape yourself. | The talking-aloud habit is what wins live coding rounds. Practicing it solo is uncomfortable; do it anyway. |

### Night block (60 min before bed)

The night block consolidates two things: **your project narratives** and **the relocation/cultural angle**. These are the things that drift in your head, not in the docs.

| Time | Task | Why |
|------|------|-----|
| 30 min | [Chapter 19 §4 — Reverse-chronology drill](19_avrioc_company_intel.md#4-the-reverse-chronology-project-walkthrough-drill-critical). Practice TrueBalance + ResMed walkthroughs **out loud, timed**. Aim 4-5 min for TrueBalance, 3-4 min for ResMed. | This is the confirmed Avrioc interviewing pattern. Most candidates lose this round because they ramble. Timing yourself is the only way to fix that. |
| 30 min | [Chapter 17 — Behavioral & UAE](17_behavioral_hr.md). Lock in your "Why Abu Dhabi" answer. | Avrioc has been burned by candidates who back out post-offer. Your sincerity here matters as much as your technical depth. |

**Bedtime: by 22:30.** No screens after 22:00. Sleep is performance.

---

## 22.3 Wednesday in detail

### Morning block (3 hours, ~9:00–12:00) — RAG + embeddings + fine-tuning

This morning fills the gaps in modern LLM workflows. RAG is the most-asked LLM design topic at this seniority level, and Sachin has a direct project tie-in (the ResMed clinical chatbot) that he should be able to narrate in his sleep.

| Time | Task | Why |
|------|------|-----|
| 60 min | [Chapter 07 — RAG](07_rag.md). Read the worked example of the ResMed clinical chatbot carefully — it's your story to tell. | RAG is the most likely "design X" question. Having a real-world example you've shipped beats hypothetical answers every time. |
| 45 min | [Chapter 04 — Embeddings](04_embeddings.md). Read interview Q&A only if time-pressed. | Embeddings are the foundation under RAG. If the interviewer probes contrastive training or fine-tuning embeddings, you need an answer. |
| 60 min | [Chapter 06 — Fine-tuning](06_fine_tuning.md). LoRA/QLoRA math, DPO vs RLHF, SFT pitfalls. | Fine-tuning is named in the JD context implicitly (model optimization). One question on this topic is likely. |
| 15 min | Break / coffee. Don't skip the break. | Cognitive fatigue compounds. A 15-minute break here saves you from declining quality through the afternoon. |

### Afternoon block (3 hours, ~13:30–16:30) — system design + monitoring

The afternoon is the deepest technical block. System design is the round where senior candidates are graded most strictly, and monitoring is your direct resume tie-in (the Datadog drift dashboard).

| Time | Task | Why |
|------|------|-----|
| 90 min | [Chapter 16 — System Design](16_system_design.md). Read all worked examples. **Then do one mock yourself**: pick "Design AI coaching for MyWhoosh" or "Design a customer support assistant for Comera." Sketch on paper, time yourself 45 min. | System design is performance under pressure. Reading is necessary but not sufficient. The mock is what builds the actual skill. |
| 60 min | [Chapter 14 — Monitoring & Drift](14_monitoring_drift.md). PSI formula, KS test, Evidently, online vs offline eval. | Your Datadog drift dashboard at ResMed is a signature story. You should be able to talk through it in three minutes without notes. |
| 30 min | [Chapter 11 — AWS & Azure](11_aws_azure.md). Refresh SageMaker endpoints, Lambda, Databricks. Skim. | AWS is your strongest cloud per resume. A refresh ensures you don't fumble familiar ground. |

### Evening block (3 hours, ~18:00–21:00) — resume rehearsal + final live coding + interview kit

The evening is consolidation, not new content. By now your technical reservoir should be full; what you're doing is making it accessible under stress.

| Time | Task | Why |
|------|------|-----|
| 75 min | [Chapter 15 — Resume Deep Dive](15_resume_deep_dive.md). For **every** resume bullet, prepare a 30-second answer to "tell me more about that." Rehearse out loud. | The Avrioc reverse-chronology pattern means they may pull on any bullet. You need an answer ready for each. |
| 60 min | [Chapter 20 §B](20_live_coding_bank.md). Implement B.4 (Ray Serve deployment), B.7 (RAG retriever) from scratch. | Round-2 live coding practice. The first round was Tuesday evening; this one solidifies. |
| 30 min | Practice your "Tell me about yourself" — Chapter 17 §17.7 has the timed version. Recite it aloud three times in front of a mirror. | The first 60 seconds of any interview set the tone. Memorize the first ten and last ten words; improvise the middle. |
| 15 min | Prep your three questions to ask them — Chapter 19 §8. Write them on a sticky note for tomorrow. | "Do you have any questions?" is not optional. Showing up with three thoughtful questions is a free signal of preparation. |

### Night block (60 min before bed)

| Time | Task | Why |
|------|------|-----|
| 30 min | Re-read [Chapter 18 — Cheatsheet](18_cheatsheet.md). Anything still fuzzy → 5 min refresh from the deep chapter. | The cheatsheet is the morning-of artifact. By tonight it should feel familiar; tomorrow morning it'll just refresh. |
| 30 min | **Logistics check**: clothes ready (ironed), charger packed, water bottle, ID/passport, second monitor / good camera angle if remote, screen-share dry run. | If you're scrambling tomorrow morning for cables or clean shirts, that's stress that compounds into the call. Front-load it. |

**Bedtime: 22:00.** Sleep is performance. Don't compromise this.

---

## 22.4 Thursday — interview day

### Morning before interview (1.5–2 hours pre-call)

**The hard rule: do not cram new material today.** Cramming hurts more than it helps. New facts crowding into your short-term memory displace the older ones you've actually rehearsed.

| Task | Why |
|------|-----|
| Light breakfast — protein and slow carbs. Avoid heavy or sugary foods. | Steady energy. Sugar crashes mid-interview. |
| 20-minute walk outside, or 10 minutes of cardio. | Blood flow → cognition. Especially important if you slept poorly. |
| Re-read [Chapter 18 — Cheatsheet](18_cheatsheet.md) **once**. | Refresh, not learn. If something on the cheatsheet is suddenly foreign today, accept it and move on — last-minute panicking won't fix it. |
| Re-read [Chapter 19 — Avrioc Company Intel](19_avrioc_company_intel.md) **once**. Refresh the four product names and the three closing questions. | These are the highest-leverage Avrioc-specific signals. |
| Recite "Tell me about yourself" out loud, in front of a mirror, **once**. | Confidence prime. Hearing your own voice say the words puts them on the surface for the call. |
| **Do NOT** open new docs, papers, or YouTube videos. | Anxiety amplifier. New material doesn't help; it just generates "wait, I don't know this" panic. |

### 30 minutes before

- Bathroom, water, sit down at your interview station.
- Open: cheat-sheet PDF, your resume, the JD on a side monitor (or printed).
- Close: Slack, email, browser tabs unrelated to the interview, anything that might ping.
- If remote: test camera and mic 10 minutes before, with a friend if possible. Background should be clean and uncluttered.
- If in-person: arrive 15 minutes early. Bring printed copies of your resume.

### During the interview — behavioral checklist

Read this before the call. Don't open it during.

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                  IN-INTERVIEW CHECKLIST                         │
   ├─────────────────────────────────────────────────────────────────┤
   │  □ Restate every question in your own words BEFORE answering.   │
   │    "So you'd like me to design X assuming Y — correct?"         │
   │                                                                 │
   │  □ Lead with NUMBERS in every project answer.                   │
   │    "p99 under 500ms"  "29.7% to 68%"  "8 models in 6 months"    │
   │                                                                 │
   │  □ SKETCH a diagram unsolicited for any system question.        │
   │    "Let me draw this." Picks up your hand, picks up the marker. │
   │                                                                 │
   │  □ When you don't know something:                               │
   │    "I haven't worked with X directly — here's how I'd          │
   │    investigate: [your search process]"                          │
   │    NEVER make up an answer.                                     │
   │                                                                 │
   │  □ When stuck on coding: think OUT LOUD.                        │
   │    Name the blocker. Propose alternatives. The interviewer      │
   │    grades reasoning, not just final code.                       │
   │                                                                 │
   │  □ Do NOT argue with a redirect. If they pivot, pivot fast.     │
   │    Multiple Avrioc reviews flag this as a failure mode.         │
   │                                                                 │
   │  □ Mention a product name (MyWhoosh / Comera / Labaiik / Hyre)  │
   │    at least once. This is the cheapest differentiator possible. │
   │                                                                 │
   │  □ Ask all three of your prepared questions at the end.         │
   │    "Do you have any questions?" — never say no.                 │
   └─────────────────────────────────────────────────────────────────┘
```

### After the interview

| Task | Timing | Why |
|------|--------|-----|
| Write a short thank-you email to the recruiter. | Within 4 hours. | UAE business norm. Cheap signal of follow-through. |
| Note what they asked that you didn't have a great answer for. | Within 1 hour, while fresh. | For round 2 prep, if any. Don't try to memorize while exhausted. |
| **Do NOT replay the interview in your head obsessively.** Reward yourself instead. | Immediately. | Cortisol from rumination compounds anxiety, hurts your next round if there is one. |

---

## 22.5 What to skip — deliberately

You have 18 deep chapters plus the new addendum chapters. You **cannot read all of them deeply** in 48 hours. Skip these:

- **Chapter 01 (Foundations)** — backprop and Word2Vec almost certainly not asked at this seniority. The interviewer will assume you know it.
- **Chapter 05 (Parameter tuning)** — quick scan only; you know temperature and top-p.
- **Chapter 08 (Vector databases)** — skim if Chapter 07 RAG covered it well; don't re-read in depth.

If you finish your scheduled blocks early, **don't add more reading**. Add more reps of the reverse-chronology drill and live coding. Fluency compounds; reading fatigues.

---

## 22.6 A note on anxiety

Two days is enough. You have eight years of real production experience. The job of these 48 hours is **not** to learn new material — it's to make sure your existing knowledge is on the surface, not buried.

You've shipped eight models in six months. You've taken NER accuracy from 29.7% to 68%. You've built an XGBoost Lambda with p99 under 500 milliseconds. You're already qualified. The interview just needs to confirm it. Walk in calm.

The thing that helps me most when nerves spike: **focus on the interviewer**, not yourself. The interview isn't "am I good enough?" — it's "is this a fit for both of us?" That subtle reframe shifts you from defensive to evaluative, which is exactly the posture senior candidates need.

You'll do well. You've done the work.

---

Continue to **[Chapter 23 — High-Probability Q&A Bank](23_high_probability_qa.md)** for the question-and-answer pack you'll use most on Thursday morning.
