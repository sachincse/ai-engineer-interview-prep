# Chapter 19 — Avrioc Technologies: Company Intelligence

> **Why this chapter exists:** Generic ML prep wins you a generic role. Avrioc-specific prep wins **this** role. Below is everything I could verify about the company, their products, their interview pattern, and the specific behaviors candidates have reported. Lean on this throughout the interview.

---

## 1. The company in one paragraph

**Avrioc Technologies LLC** is a privately-held Abu Dhabi tech holding/engineering company (registered ~2020, ~40-50 employees). Crunchbase and LinkedIn list it as self-funded — no external VC raised. The operational lead behind the product portfolio is **Akhtar Saeed Hashmi** (also CEO of MyWhoosh; you may see his name on their CEO page). Avrioc functions as the **shared engineering / AI house** for a portfolio of consumer products. The AI Engineer role you're applying for is therefore a **horizontal platform role** — your models will likely serve multiple products, not just one.

> **What to say:** "I noticed Avrioc operates as a shared engineering hub across a product portfolio — that's exactly the platform shape where I've thrived: I built ResMed's IHS MLOps platform that 8 models deployed onto in 6 months. The skill is making one piece of infrastructure carry many product use-cases."

---

## 2. Their product portfolio (memorize these names)

| Product | What it is | AI surface area you can talk about |
|---------|-----------|-----------------------------------|
| **MyWhoosh** | Virtual indoor cycling app. Official **UCI Cycling Esports World Championship** platform 2024–2026. Flagship. | Personalized coaching from telemetry (power/cadence/HR/RPM); avatar pose/animation; cheat detection (CV on power curves); workout recommendation; LLM-generated training plans. |
| **Comera** | Voice/video calling + messaging app (WhatsApp-style). | Real-time STT/TTS, language ID, content moderation, smart-reply, voice cloning safety, spam-call detection. |
| **Labaiik** | UAE grocery/e-commerce delivery. | Search ranking, demand forecasting, dynamic ETA, recommendations, query understanding (Arabic+English). |
| **Hyre** | UAE marketplace for licensed freelancers. | Matching, fraud detection, profile-completeness scoring, query routing, automated KYC summarization. |

> **Killer move:** When asked "Why Avrioc?" or "What would you build first?", **name a product** and propose something concrete and humble:
>
> *"For MyWhoosh, the goldmine is the per-rider telemetry stream — power, cadence, HR. I'd start by building a feature store (probably Snowflake + a Redis online tier — that's the architecture I shipped at TrueBalance) so an offline coaching model and an online recommendation model share the same features. Once that's solid, I'd layer an LLM on top for personalized training-plan text generation, with vLLM serving on Kubernetes."*
>
> Mentioning the product BY NAME signals you actually researched them. It's rare. It works.

---

## 3. The Avrioc interview process (verified pattern)

The pipeline below is reconstructed from Glassdoor reviews + the public JD. Treat it as the most-likely shape, not the guaranteed one.

```
HR screen (30m) → Tech screen / project deep-dive (60m) →
  Take-home assignment (48h) → Code review with 2 tech leads (60m) →
  3rd-party timed coding test (60m, HackerRank-style DS&A) →
  Hiring manager / system design (60m) → Final HR (comp + visa)
```

**Your interview is described as "pure technical" with possible live coding** → you are most likely entering at one of:
- **Tech screen / project deep-dive** (most common for first technical), OR
- **Code review of take-home** (if they sent one), OR
- **Hiring manager / system design**

**Either way, expect live coding.** Plan for both DS&A and ML coding (Chapter 20).

### Red flags from candidate reviews — mitigation

| Reviewer pain point | Your countermeasure |
|---------------------|--------------------|
| *"Interviewer was in a rush, kept countering their own scenarios"* | Stay calm. **Always restate the question** in your own words ("So you want me to design X assuming Y — correct?"). If they pivot, pivot fast — don't argue. |
| *"Reviewers couldn't run the take-home despite clear instructions"* | If asked about your take-home, lead with: "I made it one-command runnable: `docker compose up`. Here's the README." |
| *"Feedback delays and ghosting"* | Don't take silence personally. Mentally "submit and forget." |
| *"Required reverse-chronology project walkthrough"* | **Practice this** — go newest job first, deepest there, then quickly back through older ones. See §5 below. |

---

## 4. The "reverse-chronology project walkthrough" drill (CRITICAL)

This is a **confirmed Avrioc-specific pattern** from candidate reviews. You will be asked: *"Walk me through your projects in reverse chronological order, going as deep as you want."*

This rewards depth on the **most recent** work and brevity on older work. Time-budget yourself:

| Block | Time | Project | Depth |
|-------|------|---------|-------|
| **TrueBalance (Feb 2026 – present)** | 4–5 min | XGBoost Lambda + Claude ML workspace | Deepest. Diagrams, numbers, trade-offs, what you'd do differently. |
| **ResMed (Jul 2023 – Jan 2026)** | 3–4 min | IHS MLOps platform + RAG chatbot | Deep. Two stories. Numbers (8 models / 6 months). |
| **Tiger Analytics (Dec 2021 – Jul 2023)** | 1.5–2 min | SageMaker drift pipelines + Mars data quality | Concise. Hand-wave specifics, focus on "what I learned." |
| **Sopra Steria (Aug 2018 – Dec 2021)** | 1 min | Mention CV/OCR ID verification, OR-Tools (300 locations 7→5 days), Sopra Steria coding championship 1st place | One sentence each, plus the win. |

### Per-project mini-script template (memorize the shape, not the words)

```
1. CONTEXT (1 sentence): "At <company>, the team needed to <business problem>."
2. MY PROBLEM (1 sentence): "I owned <specific deliverable>."
3. THE HARD PART (2-3 sentences): "The non-obvious challenge was <X>. I considered <A> and <B>, picked <B> because <reason>."
4. OUTCOME WITH NUMBERS (1 sentence): "Result: <metric>."
5. WHAT I'D DO DIFFERENTLY (1 sentence — they love this): "Today I'd <Y> because <reason>."
```

### Worked example — TrueBalance XGBoost Lambda (~ 90 sec out loud)

> "At TrueBalance — that's my current role since Feb 2026 — the lending team was losing money on loans where the borrower would withdraw money but never repay. So we wanted to predict that withdrawal-then-default risk **before disbursal** and reroute those loans away from funding.
>
> My piece was the production predictor: real-time, p99 under 500 ms, isolated across three environments — dev, staging, prod — each in its own VPC.
>
> The non-obvious challenge was that the feature pipeline pulls from Snowflake, which has 200ms+ tail latencies — not okay inside a 500ms SLA. I considered (a) caching the full feature row in Redis, or (b) async fetching with a model fallback for cache misses. I picked (a) with a TTL-based refresher because the feature freshness window was 6 hours, not 6 seconds.
>
> The implementation: XGBoost model, Lambda container image, Terraform-managed VPC + subnets + SGs per env, CloudWatch custom metrics for p99 instrumentation, Redis-fronted feature store, and a graceful fallback to last-known features on Snowflake timeout.
>
> Result: p99 stayed sub-500ms across all three envs, and the projected portfolio profit lift from cutting funding to high-withdraw-risk borrowers was significant — exact number is confidential.
>
> Today I'd push harder on the **online feature store** abstraction — right now it's Redis with a cron refresher, but the right shape is Feast with a Snowflake offline store and Redis online store. That'd save us writing custom invalidation logic."

That's the **shape** to copy: context → ownership → the trade-off → the numbers → the retrospective.

---

## 5. Things to mention unprompted (cheap signals of preparation)

Drop one or two of these in the interview — each takes 5 seconds and signals you've done your homework:

- *"I read MyWhoosh became the official UCI Cycling Esports World Championship platform through 2026 — the real-time multiplayer scale must be substantial."*
- *"vLLM 0.6+ added chunked prefill and FP8 KV cache — very useful for chat workloads where TTFT matters more than throughput."*
- *"I assume Avrioc has a DGX cluster in Abu Dhabi, given the JD names Slurm and DGX explicitly — happy to talk about how I'd partition Slurm for training jobs and Kubernetes for inference."*
- *"Ray Serve LLM has a nice pattern for prefill/decode disaggregation — it fits naturally if your RAG pipeline has different SLA needs per stage."*

---

## 6. Salary, visa, relocation talking points

(More detail in Chapter 17 — quick reference here.)

- **Tax-free UAE salary** — your ask should be in AED, not INR. Approximate target: **AED 35,000–45,000/month base** for an AI Engineer with 8 years (negotiable up if they push deep on LLMOps responsibilities).
- **Visa sponsorship** — confirm employment visa for self + spouse + children. Ask about the medical insurance tier (gold/silver/standard).
- **Onsite-only role** — confirm the Abu Dhabi office location and whether they support relocation costs (flights, initial housing).
- **Notice period** — be honest. 60 days at TrueBalance is normal in India. They will accept it.
- **"Why Abu Dhabi"** — have 2–3 reasons ready: (1) tax-free compounding for family financial security, (2) UAE's national AI strategy and the regional demand for production-grade AI talent, (3) genuinely curious about the work — MyWhoosh is a fascinating real-time consumer product.

---

## 7. What NOT to do in an Avrioc interview

- **Don't oversell Kubernetes or Ray.** Both reviews and the JD signal strongly that they want operational depth, not buzzword bingo. Say "I've operated K8s in production via team tooling — here are the primitives I know cold" rather than "I'm a K8s expert." Honesty wins.
- **Don't trash previous employers.** UAE business culture is relationship-driven; bad-mouthing travels.
- **Don't argue with the interviewer.** Multiple Glassdoor reviews flag this as a failure mode — interviewers redirecting and candidates pushing back. Pivot, don't push.
- **Don't skip the relocation seriousness signal.** They've been burned by candidates who back out. State clearly: *"My family has discussed Abu Dhabi and we're ready to move."* (Adjust to your truth.)
- **Don't dodge salary.** Come with a number in AED.

---

## 8. The 3 questions you ask them at the end

When they say "do you have any questions?" — never say no. Ask:

1. **"What does the AI infrastructure look like today — is it primarily K8s + Ray, or does Slurm own training while K8s handles inference?"**
   → Signals: you understand the split, you're already thinking about how to operate.

2. **"Of MyWhoosh, Comera, Labaiik, Hyre — which one's AI roadmap is the team's biggest current focus?"**
   → Signals: you read about them. **This question alone has won interviews.**

3. **"What does success look like for this role in the first 6 months?"**
   → Signals: outcome-oriented, low-ego.

(Don't ask about salary/visa here — save for the HR round.)

---

Continue to **[Chapter 20 — Live Coding Bank](20_live_coding_bank.md)**.
