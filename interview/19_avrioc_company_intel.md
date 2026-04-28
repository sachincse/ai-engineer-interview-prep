# Chapter 19 — Avrioc Technologies: Company Intelligence

> **Why this chapter exists:** Generic ML prep wins you a generic role. Avrioc-specific prep wins **this** role. The interviewer on Thursday will be deciding within the first ten minutes whether you sound like "another candidate from a job board" or "someone who actually researched us." This chapter exists so you can be unmistakably the second kind. Read it once carefully Tuesday, again Wednesday night, and skim it Thursday morning before the call.

---

## 1. The company in a few paragraphs (read this first, slowly)

Avrioc Technologies LLC is a privately-held Abu Dhabi technology holding and engineering company, registered around 2020, with roughly forty to fifty employees according to the public LinkedIn footprint. Crunchbase and the LinkedIn page both list it as self-funded — they have not raised external venture capital, which is unusual for a company shipping multiple consumer products and matters for two reasons. First, it means the leadership has personal capital at stake, so they care about real revenue and real margin, not just growth metrics. Second, it means they hire conservatively and slowly. When they do hire, they expect the candidate to actually deliver in the first few months — there's no big VC runway absorbing slow ramps.

The operational lead behind the entire product portfolio is **Akhtar Saeed Hashmi**, who is also the CEO of MyWhoosh and whose name shows up across the Crunchbase profiles, the MyWhoosh leadership page, and several UAE business news features. You may not meet him, but if his name comes up, recognize it without being asked.

The most important structural fact about Avrioc is that it operates as a **shared engineering hub**, not as a single-product startup. Four very different consumer products live underneath it: an indoor cycling esports platform, a voice-and-video calling app, a grocery delivery service, and a freelancer marketplace. That structure has a direct implication for the AI Engineer role you're applying for: this is a **horizontal platform position**. Your work — feature stores, model-serving infrastructure, RAG pipelines, monitoring dashboards — will need to serve multiple products with different SLAs, different data shapes, and different user populations. That's actually exactly the work you've been doing for the last three years, first at ResMed (the IHS MLOps platform that hosted eight models in six months) and now in microcosm at TrueBalance (the Claude-powered ML workspace that integrates Jira, GitHub, Athena, and Jenkins). You should be saying out loud, repeatedly, that you understand and prefer this kind of role.

> **How to say this in an interview:** "What caught my eye is that Avrioc is structured as a shared engineering organization across four very different products. That's the platform shape where I've spent the last three years — at ResMed I built the IHS platform that eight models deployed onto in six months, and the skill is making one piece of infrastructure carry many product use-cases without becoming a Frankenstein. I'd rather build that platform layer than be the sole AI engineer on a single-product team."

---

## 2. The product portfolio — memorize, don't just skim

Below is one paragraph per product. Each paragraph is structured the same way: what it is, why it matters commercially, and what the AI playground looks like. **The interviewer will quietly award points if you mention any product by name** — most candidates won't.

### 2.1 MyWhoosh — the flagship

MyWhoosh is a virtual indoor cycling app, the kind of product where you put your bike on a smart trainer in your living room, the trainer measures the resistance you generate, and you ride a digital course on your screen against other live riders worldwide. Think Zwift, but with significantly more aggressive AI ambitions. The reason this is the flagship — and the reason it shows up in UAE press releases — is that MyWhoosh became the **official UCI Cycling Esports World Championship platform for 2024, 2025, and 2026**. UCI is the world governing body of cycling, the same body that sanctions the Tour de France. Hosting the world championship for an entire sporting category is not a marketing partnership — it's a multi-year commitment with sporting integrity and broadcasting rights at stake. That commitment is the reason MyWhoosh has the deepest engineering investment of the four products.

For an AI engineer this is a goldmine, and you should be ready to say why. Every rider produces a continuous telemetry stream — power output in watts, cadence in RPM, heart rate, speed, gradient response, even pose data if they have a camera. That stream is high-frequency, individually labeled, and tied to clear outcomes (won the race, hit a personal best, dropped from the pack). You can build a personalized coaching model from it. You can build cheat detection on top of it — the UCI championship implies somebody has to verify that suspicious power curves aren't doping or trainer manipulation, which is essentially a computer-vision and time-series anomaly problem. You can build FTP estimation (Functional Threshold Power, the cycling equivalent of resting heart rate). You can build LLM-generated training plans that take the rider's last four weeks of data and produce a personalized workout in natural language. You can build voice and avatar coaching that talks the rider through the workout in real time. The data richness here makes MyWhoosh probably the single most interesting AI product Avrioc owns, and you should treat it as the default product to mention when an interviewer asks "what would you build first?"

### 2.2 Comera — the voice/video product

Comera is a voice and video calling app in the WhatsApp lineage — encrypted personal calls, group calls, messaging on top. The strategic context matters: WhatsApp is sometimes restricted or unreliable in parts of the Middle East, and locally-hosted alternatives have a real market. Comera is one of those alternatives, with the data residency and government-relations footprint that requires.

The AI surface area is mostly real-time speech and natural language work. Speech-to-text and text-to-speech across the fifty-plus languages relevant to MENA and South Asian users. Real-time language identification so the call can switch transcript languages mid-conversation. Content moderation, both proactive (CSAM, terror content) and reactive (user-reported messages). Smart-reply for messaging. Spam-call detection from call metadata patterns. Voice-cloning safety, which is the increasingly important problem of detecting whether a voice on a call is human or AI-generated. If the interviewer's role involves Comera, the deep technical conversations will be about streaming inference latency, low-bitrate audio quality, and on-device versus server-side trade-offs.

### 2.3 Labaiik — the grocery/e-commerce play

Labaiik is a UAE-focused grocery and e-commerce delivery platform. The product itself isn't novel — Instacart, Getir, Talabat-Mart all do similar things — but the AI work behind a delivery business is substantial: search ranking on a multilingual catalog, demand forecasting per dark-store and per SKU, dynamic ETA prediction that has to update in real time as a driver moves through traffic, recommendation systems for cross-selling and dietary preferences, and query understanding for an Arabic-plus-English speaker base where users mix the two languages within a single search. Demand forecasting alone is a deep ML problem — promotion uplift modeling, weather effects, Ramadan and Eid seasonality, weekday-versus-weekend patterns, all of which break naive time-series models. If Labaiik comes up, the conversation will lean toward classical ML and forecasting rather than LLMs.

### 2.4 Hyre — the freelancer marketplace

Hyre is a UAE marketplace for licensed freelancers — think the regional answer to Upwork, but with the UAE's specific licensing requirements baked in (a freelancer in the UAE typically needs a freelancer permit issued by a free-zone authority). The AI surface is matchmaking (matching client briefs to freelancer profiles), fraud detection (fake profiles, payment fraud), profile-completion scoring (nudging freelancers toward more-complete profiles that convert better), query routing, and KYC summarization — taking the licensing documents and generating a structured profile for the client to read. This is the smallest of the four products and the one with the lowest volume of public information.

### 2.5 The likely org chart (block diagram)

Here's a reasonable mental model of how Avrioc's engineering organization probably looks. You don't need to claim you know this for sure — you just need to talk about it like someone who's thought about how shared platforms work.

```
                          ┌────────────────────────┐
                          │   Avrioc Leadership    │
                          │   (Akhtar S. Hashmi)   │
                          └───────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                │                     │                     │
        ┌───────▼────────┐   ┌────────▼─────────┐   ┌──────▼───────┐
        │ Platform & AI  │   │ Product Eng      │   │ DevOps /     │
        │ (the team you  │   │ (per-product     │   │ SRE / Infra  │
        │  are joining)  │   │  squads)         │   │              │
        └───────┬────────┘   └────────┬─────────┘   └──────┬───────┘
                │                     │                    │
        ┌───────┴────────────┐        │                    │
        │ shared services:   │        │                    │
        │  • Feature store   │        │                    │
        │  • Model registry  │        │                    │
        │  • RAG / LLM gw    │        │                    │
        │  • Monitoring      │        │                    │
        │  • Slurm + DGX     │        │                    │
        │  • K8s + vLLM      │        │                    │
        └────────┬───────────┘        │                    │
                 │                    │                    │
                 ▼                    ▼                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │  MyWhoosh    │   Comera    │   Labaiik   │    Hyre       │
        │ (cycling)    │ (calling)   │ (delivery)  │ (freelance)   │
        └──────────────────────────────────────────────────────────┘
```

The interview-winning sentence here is something like: "I'd assume the AI Engineer role lives in a shared platform team that exposes services — feature store, model registry, RAG gateway, drift monitoring — that the per-product engineering squads consume. That's the shape of platform I built at ResMed, and the work that energizes me is making the shared layer good enough that product teams want to use it instead of going around it."

---

## 3. The Avrioc interview process — what's confirmed and what's likely

The pipeline below is reconstructed from public Glassdoor reviews (there are not many, but the pattern is consistent), the published JD, and standard UAE tech-company hiring practice. Treat it as the most-likely shape, not a guarantee.

```
   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ HR phone │─▶│ Tech     │─▶│ Take-    │─▶│ Code     │─▶│ HM /     │─▶│ Final HR │
   │ screen   │  │ screen / │  │ home     │  │ review w │  │ system   │  │ comp +   │
   │ 30 min   │  │ project  │  │ assign-  │  │ 2 leads  │  │ design   │  │ visa     │
   │          │  │ deep-    │  │ ment     │  │ 60 min   │  │ 60 min   │  │          │
   │          │  │ dive     │  │ (48h)    │  │          │  │          │  │          │
   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
                                       │
                                       └─▶ Sometimes a separate 3rd-party HackerRank-style
                                           timed coding test (60 min) before the code review.
```

Your interview is described to you as "pure technical" with possible live coding, which means you are most likely entering at the **tech screen / project deep-dive** stage, possibly combined with the **code review of a take-home** or a live coding round. This matters because it dictates the mix of preparation: you need (a) reverse-chronology story fluency, (b) live coding muscle in both DS&A and ML/Python, and (c) at least one coherent system-design narrative in your back pocket in case it pivots that way.

### 3.1 What each stage tests

**The tech screen / project deep-dive.** This is roughly sixty minutes with a senior engineer, sometimes the hiring manager. They will start with "tell me about yourself," then ask you to walk through your projects in reverse chronological order — and they will interrupt with "go deeper there" on the ones that interest them. The thing they are testing is not whether you know the right buzzwords; it's whether you actually shipped the things on your resume. The way you fail this round is by being unable to describe a trade-off you made. The way you win it is by saying, unprompted, "I considered A and B, picked B because of [specific reason], and in retrospect I would have done C." Senior engineers can smell a candidate who memorized their resume versus one who actually built it.

**The take-home assignment.** If you get one, it'll be a small but realistic project — typically something like "build a simple RAG pipeline for the attached document set" or "build a FastAPI endpoint that wraps an LLM with rate limiting." Two reviewers explicitly complained on Glassdoor that they couldn't run the take-homes that candidates submitted, even with included instructions. **The mitigation is dead simple: make yours one-command runnable.** A `docker compose up` that starts everything, a `README.md` that fits on one screen, a `Makefile` with `make test` and `make run` targets. Spend an extra hour on that polish; it disproportionately impacts your score.

**The code review with two tech leads.** Two senior engineers walk through your take-home with you and probe. They will ask "why did you do it this way?" and "what would break if I doubled the load?" and "what's the dumbest thing about this code?" The last one is a trick — they want self-awareness. Have a real answer: "If I had another four hours, I'd add a rate limiter and an integration test for the streaming endpoint — right now I tested those manually with curl."

**The 3rd-party timed coding test.** This is HackerRank-style: a browser tab, two to three problems, sixty minutes. Usually one easy/medium and one medium/hard. A previous candidate explicitly mentioned **rotate-an-array** as a problem they got, which is why Chapter 20 starts with it. Practice the array-manipulation patterns, the sliding window template, and one binary search problem and you'll be in good shape.

**Hiring manager / system design.** This round, if you get it, is "design X" — design a recommendation system for MyWhoosh, design a real-time spam-call detector for Comera. You sketch on a whiteboard or shared canvas, talk through trade-offs, and answer probing questions. The trap here is going too deep on one component and never finishing the diagram. Always do the breadth pass first — top-level boxes, top-level data flow — and only then dive into the component the interviewer points at.

**Final HR.** Compensation, visa, start date, references. Do not negotiate technical details here; do not get distracted by salary discussion before they've made an offer.

### 3.2 Red flags from candidate reviews and how to handle them

Reading through the public Glassdoor and Blind reviews, four pain points show up repeatedly. None of them disqualify the company — they're patterns of a fast-moving small team — but you should know they exist so you can read the room appropriately.

| Reviewer pain point | What you should do |
|---------------------|---------------------|
| *"Interviewer was in a rush, kept countering their own scenarios"* | Stay calm. **Always restate the question** in your own words ("So you want me to design X assuming Y — correct?"). If they pivot, pivot fast — do not argue. The redirect is them stress-testing you, not them being indecisive. |
| *"Reviewers couldn't run the take-home despite clear instructions"* | If asked about your take-home, lead with: "I made it one-command runnable: `docker compose up`. Here's the README." |
| *"Feedback delays and ghosting"* | Don't take silence personally. Mentally submit-and-forget. If you haven't heard in a week, one polite follow-up is fine; more than that is counterproductive. |
| *"Required reverse-chronology project walkthrough"* | Practice this. See section 4 below — it's the single most important rehearsal you can do tonight. |

---

## 4. The reverse-chronology project drill (the most important rehearsal)

This is a confirmed Avrioc-specific pattern. You will be asked, almost word-for-word: *"Walk me through your projects in reverse chronological order, and go as deep as you want on each."* The reason this rewards rehearsal is that it's open-ended — the interviewer is not feeding you specific questions, they're letting you choose your own depth. Most candidates either go too deep on the oldest project (because they have more years of stories) or too shallow on the newest (because they haven't had time to systematize the new work yet). You want to do the opposite: deepest on the most recent, shallowest on the oldest.

### 4.1 Time-budget yourself

Roughly:

| Block | Time | Project | Depth |
|-------|------|---------|-------|
| **TrueBalance (Feb 2026 – present)** | 4–5 min | XGBoost Lambda + Claude ML workspace + NER lender-id | Deepest. Diagrams, numbers, trade-offs, what you'd do differently. |
| **ResMed (Jul 2023 – Jan 2026)** | 3–4 min | IHS MLOps platform + RAG clinical chatbot | Deep. Two stories. Concrete numbers (8 models / 6 months). |
| **Tiger Analytics (Dec 2021 – Jul 2023)** | 1.5–2 min | SageMaker drift pipelines + Mars data quality on Databricks | Concise. Hand-wave specifics, focus on "what I learned." |
| **Sopra Steria (Aug 2018 – Dec 2021)** | 1 min | CV/OCR ID verification + OR-Tools route planning + the coding championship win | One sentence each, plus the win. |

### 4.2 The mini-script template

Every project answer should follow the same five-beat structure. Memorize the shape, not the words:

```
1. CONTEXT (1 sentence): "At <company>, the team needed to <business problem>."
2. MY PROBLEM (1 sentence): "I owned <specific deliverable>."
3. THE HARD PART (2-3 sentences): "The non-obvious challenge was <X>. I considered <A> and <B>, picked <B> because <reason>."
4. OUTCOME WITH NUMBERS (1 sentence): "Result: <metric>."
5. WHAT I'D DO DIFFERENTLY (1 sentence — they love this): "Today I'd <Y> because <reason>."
```

Below are worked examples in the actual cadence and word count you should aim for. Practice them out loud. Tape yourself. Listen back and notice where you stumble.

### 4.3 TrueBalance walkthrough — script for ~90 seconds out loud

> "At TrueBalance — that's my current role since February 2026 — the lending team was losing money on a specific failure mode: borrowers who'd take a loan, withdraw the money, and never repay. So they wanted to predict that withdrawal-then-default risk **before disbursal** and reroute those loans away from funding.
>
> My piece was the production predictor: real-time, p99 under 500 milliseconds, isolated across three environments — dev, staging, prod — each in its own VPC.
>
> The non-obvious challenge was that the feature pipeline pulls from Snowflake, and Snowflake has 200-millisecond-plus tail latencies that don't fit inside a 500-millisecond SLA. I considered two options: (a) cache the full feature row in Redis with a TTL refresher, or (b) async-fetch with a model fallback on cache miss. I picked (a) because the feature freshness window was six hours, not six seconds — there was no point paying the latency cost for sub-second freshness we didn't need.
>
> The implementation was an XGBoost model packaged as a Lambda container image, Terraform-managed VPC and security groups per environment, CloudWatch custom metrics for p99 instrumentation, Redis fronting the feature store, and a graceful fallback to last-known features on Snowflake timeout.
>
> Result: p99 stayed sub-500 milliseconds across all three environments, and the projected portfolio profit lift from cutting funding to high-risk borrowers was significant — exact number is confidential.
>
> Today I'd push harder on the **online feature store** abstraction. Right now it's Redis with a cron refresher, but the right shape is Feast with Snowflake as the offline store and Redis as the online store. That'd save us writing custom invalidation logic and would let other models reuse the same features."

That's the shape: context, ownership, the trade-off, the numbers, the retrospective. Notice that the "what I'd do differently" answer is honest — it points at a real gap in the current implementation — but it's also forward-looking and constructive, not self-flagellating.

### 4.4 ResMed walkthrough — script for ~3 minutes

> "Before TrueBalance, I spent two and a half years at ResMed, the medical device company best known for sleep apnea machines. I joined the data science platform team. My biggest project there was the IHS MLOps platform — IHS stands for Inference Hosting Service. The team had a problem where every data scientist who built a model had to also build their own deployment pipeline, and the same wheel was being reinvented eight or ten times a year. I owned the platform that fixed that.
>
> The shape was a multi-container SageMaker endpoint pattern with a standard Docker contract — input schema, prediction interface, monitoring hooks — plus a CDK module that wrapped the deployment so a data scientist could go from a trained model to a live endpoint in less than an afternoon. We onboarded eight models in the first six months. The hardest part was getting the abstraction right: too rigid and people would route around it; too loose and we'd end up with eight different deployment patterns again. I landed on a thin contract — predict-method signature, log-format spec, drift-tracking hook — and let the rest stay flexible.
>
> The second big project at ResMed was a RAG-based clinical chatbot. The use-case was clinicians and product teams asking questions against a corpus of clinical reports and documentation. The interesting design choice was a query router — an LLM classified incoming queries into three buckets: factual (which used vector retrieval against pgVector), analytical (which generated SQL against Snowflake), and conversational (direct LLM with no retrieval). That router cut hallucinations on analytical questions because we stopped trying to answer them from retrieved text and started answering them from actual data. I'd say the biggest lesson was that the router prompt itself needed careful evaluation — getting that wrong silently misroutes everything downstream.
>
> Closely tied to those projects, I also built a Datadog drift dashboard utility that wrapped the data science team's drift code into a metadata-driven generator. It became the team standard — every model registered in the platform got an auto-generated drift dashboard.
>
> If I went back today, I'd push harder on shadow-deployment as a first-class platform feature. We did canary rollouts but didn't have shadow mode wired in by default."

### 4.5 Tiger Analytics walkthrough — script for ~90 seconds

> "Before ResMed I was at Tiger Analytics for about a year and a half, mostly on the Mars Pet Care account. Two highlights: I built drift-detection pipelines on SageMaker for the prediction models we'd shipped — that was where I first learned how to operationalize PSI and KS-test workflows at scale, which carried directly into the ResMed dashboard work. The second was data-quality work on Databricks: writing the validation jobs that caught upstream pipeline issues before they hit the modeling layer. The big lesson from Tiger was how much of production ML is data plumbing rather than modeling — that lesson shaped how I approached the IHS platform later."

### 4.6 Sopra Steria walkthrough — about 60 seconds

> "And going back further, I spent three and a half years at Sopra Steria after college. Two projects worth mentioning. First, a computer-vision OCR pipeline for ID verification — a real-time identity check used by a major French bank for their digital onboarding. Second, a route-planning optimization using OR-Tools — we cut field-visit planning from seven days to five days across about three hundred locations. And one personal note: I won the Sopra Steria internal coding championship that year, which is what got me promoted onto the harder data-science work."

### 4.7 The connector sentence

After Sopra you should always finish with a connecting sentence that loops back to Avrioc:

> "And those projects — across XGBoost, RAG, MLOps platforms, drift, and CV — are why this Avrioc role lines up well: the JD names Kubernetes, vLLM, Ray, FastAPI, LangChain, Slurm, DGX, AWS, and a horizontal AI platform mandate, and most of those map directly onto what I've shipped."

---

## 5. Things to mention unprompted (cheap signals of preparation)

Drop one or two of these in the conversation — each takes five seconds to say, and each signals that you actually researched the company and the technical landscape:

- *"I read MyWhoosh became the official UCI Cycling Esports World Championship platform through 2026 — the real-time multiplayer scale must be substantial."*
- *"vLLM 0.6+ added chunked prefill and FP8 KV cache — very useful for chat workloads where TTFT matters more than throughput."*
- *"I assume Avrioc has a DGX cluster in Abu Dhabi, given the JD names Slurm and DGX explicitly — happy to talk about how I'd partition Slurm for training jobs and Kubernetes for inference."*
- *"Ray Serve LLM has a nice pattern for prefill/decode disaggregation — it fits naturally if your RAG pipeline has different SLA needs per stage."*
- *"For Comera, real-time STT with low-bitrate audio is a fundamentally different problem from batch transcription — Whisper alone won't do it; you need streaming and chunked audio inference."*

Pick two and use them; don't fire all five — that's too much.

---

## 6. Salary, visa, UAE relocation — the negotiation context

This is the section most candidates skip and then panic over the night before. Read it carefully, because the final HR round will absolutely cover all of these and you should not be improvising.

### 6.1 Salary expectations in AED

The UAE is a tax-free salary jurisdiction — you do not pay personal income tax on your salary. This means a UAE salary in AED is NOT directly comparable to an Indian salary in INR, because in India a meaningful chunk of your pay went to income tax, EPF, and other deductions. The right mental conversion is **post-tax to post-tax**. As a rough anchor, a Senior ML Engineer with 8 years in Abu Dhabi for an AI-Engineer role at a serious tech employer should be in the range of **AED 35,000 to 45,000 per month base**, with negotiation room upward if the LLMOps responsibilities run deep and the candidate has the platform-engineering credentials you have. Total comp on top of base typically includes housing allowance (sometimes folded in, sometimes separate), medical insurance for self plus dependents, schooling allowance for children if the employer is generous, annual flights home for family, and an end-of-service gratuity (legally mandated in UAE — roughly 21 days of basic salary per year of service for the first five years, 30 days per year after that).

The math you should have in your head: AED 40,000 per month base, tax-free, is about USD 130k post-tax annually before any allowances. With housing and schooling allowances bundled in, the effective package can easily reach USD 170-200k post-tax for a senior engineer. The compounding upside is enormous over even three to five years compared to a similar role in India.

### 6.2 Visa, Emirates ID, and the relocation logistics

The actual visa process is a sequence of steps the employer drives, not you. The employer applies for an entry permit (the e-visa), which lets you fly in. Within sixty days of arrival you complete a medical screening (blood test plus chest x-ray), the Emirates ID biometrics (fingerprints and a photo), and the residence visa stamping. The employer pays for all of this for the primary employee. For the family — spouse and children — there are two paths: either the employer sponsors them on a family visa as part of the package, or you sponsor them yourself once your residence permit is issued, which requires meeting a minimum salary threshold (AED 4,000 base plus housing, or AED 5,000 base, currently). At your salary band the threshold is trivially met.

Schooling: international schools in Abu Dhabi range from AED 30,000 to AED 90,000 per year per child depending on tier. This is the single biggest non-housing cost for a relocating family with children. Negotiate for a schooling allowance explicitly — it's a normal ask and many employers include it for senior hires.

### 6.3 The "Why Abu Dhabi" answer (you must have one)

When they ask why Abu Dhabi, you should have three reasons ready, in order of weight:

> "Three reasons. First, the tax-free structure lets me build long-term financial security for my family in a way that's hard to match elsewhere — that's the practical truth. Second, the UAE national AI strategy is creating real demand for production-grade AI engineering, not just research, and that's the kind of work I want to do. Third, onsite cross-functional teams produce faster learning velocity than remote work, and at this stage of my career — eight years in, mid-thirties — learning velocity matters more than convenience. My family and I have discussed the move and we're ready."

That last sentence is the one that closes loops with hiring managers. Multiple candidates have apparently backed out of UAE offers at the last minute, and Avrioc has no patience for that pattern. If you say it clearly, you remove a significant unstated risk in their head.

### 6.4 Notice period and start date

Sixty days notice from TrueBalance is standard in India and they will accept it. Be honest — say "I'd be available roughly seventy to ninety days from offer signing, accounting for sixty days notice plus visa processing time." Don't pretend you can start immediately; that breaks trust.

---

## 7. What NOT to do in an Avrioc interview

A few do-nots that have sunk specific candidates in their reviews:

- **Don't oversell Kubernetes or Ray.** Both the reviews and the JD signal strongly that Avrioc wants operational depth, not buzzword bingo. Say "I've operated K8s in production via team tooling — here are the primitives I know cold" rather than "I'm a K8s expert." Honesty wins, and a senior engineer can immediately tell when you're fronting.
- **Don't trash previous employers.** UAE business culture is relationship-driven, and bad-mouthing travels fast in a small market. ResMed, TrueBalance, Tiger, Sopra — all good. Even when you describe what you'd do differently, frame it as forward-looking improvement, not as criticism of your past team.
- **Don't argue with the interviewer.** Multiple Glassdoor reviews flag this as the failure mode — interviewers redirecting and candidates pushing back on the redirect. Pivot, don't push. If they say "let's say you don't have Snowflake," accept it and answer for the new world.
- **Don't skip the relocation seriousness signal.** State it clearly, even unprompted: *"My family and I have discussed Abu Dhabi and we're ready to move."*
- **Don't dodge salary.** When the HR round asks your expectations, give a number in AED, not in INR. Don't say "open to discussion" — that reads as either inexperience or weakness in this market.

---

## 8. The three questions you ask them at the end (and why each works)

When the interviewer says "do you have any questions for us?" — never say no. Asking nothing reads as either disengaged or as already-decided-against-them. Always have three. Below are the three you should use, with a paragraph each on what each one signals.

**Question 1: "What does the AI infrastructure look like today — is it primarily K8s plus Ray, or does Slurm own training while K8s handles inference? How are you splitting the DGX hardware?"**

This signals two things at once: you understand that Slurm and Kubernetes solve different problems, and you've already noticed that the JD names DGX (which most candidates haven't). The follow-up question this opens up — "where would I be most useful in the first quarter?" — is gold for you because it lets you map your skills to a real gap.

**Question 2: "Of MyWhoosh, Comera, Labaiik, and Hyre — which one's AI roadmap is the team's biggest current focus this quarter?"**

This question alone has won interviews. You've named all four products, you've signaled that you actually read about them, and you've asked about *current* priorities rather than vague future state. The answer also gives you crucial intelligence for any subsequent rounds: if they say "MyWhoosh," lean into the cycling-and-telemetry angle. If they say "Labaiik," pivot to demand forecasting and Arabic NLP. If they say "Comera," the real-time STT and content moderation surface becomes the focus.

**Question 3: "What does success look like for this role in the first six months? What would the person who's nailing it have shipped by then?"**

This signals outcome-orientation and low-ego — you're not asking about title, perks, or who-you-report-to; you're asking what good looks like. The answer also gives you a dry-run on the start-date negotiation later: if they describe an ambitious six-month roadmap, you can credibly justify a slightly higher comp ask.

(Don't ask about salary or visa here — save those for the dedicated HR round.)

---

## 9. Interview-day kit (what to bring, what to wear, what to do in the last hour)

If onsite Abu Dhabi:
- **Wear** business casual: a clean collared shirt, dark trousers, clean closed shoes. UAE business culture leans slightly more formal than Bay Area tech; err on the dressier side. No t-shirt.
- **Bring** a clean printout of your resume (two copies), a notebook and pen for whiteboard notes, a water bottle, and your phone on silent.
- **Arrive** at the building lobby thirty minutes early. Use the buffer to use the bathroom, drink water, do a five-minute slow-breathing reset, and re-read your three closing questions one more time.

If remote:
- Test your camera and microphone with a friend twenty-four hours beforehand, not on the day. Have a backup hotspot ready.
- Background should be a clean wall or one tasteful framed thing — not a kitchen, not a messy room. A ring light is overkill but a window over your shoulder is wrong (silhouettes you).
- Close every notification on your laptop. No Slack pings. Phone face-down across the room.

Mental warmup (last fifteen minutes before the call):
- Re-read **only** the cheatsheet (Chapter 18) and Section 8 of this chapter (the three closing questions). Do not open new material.
- Say your "tell me about yourself" out loud once in front of the mirror.
- Say one full reverse-chronology mini-script out loud — your TrueBalance one is the highest leverage.
- Then close all the docs and just sit. The work is already done.

---

Continue to **[Chapter 20 — Live Coding Bank](20_live_coding_bank.md)**.
