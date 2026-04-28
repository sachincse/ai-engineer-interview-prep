# Chapter 15 — Resume Projects, Deep Dive
## Every bullet on the resume, expanded into a story you can tell at a whiteboard

> The way Avrioc interviews work — confirmed by people who've been through their loop — is reverse-chronological project walkthrough. They will start at TrueBalance and walk backwards. For every bullet, expect 3-4 follow-up questions of escalating depth. This chapter is your script. Each story is structured so that you can deliver a 60-90 second opening narrative, then go five layers deep when probed.
>
> The style throughout: imagine you are at a whiteboard with a fellow senior engineer. You speak in paragraphs, not bullets. You draw diagrams unsolicited. You name trade-offs the interviewer hasn't asked about, because that's how you signal seniority.

---

## 15.1 TrueBalance (Senior ML Engineer, Feb 2026 – present)

TrueBalance is an Indian consumer-lending NBFC — small-ticket digital loans, mostly first-time borrowers in tier-2/3 cities. The ML org is small (about 12 people across DS and ML engineering), the product is loan-disbursement, and the business pain is always the same in unsecured consumer credit: how do you cut funding to the wrong borrower without choking off legitimate growth. I joined in February 2026 to lead the production ML engineering side — basically, take what data scientists prototype and make it a real-time, regulated, observable system.

There are four things on my resume from TrueBalance, and they are deliberately the four hardest things I touched in the first quarter. I'll walk through each in interview-format.

---

### Bullet 1 — Real-time XGBoost Lambda for loan-withdrawal prediction (p99 < 500ms, 3-env VPC-isolated)

**Setup — where, when, what was the business pain?**

The business pain was loan-withdrawal fraud. In a digital-loan flow there's a window — typically 30-90 seconds — between when the loan is approved and when the money lands in the borrower's account. In that window a sophisticated bad actor can change device, change SIM, even change the bank account in some flows, and the loan gets disbursed to a different person than the one who was scored at application time. This is the "withdrawal" risk, and it was costing TrueBalance measurable basis points on the portfolio every week. Until I joined, the only mitigation was a static rule-engine — coarse, false-positive-heavy, and not learning.

**My role — what I owned**

I owned the end-to-end production pipeline: model containerization, infra as code, the feature store integration, observability, and the rollout plan. The DS team owned model training. We agreed on the contract — they hand me a registered MLflow model and a list of features by name; I deliver a sub-500ms-p99 endpoint in three environments with feature drift dashboards, audit logs, and the ability to flip the model on/off via a config without a redeploy.

**The hard part — what was non-obvious**

Three things were non-obvious.

First, the latency budget. 500ms sounds generous until you actually decompose it: ingress + auth (~10ms), feature lookup (this is where the danger is — Snowflake is a warehouse, not an OLTP, p99 can spike to 800ms on cold reads), inference (XGBoost on CPU is fast, ~5-10ms for 150 features), audit write (must not block), egress (~5ms). The Snowflake number alone could blow the entire budget. So I had to design a Redis online cache in front of Snowflake, and crucially, design what to do when the cache misses.

Second, the regulatory shape. TrueBalance is RBI-regulated; loan PII must not cross environment boundaries. That ruled out a single shared cluster with environment-tagged data. The answer was three independent VPCs, each with its own KMS customer-managed key, its own Redis cluster, its own model registry, its own CloudWatch namespace. Promotion between environments is artifact-based, not data-shared.

Third, the rejected design. The default at TrueBalance for ML services was SageMaker real-time endpoints. I rejected that for this use case because (a) the traffic is spiky around peak loan-application hours and goes nearly to zero at night, (b) the model is small enough to fit in a Lambda container, and (c) provisioned-concurrency Lambda gives me predictable cold-start behavior at roughly a third of the SageMaker endpoint cost at our QPS. I wrote up a one-page comparison and the team agreed.

**The implementation — block diagram**

```
                   ┌─────────────────────────────┐
                   │   Loan Origination Service  │
                   │   (upstream, calls predict) │
                   └──────────────┬──────────────┘
                                  │ HTTPS / mTLS
                                  ▼
                       ┌─────────────────────┐
                       │   API Gateway       │
                       │   - WAF             │
                       │   - rate limiting   │
                       │   - JWT auth        │
                       └──────────┬──────────┘
                                  │
                                  ▼
              ┌──────────────────────────────────────┐
              │   AWS Lambda (container image)       │
              │   - Provisioned concurrency = 50     │
              │   - 3GB RAM, 2 vCPU                  │
              │   - Python 3.11 + xgboost + onnxrt   │
              │   - Cold-start p99: 1.2s             │
              │   - Warm  p99:    < 80ms             │
              └─────┬───────────┬──────────┬─────────┘
                    │           │          │
        feature     │   audit   │   model  │
        fetch       │   log     │   load   │
                    ▼           ▼          ▼
            ┌────────────┐ ┌────────┐ ┌─────────────┐
            │ Redis      │ │ Kinesis│ │ S3 + MLflow │
            │ ElastiCache│ │ Firehose│ │ registry   │
            │ (online    │ │ → S3   │ │ (cold load │
            │  features) │ │  audit │ │  on init)  │
            │ p99 ~15ms  │ │ bucket │ │            │
            └─────┬──────┘ └────────┘ └─────────────┘
                  │ on miss
                  ▼
            ┌────────────────┐
            │ Snowflake      │
            │ FEATURE_LATEST │
            │ p99 ~250ms     │
            └────────────────┘
```

**The feature-fetch story — the bit that interviewers love**

The Redis cache is populated in two ways. There's a batch backfill job every six hours that materializes the latest feature row per `borrower_id` from Snowflake into Redis with a 24-hour TTL — so any borrower who has been seen in the last day is hot. And there's a write-through path from the upstream feature-computation Airflow DAG that pushes to Redis at the same time it lands in Snowflake. So in steady state, cache hit rate sits around 96-97%.

The 3-4% miss is the interesting case. On a miss I do a parallel fetch — kick off the Snowflake query *and* a default-feature lookup at the same time, then wait `min(snowflake, 200ms-budget)`. If Snowflake comes back in time, use it. If not, use the default-feature row (computed offline, represents the population mean for each feature) and tag the response with `degraded=true`. The model still scores, the response still meets SLA, and we have a metric to alert on if degraded ratio goes above 1%. This is the kind of trade-off interviewers will probe — "what if Snowflake is down for 10 minutes?" — and the answer is, the system stays up and we get an alert.

**The numbers — concrete and honest**

- p99 end-to-end: 380-450ms warm, depending on environment
- p99 of just the inference call: 65ms
- Cache hit rate: 96-97%
- Provisioned concurrency: 50 (peaks at 70-80 RPS sustained, which is small in absolute terms but high enough that cold starts would have killed the SLO)
- Three environments: dev / staging / prod, each in its own VPC with VPC endpoints to S3, Secrets Manager, and CloudWatch (so Lambda is internet-isolated)
- Projected business impact: portfolio-profit lift in basis points per disbursement cycle (specific number depends on loan volume; the DS team validated this in shadow mode for 4 weeks before flip).

**What I'd do differently — the retrospective**

If I were starting again, I would have invested earlier in shadow-mode tooling. We did shadow mode but it was hand-rolled — a Lambda extension that copied the request, called the new model, and dumped both predictions to S3 for offline comparison. Building that took two engineer-weeks. If I'd built it as the first piece, before the model was even ready, the DS team could have iterated faster. Lesson: build the evaluation lane before the production lane.

The other thing I'd do differently is the audit log. We chose Kinesis Firehose for simplicity, which writes to S3 with a 60-second buffer. For a regulated loan-decision system the buffer is a compliance grey area — what if Lambda crashes mid-buffer? I'd switch to a synchronous Kafka write with at-least-once semantics, accept the ~5ms latency hit, and sleep better at audit time.

**Speaking version — 75 seconds, memorize roughly**

> "At TrueBalance the production challenge was that loan-withdrawal fraud was eating real basis points on the portfolio, and we had no real-time signal between approval and disbursement. The DS team built an XGBoost model — about 150 features — and I owned the production pipeline. The hard part was the latency budget: 500 milliseconds p99, with feature lookup against Snowflake which is a warehouse, not OLTP. So I designed a Redis online cache in front, with a write-through path from the same Airflow DAG that lands features in Snowflake. Cache hit rate is around 96% in steady state. On a cache miss I do a parallel fetch with a 200-millisecond timeout against Snowflake, and fall back to a population-mean default if Snowflake misses the budget — so the SLO never breaks, and we alert if degraded responses exceed 1%. Deployed as a Lambda container with provisioned concurrency 50, three environments each in its own VPC with its own KMS key — that's the regulatory shape RBI requires. End-to-end p99 sits between 380 and 450 milliseconds warm. The retrospective: I'd build the shadow-mode evaluation lane before the production lane next time, and I'd switch the audit log from Kinesis Firehose to synchronous Kafka for stricter compliance."

**Likely interviewer follow-ups, with narrative answers**

*Q: "Why XGBoost and not a neural net?"*
> Tabular data, around 150 mixed numerical and categorical features, hard requirement for SHAP-based explainability — gradient-boosted trees dominate this regime. Deep tabular models like TabNet or FT-Transformer are interesting but typically lose 1-2 points of AUC versus tuned XGBoost on this kind of data, and they cost more in inference latency and explainability tooling. Not worth it for this problem.

*Q: "Why Lambda over SageMaker real-time?"*
> Three reasons. Traffic is spiky — peak hours are noon and 6pm IST, near-zero at night, so I want auto-scale-to-zero economics. The model is small enough to fit comfortably in a 3GB Lambda container. And cost-wise at our sustained QPS, provisioned-concurrency Lambda lands at about a third the cost of an equivalent SageMaker endpoint. The trade-off is cold-start risk, which I bought down with provisioned concurrency 50 — sized to peak warm capacity.

*Q: "What's your cold-start strategy?"*
> Provisioned concurrency 50, which keeps 50 init'd execution environments hot at all times. Cold-start is dominated by model load from S3 — about 800ms for the XGBoost artifact plus the ONNX runtime warm-up. With provisioned concurrency, that 800ms is paid once per environment, not per request. The first request after a deploy still has the latency of warm-up, which is why we deploy with a canary of 5% then ramp.

*Q: "How do you detect drift?"*
> Two layers. PSI (population stability index) per feature, daily, comparing serving distribution to training distribution. Threshold is PSI > 0.2 on any top-10 feature; that triggers a Slack alert and opens a Jira. And output drift — the predicted-risk distribution itself, daily, KS test against training. The combination catches both feature drift and label drift.

*Q: "What's the rollback plan if a new model is bad?"*
> Two mechanisms. First, the model is loaded at Lambda init time from MLflow registry, with the registry pointer encoded in an SSM parameter. Flipping the SSM parameter and bouncing the Lambda — about 90 seconds — rolls back to the previous model with zero deploy. Second, every deploy goes 5% → 50% → 100% with metric gates at each step (error rate, p99 latency, score-distribution KS vs baseline). Auto-rollback if any gate fails for two consecutive minutes.

---

### Bullet 2 — Lender-identification NER lifting accuracy from 29.7% to 68% across 109K tradelines

**Setup**

When you pull a credit-bureau report (CIBIL, Experian) for an Indian borrower, it comes back with "tradelines" — historical loan accounts. Each tradeline has a free-text field called `creditor_name`. To use this for credit decisioning you need to map it to TrueBalance's internal lender catalog — i.e., "is this an HDFC loan, an SBI loan, an unrecognized NBFC?" The mapping accuracy directly drives the quality of every downstream feature: lender count, lender mix, NBFC ratio, prior-default-by-lender. Until I worked on it, the legacy system was a 2,000-term keyword expansion that had drifted over the years; production mapping accuracy was 29.7%, which made every downstream feature noisy.

**My role**

Own the new system end-to-end: data labeling, model selection, training, integration with the existing keyword pipeline (zero-regression requirement), CI gating, deployment.

**The hard part**

The core difficulty is that creditor names in tradelines are genuinely ugly. You see strings like `"HDFC BK LTD-MUMBAI BR"`, `"H D F C BANK"`, `"HDFCBANKLTD"`, `"HDFC NEW DELHI"`, all of which should map to "HDFC Bank". And you see `"HDFC LIFE INSURANCE"` which should NOT map to "HDFC Bank" because it's a different legal entity. Pure keyword matching can't disambiguate. NER on context can.

But there was a constraint that ruled out the obvious "throw a big model at it" approach: zero regressions. If a tradeline mapped correctly under the legacy system, it must still map correctly under the new system. The CFO had been burned before by an "improvement" that broke 4% of tradelines and threw off a regulatory report. So my mandate was: lift the long tail without touching the head.

The non-obvious decision was the architecture. I rejected "replace keyword with NER" and went with "layered: keyword first, NER fallback, BANK-keyword booster as a tie-breaker." Specifically:

```
   creditor_name string
        │
        ▼
   ┌────────────────────────┐
   │ Layer 1: keyword       │
   │ expansion (legacy 2K   │
   │ terms, exact match)    │
   └─────────┬──────────────┘
             │
       hit?  │  no hit
   ┌─────────┴────┐
   ▼              ▼
   keep      ┌─────────────────────┐
   match     │ Layer 2: DistilBERT │
             │ NER (BIO tagging)   │
             │ extracts LENDER     │
             │ entities + scores   │
             └─────────┬───────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │ Layer 3: BANK-      │
             │ lender boost        │
             │ (if entity contains │
             │ BANK / NBFC / FIN,  │
             │ boost score)        │
             └─────────┬───────────┘
                       │
                       ▼
             match in catalog? → return; else "UNMAPPED"
```

This way the head (the cases keyword nails — clean strings like `"HDFC BANK LTD"`) is untouched. The long tail (messy strings like `"hdfc-bk personal loan delhi 2023"`) gets the NER treatment. Zero regressions are guaranteed by construction because keyword runs first.

**The implementation detail interviewers love**

The training data was the bottleneck. I labeled around 5,000 tradelines manually with BIO tagging — `B-LENDER`, `I-LENDER`, `O` — and fine-tuned DistilBERT (chose DistilBERT over base BERT for inference speed; we run NER on every tradeline on every credit pull, so latency matters). Training was unremarkable — single-GPU, two epochs, F1 ~0.91 on a held-out 500-tradeline test set.

The interesting trick was active learning. After the first deployment I instrumented the system to log every tradeline where the NER and the keyword expansion *disagreed*, sampled 200 of those weekly, and re-labeled. That gave me a ~95% F1 by week 6 because I was teaching the model exactly the cases it was wrong on, not random samples.

The BANK-lender booster is a small but important detail. NER alone sometimes returns a non-bank entity ("AGRA" as a location) with a higher score than the actual lender ("BANK OF BARODA"). Boosting any entity that contains the keywords BANK, NBFC, FIN, FINANCE, or LIMITED disambiguates these cases. It's the kind of domain heuristic that ML purists scoff at and production engineers know is essential.

**The numbers**

- Accuracy: 29.7% → 68.0% across 109K tradelines
- Decomposed: 78K softpull tradelines (60% → 71%), 31K hardpull tradelines (a different distribution, harder format, 12% → 60%)
- Zero regressions verified by intersection test in CI: every tradeline in the legacy "matched" set was still in the new "matched" set
- Inference latency: p99 28ms with DistilBERT INT8 quantized, single Lambda
- Volume: 109K tradelines per credit-pull batch, processed in ~3 minutes end-to-end

**What I'd do differently**

I'd have started the active-learning loop on day one rather than week three. The biggest accuracy gains came from the 800 actively-sampled examples in weeks 3-6 — far more than from the initial 5,000 random samples. The Pareto efficiency of "label the cases the model is wrong on" is dramatic and underrated.

I would also have built a lender-catalog quality dashboard. Some of the residual error is not the NER's fault — it's that the lender catalog has stale entries (old NBFCs that have rebranded). A catalog-staleness signal would have unblocked another few percentage points.

**Speaking version — 80 seconds**

> "At TrueBalance the credit-bureau integration was bottlenecked because creditor-name to internal-lender mapping was at 29.7% accuracy across 109,000 tradelines. The legacy was a 2,000-term keyword expansion that had drifted. The constraint was zero regressions — couldn't break any case that the legacy got right. My architecture was layered: keyword first, fine-tuned DistilBERT NER as fallback, and a BANK-keyword booster as a tie-breaker. That preserves the head and lifts the long tail. I labeled 5,000 tradelines BIO-style, fine-tuned DistilBERT, F1 0.91 on the held-out set. The accuracy lift came mostly from active learning — sampling 200 cases per week where the keyword and NER disagreed, re-labeling, retraining. By week six we were at 95% F1 on the test set and 68% mapping accuracy in production. Quantized to INT8 for sub-30ms inference. Zero regressions verified in CI by intersection test. Retrospective: I'd start the active-learning loop on day one, and build a lender-catalog staleness dashboard."

**Follow-ups**

*Q: "Why DistilBERT over a larger model?"*
> Inference latency. We run NER on every tradeline on every credit pull — millions of inferences per day. DistilBERT gives 95% of BERT's accuracy at 40% of the latency. When I tested RoBERTa-base on the same task it was 1.2 F1 points higher and 2.5x slower; not worth it.

*Q: "How did you guarantee zero regressions?"*
> By architecture, not by hope. The keyword layer runs first, identical code to legacy. If keyword matches, we return that match — no NER call. NER only runs on the legacy-misses. So the legacy-hit set is preserved by construction. CI runs an intersection test on a stored 50K-tradeline regression suite on every PR; if any keyword-match disappears, the build fails.

*Q: "What did you do about the catalog itself drifting?"*
> Two things — a quarterly review with the credit-risk team where they walk through the top 100 unmapped lenders and decide rebrand/merge/add, and an automated alert when an unmapped lender appears in more than 0.5% of new tradelines. The catalog is treated as code: versioned, reviewed, deployed.

*Q: "How did you measure 68% accuracy?"*
> Manually labeled holdout of 1,000 tradelines, stratified by lender size — top 20 lenders, mid 50, long tail. The 68% is the unweighted accuracy. Weighted by volume, it's higher, around 78%, because the head lenders are the cleanest strings.

---

### Bullet 3 — 7-entity / 29-predicate ontology replacing the regex SMS engine, 100% coverage on 170K SMS, 107 tests, zero-diff seed SQL

**Setup**

In Indian consumer lending, SMS data from the borrower's phone is gold. Every bank in India sends transactional SMS for debits, credits, balance, salary, EMI bounce. If you parse those SMS reliably you have a ground-truth view of the borrower's cash flow that no bureau report gives you. TrueBalance had been parsing SMS with a regex engine since 2020 — about 600 regex patterns, organized loosely by issuer. Every time a bank changed its SMS format (which happens roughly monthly), regexes broke. The on-call engineer would patch a regex, deploy, and pray. Coverage on production SMS fields was around 85%, and the 15% miss meant we were losing high-value features for many borrowers.

**My role**

Architect and lead the migration from regex to ontology-based parsing. I was technical lead; one junior MLE worked with me on the test harness.

**The hard part**

Three things were genuinely hard.

First, the design. What does "ontology-based" even mean here? I went back and forth between three options: pure LLM extraction (too non-deterministic for a regulated cash-flow signal), template-based DSL (still brittle), and the eventual answer — a structured ontology of *entities* (the things that appear in SMS: amount, date, account, merchant, transaction-type, balance, reference-id) and *predicates* (the rules that govern valid combinations: if `type = DEBIT`, then `amount > 0` and `balance < previous_balance`). Every SMS becomes a graph of entities linked by predicates. Parsing becomes "find the assignment of entities to spans that maximizes predicate satisfaction."

Second, the migration. We had hundreds of downstream features that depended on the regex output schema. I had to make the new system byte-for-byte compatible with the old one for any SMS the old system could parse, while extending coverage to the new SMS the old system missed. The phrase "zero-diff seed SQL" on the resume is shorthand for the CI gate I built: take the production seed dataset (170K SMS), run it through old and new parsers, hash the output SQL inserts, compare. Any non-zero diff fails CI. That gate caught two subtle bugs in my new code that I would have shipped otherwise.

Third, the test count. 107 tests sounds like a lot until you realize the breakdown: 29 predicate-level tests (one per predicate, asserting it fires on positive examples and doesn't fire on negatives), 35 issuer-format tests (one per major bank's known SMS template), and 43 regression tests on production samples — the SMS that had broken the old regex over the previous two years, each pinned as a "must-parse-correctly" test. The regression tests were the most valuable.

**The implementation — ontology shape**

```
   ENTITIES (7):
   ┌─────────┬──────────────────────────────┐
   │ amount  │ "Rs.5,000", "INR 5000", "₹5K" │
   │ date    │ ISO, DD-MMM-YY, DD/MM/YY      │
   │ account │ A/C XXXX1234, last-4 digits   │
   │ merchant│ free-form, often after AT/TO  │
   │ type    │ DEBIT/CREDIT/EMI/BALANCE      │
   │ balance │ "avl bal Rs.X"                │
   │ ref     │ ref no / txn id               │
   └─────────┴──────────────────────────────┘

   PREDICATES (29 — examples):
     type=DEBIT  → amount > 0 ∧ balance < prev_balance
     type=CREDIT → amount > 0 ∧ balance > prev_balance
     type=EMI    → amount matches loan_emi_amount_window
     account     → must end in 4 digits
     date        → must be within ±2 days of receipt time
     ...

   PARSE STEP:
     1. Tokenize SMS
     2. Run named-span extractors per entity (regex + ML hybrid)
     3. Build candidate parses (combinations of spans)
     4. Score each parse by # predicates satisfied
     5. Return highest-scoring parse with confidence
```

The ontology is in YAML, version-controlled, reviewed by the credit-risk team. Adding a new bank format is now a matter of writing a few new entity-extraction patterns and verifying predicates pass — no regex archaeology.

**The numbers**

- 100% coverage on the 170K-SMS production seed dataset
- 107 tests (29 predicate + 35 format + 43 regression)
- Zero-diff SQL: every PR's CI runs the seed through both parsers and hashes the output
- Reduced on-call SMS-parse incidents from ~3-4 per week to near zero
- The migration took 11 weeks end-to-end with one engineer-week of help from junior MLE

**What I'd do differently**

I'd use an LLM as a fallback for the long tail earlier. Right now the ontology returns "unparseable" on roughly 0.4% of SMS — formats we haven't seen. An LLM with a structured-output schema (Pydantic via Instructor) could extract entities for those, with the ontology as the validation layer. I have this designed but it landed after I left for Avrioc; the team is shipping it now.

I'd also have built a "diff explorer" UI for the migration. The CI gate caught diffs but doesn't tell you *why* — you have to grep through logs. A side-by-side UI showing old-output vs new-output per SMS would have saved days of debugging.

**Speaking version — 75 seconds**

> "At TrueBalance the SMS-parsing engine was 600 regex patterns, broke roughly weekly when banks changed formats, and had 85% coverage on production. I redesigned it as an ontology — 7 entities like amount, date, type, balance, and 29 predicates that govern valid combinations, like 'a DEBIT SMS must have amount > 0 and balance lower than the previous balance.' Parsing becomes finding the entity-span assignment that maximizes predicate satisfaction. The migration constraint was zero behavioral change for the SMS the old system already handled. I built a CI gate that hashes the SQL inserts the old and new systems produce on a 170K production seed dataset; any non-zero diff fails the build. That caught two subtle bugs I'd have shipped. Test suite is 107 tests — predicate, format, and regression. We hit 100% coverage on the seed and dropped on-call incidents from 3-4 a week to near zero. Retrospective: I'd add an LLM fallback for the unparseable long tail with the ontology as a validator, and build a diff-explorer UI for migrations like this."

**Follow-ups**

*Q: "Why ontology and not just an LLM?"*
> Determinism, auditability, latency. SMS parsing feeds regulated cash-flow features; we have to be able to explain, in a CFPB-style audit, why a given SMS produced a given feature value. LLMs are non-deterministic and not auditable in that way. Latency is the second issue — we parse millions of SMS per day. An LLM call per SMS is cost-prohibitive. The ontology approach is microseconds per SMS, deterministic, and auditable. LLM is a fine fallback for the unparseable long tail; not a replacement for the core.

*Q: "How did you build the ontology in the first place?"*
> Started from the existing regex patterns — they had implicit knowledge about what entities mattered. Extracted that into the entity list. Then sat with the credit-risk team for two days and went through 200 sample SMS each, drawing the predicate graph by hand. Codified into YAML. Reviewed by both teams.

*Q: "What's the failure mode?"*
> Two. One, an SMS whose format we've never seen and whose entity extractors don't fire — we return "unparseable" and log it for review. Two, an SMS where two parses score equally — rare, but we tie-break by recency-of-format-prior. Both are alerted on; unparseable is the metric we watch.

---

### Bullet 4 — AI-powered ML workspace on Claude with Jira/GitHub/Athena/Jenkins, on-demand GPU/CPU EC2, EFS-shared state, 3-method EBS lifecycle, per-ticket git-worktree isolation

**Setup**

The ML team at TrueBalance had a productivity problem that wasn't about ML. Engineers spent measurable parts of every day on context-switching: hopping between Jira tabs, manually provisioning EC2 instances for experiments, running `git stash` / `git checkout` dances when interrupted by a P1, hand-editing Athena queries, hitting Jenkins web UI to trigger builds. The actual ML work — feature engineering, model training, evaluation — was the minority of the day.

I'd been using Claude Code personally and noticed that with proper tool integration it could collapse most of the context-switching. So I proposed a project: a Claude-backed ML developer platform that gives every engineer a single chat interface to Jira (find my tickets), GitHub (open the right branch), Athena (run that query), Jenkins (trigger this build), and AWS (give me a GPU instance for two hours).

**My role**

I scoped, designed, built the MVP, and rolled it out to the ML team. Solo project for the first four weeks; one junior MLE joined for the second phase.

**The hard part**

The hard parts were not the LLM bits. The LLM bits — Claude Sonnet with tool calls — worked well. The hard parts were the infra primitives.

First, on-demand EC2 with sane lifecycle. The naive version is "spin up a g5.xlarge when the engineer asks, terminate when they're done." That breaks the moment someone forgets to terminate at end of day — bills explode. The next-naive version is "auto-terminate after 4 hours of idle." That breaks the moment someone walks away from a 6-hour training job. So I designed a 3-method EBS lifecycle:

```
   Method 1 — EPHEMERAL (default)
     Instance starts from AMI
     EBS root volume = delete-on-terminate
     Used for: quick experiments, debugging

   Method 2 — PERSIST
     Instance starts from AMI
     EBS data volume snapshotted on stop
     Snapshot retained 7 days
     Restored on next start
     Used for: multi-day experiments, training jobs

   Method 3 — PINNED
     Instance has a named EBS data volume
     Volume detached on stop, attached on start
     Volume retained until engineer explicitly drops it
     Used for: long-running notebooks, dataset caches
```

The engineer chooses the method when they ask Claude for an instance. Claude knows the heuristics and will recommend if the engineer's described task suggests a method.

Second, EFS-shared state. Every instance mounts a shared EFS volume at `/mnt/team`. Common datasets, shared model registry caches, team-wide pip wheels, all live there once. Saves disk + saves repeated S3 downloads. The catch — EFS performance modes matter; I ended up on "Max I/O" with provisioned throughput to handle 4-5 concurrent training jobs.

Third — and this is the bit interviewers will probe — per-ticket git-worktree isolation. The default git workflow is one working tree per repo; switching branches requires stashing or committing. With 5-8 active tickets per engineer, that's chaos. So when the engineer says "Claude, work on TBL-1234," Claude:

```
  1. Checks if TBL-1234 has a git-worktree already
  2. If not: `git worktree add /workspace/TBL-1234 origin/main -b feat/TBL-1234`
  3. cd's to that worktree
  4. Sets up a Jupyter kernel rooted there with isolated env
```

Now the engineer can have 8 tickets worth of work, each in its own directory, each on its own branch, no stashing, no checkout dance. Switching tickets is `cd`. This is the productivity multiplier.

**The architecture diagram**

```
   ┌─────────────────────────────────────────────┐
   │  ML Engineer (browser, VS Code, terminal)   │
   └────────────────────┬────────────────────────┘
                        │ chat interface
                        ▼
   ┌─────────────────────────────────────────────┐
   │  React + FastAPI app                         │
   │  - SSE chat                                  │
   │  - history persistence (Postgres)            │
   └────────────────────┬────────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────────┐
   │  Claude Sonnet (Anthropic API via Bedrock)   │
   │  - tool calling                              │
   │  - extended thinking for complex requests    │
   └─────┬─────────┬─────────┬─────────┬────────┘
         │         │         │         │
         ▼         ▼         ▼         ▼
   ┌──────┐  ┌──────┐  ┌─────────┐  ┌─────────┐
   │ Jira │  │GitHub│  │ Athena  │  │ AWS EC2 │
   │ tool │  │ tool │  │ tool    │  │ + EFS + │
   │      │  │      │  │         │  │ EBS     │
   └──────┘  └──────┘  └─────────┘  │ tools   │
                                     └─────────┘
                                          │
                                          ▼
                              ┌──────────────────────┐
                              │  EC2 fleet           │
                              │  + EFS /mnt/team     │
                              │  + EBS lifecycle mgr │
                              │  + git-worktree mgr  │
                              └──────────────────────┘
```

**The numbers**

- 12 ML engineers + DS using it daily within 6 weeks
- Median time-to-ready for "I want to run a notebook on a GPU" dropped from 18 minutes (manual) to 90 seconds (Claude)
- EBS spend per engineer dropped 35% via the lifecycle methods (no more zombie volumes)
- Adoption was voluntary; no mandate

**What I'd do differently**

I'd have shipped a minimum viable version in week one rather than week four. I over-engineered the EFS performance tuning before the platform had any users. Classic mistake. The right move was: ship the worst version with one tool (Jira) on day three, get one user on it, learn what they actually want.

I'd also have invested earlier in observability of the Claude calls. We added Langfuse-style tracing in week six, and immediately spotted that the Athena tool was burning unnecessary tokens because the result schema was being re-shipped on every turn. Fixing that — by caching schema in the tool and only shipping diffs — cut per-conversation cost by ~40%.

**Speaking version — 80 seconds**

> "The pain at TrueBalance was that ML engineers spent more time context-switching — Jira, manual EC2, git-stash dances, Jenkins UI — than actually doing ML. I built a Claude-backed developer platform with tool calls into Jira, GitHub, Athena, Jenkins, and AWS. The non-trivial parts weren't the LLM bits — they were the infra primitives. EC2 lifecycle has three modes — ephemeral, persist-with-snapshot, and pinned-volume — and Claude recommends the right one based on the task. EFS gives every instance a shared `/mnt/team` for datasets and model caches. The biggest productivity unlock was per-ticket git-worktree isolation: when an engineer says 'work on TBL-1234,' Claude creates a worktree, sets up a Jupyter kernel, and the engineer can have eight tickets active at once with no stashing. Twelve users adopted within six weeks, voluntarily. Median time-to-GPU dropped from 18 minutes to 90 seconds. EBS spend down 35%. Retrospective: I should have shipped the MVP in week one with one tool, not week four with five — and added LLM-call tracing on day one, because once we did, we cut per-conversation cost 40%."

**Follow-ups**

*Q: "Why Claude over GPT or open-source?"*
> Three reasons. Tool-calling reliability is the best I've seen in production — JSON output validity rate over 99% with Sonnet, vs 95-97% with GPT-4 class models in my testing. Long-context behavior is genuinely better — at 200K tokens Claude doesn't degrade the way GPT-4-turbo does. And Bedrock data residency, which mattered for our regulated environment. Open-source — Llama-3.3-70B — was a candidate but tool-calling latency and reliability lagged.

*Q: "How do you handle destructive operations?"*
> Two layers. IAM scope — every tool has a least-privilege role, no general "do anything in AWS" power. And explicit human-in-the-loop confirmation for destructive ops: terminate instance, drop EFS data, delete branch. The model proposes; the user clicks confirm. We logged every tool call to Langfuse so we can audit.

*Q: "What about cost at scale?"*
> Per-conversation cost averages around $0.40 with Sonnet — about a quarter goes to extended-thinking tokens for the harder requests, three-quarters to regular generation. With 12 daily users and ~10 conversations each, that's ~$50/day, ~$1,200/month. Justified by the productivity numbers. We measured by hand — surveyed engineers on hours saved per week — and the platform was net-positive in the first month.

---

## 15.2 ResMed (ML Engineer, Jul 2023 – Jan 2026)

ResMed is a US-based medical-device company best known for CPAP machines for sleep apnea, plus respiratory care, ventilators, and the digital-health platform that connects all of them. Their data is the kind that interviewers light up about — millions of patients, longitudinal physiological signals (airflow, pressure, leak rate, AHI), tied to outcomes. I joined the IHS — Intelligent Health Studio — team, which is ResMed's internal MLOps platform for clinical ML.

I spent 2.5 years there, shipped multiple models to production, and the work that's most interview-relevant is the LLM/RAG work and the multi-container SageMaker work. I'll go deep on those.

---

### Bullet 5 — GenAI query routing chatbot using LLMs and RAG over clinical knowledge

**Setup**

Clinical analysts at ResMed spent hours hunting in PDFs, internal protocols, and Snowflake to answer questions like "what's the average AHI improvement after 6 months on AirSense 11?" or "what does the protocol document say about mask-leak troubleshooting for ICU patients?" The first question is analytical (needs SQL). The second is factual (needs RAG). A naive single-pipeline answer is bad at both.

**My role**

I led the design and implementation; one MLE peer worked on the eval harness, and a clinical SME built the golden set with us.

**The hard part — and the architecture**

The non-obvious decision was to *route* queries based on intent rather than try to make one pipeline do everything. The router itself is a small LLM (Haiku-class) with a few-shot prompt and a structured output. It classifies the incoming query into one of three intents:

```
   ┌───────────────────────┐
   │  User query           │
   └──────────┬────────────┘
              │
              ▼
   ┌──────────────────────────────────────┐
   │  Router LLM (Claude Haiku)            │
   │  - intent: factual | analytical |     │
   │            conversational             │
   │  - confidence score                   │
   └─────┬────────────┬────────────┬──────┘
         │ factual    │ analytical │ conv
         ▼            ▼            ▼
   ┌──────────┐  ┌─────────────┐ ┌──────────┐
   │ RAG      │  │ Code-gen    │ │ Direct   │
   │ pipeline │  │ → Snowflake │ │ LLM      │
   │          │  │ (text-to-   │ │ (no      │
   │          │  │  SQL)       │ │ retrieval│
   │          │  │             │ │  needed) │
   └────┬─────┘  └──────┬──────┘ └────┬─────┘
        │               │             │
        └───────┬───────┴─────────────┘
                ▼
        ┌───────────────┐
        │ Answer + cita-│
        │ tions (RAG)   │
        │ or table (SQL)│
        │ or text (LLM) │
        └───────────────┘
```

**RAG pipeline — the details**

The RAG pipeline is the meatiest part. Source corpus is around 12K clinical documents — a mix of protocol PDFs, internal study reports, training material, FAQ pages.

```
   Ingestion:
     PDFs → unstructured.io parsing → section-aware chunking
            (respect History/Medications/Assessment etc.
             headers; chunk size 512 tokens, overlap 50)
     → BGE-base-en embedding (768d, batched on g5.xlarge)
     → pgVector index in Aurora Postgres (hosted, in-region)
     → also store BM25 sparse index in OpenSearch

   Query:
     query → query rewrite (resolve pronouns) → hybrid search
       (BM25 top-50 + dense top-50 → RRF fuse → top-30)
     → cross-encoder rerank (bge-reranker-base) → top-5
     → assemble context with parent-doc retrieval (embed
       small chunks, return parent paragraph for context)
     → Claude Sonnet generates answer with [n] citations
     → post-process: verify each [n] maps to a retrieved chunk
```

**Eval — the ungloriously important part**

I built a 300-pair golden set with a clinical SME over a week. Each pair: question, ideal answer, expected source doc/section. Eval ran nightly via RAGAS — faithfulness (does the answer entail from retrieved docs), answer relevance (does it address the question), context precision (are the retrieved chunks relevant), context recall (did we miss any relevant chunk). Plus an LLM-as-judge "did this answer the user's actual question" 1-5 score.

Online, I added thumbs-up/thumbs-down per response; thumb-down auto-opened a Langfuse trace review queue.

**The numbers**

- 300-pair golden set, hand-built
- Faithfulness on golden: 0.91 (target was 0.85)
- Answer relevance: 0.88
- Context recall@5: 0.84
- p95 latency E2E: 2.3s (TTFT 700ms)
- Adopted by the clinical analytics team; analyst time-to-answer dropped from average ~25 minutes to ~3 minutes for the queries that hit RAG

**What I'd do differently**

I would have started with a smaller corpus — 500 documents, the most-asked-about ones — instead of trying to ingest all 12K up front. Ingestion was 3 weeks of work; I could have shipped a usable v0 in 4 days with a curated subcorpus and let user behavior tell me what to expand into.

I would also have built better evaluation lineage: a way to trace why a given golden-set pair regressed between releases. We hit a regression in week 11 where context precision dropped from 0.86 to 0.72; tracking it down took two days because we hadn't versioned chunks consistently. Lesson: chunk content-hash + index version are part of the trace, always.

**Speaking version — 80 seconds**

> "At ResMed clinical analysts were spending hours hunting in PDFs and Snowflake. The non-obvious design was to *route* queries by intent — factual goes to RAG, analytical goes to text-to-SQL against Snowflake, conversational goes direct to LLM. The router is a small LLM with a structured output. The RAG pipeline ingests 12K clinical docs with section-aware chunking — respecting History, Medications, Assessment headers — embeds with BGE-base, indexes in pgVector and BM25-in-OpenSearch in parallel. Query path is hybrid search with RRF fusion, cross-encoder rerank to top-5, parent-doc retrieval for context, and Sonnet generates with mandatory citations that we verify post-hoc against retrieved chunks. Eval is a 300-pair golden set built with a clinical SME, RAGAS faithfulness 0.91, context recall 0.84, p95 latency 2.3 seconds. Analyst time-to-answer dropped from 25 minutes to 3. Retrospective: I should have started with a curated 500-doc subcorpus instead of all 12K, and versioned chunk content-hashes from day one for regression debugging."

**Follow-ups**

*Q: "Why route instead of one pipeline?"*
> RAG cannot answer 'average AHI by mask type for the last quarter' — that's a SQL aggregation. And running SQL-codegen for 'what does protocol X say' is overkill and lossy. Routing trades a small classification-error risk against being right per query type. The router itself we measured at 96% accuracy on a 200-query labeled set.

*Q: "Why pgVector and not Pinecone or Qdrant?"*
> Data residency and operational simplicity. ResMed is HIPAA; the corpus is internal clinical data. Self-hosted Aurora Postgres in our VPC was the simplest compliant story. Qdrant would have given better recall and faster latency, but pgVector at our scale (12K docs × ~80 chunks = ~1M vectors) was fast enough. If the corpus had been 100M, I'd have moved to Qdrant.

*Q: "How did you handle hallucination?"*
> Three layers. The system prompt requires "answer ONLY from context; if context insufficient, say so." Citations are mandatory and post-hoc verified — if the model cites `[3]` and chunk 3 doesn't support it, we flag and degrade the response. And a confidence threshold on the retrieval scores; below it, we return "I don't have enough context to answer this confidently."

*Q: "Cross-encoder rerank — why?"*
> Bi-encoder retrieval (the embedding step) is fast but imprecise; cross-encoder rerank is slow but accurate because it sees query and chunk together. The combination is the standard pattern: bi-encoder gets you 100 candidates fast, cross-encoder picks the best 5 of those slowly. We used `bge-reranker-base` on a g5.xlarge, ~50ms for 100 candidates.

---

### Bullet 6 — Datadog + Snowflake drift dashboard utility, auto-generated per-model dashboards from DS-provided drift logic

**Setup**

ResMed had 30+ production ML models when I joined. Each DS team monitored drift differently — some had Datadog dashboards (hand-built), some had Slack alerts (hand-coded), most had nothing systematic. The result was that drift was usually noticed by a downstream team noticing weird outputs, not by the model owner.

**My role**

I built a utility — a small Python library + CLI — that takes a model registration record (model name, feature list, target Snowflake tables) and a drift-logic spec (which features, which test, which thresholds) and auto-generates a Datadog dashboard for that model.

**The hard part**

The hard part was getting DS adoption without mandating it. The way I cracked it: I made the utility absurdly easy to use — `drift-utility init my_model --features f1,f2,f3` produces a working dashboard in 60 seconds. Then I onboarded one team that was already in pain (high-attrition model with frequent drift), demonstrated value, and let word spread. By month three, six teams were on it; by month six, the platform team mandated it for new models.

The technical detail interviewers ask about: how does the dashboard "auto-generate"? It's a Datadog dashboard JSON template per drift-test (PSI, KS, Wasserstein), parameterized with feature names and thresholds. The utility takes the spec, fills in the template, and POSTs to the Datadog API. Each panel is a Snowflake-backed metric — Snowflake compute runs the drift calculation nightly, writes the result to a `model_metrics` table, and Datadog scrapes it via the Snowflake integration.

**The numbers**

- 11 models on the system within 4 months, 22 by the time I left
- Mean-time-to-detect drift dropped from days (anecdotal) to hours
- The utility itself is ~800 lines of Python; the test suite is ~1500

**Speaking version — 60 seconds**

> "ResMed had 30 production ML models with inconsistent drift monitoring. I built a Python utility that takes a model spec — features, drift tests, thresholds — and auto-generates a Datadog dashboard plus the Snowflake-backed metric jobs. The hard part wasn't technical; it was adoption. I started with one team in pain, made the CLI a 60-second experience, let success spread organically. Six teams in three months, eleven in four. Mean-time-to-detect drift dropped from days to hours. By the time I left, the platform team had made it a mandatory part of model registration."

**Follow-ups**

*Q: "How did the drift logic plug in?"*
> Each drift test is a Python class with `compute(reference_df, current_df) -> float`. PSI, KS, Wasserstein, JS-divergence ship out of the box. DS teams can subclass for custom logic. Spec is a YAML file checked into the model's repo.

---

### Bullet 7 — Multi-container SageMaker endpoints (IHS platform), consolidating 8 lower-traffic models on shared infrastructure

**Setup**

IHS had a sprawl problem — 30+ models, each on its own SageMaker real-time endpoint, most under 1 RPS, all paying full ml.m5.large minimum. The cost was linear in model count and most of it was idle.

**My role**

Architect the consolidation onto multi-container endpoints. Lead implementation with two MLEs.

**The hard part**

Multi-container endpoints (MCE) on SageMaker let you put up to 15 containers on one endpoint and route by `target_container_hostname`. The hard part is *which* models can share, and how to handle the noisy-neighbor problem.

I bucketed models by resource class — CPU-small (most NLP classifiers, ~256MB RAM), CPU-large (XGBoost with big feature stores, ~2GB), GPU-T4 (small vision models), GPU-A10 (mid LLMs). Within a bucket, I co-located up to 8 models per endpoint. Routing is by model ID via `target_container_hostname` in the InvokeEndpoint call.

The noisy-neighbor risk — one model's traffic spike starves another — I mitigated with per-container CPU/memory quotas (Linux cgroups in the container) and per-container autoscaling policies. The shared endpoint scales out when *any* container's invocations-per-instance exceeds threshold.

**The numbers**

- 8 long-tail models consolidated onto 1 endpoint (CPU-small bucket)
- Cost reduction: ~40% on the long-tail models
- Latency overhead from MCE routing: single-digit milliseconds (~3ms)
- Zero isolation violations in the 6 months I monitored post-launch

**Speaking version — 50 seconds**

> "ResMed had 30+ SageMaker endpoints, most under 1 RPS, paying minimum-instance cost each. I consolidated the long tail onto multi-container endpoints — up to 15 containers per endpoint, routed by container hostname. Bucketed by resource class — CPU-small, CPU-large, GPU-T4, GPU-A10. Within a bucket, co-located up to eight models. Mitigated noisy-neighbor with per-container cgroup quotas and shared autoscaling. Cost reduction on the long tail was about 40%, routing overhead about 3ms. Zero isolation incidents in six months post-launch."

**Follow-ups**

*Q: "MCE vs MME?"*
> MCE — multi-container — is for heterogeneous models with different frameworks/dependencies. MME — multi-model — is for homogeneous models, e.g., 50 versions of the same XGBoost binary. We had heterogeneity (some PyTorch, some sklearn, some XGBoost), so MCE.

*Q: "How does invocation routing work?"*
> The InvokeEndpoint API call includes a `TargetContainerHostname` parameter. SageMaker routes the request to the named container on the chosen instance. From the client's perspective it's just an HTTP call with a header.

---

### Bullet 8 — Snowflake feature store schemas (offline + online)

**Setup and detail**

ResMed had no shared feature store; DS teams computed features ad-hoc in notebooks. Train/serve skew bugs were monthly events. I designed a two-layer Snowflake feature store:

- `FEATURE_HISTORY` — point-in-time correct, partitioned by date, used for training. Queries use `AS OF` semantics so no future data leaks.
- `FEATURE_LATEST` — last-known-good per entity, used for real-time serving. Updated by the same Airflow DAG that lands history.

Critical detail: feature computation logic lives in *one* shared Python library that both the batch (Airflow) and real-time (Lambda) paths import. Train/serve skew is prevented by construction, not by hope.

For low-latency serving, Redis fronts `FEATURE_LATEST`. Snowflake queries are 100-300ms; Redis is sub-50ms.

---

### Bullet 9 — Apache Airflow orchestration of preprocessing

DAGs for daily feature ingestion, weekly retraining triggers, monthly cold-storage archival. Operators wrap SageMaker training jobs and Snowflake stored procs. Standard MLOps; mention briefly unless probed.

---

### Bullet 10 — Async SageMaker endpoints for long-running inference

For inference jobs with 10+ second compute (full-report analysis on a multi-page clinical document), real-time endpoints don't make sense. I used SageMaker async endpoints — request goes to S3, response polled from S3. Scales to zero when idle, no client-side timeout pressure. Mention as a tool-fit story: "right tool for inference latency profile that doesn't fit real-time."

---

## 15.3 Tiger Analytics (ML Engineer, Dec 2021 – Jul 2023)

Tiger was a consulting shop — projects rotated, clients varied (Mars Petcare was the major one). Most relevant work was MLOps tooling.

---

### Bullet 11 — SageMaker training + inference pipelines with drift monitoring

Standard end-to-end MLOps: CodePipeline orchestrating SageMaker Pipelines (preprocess → train → eval → register → deploy), Model Monitor for data quality and drift, CloudWatch alarms feeding EventBridge for retraining triggers. The interesting part was the *quality gate* — the eval step had to clear a held-out F1 threshold before the model was registered, and the registration step had to clear a fairness check (no slice's F1 dropped more than 2% relative to baseline). Mention quality gates explicitly; it's a senior signal.

---

### Bullet 12 — CI/CD pipelines for model deployment

Multi-env (dev → staging → prod) via CodePipeline; blue-green deploy to SageMaker endpoints with traffic-shift; Terraform-managed infra. The senior detail: each env had different gates — dev allowed any model, staging required eval-set passing, prod required canary metrics passing for 24h.

---

### Bullet 13 — Automated retraining based on monitoring reports

EventBridge watches Model Monitor outputs; if drift > threshold for 2 days, fires a Step Function that pulls latest data, retrains, evaluates, registers, deploys to dev. Human-in-the-loop for the dev → staging promotion. Don't auto-promote to prod blindly — that's how bad models hit users.

---

### Bullet 14 — Custom data quality with Deequ on Databricks (Mars project)

**Setup**

Mars's data platform had silent data-quality bugs — null spikes in pet-purchase data, schema drift in supplier feeds — that DS only noticed when their models started producing weird outputs. I added Deequ-based DQ checks as a pre-ML gate.

**Detail**

Deequ constraints per table — `hasCompleteness("user_id", _ >= 0.99)`, `hasMin("age", _ > 0)`, `hasDistribution("country", country_profile)`. Runs as scheduled Databricks jobs. Failures published to Slack and Azure Monitor with table ownership metadata so the right DS team gets paged.

**Numbers**

Caught ~12 data-quality issues in the first quarter that would otherwise have polluted models. DS spent less time debugging "the model" when the actual issue was upstream data.

---

### Bullet 15 — Azure Databricks + Data Factory orchestration

ADF for pipeline orchestration triggering Databricks activities; Databricks for big-data preprocessing and drift-detection scripts. Standard. Brief mention.

---

## 15.4 Sopra Steria (Senior Software Engineer, Aug 2018 – Dec 2021)

Sopra was the early-career years — broad exposure across CV, anomaly detection, XAI, and combinatorial optimization. I'll keep these tight; they're useful for showing breadth and longevity, less likely to be deep-probed by a senior interviewer.

---

### Bullet 16 — CNN + YOLO + OCR ID verification with font anomaly detection

**Setup**

Client (financial services) needed automated ID verification — passport / driver's license / national ID. Manual review was the bottleneck.

**Architecture**

```
   Image upload
     → YOLO v3/v4 → detect ID + crop
     → CNN classifier → identify ID type (passport, DL, etc.)
     → Tesseract OCR → extract fields
     → font-anomaly detector (CNN trained on legitimate vs forged fonts)
       → flag suspicious ID for human review
```

YOLO trained on ~5K labeled samples (real IDs with ground-truth boxes); CNN classifier ~10K labeled per class. Font-anomaly was the differentiator — forged IDs often use slightly off fonts that pass casual review but fail a CNN trained on legitimate templates.

**Numbers**

Reduced manual review queue by ~60%; false-positive rate on font-anomaly ~3% (we tuned for high recall, low FN).

---

### Bullet 17 — Time-series anomaly on Prometheus/Grafana for server metrics

Pulled metrics from Prometheus, forecast with ARIMA + LSTM ensemble, anomaly when residual > k*std. Grafana alerts when anomaly score crossed threshold. Foundation in classical TS forecasting; mention if asked about anomaly detection.

---

### Bullet 18 — Loan-risk XAI ensemble with SHAP

Ensemble of XGBoost + Random Forest + Logistic Regression; SHAP per-prediction; UI showing "applicant rejected because feature X (value Y, contribution Z to risk score)." Senior detail: XAI is regulatory in lending — we couldn't ship a black-box model without per-decision explainability.

---

### Bullet 19 — OR-Tools route planning, 300 locations, 7→5 days

Vehicle routing problem with capacity constraints. OR-Tools CP-SAT solver. Reduced delivery cover time from 7 to 5 days for a 300-location daily route. Useful as a "I'm not just an LLM person — I've shipped classical optimization too" datapoint.

---

### Bullet 20 — Oracle PL/SQL ETL

Legacy work; mention only if asked. Useful for breadth but not interview-relevant.

---

## 15.5 Awards & certifications

- **Sopra Steria India Coding Championship — 1st place** — small mention in behavioral; shows competitive intellect.
- **Google Code Jam Qualifier 2019** — algorithmic credibility.
- **LangChain LangGraph foundations + Deep Agents** — directly relevant to JD; mention naturally when LangGraph comes up in conversation.
- **TensorFlow Developer Certificate** — classical DL proficiency; brief mention.

---

## 15.6 The 5 signature stories — ready for any behavioral question

These five stories are versatile enough to answer almost any behavioral question. Memorize them as 90-second deliveries; you can repurpose them on the fly.

### Story 1 — Real-time XGBoost Lambda (TrueBalance)
**Headline:** Shipped a production credit-risk model with sub-500ms p99 across three VPC-isolated environments.
**Use for:** "Shipping under tight constraints," "biggest technical achievement," "production rigor," "tell me about your most recent project."

### Story 2 — Lender NER 29.7% → 68% (TrueBalance)
**Headline:** Lifted a critical accuracy metric by more than 2x with zero regressions, by layered architecture and active learning.
**Use for:** "Improved a legacy system," "had to deliver without breaking what worked," "data-driven decision-making."

### Story 3 — SMS ontology migration (TrueBalance)
**Headline:** Replaced a brittle 600-regex SMS parser with an ontology-based system, 100% coverage, zero-diff migration.
**Use for:** "Hardest debugging problem," "architected something from scratch," "convinced a team to change a default."

### Story 4 — RAG clinical chatbot with intent routing (ResMed)
**Headline:** Built a routed RAG/SQL/LLM system for clinical analysts; analyst time-to-answer 25 min → 3 min.
**Use for:** "Designed for users not engineers," "cross-functional collaboration," "evaluation rigor on LLM systems."

### Story 5 — Datadog drift utility platform adoption (ResMed)
**Headline:** Built a tool that became the team standard not by mandate but by being absurdly easy to adopt.
**Use for:** "Influence without authority," "leadership," "platform thinking," "adoption strategy."

---

Continue to **[Chapter 16 — System Design](16_system_design.md)**.
