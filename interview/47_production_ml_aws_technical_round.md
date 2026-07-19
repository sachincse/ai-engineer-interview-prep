# Chapter 47 — Production ML on AWS: 2nd-Round Technical Q&A Bank (Monitoring · IaC · Scheduling · Latency/Throughput · Orchestration)

> **Why this chapter exists:** This is the **exact 21-question set from a 2nd-round technical interview** — a production/MLOps-heavy round that skips model theory and drills how you *operate* ML on AWS: monitoring & drift, safe model updates, rollback across accounts, Terraform/IaC, load balancing & auto-scaling, job scheduling, latency vs throughput, and orchestrating many models + ETL jobs as one system. Every question below is answered in **spoken-interview form** — the way you'd actually say it — plus a *gotcha* line and a *how to anchor it to my experience* hook. Read it as a script you can paraphrase, not memorize.
>
> **Pairs with:** Ch.10 (MLOps/LLMOps), Ch.11 (AWS & Azure), Ch.12 (Kubernetes/Ray), Ch.14 (Monitoring & Drift), Ch.16 (System Design), Ch.24 (Apache Airflow), Ch.38 (Lambda cold-start ML), Ch.44 (Cognine Senior MLE AWS). This chapter is the *production-operations* layer; those give the depth behind each answer.

---

## 1. How to use this in the room

This round is a **filter for operational maturity**, not algorithms. The interviewer wants to hear: monitoring, failure modes, rollback, cost, and blast-radius control *before they ask*. Every answer should carry a whiff of "…and here's how it breaks in production and how I'd catch it."

**My one-sentence frame:**
> "I build and operate production ML on AWS end-to-end — data on Spark/Glue/Airflow, features in S3/Athena, training and serving on SageMaker, wrapped in Terraform IaC, CloudWatch monitoring, and blue-green deploys with automatic rollback."

**Answer shape for every question:** *define it in one line → how I do it concretely → the failure mode / trade-off → a one-line anchor to my work.*

---

## 2. Machine Learning in Production — monitoring, metrics, drift (Q1)

### Q1a. How do you monitor the performance of a model in production?

I monitor on **four layers**, because "the model" isn't the only thing that can fail:

```
   LAYER 1  Infra / service      latency (p50/p95/p99), throughput (RPS),
                                  error rate (5xx), CPU/GPU/mem, saturation
   LAYER 2  Data / input health  schema validation, null rate, range checks,
                                  feature distribution vs training baseline
   LAYER 3  Model behaviour      prediction distribution, score/confidence
                                  histogram, class balance of outputs, drift
   LAYER 4  Business / outcome   the real KPI — approval rate, default rate,
                                  fraud caught, conversion — the ground truth
```

The trap is only watching Layer 1. A model can be **perfectly healthy on latency and 200-OK while silently making garbage predictions** because an upstream feature pipeline changed. So I always instrument Layers 2 and 3, and I close the loop with Layer 4 whenever labels are available.

Concretely on AWS: **CloudWatch** for infra metrics + alarms, **SageMaker Model Monitor** (or a custom Glue/Lambda job writing to CloudWatch custom metrics) for data-quality and drift baselines, structured request/response logging to **S3** so I can replay and compute delayed metrics once labels arrive, and a dashboard (CloudWatch / Grafana) plus **SNS/PagerDuty** alerts.

> **Say it like this:** "I don't trust a green latency dashboard. I log every request and prediction to S3, compare live feature distributions against the training baseline, and only then trust the business KPI — because the model can be up and still be wrong."

### Q1b. What metrics do you track?

Split by problem type, and always **offline (with labels)** vs **online (proxy, no labels yet)**:

| Problem | Primary metrics | Why / when |
|---------|-----------------|-----------|
| **Regression** | **RMSE, MAE**, MAPE, R² | RMSE punishes large errors (good when big misses are costly); **MAE** is robust to outliers and in the target's units; MAPE for relative error but breaks near zero. I report **both RMSE and MAE** — the gap between them tells me about outliers. |
| **Classification** | **Precision, Recall, F1**, ROC-AUC, PR-AUC, log-loss | Accuracy lies on imbalanced data. In **banking/fraud** I lead with **PR-AUC and recall at a fixed precision**, because the positive class is rare and false negatives are expensive. |
| **Ranking / reco** | NDCG, MAP, MRR, Recall@k | Order matters more than pointwise correctness. |
| **Probabilistic** | Calibration (reliability curve, ECE, Brier) | A 0.9 score should mean 90% — critical when a threshold drives a money decision. |

The senior move is choosing the metric from the **business cost of each error type**, not defaulting to accuracy. "Precision vs recall" is really "cost of a false positive vs a false negative" — for loan fraud I bias to recall; for auto-approving customers I bias to precision.

**Online, before labels land:** prediction distribution, score histogram, positive-rate, and null/coverage rate — so I catch a break in *hours* instead of waiting weeks for ground truth.

### Q1c. How do you detect model drift?

First I name the three things people lump together:

- **Data drift (covariate shift)** — P(X) changes; the input distribution moved (e.g., a new customer segment).
- **Concept drift** — P(Y|X) changes; the *relationship* moved (fraud tactics evolve, so the same inputs now mean something different).
- **Label / prior drift** — P(Y) changes; the base rate shifted.

Detection, no-labels-needed (fast) vs labels-needed (true but delayed):

```
  WITHOUT labels (input side):        WITH labels (output side):
  • PSI (Population Stability Index)   • rolling RMSE / F1 / AUC over time
      <0.1 stable, 0.1–0.25 watch,     • confusion-matrix shifts
      >0.25 significant shift          • calibration decay
  • KS test (continuous features)      • business KPI regression
  • Chi-square (categorical)
  • KL / JS divergence, Wasserstein
  • embedding drift (avg cosine dist)
```

My practical setup: **SageMaker Model Monitor** captures a training baseline and runs scheduled comparisons; for custom needs I run a **daily Glue/Lambda job** that computes **PSI per feature** and prediction-distribution shift, pushes them as CloudWatch custom metrics, and alarms past thresholds. Crucially, **drift is a signal to investigate, not an auto-retrain trigger** — I gate retraining on a labelled metric drop, because you can have harmless drift (a marketing campaign shifting traffic) that doesn't hurt accuracy at all.

> **Gotcha:** if the interviewer says "how do you know it's drifting without labels?" — the answer is PSI/KS on features and the *prediction distribution*, not accuracy. Accuracy needs labels you don't have yet.

> **Anchor:** "At TrueBalance the sharpest early drift signal was usually a feature-distribution shift right after a product or campaign change — I'd catch it on PSI a week before the default-rate KPI moved."

---

## 3. Updating a model in production — deployment strategies (Q2)

### Q2a. Describe the process of updating a model in production.

I treat a model update like a **software release with a data dependency**, not a file swap:

```
  1. TRAIN & VALIDATE   new model in the build/staging account; log to
                        model registry with version, data snapshot, metrics
  2. GATE               offline eval must beat the incumbent on the agreed
                        metric on a frozen holdout + slice checks (no
                        regression on key segments / fairness slices)
  3. PACKAGE            containerize + pin deps; register artifact + metadata
                        (git SHA, training data hash, hyperparams)
  4. SHADOW / CANARY    serve to a slice of live traffic; compare online
  5. PROMOTE            shift traffic gradually; watch metrics live
  6. MONITOR + ROLLBACK path armed the whole time (see §4)
```

Key discipline: **the model registry is the source of truth** (SageMaker Model Registry / MLflow), every deploy is reproducible from a git SHA + data snapshot, and I never promote on offline metrics alone — I want a **shadow or canary** on real traffic first.

### Q2b. Steps to ensure a smooth transition + minimize downtime and risk

- **Immutable, versioned artifacts** — new version deployed alongside old, never overwrite.
- **Backward-compatible contract** — same input schema / API, or a versioned endpoint, so callers don't break.
- **Shadow mode first** — mirror live traffic to the new model, log its predictions, compare, but **don't serve them** to users. Zero user risk.
- **Canary / gradual traffic shift** — 5% → 25% → 50% → 100%, watching metrics at each step.
- **Automated rollback triggers** — a CloudWatch alarm on error rate / latency / prediction-drift auto-reverts.
- **Warm the new fleet** before shifting traffic to avoid cold-start latency spikes.

### Q2c. Blue-Green vs Canary vs Shadow (explain the strategies)

| Strategy | What it is | Pros | Cons | When I use it |
|----------|-----------|------|------|---------------|
| **Blue-Green** | Two identical environments; **Blue** = live, **Green** = new. Deploy to Green, test, then **flip the router** 100% to Green. Blue stays hot as instant rollback. | Instant cut-over, instant rollback (flip back), zero downtime | 2× infra cost during overlap; the switch is all-or-nothing for that instant | Default for **stateless inference endpoints** where I want a clean, reversible cut-over |
| **Canary** | Route a **small % of real traffic** to the new version, ramp up as it proves out | Limits blast radius; catches real-traffic issues early; cheaper than full 2× | Slower rollout; needs good per-version metrics | When I'm less certain and want graduated exposure |
| **Shadow (mirror)** | New model gets a **copy of traffic** but its output is **not served** | Zero user risk; real-traffic validation before any exposure | 2× compute; can't measure business outcome (output not used) | Highest-stakes changes; validating a rewrite before canary |
| **Rolling** | Replace instances a few at a time in place | No extra environment cost | Mixed versions live simultaneously; slower rollback | K8s/ECS deployments where 2× cost is unwanted |

> **Say it like this:** "For an inference endpoint I default to **blue-green** because rollback is a router flip — sub-second. When I'm less sure about a model, I **shadow** it first to validate on real traffic with zero user risk, then **canary** 5–25% before going green all the way. SageMaker endpoints support production variants with weighted traffic, so I get canary/blue-green natively."

> **Gotcha:** blue-green's "instant rollback" assumes the new version didn't **write incompatible state** (e.g., a migrated feature store). If there's a stateful side-effect, rollback isn't free — I call that out.

---

## 4. Model rollback strategy across build & prod accounts (Q3)

**Scenario:** model artifacts live in a **build account**; inference runs in a **production account**. New version performs poorly — roll back.

### The design

```
  BUILD / CI ACCOUNT                        PRODUCTION ACCOUNT
  ┌───────────────────────┐                 ┌───────────────────────────┐
  │ train + validate      │   promote       │ SageMaker endpoint         │
  │ Model Registry (SoT)  │  ───────────▶   │  variant-A (v1.3)  90%     │
  │ artifacts in S3 (ver) │  cross-account  │  variant-B (v1.4)  10%     │
  │ approve → deploy       │  role assume    │ CloudWatch alarms armed    │
  └───────────────────────┘                 └───────────────────────────┘
        immutable, versioned                    traffic weights = the knob
```

**The core principle: rollback is a pointer/weight change, not a rebuild.** I keep the **previous known-good version still deployed (or one command away)** and roll back by **shifting the traffic weight back to it** — I never rebuild or redeploy under fire.

### How the cross-account part works

- Artifacts are **versioned and immutable in S3** in the build account (or replicated to a prod-account bucket); the **model registry holds the approved lineage**.
- Prod assumes a **cross-account IAM role** (or the S3 bucket policy / ECR repo policy grants the prod execution role read access) to pull the exact artifact — **no rebuild in prod**, prod just references an immutable, already-approved artifact by version/digest.
- The production **SageMaker endpoint runs two variants** (blue-green or weighted canary). v1.3 stays deployed while v1.4 ramps.

### What I do when the new version performs poorly

1. **Detect** — CloudWatch alarm fires on error rate / latency / prediction-drift / a guardrail business metric.
2. **Roll back = shift traffic weight back to the previous variant (v1.3 → 100%)** via `UpdateEndpointWeightsAndCapacities`. This is **near-instant and needs no redeploy**, because v1.3 is still warm.
3. **Automate it** — an alarm → Lambda that flips the weights, so mean-time-to-recover is seconds, not a human paging cycle.
4. **Contain & diagnose** — quarantine the bad artifact in the registry (mark **Rejected**), capture the failing requests from S3 data-capture, reproduce in the build account.
5. **Guard the data side** — if the new model wrote to a feature store / DB, I need a compensating action or a versioned feature table; a model rollback doesn't undo bad writes.

> **Say it like this:** "Rollback should be a **traffic-weight flip to a still-warm previous version**, driven automatically by a CloudWatch alarm — not a redeploy. Because the artifact is immutable and versioned in the build account and prod just references it cross-account by digest, 'go back to v1.3' is one API call. I keep the last-good version deployed precisely so recovery is seconds."

> **Gotcha to raise proactively:** the two dangerous rollbacks are (1) **schema/contract change** — v1.4 expected a new feature; you must roll back the *feature pipeline* too, and (2) **stateful side-effects** — anything the model wrote. I version the feature tables and keep them backward-compatible for exactly this reason.

---

## 5. Infrastructure as Code / Terraform (Q4–Q5)

### Q4. Are you familiar with IaC?

Yes — I define infra (VPCs, SageMaker endpoints, Lambda, S3, IAM roles, Glue jobs, alarms) as **declarative code in Terraform**, kept in git, applied through CI. The whole stack for a model is stand-up-able from the repo by a teammate.

### Q5. Why use tools like Terraform?

| Reason | What it buys me |
|--------|-----------------|
| **Reproducibility** | Same code → same infra across dev/stage/prod. Kills "works in my account" drift. |
| **Version control & review** | Infra changes are PRs — reviewed, diffed (`terraform plan`), audited, revertible. |
| **Idempotency & drift detection** | `plan` shows exactly what changes before `apply`; detects manual console drift. |
| **Multi-account / multi-region** | Same modules parametrized per environment — exactly the build-vs-prod-account setup in §4. |
| **Disaster recovery** | Rebuild an entire environment from code. |
| **Cloud-agnostic-ish** | One tool/workflow across AWS/Azure/GCP (vs CloudFormation = AWS-only). |
| **Least-privilege by design** | IAM roles/policies are code — reviewable, not clicked together in a console. |

**State is the thing to get right:** remote state in **S3 with DynamoDB locking**, separate state per environment, and **never** secrets in state — those go to Secrets Manager / SSM Parameter Store. I structure with **reusable modules** (a `model-endpoint` module) and use **workspaces or separate state** per account.

> **Say it like this:** "IaC turns infra into reviewable, reproducible, revertible code. For the build-vs-prod split, Terraform modules parametrized per account are exactly how I keep the two environments identical and promote safely. CloudFormation is the AWS-native alternative; I prefer Terraform for multi-cloud and the `plan` diff."

> **Gotcha:** Terraform vs Ansible — Terraform is **provisioning** (declarative infra), Ansible is **configuration management** (procedural, on existing machines). Don't conflate them.

---

## 6. Load Balancing & Auto Scaling on AWS (Q6–Q8)

### Q6. What is Load Balancing?

Distributing incoming requests across **multiple backend instances** so no single one is overwhelmed — for **availability** (route around a dead instance via health checks), **scalability** (add capacity horizontally), and **latency** (spread load). Algorithms: round-robin, least-connections, weighted, IP-hash.

### Q7. What is Auto Scaling?

**Automatically adding/removing compute** to match demand — scale **out** under load, scale **in** when quiet. It gives you **elasticity + cost efficiency** (pay for what you need) and **resilience** (replaces unhealthy instances). Two axes: **horizontal** (more instances — the usual) vs **vertical** (bigger instance — rarer, needs a restart).

- **Reactive** scaling: **target-tracking** (keep CPU/RPS/latency at a target), **step** scaling (thresholds), **simple** scaling.
- **Proactive**: **scheduled** scaling (I know the 9am batch spike) and **predictive** scaling.

### Q8. What AWS services have you used for Load Balancing & Auto Scaling?

**Load balancing:**
- **ALB (Application Load Balancer)** — L7, HTTP/S, path/host routing → my default for **model inference APIs** behind ECS/EKS; supports weighted target groups (canary).
- **NLB (Network Load Balancer)** — L4, ultra-low latency, high throughput, static IP — for gRPC / non-HTTP or extreme-throughput.
- **API Gateway** — managed front door for Lambda-backed inference (throttling, auth, usage plans).

**Auto scaling:**
- **EC2 Auto Scaling Groups** with target-tracking policies.
- **SageMaker endpoint auto-scaling** — scales inference instances on `InvocationsPerInstance` / latency; I set **min capacity ≥ 1** (or provisioned) to avoid cold starts, and scale-in cooldowns to avoid flapping.
- **ECS/EKS**: service auto-scaling + **Cluster Autoscaler / Karpenter**; **KEDA/HPA** on Kubernetes for event- or metric-driven scaling.
- **Lambda** — scales concurrency automatically; **provisioned concurrency** to kill cold starts for latency-sensitive inference (see Ch.38).

> **Say it like this:** "For a real-time model API I put an **ALB** in front of a SageMaker endpoint or an ECS service, with **target-tracking auto-scaling** on invocations-per-instance and p95 latency. I keep a warm floor so scale-from-zero cold starts don't hit the tail latency, and I tune cooldowns so it doesn't flap. For bursty async work I lean on Lambda with provisioned concurrency."

> **Gotcha:** GPU inference scales slowly (image pull + model load can be minutes). So for GPU endpoints I **pre-warm**, keep a higher min capacity, and scale on a **leading** signal (queue depth) rather than a lagging one (CPU).

---

## 7. Scheduling processing jobs (Q9–Q12)

### Q9. Have you scheduled jobs daily/hourly? Q10. Which tools?

Yes — daily feature builds, hourly scoring batches, nightly retraining candidates, model-monitor jobs. Tools I've used:

| Tool | I use it for |
|------|--------------|
| **Apache Airflow** (MWAA) | **Primary orchestrator** for multi-step ML/ETL DAGs with dependencies, retries, backfills, SLAs (see Ch.24) |
| **Cron** (EC2 crontab / systemd timers) | Simple, single-box, no-dependency jobs |
| **AWS Glue triggers** | Scheduled/event-driven Glue ETL jobs; trigger chains (job A success → job B) |
| **EventBridge (CloudWatch Events) Scheduler** | Cron-expression triggers for Lambda / ECS / Step Functions / SageMaker jobs — the serverless "cron" |
| **Step Functions** | Stateful workflows with retries/branching, often kicked by EventBridge on a schedule |
| **SageMaker Pipelines** | Scheduled training/eval/registration pipelines |

**Cron vs a real scheduler:** cron is fine for one isolated task on one box, but it has **no dependency management, no retry/backfill, no visibility, and a single point of failure**. The moment jobs depend on each other or need observability, I move to **Airflow / Step Functions / Glue triggers**.

### Q11. What issues did you hit while scheduling jobs?

The real-world failure catalogue:

- **Silent failures** — job "succeeded" but produced no/partial data (empty upstream). Fix: **data-quality asserts** and row-count checks in the DAG, not just exit-code success.
- **Overlapping runs** — a slow run still executing when the next fires → double-processing / race. Fix: `max_active_runs=1`, `depends_on_past`, or a lock.
- **Timezone / DST bugs** — cron in UTC vs local; DST double/skip fires. Fix: **everything in UTC**.
- **Upstream not ready** — job runs before its input lands. Fix: **sensors / event-driven triggers** instead of a hopeful fixed time.
- **Thundering herd** — many jobs at `0 0 * * *` hammer a DB. Fix: stagger schedules.
- **Retry storms & non-idempotency** — a retried job double-writes. Fix: **idempotent jobs** (upsert/overwrite a partition, not append).
- **Resource contention / cost** — everything at midnight blows the cluster. Fix: concurrency pools, spread the load.
- **Backfill pain** — needing to reprocess history. Fix: **parametrize by run-date**, never `Date.now()` inside the job.

### Q12. How did you troubleshoot failed jobs & debug logs?

1. **Centralized logs** — Airflow task logs / **CloudWatch Logs** for Lambda/Glue/ECS; I go straight to the failed task's log, not the whole run.
2. **Alerting** — SNS/Slack on failure with the run-date and task name so I know *which* partition failed.
3. **Reproduce with the exact params** — rerun the single failed task with its run-date locally / in staging (this is why jobs are parametrized and idempotent).
4. **Isolate: data vs code vs infra** — did the input change (schema/nulls), did a deploy change logic, or did the box run out of memory/quota? The log layer usually tells me which.
5. **Retries with backoff** for transient (throttling, network); **fail-fast + alert** for deterministic (bad schema) so I don't retry-storm a real bug.
6. **Fix forward, then backfill** the missed partitions idempotently.

> **Say it like this:** "Most 'scheduling' bugs aren't the scheduler — they're **non-idempotent jobs, missing upstream data, and silent partial success**. So I make every job idempotent and parametrized by run-date, add data-quality asserts so a 'green' run that produced no rows still fails loudly, and pipe failures to Slack with the partition ID so I can rerun just that one."

---

## 8. Latency vs Throughput (Q13–Q15)

### Q13. What is Latency?

The **time for a single request** to complete — end to end. Measured in ms/s. It's a **distribution, not a number** — I always talk **p50/p95/p99**, because the tail is what users feel. "Average latency" hides the p99 that's timing out.

### Q14. What is Throughput?

The **number of requests handled per unit time** — RPS / QPS, or records/sec for batch. It's about **total volume/capacity**, not the speed of any one request.

### Q15. For a heavy-read system, optimize latency or throughput — and why?

**It depends on who's reading, but for a heavy-read *user-facing* system I optimize latency first, then scale throughput horizontally** — because reads are usually **independent and embarrassingly parallel**, so throughput I can buy with more replicas/instances behind a load balancer, but **latency is felt per user on every read** and can't be fixed by just adding boxes.

The senior framing — **they're a trade-off (Little's Law: concurrency ≈ throughput × latency)**, and the answer depends on the workload:

- **User-facing reads (API, product page, real-time inference):** **latency-first.** Users abandon on slow tail latency. Then I get throughput by **horizontal scaling + caching**. Levers: **caching** (Redis/ElastiCache, CDN), **read replicas**, **denormalization / materialized views**, **indexes**, keeping the hot set in memory.
- **Analytics / batch reads (nightly aggregation, training data pull):** **throughput-first.** Nobody's waiting on a single row; I want max rows/sec — **columnar formats (Parquet), partitioning, parallel scans (Spark/Athena), bigger batches**. Per-query latency barely matters.

> **Say it like this:** "For a heavy-**read** system I first ask: interactive or batch? If it's user-facing, I optimize **latency** — because throughput on independent reads I can buy horizontally with replicas and caching, but latency is felt on every single request and adding boxes won't fix a slow query path. If it's analytics, I flip to **throughput** — columnar, partitioned, parallel scans, big batches. And I always remember they trade off: batching lifts throughput but *adds* latency, so I tune batch size to the SLA."

> **Gotcha (ML-specific):** **dynamic batching** on an inference server (Triton/vLLM) raises GPU throughput but adds queueing latency. I size the max batch/wait window to the **p99 latency SLA** — throughput is worthless if it breaks the latency budget.

---

## 9. Orchestrating multiple ML models + ETL as one system (Q16–Q21)

This is the system-design core of the round. I'll walk one concrete architecture and answer Q16–Q21 against it.

### Q16 + Q17. How do you orchestrate many models & ETL jobs as one system — explain the architecture

I orchestrate with a **DAG-based orchestrator (Airflow / Step Functions / SageMaker Pipelines)** as the control plane, and a **layered data lake (S3) as the data plane** — jobs communicate through **data + a metadata/registry layer**, not by calling each other directly. Loose coupling via storage is what makes it operable.

```
                          ORCHESTRATOR (Airflow / Step Functions)  ── control plane
                          schedules, dependencies, retries, SLAs, alerts
   ┌───────────┬──────────────┬───────────────┬───────────────┬─────────────────┐
   ▼           ▼              ▼               ▼               ▼                 ▼
 INGEST     ETL / CLEAN    FEATURE ENG     MODEL A          MODEL B          POSTPROCESS
 (raw→S3)   (Glue/Spark)   (Feature Store) (score)          (uses A's out)   / aggregate
   │           │              │               │               │                 │
   └──── S3 lake (bronze → silver → gold) + Feature Store + Model Registry ──────┘   ── data plane
                          (jobs talk through DATA, not direct calls)
   Metadata: data-quality checks, lineage, run-date partitions, CloudWatch metrics
```

**Layers:**
- **Ingestion** → raw/**bronze** in S3 (immutable, partitioned by date).
- **ETL/clean** → **silver** (validated, conformed) via Glue/Spark.
- **Feature engineering** → **gold** / **Feature Store** (shared features, versioned, point-in-time correct).
- **Model layer** → each model reads its features, writes predictions back to S3/DB (a versioned prediction table).
- **Ensembles/chains** → a downstream model consumes an upstream model's output (declared as a DAG dependency).
- **Serving/aggregation** → combine outputs into the business decision.

**Two orchestration patterns:** **batch** (Airflow DAG on schedule — the above) and **real-time** (Step Functions or an event/queue chain — SQS/Kinesis/EventBridge between stages) when models must run per-request in sequence.

### Q18. Data flow between models

- **Through the data/storage layer, not point-to-point calls.** Model A writes predictions to a **versioned, partitioned table** (S3/Parquet or a DB); Model B reads that partition as an input feature. This **decouples** them — B doesn't care *how* A ran, only that A's output for `run_date=X` exists.
- The orchestrator enforces order: **B's task `depends_on` A's task**, and a **sensor/quality-gate** confirms A's output landed and passed checks before B starts.
- For **real-time chains**, A publishes to a **queue/stream (SQS/Kinesis)** and B consumes — same decoupling, async.
- **Contracts** between stages are explicit: a schema (and ideally a schema registry) so an upstream change can't silently corrupt a downstream model.

### Q19. Managing dependencies

- **DAG dependencies** — the orchestrator is the single source of truth for "what runs after what." No job hard-codes another job's schedule.
- **Data dependencies via sensors / event triggers** — a downstream stage waits for the *data* (partition present + quality-passed), not a hopeful clock time.
- **Environment/library dependencies** — **containerize each job** (pinned deps) so Model A's TF version can't clash with Model B's; artifacts pinned by digest.
- **Idempotency + parametrized run-date** — any node can be re-run/backfilled independently.
- **Failure isolation** — one model failing shouldn't silently corrupt the pipeline: fail the branch, alert, and let independent branches continue.
- **Versioning everywhere** — data snapshots, feature versions, model versions in the registry — so a re-run is reproducible.

### Q20. Identifying primary keys across datasets

This is the join-correctness question — get it wrong and every downstream model is subtly poisoned.

- **A primary key uniquely identifies a row**; across datasets I need the **join key** that links them (a foreign key). I identify it by: business meaning (`customer_id`, `transaction_id`, `account_id`), **uniqueness + non-null validation** (`COUNT(DISTINCT k) == COUNT(*)` and 0 nulls), and profiling.
- **Natural vs surrogate keys** — if no reliable natural key exists, I mint a **surrogate key** (a generated ID or a deterministic **hash of the identifying columns**) to join on.
- **Composite keys** — often the real grain is multiple columns (e.g., `customer_id + date`, or `order_id + line_item`). I define the **grain** explicitly per table.
- **Point-in-time correctness** — for ML features I don't just join on ID, I join **as-of the event time** (`customer_id` + `event_timestamp`) to **prevent leakage** — no using future data.
- **Validation as a gate** — before any join in the pipeline I assert key uniqueness and check the join **doesn't fan out** (a many-to-many blow-up inflates rows and corrupts features). Duplicate-key detection is a standing data-quality check.

> **Gotcha:** the classic bug is a **non-unique "key"** causing a fan-out join that silently duplicates rows and biases the model. I always assert `1:1` or the intended cardinality after a join.

### Q21. Monitoring metrics & dashboards for the whole system

I monitor **per stage and end-to-end**, on the four layers from §2:

| Scope | Metrics / dashboard |
|-------|---------------------|
| **Pipeline / orchestration** | DAG success/failure rate, **task duration & SLA misses**, retry counts, end-to-end **freshness/lag** (is today's data ready by 6am?), queue depth |
| **Data quality** | Row counts vs expected, null/duplicate rates, **schema-drift**, **PSI/KS feature drift**, key-uniqueness checks — per stage |
| **Model** | Prediction distribution, score histograms, per-model latency/throughput, **drift**, and (when labels arrive) accuracy/RMSE/F1/PR-AUC |
| **Infra / cost** | CPU/GPU/mem, error rates, endpoint latency p95/p99, **per-job cost**, autoscaling events |
| **Business / outcome** | The KPI each model serves — approval rate, fraud caught, conversion — the closed loop |

**Tooling:** **CloudWatch** (metrics, alarms, custom metrics) + **CloudWatch/Grafana dashboards**, **SageMaker Model Monitor** for drift/quality baselines, **data-quality checks** (Great Expectations / Deequ / Glue Data Quality) emitting metrics, structured logs to CloudWatch/S3, and **SNS/PagerDuty/Slack alerts**. One **top-level "system health" dashboard**: is the pipeline fresh, are all models within drift bounds, are SLAs met, what's the cost — so I can answer "is the whole thing healthy?" in one glance, then drill into the failing stage.

> **Say it like this:** "I orchestrate with a **DAG** (Airflow/Step Functions) as the control plane and an **S3 lake + feature store + model registry** as the data plane. Jobs and models communicate **through versioned data, not direct calls**, so they're loosely coupled and independently re-runnable. Dependencies are DAG edges plus **data-quality gates** — a downstream model won't start until the upstream output landed *and passed checks*. And I monitor all four layers with one top-level health dashboard, drilling into the failing stage. The join-correctness piece — validating keys and cardinality before joins — is where a lot of silent ML bugs hide, so I gate on it."

---

## 10. Rapid-fire one-liners (morning-of revision)

- **Monitor a model** → 4 layers: infra, data-health, model-behaviour, business KPI. Log every request/prediction to S3; don't trust a green latency dashboard.
- **Regression metrics** → RMSE (punishes big errors), MAE (robust, in-units); report both.
- **Classification metrics** → Precision/Recall/F1; **PR-AUC + recall@precision** on imbalanced data; accuracy lies.
- **Drift** → data (P(X)) vs concept (P(Y|X)) vs label (P(Y)); detect with **PSI/KS/chi-square** (no labels) + rolling accuracy (labels). Drift = investigate, not auto-retrain.
- **Update a model** → registry as SoT → offline gate → **shadow → canary → blue-green** → armed rollback.
- **Blue-green** → two envs, flip the router, instant reversible rollback; **canary** = % traffic ramp; **shadow** = mirror traffic, don't serve.
- **Rollback (build→prod accounts)** → previous version stays warm; roll back = **shift traffic weight** via `UpdateEndpointWeightsAndCapacities`, alarm→Lambda, no rebuild. Watch schema + stateful side-effects.
- **IaC / Terraform** → reproducible, reviewable, revertible infra; `plan` diff; remote state in S3 + DynamoDB lock; modules per account. Terraform=provision, Ansible=config.
- **Load balancing** → spread requests for availability/scale/latency; **ALB** (L7) / **NLB** (L4) / API Gateway.
- **Auto scaling** → elasticity + cost + resilience; target-tracking; SageMaker endpoint scaling, EC2 ASG, Lambda concurrency; pre-warm GPU.
- **Scheduling** → Airflow (deps/retries/backfill) > cron (single box, no deps); EventBridge = serverless cron; Glue triggers; Step Functions.
- **Scheduling bugs** → non-idempotency, missing upstream, silent partial success, timezone/DST, overlapping runs. Fix: idempotent + run-date-parametrized + data asserts.
- **Latency** → time per request (p50/p95/**p99**). **Throughput** → requests/sec. **Little's Law**: concurrency ≈ throughput × latency.
- **Heavy-read** → user-facing = **latency-first** (cache, replicas, denormalize); analytics = **throughput-first** (columnar, partitioned, parallel). Batching trades latency for throughput.
- **Orchestration** → DAG control plane + S3-lake/feature-store/registry data plane; **models talk through versioned data, not direct calls**; deps = DAG edges + data-quality gates.
- **Primary keys** → validate uniqueness + non-null; surrogate/composite keys; **point-in-time join** to prevent leakage; assert cardinality to stop fan-out joins.
- **System monitoring** → per-stage + end-to-end across 4 layers; one top-level health dashboard (freshness, drift, SLA, cost) → drill into the failing stage.

---

## 11. Do-NOT-state-as-fact (honesty guardrails)

- Don't claim a specific tool if you'd fumble a follow-up. If you've used **Airflow + Glue triggers + EventBridge** but not, say, Dagster/Prefect, say *what you've used* and reason about the rest from first principles.
- If asked about **SageMaker Model Monitor / multi-account cross-account promotion** and you've done the pattern with custom Glue/Lambda instead, say exactly that — "I've implemented the equivalent with a scheduled job computing PSI to CloudWatch; the managed version is Model Monitor." Interviewers respect the honest mapping.
- Numbers (RPS, latency SLAs) — give them as *how I'd reason to a target*, not invented production figures, unless they're real.
