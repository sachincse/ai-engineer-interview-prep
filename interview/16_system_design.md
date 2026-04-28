# Chapter 16 — System Design Interviews
## Five end-to-end designs you can walk through at a whiteboard for 45 minutes each

> System design is the round where senior offers are won or lost. Avrioc, like most senior-AI shops, will give you a half-open problem ("design an X for Y") and expect you to drive the conversation. The way to win this round is not to have memorized one architecture — it is to demonstrate the *process* of architecture: clarify, scope, sketch, deepen, defend trade-offs, anticipate failure modes.
>
> This chapter gives you five worked designs, each structured exactly the way I'd narrate it to an interviewer. Each design has the requirements interrogation, a block diagram, capacity math with concrete numbers, deep-dives on the two or three most interesting components, the failure modes you should pre-empt, and a "likely follow-ups" section with full narrative answers. Read these out loud once each. You will not memorize them; you will internalize the *shape* of a senior answer.

---

## 16.1 The universal framework — the shape of every answer

Before the five designs, the framework. Every design conversation follows this skeleton, give or take. Knowing the skeleton is half the battle because you stop wasting cycles on "what do I say next."

```
   ┌─────────────────────────────────────────────────────┐
   │  1. CLARIFY (5 min)                                  │
   │     Functional requirements                          │
   │     Non-functional: latency, scale, availability     │
   │     Constraints: budget, residency, team             │
   │     Success metric: what does "working" mean?        │
   ├─────────────────────────────────────────────────────┤
   │  2. API / SCOPE (3 min)                              │
   │     Endpoints, request/response shapes               │
   │     What's in scope vs out                           │
   ├─────────────────────────────────────────────────────┤
   │  3. HIGH-LEVEL ARCHITECTURE (10 min)                 │
   │     Draw the diagram                                 │
   │     Boxes + arrows + storage choices                 │
   │     Pick the compute primitive                       │
   ├─────────────────────────────────────────────────────┤
   │  4. DEEP-DIVE 2-3 COMPONENTS (15 min)                │
   │     Hardest one first                                │
   │     Capacity math                                    │
   │     Trade-offs vs alternatives                       │
   ├─────────────────────────────────────────────────────┤
   │  5. SCALE / RELIABILITY / COST (8 min)               │
   │     Bottlenecks under load                           │
   │     Failure modes + recovery                         │
   │     Cost back-of-envelope                            │
   ├─────────────────────────────────────────────────────┤
   │  6. EXTENSIONS + WHAT'S NEXT (3 min)                 │
   │     What I'd build next                              │
   │     Known limitations                                │
   └─────────────────────────────────────────────────────┘
```

**Five rules I follow every time:**

1. **Always draw a diagram unsolicited.** Senior engineers draw before they talk. If the interviewer doesn't share a whiteboard, ask "may I share my screen and sketch?"
2. **Always ask about scale before architecting.** 10 RPS and 10K RPS produce completely different systems. The most expensive interview mistake is building for the wrong scale.
3. **Always state trade-offs explicitly.** "I'd pick X over Y because Z, even though Y has W advantage." This is how you signal seniority.
4. **Always think about failure modes before scaling.** What breaks when the cache is down? When Snowflake is slow? When the model returns garbage? Senior engineers think about the bad day, not the good day.
5. **Always cost-it-out.** Even ballpark. "This will run us roughly $30K/month" is a senior thing to say. "I dunno, depends" is a junior thing to say.

**How to say this in an interview:**

> "Before I draw anything, let me make sure I understand the problem. I'd like to ask about scale, latency targets, and any data-residency constraints — those usually drive the biggest architectural decisions. Once I have those, I'll sketch a high-level diagram, then we can deep-dive whichever component you find most interesting. Cost back-of-envelope I'll do at the end."

---

## 16.2 Design #1 — Real-time low-latency inference (the TrueBalance loan-withdrawal predictor at scale)

This is the warm-up design. You've built a version of this, so you're not bluffing. The interviewer's testing whether you can scale your specific experience to a generalized requirement.

### Step 1 — Clarify

Imagine the question is: *"Design a real-time risk-scoring service for loan applications. Predictions feed the disbursement decision."*

Your clarifying questions and assumed answers:

- *Throughput?* "Let's say peak 1,000 RPS, sustained 200 RPS, with diurnal spike around lunch and 6pm IST."
- *Latency target?* "p99 < 500ms end-to-end including feature fetch."
- *Model type?* "XGBoost-class, ~150 features, mostly numerical and categorical."
- *Environments?* "Three: dev, staging, prod, with regulatory isolation."
- *Multi-tenant?* "Single tenant for v1, but design with future tenants in mind."
- *Data residency?* "India for now, GCC for the future Avrioc-equivalent."
- *Audit?* "Every prediction logged for 7 years, regulator-readable."
- *Failure semantics?* "If feature store is down, must we fail closed (reject all) or fall back to a default? Important for credit policy."

### Step 2 — API

```
   POST /v1/predict
   Headers: X-Request-Id, X-Tenant-Id, X-Model-Version (optional pin)
   Body:
   {
     "applicant_id": "abc123",
     "context_features": {        // request-time features
       "device_type": "android",
       "app_version": "5.2.1",
       "geo": "IN-MH"
     },
     "request_meta": {"timestamp": "...", "correlation_id": "..."}
   }

   Response (p99 < 500ms):
   {
     "score": 0.23,
     "decision": "approve | reject | review",
     "model_version": "v12.3.1",
     "features_used_count": 152,
     "shap_top_k": [...],          // top 5 contributors for XAI
     "degraded": false             // true if cache miss + fallback used
   }
```

### Step 3 — High-level architecture

```
   ┌──────────────────────────────┐
   │ Loan Origination Service     │
   │ (caller)                     │
   └────────────┬─────────────────┘
                │ HTTPS / mTLS
                ▼
   ┌──────────────────────────────┐
   │ AWS API Gateway              │
   │ - WAF (DDoS, OWASP)          │
   │ - per-tenant rate limit      │
   │ - JWT validation             │
   └────────────┬─────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────┐
   │ Inference service (FastAPI / async)          │
   │ Deployment options (see deep-dive):          │
   │  - Lambda container (chosen for spiky)       │
   │  - ECS Fargate (if traffic stays warm)       │
   │  - SageMaker real-time (if model is ML-heavy)│
   └─┬────────┬──────────────┬──────────────┬────┘
     │        │              │              │
     │        │              │              │ async fire-and-forget
     ▼        ▼              ▼              ▼
   ┌─────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │Redis│ │Snowflake   │ │Model     │ │Kafka audit   │
   │(hot │ │(feature    │ │artifact  │ │log → S3      │
   │feat-│ │ store      │ │S3 +      │ │(7-yr ret.)   │
   │ures)│ │ + history) │ │MLflow)   │ │              │
   └──┬──┘ └────────────┘ └──────────┘ └──────────────┘
      │
      │ 96% hit
      │ 4% miss → Snowflake (200ms budget)
      ▼
    response
```

### Step 4 — Deep dive on three components

#### 4a. The feature-fetch path (the bottleneck)

This is where the design lives or dies. Naive feature fetch against Snowflake is 100-300ms p99 at best, 800ms+ on cold reads. That alone blows the 500ms budget.

The pattern is **online-cache + offline-source-of-truth + parallel-fallback**.

```
   request → feature_fetch():
     ┌───────────────────────────────────┐
     │ Redis GET (timeout 30ms)          │
     └─────┬─────────────┬───────────────┘
           │ hit         │ miss
           ▼             ▼
       return     ┌─────────────────────────────┐
                  │ parallel fetch:             │
                  │   T1: Snowflake SELECT      │
                  │       (timeout 200ms)       │
                  │   T2: default-feature lookup│
                  │       (S3-cached, 5ms)      │
                  └─────┬───────────────────────┘
                        │
                        │ T1.success ? T1 : T2
                        ▼
                  return (tag degraded if T2 used)
```

Cache population: Airflow DAG that lands features in Snowflake also writes to Redis (write-through). TTL 24h. Every 6h a backfill job re-pushes the latest values for any borrower not refreshed in last 24h.

**Capacity math for the cache:**

- Number of active borrowers per day: 1M (assumption)
- Features per borrower: 150, mixed types, ~600 bytes serialized (msgpack)
- Cache size: 1M × 600B = 600MB
- With 7-day TTL window: ~4GB
- Redis ElastiCache `cache.r7g.large` (13.5GB RAM) covers this 3× → fits with headroom on a single shard. Cluster mode disabled. Replica for HA.

#### 4b. The compute primitive (Lambda vs ECS vs SageMaker)

The interview gold is here.

```
                Lambda (container)    ECS Fargate           SageMaker RT
   Cold start   200ms-2s              ~30s but rare         ~30s but rare
   Auto-scale   Per-request           Task-count            Endpoint instances
   Idle cost    $0 (or small w/ PC)   Always running        Always running
   Max latency  Sub-second OK         Sub-second OK         Sub-second OK
   GPU support  No                    Yes (with G-instance) Yes (richest)
   ML features  None                  None                  Model registry,
                                                            Monitor, MCE, MME
   Cost @ 200   ~$700/mo + PC         ~$1500/mo (3 tasks    ~$2000+/mo
   RPS sustain                        always warm)          (ml.m5 endpoint)
```

For the spiky-traffic, small-model case described, Lambda with provisioned-concurrency wins on cost and complexity. For sustained 5K+ RPS with a steady floor, ECS or SageMaker becomes more economical because Lambda's per-request cost dominates.

**The interviewer-impressing line:** "I'd pick Lambda for this profile because the traffic is spiky and the model fits comfortably in a 3GB container. If sustained throughput rises above ~1,500 RPS, I'd reconsider — Fargate becomes more economical past that break-even because Lambda's per-invocation cost dominates."

#### 4c. Audit, observability, and the regulator

For regulated lending, every prediction is durable evidence. The audit log is non-negotiable.

```
   Inference path:
     - Sync: predict → return response
     - Async: write {request, features, score, model_version, ts} → Kafka

   Kafka → S3 (Parquet, partitioned by date, KMS-encrypted)
        → 7-year retention (S3 Glacier after 90d)
        → Object lock (immutable, regulatory)

   Observability:
     - Prometheus metrics: RED (rate, errors, duration) per endpoint
     - CloudWatch metrics: score distribution, degraded-response rate
     - Datadog drift dashboard: PSI per feature, daily
     - PagerDuty alerts: p99 SLO breach, error rate >1%, degraded >2%
```

### Step 5 — Capacity, scale, failure modes

**Steady state (200 RPS, all warm):**
- 200 RPS × 80ms inference = within 1 Lambda's 1000-concurrent budget easily
- Provisioned concurrency 50 is sized for peak 1,000 RPS (each PC instance handles ~20 RPS)

**Failure modes:**

| Failure | Detection | Mitigation |
|---|---|---|
| Redis cluster down | Connection error | Fall through to Snowflake |
| Snowflake slow / timeout | 200ms timeout | Default-feature fallback (degraded=true) |
| Model artifact corrupt | Lambda init fails | Auto-rollback, page on-call |
| Score drift (model bad) | Daily KS test on score dist | Manual review, possible model rollback |
| Feature drift (input bad) | Daily PSI per feature | Slack alert, retraining queue |
| Lambda concurrency cap | CloudWatch throttle metric | Increase reserved concurrency |
| Audit Kafka backed up | Kafka lag metric > 60s | Page; predictions still served (async) |

**Cost back-of-envelope (production tier, 200 RPS sustained, peak 1000):**
- Lambda: 200 RPS × 86400s × 30d = 0.5B requests; at $0.20/M + 80ms × $0.0000167/GB-s × 3GB = ~$1.5K/month base + $400 for provisioned concurrency
- Redis: r7g.large × 2 (HA) ≈ $400/month
- Kafka MSK + S3 audit: ~$600/month
- Snowflake compute (background backfill jobs): $300/month
- CloudWatch + Datadog: ~$300/month
- **Total: ~$3K/month per environment, ~$9K across three envs**

### Step 6 — Extensions

- Multi-region active-active (DR): Aurora Global + Snowflake replicas + Route53 latency routing
- Multi-tenant: tenant prefix on cache keys, per-tenant model versioning, per-tenant audit segregation
- Online learning: hourly partial-fit on recent labels (we deferred this; needs careful eval)

### Likely interviewer follow-ups

*Q: "What if the interviewer says traffic is 10,000 RPS sustained?"*
> Lambda breaks down on cost at that scale. I'd switch to ECS Fargate or EKS with FastAPI pods, HPA on RPS. ~30 pods, each handling ~350 RPS. Redis cluster mode enabled across 3 shards. Snowflake replaced by DynamoDB for online features (Snowflake doesn't keep up with 10K RPS reads). Audit goes to Kinesis Data Streams instead of MSK for the higher ingest rate.

*Q: "How do you do canary deploys on Lambda?"*
> CodeDeploy traffic-shifting on Lambda aliases — the alias points to v(n) and we shift 5% / 50% / 100% with bake time. CloudWatch alarms gate each shift. Auto-rollback on alarm.

*Q: "What's your strategy for model retraining cadence?"*
> Two triggers. Time-based: weekly retrain on rolling 90-day data. Drift-based: PSI > 0.2 on any top-10 feature, OR predicted-score KS-divergence > threshold against baseline → fires retraining DAG. Result evaluated on held-out validation before promotion.

*Q: "The model's predictions degrade silently. How do you know?"*
> Two layers. Proxy metric daily — predicted-score distribution KS test against last week. Real metric weekly — once labels (loan repayment outcome) arrive 30+ days later, compute AUC on the labelled cohort. Alert on either drift.

*Q: "How would you A/B test two models?"*
> Hash-based traffic split — `applicant_id % 100 < 5` goes to variant B. Both predictions logged with `variant` tag. After 2 weeks, compare AUC on the labelled cohort, plus business metrics — approval rate, default rate, dollar-volume. Promote on statistical significance with effect-size threshold.

*Q: "Cost spikes. What's the first thing you check?"*
> Per-tenant request volume. Lambda invocation count. PC over-provisioning. The most common cause in my experience is provisioned concurrency set too high after a load test and never reduced — checks run on it.

*Q: "How do you handle the cold-start storm after a deploy?"*
> Provisioned concurrency keeps 50 PC instances warm. Deploy is via CodeDeploy linear-traffic-shift; each shift waits for the new alias's PC instances to be initialized before it gets traffic. So no thundering-herd cold-start.

*Q: "How do you do explainability?"*
> SHAP values computed at inference time using TreeSHAP for XGBoost — costs ~3-5ms per inference. Top-5 SHAP contributors returned in the response. For the regulator-facing audit, every decision's full SHAP vector is logged.

---

## 16.3 Design #2 — RAG system at scale (100M docs, 1000 QPS)

The bigger, harder version of the ResMed RAG you built. The interviewer wants to see whether you can scale a familiar pattern by 4 orders of magnitude.

### Step 1 — Clarify

Question: *"Design a RAG system over 100M internal documents with 1000 QPS query rate. Multilingual (English + Arabic). Latency budget under 3 seconds end-to-end."*

Clarifications:
- *Update frequency?* Documents updated daily, not streaming. Daily reindex acceptable.
- *Languages?* English + Arabic, single index.
- *Output style?* Conversational, with citations.
- *Eval criteria?* Faithfulness + answer relevance + thumb-down rate.
- *Data residency?* Yes, UAE region (me-central-1 Bahrain or self-hosted UAE).
- *Multi-tenant?* Yes, with RBAC at chunk level (org_id, doc_access_level).

### Step 2 — High-level architecture

```
                  ┌────────────────────────────┐
                  │  Client (Web / Mobile)     │
                  └─────────────┬──────────────┘
                                │ SSE
                                ▼
                  ┌────────────────────────────┐
                  │  API Gateway + Auth         │
                  └─────────────┬──────────────┘
                                ▼
   ┌────────────────────────────────────────────────────────┐
   │  FastAPI Orchestrator (LangGraph)                       │
   │   - rate limit per tenant                               │
   │   - LangGraph: rewrite → retrieve → rerank → generate   │
   └──┬────────┬───────────┬──────────┬──────────┬──────────┘
      │        │           │          │          │
      ▼        ▼           ▼          ▼          ▼
   ┌──────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────────┐
   │Query │ │Hybrid   │ │Cross-   │ │vLLM     │ │Langfuse    │
   │rewrite│ │search   │ │encoder │ │(70B    │ │tracing      │
   │(Haiku)│ │(BM25 +  │ │rerank  │ │AWQ)    │ │             │
   │       │ │  dense) │ │bge-rrk │ │TP=4 on │ │             │
   │       │ │         │ │ v2-m3  │ │A100s   │ │             │
   └───────┘ └────┬────┘ └────────┘ └────────┘ └────────────┘
                  │
                  ▼
            ┌──────────────────────┐
            │ Qdrant cluster        │
            │  - 100M dense vectors │
            │  - PQ-compressed      │
            │  - sparse (BM25 head) │
            │  - 6-9 shards, 3 reps │
            └──────────┬───────────┘
                       │ daily reindex
                       ▼
            ┌──────────────────────┐
            │ Ingestion pipeline    │
            │  - Ray Data           │
            │  - parsing + chunking │
            │  - BGE-M3 embedding   │
            │    on H100 batch      │
            │  - Qdrant upsert      │
            └──────────┬───────────┘
                       ▼
            ┌──────────────────────┐
            │ Source storage        │
            │  - S3 (Parquet, raw)  │
            │  - Postgres metadata  │
            └──────────────────────┘

         Eval lane (offline):
         ┌──────────────────────────────────────┐
         │ Golden set (500 pairs) → RAGAS nightly│
         │ + LLM-as-judge → metrics dashboard    │
         └──────────────────────────────────────┘
```

### Step 3 — Deep dives

#### 3a. Index sizing — the most important math in the design

100M chunks × 1024 dimensions × 4 bytes (FP32) = **400 GB raw vectors**.

That doesn't fit in a single node's RAM cheaply. Options:

```
   Option A — HNSW raw FP32
     Memory: 400GB + HNSW graph (~30% overhead) = ~520GB
     Hardware: distribute across 6 r6i.16xlarge (512GB RAM each)
     Cost: ~$25K/month
     Latency: ~50ms p99
     Recall: ~99% @ M=32, ef=128

   Option B — HNSW with FP16 vectors
     Memory: 200GB + 30% = ~260GB
     Hardware: 4 r6i.16xlarge
     Cost: ~$17K/month
     Latency: ~50ms
     Recall: ~98.5%

   Option C — IVF-PQ (product quantization, 8x compression)
     Memory: 50GB + small graph = ~60GB
     Hardware: 2 r6i.4xlarge or even single
     Cost: ~$2K/month
     Latency: ~80ms (PQ adds compute)
     Recall: ~94% (acceptable with reranking)

   Option D — Binary quantization + reranking
     Memory: 12GB binary vectors + 400GB on disk for full-precision rerank
     Hardware: 1 r6i.4xlarge for binary, S3-backed for full
     Cost: ~$1K/month
     Latency: ~100ms (two-pass)
     Recall: ~96% with rerank
```

**The senior choice:** Option C (IVF-PQ) for the primary index, paired with cross-encoder rerank to recover quality. If recall@5 measured on the golden set drops below 0.85, escalate to Option B.

**How to say this in an interview:**

> "100 million 1024-dimensional vectors at FP32 is 400 gigabytes raw, which is too expensive to keep in RAM. I'd use IVF-PQ with 8x compression — that brings it down to roughly 50 gigabytes — and pair it with a cross-encoder rerank to recover the recall lost from quantization. If golden-set recall drops below 0.85, I'd consider FP16 instead, which costs more but preserves more signal. I'd benchmark before committing."

#### 3b. Latency budget — show the math

```
   Total budget: 3000ms
     - Network round trips:                  100ms
     - Auth + routing:                        20ms
     - Query rewrite (Haiku):                300ms
     - Retrieval (Qdrant IVF-PQ, top-100):    80ms
     - Cross-encoder rerank (top-100→top-5): 200ms
     - Context assembly + prompt build:       30ms
     - LLM TTFT (Llama-70B AWQ on TP=4):     500ms
     - LLM stream (200 tokens @ 30ms ITL):  6000ms (off the critical path; user sees first token at 1230ms)
     - Citation post-process:                 50ms

   TTFT (time-to-first-token) for the user:  ~1230ms
   Full answer (200 tokens):                 ~7300ms
```

The user sees streaming, so what matters is TTFT, not full-answer time. The TTFT budget of ~1.2s is what we promise.

#### 3c. Ingestion pipeline at 100M scale

Daily reindex of 100M docs is non-trivial.

```
   Step 1 — Diff detection
     SQL query against Postgres metadata: WHERE updated_at > yesterday
     Typically 0.5-2% of docs change daily = 500K-2M docs
     Only re-process the diff, not the full corpus

   Step 2 — Parsing + chunking
     Ray Data job, 50 workers
     Reads from S3 Parquet
     Parses (PDF / HTML / DOCX) via unstructured.io
     Section-aware chunking (semantic boundaries via header detection)
     Chunk size: 512 tokens, overlap 50

   Step 3 — Embedding
     BGE-M3 on H100 batch
     Batch size 256
     ~4000 chunks/sec per H100
     2M chunks / 4000 = 500 sec ≈ 9 min on a single H100
     We use 4 H100s in parallel to compress this further

   Step 4 — Index update
     Qdrant upsert in batches of 1000
     Mark new chunks as primary; old chunks tombstoned
     Background compaction nightly
```

Total ingestion time for 2M-doc daily diff: ~30-45 minutes. Comfortably fits a maintenance window.

### Step 4 — Capacity for query path

1000 QPS query load:

- vLLM cluster sized for 1000 QPS LLM:
  - Llama-70B AWQ on TP=4 (4 × A100-80GB) → ~30 QPS per replica at 2KB context
  - Need 1000 / 30 = 33 replicas → 33 × 4 = 132 A100s
  - **That's $200K-300K/month at on-demand pricing.** The interviewer will probe this.
  - Mitigations: prompt caching for repeated system prompts (10-30% savings), KV-cache quantization to FP8 (2× concurrent capacity), routing 90% of queries to a cheaper Sonnet-class hosted model and only escalating to 70B for hard queries.
- Qdrant: 6 shards × 3 replicas → 18 nodes. Easy at 1000 QPS (each shard handles ~150 QPS).
- Cross-encoder rerank: bge-reranker-v2-m3 on g5.xlarge → ~50 QPS per replica. Need 20 replicas.

### Step 5 — Failure modes

| Failure | Detection | Mitigation |
|---|---|---|
| Qdrant shard down | Heartbeat | Replica takes over, alert |
| LLM cluster overloaded | Queue depth metric | KEDA scales out; SLA-tier rate-limiting |
| Embedding pipeline failure (daily) | Airflow DAG failure | Yesterday's index served; retry next day |
| Hallucinated answer | Faithfulness eval (LLM-as-judge online sampling) | Trace flagged for review |
| RBAC leak (cross-tenant doc returned) | Audit query | Pre-filter chunks by access metadata before retrieval |
| Tail latency (>10s on 1% of queries) | p99.9 metric | Chunked prefill, max-tokens cap, separate replica pool for long context |

### Step 6 — Eval lane (often-skipped, senior-marker)

```
   ┌─────────────────────────────────────────────────────┐
   │ Offline eval (nightly)                              │
   │  - 500-pair golden set                              │
   │  - RAGAS: faithfulness, answer relevance,           │
   │    context precision, context recall                │
   │  - LLM-as-judge: 1-5 helpfulness                    │
   │  - Tracked in MLflow per-release                    │
   ├─────────────────────────────────────────────────────┤
   │ Online eval (continuous)                            │
   │  - 1% of traces sampled, LLM-as-judge faithfulness  │
   │  - User feedback (thumb up/down)                    │
   │  - Thumb-down rate alarm                            │
   ├─────────────────────────────────────────────────────┤
   │ Drift                                                │
   │  - Query distribution embeddings → UMAP weekly      │
   │  - Alert on novel-query cluster appearance          │
   └─────────────────────────────────────────────────────┘
```

### Cost summary

- Qdrant cluster: ~$5-10K/month (depending on quantization)
- vLLM 70B serving: ~$50-200K/month (range depending on QPS and quant)
- Reranker + embedder: ~$5-10K/month
- Storage + ops: ~$3K/month
- **Range: $63K-$220K/month** depending on quality/cost trade-offs.

The interviewer will ask "how do you cut this in half?" — answers: smaller LLM (Llama-3.1-8B for easy queries, route only hard ones to 70B), more aggressive caching, switch to managed Bedrock, accept lower SLA.

### Likely follow-ups

*Q: "How do you handle Arabic + English in one index?"*
> BGE-M3 is multilingual; it embeds Arabic and English into the same space. No language-segmenting needed. At query time, the language detection happens implicitly — the query embedding lands closer to docs in the same language for queries with strong language cues, and cross-lingually for translatable concepts. We benchmark recall on a mixed-language eval set per release.

*Q: "How do you do RBAC at chunk level?"*
> Every chunk has an `access_level` and `org_ids` payload field in Qdrant. The retrieval call includes a filter — `must={org_id: $user.org_id, access_level: <= $user.clearance}`. Filter is applied pre-retrieval (Qdrant supports filtered HNSW efficiently). Audit log retrieval queries for compliance review.

*Q: "Where's your bottleneck under 1000 QPS?"*
> The LLM. Everything else scales linearly with replicas; the LLM is GPU-bound and expensive. The single biggest lever is routing — most queries are easy and go to a smaller model; only hard queries (long context, complex reasoning) route to the 70B. That single change can cut LLM cost by 5-10x.

*Q: "How do you keep the index fresh for breaking news?"*
> The daily-reindex pattern doesn't fit time-sensitive content. For that, we'd add a streaming-update lane — Kafka topic for new docs, immediate embedding + upsert. Two-tier index: large daily for legacy, small streaming for recent. Query merges results from both.

*Q: "What if the user asks a question outside the corpus?"*
> Two layers. The retrieval scores tell us — if max similarity score across top-K is below threshold, we say "I don't have information on that" rather than hallucinate. And the system prompt explicitly instructs the model to answer ONLY from context, with a fallback "insufficient context" output.

*Q: "Cost is too high. What do you cut first?"*
> The LLM. In order: (1) route easy queries to smaller model, (2) AWQ quantize the 70B, (3) FP8 KV-cache to double effective capacity, (4) prompt-cache the system prompt. Together these typically cut LLM cost by 60-70% with minimal quality loss.

*Q: "How would you evaluate quality after a model upgrade (Llama-3 → Llama-4)?"*
> Shadow mode first — run both models on a 5% traffic sample, compare RAGAS metrics + LLM-as-judge on the same prompts. If new beats old on faithfulness and answer relevance with no regression on latency, A/B 5% → 50% → 100% with metric gates.

*Q: "What about agentic RAG — multi-hop?"*
> LangGraph node graph: classify → retrieve → check-coverage → if-incomplete → reformulate → retrieve again → synthesize. Each loop logged to Langfuse. Cap iterations at 5 to prevent runaway. Use cases: questions that span multiple documents — "compare the two protocols" type queries.

---

## 16.4 Design #3 — Multi-model LLM serving platform (10K req/s, 5 models, mixed context lengths)

This is the LLMOps platform design. Avrioc serves multiple products (MyWhoosh, Comera, Labaiik, Hyre); a unified LLM platform makes operational sense.

### Step 1 — Clarify

- *Models?* 5 — assume Llama-3.3-70B (heavy reasoning), Llama-3.1-8B (easy chat), Whisper-large-v3 (transcription), an embedding model, a small classifier.
- *QPS?* 10K req/s aggregate, varying per model.
- *Context lengths?* Mixed — 2KB chat to 128KB long-doc.
- *Multi-tenant?* Yes, 4-5 product tenants.
- *Latency targets?* TTFT <500ms for chat, batch throughput primary for embeddings.

### Step 2 — Architecture

```
   ┌─────────────────────────────┐
   │ Product apps (MyWhoosh,      │
   │ Comera, Labaiik, Hyre)       │
   └────────────────┬────────────┘
                    │
                    ▼
   ┌─────────────────────────────────────────────┐
   │ LiteLLM Gateway (Python, async)              │
   │  - Model router (routes by `model` field)    │
   │  - Per-tenant rate limit + quota             │
   │  - Prompt cache (Redis-backed)               │
   │  - Request → response logging                │
   └──┬──────────┬──────────┬──────────┬─────────┘
      │          │          │          │
      ▼          ▼          ▼          ▼
   ┌──────┐  ┌──────┐  ┌────────┐  ┌──────────┐
   │vLLM  │  │vLLM  │  │Whisper │  │Embedding │
   │70B   │  │8B    │  │service │  │service   │
   │TP=4  │  │TP=2  │  │GPU pool│  │CPU pool  │
   │A100s │  │L40Ss │  │        │  │          │
   └──┬───┘  └──┬───┘  └───┬────┘  └────┬─────┘
      │         │          │            │
   ┌──┴─────────┴──────────┴────────────┴───────┐
   │ KEDA autoscaler (queue-depth aware)         │
   │ Ray Serve / KServe orchestration            │
   └─────────────────────────────────────────────┘

      ┌──────────────────────────────────┐
      │ Cross-cutting:                    │
      │  - Langfuse traces                │
      │  - Prometheus + Grafana           │
      │  - Guardrails (input/output)      │
      │  - PII redaction                  │
      └──────────────────────────────────┘
```

### Step 3 — Deep dives

#### 3a. Routing — the senior decision

Every request hits the gateway. Two routing decisions:

```
   1. Model routing (explicit by `model` field):
      - "llama-70b" → 70B pool
      - "llama-8b"  → 8B pool
      - "auto"     → router LLM decides based on query

   2. Replica routing (load balancer):
      - Round-robin? No, queue-depth-aware
      - Pick replica with shortest queue
      - Reason: long-context requests in vLLM occupy KV cache
        slots and slow other requests on the same replica.
        Queue-depth routing avoids piling on a busy replica.
```

**KV-cache contention is the killer in LLM serving.** A single 128K-context request pins ~10GB of KV-cache on a 70B model, which crowds out 50 short-context requests. The architecture must handle this.

The pattern: **dedicated long-context pool**.

```
   Llama-70B serving:
     Pool A — short context (≤4KB), 6 replicas, max 4096 tokens
     Pool B — long context (≤128KB), 2 replicas, max 131072 tokens
     Router sends request to A or B based on input + max_tokens

   Why split? Long-context requests on shared pool cause head-of-line blocking.
   Dedicated pool isolates the variance.
```

#### 3b. Prompt caching — cost-saving feature

vLLM has `--enable-prefix-caching`. Repeated prefixes (system prompts, RAG instructions) are KV-cached.

```
   Request 1: <system_prompt_X> + <user_query_1>
     - System prompt encoded, KV cached under hash(system_prompt_X)

   Request 2: <system_prompt_X> + <user_query_2>
     - System prompt cache HIT
     - Only encode user_query_2
     - 10-30% latency reduction on warm cache
     - 30-50% cost reduction (fewer prefill tokens billed)
```

For a multi-tenant platform with shared system prompts, this is significant.

#### 3c. GPU allocation strategy

```
   Llama-70B AWQ:
     - 70B params × 2 bytes (FP16) = 140GB → too big for 1 A100
     - With AWQ INT4: ~35GB → fits 1 A100-80GB but tight with KV cache
     - TP=2 on 2 A100s comfortable
     - TP=4 on 4 A100s for higher concurrency

   Llama-8B AWQ:
     - 8B × 2 = 16GB → fits L40S (48GB) easily, room for KV cache
     - TP=1 sufficient
     - Replicate horizontally for QPS

   Whisper-large-v3:
     - 1.5B params, ~3GB
     - Single L4 instance per replica
     - Audio chunked, batched

   Embeddings:
     - BGE-M3 (~600M)
     - CPU c6i.4xlarge handles ~200 QPS with batch 32
     - Or g5.xlarge for 10x throughput
```

### Step 4 — Cost back-of-envelope

```
   Llama-70B serving (6 short + 2 long pool replicas, TP=4):
     8 replicas × 4 A100 = 32 A100s
     A100-80GB on-demand: ~$3.5/hr × 32 × 730 hr/mo = ~$82K/mo
     With 1-yr reserved: ~$50K/mo

   Llama-8B serving (10 replicas, L40S):
     10 × $1.5/hr × 730 = ~$11K/mo

   Whisper:
     5 × L4 ($0.8/hr) × 730 = ~$3K/mo

   Embeddings:
     4 × g5.xlarge ($1/hr) × 730 = ~$3K/mo

   Gateway + ops + tracing:
     ~$2K/mo

   Total: ~$70-100K/mo
   Per 1M tokens (mix): ~$0.30-0.50 self-hosted
   Compare to Anthropic Sonnet: ~$3-15 per 1M tokens
   → Self-hosting wins at this volume.
```

### Likely follow-ups

*Q: "How do you handle a noisy tenant?"*
> Per-tenant rate limit at gateway (token-bucket). Per-tenant quota (daily token cap). Per-tenant priority queue inside the gateway — premium tenants jump ahead. If a tenant exceeds quota, return 429 with retry-after. Track per-tenant cost; alert if anomalous spike.

*Q: "What if a tenant needs custom fine-tuning?"*
> Multi-LoRA serving. Train a LoRA adapter (r=16, ~30MB) per tenant, register in MLflow, hot-load into vLLM via `--enable-lora --max-loras 16`. Per-request adapter dispatch by `model` field. One base model in VRAM, many adapters swapped per request. SGMV kernels make this batch-friendly.

*Q: "Streaming token-by-token at 10K QPS — feasible?"*
> Yes, FastAPI + uvloop + Server-Sent Events handles thousands of concurrent streams per pod. The bottleneck isn't streaming — it's the LLM throughput. SSE itself is lightweight HTTP-keep-alive.

*Q: "How do you do canary on a model upgrade?"*
> Shadow first — 5% mirror traffic to new model, compare outputs offline (cost: 2x for 5%). Then A/B on real traffic, gated by faithfulness, latency, thumb-down rate. 5% → 50% → 100% with 24h soak at each step.

*Q: "What's the failover plan if vLLM cluster is down?"*
> Three-tier fallback. Primary: self-hosted vLLM. Secondary: Bedrock (managed Anthropic / Llama). Tertiary: cached responses for repeat queries. Gateway routes by health-check. Critical: do NOT fail-open — return a graceful degradation message rather than a hallucinated answer from a misbehaving model.

*Q: "How do you route 'auto' queries to the right model?"*
> A small classifier — could be a fine-tuned distilbert or even a heuristic on input length + special keywords. Trained on a labeled set of (query, ideal_model). Cost: 5ms classifier latency + sometimes-wrong routing. Validation: A/B against always-70B; measure quality and cost.

---

## 16.5 Design #4 — AI coaching for MyWhoosh (Avrioc's flagship product)

This is the killer story. MyWhoosh is Avrioc's UCI-licensed virtual indoor cycling platform. Riders pair smart trainers (Wahoo, Kickr, Tacx) and ride virtual courses, alongside thousands of other riders. The product collects real-time telemetry — power output (watts), cadence, heart rate, speed, pedal balance — at 1Hz or higher. **An AI coaching layer is the obvious next product surface.** Pitch it confidently.

### Step 1 — Clarify

Question (you bring this up unprompted in the interview, and the interviewer will love that you've researched the product): *"Imagine I'm pitching the AI coaching feature for MyWhoosh. Walk me through the design."*

- *Users?* Imagine 100K active riders, of which 10K are concurrent peak (e.g., Tour de France launch event).
- *Data?* Per-rider per-second telemetry: power, cadence, HR, speed, pedal balance, gradient, virtual position.
- *Coaching scope?* Real-time during ride (form/effort tips), post-ride (workout summary + insights), training-plan generation (long-horizon).
- *Personalization?* Yes, per-rider models or per-rider features over a global model.
- *Latency?* Real-time tips: <2s from event to tip on screen. Post-ride summary: minutes acceptable. Training plan: hours acceptable.

### Step 2 — High-level architecture

```
   ┌──────────────────────────────────────┐
   │  Smart trainer + MyWhoosh client     │
   │  (sends telemetry @ 1Hz)             │
   └───────────────┬──────────────────────┘
                   │ WebSocket / MQTT
                   ▼
   ┌──────────────────────────────────────┐
   │  Telemetry ingestion (Kinesis)       │
   │  - 100K riders × 1 evt/s = 100K evt/s│
   └───────┬──────────────────────────────┘
           │
   ┌───────┴────────────────────────────────────────────┐
   │ Stream processor (Flink / Kinesis Data Analytics) │
   │  - per-rider rolling windows (10s, 60s, 5min)      │
   │  - feature aggregation                             │
   │  - anomaly detection (cheat detection)             │
   └───┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────────┐
   │TimeSeries│ │Feature  │ │Real-time│ │Cheat-detect  │
   │ DB        │ │store    │ │coaching │ │alerts        │
   │(Timestream│ │(online: │ │model    │ │(post to mod) │
   │ Influx)   │ │ Redis;  │ │(XGBoost)│ │              │
   │           │ │ offline:│ │         │ │              │
   │           │ │ S3+Snow)│ │         │ │              │
   └────┬──────┘ └────┬────┘ └────┬────┘ └──────────────┘
        │             │           │
        │             │           ▼
        │             │     ┌──────────────────────┐
        │             │     │ Tip generator (LLM)  │
        │             │     │ - input: model output│
        │             │     │   + recent telemetry │
        │             │     │ - output: short tip  │
        │             │     │   ("ease cadence")   │
        │             │     └────────┬─────────────┘
        │             │              │
        │             │              ▼
        │             │     ┌──────────────────────┐
        │             │     │ Push to client (WS)  │
        │             │     └──────────────────────┘
        │             │
        ▼             ▼
   ┌──────────────────────────────────────────────┐
   │ Post-ride pipeline (batch, hourly)            │
   │  - workout summary                            │
   │  - FTP estimation                             │
   │  - training-zone classification               │
   │  - LLM personalization (training plan)        │
   └──────────────────────────────────────────────┘
```

### Step 3 — Deep dives

#### 3a. Models needed (and why)

**1. Real-time effort/RPE estimator.** Predicts the rider's perceived effort (1-10 scale) from objective signals (power, HR, cadence, recent history). Trained on labeled rides where riders self-reported RPE. XGBoost on engineered features (10s/60s/5min rolling stats). Used to drive tip generation: "RPE detected at 8/10 — reduce gear by one for the next minute."

**2. FTP (Functional Threshold Power) estimator.** FTP is the cornerstone of cycling training — the power you can sustain for one hour. Most riders don't do an FTP test; we estimate from recent ride data. Time-series regression model. Updated weekly per rider.

**3. Training-zone classifier.** Given current power + FTP, classify which training zone (Z1 active recovery through Z7 anaerobic). Used in the UI and in coaching tips.

**4. Anomaly / cheat detection.** MyWhoosh awards UCI-sanctioned race results — cheating is a real concern. Riders can fake power data via firmware mods. The model: trained on (normal_power_curve, suspicious_power_curve) labeled data. Flags rides with non-physiological power profiles (e.g., constant 600W for 2h with HR at 110 BPM is implausible). Used both real-time (warn rider, flag for review) and post-race (audit before podium).

**5. LLM personalization layer.** Takes the structured outputs of the above models and generates natural-language tips, ride summaries, and training plans. This is where Avrioc's LLM expertise plays. Prompt engineering with rider history, goals, recent performance. Fine-tuning on coaching-style language for tone.

#### 3b. Real-time vs batch decisions

```
   Online (during ride):
     - Telemetry → Kinesis → Flink → Feature store (Redis)
     - Real-time inference: RPE model, zone classifier, anomaly check
     - Tip generation: small LLM (Llama-8B or Haiku) for latency
     - Push to client at intervals (every 30-60s, or on event)

   Batch (post-ride / overnight):
     - Workout summary aggregation
     - FTP estimation update
     - Personalized training plan (full Sonnet-class LLM, takes seconds)
     - Insights deck for the rider

   Training (continuous):
     - Daily retraining of RPE model on yesterday's labeled rides
     - Weekly retraining of cheat-detection on new flagged samples
     - Per-rider model fine-tuning (LoRA on cumulative ride history)
       for top users (paid tier)
```

#### 3c. Personalization — global vs per-rider

The classic question. My pick:

- **Global model with rich rider features** for v1 — much faster to ship, no cold-start, works for every rider including new ones.
- **Per-rider LoRA adapters** for top-tier (paid) riders in v2 — meaningful uplift on personalized prediction once the rider has 50+ rides of data. LoRA is small (~30MB), fits in object store.
- **Hybrid serving:** vLLM with `--enable-lora`. Free-tier rider → base model. Premium-tier rider → adapter swapped in. Multi-LoRA SGMV makes this batched.

### Step 4 — Capacity math

**Telemetry ingestion:**
- 100K riders × 1Hz = 100K events/sec peak
- Each event ~200 bytes → 20 MB/s
- Kinesis: 1 shard handles 1 MB/s in, 2 MB/s out → 20-30 shards
- Cost: ~$1K/month

**Real-time inference (online coaching):**
- Tips fire every 30-60s per active rider
- 10K concurrent × 1 tip/min = 167 tips/sec
- RPE model (XGBoost) on CPU: 5K QPS per c6i.4xlarge → 1 instance handles all
- Tip-generation LLM (Llama-8B) on L40S: ~50 QPS per replica → 4 replicas
- Cost: ~$10K/month for the LLM tier

**Post-ride batch:**
- 100K rides/day × 30s of compute each = 3000 GPU-seconds = 50 GPU-minutes
- Easily handled by 2-4 L4 instances on a queue

### Step 5 — Why this is the killer pitch in interview

When the interviewer asks "what would you build first at Avrioc?", this is the answer.

**How to say it:**

> "If I were starting at Avrioc, I'd want to look at MyWhoosh data first. Indoor cycling is a perfect AI playground — high-frequency telemetry, ground-truth outcomes (race finishes, FTP improvements), engaged user base. The first feature I'd ship is a real-time coaching layer: lightweight effort and zone classifiers, an LLM tip generator that turns model outputs into natural language, and a post-ride summary with personalized training plan. The architecture is Kinesis ingestion, Flink for windowed features, online feature store on Redis, real-time XGBoost for effort, Llama-8B for tip generation. Cheat detection is a second model that flags non-physiological power curves before podium audits — this matters because MyWhoosh is UCI-sanctioned. Personalization v1 is a global model with rich rider features; v2 is per-rider LoRA for premium tier, served with multi-LoRA vLLM. The whole thing fits in $30-50K/month and would meaningfully drive engagement and the premium tier."

That answer is your differentiator in this interview.

### Likely follow-ups

*Q: "How do you handle riders who only ride twice a month?"*
> Cold-start problem. Use the global model with population priors; flag them as "low-data" so the UI doesn't promise too much. As they accumulate data, gradually personalize — e.g., simple Bayesian update of FTP estimate, switch to LoRA after 50 rides.

*Q: "Cheat detection — how do you avoid false positives that anger legitimate strong riders?"*
> Two-tier. Tier-1: model flags suspicious rides automatically; rider gets a warning, not a ban. Tier-2: human reviewer (the moderation team) confirms before any podium adjustment. Track precision/recall on a labeled set of confirmed cheats; tune threshold to optimize for precision (avoid FPs that frustrate users).

*Q: "How does the real-time tip generation not feel spammy?"*
> Rate limit per rider (max 1 tip per 60s). Tip relevance threshold — only push if model confidence > X. User control — riders can opt down ("only safety tips" / "only effort tips"). Track engagement signal — tips that riders dismiss within 2 seconds get suppressed for that rider.

*Q: "How do you test this without bothering real riders?"*
> Offline replay. We have historical telemetry for millions of rides; run the new model against those rides, compare its tips to the labeled "what-good-coaching-would-say" set. A/B on synthetic shadow traffic. Then small-percentage live A/B with metrics: tip acceptance rate, post-ride FTP delta, retention.

---

## 16.6 Design #5 — Recommender system (Labaiik grocery delivery)

Avrioc's Labaiik is a UAE grocery delivery platform. Recommendation is a core feature: what to suggest on the home screen, in cart-add cross-sells, in re-engagement notifications.

### Step 1 — Clarify

- *Catalog size?* ~50K SKUs.
- *Users?* ~1M registered, 200K MAU.
- *Throughput?* 5K recommendations/sec at peak.
- *Latency?* <200ms for in-app feeds.
- *Data?* Order history, view/click logs, search queries, time-of-day, location.

### Step 2 — Two-stage recommender

```
   Request: user_id, context (page, device, time)
        │
        ▼
   ┌────────────────────────────────────────────┐
   │ Stage 1: Candidate generation (retrieval)  │
   │  - Multiple recallers run in parallel:     │
   │    a. Collaborative filtering (ALS, daily) │
   │    b. Content-based (item embeddings)      │
   │    c. Recency / trend                      │
   │    d. User-history "frequent buys"         │
   │  - Each returns top-200 candidates         │
   │  - Union → ~500 candidates                 │
   └─────────────┬──────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────────────────────┐
   │ Stage 2: Ranking (deep model)              │
   │  - DLRM-style model                        │
   │  - User features × Item features × Context │
   │  - Two-tower or sparse embeddings + DNN    │
   │  - Predicts P(click), P(add-to-cart)       │
   │  - Score each of 500 candidates            │
   └─────────────┬──────────────────────────────┘
                 │
                 ▼
   ┌────────────────────────────────────────────┐
   │ Re-ranking (business logic)                │
   │  - Diversity (don't show 8 yogurts)         │
   │  - Inventory filter (in-stock only)        │
   │  - Promotion boost                         │
   │  - User-preference filters (vegetarian)    │
   └─────────────┬──────────────────────────────┘
                 │
                 ▼
   Return top-20 to client
```

### Step 3 — Architecture

```
   ┌──────────────────────────────────────────┐
   │ Mobile app                                │
   └─────────────┬────────────────────────────┘
                 │
                 ▼
   ┌──────────────────────────────────────────┐
   │ Recommender API (FastAPI on EKS)          │
   │  - HPA, ~30 pods                          │
   └─────┬───────────┬───────────┬────────────┘
         │           │           │
         ▼           ▼           ▼
   ┌─────────┐ ┌──────────┐ ┌──────────┐
   │ Online  │ │ Candidate│ │ Ranker   │
   │ feature │ │ generator│ │ (Triton  │
   │ store   │ │ (Redis + │ │  on GPU) │
   │ (Redis) │ │  ANN     │ │          │
   │         │ │  index)  │ │          │
   └─────────┘ └─────┬────┘ └────┬─────┘
                     │            │
                     ▼            ▼
                ┌─────────┐  ┌──────────┐
                │ ANN idx │  │ Model    │
                │ (Faiss /│  │ artifact │
                │ Qdrant) │  │ S3       │
                └─────────┘  └──────────┘

      Offline (batch, nightly):
      ┌──────────────────────────────────────┐
      │ Spark on EMR                          │
      │  - feature engineering                │
      │  - ALS retraining                     │
      │  - DLRM retraining (PyTorch on GPU)   │
      │  - Item/user embedding refresh        │
      └──────────────────────────────────────┘
```

### Step 4 — Cold start

Three faces of cold-start:

- **New user:** No history. Use demographic + location + popular-items fallback. Ramp into personalized as the user clicks.
- **New item:** No interactions. Use content-based embedding (image + description encoded). Surfaces in similar-item recommendations until interaction history accumulates.
- **New region:** Limited data. Use aggregate from similar regions; gradually transition.

### Step 5 — Online vs offline features

```
   Offline features (batch, daily):
     - User: avg_basket_size_30d, fav_categories, lifetime_spend
     - Item: rolling_sales_30d, avg_rating, return_rate

   Online features (real-time, Redis):
     - User: items_in_cart, last_search_query, items_viewed_session
     - Item: stock_level (live), price (live), promotion_active
     - Context: time_of_day, weather (Dubai-summer-AC-load matters!)
```

The ranker concatenates both sets at inference time.

### Likely follow-ups

*Q: "How do you measure success?"*
> Online: CTR, add-to-cart rate, conversion, GMV. Counter-metrics: diversity index, novelty (avg similarity between recommended and historically purchased items). All sliced by user segment.

*Q: "How do you avoid filter bubbles?"*
> Diversity term in re-ranking — penalize successive items from same category. Periodic exploration — 5% of slots reserved for items the model is uncertain about (epsilon-greedy). Track novelty as a counter-metric.

*Q: "How do you handle real-time inventory?"*
> Inventory is a hard filter — out-of-stock items never recommended. Done at re-ranking stage with live Redis inventory check. If a recommended item goes out-of-stock between ranking and serving, re-rank locally.

*Q: "How do you re-engage churning users?"*
> Notification recommender — same architecture but optimizing for "send-likelihood-to-click". Personalized timing (model the user's typical order time). Frequency cap to avoid spam. Hold-out group for measuring incrementality.

*Q: "Cost?"*
> Ranker on Triton GPU: ~$5K/month for 5K QPS. Candidate gen + ANN: ~$2K. Feature store: ~$1K. Spark training: ~$2K. **~$10K/month** for the AI side. Modest.

---

## 16.7 Closing the system design round

When the interviewer winds down, recap and offer extensions.

**How to say this:**

> "Recapping — I designed [X] with [key components], optimizing for [their stated priority]. The two trade-offs I made deliberately are [Y] and [Z]. If we had more time, I'd deep-dive on [specific component], cover the cost optimization angle, and walk through the rollout plan from prototype to production. Anything you want to revisit?"

That sentence demonstrates: structured thinking, awareness of trade-offs, awareness of what's still open, and a collaborative tone. All senior signals.

### Strong signals to give throughout
- "Let me draw this." (sketch unprompted)
- "What's the scale?" / "What's the latency target?" (requirement discipline)
- "I'd pick X over Y because Z." (trade-off explicit)
- "The bottleneck under load is..." (analytical)
- "We should alert on X." (production mindset)
- "Cost-wise this is roughly..." (commercial awareness)
- "If we doubled the scale, I'd change..." (forward-looking)

### Weak signals to avoid
- Jumping to implementation without clarifying scale
- Listing tools without rationale
- No diagram
- Ignoring failure modes
- Refusing to commit to a choice
- Over-engineering for hypothetical futures

---

Continue to **[Chapter 17 — Behavioral & HR](17_behavioral_hr.md)**.
