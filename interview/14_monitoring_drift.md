# Chapter 14 — Monitoring, Observability, Drift Detection
## Keeping models healthy after they're shipped

> JD: "Setting up monitoring, observability, and feedback loops for model performance and drift detection." Your ResMed Datadog + Snowflake dashboards are your flagship story here.

---

## 14.1 The three pillars of observability (same for ML as for software)

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│     METRICS      │  │      LOGS        │  │     TRACES       │
│                  │  │                  │  │                  │
│ What (numerical) │  │ What happened    │  │ Why (per-request │
│ per-time-slice   │  │ (narrative)      │  │  causation)      │
│                  │  │                  │  │                  │
│ Prometheus       │  │ Loki             │  │ Tempo / Jaeger   │
│ Datadog          │  │ CloudWatch Logs  │  │ Langfuse / OTEL  │
│ CloudWatch       │  │ Splunk           │  │ X-Ray            │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

For ML, add two ML-specific pillars:
- **Data / feature drift**
- **Prediction / outcome monitoring**

---

## 14.2 Model-specific drift — three flavors

### 14.2.1 Data drift
**P(X) changes.** Input distribution shifts. Example: your lender-identification model was trained on pre-2024 applicant data; in 2026 the applicant demographic shifted.

**Detection:** statistical tests on feature distributions.

### 14.2.2 Concept drift
**P(Y|X) changes.** The relationship between inputs and outputs shifts. Example: the same applicant profile that predicted low risk in 2020 predicts higher risk in 2026 because of macro-economic shift.

**Detection:** hardest — needs labels (often delayed).

### 14.2.3 Label drift (prior shift)
**P(Y) changes.** Class balance shifts. Example: default rate goes from 2% to 5%.

**Detection:** track predicted class distribution; alert on shifts.

### The relationship
- **Data drift** can happen without performance loss (if the feature isn't important)
- **Concept drift** always hurts
- Monitor both; prioritize acting on concept drift

---

## 14.3 Statistical tests for drift

### KS (Kolmogorov-Smirnov) test
- Two-sample test for continuous distributions
- Outputs p-value
- Pro: rigorous
- Con: sensitive at large N (tiny drift triggers alert)

### PSI (Population Stability Index)
- Binned distribution comparison
- Interpretable thresholds:
  - PSI < 0.1: stable
  - 0.1-0.25: moderate drift
  - > 0.25: significant drift
- **Industry standard in finance** (credit, banking)

```
PSI = Σ (P_actual - P_expected) * ln(P_actual / P_expected)
```

### Wasserstein distance (Earth Mover's Distance)
- Captures distance even when distributions don't overlap
- Good for **embedding drift** (vector distributions)

### Chi-squared
- Categorical features
- Compares observed vs expected counts

### MMD (Maximum Mean Discrepancy)
- Kernel-based distance between distributions
- Works in high dimensions
- Used for image / embedding drift

---

## 14.4 Drift monitoring for LLMs / embeddings

### Text input drift
- Query length distribution
- Language distribution
- Topic clusters (embed + UMAP)
- Vocabulary OOV rates

### Output drift
- Response length distribution
- Refusal rate ("I don't know" / "I can't help")
- Confidence distribution (if logprobs available)
- Toxicity / sentiment distribution

### Embedding drift
- Compute embeddings with a consistent model
- Track Wasserstein / MMD between reference and current distributions
- Visualize with UMAP (Arize Phoenix does this well)

---

## 14.5 The monitoring pipeline architecture

```
┌─────────────── Serving ───────────────┐
│                                        │
│  Request → Feature fetch → Predict     │
│     │         │              │         │
│     ▼         ▼              ▼         │
│  Log req   Log features   Log prediction
│  to Kafka  to Kafka       to Kafka    │
└──────────────┬─────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Stream processor        │
    │  (Flink / Spark)         │
    │  - Compute stats per     │
    │    window (5min / 1hr)   │
    │  - Drift tests vs baseline│
    └──────────────┬───────────┘
                   │
        ┌──────────┼──────────────┐
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│ Time-series   │         │ Alerting       │
│ store:        │         │ - Datadog       │
│ Prometheus    │         │ - PagerDuty     │
│ Datadog       │         │ - Slack         │
└───────────────┘         └───────────────┘
```

---

## 14.6 Your ResMed Datadog drift story — expanded

You said: "Built a Python-based utility integrating Datadog and Snowflake to generate automated drift monitoring dashboards."

Expand this into a 90-second narrative:

> At ResMed, each DS team shipping a model had to define drift monitoring in their own way — some used Snowflake SQL queries, some used notebooks, some had nothing. I built a utility that took a standardized YAML config (`model_id, features, thresholds, schedule`), computed KS/PSI per feature using Snowflake as the compute layer (model predictions + features were already logged there), and pushed metrics to Datadog via their API. The utility auto-generated a Datadog dashboard per model with:
> - Per-feature drift time series (KS + PSI)
> - Prediction distribution drift
> - Anomaly detection on drift scores
> - Annotations when retraining happened
>
> This unified monitoring across the MLOps platform. It was adopted by every team shipping models on IHS, and drift alerts became a CoE (center of excellence) standard.

---

## 14.7 Evaluating drift vs performance

**Trap:** drift without degradation. Features drift, but model accuracy holds. Don't retrain reflexively.

### Weight drift by feature importance
Instead of raw-feature drift, use SHAP / gain-weighted drift:

```
weighted_drift = Σ_feature (drift_score_feature × feature_importance)
```

An important feature drifting 0.15 PSI is more actionable than 10 unimportant features drifting 0.2.

### Tie retraining triggers to quality proxies
- When ground-truth labels are delayed, use proxies:
  - Prediction distribution drift
  - Confidence / entropy shifts
  - Business KPIs (click-through, conversion, default rate)
  - Shadow-model agreement rate

---

## 14.8 Alerts you want (and don't want)

### Good alerts
- p99 latency > SLO (actionable, auto-scale)
- Error rate > X% (actionable, investigate)
- Drift score > X AND prediction distribution shifted (concept drift signal)
- Feature pipeline freshness > X (upstream issue)
- Model version mismatch in prod (deploy gone wrong)

### Bad alerts (alert fatigue)
- Raw drift threshold on unimportant feature
- Single-window anomaly without hysteresis
- Weekday/weekend seasonal noise
- Deployment spikes marked as errors

### Alert discipline
- Every alert has a **runbook** (what to do)
- No alert fires on expected seasonality (use baselines)
- Alert fatigue = silent failures — trim aggressively

---

## 14.9 Tools comparison

| | **Evidently AI** | **Datadog** | **Arize** | **Fiddler** | **WhyLabs** |
|--|------------------|-------------|-----------|-------------|-------------|
| License | OSS + Cloud | Commercial SaaS | Commercial SaaS | Commercial | OSS + Cloud |
| Self-host | ✅ | ✅ (agents on-prem) | Limited | ✅ | ✅ |
| Strengths | Flexible reports, batch pipelines | Full stack (infra + ML), great alerting | LLM + embeddings, RAG-aware | Explainability + fairness | Statistical rigor, budget-friendly |
| Weaknesses | Less slick UI | Expensive at scale | SaaS-only by default | Smaller ecosystem | Smaller ecosystem |
| Best for | On-prem, batch | Existing Datadog shops | LLM-heavy | Regulated (XAI) | Cost-conscious |

For Avrioc UAE: **Evidently self-hosted** is the conservative pick (OSS, on-prem, data residency). Consider Datadog if they're already a Datadog shop.

---

## 14.10 Monitoring infra for LLM apps specifically

### What to log per request
- User / tenant / session ID
- Full prompt (maybe redacted PII)
- Full completion
- Tool calls + results
- Model + version + system fingerprint
- Tokens in / out
- Latency (TTFT, TPOT, E2E)
- User feedback (thumb up/down, regenerate clicked)
- Eval scores (faithfulness, relevance — computed async)

### Tools
- **Langfuse** (OSS, self-hostable)
- **Arize Phoenix** (OSS, embedding-focused)
- **Helicone** (proxy-based, one-header)
- **LangSmith** (LangChain's native)

### Sampling
At 1000 QPS, logging every trace is expensive. Sample strategies:
- **Head-based sampling** — log 1% uniformly
- **Tail-based sampling** — log all errors + slow requests + low-confidence
- **Per-tenant** — always log premium tenants
- **Random-per-session** — keep full sessions for some %

---

## 14.11 Feedback loops — closing the loop

### Implicit feedback
- Thumbs up/down in UI
- "Regenerate" clicked
- Copy button clicked
- Session ended without message (good or bad)
- Conversion events (clicked recommended item, completed purchase)

### Explicit feedback
- Rate 1-5 stars
- Select "helpful / not helpful"
- Annotate specific tokens as wrong

### Using feedback
- Aggregate into quality metrics (thumb-up rate per prompt version / model)
- Feed bad examples into golden set expansion
- Fine-tuning data (DPO / RLHF pairs)

---

## 14.12 Closed-loop retraining

```
  Production
      │
      ▼
  Log predictions + labels (delayed)
      │
      ▼
  Drift + performance metrics
      │
      ▼
  Trigger: drift > X AND (proxy metric ↓ OR time-since-retrain > 30d)
      │
      ▼
  Retraining pipeline (with human approval gate)
      │
      ▼
  Candidate model → Eval on golden set
      │
      ▼
  Shadow → Canary → Full rollout
```

**Critical:** human approval gate for production promotion in regulated GCC contexts (UAE PDPL, healthcare).

---

## 14.13 Incident response

### The runbook
1. **Detect** — alert fires
2. **Assess** — severity? how many users affected?
3. **Mitigate** — quick fix (flip prompt version, roll back model)
4. **Communicate** — status page, affected team, Slack
5. **Root-cause** — trace back
6. **Post-mortem** — blameless, documented
7. **Prevent** — add alert, add test, update runbook

### For LLM-specific incidents
- Toxic output → guardrail block + canned response + rate-limit tenant
- Hallucination spike → lower T, increase reranker top-N, flag prompt version
- Latency spike → check KV-cache eviction, Prometheus scrape, GC
- Cost spike → audit prompts for loops, check cache hit rate, verify gateway routing

---

## 14.14 Interview Q&A — Monitoring / Drift

**Q1. Data drift vs concept drift vs label drift?**
> Data: P(X) changes (input distribution). Concept: P(Y|X) changes (relationship between X and Y). Label: P(Y) changes (class balance). Data can happen without performance loss; concept always hurts.

**Q2. KS vs PSI vs Wasserstein — when each?**
> KS: rigorous p-value for continuous; sensitive at large N. PSI: interpretable thresholds (0.1 stable, 0.25 significant); finance standard. Wasserstein: robust when distributions don't overlap; good for embeddings.

**Q3. PSI formula and thresholds?**
> `PSI = Σ (P_actual - P_expected) * ln(P_actual / P_expected)`. <0.1 stable, 0.1-0.25 moderate, >0.25 significant.

**Q4. [Gotcha] Drift alerts fire daily but accuracy is stable. What do you do?**
> Don't retrain reflexively. Drift on unimportant features is noise. Use SHAP-weighted drift. Reduce alert sensitivity. Route alerts by feature importance. Tie retraining triggers to quality proxies, not raw drift.

**Q5. How do you monitor drift for text / embedding inputs?**
> Compute embeddings with a consistent model. Track Wasserstein / MMD between reference and current. Visualize UMAP (Arize Phoenix). For LLM prompts: token length, language distribution, topic clusters.

**Q6. [Gotcha] Labels arrive weeks late. How do you detect problems?**
> Proxy signals: prediction distribution drift, confidence/entropy shifts, business KPIs, shadow-model agreement. Deploy challenger model in shadow — compare prediction agreement as early signal.

**Q7. Evidently AI vs Datadog vs Arize?**
> Evidently: OSS, self-host, flexible reports, great for batch pipelines and on-prem/UAE. Datadog: integrated with infra stack, best if already a Datadog shop. Arize: ML-native, SaaS-first, strong on LLMs + embeddings.

**Q8. Closed-loop retraining trigger — design?**
> Combine drift detection + delayed-label performance + time cadence. Policy engine decides. Log every trigger decision to audit table. Human approval gate for prod promotion (mandatory for regulated GCC).

**Q9. [Gotcha] 2% accuracy drop overnight — drift or bug?**
> Triage: (1) data pipeline freshness + schema changes (most common). (2) Week-over-week feature distribution diff. (3) Serving-code or library upgrade. (4) Only then suspect concept drift. Rule out data engineering before retraining.

**Q10. How do you set monitoring alerts that don't cause fatigue?**
> Every alert has a runbook. No alerts on expected seasonality — use baselines. Use hysteresis (alert on sustained breach, not single-point). SHAP-weighted instead of raw-feature drift. Trim aggressively.

**Q11. What to log per LLM request?**
> User/tenant/session, full prompt + completion (redacted PII), tool calls + results, model + version + system fingerprint, tokens in/out, TTFT/TPOT/E2E, user feedback, eval scores.

**Q12. Sampling strategies for high-QPS tracing?**
> Head-based (log 1% uniformly). Tail-based (log all errors + slow + low-confidence). Per-tenant (premium always). Per-session (full session for some %).

**Q13. [Gotcha] Toxic output detected in production. Steps?**
> Immediate: strict output guardrail (block + canned response), flip prompt to prior-good, rate-limit tenant. Short-term: add failing example to eval, update guardrails. Post-mortem: trace to prompt change / model update / RAG poisoning. Blameless.

**Q14. How do you quantify implicit feedback into quality metrics?**
> Thumb-up rate, regenerate rate, copy-button rate, conversion rate, session-end silently. Aggregate per prompt version / model / tenant. Track weekly; alert on significant drops.

**Q15. What's the role of shadow traffic in monitoring?**
> Deploy a new model alongside the prod model; mirror traffic; log both predictions. Compare agreement rate — a pre-production proxy for concept drift before full rollout.

**Q16. System fingerprint — why monitor?**
> Model providers update models silently. Fingerprint change = behavior may change. Log it per trace; alert on changes; re-run golden set if it changes.

**Q17. How do you measure faithfulness of RAG answers in production?**
> Async LLM-judge on sampled traces — scores whether the answer is supported by the context. RAGAS, TruLens, Arize Phoenix. Track rolling p95 faithfulness; alert on drops.

**Q18. Observability tools — Langfuse vs LangSmith vs Phoenix?**
> Langfuse: OSS, self-hostable, strong on prompt management. LangSmith: LangChain-native, SaaS. Phoenix: OSS by Arize, embedding / drift focus. For UAE data residency, Langfuse self-hosted.

**Q19. What is a "golden set" in LLM monitoring?**
> 100-300 labeled (query, expected_output, rubric) samples. Your eval ground truth. Run on every prompt/model change in CI. Expand with real production failures.

**Q20. How do you monitor embedding drift?**
> Sample current embeddings, compare to reference distribution via Wasserstein or MMD. Also monitor downstream retrieval metrics (recall@k on a labeled golden set). UMAP visualization for exploratory.

**Q21. [Gotcha] p99 latency spikes every 5 min on vLLM. Diagnose.**
> Possible causes: Prometheus scrape hitting a busy worker, GC pauses, KV-cache eviction storms at concurrency peaks, K8s cron HPA re-evaluation, log flushes. Per-request traces + correlate with GPU util + nvidia-smi dmon.

**Q22. How do you monitor cost for LLM apps?**
> Gateway (LiteLLM, Portkey) attaches metadata (user, tenant, feature). Log token counts + cost per call. Aggregate in ClickHouse / BigQuery. Dashboards per model/tenant. Alert on budget breach.

**Q23. Data pipeline SLAs for ML?**
> Feature freshness (data arrived within X hours). Training data completeness (no missing shards). Feature store read latency p99 (e.g., <50ms). Alert on breach; every model has an explicit data SLA.

**Q24. What is RED method?**
> Rate (req/sec), Errors (per sec), Duration (latency p50/p95/p99). Standard for service monitoring. For ML add: Features (drift), Predictions (distribution).

**Q25. For Avrioc UAE — monitoring stack recommendation?**
> Self-host Langfuse (LLM observability), Evidently (drift), Prometheus + Grafana (infra), Loki (logs), Tempo (traces). All on-prem or in me-central-1. Datadog as a secondary if they already use it. Retention policies aligned with UAE PDPL (90d hot, 1y cold).

---

## 14.15 Resume tie-ins

Your headline stories here are strong:
- **ResMed Datadog-drift utility** → expand per Section 14.6
- **Deequ data quality (Tiger, Mars)** → talk about constraints (completeness, uniqueness, statistical tests); running as scheduled Databricks job
- **CI/CD with drift + retraining automation (Tiger)** → "automated model retraining based on monitoring reports"
- **Server anomaly detection with Prometheus + Grafana (Sopra)** → classic infra monitoring; connect to ML monitoring

---

Continue to **[Chapter 15 — Resume Projects, Deep Dive](15_resume_deep_dive.md)**.
