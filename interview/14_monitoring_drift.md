# Chapter 14 — Monitoring, Observability, Drift Detection
## Keeping models healthy after they're shipped

> JD: "Setting up monitoring, observability, and feedback loops for model performance and drift detection." Sachin's ResMed Datadog + Snowflake utility is the signature story here. Walk it like a system design — slowly, with diagrams.

---

## 14.1 Why monitoring matters more for ML than for software

### 14.1.1 The silent failure problem

A web service fails loud. The endpoint returns 500, latency spikes, the alarm fires, the on-call engineer wakes up. A machine learning model fails **silent**. The endpoint keeps returning 200. Predictions keep flowing. Latency stays flat. The model is just slowly, steadily getting wrong — and unless someone is specifically watching, accuracy can decay 5%, 10%, 20% before anyone notices because the user impact is gradual and the engineering metrics are unaffected.

This is the central reason ML monitoring deserves its own discipline. Standard software observability — metrics, logs, traces — answers "is the system up?" Machine learning monitoring answers "is the system **right**?"

### 14.1.2 The story I tell new engineers

When I joined ResMed, one of the first things I noticed was that we had a fraud-scoring model running for eighteen months without anyone monitoring its accuracy. The endpoint was healthy — 50ms p99, 0.01% error rate, beautiful Datadog dashboards. But the production data had drifted because customer behavior changed during COVID, and the model's actual fraud-catch-rate had degraded from 91% to 76%. We only discovered it because a finance analyst noticed the chargeback rate was creeping up. Three months of recovery work to retrain, validate, and ship a new model.

That story is why I built the drift utility. Monitoring engineering metrics keeps the system **alive**. Monitoring data and prediction distributions keeps the system **correct**. Both are non-negotiable.

### 14.1.3 The five pillars

Standard observability has three pillars. ML monitoring has five.

```
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │  METRICS   │  │    LOGS    │  │   TRACES   │   <- standard observability
   │            │  │            │  │            │
   │ Prometheus │  │ Loki       │  │ Tempo      │
   │ Datadog    │  │ CloudWatch │  │ Jaeger     │
   │ CloudWatch │  │ Splunk     │  │ X-Ray      │
   └────────────┘  └────────────┘  └────────────┘

   ┌────────────────────────┐  ┌────────────────────────┐
   │  DATA / FEATURE DRIFT  │  │  PREDICTION / OUTCOME  │   <- ML-specific
   │                        │  │                        │
   │ Evidently / Arize      │  │ Predicted distribution │
   │ Fiddler / WhyLabs      │  │ Outcome metrics        │
   │ Datadog custom metrics │  │ Business KPIs          │
   └────────────────────────┘  └────────────────────────┘
```

---

## 14.2 The drift taxonomy — three flavors that matter

### 14.2.1 Data drift (covariate shift) — P(X) changes

The input distribution shifts. The model's relationship between inputs and outputs is unchanged, but the inputs you're seeing now don't look like the inputs you trained on.

**Example:** You trained a credit risk model in 2019 when the typical applicant was 32 years old earning ₹40,000/month. In 2026 the typical applicant is 26 years old earning ₹65,000/month. The model still works *in principle* — its weights are correct — but it's extrapolating outside its training distribution. Some predictions are reasonable; many aren't.

**Detection:** statistical tests on input feature distributions vs reference window. PSI, KS test, Wasserstein. This is the easy kind to detect because you don't need labels — just the inputs.

### 14.2.2 Concept drift — P(Y|X) changes

The relationship itself shifts. The same input now produces a different correct output.

**Example:** Pre-pandemic, an applicant with stable employment, ₹50k income, no defaults was a low-risk loan. Post-pandemic, the same profile has higher default risk because of macroeconomic conditions. The input X is identical; the correct output Y has changed. The model's weights are now actually wrong, not just extrapolating.

**Detection:** the hard one — needs labels, which often arrive weeks or months late. Use proxy signals: prediction distribution drift, business KPIs, shadow-model agreement.

### 14.2.3 Label drift (prior shift) — P(Y) changes

The class balance shifts. The relationship P(Y|X) and the input distribution P(X) might be stable, but the prior probability of each class is different.

**Example:** Default rate drops from 5% to 2% because of a stricter underwriting policy upstream. The model still calibrates well per-applicant but its overall positive-class rate is now miscalibrated.

**Detection:** track predicted class distribution over time; alert on significant shifts. Easy to detect.

### 14.2.4 The visualization

```
   ┌─────────────────────────────────────────────────┐
   │                  DRIFT TYPES                    │
   └─────────────────────────────────────────────────┘

   DATA DRIFT (P(X) changes)
   ─────────────────────────
   Reference X distribution:    [..▁▂▃▅▇▅▃▂▁..]   age 32 mean
   Current   X distribution:    [▁▃▅▇▅▃▂▁........]  age 26 mean
                                     ─── shift left

   CONCEPT DRIFT (P(Y|X) changes)
   ──────────────────────────────
   Reference: same X -> same Y
                                P(default | X=stable_emp) = 0.05
   Current:   same X -> diff Y
                                P(default | X=stable_emp) = 0.12
                                ─── relationship changed

   LABEL DRIFT (P(Y) changes)
   ──────────────────────────
   Reference: 95% non-default, 5% default
   Current:   98% non-default, 2% default
                                ─── prior shifted
```

### 14.2.5 The relationship and the trap

Data drift can happen **without performance loss** — the feature that drifted might be unimportant. Concept drift always hurts. The standard rookie mistake is treating raw data drift as automatically actionable. It isn't. **Don't retrain on data drift alone.** Drift is diagnostic; retraining is the cure for actual degradation, signaled by quality metrics.

### 14.2.6 How to say this in an interview

> "When I think about model drift I separate three things. Data drift is when the input distribution shifts — easy to detect because you don't need labels, but it doesn't always hurt the model because the drifted feature might be unimportant. Concept drift is when the input-output relationship itself changes — this always hurts, but it's hard to detect because you need ground truth labels which often arrive late. Label drift is when class balance shifts — easy to detect on predictions. The standard mistake is reflexively retraining on raw data drift; I'd rather use SHAP-importance-weighted drift as a diagnostic and tie retraining to actual quality metrics or proxies like prediction confidence shifts."

---

## 14.3 PSI — Population Stability Index, the workhorse

### 14.3.1 Why PSI is the industry standard

In finance — credit, banking, insurance — PSI is the de facto drift metric. It has a simple interpretation, well-known thresholds, and produces a single number per feature that you can chart over time. Both finance regulators and ML platforms use it.

### 14.3.2 The intuition first

PSI asks: "Compared to my reference distribution, how much has the actual distribution shifted, weighted by how meaningful that shift is?" It's essentially the KL divergence between two binned distributions, made symmetric. Big PSI = big shift. Small PSI = stable.

### 14.3.3 The formula

```
       k
PSI = Σ  (P_i - Q_i) × ln(P_i / Q_i)
      i=1
```

**Symbol meanings:**
- `k` — number of bins (typically 10)
- `P_i` — fraction of points in bin i in the **current** (actual) distribution
- `Q_i` — fraction of points in bin i in the **reference** (expected) distribution
- The factor `(P_i - Q_i)` measures the directional shift; `ln(P_i / Q_i)` measures the relative change

### 14.3.4 Worked example you can do on a whiteboard

Reference distribution Q: a feature like "credit score" binned into 5 buckets. Current distribution P: same feature, this week.

| Bin | Q (reference) | P (current) | P - Q | ln(P/Q) | Contribution |
|-----|---------------|-------------|-------|---------|--------------|
| 1 (300-500) | 0.10 | 0.15 | 0.05 | ln(1.5) = 0.405 | 0.05 × 0.405 = **0.020** |
| 2 (500-600) | 0.20 | 0.25 | 0.05 | ln(1.25) = 0.223 | 0.05 × 0.223 = **0.011** |
| 3 (600-700) | 0.40 | 0.30 | -0.10 | ln(0.75) = -0.288 | -0.10 × -0.288 = **0.029** |
| 4 (700-800) | 0.20 | 0.20 | 0.00 | ln(1.0) = 0 | **0.000** |
| 5 (800-900) | 0.10 | 0.10 | 0.00 | ln(1.0) = 0 | **0.000** |
| **Total** | | | | | **PSI = 0.060** |

PSI = 0.060. Below 0.1 → stable. No alert.

Now suppose next week the distribution shifts dramatically:

| Bin | Q (reference) | P (current) | P - Q | ln(P/Q) | Contribution |
|-----|---------------|-------------|-------|---------|--------------|
| 1 (300-500) | 0.10 | 0.30 | 0.20 | ln(3.0) = 1.099 | **0.220** |
| 2 (500-600) | 0.20 | 0.30 | 0.10 | ln(1.5) = 0.405 | **0.041** |
| 3 (600-700) | 0.40 | 0.20 | -0.20 | ln(0.5) = -0.693 | **0.139** |
| 4 (700-800) | 0.20 | 0.15 | -0.05 | ln(0.75) = -0.288 | **0.014** |
| 5 (800-900) | 0.10 | 0.05 | -0.05 | ln(0.5) = -0.693 | **0.035** |
| **Total** | | | | | **PSI = 0.449** |

PSI = 0.449 → significant drift. Alert and investigate.

### 14.3.5 The threshold mnemonic

```
   PSI < 0.10  →  "stable, no concern"
   PSI 0.10 - 0.25  →  "moderate drift, investigate"
   PSI > 0.25  →  "significant drift, action required"
```

### 14.3.6 Common PSI mistakes

1. **Using too few bins.** With 3 bins you'll miss real drift; with 30 you'll get noisy alerts. 10 bins is the standard.
2. **Empty bins.** ln(0) is undefined; ln(P/0) is infinite. Add a small epsilon (1e-6) to both P and Q in every bin to avoid blow-ups.
3. **Letting the reference window stale.** A reference from 2 years ago will show drift even when nothing has changed because of long-term gradual shift. Update the reference window quarterly.

### 14.3.7 Code skeleton

```python
import numpy as np

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two distributions."""
    # Use reference quantiles as bin edges
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Avoid log(0) with small epsilon
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi_per_bin = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(psi_per_bin.sum())
```

Walking the key lines: bin edges come from reference quantiles so each reference bin holds 10% of mass. Open-ended outer edges handle out-of-range new data. Epsilon clamp prevents log-zero blowups. The summed contribution is symmetric and always non-negative.

---

## 14.4 KS test — for continuous distributions with statistical rigor

### 14.4.1 What KS does

The Kolmogorov-Smirnov test computes the **maximum absolute difference between two cumulative distribution functions**. Outputs a test statistic D and a p-value.

```
       D = sup |F_ref(x) - F_cur(x)|
            x

   Two CDFs:
        1 ─┐                        ┌──────
           │            ┌──────────┘
   F_cur   │      ┌────┘
           │ ┌───┘  ↑ max gap = D
        0 ─┴─────────────────────── x

           │  ┌─────┐
   F_ref   │ ┌┘     └─────┐
           │┌            ┌┘
        0 ─┴─────────────────────── x
```

### 14.4.2 When to use KS vs PSI

| Aspect | KS | PSI |
|--------|----|----|
| Output | D-statistic + p-value | Single PSI score |
| Interpretation | Statistical (p < 0.05) | Threshold-based (0.1, 0.25) |
| Sample size sensitivity | **Sensitive** at large N | Stable |
| Use case | Small samples, rigor needed | Production, binned features |
| Ease of communication | Statistical jargon | Easy for business stakeholders |

The big practical caveat: **KS p-values become useless at production scale**. With 1 million samples, even a 0.5% real shift produces p << 0.05. You'd alert constantly. PSI's threshold-based interpretation scales better. In practice, I report both: D-statistic (effect size) plus PSI score (interpretable threshold).

### 14.4.3 Common mistake

Using KS p-value as the alert criterion at high traffic. You'll alert on every imperceptible shift. Use the D-statistic itself or PSI thresholds.

---

## 14.5 Wasserstein distance — for embeddings and continuous spaces

### 14.5.1 Why this exists

KS and PSI work fine for 1D distributions. For multidimensional distributions — like LLM embedding vectors of dimension 768 or 1536 — they don't generalize cleanly. Wasserstein distance (Earth Mover's Distance) measures "how much work to transform distribution P into distribution Q" and works in arbitrary dimensions.

### 14.5.2 The intuition

Imagine each distribution is a pile of dirt. Wasserstein distance is the minimum total work — mass × distance moved — required to reshape pile P into pile Q. If the piles are identical, distance = 0. If they're far apart, distance is large.

### 14.5.3 When to use

- **Embedding drift** — the modern LLM monitoring use case. Sample current embeddings, compare via Wasserstein to a reference embedding distribution.
- **High-cardinality categorical features** — when bin-based PSI is fragile.
- **Visual data drift** — image embeddings.

### 14.5.4 MMD — Maximum Mean Discrepancy

A close cousin: kernel-based distance between distributions. Computes the mean embedding of each sample in a reproducing kernel Hilbert space and takes the squared distance. Used by drift libraries (Alibi-Detect, Evidently) for high-dimensional drift. Faster than full Wasserstein on big samples.

---

## 14.6 Embedding drift for LLMs — the new frontier

### 14.6.1 Why monitor embeddings

For LLM applications, the user input is text. Text drift in raw form (word frequencies, n-grams) is noisy and hard to interpret. Embedding the text into a fixed dimensional vector and tracking the embedding distribution gives a cleaner, semantically-meaningful drift signal.

### 14.6.2 The pipeline

```
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ Production   │───▶│  Embedder    │───▶│ Sample       │
   │ user prompts │    │ (consistent  │    │ embedding    │
   │              │    │  model!)     │    │ store        │
   └──────────────┘    └──────────────┘    └──────┬───────┘
                                                   │
                              ┌────────────────────┘
                              ▼
                       ┌──────────────────┐
                       │ Reference        │
                       │ embedding sample │
                       │ (training-time)  │
                       └──────┬───────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Wasserstein /    │
                       │ MMD between sets │
                       └──────┬───────────┘
                              ▼
                       ┌──────────────────┐
                       │ UMAP visualize   │
                       │ Cosine to        │
                       │ centroid         │
                       │ Drift metric     │
                       └──────────────────┘
```

### 14.6.3 What to track for LLM apps

| Signal | What drift means |
|--------|------------------|
| Prompt embedding distribution | Topic distribution shift |
| Prompt length distribution | Usage pattern shift |
| Refusal rate ("I can't help with that") | Model becoming more cautious |
| Output length distribution | Verbosity drift |
| Output embedding distribution | Response style drift |
| Cosine to training centroid | Out-of-distribution requests |
| Vocabulary OOV rate | Unknown-domain drift |

### 14.6.4 Tooling

Arize Phoenix has the best UMAP visualization for embedding drift — you can scrub through time and watch clusters appear and disappear. Langfuse can ingest custom embedding metrics. Evidently has built-in embedding drift reports.

---

## 14.7 The full monitoring pipeline architecture

### 14.7.1 The system diagram

```
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ Production   │───▶│  Logger      │───▶│  Snowflake   │
   │  Endpoint    │    │  (request_id,│    │ predictions  │
   │  (SageMaker, │    │  features,   │    │ + features   │
   │   Lambda,    │    │  prediction) │    │   table      │
   │   vLLM)      │    │              │    │              │
   └──────────────┘    └──────────────┘    └──────┬───────┘
                                                   │
                              ┌────────────────────┘
                              ▼
                       ┌──────────────┐
                       │ Drift script │      Daily / hourly schedule
                       │ (scheduled)  │      Pulls reference window
                       │ - PSI        │      vs current window
                       │ - KS         │      Per feature
                       │ - MMD (emb)  │
                       └──────┬───────┘
                              │ push metrics
                              ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │   Datadog    │◀───│ Custom       │───▶│   Slack      │
   │  Dashboards  │    │ metrics API  │    │  PagerDuty   │
   │              │    │              │    │              │
   │ - Drift line │    │              │    │ alerts when  │
   │ - PSI heat   │    │              │    │ thresholds   │
   │ - Anomaly    │    │              │    │ breached     │
   │   monitors   │    │              │    │              │
   └──────────────┘    └──────────────┘    └──────────────┘
```

### 14.7.2 Why this architecture works

- **Snowflake as compute layer**: predictions and features are already there for analytical reasons. Don't move them.
- **Datadog as visualization**: ops teams already have it; reuse infrastructure.
- **Custom metrics push**: gives you full control over the alert logic without coupling to Datadog's built-in drift detection.
- **Scheduled batch, not stream**: drift is a slow-moving signal; batch is fine and cheaper than streaming.

---

## 14.8 The ResMed Datadog drift utility — Sachin's signature story

This is the story you should be ready to spend 5 minutes on. Walk it like a system design.

### 14.8.1 The 90-second pitch

> "At ResMed each data science team shipping a model on the IHS platform was reinventing drift monitoring. Some used Snowflake SQL queries, some had Jupyter notebooks running on a cron, some had nothing. There was no central view of how every model was behaving. I built a utility that unified all of this into a per-model YAML config and an automated Datadog dashboard. Eight production models adopted it; it became the CoE standard for the platform."

### 14.8.2 The architecture in detail

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                  RESMED DRIFT UTILITY                           │
   └─────────────────────────────────────────────────────────────────┘

   STEP 1: Data scientist writes config
   ─────────────────────────────────────
   model_id: ihs_session_quality_v3
   features:
     - name: session_duration
       drift_metric: psi
       bins: 10
       threshold: 0.25
     - name: device_type
       drift_metric: chi_squared
       threshold: p < 0.01
     - name: prompt_embedding
       drift_metric: mmd
       threshold: 0.15
   schedule: "0 2 * * *"   # daily 2 AM UTC
   reference_window: "30d-60d ago"
   current_window:   "0d-1d ago"
   alert_channels:
     - slack: "#ihs-mlops-alerts"
     - datadog: "anomaly-monitor"

   STEP 2: CI registers the config
   ─────────────────────────────────
   Config validated -> registered to MLOps platform DB
                    -> Airflow DAG generated
                    -> Datadog dashboard auto-generated via API

   STEP 3: Daily run
   ─────────────────
   Airflow triggers Python job:
     1. Pull reference window from Snowflake
     2. Pull current window from Snowflake
     3. Compute drift metric per feature
     4. Push metrics to Datadog API as custom metrics
     5. Annotate Datadog timeline with retraining events

   STEP 4: Alerting
   ────────────────
   Datadog anomaly-detection monitor:
     condition: drift score > threshold for 2+ consecutive runs
     severity: P2 (drift), P1 (drift + perf proxy degraded)
     -> Slack #ihs-mlops-alerts
     -> PagerDuty if P1
```

### 14.8.3 The dashboard layout

Each model got an auto-generated Datadog dashboard with the following sections:

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Model: IHS Session Quality v3                              │
   │  Last retrained: 2026-03-15  |  Next scheduled: 2026-04-30  │
   ├─────────────────────────────────────────────────────────────┤
   │                                                             │
   │  ┌──────────────────────────┐  ┌──────────────────────────┐ │
   │  │ PSI per feature (line)   │  │ Prediction distribution  │ │
   │  │  ──── session_duration   │  │ histogram                │ │
   │  │  ──── device_type        │  │                          │ │
   │  │  ──── prompt_embed       │  │                          │ │
   │  └──────────────────────────┘  └──────────────────────────┘ │
   │                                                             │
   │  ┌──────────────────────────┐  ┌──────────────────────────┐ │
   │  │ PSI heatmap (feature ×   │  │ Anomaly detection on     │ │
   │  │ time, color = drift)     │  │ drift score              │ │
   │  └──────────────────────────┘  └──────────────────────────┘ │
   │                                                             │
   │  ┌──────────────────────────┐  ┌──────────────────────────┐ │
   │  │ Latency p50/p95/p99      │  │ Error rate               │ │
   │  └──────────────────────────┘  └──────────────────────────┘ │
   │                                                             │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │ Annotations: retraining events, deploys, incidents   │   │
   │  └──────────────────────────────────────────────────────┘   │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
```

### 14.8.4 What made it work

- **Self-service config-driven**: data scientists owned the YAML, MLEs maintained the framework.
- **Reused existing infra**: no new tools, just Snowflake + Datadog + Airflow.
- **Anomaly detection built in**: not just thresholds — Datadog's anomaly-detection monitors caught gradual drift.
- **Annotated with deploys**: every retraining event was a vertical line on the dashboard, so you could see "drift dropped after Mar 15 retrain."

### 14.8.5 The interview line

> "The reason that utility was successful wasn't technical sophistication — it was the right level of abstraction. Data scientists got drift monitoring with thirty lines of YAML; MLEs got a single codebase to maintain instead of eight bespoke scripts. The metric definitions were portable across models, the alerting was uniform, and the dashboards looked the same so any on-call engineer could read them. That's what platform thinking looks like in MLOps."

---

## 14.9 Drift vs performance — the senior trap

### 14.9.1 The trap

Drift alerts fire daily. Engineering team rushes to retrain. New model ships. Drift alerts continue firing. Cycle repeats. Meanwhile actual production accuracy was fine the whole time.

This pattern is everywhere. The trap is treating raw drift as automatically actionable.

### 14.9.2 SHAP-importance-weighted drift

Instead of raw per-feature drift, weight each feature's drift by its SHAP importance to the model:

```
weighted_drift = Σ_feature  (drift_score × |SHAP_importance|)
```

A feature with PSI = 0.3 but SHAP importance near zero contributes nothing to weighted drift. A feature with PSI = 0.15 but high SHAP importance contributes a lot. Now the alert fires only when **drift on important features** crosses a threshold.

### 14.9.3 Tying retraining to quality, not drift

When ground-truth labels are delayed, use **proxy metrics**:

- **Prediction distribution drift** — if the predicted positive rate jumps from 5% to 15%, something's off
- **Confidence/entropy shifts** — model getting less confident on average suggests OOD inputs
- **Business KPIs** — chargeback rate, conversion rate, click-through
- **Shadow-model agreement** — challenger model in shadow; agreement rate is a pre-label proxy

### 14.9.4 Common mistakes

1. **Auto-retraining on raw drift.** Wastes compute and risks regressing.
2. **Single-window thresholds.** One bad batch shouldn't trigger; require sustained breach.
3. **Same threshold for all features.** Customize per feature based on baseline variability.

---

## 14.10 Tools comparison — Evidently vs Arize vs WhyLabs vs Fiddler

```
   ┌───────────────────────────────────────────────────────────────┐
   │                  DRIFT MONITORING TOOLS                        │
   └───────────────────────────────────────────────────────────────┘
```

| Aspect | **Evidently** | **Arize** | **WhyLabs** | **Fiddler** |
|--------|---------------|-----------|-------------|-------------|
| License | OSS + Cloud | Commercial SaaS | OSS + Cloud | Commercial |
| Self-host | ✅ canonical | Limited | ✅ | ✅ |
| Strengths | Flexible reports, batch pipelines, OSS | LLM + embeddings, RAG-aware, polished UI | Statistical rigor, budget-friendly | Explainability + fairness focus |
| Weaknesses | Less polished UI | SaaS-first | Smaller ecosystem | Smaller ecosystem |
| LLM support | Decent | Excellent (Phoenix) | OK | Limited |
| Sweet spot | On-prem batch monitoring | LLM-heavy, complex apps | Cost-conscious teams | Regulated, XAI requirements |

**For Avrioc UAE:** Evidently AI self-hosted is the conservative pick — OSS, on-prem, data residency, integrates cleanly with batch pipelines. Add Arize Phoenix if they have heavy LLM workloads needing embedding visualizations.

---

## 14.11 Datadog + Prometheus + Grafana — the system metrics stack

### 14.11.1 The split

ML monitoring covers data and predictions. **System monitoring** covers the infrastructure: latency, error rate, GPU utilization, memory, throughput. Different tools for different concerns.

```
   ┌────────────────────────────────────────────────────────┐
   │                  MONITORING SPLIT                       │
   └────────────────────────────────────────────────────────┘

   ML MONITORING                   SYSTEM MONITORING
   ─────────────                   ─────────────────
   - Drift (PSI, KS, MMD)         - Latency (p50, p95, p99)
   - Prediction distribution      - QPS / RPS
   - Feature freshness            - Error rate
   - Outcome metrics              - GPU utilization
   - Faithfulness scores           - Memory / disk
                                   - Network I/O

   Tools:                          Tools:
   - Evidently                     - Prometheus + Grafana
   - Arize / WhyLabs              - Datadog
   - Custom + Datadog              - CloudWatch
```

### 14.11.2 Prometheus + Grafana — the OSS stack

```
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ Application     │    │ Prometheus      │    │ Grafana         │
   │ /metrics        │───▶│ scrape every    │───▶│ dashboards      │
   │ endpoint        │    │ 15-30 seconds   │    │ alert rules     │
   └─────────────────┘    └─────────────────┘    └─────────┬───────┘
                                                            │
                          ┌─────────────────┐              │
                          │ AlertManager    │◀─────────────┘
                          │ - PagerDuty     │
                          │ - Slack         │
                          │ - email         │
                          └─────────────────┘
```

### 14.11.3 Histograms vs summaries

A frequent gotcha. Both are Prometheus metric types for latency.

```
   HISTOGRAM (prometheus_client.Histogram)
   ────────────────────────────────────────
   Pre-defined buckets, e.g. [10ms, 50ms, 100ms, 500ms, 1s]
   Records count of requests in each bucket
   Quantiles computed at QUERY time across instances
   ✅ Aggregatable across pods
   ✗  Quantile accuracy depends on bucket choice

   SUMMARY (prometheus_client.Summary)
   ───────────────────────────────────
   Computes quantiles AT INGEST time per instance
   ✅ Exact quantile values
   ✗  NOT aggregatable across pods (cannot compute global p99)
```

**Rule of thumb:** Use histograms for latency in distributed systems. Use summaries only when you have a single instance and need exact quantiles.

### 14.11.4 Recording rules — the optimization

Computing complex Prometheus queries on every dashboard load is slow. Recording rules pre-compute and store the result as a new time series:

```yaml
# prometheus-rules.yml
groups:
- name: ml_serving
  interval: 30s
  rules:
  - record: model:request_latency:p99_5m
    expr: |
      histogram_quantile(0.99,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le, model))
```

Now dashboards query `model:request_latency:p99_5m` directly. Sub-second load times.

### 14.11.5 Resume tie-in

> "At Sopra I built infrastructure anomaly detection using Prometheus and Grafana — server CPU, memory, disk, network metrics scraped every 30 seconds, custom histograms for application latency, AlertManager piping P1 alerts to PagerDuty. The same patterns applied later when I built the ResMed ML monitoring — Prometheus for system metrics on SageMaker endpoints, Datadog custom metrics for drift signals. The split between system and ML monitoring is essential; using one tool for both inevitably means one of them gets short-changed."

---

## 14.12 Alerting strategy — SLOs and error budgets

### 14.12.1 SLI vs SLO vs SLA

```
   SLI  Service Level Indicator    Measurement (e.g. "p99 latency = 280ms")
   SLO  Service Level Objective    Target (e.g. "p99 < 500ms 99.9% of time")
   SLA  Service Level Agreement    Contract with consequences
```

Define SLOs for ML services. Examples:

- **Latency SLO**: p99 inference latency < 500ms, 99.9% of 30-day window
- **Availability SLO**: endpoint successful response 99.9% of 30-day window
- **Drift SLO**: PSI on top-5 SHAP features < 0.25, 95% of weekly windows
- **Quality SLO**: weekly thumb-up rate > 70% on production traffic

### 14.12.2 Error budgets

If your SLO is 99.9% over 30 days, your error budget is 0.1% — about 43 minutes of allowed failure per month. **Budget burn-rate alerts** fire when you're consuming the budget faster than allowed:

- **Fast burn** — burning 5% of monthly budget in 1 hour → page immediately
- **Slow burn** — burning 10% of budget in 6 hours → ticket, investigate during work hours

### 14.12.3 Good alerts vs bad alerts

```
   ┌──────────────────────────────────────────────────────────┐
   │  GOOD ALERTS                                             │
   ├──────────────────────────────────────────────────────────┤
   │  - p99 latency > SLO (actionable, autoscale or rollback) │
   │  - Error rate > X% sustained (investigate)               │
   │  - Drift on important features + perf proxy degraded     │
   │  - Feature pipeline freshness > X (upstream broken)      │
   │  - Model version mismatch in prod (deploy went wrong)    │
   └──────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────┐
   │  BAD ALERTS (cause fatigue)                              │
   ├──────────────────────────────────────────────────────────┤
   │  - Raw drift on unimportant feature                      │
   │  - Single-window anomaly without hysteresis              │
   │  - Weekday/weekend seasonal noise                        │
   │  - Deployment spikes flagged as errors                   │
   │  - "FYI" alerts with no runbook                          │
   └──────────────────────────────────────────────────────────┘
```

### 14.12.4 Alert discipline

Three rules:

1. **Every alert has a runbook.** What does the on-call do when this fires? If you can't write that down, the alert shouldn't exist.
2. **No alert fires on expected seasonality.** Use baselines and anomaly detection, not raw thresholds.
3. **Trim aggressively.** Alert fatigue causes silent failures. If an alert hasn't been actionable in 30 days, retire it.

### 14.12.5 Page vs ticket vs ignore

```
   PAGE          - immediate user impact, requires action now
                 - examples: endpoint down, latency >> SLO,
                            quality crashed
                 - target: 1-2 pages per week per engineer

   TICKET        - non-urgent, investigate during work hours
                 - examples: drift trending up, slow burn
                 - target: <5 tickets per week per engineer

   IGNORE        - informational, automatic correction expected
                 - examples: cache miss spikes, deploy ramping
```

---

## 14.13 LLM-specific monitoring

### 14.13.1 What's different

For classical ML you monitor input distributions and output predictions. For LLMs add:

- **Prompt length distribution** — usage pattern shifts
- **Response length distribution** — model-side verbosity drift
- **Refusal rate** — "I can't help with that" frequency, indicates over-cautiousness or jailbreak attempts
- **Embedding drift** — semantic distribution shift on inputs
- **Faithfulness scores** — async LLM-judge on RAG outputs
- **Tool call patterns** — what tools is the agent invoking?
- **Cost per request** — token efficiency drift
- **Latency breakdown** — TTFT (time to first token), TPOT (time per output token), E2E

### 14.13.2 What to log per LLM request

```json
{
  "request_id": "uuid",
  "tenant_id": "acme",
  "user_id": "user42",
  "session_id": "sess-1234",

  "prompt_version_id": "clinical_rag_v3.2",
  "prompt_text": "[redacted PII]",
  "prompt_token_count": 4521,
  "rag_doc_ids": [101, 245, 378],

  "model": "claude-sonnet-4-7",
  "system_fingerprint": "fp_abc123",
  "temperature": 0.2,

  "response_text": "[redacted]",
  "response_token_count": 312,
  "tool_calls": [{...}],

  "ttft_ms": 280,
  "tpot_ms": 22,
  "e2e_ms": 1840,

  "cost_usd": 0.0142,

  "user_feedback": null,
  "eval_scores": {
    "faithfulness": 0.92,
    "relevance": 0.87,
    "refusal": false
  }
}
```

### 14.13.3 Sampling at high QPS

At 1000 QPS, full-trace logging is expensive. Strategies:

- **Head-based** — random 1-5% uniformly
- **Tail-based** — log all errors, all p99 slow requests, all low-confidence
- **Per-tenant** — premium tenants always logged
- **Per-session** — full session traces for some random %

OpenTelemetry's tail-based sampling is the production gold standard.

### 14.13.4 LLM observability tools

- **Langfuse** — OSS, self-hostable, strong on traces + prompts + evals
- **Arize Phoenix** — OSS, embedding-drift focus, UMAP visualization
- **Helicone** — proxy-based, single-header integration
- **LangSmith** — LangChain-native SaaS

For Avrioc with UAE residency: **Langfuse self-hosted on Postgres**.

---

## 14.14 Closed-loop retraining

### 14.14.1 The flow

```
       Production traffic
             │
             ▼
   ┌──────────────────────┐
   │ Log predictions +    │
   │ features + outcomes  │ <- outcomes often delayed
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Drift detection      │
   │ + perf proxy metrics │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────────────────┐
   │ Trigger policy:                  │
   │  drift > X on important features │
   │   AND                            │
   │  (proxy degraded OR              │
   │   time_since_retrain > 30d)      │
   └──────────┬───────────────────────┘
              ▼ trigger
   ┌──────────────────────┐
   │ Retraining pipeline  │
   │ (CI runs full train) │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Eval golden set      │
   │ Must beat prod by X% │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ HUMAN APPROVAL GATE  │  <- mandatory in regulated GCC
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ Shadow → Canary →    │
   │ Full rollout         │
   └──────────────────────┘
```

### 14.14.2 The non-negotiable

In regulated environments — UAE PDPL, healthcare, finance — never wire automated retraining directly to production deploy without human approval. Auditors will reject the system. The trigger can be automatic; the production promotion must be human-gated.

---

## 14.15 Incident response runbook

### 14.15.1 The seven-step runbook

```
   1. DETECT       Alert fires; confirm not a false positive
   2. ASSESS       Severity? How many users affected?
                   What's the blast radius?
   3. MITIGATE     Immediate fix - flip prompt, roll back model,
                   shed load, rate-limit tenant
   4. COMMUNICATE  Status page if user-facing, Slack to team,
                   incident channel
   5. ROOT CAUSE   Trace through logs, recent changes,
                   data pipeline events
   6. POST-MORTEM  Blameless, documented, action items
   7. PREVENT      Add alert, add test, update runbook,
                   close action items
```

### 14.15.2 LLM-specific incidents

| Incident | Mitigation | Root cause to investigate |
|----------|------------|--------------------------|
| Toxic output | Strict guardrail + canned reply, rate-limit | Prompt change, model update, RAG poisoning |
| Hallucination spike | Lower temperature, increase reranker top-N | Embedding drift, RAG context quality |
| Cost spike | Audit prompts for loops, check cache hit rate | Prompt verbosity, agent loop bug |
| Latency spike | Check KV-cache eviction, GPU util | Concurrency surge, GC pauses |
| Refusal rate spike | Roll back to known-good prompt | Model provider update, jailbreak attempts |

---

## 14.16 Resume tie-ins — Sachin's monitoring portfolio

### 14.16.1 The headline stories

> 1. **ResMed Datadog drift utility** (signature) — see Section 14.8 for the 5-minute walkthrough
> 2. **Tiger Analytics drift + retraining** — automated retraining pipelines triggered by drift signals from Snowflake-logged predictions
> 3. **TrueBalance real-time monitoring** — Lambda XGBoost p99 < 500ms with CloudWatch + Datadog stack
> 4. **Sopra infrastructure monitoring** — Prometheus + Grafana for server anomaly detection; pattern reused for ML systems later
> 5. **Deequ data quality at Tiger** — schedule-based data validation; constraints on completeness, uniqueness, statistical sanity

### 14.16.2 The 3-minute deep narrative for the ResMed drift utility

Use this verbatim if asked "tell me about a monitoring system you built."

> "The drift utility started because every data scientist on the IHS platform was reinventing the same wheel — drift monitoring with custom Snowflake queries, custom notebooks on cron, custom alerting via email. There was no central view. I sat down with three teams to understand what they all actually wanted, and I designed a config-driven framework.
>
> The interface was a YAML file per model. Data scientist defines: model_id, the features to monitor, the drift metric per feature — PSI for continuous, chi-squared for categorical, MMD for embeddings — thresholds, schedule, alert channels. Reference window typically 30-to-60 days ago, current window 0-to-1 day ago.
>
> CI registered the config to a metadata DB and auto-generated three things. One — an Airflow DAG that ran daily. Two — a Datadog dashboard built via the Datadog API with PSI line charts per feature, a heatmap of drift across features over time, prediction distribution histograms, latency and error rate panels, and an annotation track for retraining and deploy events. Three — Datadog anomaly-detection monitors with Slack and PagerDuty integrations.
>
> The daily Airflow job pulled both windows from Snowflake — predictions and features were already there because of our analytics pipeline, so we leveraged Snowflake's compute rather than moving data. Computed drift per feature in pandas. Pushed metrics to Datadog via the custom-metrics API. Annotated the timeline with any retraining events from the last 24 hours.
>
> Alerting had two tiers. Tier two was raw drift exceeding threshold for two consecutive runs — Slack notification, ticketed for investigation. Tier one was drift plus a degraded performance proxy — confidence drift or prediction distribution drift — that paged the on-call MLE.
>
> The result was that eight production models on IHS adopted it, drift monitoring became a CoE standard, and the average time-to-detect for a real model issue dropped from weeks to a day or two. The thing I'm most proud of is that it required almost no work from the data scientists — thirty lines of YAML and they got production-grade monitoring."

---

## 14.17 Master Q&A — monitoring and drift interview spread

**Q1. Walk me through the difference between data drift, concept drift, and label drift.**
> Data drift means P of X changes — the input distribution shifts but the input-output relationship is unchanged. Concept drift means P of Y given X changes — the same input now produces a different correct output, the relationship itself moved. Label drift means P of Y changes — class balance shifts. Data drift is easy to detect because you don't need labels but it doesn't always hurt the model since the drifted feature might be unimportant. Concept drift always hurts but it's hard to detect because you need ground truth which often arrives weeks late. The standard mistake is treating raw data drift as automatically actionable; in production I tie retraining to actual quality metrics or proxies, not raw drift.

**Q2. KS versus PSI versus Wasserstein — when each?**
> KS for small samples where I want statistical rigor with a p-value; the catch is that at production scale with millions of samples even tiny shifts produce p less than 0.05, so the p-value becomes useless and I rely on the D-statistic effect size instead. PSI for production-scale tabular features because it has interpretable thresholds — under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 significant — and those thresholds are the finance industry standard. Wasserstein and MMD for high-dimensional or embedding distributions where bin-based PSI doesn't generalize. For LLM apps with embedding monitoring, Wasserstein is the modern choice.

**Q3. Give me the PSI formula and walk through a worked example.**
> PSI equals the sum over k bins of P_i minus Q_i times the natural log of P_i over Q_i, where P is current and Q is reference. The intuition is it's a symmetrized KL divergence binned. Worked example — credit score binned into five buckets. Reference distribution Q is 10, 20, 40, 20, 10 percent across the bins. If current P is 15, 25, 30, 20, 10 percent, the contributions per bin are 0.020, 0.011, 0.029, 0, 0 — sum is 0.060. That's under 0.1, so stable. If next week P shifts to 30, 30, 20, 15, 5 percent, the contributions become 0.220, 0.041, 0.139, 0.014, 0.035 — sum is 0.449. That's well over 0.25, significant drift, time to investigate.

**Q4. What's the gotcha with KS test at production scale?**
> The gotcha is that p-values become useless. At a million samples, even a 0.5 percent real shift produces p much less than 0.05 because the test has so much statistical power. You'd alert constantly and your team would tune you out within a week. The fix is to ignore the p-value and use the D-statistic as an effect size, or just use PSI which has threshold-based interpretation that scales naturally. I usually report both — D for rigorous size, PSI for actionable threshold.

**Q5. Drift alerts fire daily but accuracy is stable. What do you do?**
> First, don't retrain reflexively — drift without performance degradation is noise. Second, switch from raw-feature drift to SHAP-importance-weighted drift — drift on unimportant features should not fire alerts. Third, increase the alerting hysteresis — single-window anomalies are noisy, sustained drift over multiple windows is signal. Fourth, tie retraining triggers to quality proxies rather than raw drift — confidence drift, prediction distribution drift, business KPI changes. The runbook for a drift alert should be "investigate," not "retrain." Drift is diagnostic; retraining is the cure for actual degradation.

**Q6. How do you monitor drift for text or embedding inputs?**
> Embed every prompt with a consistent embedding model — never change embedders without re-baselining. Sample embeddings into a reference set during training and a current set during production. Compute Wasserstein or MMD between the two distributions. For visualization, project both with UMAP and overlay them — Arize Phoenix does this beautifully. For LLM apps specifically I also track prompt length distribution, refusal rate which is the rate of "I can't help with that" responses, output length distribution, and cosine to the training-corpus centroid as an out-of-distribution signal.

**Q7. Labels arrive weeks late. How do you detect problems early?**
> Proxy signals. First, prediction distribution drift — if the predicted positive rate jumps from 5 to 15 percent overnight, something's off even if you can't yet measure accuracy. Second, confidence or entropy shifts — model getting less confident on average suggests OOD inputs. Third, business KPIs that correlate with model output — chargeback rate for fraud, conversion rate for recommendations. Fourth, deploy a challenger model in shadow mode and track agreement rate between champion and challenger; divergence is a pre-label proxy for concept drift. The combination of these gives you 60 to 70 percent of the signal you'd get from real labels.

**Q8. Evidently AI versus Datadog versus Arize — pick for Avrioc.**
> Evidently AI self-hosted is the conservative pick. Open source, runs on Postgres or whatever you already have, integrates cleanly with batch pipelines. Strong on flexible report generation. Less polished UI than Arize but the on-prem story for UAE data residency is excellent. Datadog if Avrioc is already a Datadog shop — agents work on-prem and the existing infrastructure is a force multiplier. Arize is best for LLM-heavy workloads with embedding visualization needs but it's SaaS-first which complicates UAE residency. My recommendation for a typical Avrioc setup is Evidently for the drift core, plus self-hosted Langfuse for any LLM observability needs.

**Q9. Closed-loop retraining trigger — how would you design it?**
> Combine three signals through a policy engine. Signal one is drift detection — PSI or KS on features, weighted by SHAP importance to the model. Signal two is delayed-label performance — actual AUC or F1 computed when ground truth arrives. Signal three is time cadence — minimum time since last retrain to prevent ping-ponging, maximum time since last retrain to catch slow drift. The policy fires only when at least two of the three are crossed, with hysteresis on each. Every trigger decision logs to an audit table — who triggered, what signals, what model resulted. In regulated GCC contexts there's always a human approval gate before production promotion; automated retrain-and-deploy is a compliance failure waiting to happen.

**Q10. Two percent accuracy drop overnight — drift or bug?**
> Triage in this order, and the order matters because data engineering issues are far more common than concept drift. First, check the data pipeline — feature freshness, schema changes, missing partitions, upstream system outages. About sixty percent of "model regressions" are actually broken pipelines. Second, week-over-week feature distribution diff — has any input feature shifted suddenly? Third, serving-code or library upgrade — did anyone deploy a new container? Fourth, model itself — is it the same registered version or did promotion happen unexpectedly? Only if all four are clean do I suspect concept drift. Rule out engineering before rule in modeling.

**Q11. How do you set monitoring alerts that don't cause fatigue?**
> Three rules. One — every alert has a runbook documenting what the on-call does when it fires; if I can't write that down, the alert shouldn't exist. Two — no alerts on expected seasonality; use anomaly detection or baselines instead of static thresholds. Three — trim aggressively; if an alert hasn't been actionable in thirty days, retire it. Specifically for drift alerts I use SHAP-importance-weighted drift instead of raw, require sustained breach over multiple windows rather than single-point spikes, and tier alerts as page versus ticket versus ignore. Page only on user-impacting issues, ticket on investigation needs, ignore on informational.

**Q12. What would you log per LLM request in production?**
> Request ID, tenant, user, session — the dimensions for slicing later. Prompt version ID linking to the registry. Prompt text and completion text, both PII-redacted. RAG context document IDs. Model name, system fingerprint, temperature. Token counts in and out. Cost in USD. Latency breakdown — TTFT, TPOT, end-to-end. User feedback when available. Eval scores from async LLM-judge sampling. Tool calls if any. The trace gets ingested into Langfuse or equivalent, with tail-based sampling because at high QPS full-trace logging is expensive — log all errors, all p99-slow, all low-confidence, plus a one-percent random sample for healthy traffic.

**Q13. Sampling strategies for high-QPS tracing?**
> Four patterns. Head-based — random one to five percent uniformly. Tail-based — log everything that errored, was slow, or scored low confidence; this is OpenTelemetry's gold standard. Per-tenant — premium tenants always logged for SLA reasons, free tenants sampled. Per-session — full session traces kept for some random percentage, useful for debugging multi-turn issues. In practice I combine tail-based plus per-tenant — it gives you all the interesting cases for premium customers without breaking the budget on free-tier traffic.

**Q14. Toxic output detected in production — incident response?**
> Immediate mitigation first. Enable strict output guardrails to block the specific failure mode. Flip the prompt to the last known-good version via the registry pointer. Rate-limit the affected tenant if it looks like targeted abuse. Communicate via status page if user-facing and the team Slack. Short-term fix — add the failing example to the eval set as a regression test, update guardrail rules to catch the specific pattern. Post-mortem — trace through prompt version history to find what changed, check for model-provider updates that might have shifted behavior, audit RAG context for poisoning. Always blameless, always documented action items, automated tests so the same failure can't recur silently.

**Q15. How do you quantify implicit feedback into quality metrics?**
> Aggregate per-prompt-version, per-model, per-tenant. Thumb-up rate is the obvious one but only works when the UI surfaces a thumb. Regenerate-clicked rate is a strong implicit dissatisfaction signal. Copy-button-clicked rate is implicit satisfaction. Conversion or task-completion rate ties to business outcome. Session-end-without-message rate is ambiguous — could be satisfied or could be giving up. Track all of these weekly and alert on significant week-over-week drops. The composite "quality score" I usually build is a weighted sum of these, calibrated against a small human-eval ground truth.

**Q16. Shadow traffic — what's it for in monitoring?**
> Shadow traffic mirrors production requests to a candidate model and logs both predictions without the candidate response reaching users. The agreement rate between champion and shadow is a pre-production proxy for concept drift — if a freshly-retrained model substantially disagrees with production on recent traffic, that's a strong signal something has shifted. It's also how I'd validate a new model before canary — if shadow agrees with production at 95 percent and disagrees on the cases where production was actually wrong, the new model is probably better. Cheap to run, valuable signal.

**Q17. System fingerprint — why monitor it?**
> Because model providers update models silently. You pin the model ID to gpt-4o-2024-11-20 thinking you've locked in behavior, but provider-side fingerprint changes can still alter responses subtly. Logging system fingerprint per request and alerting on changes catches this. When fingerprint changes, automatically re-run the golden eval set against the new fingerprint and alert if any metric drops more than two percent. This single discipline has saved me from at least three "the model started behaving weirdly" incidents over the years.

**Q18. How do you measure faithfulness of RAG answers in production?**
> Async LLM-judge on sampled traces. Sample one to two percent of production RAG responses. For each sample, send the question, the retrieved context, and the generated answer to a judge model — typically a stronger model than the production one — with a rubric: "Score 1 to 5 on whether the answer is supported by the context." Aggregate rolling p95 faithfulness score per prompt version. Alert on drops greater than 10 percent. RAGAS, TruLens, and Arize Phoenix all implement this pattern. The key is rotating the judge models periodically so you don't bias toward one judge's preferences.

**Q19. Langfuse versus LangSmith versus Phoenix — pick one.**
> Langfuse self-hosted for Avrioc. Open source, deploys on Postgres, has the strongest combination of trace ingestion, prompt registry, and eval framework, and deploys cleanly on-prem for UAE residency. LangSmith is LangChain-native and very polished but SaaS-first which complicates residency. Phoenix is open source from Arize, beautiful for embedding-drift visualization, but lighter on prompt management features. For a regulated production environment, Langfuse hits the sweet spot — production-grade, OSS, self-hostable, integrated.

**Q20. Define a "golden set" in LLM monitoring.**
> A curated set of 100 to 300 examples each containing a prompt, an expected output or rubric, and ideally metadata about why this example matters. Maintained by domain experts, treated as the canonical eval ground truth. Run on every prompt or model change in CI before merge — fail blocks the PR. Expanded continuously with real production failures, so the set evolves with the application. Run daily against production to detect regression. The golden set is the unit-test suite of LLM apps — without one you have no idea whether changes are improving or regressing the system.

**Q21. p99 latency spikes every 5 minutes on vLLM. Diagnose.**
> Several candidates. One — Prometheus scrape hitting a busy worker, contention on the metrics endpoint. Two — Python garbage collection pauses; check gc statistics. Three — KV-cache eviction storms when concurrency peaks fill the cache. Four — Kubernetes HPA re-evaluating and re-balancing pods every five minutes. Five — log flushes if logging is synchronous. Diagnostic: per-request traces correlated with GPU utilization from nvidia-smi dmon, container metrics from cAdvisor, and Prometheus scrape times. Usually the answer is either KV-cache pressure or GC; both have well-known fixes.

**Q22. Cost monitoring for LLM apps — implementation?**
> Gateway-centric. Every LLM call goes through LiteLLM or Portkey, which attaches metadata — user, tenant, feature flag — and computes the cost from the current rate card. Logs go to ClickHouse or BigQuery for fast aggregation. Grafana or Datadog dashboards for total spend, spend by model, spend by tenant, cache hit rate, average tokens per request. Alerts at 80 percent of daily or weekly budget. Per-tenant breakdown is essential because the cost-per-tenant signal often catches abuse — one customer is using 40 percent of compute, that's a billing or rate-limit issue.

**Q23. Data pipeline SLAs for ML — what should they cover?**
> Four SLAs minimum. Feature freshness — data has arrived within X hours of the source-of-truth update; alert on breach because stale features mean wrong predictions. Training data completeness — no missing partitions or schema-broken rows; this is what blocks training. Feature store read latency p99 — typically under 50 milliseconds for online serving; breach means the inference endpoint is going to time out. Drift on feature pipeline outputs versus source — catches silent transformation bugs. Every model has its own explicit data SLA documented in the model card and monitored via the same Datadog dashboards as the model itself.

**Q24. What's the RED method?**
> RED stands for Rate, Errors, and Duration — the three core metrics for service monitoring. Rate is requests per second. Errors is the percentage that fail. Duration is latency, typically tracked at p50, p95, and p99. It's a Brendan Gregg concept that became the standard for SRE-style observability. For ML services I extend it to RED-FP — Features, where I add drift metrics, and Predictions, where I add prediction distribution. Same dashboard, two more rows. Gives you the full picture of "is the system up, and is it right?"

**Q25. Stack recommendation for Avrioc UAE monitoring.**
> Self-hosted Langfuse on Postgres for LLM observability — traces, prompts, evals — runs in UAE region for residency. Evidently AI self-hosted for tabular and embedding drift on classical ML and RAG. Prometheus plus Grafana for system metrics on EKS — CPU, memory, GPU, latency. Loki for logs, Tempo for traces. AlertManager piping P1 alerts to PagerDuty, P2 to Slack. Custom drift utility on Snowflake plus Datadog if they have those — pattern from my ResMed work. Retention policies aligned with UAE PDPL — typically 90 days hot, 1 year cold for non-PII metrics, redacted PII for any user-visible logs. Everything runs on-prem or in me-central-1 region.

---

Continue to **[Chapter 15 — Resume Projects, Deep Dive](15_resume_deep_dive.md)**.
