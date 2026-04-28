# Chapter 10 — MLOps & LLMOps
## From training to production — the operational backbone

> This is a signature chapter for Sachin. Two and a half years building MLOps at ResMed, plus current TrueBalance work. Own this conversation. Lead with stories, not theory.

---

## 10.1 What MLOps actually is — the conversation that changed how I think

### 10.1.1 Why this exists

In 2018 a McKinsey survey found that around 87% of data science projects never made it to production. Models trained on someone's laptop, evaluated in a notebook, demoed to stakeholders, then never deployed. The reason wasn't usually the model — it was that nobody had figured out how to **operate** the model. Whose job is it when accuracy degrades? How do you retrain when the data shifts? How do you roll back a bad model deploy at 2 AM? How do you reproduce a model from two years ago when the auditor asks?

MLOps is the discipline that answers those questions. It's DevOps with two extra moving parts — **data** and **trained models** — each of which has its own lifecycle, its own versioning challenges, its own failure modes. If DevOps is "code → build → deploy → monitor," MLOps is the same loop but with three artifacts that all need to stay in sync.

### 10.1.2 The mental model — analogy

Think of a software engineer's job as **maintaining a recipe**. The recipe is the code. As long as the recipe is good, the dish comes out the same every time you cook.

A machine learning engineer's job is more like **running a restaurant**. Yes there's a recipe (code), but you also need fresh ingredients (data — different every day), a chef trained for this kitchen (model — needs retraining as menu evolves), and a tasting protocol (evaluation). When the broccoli is unusually bitter that week (data drift), the dish suffers even though the recipe never changed. MLOps is the kitchen management discipline that keeps all three in sync.

### 10.1.3 The three artifacts

```
   ┌───────────────────────────────────────────────────────────┐
   │                   MLOPS ARTIFACTS                         │
   └───────────────────────────────────────────────────────────┘

      CODE                    DATA                    MODEL
      ────                    ────                    ─────
   ┌───────────┐         ┌───────────┐         ┌───────────┐
   │ Git SHA   │         │ Dataset   │         │ Weights   │
   │ Container │         │ hash      │         │ + config  │
   │ image     │         │ Schema    │         │ + metrics │
   │ digest    │         │ version   │         │ MLflow    │
   └───────────┘         └───────────┘         │ registry  │
                                                └───────────┘
   Versioned in:         Versioned in:         Versioned in:
   - Git                 - DVC, LakeFS         - MLflow,
   - ECR/GCR             - S3 object           - W&B,
                          versioning           - SageMaker
                                                Model Registry

   Fails when:           Fails when:           Fails when:
   - Branch out of sync  - Schema breaks       - Drift untracked
   - Image tag mutable   - Pipeline stale      - Promotion manual
```

Every MLOps platform answers: how do we version, test, deploy, and roll back each of these three artifacts in a way that the data scientist can self-serve and the SRE can sleep at night?

---

## 10.2 The MLOps maturity ladder

This is a question Avrioc will probably ask. Have a clean answer.

```
   ┌───────────────────────────────────────────────────────────┐
   │                  MLOPS MATURITY LADDER                    │
   └───────────────────────────────────────────────────────────┘

   LEVEL 0  ─────  Manual notebooks
                   - Train in Jupyter
                   - Deploy by hand-copying weights
                   - No versioning, no monitoring
                   - "It works on Sonia's laptop"

   LEVEL 1  ─────  CI/CD for code, manual for data/model
                   - Code in Git, runs in Docker
                   - Training still manual
                   - No model registry
                   - One model, one deploy

   LEVEL 2  ─────  Automated training + registry
                   - Pipeline triggered on data arrival
                   - MLflow tracks experiments + models
                   - Manual promotion to production
                   - Basic latency/error monitoring

   LEVEL 3  ─────  Closed-loop with drift triggers
                   - Drift detection -> retraining triggers
                   - Automated canary + shadow deploys
                   - SHAP-weighted drift, not raw drift
                   - Datadog dashboards per model

   LEVEL 4  ─────  Full self-service platform
                   - Feature store with online + offline parity
                   - Multi-region, multi-tenant
                   - Eval-driven promotion
                   - Cost dashboards per model/tenant
```

**Sachin's pitch line:** "At ResMed I built the framework that took the IHS team from Level 1 to Level 3 — eight models in six months on a standardized pipeline. At TrueBalance I'm operating at Level 3 with real-time XGBoost on Lambda. For Avrioc I'd start by auditing where they sit and identifying the next-level upgrade."

---

## 10.3 The full ML lifecycle — the master diagram

```
         ┌─────────────────────────────────────────────────┐
         │              ML LIFECYCLE                       │
         └─────────────────────────────────────────────────┘

   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ 1. Data  │────▶│2. Feature│────▶│3. Train  │
   │ ingestion│     │ engineer │     │ pipelines│
   │          │     │          │     │          │
   │ Airflow  │     │ Spark    │     │ MLflow   │
   │ Fivetran │     │ dbt      │     │ Sagemaker│
   │ Airbyte  │     │ Feast    │     │ KubeFlow │
   └──────────┘     └──────────┘     └────┬─────┘
                                           │
                                           ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ 6.Monitor│◀────│5. Serve  │◀────│4.Register│
   │ + Retrain│     │ endpoint │     │  & Test  │
   │          │     │          │     │          │
   │ Evidently│     │ Sagemaker│     │ MLflow   │
   │ Datadog  │     │ vLLM     │     │ Eval set │
   │ Arize    │     │ Lambda   │     │ Canary   │
   └────┬─────┘     └──────────┘     └──────────┘
        │
        │ drift signal triggers retraining
        ▼
   loop back to Step 3
```

Every box on this diagram is a separate technology decision and a separate failure mode. A senior MLE earns their salary by knowing where the failure modes are and which combinations of tools work in which environments.

---

## 10.4 Reproducibility — the foundation everything else rests on

### 10.4.1 Why this matters

Two years from now an auditor walks in and asks "Show me byte-identical output from your March 2024 model." If you can't, you don't have MLOps. You have an artisan workshop that occasionally ships software.

In regulated domains — healthcare (Sachin's ResMed work), finance (TrueBalance), automotive (Avrioc) — this isn't theoretical. It's a compliance requirement.

### 10.4.2 What to pin

```
   ┌──────────────────────────────────────────────────────┐
   │  FULL REPRODUCIBILITY STACK                          │
   ├──────────────────────────────────────────────────────┤
   │                                                      │
   │  Code           ─▶ Git SHA (not branch name)         │
   │  Container      ─▶ Image digest (not tag)            │
   │  Dependencies   ─▶ pip freeze, not requirements.txt  │
   │  Data           ─▶ DVC hash or S3 version-id         │
   │  Hyperparams    ─▶ Hydra YAML committed              │
   │  Random seeds   ─▶ numpy + torch + cuda              │
   │  Hardware       ─▶ GPU type, CUDA version            │
   │  Framework      ─▶ pytorch==2.1.0+cu118              │
   │  Run metadata   ─▶ MLflow run_id                     │
   │                                                      │
   └──────────────────────────────────────────────────────┘
```

**The two-year reproducibility test:** pick any production run from two years ago. Can you reproduce its output byte-for-byte? If yes, you have real MLOps. If no, you're cosplaying.

### 10.4.3 Common mistakes

1. **Using docker tag `:latest` in production.** Tags are mutable. The image you ran today isn't the image you'll run tomorrow. Always pin to digest: `myapp@sha256:abc123...`.
2. **Storing `requirements.txt` instead of `pip freeze`.** `pandas>=1.5.0` is not a pin — pip might install 1.5.0 today and 2.1.3 next week.
3. **Not committing hyperparameters with the code.** A few months later nobody remembers the learning rate that won.

---

## 10.5 Experiment tracking — MLflow vs W&B vs SageMaker

### 10.5.1 The problem

Data scientists run hundreds of experiments. Without tracking, you can't answer "what was the configuration that got our best F1 last quarter?" or "did adding feature X actually help?" Experiment tracking turns ML research from a craft into engineering.

### 10.5.2 MLflow — the OSS workhorse

```
   ┌──────────────────────────────────────────────────────┐
   │                    MLFLOW                            │
   ├──────────────────────────────────────────────────────┤
   │                                                      │
   │  ┌──────────┐   ┌──────────┐   ┌──────────────┐     │
   │  │Tracking  │──▶│ Registry │──▶│ Model Serving│     │
   │  │ Server   │   │          │   │ (mlflow      │     │
   │  │          │   │ Stages:  │   │  models      │     │
   │  │ Logs to  │   │ None     │   │  serve)      │     │
   │  │ Postgres │   │ Staging  │   │              │     │
   │  │ + S3     │   │ Prod     │   │              │     │
   │  │          │   │ Archive  │   │              │     │
   │  └──────────┘   └──────────┘   └──────────────┘     │
   │                                                      │
   └──────────────────────────────────────────────────────┘
```

The MLflow API is simple: `mlflow.start_run()`, `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()`. Self-hosted on EC2 + RDS + S3 is the canonical setup for regulated environments. MLflow's model registry has four canonical stages — None, Staging, Production, Archived — and every endpoint queries the registry by stage so promotion = registry pointer flip.

### 10.5.3 Comparison

| Feature | MLflow | W&B | SageMaker MR |
|---------|--------|-----|--------------|
| License | Apache 2.0 OSS | Commercial SaaS | AWS managed |
| Self-host | ✅ (canonical) | Limited | ✗ |
| Sweeps | Limited | ✅ killer feature | ✗ |
| Polish | Functional | Best UI | Functional |
| Data residency | On-prem possible | US/EU/private | AWS regions |
| Best for | Regulated, on-prem | R&D speed | All-AWS shops |

For Avrioc UAE — likely data residency requirements — **MLflow self-hosted on UAE region or on-prem** is the conservative pick.

### 10.5.4 Resume tie-in

> "The MLOps framework I built at ResMed used MLflow self-hosted on EC2 with Postgres-backed tracking and S3-backed artifacts. Each data scientist's training job auto-registered the resulting model to MLflow with stage None, our CI ran the eval golden set against it, and on pass it auto-promoted to Staging. Production promotion was a manual approval gate per regulatory requirements — but the path from Staging to Production was a single button click that flipped the registry pointer and triggered a SageMaker endpoint update."

---

## 10.6 Feature stores — solving train-serve skew

### 10.6.1 Why this exists

The single most common production bug in ML systems: **the feature is computed differently in training than at serving**. Training pipeline computed `user_avg_purchase_30d` from a Spark batch over historical Snowflake data. Serving endpoint computes the same feature from a real-time DynamoDB lookup that uses a different time window or a different aggregation rule. Predictions diverge silently. The model "works" in offline eval and "is broken" in production.

A feature store solves this by making **the feature definition the single source of truth**. Both training and serving go through the same definition and read from the same logical store — just from different physical backends optimized for batch versus low-latency.

### 10.6.2 The architecture

```
   ┌──────────────────────────────────────────────────────────────┐
   │                    FEATURE STORE                             │
   │                                                              │
   │  ┌────────────────────────────────────────────────────┐      │
   │  │             FEATURE DEFINITIONS                    │      │
   │  │  user_avg_purchase_30d:                            │      │
   │  │    source: transactions                            │      │
   │  │    aggregation: AVG(amount)                        │      │
   │  │    window: 30 days                                 │      │
   │  │    entity: user_id                                 │      │
   │  └────────────────────────────────────────────────────┘      │
   │             │                          │                     │
   │             ▼                          ▼                     │
   │  ┌──────────────────┐        ┌──────────────────┐            │
   │  │  OFFLINE STORE   │        │  ONLINE STORE    │            │
   │  │                  │        │                  │            │
   │  │ S3 / Snowflake / │        │ Redis / Dynamo / │            │
   │  │ Parquet / Hive   │        │ Snowflake        │            │
   │  │                  │        │                  │            │
   │  │ Point-in-time    │        │ <50ms p99 lookup │            │
   │  │ correct joins    │        │                  │            │
   │  │ for training     │        │ Real-time serving│            │
   │  └──────────────────┘        └──────────────────┘            │
   │             ▲                          ▲                     │
   └─────────────┼──────────────────────────┼─────────────────────┘
                 │                          │
        ┌────────┴───────┐         ┌────────┴───────┐
        │ Training jobs  │         │ Inference API  │
        │ (Spark, etc.)  │         │ (low latency)  │
        └────────────────┘         └────────────────┘
```

The two-store pattern is the heart of it. Offline store is huge but slow — perfect for training jobs that scan terabytes. Online store is small but fast — perfect for sub-50-ms lookups. Both are populated from the same source-of-truth feature pipeline so they can never disagree.

### 10.6.3 Worked example — point-in-time correctness

Suppose you're training a churn model on December 1. For user U who churned on November 15, what value of `user_avg_purchase_30d` should you use?

**The wrong answer:** `AVG(transactions.amount) WHERE user_id = U AND ts > today() - 30`. You'd be using December's average to predict November's churn. **Data leakage.**

**The right answer (what feature stores enforce):** `AVG(transactions.amount) WHERE user_id = U AND ts BETWEEN '2024-11-15' - 30 days AND '2024-11-15'`. You compute the feature value as it would have been at the time of the label. This is **point-in-time correctness** and it's a major reason to use a feature store rather than rolling your own.

### 10.6.4 Products

- **Feast** — open-source, Kubernetes-friendly, the OSS standard
- **Tecton** — managed enterprise version, built by Uber Michelangelo alumni
- **Databricks Feature Store** — if you live in Databricks
- **AWS SageMaker Feature Store** — managed, integrated with the rest of SageMaker
- **Snowflake Feature Store** — Sachin's ResMed setup, leverages Snowflake's compute

### 10.6.5 When it's overkill

- Single model, fewer than 10 features
- Pure offline batch inference
- One team, one feature definition

### 10.6.6 When it pays off

- Multiple teams sharing features
- Sub-50-ms online serving
- Compliance audits requiring feature lineage
- Risk of train-serve skew is unacceptable

### 10.6.7 Resume tie-in

> "At ResMed we used Snowflake's feature store as our online and offline backend. The IHS team had eight models in production sharing about thirty common features — without the feature store we'd have had eight different versions of `patient_avg_session_duration_7d`, computed slightly differently, drifting silently. The feature store made the definition canonical, the offline join logic point-in-time correct, and the online lookup fast enough — about 30 ms p99 — for our SageMaker real-time endpoints."

---

## 10.7 CI/CD for ML — beyond just code

### 10.7.1 What's different from software CI/CD

Standard software CI: pull request triggers tests, tests pass, deploy. For ML you need:

- Tests on **code** (unit tests for feature engineering, training utilities)
- Tests on **data** (schema, freshness, distribution sanity)
- Tests on **model quality** (eval against golden set, must beat baseline)
- Tests on **serving behavior** (latency budget, output schema)
- Promotion **stages** (Staging → Canary → Production)

### 10.7.2 The pipeline diagram

```
   ┌──────────────────────────────────────────────────────────────┐
   │                    ML CI/CD PIPELINE                         │
   └──────────────────────────────────────────────────────────────┘

   PR opened
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Unit tests (feature eng, utils)  │
      │    └──────────────────────────────────┘
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Data validation (Great Expectations)
      │    └──────────────────────────────────┘
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Smoke train (1 epoch, tiny set)  │
      │    └──────────────────────────────────┘
      │
   Merge to main
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Full training on latest data     │
      │    └──────────────────────────────────┘
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Eval golden set                  │
      │    │ Must beat prod baseline by >0.5% │
      │    └──────────────────────────────────┘
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Bias / fairness checks           │
      │    │ Per-subgroup AUC gap < threshold │
      │    └──────────────────────────────────┘
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Register in MLflow (Staging)     │
      │    └──────────────────────────────────┘
      │
   Manual approval
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Shadow deploy                    │
      │    │ Mirror prod traffic, log only    │
      │    └──────────────────────────────────┘
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Canary 10% traffic, 24h          │
      │    │ Monitor latency + error          │
      │    └──────────────────────────────────┘
      │
      ├──▶ ┌──────────────────────────────────┐
      │    │ Full rollout                     │
      │    │ Monitor drift + perf metrics     │
      │    └──────────────────────────────────┘
      │
      └──▶ Rollback path: flip MLflow registry pointer
```

### 10.7.3 Tools

- **GitHub Actions / GitLab CI / Jenkins / CodePipeline** for orchestration
- **Argo Workflows / Airflow / Prefect** for scheduled training
- **Argo CD / Flux** for GitOps deploy to Kubernetes
- **Terraform / CloudFormation / CDK** for infrastructure-as-code
- **Great Expectations / Deequ** for data validation

### 10.7.4 Quality gates

Every ML CI pipeline needs hard gates that block bad models from reaching production. The minimum:

| Gate | Check | Fail action |
|------|-------|-------------|
| Baseline metric | New model AUC ≥ prod - 0.5% | Block merge |
| Latency budget | p99 < target on canary | Auto-rollback |
| Fairness | Per-subgroup AUC gap < threshold | Block merge |
| Schema | Predictions match contract | Block merge |
| Canary regression | No metric drops 5%+ in canary 24h | Auto-rollback |

### 10.7.5 Resume tie-in

> "At Tiger Analytics I built CI/CD pipelines that automated retraining based on drift signals. The pipeline ran weekly: pulled latest data from Snowflake, computed feature drift via PSI, and if PSI exceeded 0.25 on important features it triggered a retrain job. The retrained model went through eval against a golden set; if it beat the production model by at least 0.5% AUC, it was registered in MLflow Staging and a Slack notification went to the on-call engineer for manual promotion to canary. End-to-end the loop closed in about three hours."

---

## 10.8 Model registry — the traffic light

The registry is the **source of truth** for what's running where.

```
   ┌────────────────────────────────────────────────────┐
   │              MLFLOW MODEL REGISTRY                 │
   └────────────────────────────────────────────────────┘

   None  ─────▶  Staging  ─────▶  Production  ─────▶  Archived
     │             │                  │
     │             │                  │
   newly         passed CI,         promoted to      retired
   trained,      awaiting           prod traffic     no longer
   awaiting      manual                              served
   eval          approval

   Each transition logged: who, when, why, MLflow run_id
```

Production endpoints query registry by stage alias — `models:/clinical-rag/Production` — and the actual model artifact behind that alias can change without endpoint redeploy.

**Rollback** in this model: change the alias pointer back to the previous version. Atomic, seconds, zero-downtime.

---

## 10.9 LLMOps — what's actually different

### 10.9.1 The shift in mental model

Classical MLOps optimized for **training pipelines**. You trained models, you cared about training data quality, you cared about reproducible training runs. Most LLM consumers in 2026 don't train LLMs — they integrate them, prompt them, and put guardrails around them.

LLMOps shifts the focus from training pipelines to **prompt engineering, evaluation, observability, and cost**.

```
   ┌─────────────────────────────────────────────────────────┐
   │           MLOPS  vs  LLMOPS — primary concerns          │
   ├─────────────────────────────────────────────────────────┤
   │                                                         │
   │   MLOps                          LLMOps                 │
   │   ─────                          ──────                 │
   │   Training data pipelines        Prompts as code         │
   │   Model registry (artifacts)     Prompt registry        │
   │   Feature store                  Vector store / RAG     │
   │   Drift on features              Drift on prompts/output│
   │   Accuracy / AUC                 LLM-judge / RAGAS      │
   │   Per-model retraining           Per-prompt A/B testing │
   │   Latency budget                 Cost / token budget    │
   │   Fairness audits                Guardrails (PII, jail) │
   │                                                         │
   │   (still relevant for fine-tuning)                      │
   └─────────────────────────────────────────────────────────┘
```

### 10.9.2 The LLMOps stack

```
   ┌──────────────────────────────────────────────────────────────┐
   │                    APPLICATION LAYER                         │
   │  ┌──────────────────────────────────────────────────────┐    │
   │  │  FastAPI / Next.js + LangGraph orchestration         │    │
   │  │  Guardrails wrapper (input + output validation)      │    │
   │  └──────────────────────────────────────────────────────┘    │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    LLM GATEWAY                               │
   │  ┌──────────────────────────────────────────────────────┐    │
   │  │  LiteLLM / Portkey                                   │    │
   │  │  - Routing across providers                          │    │
   │  │  - Auth + rate limit                                 │    │
   │  │  - Semantic caching                                  │    │
   │  │  - Per-tenant cost tracking                          │    │
   │  └──────────────────────────────────────────────────────┘    │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    OBSERVABILITY                             │
   │  ┌──────────────────────────────────────────────────────┐    │
   │  │  Langfuse / Phoenix / Helicone                       │    │
   │  │  - Trace ingestion                                   │    │
   │  │  - Prompt registry + versioning                      │    │
   │  │  - Eval scores (faithfulness, relevance)             │    │
   │  │  - Cost dashboards                                   │    │
   │  └──────────────────────────────────────────────────────┘    │
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    MODEL LAYER                               │
   │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
   │  │ Anthropic  │  │  OpenAI    │  │ Self-hosted│              │
   │  │ Claude     │  │  GPT-4o    │  │ Llama vLLM │              │
   │  └────────────┘  └────────────┘  └────────────┘              │
   └──────────────────────────────────────────────────────────────┘
```

---

## 10.10 Prompt management — treating prompts as code

### 10.10.1 Why this matters

In a real LLM application, the prompt is the model's behavior. Change the prompt, change the product. Yet I've watched teams put prompts inside Python f-strings deep in the codebase, with no versioning, no eval scores attached, no rollback story. When something breaks in production at 3 AM, nobody knows what prompt is actually running.

Prompts are code. Treat them like code.

### 10.10.2 Three patterns

```
   ┌────────────────────────────────────────────────────┐
   │  PATTERN 1: Inline in code (DO NOT DO THIS)        │
   ├────────────────────────────────────────────────────┤
   │  prompt = f"You are a clinical assistant..."        │
   │  response = client.completions.create(prompt=...)   │
   │                                                     │
   │  Problems: not versioned independently of code,    │
   │  rollback requires redeploy, no eval linkage       │
   └────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────┐
   │  PATTERN 2: YAML in git (acceptable)               │
   ├────────────────────────────────────────────────────┤
   │  prompts/clinical_rag_v3.yaml in git               │
   │  Loaded at startup or per request                  │
   │                                                     │
   │  Better: versioned independently of code,          │
   │  rollback via git, but no runtime A/B,             │
   │  no eval-score linkage                             │
   └────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────┐
   │  PATTERN 3: Prompt registry (best for production)  │
   ├────────────────────────────────────────────────────┤
   │  Langfuse / PromptLayer / Helicone                 │
   │  Runtime-loadable, versioned, A/B-able,            │
   │  eval scores attached per version,                 │
   │  rollback = pointer flip                           │
   └────────────────────────────────────────────────────┘
```

### 10.10.3 The canonical prompt object

```yaml
name: clinical_rag_system
version: v3.2
model: claude-sonnet-4-7
temperature: 0.2
max_tokens: 1024
template: |
  You are a clinical assistant. Answer using only the
  provided context. Cite sources with [N] brackets.
  ...
linked_eval_run: ragas_run_2026_04_25
last_eval_score: 0.89
created_by: sachin@truebalance
status: production
```

Every LLM call logs the `prompt_version_id` to the trace. Now you can answer: "in the past hour, which prompt version was each request using, and what was the average user thumbs-up rate per version?"

### 10.10.4 Tools

- **Langfuse** — OSS, self-hostable, integrated traces + prompts + evals
- **PromptLayer** — SaaS, prompt-focused
- **Helicone** — proxy-based, single-header integration
- **LangSmith** — LangChain's native, SaaS

### 10.10.5 Resume tie-in

> "On the IHS clinical RAG chatbot at ResMed we used Langfuse self-hosted as both our prompt registry and trace store. The system prompt went through about a dozen iterations over four months — each version registered, eval-scored against a hundred-sample golden set of clinical questions with rubric grading. When we shipped a new version, we ran a 24-hour shadow deploy with both versions and compared faithfulness scores via async LLM-judge before promoting."

---

## 10.11 LLM evaluation — three layers

### 10.11.1 Why LLM eval is different

For a classifier you have ground-truth labels and you compute F1. For an LLM, the "right answer" is often subjective, multiple answers can be correct, and the failure modes are more subtle (hallucination, bias, refusal, off-topic, length explosion). You need a richer eval stack.

### 10.11.2 The three layers

```
   ┌───────────────────────────────────────────────────────┐
   │  LAYER 1: OFFLINE EVAL (CI gate)                      │
   │  ─────────────────────────────────                    │
   │                                                       │
   │  Golden set: 100-300 (prompt, expected, rubric)       │
   │  triples curated by domain experts                    │
   │                                                       │
   │  Scored via:                                          │
   │  - LLM-as-judge (pairwise or rubric)                  │
   │  - Reference metrics (BLEU, ROUGE, exact-match)       │
   │  - Task-specific (SQL execution, JSON schema)         │
   │                                                       │
   │  Frameworks: RAGAS, DeepEval, Promptfoo,              │
   │              TruLens, Arize Phoenix                    │
   │                                                       │
   │  Fail this -> block PR                                 │
   └───────────────────────────────────────────────────────┘

   ┌───────────────────────────────────────────────────────┐
   │  LAYER 2: ONLINE A/B TESTING                          │
   │  ─────────────────────────                            │
   │                                                       │
   │  Route X% traffic to new prompt/model/pipeline        │
   │  Compare:                                             │
   │  - Task success proxy                                 │
   │  - User feedback (thumbs, regenerate)                 │
   │  - Cost & latency                                     │
   │                                                       │
   │  Requires statistical power (1-2 weeks per arm)       │
   └───────────────────────────────────────────────────────┘

   ┌───────────────────────────────────────────────────────┐
   │  LAYER 3: CONTINUOUS USER FEEDBACK                    │
   │  ──────────────────────────────                       │
   │                                                       │
   │  Every production trace gets a score:                 │
   │  - Thumbs up/down                                     │
   │  - Regenerate clicked                                 │
   │  - Conversion / task completion                       │
   │  - Async LLM-judge sampling                           │
   │                                                       │
   │  Aggregates -> dashboards -> alerts on drops          │
   └───────────────────────────────────────────────────────┘
```

### 10.11.3 LLM-as-judge — the workhorse pattern

You ask a strong model (Claude Opus, GPT-4) to score outputs from your production model on a rubric. This isn't perfect — judges are biased toward longer answers, toward their own style — but it's the only scalable eval for open-ended generation.

```
   ┌──────────────────┐
   │ Question, Answer,│
   │ Reference / ctx  │
   └────────┬─────────┘
            ▼
   ┌──────────────────────────────────────────┐
   │ Judge prompt:                            │
   │ "Score this answer 1-5 on:               │
   │   - Faithfulness to context              │
   │   - Relevance to question                │
   │   - Conciseness                          │
   │  Provide reasoning then a JSON score."   │
   └────────┬─────────────────────────────────┘
            ▼
   ┌──────────────────┐
   │ {faithfulness:4, │
   │  relevance:5,    │
   │  conciseness:3}  │
   └──────────────────┘
```

### 10.11.4 Eval-driven development

The engineering practice: every prompt or model change runs the golden set in CI before merge. Fail = blocked PR. Treat your LLM application like software with unit tests.

---

## 10.12 Guardrails — input and output safety

### 10.12.1 Why guardrails

LLMs can leak PII, generate toxic content, follow jailbreak instructions, hallucinate citations, output malformed JSON. Guardrails are the input and output validators that catch these before they reach the user.

### 10.12.2 The architecture

```
        ┌──────────────────────────────────────────────────────┐
        │                  USER REQUEST                        │
        └─────────────────────────┬────────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │  INPUT GUARDRAILS (run in parallel with LLM call)    │
        │  ─────────────────                                   │
        │  - PII detection (Presidio, Comprehend)              │
        │  - Jailbreak detection (Rebuff, NeMo)                │
        │  - Topic restriction                                 │
        │  - Profanity / toxicity                              │
        └─────────────────────────┬────────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │                LLM CALL                              │
        └─────────────────────────┬────────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │  OUTPUT GUARDRAILS                                   │
        │  ──────────────                                      │
        │  - Hallucination detection (faithfulness)            │
        │  - PII leak detection                                │
        │  - JSON schema validation                            │
        │  - Brand/tone check                                  │
        └─────────────────────────┬────────────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────────┐
        │  IF ANY GUARDRAIL FAILS:                             │
        │   - Block response                                   │
        │   - Return canned safety message                     │
        │   - Log incident                                     │
        │   - Increment alert counter                          │
        └──────────────────────────────────────────────────────┘
```

### 10.12.3 The latency trick

Run input guardrails **in parallel with** the LLM call, not before. If the input is fine, the LLM result is already ready. If the input fails, you abort and return the safety message. This hides the guardrail latency for the 99% of requests that are clean.

### 10.12.4 Tools

- **NVIDIA NeMo Guardrails** — flexible, rule + model-based, Colang DSL
- **Guardrails AI** — OSS, schema-centric, JSON validation focus
- **LlamaGuard** (Meta) — open-source safety classifier
- **AWS Bedrock Guardrails** — managed service in Bedrock
- **OpenAI Moderation API** — simple toxicity classifier

---

## 10.13 Observability for LLM apps

### 10.13.1 The four pillars

Standard software observability is three pillars: metrics, logs, traces. For LLMs add a fourth: **eval scores**.

```
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │   METRICS   │  │    LOGS     │  │   TRACES    │  │    EVALS    │
   │             │  │             │  │             │  │             │
   │ Prometheus  │  │ Loki        │  │ Tempo       │  │ Langfuse    │
   │ Datadog     │  │ CloudWatch  │  │ Jaeger      │  │ RAGAS       │
   │             │  │             │  │             │  │ Phoenix     │
   │ - QPS       │  │ - Errors    │  │ - Per-req   │  │ - Faithful- │
   │ - p99 lat   │  │ - PII       │  │   span      │  │   ness      │
   │ - Error %   │  │   redacted  │  │ - LLM call  │  │ - Relevance │
   │ - Tokens    │  │             │  │   tree      │  │ - Toxicity  │
   │ - Cost      │  │             │  │             │  │             │
   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

### 10.13.2 What to log per LLM request

```
{
  "request_id": "uuid",
  "tenant_id": "acme-corp",
  "user_id": "user-42",
  "session_id": "sess-1234",

  "prompt_version_id": "clinical_rag_v3.2",
  "system_prompt": "[redacted PII]",
  "user_prompt": "[redacted]",
  "rag_context_doc_ids": [101, 245, 378],

  "model": "claude-sonnet-4-7",
  "system_fingerprint": "fp_abc123",
  "temperature": 0.2,

  "response": "[redacted]",
  "tool_calls": [...],

  "tokens_in": 4521,
  "tokens_out": 312,
  "cost_usd": 0.0142,

  "ttft_ms": 280,
  "e2e_ms": 1840,

  "user_feedback": null,  // populated async
  "eval_scores": {        // populated async
    "faithfulness": 0.92,
    "relevance": 0.87
  }
}
```

### 10.13.3 Sampling at high QPS

At 1000 QPS, logging every trace is expensive. Sample strategies:

- **Head-based** — random 1-5% uniformly
- **Tail-based** — log all errors, all slow requests, all low-confidence
- **Per-tenant** — premium tenants always logged, free tenants sampled
- **Per-session** — full session traces for some random %

OpenTelemetry's tail-based sampling is the gold standard for production LLM observability.

---

## 10.14 A/B testing LLMs

### 10.14.1 Why this is different from classic A/B

Classic A/B has ground truth — clicks, conversions, revenue. LLM A/B has noisy proxies. Users don't always tell you when an answer was bad. You need to combine implicit signals (regenerate-clicks, session length, copy-button) with async LLM-judge scoring on samples.

### 10.14.2 The progressive rollout pattern

```
   Stage 1: SHADOW (0% user-facing)
   ────────────────────────────────
   ┌─────────────┐
   │ Production  │───▶ user
   │ model       │
   └──────┬──────┘
          │ mirror traffic
          ▼
   ┌─────────────┐
   │ Candidate   │ log only, no response
   │ model       │
   └─────────────┘

   Stage 2: CANARY (5-10% user-facing)
   ───────────────────────────────────
   ┌─────────────┐
   │ Production  │ 90% ───▶ user
   └─────────────┘
   ┌─────────────┐
   │ Candidate   │ 10% ───▶ user
   └─────────────┘
   Monitor for 24-48 h: latency, error rate, feedback rate

   Stage 3: PROGRESSIVE (10 -> 25 -> 50 -> 100%)
   ─────────────────────────────────────────────
   Each step: monitor for X hours, auto-rollback on metric drop

   Stage 4: FULL ROLLOUT
   ─────────────────────
   Old version archived but reachable for instant rollback
```

### 10.14.3 Metrics to compare

| Metric | What it tells you |
|--------|-------------------|
| TTFT (time to first token) | UX responsiveness |
| TPOT (time per output token) | Steady-state throughput |
| E2E latency | Total user wait |
| Win rate (LLM-judge pairwise) | Subjective quality |
| Refusal rate | Over-cautious flag |
| Output length distribution | Verbosity drift |
| Token cost per request | Economics |
| Thumbs-up rate | User satisfaction |
| Regenerate rate | Implicit dissatisfaction |

---

## 10.15 Cost tracking — $/1M tokens dashboards

### 10.15.1 The cost formula

```
cost_per_request = (prompt_tokens × in_rate)
                 + (completion_tokens × out_rate)
                 + cached_tokens × cache_rate
```

Each provider has different rates per model. A gateway like LiteLLM or Portkey logs every call with metadata (user, tenant, feature) plus the cost computed from current rate cards. Aggregate to ClickHouse or BigQuery, dashboard in Grafana or Datadog.

### 10.15.2 Cost levers (ordered by typical impact)

1. **Model cascade** — route easy queries to Haiku/Nova, hard to Sonnet/Opus. 40-60% reduction on real workloads.
2. **Prompt caching** (Anthropic, OpenAI, Bedrock) — 90% discount on cached tokens. Massive for long system prompts.
3. **Semantic caching** (GPTCache + Redis) — dedupe semantically-similar queries. Saves entire LLM call.
4. **Batch API** — 50% discount for async workloads.
5. **Self-hosting top 20%** — for highest-volume prompts, break-even ~1M tokens/hour.
6. **Output shortening** — structured schemas, prose discipline.
7. **Prompt optimization** — fewer few-shot examples, more concise instructions.

### 10.15.3 Cost dashboard structure

```
   ┌──────────────────────────────────────────────────────┐
   │  COST DASHBOARD (Grafana over ClickHouse)            │
   ├──────────────────────────────────────────────────────┤
   │                                                      │
   │  Total spend today: $4,231 / $5,000 budget           │
   │  Spend rate: $176/hour (alert if > $250)             │
   │                                                      │
   │  By model:                                           │
   │   Sonnet-4   $2,815  (66%)                           │
   │   Haiku      $912    (22%)                           │
   │   Opus-4     $504    (12%)                           │
   │                                                      │
   │  By tenant (top 5):                                  │
   │   acme       $1,402  (33%)                           │
   │   foobar     $887    (21%)                           │
   │   ...                                                │
   │                                                      │
   │  Cache hit rate: 47% (target > 30%)                  │
   │  Avg tokens/req: in=4521, out=312                    │
   │                                                      │
   └──────────────────────────────────────────────────────┘
```

---

## 10.16 Resume tie-in — Sachin's signature MLOps stories

### 10.16.1 The IHS platform story (use this as the system-design opening)

> "At ResMed I orchestrated the MLOps framework on the IHS platform that took the team from manual deploys to shipping eight production models in six months. The core insight was that data scientists were each rebuilding the same scaffolding — training scripts, eval harnesses, deploy configs, monitoring dashboards — slightly differently. So I standardized it.
>
> The data scientist's interface was a single `model_config.yaml` — about thirty fields covering data sources, training params, eval thresholds, deploy targets, monitoring config. CI parsed that config, built a SageMaker training container with a pinned image digest, ran the training in a CodeBuild job, registered the model to MLflow self-hosted on EC2, ran the eval golden set, and on pass auto-promoted to MLflow Staging. A Slack notification went to the on-call MLE for production promotion approval.
>
> Production deploy was GitOps — the framework wrote a Kubernetes InferenceService manifest pointing at the registry alias and committed it to a deploy repo, ArgoCD picked it up, and SageMaker rolled out a new endpoint version with 10% canary traffic for 24 hours.
>
> Monitoring was the part I'm most proud of. I built a utility that took a YAML config — `model_id, features, thresholds, schedule` — and auto-generated a per-model Datadog dashboard with KS and PSI drift charts, prediction distribution plots, anomaly detection on drift scores, and annotation tracks for retraining events. Snowflake was the compute layer; Datadog API was the dashboard target. Every team shipping models got drift monitoring for free, with no work from them.
>
> End-to-end, a data scientist could go from notebook to production endpoint with a config file and an approval click. That's the framework."

### 10.16.2 The clinical RAG LLMOps story

> "On the same platform I shipped a clinical RAG chatbot for IHS users. The full LLMOps stack: Langfuse self-hosted on Postgres for prompt registry, traces, and eval scores. LiteLLM as the gateway for routing between Bedrock Claude and a self-hosted Llama-3 fallback. Guardrails using NeMo Guardrails for jailbreak detection on input, faithfulness LLM-judge async on output. RAGAS golden set running daily in CI — about 150 clinical Q&A pairs with rubrics. A/B testing prompt versions via Langfuse's tag system, comparing thumb-up rates and async faithfulness scores. Cost tracking per tenant in ClickHouse, dashboarded in Grafana, alerting at 80% of weekly budget. That stack ran for fourteen months without a P0 incident."

### 10.16.3 The TrueBalance real-time XGBoost story

> "Currently at TrueBalance I run a real-time XGBoost model on Lambda for credit risk scoring. p99 latency budget is 500 ms; we're at 320 ms steady state. The model itself is small — a few hundred trees — so most of the latency is feature lookup from DynamoDB. I built the deploy pipeline using SAM CDK with versioned Lambda aliases, traffic-shifting deploys for canary, and CloudWatch metrics piped to Datadog for dashboarding. Drift monitoring runs nightly via a separate Lambda that pulls the day's predictions from S3, computes PSI per feature versus a 30-day reference window, and pushes to Datadog. Alert at PSI > 0.25 on any of the top-five SHAP-importance features."

---

## 10.17 Master Q&A — MLOps and LLMOps interview spread

**Q1. What's actually different about ML CI/CD compared to standard software CI/CD?**
> Three artifacts instead of one. In standard software CI you version code; in ML CI you version code, data, and trained models, and they all need to stay consistent. The pipeline runs different gates — data validation with Great Expectations, smoke training to catch broken pipelines, eval against a golden set that must beat the production baseline, fairness checks per subgroup, and canary or shadow deploys before full rollout. Promotion is tied to model registry stages — None to Staging to Production — not to Docker tags. And rollback is registry-pointer flip rather than redeploy. The mindset shift is that the model itself is a versioned artifact, not just a build output.

**Q2. MLflow versus Weights and Biases — which would you pick for Avrioc?**
> For Avrioc, where data residency in the UAE is likely a hard requirement, I'd pick MLflow self-hosted. MLflow is Apache-licensed, runs on a Postgres-plus-S3 backend you control, and has the canonical model registry pattern with None-Staging-Production-Archived stages. W&B has a much better UI, better hyperparameter sweeps, and richer dashboards — but it's SaaS-first and on-prem deployment is more limited. For fast-moving R&D teams in non-regulated environments W&B is fantastic; for regulated GCC contexts, MLflow self-hosted on a UAE-region EC2 plus RDS plus S3 is the conservative pick. SageMaker Model Registry is also fine if they're all-in on AWS.

**Q3. When is a feature store overkill, and when is it essential?**
> Overkill when you have one model, fewer than ten features, and pure offline batch inference — the operational overhead exceeds the benefit. Essential when you have multiple models sharing features, need sub-fifty-millisecond online lookups, or have compliance audits that require feature lineage tracking. The single biggest reason to adopt one is preventing train-serve skew — when the same feature is computed differently in batch versus online, your model breaks silently in production. A feature store enforces a single feature definition with point-in-time correct training joins and matching online serving.

**Q4. Walk me through how you'd prevent training-serving skew.**
> Three layers. First, share the feature transformation code — both the offline pipeline and the online serving call the same `compute_features()` function, ideally packaged in a shared library. Second, use a feature store that enforces single-source-of-truth definitions for any feature consumed by more than one model. Third, monitor feature distributions continuously — log production feature values to a metrics store, compare distributions versus training data via PSI or KS, alert on divergence. The first two prevent skew; the third catches it when the prevention fails.

**Q5. How do you guarantee a model is reproducible two years later?**
> Pin everything. Container image by digest, not by tag. Dataset by hash via DVC or LakeFS, not by S3 key alone. Git SHA for code, full pip-freeze output for dependencies, exact CUDA and driver versions for hardware, random seeds for numpy and torch and CUDA, all hyperparameters in a Hydra YAML committed alongside the code. All of that gets logged into the MLflow run metadata. The two-year test is real — pick any production run from two years ago and reproduce its output byte-for-byte. If you can't, you don't have MLOps yet.

**Q6. Walk me through your model promotion workflow.**
> PR triggers training in CI, model registers to MLflow with stage None. CI runs the eval golden set; if it beats the production baseline by at least 0.5% on the primary metric, it auto-promotes to Staging. The on-call MLE gets a Slack notification, reviews the eval report, and approves promotion to Production. GitOps takes over — the registry pointer change triggers Argo CD or a SageMaker endpoint update. Production starts at 10% canary traffic for 24 hours; if latency, error rate, and any quality proxies hold, it goes to 100%. Rollback at any stage is a registry pointer flip — atomic, zero-downtime.

**Q7. What's the actual difference between LLMOps and MLOps?**
> LLMOps refocuses MLOps from training pipelines to integration and serving. Most LLM consumers don't train the model — they integrate someone else's model — so the operational concerns shift. Prompt versioning and registry replace feature engineering. Prompt registries replace feature stores. Eval-as-code with LLM-judge replaces accuracy metrics. Guardrails for PII and jailbreak replace fairness audits. Cost tracking per token replaces cost tracking per training run. The fundamentals — observability, A/B testing, gradual rollout, rollback paths — are the same; the artifacts and metrics differ.

**Q8. How do you version prompts in production?**
> Prompt registry. Langfuse, PromptLayer, or Helicone. Never inline them in Python code where they can't be rolled back without a deploy. Every prompt has a semantic version, a model and temperature it's tested against, and a linked eval-set score. At runtime the application loads the prompt by name and active version pointer; the trace logs which version was used. To roll back, flip the version pointer. To A/B test, route traffic between two version aliases. This is exactly the same pattern as model registries — prompts are versioned artifacts with environment stages.

**Q9. How would you implement guardrails?**
> Two layers, parallel to the LLM call. Input guardrails check for PII, prompt injection, jailbreak attempts, and topic restrictions — running concurrently with the LLM call so they don't add latency for the 99% of requests that are clean. Output guardrails check for hallucination via faithfulness scoring, PII leakage, JSON schema compliance, and brand-tone violations. If any guardrail fails, abort the response and return a canned safety message, log the incident, and increment an alert counter. Tools: NeMo Guardrails for the rule engine, LlamaGuard or Bedrock Guardrails for the safety classifier, Guardrails-AI for schema validation. The architecture matters more than the tool — input parallel to LLM, output sequential, fail-fast on either.

**Q10. How do you track LLM costs?**
> Gateway-centric architecture. Every LLM call goes through LiteLLM or Portkey, which attaches metadata — user ID, tenant, feature flag — and computes the cost from current provider rate cards. Logs go to ClickHouse or BigQuery for fast aggregation. Grafana dashboards for total spend, spend by model, spend by tenant, cache hit rate. Alerts at 80% of daily or weekly budget. The biggest cost levers are model cascading — route easy queries to cheap models — prompt caching with Anthropic or OpenAI's caching APIs, and semantic caching for FAQs.

**Q11. How do you detect prompt regression when a model provider updates their model?**
> Three defenses. One — pin model versions, use dated model IDs like `gpt-4o-2024-11-20` rather than `gpt-4o-latest`. Two — monitor system_fingerprint per trace and alert on changes; even pinned model IDs can have silent updates. Three — run the golden eval set daily in CI and alert when any metric drops more than two percent. For any new model version, A/B against the old version with shadow traffic for at least a week before switching production traffic.

**Q12. Walk me through semantic caching.**
> Embed the incoming user query, search a vector store of past queries for similar ones, and if the cosine similarity exceeds a threshold — typically 0.95 to 0.99 — return the cached answer without calling the LLM. Implementation is GPTCache or a custom Redis-Vector setup. Massive cost win for FAQ or support bots. The risk is cache poisoning — two semantically-similar but actually-different questions mapping to the same cached wrong answer, like "how do I cancel?" and "how do I cancel my premium subscription?" Threshold tuning matters; 0.99 is safe for factual FAQ, 0.95 is risky. I'd also include user metadata in the cache key for personalized contexts.

**Q13. RAG versus fine-tuning — operational difference?**
> RAG updates near-instantly — reindex your knowledge base and the next query sees the new info. It's auditable because you can show citations. Cost is mostly retrieval plus inference. Fine-tuning bakes knowledge into the weights — better style adherence, no retrieval cost, but updating means a new training run. Risk of catastrophic forgetting. For about eighty percent of enterprise use cases — knowledge bases that change frequently, need source citations, and don't have specific style requirements — RAG is the right answer. Fine-tuning when you need behavioral mimicry — brand voice, structured output formats, domain-specific reasoning patterns — that few-shot prompting can't reliably achieve.

**Q14. How do you evaluate LLM applications in production?**
> Three layers running continuously. Offline golden set in CI — block any prompt or model change that fails. Online A/B testing with progressive rollout — shadow, canary, full. Continuous user feedback — thumbs, regenerate clicks, conversion events — aggregated per prompt version. Add async LLM-judge scoring on a 1% sample of production traces for faithfulness and relevance trending. Rotate the judge models periodically to avoid bias toward one judge's preferences. The goal is a continuous quality signal that catches degradation before users complain.

**Q15. Drift alerts fire daily but accuracy is stable. What do you do?**
> First, don't retrain reflexively — drift without performance degradation is often noise. Second, switch from raw-feature drift to SHAP-importance-weighted drift — drift on unimportant features shouldn't fire alerts. Third, increase the alert window — single-window anomalies are noisy, sustained drift over multiple windows is signal. Fourth, tie retraining triggers to quality proxies rather than raw drift — confidence drift, prediction distribution drift, business KPI changes. Drift monitoring should be diagnostic, not directly action-triggering. The runbook for a drift alert should be "investigate," not "retrain."

**Q16. How do you design a closed-loop retraining trigger?**
> Combine three signals through a policy engine. Drift detection — PSI or KS on features, weighted by importance. Delayed-label performance — actual accuracy or AUC computed when ground truth arrives. Time cadence — minimum time since last retrain, maximum time since last retrain. Each signal has a threshold; the policy fires only when at least two of three are crossed, with hysteresis. Every trigger decision logs to an audit table — who triggered, what signals fired, what the resulting model was. In regulated GCC contexts there's always a human approval gate before production promotion — automated retrain-and-deploy is a compliance failure waiting to happen.

**Q17. Accuracy dropped 2% overnight. Drift or bug?**
> Triage in this order — and this order matters because data engineering issues are far more common than concept drift. First, check the data pipeline — feature freshness, schema changes, missing partitions, upstream system outages. About sixty percent of "model regressions" are actually broken pipelines. Second, week-over-week feature distribution diff — has any input feature shifted suddenly? Third, serving-code or library upgrade — did anyone deploy a new container? Fourth, model itself — is it the same registered version or did promotion happen unexpectedly? Only if all four are clean do you suspect concept drift. Rule out engineering before rule in modeling.

**Q18. How do you roll back a bad prompt change?**
> If you have a prompt registry — Langfuse, PromptLayer — flip the version alias. Atomic, takes seconds. The next request loads the previous version. If prompts are in code, redeploy the previous container. Importantly, also invalidate any semantic cache entries created with the bad prompt — those will keep serving bad answers if not flushed. And run the golden eval set against the rolled-back version to confirm metrics are back to baseline before declaring all-clear.

**Q19. Multi-LoRA serving — explain.**
> Single base model loaded once into GPU memory; many LoRA adapters swappable at request time. Instead of running fifty separate deployments for fifty fine-tuned model variants, you run one deployment with one base model and fifty adapters in CPU memory, hot-loaded to GPU per request. Latency overhead per swap is in the milliseconds. Cost reduction is ten to twenty times versus separate deployments. Tools: vLLM has multi-LoRA support, LoRAX is a dedicated implementation, Punica is research-grade. This is the standard pattern for SaaS LLM products with per-customer fine-tunes.

**Q20. Langfuse vs Helicone vs Phoenix vs LangSmith — pick one.**
> For Avrioc, Langfuse self-hosted. It's open-source, runs on Postgres, has the strongest combination of trace ingestion, prompt registry, and eval framework, and it deploys on-prem cleanly for UAE data residency. Helicone is great if you want a one-line proxy integration without changing application code, but it's SaaS-first. Arize Phoenix is open-source and strong on embedding drift visualization but lighter on prompt management. LangSmith is the LangChain-native option, great if you're all-in on LangChain, but SaaS. Langfuse hits the sweet spot for self-hosted production LLMOps.

**Q21. Model cascading — what is it?**
> Tier your models by cost and capability. Route easy queries to a cheap model — Haiku, Nova-Lite, GPT-4o-mini. Route hard queries to a strong model — Sonnet, Opus, GPT-4o. The classification can be a small classifier model trained on past traffic, a confidence-based check on the cheap model's output, or a heuristic on query length and complexity. On real production workloads — support bots, FAQ retrieval — model cascading typically saves forty to sixty percent of cost without measurable quality loss because most queries genuinely don't need a frontier model.

**Q22. Eval-driven development — what does it look like in practice?**
> Every prompt or model change runs the golden eval set in CI before merge. The eval set is a curated 100-300 examples of (prompt, expected_output, rubric) maintained by domain experts. Failure on the golden set blocks the PR — same as a unit test failure. Bad answers from production get added to the golden set as regression tests. The team treats LLM applications as software with tests, not as experiments to demo. This single discipline catches the vast majority of "we shipped a prompt change and now things are weird" problems.

**Q23. Continuous retraining cadence — how do you decide?**
> Three factors. Drift velocity — how fast does the data shift? Chat or recommendations might shift weekly; medical models monthly or quarterly. Label availability — fraud labels arrive in days, churn labels in weeks. Validation cost — never retrain faster than you can validate; if eval takes two days, daily retraining doesn't work. The pattern: weekly for fast-moving domains like chat or product recommendations, daily-or-event-triggered for fraud, quarterly for regulated domains like medical. Always with human approval gates for regulated production.

**Q24. Incident response for toxic LLM output — walk me through it.**
> Immediate mitigation — enable strict output guardrails, flip prompt to last known-good version, rate-limit the affected tenant if it's targeted abuse. Communicate — status page update if user-facing, internal Slack to the on-call team. Short-term — add the failing example to the golden eval, update guardrail rules to catch the specific failure mode, ship a hotfix. Post-mortem — trace back through the prompt history to find what changed, whether it was a prompt update, a model provider update, or RAG context poisoning. Blameless culture, documented action items, automated tests added so the same failure can't recur silently.

**Q25. Stack recommendation for Avrioc UAE LLMOps.**
> Self-hosted Langfuse on Postgres for observability, prompt registry, and evals — gives you data residency. LiteLLM as the gateway for routing, caching, and per-tenant cost tracking. Anthropic Claude via Bedrock me-central-1 if AWS, or self-hosted Llama-3 on EKS via vLLM if they want full control. Evidently AI self-hosted for drift detection on classical models. Postgres as the metrics warehouse plus Grafana for dashboards. GitOps via Argo CD or Flux on EKS, with Terraform for infrastructure. Everything runs in UAE region or on-prem to satisfy PDPL. That stack supports both classical ML and LLM workloads, has zero non-self-hostable dependencies, and scales to dozens of models without rearchitecting.

---

Continue to **[Chapter 11 — AWS & Azure](11_aws_azure.md)**.
