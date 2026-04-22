# Chapter 10 — MLOps & LLMOps
## From training to production — the operational backbone

> You spent two and a half years building MLOps at ResMed. Be ready to own this conversation.

---

## 10.1 What is MLOps, really?

MLOps = DevOps + Data + Models. Three artifacts to version instead of one.

```
DevOps world:     Code → Build → Test → Deploy → Monitor
MLOps world:      Code ┐
                  Data ┤→ Build → Test → Deploy → Monitor → Retrain loop
                  Model┘
```

Each has different failure modes. Each needs its own pipeline.

---

## 10.2 The ML lifecycle — every stage needs attention

```
┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ 1. Data    │→│ 2. Feature │→│ 3. Training│→│ 4. Serving │→│ 5. Monitor │
│ ingestion  │  │ engineering│  │ pipelines  │  │  endpoints │  │ & Retrain  │
│            │  │            │  │            │  │            │  │            │
│ - Airflow  │  │ - Features │  │ - MLflow   │  │ - Sagemaker│  │ - Evidently│
│ - Fivetran │  │ - Feature  │  │ - W&B      │  │ - KServe   │  │ - Datadog  │
│ - Airbyte  │  │   store    │  │ - DVC      │  │ - vLLM     │  │ - Arize    │
└────────────┘  └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

---

## 10.3 Reproducibility — the foundation

### What to pin
- **Docker image digest** (not tag — tags are mutable)
- **Dataset hash** (DVC, LakeFS, or S3 object version)
- **Git SHA**
- **Random seeds** (numpy, torch, cuda)
- **Framework versions** (PyTorch, Transformers, vLLM)
- **Hardware** (GPU model, CUDA version)
- **Hyperparameters** (MLflow config or Hydra YAML)

### Tooling
- **MLflow** — experiment tracking + model registry; OSS, self-hostable
- **Weights & Biases** — richer dashboards, sweeps, SaaS
- **DVC** — data + model versioning in git
- **Hydra / OmegaConf** — hierarchical config management

### Two-year reproducibility test
Pick any production run from 2 years ago. Can you reproduce byte-identical output? If no, you don't have real MLOps.

---

## 10.4 Experiment tracking — MLflow or W&B?

| | MLflow | Weights & Biases |
|--|--------|------------------|
| License | OSS | Commercial |
| Deploy | Self-host or managed (Databricks) | SaaS primarily |
| Model registry | ✅ (canonical) | ✅ (Model Registry) |
| Sweeps | Limited | ✅ (killer feature) |
| Collab UI | Functional | Polished |
| Data residency | On-prem available | US / EU / private cloud |
| Best for | Open source teams, Databricks shops, regulated deployments | Fast-moving ML research |

For **Avrioc UAE** — probable data residency → **MLflow self-hosted** is the conservative pick.

---

## 10.5 Feature stores

### The problem they solve
Train with feature F computed one way (Spark batch). Serve with F computed a different way (real-time API). Train-serve skew → broken model in production.

### The pattern
```
           ┌─────────────── Feature Store ───────────────┐
           │                                             │
           │  ┌──────────┐          ┌──────────────┐    │
           │  │ Batch    │          │ Online       │    │
           │  │ storage  │          │ storage      │    │
           │  │ (S3,     │          │ (Redis,      │    │
           │  │  Parquet)│          │  DynamoDB,   │    │
           │  └────┬─────┘          │  Snowflake)  │    │
           │       │                 └────┬─────────┘    │
           └───────┼──────────────────────┼──────────────┘
                   │                      │
         training jobs          real-time inference
         (offline)              (<50ms p99)
```

- Same feature **definition** used for both paths
- Point-in-time correctness for training (no data leakage)
- Low-latency online lookups

### Products
- **Feast** (open source)
- **Tecton** (managed, enterprise)
- **Databricks Feature Store**
- **AWS SageMaker Feature Store**
- **Snowflake Feature Store** (your ResMed experience)

### When it's overkill
- Single model, <10 features
- No online inference

### When it pays off
- Multiple models share features
- Sub-50ms online serving
- Compliance (need audit of feature lineage)

---

## 10.6 CI/CD for ML

### Typical pipeline
```
PR opened
  │
  ├─▶ Unit tests (feature engineering, model code)
  ├─▶ Data validation (Great Expectations)
  ├─▶ Training smoke test (tiny dataset, 1 epoch)
  ├─▶ Model card auto-gen
  │
Merge to main
  │
  ├─▶ Full training on latest data
  ├─▶ Eval on golden set (must beat production baseline)
  ├─▶ Fairness / bias checks
  ├─▶ Register model in MLflow (stage: Staging)
  │
Manual approval
  │
  ├─▶ Shadow deploy (mirror prod traffic, log only)
  ├─▶ Canary 10% traffic
  ├─▶ Full rollout
  └─▶ Monitor drift, regress, rollback if needed
```

### Tools
- **GitHub Actions / GitLab CI / Jenkins** for orchestration (your TrueBalance MLOps likely uses CodePipeline)
- **Argo Workflows / Airflow / Prefect** for scheduled training
- **Argo CD** for GitOps deploy to Kubernetes

### Quality gates (must-have)
- Baseline metric threshold (e.g., AUC ≥ current prod - 0.5%)
- Latency budget on test traffic
- Fairness metric (e.g., per-subgroup AUC gap < X)
- No regressions on held-out "canary" queries

---

## 10.7 Model registry — the traffic light

Stages (MLflow canonical):
```
None → Staging → Production → Archived
```

- **None** — newly trained
- **Staging** — passed CI, awaiting human review
- **Production** — promoted to prod endpoint
- **Archived** — retired

Every endpoint queries the registry by alias/stage. Rollback = change registry pointer; zero-downtime.

---

## 10.8 Training-serving skew

### Causes
1. Different feature engineering code online vs offline
2. Different preprocessing (tokenization, normalization)
3. Different library versions
4. Concept drift between training data and serving data
5. Timezone / date-handling bugs

### Prevention
- Share code: offline and online call the same `compute_features()` function
- Package transforms into the model artifact (sklearn Pipeline, ONNX with preprocessing)
- Or use a feature store enforcing a single definition
- Continuous monitoring: feature distribution drift alerts

---

## 10.9 LLMOps — what's different?

MLOps for classical ML focuses on training. LLMs are mostly **served**, not trained by the consumer. LLMOps focuses on:

| LLMOps concern | What it is |
|----------------|------------|
| **Prompt versioning** | Prompts are code — register, version, link to evals |
| **Evaluation** | LLM-as-judge, RAGAS, golden sets, user feedback |
| **Guardrails** | PII redaction, jailbreak detection, schema enforcement |
| **Cost tracking** | Tokens in/out, per user/tenant, budget alerts |
| **Observability** | Trace every LLM call; prompt + completion + tools + latency |
| **Semantic caching** | Cache responses for similar queries |
| **Prompt regression detection** | Model provider changes → test golden set again |

### The LLMOps stack (2026)

```
┌──────────────────────────────────────────────────────────┐
│  Application (FastAPI + LangGraph + guardrails)          │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  LLM Gateway (LiteLLM, Portkey) — routing, cache, auth   │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Observability (Langfuse, Arize Phoenix, Helicone)       │
│  - Trace ingestion                                        │
│  - Eval scores                                            │
│  - Cost tracking                                          │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│  Model layer (OpenAI / Anthropic / Bedrock / self-host)  │
└──────────────────────────────────────────────────────────┘
```

---

## 10.10 Prompt management

### The patterns
- **Inline in code** — only for tiny projects. Becomes unmaintainable.
- **Version-controlled YAML / templates** — in git, parametrized.
- **Prompt registry** (Langfuse, PromptLayer) — runtime-loadable versions, linked to eval scores, A/B testable.

### Canonical prompt object
```yaml
name: clinical_rag_system
version: v3.2
model: claude-sonnet-4-6
temperature: 0.2
max_tokens: 1024
template: |
  You are a clinical assistant. Answer using only the context.
  ...
eval_score: 0.89  # last RAGAS score
```

---

## 10.11 LLM evaluation — three layers

### 1. Offline evaluation
Golden set of (prompt, expected_output, rubric). Score via:
- **LLM-as-judge** (pairwise or rubric-based)
- **Reference-based metrics** (BLEU, ROUGE — limited usefulness)
- **Task-specific** (exact match, F1, SQL execution accuracy)

Frameworks: **RAGAS**, **DeepEval**, **Promptfoo**, **TruLens**, **Arize Phoenix**.

### 2. Online A/B test
Route X% of traffic to new prompt/model/pipeline. Compare:
- Task success proxy (completions, clicks)
- User feedback (thumbs, regenerate rate)
- Cost and latency

Requires enough traffic for statistical power (usually 1-2 weeks per segment).

### 3. Continuous user feedback
Every production trace gets a score (implicit or explicit):
- Thumbs up / down
- Regenerate triggered
- Conversion / task completion
- Human review of low-confidence traces

---

## 10.12 Guardrails

Input and output validators wrapping the LLM call.

### Input guardrails
- PII detection / redaction (Presidio, AWS Comprehend)
- Jailbreak / prompt injection detection (Rebuff, NeMo Guardrails)
- Topic restriction
- Profanity / safety

### Output guardrails
- Hallucination detection (faithfulness to context)
- PII leak (model emitting user data)
- Schema validation (JSON output)
- Brand/tone conformance

### Placement
Run input + output guardrails in **parallel with** (not before) the LLM call where possible — hides latency. Abort the response if any guardrail fails.

### Tools
- **NVIDIA NeMo Guardrails** — flexible, rule + model-based
- **Guardrails AI** — open-source, schema-centric
- **LlamaGuard** (Meta) — safety classifier

---

## 10.13 LLM cost tracking

### Per-request cost math
```
cost = prompt_tokens × in_rate + completion_tokens × out_rate
```

Log both to a gateway (LiteLLM, Portkey) with metadata (user, tenant, feature). Aggregate in ClickHouse / BigQuery.

### Cost-optimization levers (ordered by impact)

1. **Model cascade** — route easy → Haiku/Nova; hard → Sonnet/Opus. Cuts 40-60% on realistic workloads.
2. **Prompt caching** (Anthropic, OpenAI, Bedrock) — 90% discount on cached tokens. Massive on long system prompts / RAG contexts.
3. **Semantic caching** (GPTCache) — dedupe similar queries.
4. **Batch API** (OpenAI, Anthropic) — 50% discount for async workloads.
5. **Self-host top 20% highest-volume prompts** — break-even ~1M tokens/hour.
6. **Output shortening** — structured schemas, prose discipline.
7. **Prompt optimization** — shorter prompts, few-shot reduction.

---

## 10.14 Semantic caching

```
Incoming query → embed → search cache vector store
  │
  ├─ similarity ≥ 0.97? → return cached answer (save 100% LLM cost)
  └─ else → LLM call → store result in cache
```

Thresholds:
- 0.97-0.99: safe for factual/FAQ
- 0.90-0.95: risky, can cause wrong answers if queries differ in subtle details

Watch for **cache poisoning**: two different questions mapping to the same cached wrong answer.

Tool: **GPTCache**, Redis Vector, Qdrant.

---

## 10.15 Prompt regression detection

Problem: model provider silently updates their model (GPT-4o-2025-06 → GPT-4o-2025-09) and your prompts now misbehave.

Defense:
- **Pin model versions** (use dated model IDs)
- **System fingerprint** monitoring — alert on changes
- **Run golden set daily** — detect regressions early
- **A/B on new versions** before switching production

---

## 10.16 Observability for LLM apps

Trace every LLM call end-to-end:
- User ID + tenant + session
- Full prompt (maybe redacted PII)
- Full response + tool calls + tool results
- Tokens in/out, latency, model, temperature
- User feedback score
- Eval scores (faithfulness, relevance)

### Tools
- **Langfuse** (OSS, self-hostable) — strong on traces/evals/prompt mgmt; ideal for UAE residency
- **Helicone** — proxy-based, one-header integration
- **Arize Phoenix** — open-source, embedding-drift focus
- **LangSmith** — LangChain's native
- **TrueFoundry** — integrated platform (observability + deploy + gateway + GPU autoscaling)

---

## 10.17 The MLOps maturity model

| Level | Characteristics |
|-------|-----------------|
| **0** | Manual notebooks, no versioning, no monitoring |
| **1** | CI/CD for code only; model training manual |
| **2** | Automated training + registry; manual promotion; basic monitoring |
| **3** | Automated promotion with canary; drift detection; closed-loop retraining |
| **4** | Feature store; online eval; automated retraining triggers; multi-region |

Avrioc likely has Level 2-3; your ResMed framework sounds Level 3. **Lead with that.**

---

## 10.18 Interview Q&A — MLOps/LLMOps

**Q1. What's different about ML CI/CD vs standard software CI/CD?**
> Three artifacts instead of one: code, data, model. You run data validation, training reproducibility, quality gates (min accuracy, fairness), and canary/shadow deploys. Promotion is tied to model registry stages, not just Docker tags.

**Q2. MLflow vs W&B — pick?**
> MLflow: OSS, self-hosted, great for Databricks or on-prem. W&B: SaaS-first, richer dashboards, better sweeps. For regulated GCC (Abu Dhabi), MLflow self-hosted is conservative. For fast-moving R&D, W&B.

**Q3. When is a feature store overkill?**
> Single model, <10 features, no online serving. Overhead outweighs benefit. It pays off with multiple models sharing features, sub-50ms online lookups, or compliance-grade audit needs.

**Q4. Training-serving skew — how do you prevent it?**
> Share feature transformation code between offline and online paths via a feature store or by packaging transforms inside the model artifact. Monitor online feature distributions vs training distributions. Alert on divergence.

**Q5. How do you guarantee reproducibility two years later?**
> Pin Docker image digest, dataset hash (DVC/LakeFS), git SHA, seeds, hardware, framework versions. Store resolved `pip freeze`, not `requirements.txt`. Log all to MLflow run metadata.

**Q6. Model promotion workflow?**
> PR → train → MLflow log. If eval beats baseline, auto-promote to Staging. Human approves → GitOps to KServe InferenceService. 10% canary 24h → full rollout. Rollback = flip registry pointer.

**Q7. LLMOps vs MLOps — what's different?**
> LLMOps focuses on prompts (versioning, eval), guardrails, cost tracking, observability of every call. Most consumers don't train LLMs — they integrate them. So less training pipeline, more prompt/model pipeline.

**Q8. How do you version prompts?**
> Prompt registry (Langfuse, PromptLayer) with semantic versions and eval-score links. Never inline in application code. Log the prompt_version_id in every trace for rollback and A/B comparison.

**Q9. LLM guardrails — implementation?**
> Input and output validators around the LLM call: PII redaction, jailbreak detection, topic restriction, JSON schema, hallucination checks. NVIDIA NeMo Guardrails or Guardrails AI. Run input/output guardrails parallel to LLM call to hide latency.

**Q10. LLM cost tracking — approach?**
> Attach metadata (user, tenant, feature) to every call via a gateway (LiteLLM). Log token counts + cost. Aggregate in ClickHouse / BigQuery. Alert on per-tenant budget breaches.

**Q11. How do you detect prompt regression?**
> Pin model versions. Monitor system_fingerprint. Run golden eval set daily — alert when metrics drop >2%. A/B test any prompt/model change.

**Q12. Semantic caching — when and how?**
> Embed query, search vector store for similar past queries (threshold 0.95-0.99), return cached answer if found. Huge cost win for FAQ / support bots. Watch cache poisoning for semantically-similar-but-different queries.

**Q13. RAG vs fine-tuning — operational difference?**
> RAG: near-instant updates (reindex), cheap, auditable (citations). Fine-tuning: bakes knowledge in, better style adherence, requires retraining pipeline, risks forgetting. 80% of enterprise use cases are RAG-first.

**Q14. How do you evaluate LLM apps in prod?**
> Three layers: (1) offline golden set with LLM-as-judge, (2) online A/B (task success, latency, cost), (3) continuous user feedback (thumbs, implicit). Rotate judges to avoid bias.

**Q15. [Gotcha] Your drift alerts fire daily but accuracy is stable. What do you do?**
> Drift without degradation. Don't retrain reflexively. Check SHAP importance-weighted drift — raw-feature drift on unimportant features is noise. Tie retraining triggers to quality proxies, not raw drift.

**Q16. Closed-loop retraining trigger — design?**
> Combine drift detection + delayed-label performance + time-based cadence. Policy engine decides. Log every trigger decision to audit table. Human approval gate before production promotion (mandatory for regulated GCC).

**Q17. [Gotcha] Accuracy dropped 2% overnight. Drift or bug?**
> Triage: (1) data pipeline freshness + schema changes (most common). (2) Feature distribution diff week-over-week. (3) Serving-code or library upgrade. (4) Only then suspect concept drift. Rule out data engineering before retraining.

**Q18. How do you roll back a bad prompt change?**
> Prompt registry + active-version pointer flip — atomic, seconds. If prompts are in code → redeploy. Also invalidate semantic cache entries from the bad prompt.

**Q19. Multi-LoRA serving — why?**
> One base model in VRAM; adapters swapped per request (ms). Beats 50 separate deployments by 10-20× on cost. vLLM, LoRAX, Punica.

**Q20. Langfuse vs Helicone vs Phoenix vs TrueFoundry?**
> Langfuse: OSS, self-host, strong on traces/evals/prompt mgmt — ideal for UAE residency. Helicone: proxy-based, one-header integration. Phoenix: Arize's open-source, embedding drift focus. TrueFoundry: bundles LLMOps + deploy + GPU autoscaling.

**Q21. Model cascading — what and why?**
> Route easy queries to cheap models (Haiku, Nova-Lite), hard to strong (Sonnet, Opus). 40-60% cost reduction without quality loss. Implementation: classifier or confidence-based routing.

**Q22. What are eval-driven development workflows?**
> Every prompt/model change runs against the golden set in CI. Failing eval blocks the PR. Treat LLM apps like software tests, not like experiments.

**Q23. Continuous retraining cadence?**
> Depends on drift velocity. Chat / product recommendations: weekly. Fraud detection: daily or triggered on drift. Medical models: quarterly with heavy validation. Never retrain faster than you can validate.

**Q24. Incident response for toxic LLM output?**
> Immediate: enable strict output guardrail, flip prompt to prior-known-good, rate-limit tenant. Short-term: add failing example to eval, update guardrail rules. Post-mortem: trace back to prompt change, model update, or RAG context poisoning. Blameless.

**Q25. For Avrioc UAE — stack recommendation for LLMOps?**
> Self-hosted Langfuse for observability (data residency). LiteLLM gateway for routing/caching. Claude via Bedrock me-central-1 OR self-hosted Llama-3 via vLLM on EKS. Evidently self-hosted for drift. Postgres as prompt registry + metrics store. GitOps via Argo CD. All on-prem or UAE region.

---

## 10.19 Resume tie-ins — your MLOps stories

### Story: "Orchestrated MLOps framework on AWS, 8 models in 6 months"
Expand with:
- Standardized SageMaker training container (CodeBuild → ECR)
- Inference pipeline templates (SageMaker Model → Endpoint Config → Endpoint)
- Airflow DAGs for preprocessing
- MLflow self-hosted on EC2 + RDS for tracking + registry
- Datadog drift dashboards auto-wired via the framework
- Snowflake feature store for online features
- Data scientists submitted `model_config.yaml`; framework did everything else

### Story: "Developed utility to integrate drift monitoring into Datadog"
- Per-model drift configs (KS, PSI per feature, thresholds)
- Snowflake logs → Datadog custom metrics via push API
- Anomaly detection monitors on drift scores
- Dashboards template per model (heatmap + line charts + annotations)

### Story: "7-entity / 29-predicate SMS ontology, 107 tests"
- Zero-diff migration test gates — every PR must produce identical seed SQL
- Ontology versioned in git; test suite asserts coverage on production SMS fields
- Example of MLOps for rule-based systems; same rigour applies

---

Continue to **[Chapter 11 — AWS & Azure](11_aws_azure.md)**.
