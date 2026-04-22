# Chapter 15 — Resume Projects, Deep Dive
## Every bullet explained as an interview-ready story

> For each bullet on your resume, this chapter gives you: context, the story (STAR), the technical detail the interviewer will drill into, and the follow-up Qs you should expect.

---

## 15.1 TrueBalance (Senior ML Engineer, Feb 2026 – Present)

### Bullet 1 — Real-time XGBoost Lambda (p99 < 500ms, 3-env VPC-isolated)

**Context:** TrueBalance is an Indian consumer lender. Loan-withdrawal risk (borrower takes the money, withdraws before full disbursal) bleeds portfolio profit.

**STAR:**
- **Situation:** Weekly loan-withdrawal losses were measurable and growing. No production signal to cut funding on high-risk loans before disbursement.
- **Task:** Ship a production predictor — real-time, sub-500ms p99, deployed across 3 environments with strict VPC isolation for PII.
- **Action:** Built a binary classifier (XGBoost, ~150 features from Snowflake feature store, SHAP for explainability) → packaged as container → deployed to AWS Lambda container runtime → 3-environment Terraform modules (dev/staging/prod), each with dedicated VPC, private subnets, VPC endpoints (S3, Secrets Manager, CloudWatch Logs), provisioned concurrency. Feature fetch from Redis-fronted Snowflake cache for sub-50ms p99 on the feature call; fallback to default features on Redis timeout.
- **Result:** p99 consistently <500ms across 3 envs. Projected portfolio-profit lift from cutting high-withdraw-risk funding pre-disbursement.

**Technical drill-downs likely:**
- Why XGBoost not neural? — Tabular, ~150 features, explainability needed. Trees dominate.
- Why Lambda? — Spiky traffic, sub-500ms budget, no GPU needed, tight AWS integration.
- How did you hit p99 <500ms? — Provisioned concurrency (no cold starts), Redis for features (<50ms p99), container size optimized, feature computation in-Lambda.
- Feature drift? — Weekly KS + PSI per feature; Datadog dashboards; retraining trigger at PSI > 0.2 on any top-5 feature.
- Rollback? — Model registry pointer flip; redeploy the previous container tag via CodePipeline.
- Why 3 VPCs? — Blast-radius containment; PII in feature store must not cross environments; each env has its own KMS CMK.

### Bullet 2 — Lender-identification NER (29.7% → 68.0%)

**Context:** Credit bureau tradelines contain a "creditor_name" field. Matching it to the correct lender in your internal catalog is a hard NER problem because of inconsistent formatting, abbreviations, typos.

**STAR:**
- **Situation:** Legacy system used a 2,000-term keyword expansion. Production accuracy on 109K tradelines was only 29.7% — blocking a credit-reporting feature launch.
- **Task:** Lift accuracy materially without losing any existing matches (zero-regression requirement).
- **Action:** Layered an NER-based lender-entity extractor (fine-tuned BERT-family NER model) over the legacy keyword expansion. Added a BANK-lender boosting heuristic (if extracted entity has BANK / NBFC keyword, boost its score). Validated on full 109K tradelines (78K softpull + 31K hardpull) with intersection test — every legacy match still matched.
- **Result:** 29.7% → 68.0% accuracy. Zero regressions. Feature unblocked.

**Technical drill-downs:**
- Why NER? — "lender name" isn't always first/last token; it's embedded in messy text. Keyword matching is brittle; NER captures context.
- How did you fine-tune? — Labeled ~5K examples of real tradelines, BIO tagging, fine-tuned DistilBERT or similar for speed.
- Zero-regression validation? — Intersection test: every match in legacy must also match in new system. Automated in CI.
- Latency? — Lambda inference, <200ms p99 (DistilBERT quantized INT8).
- Error analysis? — Confusion matrix on labeled set, sliced by creditor category; found NBFCs had weaker performance due to fewer training examples.

### Bullet 3 — 7-entity / 29-predicate SMS ontology (107 tests, zero-diff seed SQL)

**Context:** Legacy SMS engine used regex to parse SMS transactions (banks SMS format: "Rs.5000 debited from A/C ...XXX on 01-APR-26"). Fragile; frequent breakage on new bank formats.

**STAR:**
- **Situation:** Regex-based SMS parsing had ~85% coverage on production SMS fields; broke often on new issuer formats; caused downstream feature quality issues.
- **Task:** Replace with an ontology-based system at 100% coverage, fully testable.
- **Action:** Designed an ontology of 7 entities (amount, date, account, merchant, type, balance, reference) and 29 predicates (governing valid combinations). Migrated Phase 1 into a standalone ML repository with 107 unit/integration tests. Seed SQL migrations validated for byte-identical output via zero-diff gate in CI.
- **Result:** 100% coverage on 170K production SMS fields. Migration landed with zero diffs in downstream features. Reduced on-call SMS-parse-failure incidents.

**Technical drill-downs:**
- Ontology-based vs LLM-based? — Ontology-based is deterministic, auditable, auditable (compliance); LLM would be a fallback for the long tail.
- How did the predicates work? — Rules like "if {type} = DEBIT, {amount} > 0 and {balance} < previous_balance". Validation logic runs per-parse.
- Why 107 tests? — One per predicate + integration per bank format + regression on production samples.
- Zero-diff seed SQL? — Golden seed dataset; migration generates SQL; diff against baseline SQL is hashed and compared. CI blocks PR if diff is non-zero.

### Bullet 4 — AI-powered ML workspace on Claude

**Context:** ML engineers at TrueBalance spent too much time on ticket isolation, GPU/CPU provisioning, Athena queries, Jenkins triggers — context-switching killed productivity.

**STAR:**
- **Situation:** Scattered workflow: Jira for tickets, manual EC2 provisioning, git worktree manual setup per branch, separate tools for Athena + Jenkins.
- **Task:** Build a unified AI-driven developer platform.
- **Action:** Claude Sonnet backbone with tool-calling into Jira, GitHub, AWS Athena, Jenkins APIs. On-demand GPU/CPU EC2 provisioning (EFS-shared state + 3-method EBS lifecycle for fast start/stop/archive). Automated per-ticket git-worktree isolation so engineers could work on multiple tickets simultaneously without branch conflicts. FastAPI backend + React frontend.
- **Result:** ML engineer productivity improved by eliminating context-switch overhead; safe parallel experimentation across shared repos; standardized on-demand compute provisioning.

**Technical drill-downs:**
- Why Claude Sonnet vs GPT-5 or open model? — Strong tool-calling quality, JSON mode reliability, multi-step reasoning; cost acceptable for internal use.
- Tool-calling architecture? — Tools defined as JSON schemas; Claude emits tool calls; dispatcher routes to per-tool handler; response injected back into conversation.
- On-demand EC2 3-method EBS lifecycle? — (1) Instant snapshot restore, (2) snapshot-on-stop for warm-archive, (3) delete-on-terminate for ephemeral. Different method per use case.
- EFS for shared state? — Mount across instances to share git repos, Jupyter workspaces, cached datasets without re-downloading.
- Git worktree per ticket? — `git worktree add /path branch-name` per Jira ticket; isolated working trees; avoids stashing/switching.
- Safety? — Tools have IAM scope; no direct shell exec — only well-defined API calls. Rate limits and approval gates for destructive ops (terminate instance).

---

## 15.2 ResMed (ML Engineer, Jul 2023 – Jan 2026)

### Bullet 1 — GenAI-powered query routing system (knowledge-base chatbot, RAG)

**Context:** ResMed (medical device company; sleep apnea CPAP, respiratory care) has vast unstructured clinical knowledge — protocols, studies, internal research.

**STAR:**
- **Situation:** Clinical analysts spent hours hunting for answers in PDFs/reports. Generic LLM answers hallucinated medical details.
- **Task:** Build a production chatbot that reliably answers factual and analytical questions on the clinical knowledge base.
- **Action:** Three-layer architecture: (1) document-aware chunker respecting clinical section headers (History, Medications, Assessment), (2) embedding + pgVector index with hybrid BM25+dense retrieval, (3) LLM query router that classifies queries — factual → RAG pipeline, analytical → auto-generated Python/SQL, conversational → direct LLM. Answers included citations. RAGAS eval on a 300-pair golden set built with clinical SMEs.
- **Result:** Adopted by clinical team; analyst time-to-answer cut significantly; answers grounded in source documents.

**Technical drill-downs:**
- Why route instead of one pipeline? — RAG is poor at "what's the average age of diabetic patients in this cohort?" — needs structured SQL. Router trades complexity for correctness per query type.
- Embedding model choice? — Evaluated OpenAI text-embedding-3-small and domain-specific medical embeddings on labeled 500-pair test; picked based on recall@10 + inference latency on pgVector.
- Hybrid search? — BM25 for exact medical terms (drug names, ICD codes); dense for semantic. RRF fusion.
- Chunking? — Section-aware because clinical reports have standardized structure. Parent-doc retrieval (embed small, return parent) for context.
- Evaluation? — RAGAS faithfulness, answer relevance, context precision, context recall. Golden set of 300 (query, answer, doc_id) built with a clinical analyst over a week.
- Citations? — Chunks numbered in context; prompt requires `[1]` style citations; post-process verifies citations map to real chunks.
- Hallucination handling? — Confidence threshold on retrieval scores; below threshold → "I don't have information" response.

### Bullet 2 — Datadog + Snowflake drift dashboards

See Chapter 14.6 for the full story.

### Bullet 3 — Multi-container SageMaker endpoints (IHS platform)

**Context:** IHS = Intelligent Health Studio (ResMed's MLOps platform for clinical ML).

**STAR:**
- **Situation:** Each model deployed on its own SageMaker endpoint — cost was linear in model count, many endpoints underutilized.
- **Task:** Reduce cost per model without sacrificing latency or isolation.
- **Action:** Deployed multi-container endpoints (up to 15 containers per endpoint, invokable by target name). Grouped models by resource class (CPU-small, CPU-large, GPU-T4, GPU-A10). Shared endpoint across a resource class; routing by model ID. Autoscaling per endpoint based on invocations-per-instance.
- **Result:** ~40% cost reduction on the long-tail models; single-digit-ms overhead per invocation; clean isolation between containers.

**Technical drill-downs:**
- MCE vs MME vs dedicated? — MCE for heterogeneous models (different frameworks); MME for homogeneous (many similar); dedicated for top-traffic models needing isolation.
- How does invocation routing work? — `target_container_hostname` in the InvokeEndpoint request; endpoint forwards to the named container.
- Cold-start? — Containers always loaded; cold only when endpoint first spins.
- A/B via MCE? — Two versions of same model as two containers; gateway splits traffic.

### Bullet 4 — Snowflake feature store

**STAR:**
- **Situation:** No shared feature store; DS teams computed features in notebooks; train/serve skew bugs.
- **Task:** Design schemas for an online + offline feature store in Snowflake.
- **Action:** Two-layer design: offline `FEATURE_HISTORY` (point-in-time correct, used for training), online `FEATURE_LATEST` (last-value, used for real-time serving). Features ingested via Airflow (batch) and Snowpipe (streaming). Ingestion and serving code in a shared library to avoid skew.
- **Result:** Multiple models consumed shared features; train/serve consistency validated via daily diff; feature lineage in a central catalog.

**Technical drill-downs:**
- Point-in-time correctness? — Training queries use `AS OF` timestamp, ensuring no data from after the prediction time leaks.
- Online serving latency? — Snowflake direct reads were 100-300ms; added Redis cache in front for sub-50ms p99.
- Online / offline skew prevention? — Shared Python library computes features identically in both paths; feature-store SDK wrapper enforces it.

### Bullets 5-6 — Airflow + async endpoints

Short-form:
- **Airflow:** DAGs for daily feature ingestion, weekly retraining, monthly cold-storage archival. Operators wrap SageMaker + Snowflake tasks.
- **Async SageMaker endpoints:** For long-running inference (e.g., full-report analysis); SQS-backed; scale-to-zero.

---

## 15.3 Tiger Analytics (ML Engineer, Dec 2021 – Jul 2023)

### Bullet 1 — SageMaker training + inference pipelines with drift monitoring

**STAR:**
- **Situation:** Client had ad-hoc SageMaker jobs. No drift monitoring, manual retraining, no versioning discipline.
- **Task:** Build end-to-end pipelines with quality gates and drift detection.
- **Action:** CodePipeline orchestrating SageMaker Pipelines (preprocess → train → eval → register → deploy). Model Monitor for data quality + drift. CloudWatch alarms on drift thresholds triggered retraining via EventBridge.
- **Result:** Reduced deployment errors; automated drift-triggered retraining kept models within accuracy SLAs.

### Bullet 2 — CI/CD pipelines for model deployment

**STAR:**
- Standardized multi-env deploy (dev → staging → prod) via CodePipeline.
- Blue-green deploys to SageMaker endpoints with traffic-shift steps.
- Terraform-managed infra.

### Bullet 3 — Custom data quality checks with Deequ on Databricks (Mars)

**STAR:**
- **Situation:** Mars' large-scale data pipelines had silent data-quality issues (null spikes, schema drifts).
- **Task:** Add automated data quality gates before downstream ML consumption.
- **Action:** Deequ constraints (completeness, uniqueness, value-range, statistical-anomaly) configured per table. Ran as scheduled Databricks jobs. Failures published to Slack + Azure Monitor.
- **Result:** Caught issues upstream; DS didn't debug models mid-sprint to find it was bad data.

**Technical drill-downs:**
- Deequ constraint examples? — `hasCompleteness("user_id", _ >= 0.99)`, `hasMin("age", _ > 0)`, `hasDistribution("country", some_profile)`.
- Scalable? — Runs in Spark; computes approximate stats where possible (HyperLogLog for cardinality).
- Alert routing? — Per-table ownership metadata; notifications go to the right DS team's Slack.

### Bullets 4-5 — Azure Databricks + Data Factory

Short-form:
- Azure Databricks for big-data preprocessing, drift detection scripts
- ADF for pipeline orchestration, triggering Databricks activities

---

## 15.4 Sopra Steria (Senior Software Engineer, Aug 2018 – Dec 2021)

### Bullet 1 — CNN + YOLO + OCR ID verification

**STAR:**
- **Situation:** Client needed automated ID card verification.
- **Task:** End-to-end system: detect ID in image, crop, OCR, verify formatting and font anomalies.
- **Action:** YOLO v3/v4 for ID detection (trained on labeled samples). CNN for ID-type classification. OCR via Tesseract + a trained font anomaly detector (to catch forgeries).
- **Result:** Improved automation rate; reduced manual review.

### Bullet 2 — Time-series anomaly on Prometheus/Grafana

**STAR:**
- Extracted server metrics from Prometheus.
- Forecasting with ARIMA + LSTM.
- Anomaly when |actual - forecast| > threshold.
- Grafana alerts.

### Bullet 3 — Loan-risk XAI ensemble

**STAR:**
- Ensemble of XGBoost + RF + LR.
- SHAP for feature-level explanations per prediction.
- Custom UI showing "this applicant was rejected because of feature X (value Y, contribution Z)".

### Bullet 4 — OR-Tools logistics

**STAR:**
- 300 delivery locations; vehicle routing problem.
- OR-Tools CP-SAT solver.
- Reduced cover time from 7 days to 5.

### Bullet 5 — Oracle Apps ETL

- PL/SQL procedures, scheduled via Oracle Concurrent Manager.
- Legacy work — mention for breadth.

---

## 15.5 Awards / Certifications

- **Sopra Steria India Coding Championship** — 1st place — mention in behavioral ("I enjoy hard problems")
- **Google Code Jam Qualifier 2019** — mention for algorithmic bona fides
- **LangChain certifications** (LangGraph foundations + Deep Agents) — relevant to JD
- **TensorFlow Developer Certificate** — classical DL proficiency

---

## 15.6 Cross-cutting STAR-ready summaries

### "Tell me about a time you shipped under tight constraints."
→ Real-time XGBoost Lambda (500ms p99, 3 envs, VPC-isolated). 6-week build including Terraform modules.

### "Tell me about a time you improved a legacy system."
→ Lender-identification NER (29.7 → 68%). Zero-regression by design.

### "Tell me about a time you led technical strategy."
→ ML workspace assistant — you chose Claude vs open-source, designed the tool ecosystem, influenced team adoption patterns.

### "Tell me about a difficult debugging problem."
→ SMS ontology migration: 107 tests, zero-diff seed SQL. Any diff in generated SQL across environments failed the CI gate; tracking down per-environment locale differences in SQL generation.

### "Tell me about a time you had to collaborate across teams."
→ ResMed drift monitoring utility: worked with every DS team to standardize their metrics, embedded in their retraining pipeline, became team standard.

### "Tell me about a time you disagreed with a technical decision."
→ (Prepare one real story — e.g., you advocated for multi-container SageMaker endpoints against "one endpoint per model" default; used cost projection + latency data to convince the team.)

---

Continue to **[Chapter 16 — System Design](16_system_design.md)**.
