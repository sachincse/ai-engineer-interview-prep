# Interview Preparation — Sachin Singh
## Avrioc Technologies | AI Engineer | Abu Dhabi (Onsite)

> **Candidate:** Sachin Singh — AI Engineer | ML Solutions Architect | MLOps & LLMOps Expert
> **Role:** AI Engineer, Avrioc Technologies, Abu Dhabi (Onsite, Permanent, Visa Sponsorship)
> **Hiring Manager Contact:** Wilma Herron (HR Leader)
> **Interview Date (target):** 2026 Q2
> **Experience:** 8+ years (TrueBalance, ResMed, Tiger Analytics, Sopra Steria)

---

## How this document is organized

This is a **18-chapter interview-preparation pack** (~100+ printed pages). Every resume bullet, every JD line item, and every "must-know" concept for a Senior AI/LLMOps Engineer is covered with:

- **Conceptual explanation** — plain English, first principles
- **Block diagrams** — ASCII / Mermaid diagrams you can sketch on a whiteboard
- **Math / formulas** — where relevant
- **Code snippets** — minimal but correct
- **Interview Q&A** — canonical questions + model answers (2025-2026 patterns)
- **"Red-flag" traps** — common gotcha questions and how to handle them
- **Resume tie-ins** — how to bridge each concept to your actual projects

---

## Table of Contents

| # | Chapter | Est. Pages | What it covers |
|---|---------|-----------|----------------|
| 00 | [Master Index & JD Alignment](00_index.md) | 5 | This file — JD analysis, resume map, strategy |
| 01 | [Foundations — Neural Nets, Word Embeddings](01_foundations.md) | 6 | Backprop, activations, Word2Vec, GloVe, FastText |
| 02 | [Transformer Architecture Deep Dive](02_transformers.md) | 10 | Attention math, MHA, MQA, GQA, RoPE, ALiBi, FlashAttention |
| 03 | [How LLMs Work End-to-End](03_llms.md) | 10 | Tokenization, pretraining, SFT, RLHF, DPO, inference, KV-cache |
| 04 | [Embedding Models — How They're Created](04_embeddings.md) | 7 | Contrastive learning, SBERT, BGE, E5, Matryoshka, ColBERT |
| 05 | [LLM Parameter Tuning (Inference)](05_parameter_tuning.md) | 8 | Temperature, top-p/k, min-p, penalties, beam search, when to use what |
| 06 | [Fine-tuning — SFT, RLHF, DPO, LoRA, QLoRA, PEFT](06_fine_tuning.md) | 8 | PEFT theory, adapter math, DPO vs RLHF, hyperparameters |
| 07 | [RAG — Retrieval-Augmented Generation](07_rag.md) | 10 | Naive → advanced RAG, hybrid search, reranking, GraphRAG, Self/Corrective RAG, RAGAS |
| 08 | [Vector Databases & Search Indexes](08_vector_databases.md) | 5 | HNSW, IVF, PQ; pgVector, Pinecone, Qdrant, Weaviate, Milvus |
| 09 | [Model Optimization — Quantization, Pruning, Distillation](09_model_optimization.md) | 10 | INT8/INT4, GPTQ, AWQ, GGUF, SmoothQuant, magnitude/structured pruning, KD |
| 10 | [MLOps & LLMOps](10_mlops_llmops.md) | 8 | CI/CD, registries, feature stores, MLflow, LangFuse, evaluation, guardrails |
| 11 | [AWS & Azure for AI](11_aws_azure.md) | 7 | SageMaker endpoints, Lambda, ECR/EC2/EFS, VPC, Databricks, Azure ML |
| 12 | [Kubernetes, Ray, Docker](12_kubernetes_ray.md) | 7 | K8s fundamentals, KServe, KEDA, Ray Core/Serve/Train/Tune, Docker |
| 13 | [Frameworks — FastAPI, LangChain, LangGraph, CrewAI, vLLM, Chainlit, Streamlit](13_frameworks.md) | 10 | Production patterns, pitfalls, choosing between them |
| 14 | [Monitoring, Observability, Drift Detection](14_monitoring_drift.md) | 6 | PSI, KS, Evidently, Datadog, Prometheus, Grafana, alerts |
| 15 | [Resume Projects — Deep Dive](15_resume_deep_dive.md) | 12 | Every bullet from your resume explained with STAR answers |
| 16 | [System Design Interviews](16_system_design.md) | 8 | Real-time inference, RAG at scale, LLM chatbot, recommender |
| 17 | [Behavioral, HR, UAE Relocation](17_behavioral_hr.md) | 5 | STAR answers, UAE-specific, salary, visa, culture |
| 18 | [Cheatsheets — Formulas, Commands, Numbers](18_cheatsheet.md) | 3 | Last-minute revision reference |

**Total: ~130 pages** of focused, depth-first content.

---

## 1. Job Description Analysis

### 1.1 The Role — Avrioc Technologies AI Engineer

**Avrioc Technologies** is a product-based engineering company (globally growing product, per the JD). The role is:

- **Location:** Abu Dhabi, UAE (**onsite**, no remote)
- **Type:** Permanent, full-time
- **Salary:** Tax-free (UAE has no personal income tax)
- **Benefits:** Visa sponsorship, relocation assistance, family medical coverage (spouse + up to 3 children)
- **Team shape:** Cross-functional — engineers + data scientists + product owners
- **Reporting contact:** Wilma Herron (HR Leader)

### 1.2 JD Responsibilities — Decoded

| JD Line | What They Actually Mean | Your Resume Evidence |
|---------|------------------------|---------------------|
| "Building, deploying, maintaining AI/ML models in production" | Full ML lifecycle ownership — not just notebooks | 8 models deployed to prod at ResMed IHS platform; real-time XGBoost Lambda at TrueBalance |
| "Productionizing LLM apps using Kubernetes, Ray, LLMOps" | Scaled serving, not toy demos | Claude-based ML workspace assistant; RAG query routing at ResMed |
| "Developing scalable APIs with FastAPI" | Async, concurrent, production-hardened | FastAPI mentioned in skills; used across TrueBalance + ResMed |
| "Creating interactive AI apps using Chainlit, Streamlit, vLLM" | Chatbots/agents + model serving | Streamlit dashboards; LLM-assisted workspaces |
| "Integrating LLMs with external APIs" | Function-calling, MCP, tool use | ML workspace integrates Jira, GitHub, Athena, Jenkins |
| "Optimizing models using quantization, pruning, distillation" | Inference optimization | Listed in skills; prepare to show you've actually done INT8/LoRA |
| "Monitoring, observability, feedback loops, drift detection" | Not just "model served" — model WATCHED | Datadog drift dashboards at ResMed; Deequ data quality at Tiger |
| "Working across AWS/Azure" | Polyglot cloud engineer | AWS (SageMaker, Lambda, VPC); Azure (Databricks, DF, ML Studio) |
| "Collaborating with data scientists and engineers" | Bridge role | Explicit in ResMed IHS — "enabled DS to integrate with minimal code changes" |

### 1.3 Hidden signals in the JD

1. **"Bridge the gap between research and real-world deployment"** — They have researchers producing notebooks and need someone to productionize. This matches your ResMed IHS work perfectly. **Lead with that story.**
2. **"Ensure models run reliably, efficiently, and at scale"** — Reliability + efficiency + scale = SRE-flavored ML. Emphasize p99 latency numbers (your <500ms XGBoost Lambda), cost-efficiency (SageMaker multi-container endpoints), and drift monitoring.
3. **"Globally growing product"** — Product company, not consulting. They want ownership, iteration, long-horizon thinking. Highlight 2-year+ engagements (ResMed 2.5yrs, Sopra 3.5yrs).
4. **No specific LLM framework named** (no "LangChain" / "LlamaIndex") — They're open to your choice. Good place to demonstrate judgement.
5. **Chainlit + Streamlit + vLLM together** — Suggests they're building **internal AI tools** or **agent UIs** on top of **self-hosted** models (vLLM = open-source serving, likely llama-3/Qwen/Mistral family).

### 1.4 Likely interview rounds (UAE / Avrioc pattern)

1. **HR/Screening (30 min)** — Notice period, compensation, visa, relocation willingness, family questions.
2. **Technical screen 1 (1 hr)** — ML fundamentals + LLM concepts + your projects. Driven by a tech lead or senior DS.
3. **Technical deep dive 2 (1-1.5 hrs)** — Pick 1-2 projects from your resume, grill deeply. Expect whiteboard/shared doc.
4. **System design (1 hr)** — "Design a RAG system for X" OR "Design an LLM-powered feature for our product."
5. **Culture + hiring manager (45 min)** — Team fit, leadership, conflict stories, why Abu Dhabi, long-term plans.
6. **(Optional) Onsite / founder round** — If they fly you to Abu Dhabi.

---

## 2. Resume-to-JD Coverage Matrix

This maps every resume bullet to JD requirements — so you can steer stories into JD-relevant lanes.

### 2.1 Resume bullet → JD requirement map

| Resume Bullet | JD Bucket | Chapter to Study |
|---------------|-----------|------------------|
| **Real-time XGBoost Lambda (p99 < 500 ms, 3-env VPC)** | Production ML, AWS, FastAPI-like APIs, scale | 11 (AWS), 15 (Resume), 16 (System Design) |
| **Lender-identification NER (29.7% → 68.0%)** | Model optimization, NLP, eval | 04 (Embeddings), 15 |
| **7-entity / 29-predicate ontology, 107 tests** | Engineering rigor, zero-diff migrations | 10 (MLOps), 15 |
| **AI-powered ML workspace on Claude (Jira+GitHub+Athena+Jenkins)** | LLM + external APIs, agent systems | 07 (RAG), 13 (LangGraph), 15 |
| **RAG-based knowledge-base chatbot (ResMed)** | LLMOps, RAG, production LLMs | 07, 13, 15 |
| **Datadog + Snowflake drift dashboards** | Monitoring, observability, drift | 14, 15 |
| **Multi-container SageMaker endpoints (IHS)** | AWS scale, cost-efficiency | 11, 15 |
| **Snowflake feature store, batch + real-time inference** | Feature stores, MLOps | 10, 11 |
| **Apache Airflow orchestration** | MLOps | 10 |
| **SageMaker drift + retraining pipelines (Tiger)** | MLOps | 10, 11, 15 |
| **Deequ data quality on Databricks (Mars)** | Azure, data quality | 11, 14 |
| **CNN + YOLO + OCR ID verification** | Classical CV | 15 (brief) |
| **Time-series anomaly on Prometheus/Grafana** | Observability | 14 |
| **Loan-risk XAI ensemble** | Classical ML, explainability | 15 |
| **OR-Tools logistics** | Optimization | 15 (brief) |

### 2.2 JD requirement → Resume evidence + bench study

| JD Requirement | Your Evidence | Gap / Extra Study |
|----------------|---------------|-------------------|
| Kubernetes | Skills list only | **Study Ch. 12 harder** — be able to explain pods, deployments, services, HPA, KEDA, node affinity, GPU scheduling |
| Ray | Skills list only | **Study Ch. 12** — Ray Core vs Ray Serve vs Ray Train vs Ray Tune |
| LLMOps practices | Claude workspace, RAG at ResMed | **Ch. 10** — LangFuse, Phoenix, prompt versioning |
| FastAPI | Skills + implied in Lambda work | **Ch. 13** — async, Pydantic v2, dependency injection |
| Chainlit | Skills list only | **Ch. 13** — know what it does (chat UI for LangChain); practice 5 min demo in head |
| Streamlit | Skills list | **Ch. 13** — stateful widgets, caching, deployment |
| vLLM | Skills list | **Ch. 13 + 09** — PagedAttention, continuous batching, quantization support |
| Quantization/pruning/distillation | Skills list (need project example) | **Ch. 09** — pick one hands-on story (e.g., INT8 a BERT for local inference) |
| AWS + Azure | Strong on AWS, thinner Azure | **Ch. 11** — brush up Azure ML, AKS |
| Drift detection | Datadog dashboards at ResMed | **Ch. 14** — Evidently, PSI formula, statistical tests |

> **Top 3 bench gaps to close:** Kubernetes (prod depth), Ray, vLLM internals. Chapters 12, 13 prioritize these.

---

## 3. The 5 "signature stories" you will reuse

Every interview asks behavioral / project questions. Prepare 5 tight STAR-format stories you can rotate. Each should take **90-120 seconds** to tell.

### Story 1 — "The Real-Time Credit-Risk Lambda"
**S:** TrueBalance needed to cut funding on loans where borrower would withdraw before full disbursal; cost the book real money every week.
**T:** Ship a production predictor with p99 < 500 ms, VPC-isolated, across 3 environments.
**A:** Built an XGBoost model served via AWS Lambda (container image), Terraform'd VPC + subnets + SGs per env, p99 instrumentation via CloudWatch + custom metrics; feature fetch from a Redis-fronted Snowflake feature store; fallback to cached features on timeout.
**R:** p99 stayed under 500 ms across 3 envs; projected portfolio profit lift from cutting high-withdraw-risk funding. **Tie to JD:** scalable APIs, AWS, drift-aware (we monitor withdraw-rate drift weekly).

### Story 2 — "NER bump from 29.7% to 68% lender accuracy"
**S:** Production lender-identification on ~109K tradelines was only 29.7% accurate, blocking a credit-reporting feature.
**T:** Ship an accuracy lift without losing any existing matches (zero-regression).
**A:** Layered an NER lender-entity extractor over the legacy 2K-term keyword expansion, added BANK-lender boosting heuristic; validated on 78K softpull + 31K hardpull tradelines, zero lost matches via intersection test.
**R:** 29.7% → 68.0% — a 129% relative lift, zero regressions. **Tie to JD:** model lifecycle, evaluation discipline, productionization.

### Story 3 — "Claude-powered ML workspace"
**S:** ML engineers spent too much time on Jira ticket isolation, GPU provisioning, Jenkins triggers — context-switching everywhere.
**T:** Build a single LLM-driven developer platform to unify these.
**A:** Claude Sonnet backbone with tool calling into Jira, GitHub, AWS Athena, Jenkins; on-demand GPU/CPU EC2 provisioning (EFS-shared state + 3-method EBS lifecycle); automated per-ticket git-worktree isolation.
**R:** ML engineer productivity up; safe parallel experimentation on shared repos. **Tie to JD:** LLM + external APIs, agent systems, LLMOps.

### Story 4 — "MLOps framework on AWS — 8 models in 6 months"
**S:** DS team at ResMed was hand-deploying each model — inconsistent, slow, error-prone.
**T:** Build a reusable MLOps framework so DS could integrate with minimal code changes.
**A:** Standardized training + inference pipelines on SageMaker; CodePipeline/CodeBuild CI; multi-container endpoints for cost efficiency; Snowflake feature store; Datadog drift dashboards auto-wired.
**R:** 8 models to production in 6 months with near-zero DS engineering time. **Tie to JD:** MLOps, AWS, collab with DS, scale.

### Story 5 — "RAG-powered clinical chatbot"
**S:** Clinical report data at ResMed was unstructured, locked in PDFs; answering even simple questions took analysts hours.
**T:** Ship a chatbot that could answer factual + analytical queries on the corpus.
**A:** RAG architecture with pgVector, domain-aware chunking (respecting clinical sections), LLM-based query router — factual → vector search; analytical → code generation; conversational → direct LLM; citations in responses.
**R:** Analyst time-to-answer cut significantly; clinical team adopted it. **Tie to JD:** RAG, LLMOps, AWS, production.

---

## 4. Your 60-Second "Tell me about yourself"

> I'm Sachin, a Senior ML Engineer with 8 years of experience productionizing ML and LLM systems across fintech, healthcare, and enterprise. At TrueBalance today I own a real-time XGBoost Lambda with p99 under 500 milliseconds and VPC isolation across three environments, plus a Claude-powered developer platform that unifies Jira, GitHub, Athena, and Jenkins for the ML team. Before that I spent two and a half years at ResMed building their MLOps platform — we shipped 8 models to production in 6 months, a RAG-based clinical chatbot, and automated drift dashboards that became the team standard. My sweet spot is the handoff from research to production: LLMOps with vLLM and Kubernetes, real-time inference, quantization and distillation for cost, and the observability that keeps models healthy at scale. That's exactly the bridge Avrioc's JD describes, which is why this onsite Abu Dhabi role excites me.

Time this at home — should land at 55-65 seconds. Memorize the first 10 and last 10 words; improvise the middle.

---

## 5. Interview strategy (one-pager)

### Do
- **Lead with numbers.** Every project statement should have a number: p99, accuracy lift, SMS coverage %, time-to-answer cut, models shipped.
- **Use the word "production" a lot.** The JD uses it twice.
- **Draw a diagram unsolicited.** For any system question, say "let me sketch this" — you will immediately stand out.
- **Say "I don't know, but here's how I'd find out."** When stumped, describe your search process — Claude, docs, GitHub issues, papers.
- **Ask 2-3 strong questions at the end.** See Ch. 17.

### Don't
- **Don't oversell Kubernetes and Ray.** Say "I've used them via team tooling; I understand the primitives, and here's how I'd operate them in production" — they'll respect honesty over fake depth.
- **Don't trash previous employers.** UAE is relationship-driven; bad-mouthing travels.
- **Don't dodge compensation questions.** Come with a tax-adjusted number in AED.
- **Don't skip the "why Abu Dhabi" prep.** Have 2-3 real reasons.

---

## 6. How to use this pack (last 14 days)

| Day | Focus |
|-----|-------|
| D-14 → D-12 | Read Ch. 02 (Transformers), 03 (LLMs), 04 (Embeddings) |
| D-11 → D-9 | Ch. 05 (Parameter tuning), 06 (Fine-tuning), 07 (RAG) |
| D-8 → D-6 | Ch. 09 (Optimization), 10 (MLOps), 12 (K8s/Ray) |
| D-5 → D-4 | Ch. 11 (Cloud), 13 (Frameworks), 14 (Monitoring) |
| D-3 | Ch. 15 (Resume deep dive) — rehearse stories out loud |
| D-2 | Ch. 16 (System design) — do 2 mock designs |
| D-1 | Ch. 17 (Behavioral), Ch. 18 (Cheatsheet) — light review only |
| Day of | Re-read Ch. 18. Eat. Sleep 8 hrs. |

---

Continue to **[Chapter 01 — Foundations](01_foundations.md)**.
