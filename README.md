# AI Engineer Interview Prep Pack

> Comprehensive interview-preparation pack for a **Senior AI Engineer / ML Solutions Architect / MLOps & LLMOps** role.
> Originally compiled for an onsite role at Avrioc Technologies (Abu Dhabi, UAE). Generic enough to reuse for any senior AI engineering interview at a product company with a modern LLM stack (vLLM, Kubernetes, Ray, LangGraph, RAG).

## What's inside

18 chapters, ~50,000 words, ~200 printed pages of depth-first content. Every chapter has: concepts, block diagrams, math, code snippets, 20–30 Q&A, and "gotcha" traps.

| # | Chapter | Covers |
|---|---------|--------|
| 00 | [Master Index & JD Alignment](interview/00_index.md) | JD analysis, resume→JD mapping, 5 signature stories |
| 01 | [Foundations](interview/01_foundations.md) | Neural nets, activations, Word2Vec / GloVe / FastText |
| 02 | [Transformer Architecture](interview/02_transformers.md) | Attention, MHA/MQA/GQA/MLA, RoPE, YaRN, ALiBi, FlashAttention |
| 03 | [How LLMs Work](interview/03_llms.md) | Tokenization, pretraining, SFT, RLHF, DPO, KV-cache, inference |
| 04 | [Embedding Models](interview/04_embeddings.md) | Contrastive learning, InfoNCE, BGE/E5/ColBERT, Matryoshka |
| 05 | [LLM Parameter Tuning](interview/05_parameter_tuning.md) | Temperature, top-p/k, min-p, penalties, beam, mirostat |
| 06 | [Fine-tuning](interview/06_fine_tuning.md) | LoRA, QLoRA, DoRA, PEFT, DPO recipes |
| 07 | [RAG](interview/07_rag.md) | Naive → advanced, hybrid, HyDE, rerank, GraphRAG, RAGAS |
| 08 | [Vector Databases](interview/08_vector_databases.md) | HNSW/IVF/PQ, pgVector/Qdrant/Pinecone/Milvus |
| 09 | [Model Optimization](interview/09_model_optimization.md) | GPTQ/AWQ/GGUF/SmoothQuant/FP8, Wanda, distillation |
| 10 | [MLOps & LLMOps](interview/10_mlops_llmops.md) | MLflow, feature stores, Langfuse, guardrails, cost cascades |
| 11 | [AWS & Azure](interview/11_aws_azure.md) | SageMaker, Lambda, Bedrock, VPC, Databricks, Azure ML |
| 12 | [Kubernetes, Ray, Docker](interview/12_kubernetes_ray.md) | K8s primitives, KServe, KEDA, Ray Core/Serve/Train/Tune |
| 13 | [Frameworks](interview/13_frameworks.md) | FastAPI, LangChain, LangGraph, CrewAI, vLLM, Chainlit, Streamlit |
| 14 | [Monitoring & Drift](interview/14_monitoring_drift.md) | KS/PSI/Wasserstein, Evidently, Datadog, closed-loop retraining |
| 15 | [Resume Deep Dive](interview/15_resume_deep_dive.md) | Every resume bullet with STAR + technical drill-downs |
| 16 | [System Design](interview/16_system_design.md) | 4 full designs: RAG, real-time inference, multi-LoRA, streaming agent |
| 17 | [Behavioral & HR](interview/17_behavioral_hr.md) | 15 behavioral Qs, UAE specifics, compensation negotiation |
| 18 | [Cheatsheet](interview/18_cheatsheet.md) | Formulas, numbers, commands, names — morning-of revision |

## Domain & role-specific packs

Beyond the 18 core chapters, `interview/` holds domain-, role- and company-specific prep banks (chapters 19+). The most comprehensive recent ones:

| # | Pack | Covers |
|---|------|--------|
| **48–53** | **Patent & Prior-Art AI — Novelty Checking & Design-Around** — [48 Orientation & strategy](interview/48_patent_prior_art_ai_orientation.md) · [49 Domain primer](interview/49_patent_domain_primer_for_ai.md) · [50 System design](interview/50_prior_art_novelty_system_design.md) · [51 Measurement & evaluation](interview/51_novelty_measurement_and_evaluation.md) · [52 Q&A bank](interview/52_patent_ai_qa_bank.md) · [53 Explain it simply](interview/53_explaining_prior_art_ai_simply.md) · **[runnable demo →](examples/prior_art_demo/)** | Six-chapter domain pack (~50k words) **plus working code** on building AI for **patent prior-art search, novelty assessment and design-around analysis** — written for an ML engineer with no patent-law or chemistry background. **48:** how to scope an unfamiliar domain fast — the three discovery questions, cost-asymmetry framing, a traps list, a 30-minute conversation plan. **49:** novelty (EPC Art. 54) vs inventive step (Art. 56), the X/Y/A/E/P examiner citation categories, the **18-month publication blackout** and Art. 54(3) secret prior art, DOCDB vs INPADOC families, IPC/CPC, selection inventions and the Art. 123(2) added-matter constraint, plus the chemistry layer — **Markush structures**, SMILES/InChIKey/SMARTS, fingerprints, **Tanimoto and its `T ≤ min(a,b)/max(a,b)` size bound**, OPSIN/OCSR/chemical NER — and a free-vs-licensed data map with the embedding/licensing trap. **50:** a whiteboard-able reference architecture — nine retrieval channels (BM25, dense, structure, Markush, graph, metadata, citation) fused with RRF, four-granularity chunking, the **element × document coverage matrix** that turns "one document kills novelty" into an objective function, the design-around/white-space module, an allowed/forbidden table for the LLM, an MCP tool layer, EU AI Act & confidentiality patterns, build-vs-buy, and a phased plan with kill criteria. **51:** **TF-BIDF** and the full Kelly–Papanikolaou–Seru–Taddy construction, Hall's exact Herfindahl bias correction, Uzzi atypicality, Trajtenberg originality/generality, calibration (Platt/isotonic, ECE/Brier), asymmetric-cost review-depth thresholds, **PRES**, Lincoln–Petersen/Chapman and **Chao1** for estimating the recall you cannot observe, TAR stopping rules, and the temporal-leakage traps that make offline numbers lie. **52:** ~75 answered questions (RAG, agents/MCP, knowledge graphs, MLOps, architecture judgement, STAR) + 20 to ask back. **53:** the plain-language version — the whole system explained without jargon, with block diagrams, a fully worked example, a three-minute narration script, and the obvious questions answered simply. **[`examples/prior_art_demo/`](examples/prior_art_demo/):** a dependency-free Python implementation you can actually run (`python run_demo.py`) — three independent retrieval channels with RRF fusion, the EPC date engine, the element × document coverage matrix with quoted evidence spans, Chao1 recall estimation and white-space mapping, over a toy corpus of 12 fictional documents, with 25 tests |
| **54** | **[Smartsheet · Senior AI/ML Ops Engineer — Python CoderPad round](interview/54_smartsheet_python_coderpad.md)** | ~880 KB / 250 runnable stdlib-only Python blocks, every one executed on 3.11 before commit. A **provenance-first** pack for a 60-minute CoderPad round: **§0** separates what candidates actually report (Glassdoor/LeetCode/AlgoDaily/TechPrep — ~60 min shared editor, 10/40/10 split, difficulty ~3/5, arrays-strings-hashmap bias, eleven problems reported by name) from what is inference, and covers the reported quirk that **some interviewers invite you to use an LLM for a first pass then grade how you test, explain and refactor it**. **§1** CoderPad mechanics + a minute-by-minute playbook + narration scripts. **§2** Smartsheet's object model (Workspace/Sheet/Column/Row/Cell, cross-sheet refs, dependencies, automation) and what permission-aware RAG over sheets demands. **§3** 60+ Python-internals questions (data model, mutability traps, GIL, generators, decorators, MRO, asyncio, typing) — six blocks were rewritten because their printed output contradicted their own claim, including the race-condition demo that reliably shows **zero** lost updates on 3.11. **§4** 22 classics, cross-checked against brute-force references. **§5** grid/formula/dependency-graph problems (inference only). **§6** MLOps utilities — retry/backoff, token bucket, TTL-LRU, size-aware batching, PSI/KS drift, train/serve feature-parity checker, a 60-line DAG scheduler. **§7** OOP design, 12 broken snippets, testing fast. **§8** STAR stories + honesty guardrails. **§9** hour-of cheatsheet |
| **55** | **[Smartsheet · Senior AI/ML Ops Engineer — Round 2: AI/ML Ops Design & Domain](interview/55_smartsheet_r2_design_domain.md)** | ~198k words. The round-2 companion to Ch.54, for an invite tagged `COMPETENCY ASSIGNMENT: AI/ML Ops Design and Domain` in front of a two-peer IC panel. **§0** decodes the round and gives a 60-minute playbook in three variants plus the pad skeleton. **§1** company delta for a design round. **§2** one complete 40-minute answer to “design the AI/ML Ops platform”, arithmetic included. **§3–§4** twelve more worked designs. **§5–§8** four domain primers — Databricks (Unity Catalog, MLflow 3, Mosaic AI, Delta, Asset Bundles, Lakehouse Monitoring) with a translation table onto AWS/OSS equivalents; AWS Bedrock and the AWS ML plane plus Monte Carlo, the observability landscape and the drift statistics; Kubernetes, serving patterns, promotion gates, Terraform and design-to-cost; production RAG, permission-aware multi-tenant retrieval, vector stores, GraphRAG, agent operations and LLMOps governance. **§9** 205 probes organised by JD line, 25 flagged highest-probability, 10 debugging ladders, 5 SQL exercises. **§10** platform stories, the gap playbook, the consistency sheet. **§11** hour-of cheatsheet. Every section adversarially reviewed and revised before commit; disclosed tool gaps are stated as gaps, never implied as experience |
| **56** | **[Smartsheet Round 2 — Simple English, with diagrams](interview/56_smartsheet_r2_simple_english.md)** | ~77k words, **61 Mermaid diagrams + 66 ASCII diagrams**. The same round written to be understood fast rather than read exhaustively: short sentences, every piece of jargon explained on first use, and a picture for almost every idea. Each diagram appears twice — Mermaid for reading here, ASCII under 76 columns for typing into a shared editor mid-answer. Every question uses one drillable shape: *the question as an interviewer would say it → what they are really checking → a short answer you can speak → what to say if they push*. **§0** the panel and what a peer-IC pair changes, plus what the company's own job board reveals about the team. **§2** one master diagram of the seven planes of an ML platform. **§3** 14 core ideas (registry, CI/CD for ML, feature stores, train/serve skew, drift, PSI with a worked example, serving patterns, Kubernetes, RAG, vectors, knowledge graphs, agents, gateways, cost). **§4–§5** Databricks and AWS/Bedrock/monitoring in plain English, with the honest gap scripts. **§6** 60 likely questions, 15 starred. **§7** five full designs. **§8** stories, gap answers, numbers sheet, questions to ask, pre-flight |
| 47 | [Production ML on AWS · 2nd-Round Technical Q&A Bank](interview/47_production_ml_aws_technical_round.md) | Full model answers to a 21-question production/MLOps 2nd-round: model monitoring (4 layers) & metrics (RMSE/MAE, Precision/Recall/F1), **drift detection** (PSI/KS, data vs concept vs label), safe **model updates** (shadow→canary→**blue-green**), **rollback across build/prod accounts** (traffic-weight flip, no rebuild), **IaC/Terraform**, **load balancing & auto-scaling** (ALB/NLB, SageMaker/EC2/Lambda scaling), **job scheduling** (Airflow/Glue/EventBridge, idempotency & debugging), **latency vs throughput** (Little's Law, read-heavy trade-off), and **orchestrating many models + ETL as one system** (DAG control plane + lake/feature-store/registry data plane, data-flow, dependencies, **primary keys & join correctness**, system dashboards). Each answer in spoken-interview form with gotchas + a morning-of one-liner sheet |
| 46 | [Logic20/20 · Offshore Senior MLE (SDG&E Vegetation Management)](interview/46_logic2020_senior_mle_sdge_veg.md) | Full 2-round notebook for a wildfire vegetation-management MLE seat: Logic20/20 + SDG&E intel (VRI, TreeVision, WiNGS, WMP/OEIS/CPUC, HFTD, grow-in vs fall-in, PSPS), the vegetation-ML domain (LiDAR point clouds, aerial/satellite segmentation, encroachment risk, geospatial stack), resume→role skill-map with **honest-gap bridges**, a whiteboard-able reference architecture, worked system design, live-coding, **per-interviewer game plans**, a ~54-question bank, market context, a morning-of cheatsheet, and a **do-NOT-state-as-fact** honesty section |
| 45 | [Google · Staff AI/ML Engineer, YouTube Create](interview/45_google_youtube_create_staff_aiml.md) | ~50-page L6 pack: JD decode, YouTube Create product & tech intel, the full interview loop, coding bank, CV/video ML, diffusion (Imagen/Veo), ASR & audio, **on-device/mobile ML**, **6 worked ML system designs** with diagrams, Googleyness & leadership, a mock design transcript, and a 40-question rapid-fire bank |

## Suggested 2-week study plan

| Day | Focus |
|-----|-------|
| D-14 → D-12 | Ch 02 (Transformers), 03 (LLMs), 04 (Embeddings) |
| D-11 → D-9 | Ch 05 (Parameters), 06 (Fine-tuning), 07 (RAG) |
| D-8 → D-6 | Ch 09 (Optimization), 10 (MLOps), 12 (K8s/Ray) |
| D-5 → D-4 | Ch 11 (Cloud), 13 (Frameworks), 14 (Monitoring) |
| D-3 | Ch 15 (Resume) — rehearse out loud |
| D-2 | Ch 16 (System design) — 2 mock designs |
| D-1 | Ch 17 (Behavioral), 18 (Cheatsheet) — light review |
| Day of | Re-read Ch 18. Eat. Sleep 8 hrs. |

## Disclaimer

- Content reflects understanding as of 2026-04. Fast-moving space — verify model names, benchmarks, and prices before quoting in an interview.
- Project narratives in Chapter 15 describe one candidate's specific experience; adapt them to your own resume.
- Not affiliated with Avrioc Technologies or any company mentioned. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE).
