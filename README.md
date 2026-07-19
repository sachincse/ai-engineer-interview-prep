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

## Company-specific packs

Beyond the 18 core chapters, `interview/` holds role- and company-specific prep banks (chapters 19+). The most comprehensive recent one:

| # | Pack | Covers |
|---|------|--------|
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
