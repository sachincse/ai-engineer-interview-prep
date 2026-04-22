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
