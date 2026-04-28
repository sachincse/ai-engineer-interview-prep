# Chapter 23 — High-Probability Q&A Bank (Avrioc-Tuned)

> **Why this chapter exists:** Below are the 40 questions most likely to be asked given (a) the Avrioc JD keywords, (b) the public Avrioc interview pattern, (c) standard MLOps/LLMOps interview banks. Each has a model answer **calibrated to your resume** so you can speak with conviction, not generic textbook phrases.

**Format:** Each Q has a 1–2 sentence answer + the deeper "if they push" follow-up. Read out loud. Adjust the wording so it sounds like *you*, not a textbook.

---

## A. vLLM & LLM serving (very high probability — JD names vLLM)

### Q1. What is PagedAttention and why does it matter?
**Short:** It's vLLM's KV-cache management — borrowed from OS virtual memory. Each request's KV cache is broken into fixed-size blocks (default 16 tokens) that don't have to be contiguous in GPU memory. Fragmentation drops from ~60–80% to <4%, which means 2–4x more concurrent requests on the same GPU.
**If pushed:** Explain logical-vs-physical block tables, copy-on-write for prefix sharing across siblings (beam search, parallel sampling), and that block size is a tunable trade-off — smaller blocks = less internal frag but more bookkeeping overhead.

### Q2. What is continuous batching?
**Short:** Iteration-level scheduling — after every forward pass, the scheduler can add new sequences to the running batch and remove ones that finished. So a short request doesn't have to wait for a long one in the same batch to complete.
**If pushed:** Mention that vLLM's scheduler maintains three queues — `running`, `waiting`, `swapped` (when KV cache pressure forces eviction). Compare to static batching (HuggingFace's default), which idles the GPU until the slowest sequence in a batch finishes — wasteful for chat workloads with very different output lengths.

### Q3. How do you deploy a 70B model on a node of 8x A100s?
**Short:** Tensor-parallel = 8 within the node so NVLink carries the high-bandwidth all-reduce traffic. With FP16 a 70B model is 140 GB weights, so it just fits across 8x80GB GPUs with KV-cache headroom. I'd also enable INT8 (AWQ or GPTQ) if I can sacrifice 1-2 points of quality for 2x throughput.
**If pushed:** Talk through `--tensor-parallel-size 8 --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.9` as actual vLLM CLI flags. Mention chunked prefill if the use-case is chat (lowers TTFT), or speculative decoding with a draft model.

### Q4. TTFT vs ITL vs throughput — when does each matter?
**Short:** Chat → TTFT (time to first token) and ITL (inter-token latency); anything below ~50 ms ITL feels real-time. Batch summarization → throughput (tokens/sec) is king and TTFT is irrelevant.
**If pushed:** Add p95/p99 framing, not just p50 — chat UX is dominated by tail latency. The reason to use *chunked prefill* in vLLM is to interleave prefill with decode tokens, smoothing the latency for in-flight chat requests when a new long-context user joins.

### Q5. What's the difference between vLLM and SGLang / TensorRT-LLM?
**Short:** vLLM is general-purpose with PagedAttention + continuous batching, easy to deploy. SGLang adds RadixAttention for prefix caching across requests — very fast for agents/chat with shared system prompts. TensorRT-LLM is NVIDIA's optimized backend with the best raw H100 throughput but tighter tooling.
**If pushed:** I'd reach for vLLM by default; SGLang for an agent product where the system prompt is identical across users; TRT-LLM if I have NVIDIA support and need every last token/sec.

---

## B. Ray (high probability — JD names Ray)

### Q6. When would you choose Ray Serve over plain Kubernetes Deployments?
**Short:** When you have **multi-stage AI pipelines** that need to compose together — a RAG flow with embed → retrieve → rerank → generate is naturally a Ray Serve graph with each stage as a deployment. Ray Serve gives you fractional GPUs, dynamic batching, and per-stage autoscaling for free.
**If pushed:** For a single model with simple in/out, plain K8s + KEDA is simpler and I'd pick that. The Ray Serve sweet spot is when you have a DAG, not a single endpoint.

### Q7. Ray actors vs tasks?
**Short:** Tasks are stateless — `@ray.remote` on a function. Actors are stateful — `@ray.remote` on a class, get their own process, and can hold a model in GPU memory across many calls. For ML serving, actors win because you load weights once.
**If pushed:** Trade-off is fault tolerance — if an actor process dies, its state is gone (you can mitigate with checkpointing). Tasks are restarted automatically by Ray.

### Q8. How does Ray Serve LLM integrate with vLLM?
**Short:** Ray Serve LLM wraps vLLM as a Ray Serve deployment, exposing OpenAI-compatible endpoints. You get vLLM's serving optimizations *plus* Ray Serve's composition — so a RAG node + a vLLM node share one autoscaling layer.
**If pushed:** The newer Ray Serve has *prefill-decode disaggregation* — you run prefill (compute-bound) and decode (memory-bound) on separate replica pools, scale them independently, and shave p99 by 30%+.

---

## C. Kubernetes (very high probability — JD names K8s + EKS)

### Q9. Walk me through deploying a vLLM service on K8s.
**Short:** Containerize vLLM (NVIDIA base image), set `resources.limits.nvidia.com/gpu: 1`, install the NVIDIA GPU operator on the cluster, weight loading via init container pulling from S3 to a PVC. Liveness probe on `/health`, readiness on `/v1/models`, KEDA autoscaler on `vllm:num_requests_waiting` from Prometheus.
**If pushed:** PodDisruptionBudget so an HPA scale-down doesn't kill all replicas at once; `terminationGracePeriodSeconds: 60` and a `preStop` hook that calls vLLM's drain endpoint so in-flight requests finish before pod terminates.

### Q10. How do you handle 140 GB model weights in K8s?
**Short:** Three options: (a) init container `s5cmd` to PVC at startup, (b) bake weights into a "fat" image with eStargz/SOCI for lazy pull, (c) shared filesystem (Fluid/JuiceFS/EFS) where one download serves many pods.
**If pushed:** I prefer (a) for simplicity; (c) when I have many replicas downloading simultaneously and the egress cost or startup time hurts. Option (b) is brittle — the image is huge and pinned to weight version.

### Q11. HPA vs KEDA for AI inference?
**Short:** HPA scales on CPU/memory by default — wrong for GPU inference because the GPU is already pegged at full load. KEDA can scale on **any metric**, including queue depth from Prometheus or RabbitMQ length. Always KEDA for inference.
**If pushed:** Custom metric like `vllm:num_requests_waiting > 50` is the right signal — it's a leading indicator of latency degradation, whereas GPU util is a lagging indicator.

### Q12. Rolling update without dropping in-flight LLM requests?
**Short:** maxSurge=1 + maxUnavailable=0, terminationGracePeriodSeconds long enough for the longest expected response (~120s), preStop hook hits a `/drain` endpoint that stops accepting new requests but finishes current ones, readiness gate flips to false on drain so service mesh stops routing.

---

## D. RAG, LangChain, agents (very high probability — JD names LangChain + LLM+API integrations)

### Q13. Walk me through your RAG architecture at ResMed.
**Short:** Knowledge-base chatbot for clinical reports. LLM-based query router classified queries into three buckets — factual (vector retrieval), analytical (code generation against Snowflake), conversational (direct LLM). Used pgVector for retrieval, domain-aware chunking respecting clinical sections, citations in responses.
**If pushed:** Talk through how the router was prompted (few-shot with examples), evaluation (RAGAS — faithfulness, answer relevance, context precision), what failed (initial naive chunking split sentences mid-clinical-finding; fixed by section-aware chunker).

### Q14. Naive RAG vs Advanced RAG?
**Short:** Naive = chunk + embed + cosine retrieve + stuff into prompt. Advanced adds: query rewriting, hybrid search (BM25 + dense), reranking (Cohere or BGE-reranker), sentence-window retrieval, parent-document retrieval, query routing.
**If pushed:** Each stage adds cost. Trade-off: naive RAG with a good embedding model often beats badly-tuned advanced RAG. I'd add complexity only when offline RAGAS evals show the bottleneck (low context_precision → add reranker; low context_recall → add hybrid + query rewrite).

### Q15. How do you evaluate a RAG system?
**Short:** Offline — golden Q/A set + RAGAS metrics (faithfulness, answer relevance, context precision, context recall). Online — thumbs-up/down from users, LLM-as-judge with Claude on a sample, latency dashboards.
**If pushed:** Faithfulness vs answer relevance — easy to confuse. Faithfulness = does the answer come from the context (hallucination check). Answer relevance = does the answer actually address the question. Both can fail independently.

### Q16. What is agentic RAG and when do you skip it?
**Short:** The LLM decides if/when/what to retrieve, often multi-hop. Skip it when latency budget is tight or queries are uniform — adds 2-5x latency for marginal gains.
**If pushed:** Self-RAG / Corrective-RAG variants add reflection loops — LLM grades retrieved docs and re-retrieves if low-quality. Useful when retrieval quality varies wildly. I'd default to a simple grade-and-rewrite loop, not full agentic, for production.

### Q17. How does LangChain compare to LangGraph?
**Short:** LangChain = chains (linear or simple branches). LangGraph = stateful graph with explicit nodes/edges, conditional routing, persistence, human-in-the-loop. For anything beyond a single-shot chain, LangGraph is the right primitive — LangChain shines for prototypes.
**If pushed:** Mention that LangGraph compiles to a graph that can be checkpointed (durable execution), which is what I lean on for multi-step agents that can resume after failure.

---

## E. FastAPI / Python production (high probability)

### Q18. Sync vs async endpoints in FastAPI — when does each win?
**Short:** Async wins **only if your downstream is async** — httpx instead of requests, asyncpg instead of psycopg, motor instead of pymongo. CPU-bound ML inference inside an async handler will block the event loop and stall every other request. Use `run_in_threadpool` or offload to a worker (Ray, Celery, Triton).
**If pushed:** A common bug — `def predict(req)` (sync) vs `async def predict(req)` (async). Sync handlers run in a threadpool by FastAPI design, so they DO get concurrency. The lurking failure mode is mixing async handler + sync ML call.

### Q19. Production checklist for a FastAPI service?
**Short:** Pydantic v2 models on every endpoint; structured JSON logs with `request_id` correlation (structlog); OpenTelemetry tracing; Prometheus middleware; rate limiting via slowapi; gunicorn + uvicorn workers (`2*CPU + 1`); health/ready endpoints distinguished; request body size limits; graceful shutdown.
**If pushed:** /health vs /ready — health = process is up; ready = dependencies are healthy (model loaded, DB reachable). K8s uses readiness to gate traffic; liveness to restart.

### Q20. FastAPI streaming for LLMs — how?
**Short:** `StreamingResponse` with an async generator yielding SSE-formatted chunks. Crucially, check `request.is_disconnected()` inside the generator so you can free the upstream LLM call when client disconnects.
**If pushed:** Mention that the OpenAI-compatible API spec uses `text/event-stream` and `data: {...}\n\n` framing; HTTPX with `client.stream()` is the right upstream client.

---

## F. PyTorch / fundamentals (mid probability — JD names PyTorch/TF)

### Q21. Why scale by sqrt(d_k) in attention?
**Short:** Without scaling, dot-product variance grows with d_k. Large dot products push softmax into saturation (one element ≈ 1, rest ≈ 0), gradients vanish. Scaling by 1/√d_k keeps variance ≈ 1.
**If pushed:** Quick sketch: if Q and K have unit-variance components, `Q·K = sum of d_k products`, variance scales as d_k, std as √d_k.

### Q22. Why are BF16 and FP16 different, and which do you pick?
**Short:** BF16 has the same exponent range as FP32 (8 bits) with reduced mantissa (7 bits). FP16 has shrunken range (5-bit exp). Result: FP16 trains require a `GradScaler` to avoid underflow; BF16 doesn't. Default to BF16 on A100/H100.
**If pushed:** FP16 has slightly better precision when in range — useful for inference where range is bounded; BF16 wins for training. FP8 (H100+) is now common for inference too.

### Q23. What is FlashAttention?
**Short:** Same attention math, but tiles Q/K/V into SRAM-resident blocks, computes softmax block-wise, and **recomputes** the attention matrix during backward instead of storing it. O(N) memory instead of O(N²), 2-4x faster end-to-end.
**If pushed:** FlashAttention-2 added better work partitioning across warps; FlashAttention-3 (H100) uses TMA + warp specialization for FP8.

### Q24. `torch.compile` — when does it help and when does it break?
**Short:** Helps on transformer training/inference by graph-capturing common patterns and fusing kernels (~30-50% speedup). Breaks on dynamic shapes (use `mark_dynamic`), data-dependent control flow (`if x.sum() > 0`), and unregistered custom ops.
**If pushed:** Mention `torch.compile(model, mode="reduce-overhead")` for inference, `mode="max-autotune"` for training. For LLM serving, vLLM has its own kernel layer so compile less relevant there.

---

## G. MLOps and observability (very high probability — your Datadog dashboard story)

### Q25. How do you detect data drift in production?
**Short:** Population Stability Index (PSI) on numerical features against a training reference; KS test for distribution shape; embedding drift via cosine distance from training centroid for text. Tooling: Evidently AI, Arize Phoenix, or my custom Datadog/Snowflake dashboards (which I built at ResMed).
**If pushed:** PSI math: `PSI = sum((p_i - q_i) * ln(p_i / q_i))` over bins. >0.25 = significant drift. Action: investigate feature engineering pipeline, data source, then retrain.

### Q26. Concept drift vs data drift — what's different?
**Short:** Data drift = `P(X)` changed, inputs are different. Concept drift = `P(y|X)` changed, the relationship between inputs and target shifted. Data drift you can sometimes fix with re-weighting; concept drift always requires fresh labels and retraining.
**If pushed:** Concept drift is sneakier because input distributions can look stable while the label semantics shift (e.g., COVID changed what "normal" lab values meant for many models). Use prediction-vs-ground-truth deltas as the canary.

### Q27. How do you A/B test two LLM versions?
**Short:** Shadow mode → 1% canary → 5% → 50% → 100%. Metrics: TTFT, refusal rate, output length, user thumbs, LLM-as-judge head-to-head win rate. Auto-rollback on guardrail breach (toxicity, refusal-rate spike).
**If pushed:** LLM-as-judge variance is real — sample more (n>200), use Claude-Opus-class judge, anchor with reference answers, and prefer pairwise comparisons over absolute scoring.

### Q28. Walk me through the drift dashboard you built at ResMed.
**Short:** Data Science team computed drift metrics (PSI, KS) in Python; I built a utility that wrapped their logic, ran it on Snowflake-stored prediction logs, pushed metrics to Datadog as custom metrics with feature/model tags, and auto-generated dashboards per model. Became the team standard.
**If pushed:** Hardest part: **schema drift** — different models had different feature sets, so the dashboard generation had to be metadata-driven, not hand-built per model. Used a YAML config per model + a generator script.

---

## H. AWS specifically (your strongest cloud — they'll lean here)

### Q29. SageMaker single endpoint vs multi-container — when?
**Short:** Single = one container, one model, simple. Multi-container = up to 15 models on one endpoint, sharing infrastructure cost. Used multi-container at ResMed IHS to consolidate 8 lower-traffic models — cost dropped without latency impact because GPU was underutilized per-model.
**If pushed:** Multi-model endpoints (different from multi-container) load models on demand into one container — cheaper still but cold-start hit on first invocation. Choice depends on traffic shape.

### Q30. Walk me through your real-time XGBoost Lambda architecture.
**Short:** API Gateway → Lambda (container image with XGBoost in Python) → Redis (online feature cache) → fallback to Snowflake on cache miss → CloudWatch custom metric for p99 instrumentation. VPC-isolated across dev/staging/prod, Terraform-managed.
**If pushed:** The 500ms p99 budget broke down: 100ms Lambda cold start (cold) or ~10ms (warm), 50ms Redis fetch, 100ms model inference, 50ms response build. Snowflake fallback was a circuit-breaker — last-known features, not a synchronous fetch.

### Q31. Lambda for ML — when does it fail?
**Short:** Three failure modes: (1) cold-start with large containers (>1 GB) — provisioned concurrency or stay warm with EventBridge ping; (2) GPU-required inference — Lambda doesn't have GPU, switch to ECS/EKS or SageMaker; (3) >15 min runtime — switch to Step Functions or batch.
**If pushed:** Lambda with container images can be up to 10 GB image / 10 GB memory — fits most CPU-side ML, including XGBoost, sklearn, small transformers (DistilBERT). For LLMs, Lambda is wrong.

---

## I. Behavioral / situational (Avrioc-confirmed reverse-chronology)

### Q32. Tell me about yourself.
**Short:** Use the 60-second version from [Chapter 00 §4](00_index.md). Lead with current role (TrueBalance + XGBoost Lambda + Claude workspace), then ResMed (8 models in 6 months), then close with the Avrioc-relevant skills (LLMOps, vLLM, K8s) and why Abu Dhabi.

### Q33. Walk me through your projects in reverse chronological order.
**Short:** Use the structured drill in [Chapter 19 §4](19_avrioc_company_intel.md#4-the-reverse-chronology-project-walkthrough-drill-critical). 4-5 min on TrueBalance, 3-4 min on ResMed, 1.5-2 min on Tiger, 1 min on Sopra Steria.

### Q34. Tell me about a production incident you owned.
**Short:** Pick **one specific story**. Suggested: at ResMed, a SageMaker endpoint started returning stale predictions because the feature store backfill cron silently failed. I caught it via the Datadog drift dashboard alerting on a sudden distribution shift, traced it back to the ETL job, fixed it, then added a freshness SLO with a separate alert.
**If pushed:** "Five whys" to root cause: ETL job → cron schedule misconfigured after a region migration → no monitoring on cron success → only data drift caught it. Added cron success alert as the long-term fix.

### Q35. Tell me about a technical decision you regret.
**Short:** Show learning. Suggested: at TrueBalance early on, I used Lambda layers for ML deps; layer size limit (250 MB) bit me when I added one library. Migrated to container images. Lesson: pick the artifact format based on the **upper bound** of dependencies, not the current state.

### Q36. How do you handle disagreement with a tech lead?
**Short:** Restate their position to confirm I understand it, share my reasoning with concrete trade-offs (numbers if possible), and ask what would change their mind. If we still disagree and they're the decision-maker, I commit and move forward — disagree-and-commit. Avoid making it personal.

### Q37. Why are you leaving TrueBalance after only 3 months?
**Short:** Honestly — Avrioc's onsite Abu Dhabi role with visa sponsorship is a major life and career change my family has been planning toward. The role specifically matches my LLMOps strength on Kubernetes + vLLM + Ray, which is the next step I want my career to take. TrueBalance has been a great chapter but wasn't designed around relocation.
**If pushed:** Don't disparage TrueBalance. Reinforce: this was always the plan; the timing aligned.

### Q38. Why Abu Dhabi?
**Short:** Three reasons — (1) tax-free compounding lets me build long-term family financial security, (2) UAE's national AI strategy is creating real demand for production ML talent, the kind of work I want to do, (3) onsite cross-functional teams produce faster learning velocity than remote — and at this stage of my career, learning velocity matters more than convenience.

### Q39. What would you build first in your first 90 days?
**Short:** Listen first 30 days — sit with the data scientists and product owners, understand the pipelines and pain points. Days 30-60: pick one product (maybe MyWhoosh, given the telemetry richness) and ship one concrete LLMOps win — could be a vLLM-based serving migration, a drift dashboard for an existing model, or a RAG-powered internal tool. Days 60-90: turn that into a reusable platform pattern others can adopt.
**If pushed:** Avoid "I'd rebuild everything" energy. The first quarter is about earning trust and shipping one meaningful thing, not re-architecting the world.

### Q40. Do you have any questions for us?
**Short:** Yes. Use the 3 prepared questions from [Chapter 19 §8](19_avrioc_company_intel.md#8-the-3-questions-you-ask-them-at-the-end):
1. Slurm-vs-K8s split in current infra?
2. Of MyWhoosh / Comera / Labaiik / Hyre, which AI roadmap is the team's biggest current focus?
3. What does success look like for this role in 6 months?

---

## How to use this chapter

1. **First pass:** read every question, answer it out loud (don't skim the model answer). Note where you stumble.
2. **Second pass:** drill the 5-10 you stumbled on. Use the deeper chapters to backfill.
3. **Day-of:** read **only the section headings and your stumble list** — not the full chapter.

You're not memorizing; you're **rehearsing fluency**. The goal is that when one of these questions comes up, your mouth knows what to say while your brain works on the next thing.

---

End of pack. Continue back to **[Chapter 00 — Master Index](00_index.md)** to navigate other chapters.
