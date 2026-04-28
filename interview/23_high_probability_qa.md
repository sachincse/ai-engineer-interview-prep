# Chapter 23 — High-Probability Q&A Bank (Avrioc-Tuned)

> **Why this chapter exists:** This is the chapter you will use most on the morning of Thursday's interview. It contains the questions most likely to come up given (a) the Avrioc JD keywords, (b) the public Avrioc interview pattern, and (c) the standard MLOps/LLMOps interview banks I cross-referenced from Glassdoor, GitHub interview repos, and Medium write-ups.
>
> Every answer is written as a **full narrative the candidate can verbalize** — not a bullet list, not a "short version" with caveats. Read each answer aloud at least once. Adjust the wording so it sounds like *you* rather than a textbook. The goal is fluency: when the question lands in the interview, your mouth should already know the rhythm of the answer while your brain works ahead.

---

## How to use this chapter

There are roughly fifty questions across nine sections. Don't try to memorize all of them. Instead, do this:

1. **Tuesday evening (today)**: Read every question and every answer once, slowly, out loud. Mark with a star any question where you stumbled, where the answer felt foreign, or where you couldn't tie it to your resume.
2. **Wednesday afternoon**: Re-read only the starred questions. For each, also re-read the deeper chapter that backs it. The goal is to fill the gap, not to memorize the text here.
3. **Thursday morning**: Read only the section headings and your starred-question list. Five-minute refresh, no more.

You're not memorizing — you're rehearsing fluency. When a question lands on Thursday, you want your mouth to know what to say so your brain can plan the next sentence.

---

## A. vLLM and LLM serving (very high probability — vLLM is named in the JD)

### Q1. What is PagedAttention and why does it matter?

PagedAttention is the core innovation that vLLM brought to LLM serving, and it borrows the virtual-memory idea from operating systems and applies it to KV-cache management. The problem it solves is that without it, you have to allocate a contiguous KV-cache block sized to the maximum context length for every request, even if the actual generation turns out to be short. That wastes sixty to eighty percent of GPU memory through fragmentation. PagedAttention breaks each request's KV cache into fixed-size blocks of typically sixteen tokens, and these blocks don't have to be contiguous in physical memory — vLLM maintains a logical-to-physical block table per request, just like a page table in an operating system. The result is that fragmentation drops to under four percent, which means you can fit two to four times more concurrent requests on the same GPU. It also enables prefix sharing: if two requests share the first part of a prompt — like a system prompt — they can share the same physical blocks via copy-on-write, which is huge for chat applications where the system prompt is identical across thousands of users.

**Resume tie-in:** I haven't deployed vLLM at scale at TrueBalance — we use simpler XGBoost models there — but I've studied this carefully because it's named in the Avrioc JD, and our LangGraph workspace assistant would benefit substantially if we moved to a vLLM-served Llama as a backbone alongside Claude.

**Follow-up: Why is the default block size sixteen tokens?**

It's a tradeoff between internal fragmentation and bookkeeping overhead. Smaller blocks waste less memory inside the last partial block per sequence, but they require more entries in the page table and more lookup overhead per attention step. Sixteen tokens turns out to be the empirical sweet spot — small enough that the last-block waste per sequence is bounded to about sixteen divided by the maximum sequence length, large enough that the page table and the attention kernel stay efficient.

---

### Q2. What is continuous batching and how is it different from static batching?

Continuous batching is iteration-level scheduling, which means after every single forward pass through the model, the scheduler can add new sequences to the running batch and remove ones that just finished. The contrast is static batching, which is what the HuggingFace generate function does by default: you assemble a batch of N sequences, you wait for the slowest sequence in the batch to finish generating, and only then can you start a new batch. The wasted GPU time in that waiting window is enormous, especially for chat workloads where output lengths vary from ten tokens to a thousand tokens within the same batch. With continuous batching, a short request finishes quickly, frees up its slot, and a new request joins the running batch on the very next iteration — the GPU is never idle. vLLM's scheduler maintains three queues internally: running, waiting, and swapped, where swapped means a request had its KV cache evicted to CPU memory because of GPU memory pressure and will resume later. The combination of PagedAttention and continuous batching is why vLLM gets two to four times the throughput of naive serving stacks on the same hardware.

**Follow-up: When does static batching actually win?**

Static batching wins exactly one workload: offline batch inference where all your sequences have nearly identical lengths and you don't care about latency at all. Think nightly jobs that re-embed a million documents. There the fixed-batch overhead is amortized and the simplicity of static batching is fine.

---

### Q3. How would you serve a 70B-parameter model on a node of 8 A100 or H100 GPUs?

I would use tensor parallelism set to eight, so the model is sharded across all eight GPUs within the node, leveraging NVLink for the high-bandwidth all-reduce traffic that tensor parallelism requires. With FP16 a 70B model is 140GB of weights, plus optimizer states for training or KV cache for inference; on 8x80GB A100s that fits with headroom, and on 8x80GB H100s I have generous KV-cache budget for larger batch sizes. If I can sacrifice one or two points of quality for throughput, I'd quantize to INT8 with AWQ or GPTQ, which roughly halves memory and doubles throughput. The vLLM CLI for this would be something like `--tensor-parallel-size 8 --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.9 --enable-chunked-prefill`. Chunked prefill is important because it interleaves prefill and decode tokens, which smooths out latency when a long-context user joins an in-flight chat batch. If TTFT is critical I'd also enable speculative decoding with a 1B draft model, which can drop TTFT by thirty to fifty percent without quality loss.

**Follow-up: What if you had two nodes of eight GPUs each, total sixteen GPUs?**

Then I have a real choice. The first option is tensor parallelism eight within node and pipeline parallelism two across nodes. The second is tensor parallelism eight within node and data parallelism two across nodes. Pipeline parallelism is good when memory is the bottleneck — splitting the model across nodes makes the per-node footprint smaller. Data parallelism is good when throughput is the bottleneck — each node serves the full model and I'm just doubling effective concurrency. For Llama-70B on 16x80GB GPUs I'd default to TP=8 plus DP=2 because we have memory headroom and we want to maximize concurrent users.

---

### Q4. TTFT, ITL, and throughput — which do you optimize for and when?

These three are the latency triangle of LLM serving and they trade off against each other. TTFT is time-to-first-token, the time from the user pressing send to the first token appearing on screen — for a chat product this should be under five hundred milliseconds at p95, otherwise the experience feels broken. ITL is inter-token latency, the time between subsequent tokens — humans read at about two hundred to three hundred words per minute, so anything under fifty milliseconds per token feels real-time. Throughput is total tokens per second across all concurrent users, and it's what drives your dollars-per-million-tokens cost.

For a chat product like a customer support bot or a coding assistant, I optimize for TTFT and ITL — chunked prefill, smaller batch sizes, lower max-model-len, possibly speculative decoding. For a batch summarization product like nightly document processing, I optimize for throughput — large batch sizes, full-context prefill, no streaming, and I'm happy to let TTFT be ten seconds because no one is watching. The conflict is real: bigger batches give better throughput but worse per-request latency, because each user waits for the batch's slowest sequence at every step. The art of LLM serving is choosing the right point on this triangle for each product.

**Follow-up: Why does p99 matter more than p50 for a chat product?**

Because users notice tail latency, not average latency. If your p50 is two hundred milliseconds and your p99 is twelve seconds, two percent of users hate your product, write angry tweets, and never come back. Tail latency in LLM serving comes from a few specific causes: long prefills blocking the GPU, KV-cache pressure forcing eviction, batch coalescence delays. Every one of those needs an explicit mitigation — chunked prefill for the first, swap-to-CPU policy for the second, careful batch-wait-timeout tuning for the third.

---

### Q5. How does vLLM compare to TGI, SGLang, and TensorRT-LLM?

vLLM is the general-purpose default — easy to deploy, OpenAI-compatible API out of the box, PagedAttention plus continuous batching plus a healthy ecosystem of quantization formats. TGI from HuggingFace is similar in spirit, slightly behind vLLM on raw throughput in recent benchmarks, but tightly integrated with the HuggingFace stack which is convenient if your team lives in that ecosystem. SGLang is the new contender — it adds RadixAttention for prefix caching across requests, which is a step beyond vLLM's prefix-share capability and very fast for agent-style workloads where many requests share long system prompts. TensorRT-LLM is NVIDIA's own optimized backend with the best raw H100 throughput numbers, but it has tighter tooling, less ecosystem support, and you need to be willing to recompile engines for different shapes. My default choice is vLLM. I'd reach for SGLang if I had an agent product where the system prompt is identical across all users and I'm losing money on redundant prefill compute. I'd reach for TensorRT-LLM if I had NVIDIA support and I needed every last token per second on H100 hardware.

---

## B. Ray and Ray Serve (high probability — JD names Ray)

### Q6. When would you choose Ray Serve over plain Kubernetes Deployments?

Ray Serve wins when you have multi-stage AI pipelines that need to compose together. A real RAG flow has an embedder, a retriever, a reranker, and a generator, each of which has different resource needs — the embedder might run on a small GPU, the reranker is CPU-or-small-GPU, the generator wants a big GPU with TP. With plain Kubernetes you'd have to hand-build four deployments, four services, and an orchestration layer in front of them, plus your own dynamic batching, plus your own per-stage autoscaling. Ray Serve gives you all of that natively: each stage is a Ray Serve deployment, the deployments compose into a graph, you get dynamic batching for free with the `serve.batch` decorator, you get fractional GPU support so the embedder can take a quarter of a GPU, and the Ray autoscaler handles per-replica scaling. For a single-stage in-out service Ray Serve is overkill — plain Kubernetes plus KEDA is simpler and cheaper. Ray Serve's sweet spot is the DAG.

**Follow-up: How does Ray Serve LLM integrate with vLLM?**

Ray Serve LLM wraps vLLM as a Ray Serve deployment and exposes OpenAI-compatible endpoints, so you get vLLM's PagedAttention and continuous batching plus Ray Serve's composition. The newer feature is prefill-decode disaggregation, where the prefill phase, which is compute-bound, runs on one replica pool and the decode phase, which is memory-bandwidth-bound, runs on another. Each pool autoscales independently, which can shave thirty percent off p99 latency in chat workloads.

---

### Q7. Tasks versus actors in Ray — when does each apply?

Tasks are stateless — you decorate a Python function with `@ray.remote` and call it as `func.remote(args)`, which schedules the function to run anywhere in the cluster. Actors are stateful — you decorate a Python class with `@ray.remote` and instantiate it with `MyActor.remote()`, which spawns a long-lived process that holds the class instance. The mental model: tasks are like AWS Lambda invocations, actors are like long-running EC2 services.

For ML serving, actors win almost every time, because you load the model weights into memory once at actor startup and serve thousands of requests against that loaded state. With tasks, every invocation would re-pickle and re-load the model, which is fatally expensive for any nontrivial model. The tradeoff is fault tolerance: if an actor process dies, its in-memory state is gone, so you have to either checkpoint state explicitly or rely on Ray's actor restart with re-initialization from durable storage. Tasks are restarted automatically by Ray so they're more robust by default.

---

### Q8. Walk me through writing a Ray Serve deployment that batches requests.

The pattern is to decorate a class with `@serve.deployment` to declare it as a Ray Serve deployment, then add `@serve.batch` on the call method to enable dynamic batching. Here's the canonical shape:

```python
from ray import serve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class LLMDeployment:
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.01)
    async def __call__(self, prompts: list[str]) -> list[str]:
        inputs = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = self.model.generate(**inputs, max_new_tokens=128)
        return self.tok.batch_decode(out, skip_special_tokens=True)

deploy = LLMDeployment.bind()
```

The interesting part is the `serve.batch` decorator. Ray takes incoming individual requests and coalesces them into a batch up to `max_batch_size` or until `batch_wait_timeout_s` elapses, whichever comes first. Inside the call method I receive a list of prompts and return a list of responses, and Ray takes care of unzipping the response back to the individual callers.

**Follow-up: Why would you use `serve.batch` instead of vLLM's continuous batching?**

You wouldn't, for an LLM. vLLM's iteration-level continuous batching is strictly better for autoregressive generation because finished sequences free their GPU slots immediately, and new requests join mid-batch. `serve.batch` is request-level batching with a static window, which is great for upstream stages of a Ray Serve graph — a reranker that takes a batch of (query, document) pairs, an embedding model that batches text inputs, a classifier that scores a batch of items. For the LLM itself, hand off to vLLM.

---

## C. Kubernetes (very high probability — JD explicitly names K8s and EKS)

### Q9. Walk me through deploying a vLLM model server on Kubernetes.

The first decision is the container image. I start from `nvidia/cuda` or directly from `vllm/vllm-openai`, install vLLM, and copy in any inference scripts. The pod itself needs `resources.limits` set to `nvidia.com/gpu: 1` for a single-GPU model, or higher for tensor-parallel deployments. The cluster needs the NVIDIA GPU operator installed, which provides the device plugin, the runtime, and node labels for GPU types.

Weight loading is the next decision because a 70B model is 140GB. The simplest pattern is an init container that runs `s5cmd` to pull weights from S3 to a PersistentVolumeClaim mounted at, say, `/models`. The vLLM container reads from that PVC. For deployments with many replicas downloading the same weights, I'd switch to a shared filesystem pattern like Fluid or JuiceFS so we download once and many pods read.

The probes matter for stability. I set a liveness probe on `/health` so Kubernetes can restart a stuck container. I set a readiness probe on `/v1/models` so the load balancer doesn't send traffic until the model is loaded — important because cold-start can take three to five minutes for large models. I add a `terminationGracePeriodSeconds` of sixty seconds plus a `preStop` hook that calls vLLM's drain endpoint, so an in-flight ten-second response can finish before the pod terminates during a rolling update.

For autoscaling I do not use the default HPA on CPU because GPU inference doesn't show up in CPU metrics. I install KEDA with a Prometheus scaler reading `vllm:num_requests_waiting` from vLLM's exported metrics, and scale on queue depth, which is a leading indicator of latency degradation. I add a PodDisruptionBudget so an HPA scale-down doesn't kill all replicas simultaneously, and a NetworkPolicy for security.

**How to say this in an interview:** "Weight loading via S3 init container to a PVC, GPU operator installed cluster-wide, KEDA autoscaling on queue depth from vLLM Prometheus metrics, and a preStop drain hook so we don't drop in-flight requests on rolling updates. The big rookie mistake is autoscaling on CPU — for GPU inference it doesn't move, you have to scale on a request-level metric."

---

### Q10. How do you handle very large model weights on Kubernetes?

For a 140GB Llama-70B I have three patterns and I pick based on traffic. First option: init container with `s5cmd` pulling from S3 to a per-pod ephemeral volume or a shared PVC. This is simple and works for small replica counts, but if I scale to fifty pods all pulling the same weights, my egress bill and my startup latency both blow up. Second option: bake weights into a fat container image with eStargz or SOCI for lazy pull, so the image streams in chunks as the kernel reads them. This pattern is brittle in practice — the image is gigantic, the registry has to support these formats, and you've coupled image versioning to weight versioning. Third option, my preferred for production: a shared parallel filesystem like Fluid, JuiceFS, or AWS EFS where one download serves all pods. Cold start drops to seconds because the OS page cache warms up across pods, and weight versioning is decoupled from container image versioning.

**Follow-up: How do you handle a 70B-to-405B model upgrade without dropping traffic?**

Side-by-side deployments behind a routing layer. I deploy the new 405B as a new Deployment with its own replica set, let it warm up to readiness, then progressively shift traffic from old to new at the service mesh — five percent canary, fifty percent if metrics are green, one hundred percent. Old replicas drain via preStop hook, new replicas scale up under traffic. If the new model regresses on guardrail metrics like refusal rate or LLM-as-judge win rate, automatic rollback shifts traffic back.

---

### Q11. HPA versus KEDA for inference workloads — which and why?

HPA, the default Horizontal Pod Autoscaler, scales on CPU and memory metrics out of the box. For GPU inference this is exactly the wrong signal: when the GPU is fully busy, the CPU is often idle because the model forward pass is happening on the accelerator while the Python orchestrator is waiting. So HPA looks at twenty percent CPU usage, decides nothing is happening, and refuses to scale up while requests are queueing and latency is exploding. KEDA, the Kubernetes Event-Driven Autoscaler, fixes this by letting you scale on any metric available through any of dozens of scalers — Prometheus, RabbitMQ queue length, Kafka lag, AWS SQS depth, custom external APIs. For LLM inference the right metric is `vllm:num_requests_waiting` from vLLM's exported Prometheus metrics. That's a leading indicator: when the queue grows, latency is about to degrade, scale up now. CPU utilization is a lagging indicator that's already broken when it triggers.

**Follow-up: Could you scale on GPU utilization?**

Technically yes, with the NVIDIA DCGM exporter, but it's still a lagging indicator. By the time GPU utilization is at one hundred percent, your queue is already growing and your p99 is already degrading. The leading indicator — queue depth or request rate — gives you the headroom to scale up before users notice. I use GPU util as a *health metric*, not a *scaling metric*.

---

### Q12. How do you do a rolling update of a vLLM deployment without dropping in-flight requests?

The mechanics are: set `maxSurge: 1` and `maxUnavailable: 0` in the deployment strategy so we add one new pod before removing an old one. Set `terminationGracePeriodSeconds` long enough to cover the longest expected response — for chat, two minutes is generous. Wire a `preStop` lifecycle hook that calls vLLM's drain endpoint, which tells vLLM to stop accepting new requests but to keep serving the in-flight ones. The readiness gate flips to false on drain so the service mesh stops routing new traffic to that pod. By the time the grace period expires, the pod's in-flight requests are done and it can terminate cleanly.

The subtle point is the service mesh layer. If you're using plain Kubernetes Services with iptables routing, removing a pod's endpoint can take a few seconds to propagate, and traffic can briefly land on a draining pod. With Envoy or Istio, the service mesh respects the readiness gate immediately, so traffic stops the moment the pod transitions to NotReady.

---

## D. RAG, LangChain, and agents (very high probability — JD names LangChain and LLM-plus-API)

### Q13. Walk me through your RAG architecture at ResMed.

We were building a knowledge-base chatbot over clinical reports. The naive RAG pattern was failing because user queries were heterogeneous — some were factual lookups, some were analytical aggregations across many records, some were just conversational follow-ups. So I designed a query router as the front door. An LLM-based classifier — Claude Haiku for cost — read the user question and routed it into one of three buckets. Factual queries went through standard vector retrieval over our chunked clinical corpus stored in pgVector. Analytical queries triggered a code-generation path: the LLM wrote a SQL query against our Snowflake clinical tables, we executed it, and the answer was the result set. Conversational queries went to the LLM directly with no retrieval. Every answer included citations linking back to the source document.

The hardest part was the chunking strategy. Naive token-based chunking split clinical findings mid-sentence, which destroyed retrieval quality. I built a section-aware chunker that respected the structure of clinical reports — Findings, Impressions, Recommendations as natural boundaries — and that single change moved our context-precision metric from sixty percent to over eighty. Evaluation was RAGAS for offline metrics — faithfulness, answer relevance, context precision, context recall — plus a thumbs-up-thumbs-down logging hook in the UI for online signal.

**How to say this in an interview:** "The signature design was a router-LLM front door that classified queries as factual, analytical, or conversational, and dispatched to vector retrieval, code generation, or direct LLM respectively. The biggest single quality win was domain-aware chunking — respecting section boundaries in clinical reports."

---

### Q14. Naive RAG versus advanced RAG — when do you add complexity?

Naive RAG is the four-step pipeline: chunk the corpus, embed each chunk, on query embed the query, run cosine similarity against the chunks, stuff the top-k into the prompt and ask the LLM. It works surprisingly well on clean, well-curated corpora with uniform queries. Advanced RAG adds stages around that core: pre-retrieval techniques like query rewriting and HyDE — Hypothetical Document Embedding — where the LLM writes a fake document for the query and you retrieve against that; hybrid retrieval combining BM25 keyword search with dense embedding search and fusing them with Reciprocal Rank Fusion; post-retrieval techniques like cross-encoder reranking to push the most relevant documents to the top of the context window.

My rule for adding complexity: do not add a stage until offline evaluation tells you which axis is broken. If RAGAS context precision is low — meaning the retrieved chunks don't actually contain the answer — add a reranker, which is a stronger but slower model that scores query-document pairs jointly. If context recall is low — meaning the right chunks aren't even in the top-k — add hybrid search or query rewriting. Throwing every advanced technique at a problem without measuring just adds latency and cost without principled improvement.

**Follow-up: How does Reciprocal Rank Fusion work?**

RRF combines multiple ranked lists into a single ranking by summing the reciprocal of each item's rank across the lists. The formula is `score(d) = sum over rankers of 1 / (k + rank_i(d))`, where `k` is a damping constant typically around sixty. The intuition is that an item ranked second in two different rankers is probably more relevant than an item ranked first in one ranker and missing from the other, so combining ranks rewards consistency. RRF is parameter-free except for k and works without needing the underlying scores to be on the same scale, which makes it robust for fusing BM25 and cosine-similarity rankings.

---

### Q15. How do you evaluate a RAG system, both offline and online?

Offline evaluation is RAGAS or a similar framework over a curated golden set of question-answer pairs. RAGAS gives me four metrics that decompose RAG failure modes. Faithfulness asks whether the answer is grounded in the retrieved context — a low score means hallucination, the LLM is making things up despite having the context. Answer relevance asks whether the answer actually addresses the question — a low score means the LLM is confidently answering the wrong question. Context precision asks whether the retrieved chunks actually contain the answer — a low score means the reranker or the embedding model is broken. Context recall asks whether all the relevant information was retrieved — a low score means the retriever's top-k is too small or hybrid search is needed.

Online evaluation is harder because users don't tell you when the bot was wrong. I instrument three signals: a thumbs-up-thumbs-down widget in the UI; LLM-as-judge with a stronger model like Claude Opus running on a sample of production traffic, scoring answers against the retrieved context for groundedness; and embedding-drift monitoring over the user query distribution, which tells me when production queries are diverging from my offline test set.

**Follow-up: How do you avoid LLM-as-judge variance making your evals noisy?**

Three practices: sample more, use a stronger judge, and prefer pairwise comparisons over absolute scoring. Anchor each judgment with a reference answer where possible. Run the judge multiple times per item and look at agreement; if the judge disagrees with itself thirty percent of the time, your prompt to the judge needs more structure. Pairwise — "which of these two answers is better" — is much lower variance than absolute — "rate this answer one to five" — because the judge anchors against the comparison rather than an internal scale.

---

### Q16. What is agentic RAG and when do you not use it?

Agentic RAG is a pattern where the LLM itself decides whether and what to retrieve, often making multiple retrieval calls in a loop. A standard variant is a self-RAG or corrective-RAG flow: the LLM gets the user query, asks itself "do I have enough information to answer this," and if not, formulates a retrieval query, scores the retrieved documents, and either answers or re-retrieves with a different query. This is powerful for multi-hop questions where one retrieval round isn't enough — questions like "compare the side effects of medication A and medication B" require two retrievals.

The cost is latency and complexity. Each agentic step adds an LLM call, and the loop can take five to ten seconds end-to-end versus one second for naive RAG. So I do not use agentic RAG when the latency budget is tight, when queries are uniform and don't actually need multi-hop, or when retrieval quality is high enough that a single round suffices. My default is a single retrieval round with a self-grading step — the LLM scores its own answer for groundedness and triggers a re-retrieval only if the score is below a threshold. That captures most of the upside with much less latency.

---

### Q17. Compare LangChain, LangGraph, and CrewAI.

LangChain is the original modular SDK — chains, agents, memory, tools, callbacks. It shines for prototypes because you can stand up a basic RAG chain in twenty lines, but it has a reputation for verbose abstractions and frequent breaking changes between versions. For production I find myself fighting LangChain's abstractions as often as I'm using them. LangGraph is the same team's answer to that critique: a stateful graph framework with explicit nodes and edges, conditional routing, persistence, and human-in-the-loop. The graph compiles to a checkpointable executor, which means you can pause an agent, save its state to a database, and resume later — that's what I leverage at TrueBalance for the Claude-powered ML workspace, where a Jira-ticket-resolution agent can take hours and survive process restarts.

CrewAI is a higher-level multi-agent orchestration framework where you define agents by role, goal, and backstory, and they collaborate through structured task passing. The mental model is closer to "team of LLM personas" than "graph of LLM steps." I'd choose CrewAI when the problem decomposes naturally into specialized agents — a researcher, a writer, an editor — and the orchestration is conversational rather than algorithmic. For anything where I need precise control over the state machine, LangGraph wins.

**How to say this in an interview:** "I default to LangGraph for production agents because the graph is stateful and checkpointable — I can pause, resume, and inspect agent state, which matters when an agent is doing real work over hours. I keep LangChain in the toolbox for quick prototypes and wrappers around vector stores and chat models."

---

## E. FastAPI and Python production (high probability — JD explicitly names FastAPI)

### Q18. Sync versus async endpoints in FastAPI — when does each win?

The mental model is that async helps when your endpoint spends most of its time waiting on I/O — a database query, an HTTP call to another service, a cache lookup — and the event loop can context-switch to handling another request during that wait. Async only delivers if your downstream is async too: `httpx.AsyncClient` instead of `requests`, `asyncpg` instead of `psycopg2`, `motor` instead of `pymongo`. If you put a synchronous library inside an `async def` handler, you block the event loop and serialize every request that's hitting that worker, which is dramatically worse than just using sync handlers.

The flip side is that CPU-bound work — model inference, image processing, big numpy operations — also blocks the event loop in async handlers. For an ML endpoint where the inference is happening in-process, the right pattern is either a sync handler, which FastAPI runs in a threadpool by design and does get concurrency, or an async handler that offloads the inference call via `run_in_threadpool` or by hitting a separate inference server like Ray Serve or Triton.

**How to say this in an interview:** "Async wins for I/O-heavy endpoints when the whole stack is async. For CPU-bound ML inference inside the same process I'd use a sync handler, which FastAPI runs in a threadpool, or I'd offload the inference to a dedicated server and keep the FastAPI layer thin and async."

---

### Q19. Walk me through a production FastAPI service checklist.

Pydantic v2 models on every endpoint for input validation and structured documentation. Structured JSON logs with a `request_id` correlation field that propagates through every downstream call — I use `structlog` and a middleware that injects the ID. OpenTelemetry tracing with the FastAPI auto-instrumentation, sending spans to a backend like Jaeger or Tempo. A Prometheus metrics middleware exposing latency histograms, request counts, and error rates per endpoint. Rate limiting via slowapi or an upstream gateway because untrusted callers will hammer your endpoints. Process management with Gunicorn as the master and Uvicorn workers as the children, configured to roughly two times CPU plus one workers, with `--lifespan on` so startup and shutdown events run correctly. Distinct `/health` and `/ready` endpoints — `/health` returns 200 if the process is alive, `/ready` returns 200 only if the model is loaded and downstream dependencies are reachable. Kubernetes uses the readiness probe to gate traffic and the liveness probe to restart unhealthy containers; conflating them is a common mistake. Request body size limits to prevent memory exhaustion attacks. Graceful shutdown via the lifespan context manager so in-flight requests get a chance to finish before the process exits.

---

### Q20. How do you stream LLM responses from FastAPI?

The pattern is `StreamingResponse` with an async generator that yields server-sent-events-formatted chunks. The trap that catches everyone the first time is that returning a generator from a regular `JSONResponse` does not stream — the response is buffered and sent at once. You have to use `StreamingResponse` and set `media_type="text/event-stream"` for proper SSE framing. Inside the generator, you check `request.is_disconnected()` on every yield so you can free the upstream LLM call when the client disconnects mid-stream. Without that check, your vLLM call keeps generating tokens that no one is reading, wasting GPU time. The upstream client should be `httpx.AsyncClient` with `client.stream()`, which streams tokens as they arrive without buffering the whole response. The combination — async upstream, async generator, disconnect check — gives you a streaming endpoint that scales to thousands of concurrent users on a single FastAPI worker.

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

@app.post("/chat")
async def chat(req: dict, request: Request):
    async def stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", "http://vllm:8000/v1/completions",
                                     json={**req, "stream": True}) as r:
                async for chunk in r.aiter_lines():
                    if await request.is_disconnected():
                        break
                    if chunk.startswith("data: "):
                        yield chunk + "\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream")
```

---

## F. PyTorch and ML fundamentals (mid probability — JD names PyTorch)

### Q21. Why do we scale by the square root of d_k in self-attention?

The math reason is that without scaling, dot products of query and key vectors grow in magnitude proportionally to the dimension d_k. If Q and K have unit-variance components, the dot product is a sum of d_k products, so its variance scales as d_k and its standard deviation as the square root of d_k. Large dot products push the softmax into saturation — one element approaches one and all the others approach zero — and the gradient through softmax becomes nearly zero. That kills learning. Dividing by the square root of d_k normalizes the variance back to one, keeping the softmax in a regime where gradients flow.

The intuition I'd give an interviewer: imagine you're computing softmax over scores like ten, twenty, thirty versus scores like one, two, three. The first set is so peaked that the gradient is essentially flat — softmax is "decided" — while the second set is gentle and gradients flow. Scaling by the square root of d_k keeps us in the second regime regardless of dimension.

---

### Q22. BF16 versus FP16 — when does each win for training?

Both are 16-bit floats, but they distribute the bits differently. FP16 has a five-bit exponent and a ten-bit mantissa, so it has good precision but a narrow dynamic range — it overflows around sixty thousand and underflows around six times ten to the minus eight. BF16 has an eight-bit exponent, the same as FP32, and a seven-bit mantissa, so it has the same dynamic range as FP32 but worse precision per representable number. For training, the dynamic-range issue dominates: gradients can be tiny, and FP16 underflows them to zero, which kills learning. The classical fix is gradient scaling, where you multiply the loss by a large constant before the backward pass and unscale gradients before the optimizer step — this is what PyTorch's `GradScaler` does. BF16 doesn't need that fix because its range matches FP32's, so most modern training pipelines on A100 and H100 default to BF16.

For inference, FP16 has slightly better precision when the values are in range, which is usually true post-training, so FP16 is often preferred for serving while BF16 wins for training. FP8 is the new frontier on H100 and B200 — you typically use FP8 for the linear layers and FP16 or BF16 for accumulation, with both E4M3 and E5M2 formats available depending on whether you need more precision or more range.

---

### Q23. What is FlashAttention?

FlashAttention computes the same self-attention output as the textbook formula, but with a different memory access pattern that makes it dramatically faster on modern GPUs. The textbook implementation materializes the full attention matrix of shape (sequence_length, sequence_length), which for a sequence of eight thousand tokens is sixty-four million entries — gigabytes per layer in FP16. That matrix has to be written to HBM, the GPU's main memory, then read back to compute the softmax, then read again to multiply by V. HBM bandwidth is the bottleneck.

FlashAttention's trick is to tile the computation: it processes Q, K, and V in blocks that fit in SRAM, the much faster on-chip memory, and computes the softmax incrementally using the online-softmax algorithm so it never needs the full attention matrix in memory. It also recomputes attention during the backward pass instead of storing it from the forward pass, trading a bit of compute for a lot of memory savings. The net effect is two to four times faster attention with O(N) memory instead of O(N squared). FlashAttention-2 improved work partitioning across warps; FlashAttention-3 uses the H100's TMA — Tensor Memory Accelerator — and warp specialization for FP8.

---

### Q24. When does `torch.compile` help and when does it break?

`torch.compile` traces your PyTorch code through TorchDynamo, captures it as an FX graph, and lowers that graph through Inductor into fused, optimized kernels. The wins are real — thirty to fifty percent training throughput improvement is common on transformer models, more on smaller models where Python overhead dominates. It works well on static-shape, control-flow-light forward passes, which most transformers are.

It breaks on dynamic shapes unless you mark the dynamic dimension with `mark_dynamic`. It breaks on data-dependent control flow like `if x.sum() > 0`, because the compiler can't know which branch to compile. It breaks on custom CUDA kernels that aren't registered as FX operators. For LLM serving I rarely use `torch.compile` directly because vLLM has its own kernel layer with PagedAttention and continuous batching that's more important than graph-level fusion.

---

## G. MLOps and observability (very high probability — Sachin's Datadog/drift signature)

### Q25. How do you detect data drift in production?

The standard tool is the Population Stability Index, PSI, which compares a baseline distribution against a production distribution by binning the values and computing a weighted log-ratio: PSI equals the sum over bins of `(p_i minus q_i) times the log of p_i over q_i`, where p is the production proportion in bin i and q is the baseline proportion. The conventional thresholds are below zero point one is stable, between zero point one and zero point two five is minor drift worth investigating, and above zero point two five is significant drift that probably warrants retraining. PSI is robust because it's symmetric and bounded, but it requires you to choose a bin count — typically ten — and to handle empty bins with a small epsilon.

For categorical features I use the Kolmogorov-Smirnov test or chi-squared, depending on whether the feature is ordered. For high-dimensional embeddings I use cosine distance to a training-data centroid or maximum mean discrepancy. The tooling spectrum runs from open-source like Evidently AI, which is what I'd reach for first, through commercial products like Arize, WhyLabs, and Fiddler that offer richer dashboards and alerting.

**Resume tie-in:** At ResMed I built a custom utility integrating Datadog and Snowflake for this — the data science team computed drift metrics in Python, my utility wrapped their logic, ran it on Snowflake-stored prediction logs, pushed metrics to Datadog as custom metrics, and auto-generated dashboards per model. It became the team standard.

---

### Q26. Concept drift versus data drift — what's different and how do you respond differently?

Data drift, sometimes called covariate shift, is when the input distribution P(X) changes — your users start sending different prompts, your features have a new value distribution. The model itself may still be correct on the inputs it sees; it's just that the inputs are shifted. The response is sometimes feature reweighting or retraining on a more recent window of data.

Concept drift is when the relationship P(y | X) changes — given the same inputs, the right output is now different. COVID was a textbook example: lab values that previously meant "healthy" suddenly meant something different because the population's baseline shifted. Concept drift is sneakier because input distributions can look stable while the labels' meaning shifts underneath. The only response is retraining with fresh labels, and if labels are expensive to collect, you have to invest in active learning or human-in-the-loop relabeling.

The diagnostic that distinguishes them is monitoring prediction-versus-ground-truth deltas, not just input distributions. If inputs look stable but accuracy is dropping, that's concept drift.

---

### Q27. How do you A/B test two LLM versions safely?

The progressive rollout pattern. First, shadow mode: route copies of production traffic to the new model and compare offline, no impact on users. Then a one-percent canary, watching latency, refusal rate, output length, and user thumbs-up-thumbs-down. Then five percent, fifty percent, one hundred percent, with automatic rollback triggered by any guardrail breach — a refusal-rate spike, a toxicity-classifier spike, a latency p99 regression. The metrics that matter for LLM A/B are different from classifier A/B: TTFT and ITL for latency, win-rate via LLM-as-judge for quality, refusal rate as a safety canary, output length distribution as a sanity check that the model isn't suddenly verbose or terse, and user-side thumbs as the ultimate ground truth.

The hardest part of LLM A/B testing is sample size: LLM-as-judge is noisy, so you need hundreds of judgments to get a reliable signal, which means even at one percent of traffic, you might need a multi-day window to get statistical significance. Don't move to fifty percent until your canary has produced at least a few hundred high-confidence judgments.

---

### Q28. Walk me through the Datadog drift dashboard you built at ResMed.

The data science team had drift metrics they computed in Python — PSI, KS, embedding distance — but they were running these in notebooks that no one outside the team saw. I built a utility that took their Python logic, wrapped it in a scheduled job that read prediction logs from Snowflake on a daily schedule, computed drift metrics per feature per model, pushed those metrics to Datadog as custom metrics with feature and model tags, and generated a Datadog dashboard from a YAML config file that mapped each model to its features and thresholds.

The hardest engineering problem was schema heterogeneity. Different models had different feature sets, different drift thresholds, different baselines. I solved it with a metadata-driven generator: each model had a YAML manifest declaring its features, baselines, and alerts, and a single Python script consumed the manifest to produce a Datadog dashboard JSON via the Datadog API. Adding a new model became a five-minute YAML PR rather than a half-day of dashboard hand-building. The utility became the team standard, and adoption went from "data science team only" to "every model owner uses it" within two months.

---

## H. AWS specifically (your strongest cloud — they will lean here)

### Q29. SageMaker single-container, multi-container, and multi-model endpoints — when do you use which?

A single-container SageMaker endpoint runs one container with one model and is the right default when you have meaningful traffic and want simplicity. A multi-container endpoint runs up to fifteen containers behind one endpoint, sharing the underlying instance — this is what I used at ResMed's IHS platform to consolidate eight lower-traffic models on one endpoint, dramatically reducing cost because the GPU was being underutilized per-model. The catch is that all containers on a multi-container endpoint share the same compute, so noisy-neighbor effects are real if traffic is uneven.

A multi-model endpoint, which is different from multi-container, loads model artifacts on demand into one container that's prepared to serve any of them. The first request to a cold model takes a hit while the artifact loads, but afterward it stays in memory until evicted. This pattern is great for long-tail traffic where you have hundreds of models, each with low individual traffic. The cold-start latency is the tradeoff — if your SLA can absorb a one-second first-call penalty, multi-model endpoints are much cheaper than running hundreds of containers.

---

### Q30. Walk me through your TrueBalance real-time XGBoost Lambda architecture.

The user-facing path was API Gateway, then a Lambda function packaged as a container image because the XGBoost dependencies plus Pydantic plus our internal libraries blew through the 250MB layer limit. Inside Lambda I had three steps: feature fetch from a Redis cache that mirrored a Snowflake feature store, model inference, response build. Redis was warmed by a refresher Lambda triggered by EventBridge every five minutes that pulled the latest feature snapshot from Snowflake. On Redis cache miss I had a circuit-breaker fallback to last-known features rather than a synchronous Snowflake fetch, because Snowflake p99 was over two hundred milliseconds and would have blown my five-hundred-millisecond budget.

The infrastructure was Terraform-managed across three environments — dev, staging, prod — each in its own VPC, each with its own subnets, security groups, NAT gateway, and Redis cluster. I instrumented p99 latency as a CloudWatch custom metric with a threshold alarm, plus a separate alarm on the cache-miss-fallback rate as a leading indicator that something upstream had broken. The model itself was loaded into Lambda at cold start from S3 — about thirty megabytes of XGBoost — and provisioned concurrency kept five Lambda instances warm at all times to avoid cold-start spikes.

The result: p99 stayed under five hundred milliseconds across all three environments, and the projected portfolio profit lift from cutting funding to high-withdrawal-risk borrowers was significant.

**How to say this in an interview:** "API Gateway, Lambda container image, Redis-fronted Snowflake feature store with a circuit breaker on misses, Terraform-managed VPC isolation across three environments. The non-obvious decisions were the cache-not-fetch pattern for sub-five-hundred-millisecond p99 and provisioned concurrency to absorb cold starts."

---

### Q31. Lambda for ML — when does it work and when does it fail?

It works for CPU-bound, low-memory, low-latency models — XGBoost, scikit-learn, small Hugging Face encoder models like DistilBERT for classification or NER. The container-image format gives you up to ten gigabytes of image, which fits most CPU-side ML, and provisioned concurrency keeps cold starts manageable.

It fails on three workloads. First, GPU inference: Lambda has no GPU, so anything LLM-shaped goes to ECS, EKS, or SageMaker. Second, long-running inference: Lambda has a fifteen-minute hard limit, so anything beyond that needs Step Functions or batch processing. Third, very large models: even with the ten-gigabyte image limit, loading a multi-gigabyte model on every cold start kills your latency budget, so for anything over about a gigabyte of model weights I'd run on ECS Fargate or EKS where the model loads once per container lifetime.

The other underestimated failure mode is VPC cold starts: Lambdas in VPC mode used to take five-plus seconds to cold-start while the ENI was attached. AWS has improved this with Hyperplane, but the gotcha still bites. If your Lambda needs VPC access — to reach a private RDS or a private VPC endpoint — keep provisioned concurrency on for predictable performance.

---

## I. System design (one whiteboard question, 45 min — almost certain at the senior round)

### Q32. Design an LLM serving platform handling 10,000 requests per second across five models with mixed context windows.

I would draw the architecture in tiers from left to right. At the front, an API gateway — Envoy or AWS API Gateway — handles authentication, rate limiting, and routing. Behind it, a model router service that inspects the request and dispatches to the right model pool — this matters because different models have different context windows, different costs, and different latencies, and I don't want a one-thousand-token chat going to the same pool as a forty-thousand-token RAG analytical request. The router also handles per-tenant quotas and circuit-breaking.

Each model has its own pool of vLLM replicas managed by Ray Serve LLM, deployed on Kubernetes with the GPU operator. Tensor parallelism is set within node — usually TP equals the number of GPUs per node — and data parallelism via replica count. KEDA autoscales each pool independently based on `vllm:num_requests_waiting`. A KV-cache-aware load balancer routes requests to the replica with the warmest prefix cache for that user — for chat workloads this is a huge win because the system prompt is identical across all of one user's turns.

For caching, I add a Redis layer for prompt-level response caching — exact-match semantic cache, with optional fuzzy matching via embedding similarity. For RAG-shaped workloads, I cache embeddings and retrieved chunks with a longer TTL.

For observability, I have three lanes: structured JSON logs with a request ID propagating through every hop; Prometheus metrics with histograms for TTFT, ITL, and full-response latency; Langfuse or Helicone for LLM-specific tracing — token counts, prompt versions, eval scores. I add an offline eval lane that samples production traffic to a golden eval set and runs LLM-as-judge daily.

The capacity math: ten thousand requests per second times an average two thousand tokens per response is twenty million tokens per second of generation throughput. A single H100 with a 70B model in INT8 does roughly two thousand tokens per second of decode. So I need ten thousand H100s of decode capacity, distributed across pools by traffic mix. That's an extreme number, so in practice I'd push hard on quantization to INT4 — roughly doubling throughput — and on speculative decoding, also another thirty to fifty percent on hot paths.

---

### Q33. Design an AI coaching feature for MyWhoosh, Avrioc's flagship cycling product.

This is the answer where I show I researched their product. The setup is that MyWhoosh produces continuous high-frequency telemetry from every rider — power output in watts, cadence in RPM, heart rate, gradient response, even pose if they have a camera. That stream is the gold mine.

Architecture, left to right. Riders send telemetry over a low-latency channel — WebSocket or MQTT — into an ingestion gateway. The gateway writes to a streaming buffer, Kafka or AWS Kinesis, with a topic per metric type. From the buffer, two parallel consumers: a stream processor for real-time features like rolling power averages and personal-zone classifiers, writing into a Redis-backed online feature store; and a batch sink to a time-series database — Timestream or InfluxDB — for offline analytics and model training.

Three model surfaces. First, a cheat-detection model — a CV-style anomaly detector on power curves, looking for impossible accelerations or trainer manipulation. This runs in real time during sanctioned events, with a human-review queue for flagged riders. Second, a workout-recommendation model — given a rider's last four weeks of training, what's the next workout? This is per-rider personalization, trained nightly on the offline feature store. Third, an LLM-personalization layer — the workout recommendation generates structured zones and durations, and the LLM converts that into natural-language coaching text, possibly with voice synthesis for in-ride audio.

The serving stack is the standard one: vLLM on EKS for the LLM, KServe for the workout-recommendation model, the feature store splits between Redis online and Snowflake offline. The interesting part is the data pipeline: most of the value is in the telemetry features, not in the LLM. So the first ninety days of work would be the feature store, the cheat-detection model, and the recommendation model. The LLM personalization is the last twenty percent of effort that delivers the user-facing magic.

**How to say this in an interview:** "For MyWhoosh, the value lives in the telemetry stream — power, cadence, heart rate, gradient. I'd build the feature store first, the cheat-detection and workout-recommendation models second, and only then layer an LLM for personalized coaching text. Most of the engineering is the data pipeline; the LLM is the user-facing icing."

---

## J. Behavioral and project deep-dive (Avrioc-confirmed reverse-chronology pattern)

### Q34. Tell me about yourself.

I'm Sachin, a senior ML engineer with eight years of experience productionizing ML and LLM systems across fintech, healthcare, and enterprise. At TrueBalance today I own a real-time XGBoost Lambda for loan-withdrawal prediction with p99 under five hundred milliseconds and three-environment VPC isolation, plus a Claude-powered developer platform that integrates Jira, GitHub, AWS Athena, and Jenkins for our ML team. Before that I spent two and a half years at ResMed building their IHS MLOps platform — we shipped eight models to production in six months, a RAG-based clinical chatbot, and the Datadog drift dashboard utility that became the team standard. Before that, time at Tiger Analytics and Sopra Steria building SageMaker pipelines and computer-vision systems. My sweet spot is the handoff from research to production — LLMOps with vLLM and Kubernetes, real-time inference, quantization and distillation for cost, and the observability that keeps models healthy at scale. That bridge from research to production is exactly what Avrioc's JD describes, which is why this onsite Abu Dhabi role excites me.

(Time this at home. Should land at fifty-five to sixty-five seconds. Memorize the first ten and last ten words; improvise the middle.)

---

### Q35. Walk me through your projects in reverse chronological order.

Use the timing structure from Chapter 19, section 4: four to five minutes on TrueBalance, three to four minutes on ResMed, one and a half to two minutes on Tiger Analytics, one minute on Sopra Steria. For each, the shape is context, my role, the hard part, the implementation, the numbers, and a single retrospective sentence on what you'd do differently today. Do not narrate every detail; lead with the most interesting technical decision and let the interviewer pull on threads.

---

### Q36. Tell me about a production incident you owned end-to-end.

At ResMed, a SageMaker endpoint started returning stale predictions about three weeks after a quiet region migration. The Datadog drift dashboard I'd built fired an alert because the input distribution had shifted — but only on one specific feature, which was suspicious. I traced the feature back through the pipeline to a cron job that backfilled feature values nightly, and found the cron had silently failed for two weeks because its IAM role had been scoped down during the region migration and lost a permission. The endpoint was using the last-good feature snapshot, which is why predictions degraded gradually rather than catastrophically — exactly the failure mode that's hardest to catch.

The immediate fix was to restore the IAM permission and run a backfill job. The longer-term fix was to add a separate freshness SLO and alert on cron success directly, rather than relying on data drift as the canary. Five-whys root cause: cron failed, no alarm on cron, only data drift caught it, only because we happened to have a drift dashboard in place. Today I'd push freshness SLOs as a first-class feature of any feature store I build.

---

### Q37. Tell me about a technical decision you regret.

At TrueBalance early on, I deployed our XGBoost model using Lambda layers for the dependencies. Layers have a 250MB limit and at the time we were at about 200MB, with comfortable headroom. Three months later one of the data scientists added a new library and we hit the layer limit overnight. The migration to container-image format took a day of work and an unplanned deployment window, which annoyed everyone.

The lesson was that I picked the artifact format based on the current size of dependencies rather than the upper bound. Container images would have given us up to ten gigabytes from day one, with no real cost overhead. Today I default to container images for any Lambda with non-trivial ML dependencies, even when the current footprint is small. The cost of the migration later is much higher than the cost of starting with the right format.

---

### Q38. How do you handle disagreement with a tech lead?

I restate their position back to them in my own words to confirm I actually understand what they're proposing — at least half the time disagreements turn out to be misunderstandings, and the restate-and-confirm step ends them quickly. Then I share my reasoning with concrete trade-offs, ideally with numbers. If we still disagree and they're the decision-maker, I commit and move forward — disagree-and-commit. The exception is if the decision has serious safety, security, or compliance implications, in which case I'd escalate, calmly. I try not to make any disagreement personal; technical decisions are about the work, not the people.

---

### Q39. Why are you leaving TrueBalance after only three months?

Honestly: Avrioc's onsite Abu Dhabi role with visa sponsorship is a major life and career change my family has been planning toward for some time. The role specifically matches my LLMOps strength on Kubernetes plus vLLM plus Ray, which is the next step I want my career to take. TrueBalance has been a great chapter — I've learned a lot about real-time inference at fintech scale — but the role wasn't designed around a relocation that we've been preparing for. Timing and opportunity aligned. I'll continue to deliver at TrueBalance through my notice period and leave the systems I've built well-documented for whoever picks them up.

---

### Q40. Why Abu Dhabi specifically?

Three reasons, in the order they matter to me. First, tax-free compounding. UAE has no personal income tax, which over a decade is a meaningful difference in long-term family financial security versus India's tax structure. Second, the UAE's national AI strategy is creating real demand for production ML talent — I want to do the kind of work where ML directly drives consumer products, and Abu Dhabi is one of the few cities outside the usual SF-NYC-London axis where that work is concentrated. Third, onsite cross-functional teams produce faster learning velocity than remote does, and at this stage of my career — eight years in — learning velocity matters more than convenience. Avrioc is specifically a shared-engineering shop across multiple consumer products, which means my work would touch a wide surface area, not just one team.

---

### Q41. What would you build first in your first ninety days?

The first thirty days I'd listen — sit with each product's data scientists and product owners, understand the existing pipelines, the pain points, the performance gaps. I would not propose anything in the first thirty days. Days thirty to sixty I'd pick one product — likely MyWhoosh given the telemetry richness — and ship one concrete LLMOps win. Could be a vLLM-based serving migration for whatever model is currently underperforming on cost, could be a drift dashboard for an existing model, could be a RAG-powered internal tool. Days sixty to ninety I'd turn that one win into a reusable platform pattern others can adopt — the same shape I followed at ResMed where one model's pipeline became the template for seven more.

The mindset I'd bring: avoid "I'd rebuild everything" energy. The first quarter is about earning trust by shipping one meaningful thing, not re-architecting the world.

---

### Q42. Do you have any questions for us?

Always yes — three of them.

First: "What does the AI infrastructure look like today — is it primarily Kubernetes plus Ray, or does Slurm own training while Kubernetes handles inference?" This signals you understand the split and that you've thought about how you'd operate.

Second: "Of MyWhoosh, Comera, Labaiik, and Hyre, which one's AI roadmap is the team's biggest current focus?" This signals you actually researched the company. Most candidates don't.

Third: "What does success look like for this role in the first six months?" This signals you're outcome-oriented and willing to commit to concrete deliverables.

Save salary, visa, and relocation logistics for the HR round. The technical interviewer doesn't decide those.

---

## How to use this chapter, redux

These fifty answers are not a script — they are a rhythm. Read the question, hear yourself say the answer, and trust that on Thursday your mouth will know the rhythm even if the words come out slightly differently. The interview doesn't grade word-for-word recall. It grades whether you sound like someone who's actually done the work. The work shows up in your voice when you've internalized the answers.

Walk in calm. You're already qualified. The interview just needs to confirm it.

---

End of pack. Continue back to **[Chapter 00 — Master Index](00_index.md)** to navigate other chapters.
