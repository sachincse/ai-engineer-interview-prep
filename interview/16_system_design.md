# Chapter 16 — System Design Interviews
## Four full walk-throughs for the questions you'll hear at Avrioc

> System design rounds for Senior AI Engineer roles are usually "design an LLM-powered X" or "design a real-time ML feature." This chapter gives you four polished answers; mix-and-match elements for any variation.

---

## 16.1 The universal system-design framework

Every design answer follows this skeleton. Use it every time.

```
1. Clarify requirements (5 min)
   - Functional: what must it do?
   - Non-functional: latency, scale, availability, data residency
   - Constraints: budget, team size, existing infra

2. Define APIs / scope (3 min)
   - Endpoints
   - Request/response schemas

3. High-level architecture (10 min)
   - Components + data flow (DRAW A DIAGRAM)
   - Storage choices
   - Compute choices

4. Deep-dive key components (15 min)
   - Pick 2-3 critical components; go deep
   - Show trade-offs

5. Address scale, reliability, cost (10 min)
   - Bottlenecks
   - Scaling strategy
   - Monitoring / drift / on-call

6. Future / extensions (3 min)
   - What next?
   - Known limitations
```

**Golden rules:**
- **Always draw a diagram.** Unsolicited. It separates you.
- **Always ask about scale first.** 1 QPS vs 10K QPS = different systems.
- **Always think about data residency** for Avrioc (UAE).
- **State trade-offs explicitly.** "I'd pick X because Y, even though Z is tempting."

---

## 16.2 Design #1 — Production RAG System for Internal Knowledge

### The ask
"Design a RAG system for Avrioc's internal compliance + product docs (say, 50K documents, English + Arabic). Must stay in UAE."

### Step 1 — Clarify
- **Volume:** 50K docs → ~5M chunks (100 chunks/doc avg)
- **Query QPS:** 100 QPS peak (internal tool)
- **Latency:** TTFT <1s, E2E <3s for chat
- **Languages:** English + Arabic
- **Data residency:** All data and compute in UAE (me-central-1 Bahrain OR UAE North)
- **Users:** Internal staff (~500 active)
- **Update frequency:** Docs update weekly; reindex cadence daily

### Step 2 — API
```
POST /chat
{
  "query": "What does the compliance doc say about KYC for corporate accounts?",
  "thread_id": "abc-123",      // for multi-turn
  "lang_hint": "en|ar|auto"     // optional
}

Response (SSE stream):
  data: {"token": "..."}
  data: {"citations": [{"doc_id": "...", "page": 4}]}
  data: [DONE]
```

### Step 3 — High-level architecture

```
┌──────────────┐
│   Client UI  │ (Chainlit or React)
└──────┬───────┘
       │ SSE
       ▼
┌──────────────┐
│ API Gateway  │ (AWS API Gateway or Kong)
│ + Auth       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│ FastAPI Gateway (Python, async)          │
│   - Auth, rate-limit, log                │
│   - LangGraph orchestration              │
└──┬───────┬───────┬───────┬──────────┬────┘
   │       │       │       │          │
   ▼       ▼       ▼       ▼          ▼
┌──────┐┌──────┐┌──────┐┌──────┐ ┌──────────┐
│Query ││Hybrid││Re-   ││LLM   │ │Langfuse  │
│rewrit││search││rank  ││     │ │tracing   │
│LLM   ││(BM25+││cross-││vLLM  │ │          │
│      ││dense)││enc.  ││Llama-│ │          │
│      ││      ││      ││3 70B │ │          │
└──────┘└──┬───┘└──┬───┘└──────┘ └──────────┘
           │      ▲
           ▼      │
      ┌──────────────────┐
      │ Qdrant cluster   │
      │  - dense + sparse│
      │  - 5M vectors    │
      │  - 3 replicas    │
      └──────────────────┘
           ▲
           │ daily reindex
      ┌──────────────────┐
      │ Embedding job    │
      │ (Ray Data +      │
      │  BGE-M3)         │
      └──────────────────┘
           ▲
      ┌──────────────────┐
      │ Doc storage      │
      │ S3 (UAE region)  │
      │ + metadata DB    │
      │ (Postgres)       │
      └──────────────────┘
```

### Step 4 — Deep dive on 3 components

#### 4a. Chunking + indexing

- **Chunking:** contextual retrieval (Anthropic 2024) — LLM-generated per-chunk context prefix; 512-token chunks with 50 overlap; respect structural headers.
- **Multilingual:** don't language-segment; embed all languages in one index — BGE-M3 handles cross-lingual retrieval.
- **Reindex pipeline:** Ray Data job runs daily. Incremental — hash chunks, only re-embed changed ones. Write to Qdrant via batch upserts.
- **Metadata:** doc_id, section, last_updated, lang, access_level for RBAC.

#### 4b. Query → answer flow

```
User query
   │
   ▼
1. Query rewriting (LLM, small model like Haiku)
   - Resolve pronouns, inject chat history
2. Hybrid retrieval (Qdrant)
   - Dense via BGE-M3 (top-50)
   - Sparse via BGE-M3 sparse head (top-50)
   - RRF fusion → top-100
3. Reranking (bge-reranker-v2-m3 on a g5.xlarge)
   - Rerank top-100 → top-5
4. Context building
   - Top-5 chunks with source metadata
   - System prompt: "answer ONLY from context"
5. Generation (vLLM serving Llama-3.3-70B AWQ)
   - Stream tokens to client
6. Post-processing
   - Parse citations
   - Verify citations map to retrieved chunks
   - Return SSE stream
```

Total latency budget:
- Query rewriting: ~300ms (Haiku)
- Retrieval: ~50ms (Qdrant)
- Reranking: ~200ms (g5.xlarge)
- TTFT for LLM: ~500ms (70B model, 2KB context)
- E2E TTFT target: <1.2s

#### 4c. Observability + evaluation

- **Langfuse self-hosted** for all traces (query, retrieval, rerank, generation, feedback)
- **RAGAS eval** — nightly on 300-pair golden set
- **Online feedback:** thumbs in UI → feedback score per trace
- **Drift monitoring:** query distribution via embedding clustering + UMAP
- **Alerts:** p99 latency > 3s, thumb-down rate > 15%, retrieval recall@5 < 0.8 on canary queries

### Step 5 — Scale, reliability, cost

- **Scale:** 100 QPS peak → Qdrant 3 replicas ample. vLLM 2 replicas on 4xA100-80GB TP=4 → ~50 QPS each. Headroom 3x.
- **Reliability:** KServe on EKS, HPA scaling on queue depth. Cross-AZ replication for Qdrant.
- **Data residency:** All AWS me-central-1 (Bahrain) or on-prem UAE. No egress.
- **Cost estimate (monthly):**
  - vLLM: 2× p5.48xlarge = ~$60K (or swap to 4× A100 = ~$25K)
  - Qdrant: 3× r6i.4xlarge = ~$2.5K
  - Reranker + embedder: 2× g5.xlarge = ~$1.5K
  - Storage + ops: ~$2K
  - Total: ~$30-65K/month depending on LLM size

### Step 6 — Future

- Multi-region for HA
- Fine-tune embeddings on domain data
- Add GraphRAG for "corpus-wide" queries
- Agentic RAG (multi-hop via LangGraph)

---

## 16.3 Design #2 — Real-Time ML Inference Service (like your XGBoost Lambda)

### The ask
"Design a real-time credit-risk prediction service. 10K predictions per second peak, p99 <200ms, heavily regulated data."

### Clarify
- 10K QPS peak
- p99 <200ms end-to-end
- Features: ~200 numerical + categorical
- Data residency: strict

### API
```
POST /predict
{
  "applicant_id": "...",
  "context_features": {"device_type": "...", "app_version": "..."}
}

Response:
{
  "risk_score": 0.23,
  "decision": "approve|reject|review",
  "features_used": {...},    // for XAI / audit
  "model_version": "v12.3.1"
}
```

### Architecture

```
┌────────────┐
│ Client     │ (mobile app / upstream service)
└─────┬──────┘
      │
      ▼
┌─────────────────────┐
│ API Gateway         │
│ + WAF + Auth        │
└─────────┬───────────┘
          │
          ▼
┌────────────────────────────────┐
│ FastAPI service (async, K8s)   │
│ - 20 pods, HPA on QPS          │
│ - Feature fetch from cache     │
│ - Model inference              │
│ - Audit log (Kafka)            │
└────────┬────────────┬───────┬──┘
         │            │       │
         ▼            ▼       ▼
   ┌─────────┐   ┌────────┐ ┌──────┐
   │ Redis   │   │Snowflake│ │Kafka │
   │ (online │   │(offline │ │(log) │
   │features)│   │features)│ │      │
   └─────────┘   └────────┘ └──┬───┘
                                │
                                ▼
                       ┌──────────────┐
                       │ Model training│
                       │ + monitoring │
                       │ pipeline     │
                       └──────────────┘
```

### Deep dive

**Latency budget:**
- Network ingress: 5ms
- Auth + routing: 3ms
- Feature fetch (Redis): 20ms (p99)
- Fallback to Snowflake on Redis miss: 200ms (rare)
- XGBoost inference: 10ms (ONNX runtime, CPU)
- Audit log (async to Kafka): 0ms (fire-and-forget)
- Response serialization: 2ms
- Total budget: ~40ms p99 → well under 200ms SLO

**Serving pattern:** FastAPI on K8s, not Lambda — 10K QPS sustained is past Lambda break-even. Lambda only if traffic is super spiky.

**Feature fetch:**
- Online: Redis cluster, 3 shards, 99.9% hit rate
- Offline: Snowflake feature store as source of truth
- Write path: Kafka stream from feature computation jobs → Redis + Snowflake

**Model versioning:**
- Registry: MLflow on RDS; ONNX artifact in S3
- Deploy: argo CD watches registry → rolls out new model to K8s
- A/B: Istio traffic splitting 90/10 new version; compare metrics for 1 week

**Audit & compliance:**
- Every request + response + features logged to Kafka (for regulated data, retention 7y)
- KMS-encrypted at rest
- Separate audit DB (append-only, immutable via S3 object lock)

### Monitoring
- Prometheus: RED metrics
- CloudWatch: custom metrics (risk score distribution, approval rate)
- Drift: daily job compares production feature distribution vs training (PSI per feature)
- Alerts: p99 > SLO, approval rate shift >5%, feature freshness >1hr

### Failure modes + mitigations
- Redis down → Snowflake fallback (slower but works)
- Snowflake slow → return cached default + log degraded response
- Model inference slow → circuit breaker → default-decision path
- 5xx spike → auto rollback via health-check-aware canary

---

## 16.4 Design #3 — Multi-Tenant LLM Chatbot with Fine-tuned Adapters

### The ask
"Avrioc needs to serve 50 different tenants, each with their own fine-tuned LLM behavior. Design a cost-efficient architecture."

### Key insight
**Multi-LoRA serving.** One base model in GPU VRAM; 50 tenant-specific LoRA adapters swapped per request.

### Architecture

```
┌──────────────────────────────────────────┐
│ FastAPI gateway                           │
│ - Auth → tenant_id                        │
│ - Per-tenant rate limits + quotas         │
│ - Forward to vLLM with tenant adapter id  │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│ vLLM cluster (Ray Serve)                  │
│ - Base: Llama-3.3-70B AWQ (loaded once)   │
│ - Adapters: 50× LoRA (r=16), swapped      │
│ - Concurrent batched LoRA kernels (SGMV)  │
│ - Autoscale via KEDA on queue depth       │
└──────────────────────────────────────────┘

Offline:
┌──────────────────────────────────────────┐
│ Fine-tuning pipeline (per tenant)         │
│ - Ray Train on labeled tenant data        │
│ - QLoRA r=16, 1 epoch                     │
│ - Register adapter in MLflow              │
│ - Deploy adapter to vLLM (hot-swap)       │
└──────────────────────────────────────────┘
```

### Why this wins
- 50 separate endpoints = 50× compute + memory
- Multi-LoRA: 1 base + 50 small adapters (~20MB each) = ~1GB total adapter storage
- Cost reduction: 10-20× vs naive per-tenant deployment

### Gotchas
- Concurrent requests with different adapters need batched LoRA kernels (vLLM + Punica/SGMV). Naive implementations serialize.
- Adapter swap is milliseconds (memcpy) but sustained throughput requires proper kernel support.
- Tenant isolation: strict per-request adapter dispatch; no cross-tenant leakage.

### Eval + monitoring
- Per-tenant golden sets (30-100 pairs each)
- Per-tenant drift monitoring (query distribution, thumb-down rate)
- Alert if any tenant's metric diverges from fleet

---

## 16.5 Design #4 — Real-Time LLM Streaming + Agent Tool Use

### The ask
"Design an LLM-powered support agent that can stream responses AND call tools (lookup order, issue refund). 1K concurrent users, p99 TTFT <1s."

### Core loop

```
User → LLM (decide: answer or tool?)
           │
           ├── answer → stream tokens to user
           └── tool_call → execute tool → inject result → loop
```

### Architecture

```
┌─────────────┐
│ Next.js UI  │
│ (SSE client)│
└──────┬──────┘
       │ SSE
       ▼
┌──────────────────┐
│ FastAPI (async)  │
│ + LangGraph agent│
└──────┬───────────┘
       │
   ┌───┴───┬──────────┬───────────┐
   ▼       ▼          ▼           ▼
┌─────┐┌──────────┐┌─────────┐┌──────────┐
│ vLLM││Tool dispatch││Langfuse││Guardrails│
│(LLM)││  Jira     ││(trace) ││(input/out│
│     ││  Refunds  ││        ││          │
│     ││  Orders   ││        ││          │
└─────┘└───────────┘└────────┘└──────────┘
```

### Tool-calling flow (LangGraph)

```
┌──────────────┐
│ classify/    │
│ plan (LLM)   │
└──────┬───────┘
       │ decide tool vs answer
   ┌───┴────────┐
   ▼            ▼
┌──────┐    ┌──────────┐
│answer│    │call tool │
└──┬───┘    └────┬─────┘
   │             │
   ▼             ▼
┌──────┐    ┌──────────┐
│stream│    │inject    │
│tokens│    │result →  │
└──────┘    │loop back │
            └──────────┘
```

### Streaming with tool use
Challenge: server-sent events can't "pause" mid-stream for a tool call cleanly.

Pattern: stream `thinking` tokens first, then tool call (as event), then continue tokens after tool result.

```
data: {"type":"thinking","token":"I'll check your order..."}
data: {"type":"tool_call","name":"get_order","args":{"id":"123"}}
data: {"type":"tool_result","output":"shipped 2 days ago"}
data: {"type":"answer","token":"Your order #123 shipped 2 days ago."}
data: [DONE]
```

### Reliability
- Tool timeouts (3s default)
- Max agent iterations (5)
- Circuit breakers on tools
- Fallback: "I couldn't complete that — please contact support"
- Guardrails: input injection detection, output PII redaction

### Scale
- 1K concurrent streaming → 1K WebSocket/SSE connections on FastAPI
- FastAPI async handles this fine (uvloop + asyncio)
- vLLM behind with 10 replicas, continuous batching

### Observability
- Every LangGraph node logged to Langfuse (prompt, response, latency)
- Trace includes tool calls + results
- User feedback (thumb) tied to trace ID
- Metrics: agent iterations p99, tool success rate, E2E latency

---

## 16.6 Interview Q&A — System Design

**Q1. Where do you start in a system design interview?**
> Clarify requirements first — functional, non-functional, constraints. Never dive into the architecture before knowing scale and latency targets. Draw a simple box diagram before deep-diving.

**Q2. How do you show trade-offs explicitly?**
> State the alternative you're NOT picking, and why. "I'd pick Qdrant over Pinecone because data residency matters more than managed simplicity, and Qdrant's Rust performance is ample at our scale."

**Q3. When would you use Lambda vs K8s for an ML API?**
> Lambda: spiky traffic, small models, tight AWS integration, <1 req/sec sustained. K8s: steady traffic, large models, GPU, multi-region. Break-even ~1 req/sec for cost.

**Q4. Multi-tenant LLM — how do you serve cheaply?**
> Multi-LoRA: one base model in VRAM, 50+ tenant-specific adapters swapped per request. 10-20× cost reduction vs per-tenant endpoints. vLLM with enable_lora.

**Q5. Scale bottleneck in an LLM system — where do you look first?**
> (1) GPU memory / KV-cache pressure. (2) CPU on gateway / router. (3) Vector DB query latency. (4) Reranker latency. Profile with traces; don't guess.

**Q6. How do you handle long-context queries without blowing up latency?**
> Prefix caching (cache system + common prompt parts). Chunked prefill to interleave with decode. KV-cache quantization (FP8 → 2× concurrent requests). Route long-context to dedicated replicas.

**Q7. [Gotcha] Your RAG system's p99 latency is great for 95% of queries but 5% are >10s. Why?**
> Tail latency from outlier long-context requests starving the GPU. Head-of-line blocking in the vLLM queue. Fix: chunked prefill, per-request timeout, separate replica pool for long context.

**Q8. How do you handle data residency for UAE?**
> AWS me-central-1 (Bahrain) or Azure UAE regions. Self-hosted models via vLLM for sensitive data. Bedrock only if data classification allows. All observability tools self-hosted (Langfuse, Evidently). Audit logs in S3 with object lock, KMS customer-managed keys.

**Q9. A/B testing two RAG pipelines in production — how?**
> Feature flag at the gateway (LaunchDarkly, Unleash) routes % of traffic to variant B. Both log traces to Langfuse with a `variant` tag. Compare faithfulness, answer relevance, latency, cost, thumb-down rate. Promote after statistical significance (2+ weeks).

**Q10. How do you size vLLM infra for 100 QPS with Llama-70B?**
> TP=4 on 4×A100-80GB → ~25 QPS per replica at typical chat context. 4-5 replicas needed. Add headroom for peak → 6 replicas, auto-scaled by KEDA on queue depth.

**Q11. How do you do canary deploys for a model?**
> 5% → 50% → 100% traffic shift via ingress/service mesh. Metrics gating at each stage (error rate, latency, accuracy proxy). Automatic rollback on gate failure. 24-hour soak at 50% before full cutover.

**Q12. What's the role of Kafka in an ML architecture?**
> Durable audit log (feature values + prediction + timestamp), async feature computation (Kafka → Flink → feature store), decoupling producers from consumers for scale. Alternative: Kinesis on AWS.

**Q13. How do you architect for multi-region HA?**
> Active-active: read replicas in both regions; write primary with async replication; Route53 latency-based routing. Active-passive: primary handles all traffic; DR region pre-warmed; failover on health check.

**Q14. How would you monitor your RAG system's quality continuously?**
> (1) Offline: RAGAS on golden set nightly. (2) Online: A/B compare metrics across variants; user thumb-down rate; regenerate rate. (3) Drift: query distribution embedding clustering. (4) Traces: Langfuse per-call faithfulness score.

**Q15. Cost of running Llama-70B at 100 QPS for a year?**
> Rough: 2× p5.48xlarge (8×H100) = ~$80K/month = ~$1M/year on demand. Reserved: ~40% off = $600K. Cheaper alternatives: AWQ quant + 4×A100-80GB = ~$25-30K/month. Trade-offs: quant lowers quality slightly.

**Q16. [Gotcha] Cost spikes unexpectedly. First thing you check?**
> LLM token usage. Per-tenant/per-user breakdown from gateway logs. Common causes: (1) agent loop (tools called in circles), (2) model upgraded from Haiku to Sonnet silently, (3) new feature generating massive prompts, (4) cache miss spike.

**Q17. How do you design for "sensitive data never leaves the region"?**
> Self-hosted models (vLLM on EKS in region). No SaaS LLM calls. S3 with region-locked bucket policies. KMS keys in region. VPC with no internet gateway (private subnets). Audit log in region with retention.

**Q18. How do you support Arabic + English in one RAG system?**
> Multilingual embedder (BGE-M3, Cohere Embed v3). Single index, language-agnostic. Query-time language detection optional (for analytics). Tokenizer: Qwen or Gemma family for Arabic efficiency.

**Q19. What's your pick for a vector DB in UAE?**
> Qdrant self-hosted on EKS in me-central-1. Rust performance, simple ops, native hybrid + binary quant, payload filtering, data residency compliance. Pinecone if data residency is relaxed.

**Q20. How do you roll out a new base LLM (e.g., Llama-3.3 → Llama-4)?**
> Shadow mode: run both, compare outputs on sampled traces. A/B: 5% traffic to new, compare thumb-down rate + latency. Gradual rollout with gates. Keep old version for 30 days for rollback.

---

## 16.7 Closing signals during system design

### Strong signals to give
- "Let me draw this" → sketch a diagram
- "What's the scale?" / "What's the latency target?" → requirements discipline
- "I'd pick X over Y because..." → trade-off awareness
- "This is the bottleneck under Z load" → analytical thinking
- "We should alert on X" → production mindset

### Weak signals to avoid
- Jumping to implementation without clarifying
- Reciting tools without rationale
- Ignoring cost / scale / data residency
- Not drawing anything
- Over-engineering for phantom requirements

---

Continue to **[Chapter 17 — Behavioral & HR](17_behavioral_hr.md)**.
