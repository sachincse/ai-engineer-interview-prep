# Chapter 18 — Cheatsheet
## Last-minute revision — formulas, commands, numbers, names

> Skim this on the morning of the interview. Don't try to memorize — just refresh.

---

## 18.1 Formulas you might need to write on a whiteboard

### Self-attention
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

### Multi-head attention
```
MHA(X) = Concat(head_1, ..., head_h) W_O
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
```

### RoPE rotation (per dim-pair)
```
θ_i = 10000^(-2i/d)
R(pos) rotates (x_2i, x_2i+1) by pos · θ_i
```

### LoRA update
```
y = Wx + B·A·x     where B ∈ ℝ^(d_out × r), A ∈ ℝ^(r × d_in), B init = 0
Effective scale = α / r
```

### Softmax with temperature
```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

### InfoNCE (contrastive)
```
L = -log[ exp(sim(q, p+)/τ) / Σ exp(sim(q, p_i)/τ) ]
```

### PSI (drift)
```
PSI = Σ (p_actual - p_expected) · ln(p_actual / p_expected)
<0.1 stable, 0.1-0.25 moderate, >0.25 significant
```

### Chinchilla scaling
```
Compute-optimal: 20 tokens per parameter
```

### FLOP estimate (training)
```
Forward + backward ≈ 6 · params · tokens
```

### DPO loss
```
L = -log σ( β · log[π(y_w|x)/π_ref(y_w|x)] - β · log[π(y_l|x)/π_ref(y_l|x)] )
```

### Cross-entropy (next-token)
```
L = - (1/T) Σ_t log P(x_t | x_<t)
```

### Perplexity
```
PPL = exp(CE loss)
```

### Cosine similarity
```
cos(a, b) = (a·b) / (||a|| · ||b||)
```

---

## 18.2 Numbers to know (orders of magnitude)

### Model sizes
- GPT-3: 175B params, 300B tokens
- LLaMA-2 70B: 70B, 2T tokens
- LLaMA-3 8B: 8B, 15T tokens (over-trained per Chinchilla)
- LLaMA-3 70B: 70B, 15T tokens
- LLaMA-3 405B: 405B, 15T tokens

### Memory (FP16, inference)
- 7B model: ~14 GB
- 13B model: ~26 GB
- 70B model: ~140 GB (needs 2×H100 with TP=2)
- 405B model: ~810 GB (multi-node TP + PP)

### KV-cache per token (FP16)
```
2 (K,V) × layers × hidden_dim × 2 bytes
Llama-2 7B: 2 × 32 × 4096 × 2 = 512 KB/token
Llama-2 70B: 2 × 80 × 8192 × 2 = 2.5 MB/token (MHA)
            ÷ 8 (GQA) = 320 KB/token
```

### Latency targets (chat)
- TTFT: <500ms
- TPOT: <50ms
- E2E: <3s

### Latency targets (real-time inference)
- p50: <50ms
- p99: <500ms (your TrueBalance SLO)

### Quantization memory reduction
- FP16: 1×
- INT8 / FP8: 0.5× (2× savings)
- INT4: 0.25× (4× savings)

### Vector DB scale
- <1M: anything
- <10M: HNSW (pgVector, Qdrant)
- <100M: IVF-Flat / HNSW
- 100M-1B: IVF-PQ
- 1B+: IVF-PQ + DiskANN

### Typical LLM pricing (check before interview)
- Claude Opus 4.x: ~$15/$75 per M input/output tokens
- Claude Sonnet 4.x: ~$3/$15 per M
- Claude Haiku 4.x: ~$1/$5 per M
- GPT-5: ~$5/$20 per M (ballpark)
- Self-hosted Llama-70B: ~$0.5-1 per M tokens at scale

---

## 18.3 Names to know (don't misspeak)

### Papers
- **Attention Is All You Need** (2017, Vaswani et al.) — the Transformer
- **BERT** (2018, Devlin et al.)
- **GPT** (2018-2020)
- **Chinchilla** (2022, Hoffmann et al.)
- **LoRA** (2021, Hu et al.)
- **FlashAttention** (2022, Dao)
- **DPO** (2023, Rafailov et al.)
- **RoFormer / RoPE** (2021, Su et al.)

### Architectures
- **GQA** (LLaMA-2 introduced)
- **MLA** (DeepSeek-V2/V3)
- **SwiGLU** (LLaMA family)
- **RMSNorm** (LLaMA family)
- **Mixture of Experts** (Mixtral, DeepSeek)

### Frameworks
- **vLLM** — PagedAttention, continuous batching
- **TensorRT-LLM** — NVIDIA, FP8, top throughput
- **SGLang** — RadixAttention, agentic
- **Ray** — distributed Python ML
- **KServe** — K8s model serving
- **LangGraph** — stateful agents
- **RAGAS** — RAG evaluation
- **MLflow** — experiment tracking + registry

### Embedding models
- **BGE-M3** (BAAI) — multi-granularity, multilingual
- **E5-Mistral-7B** — decoder-based
- **NV-Embed-v2** — decoder-based
- **OpenAI text-embedding-3-large** — Matryoshka
- **Cohere Embed v3** — managed, multilingual

### Quantization names
- **AWQ** — Activation-aware Weight Quantization
- **GPTQ** — Second-order-aware PTQ
- **GGUF** — llama.cpp container (Q4_K_M sweet spot)
- **SmoothQuant** — W8A8 unlock
- **NF4** — QLoRA's 4-bit datatype

### Alignment methods
- **SFT** — Supervised Fine-Tuning
- **RLHF** — RL from Human Feedback (PPO-based)
- **DPO** — Direct Preference Optimization (2026 default)
- **KTO** — Kahneman-Tversky Optimization
- **ORPO**, **SimPO**, **IPO** — variants

---

## 18.4 Commands you might need to demo

### Docker / K8s
```bash
# Build + tag
docker build -t acme/rag-api:v12 -f Dockerfile .

# Deploy
kubectl apply -f deployment.yaml
kubectl rollout status deployment/rag-api
kubectl logs -f deployment/rag-api

# Check autoscale
kubectl get hpa
```

### vLLM launch
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct-AWQ \
  --quantization awq \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 4 \
  --enable-prefix-caching \
  --enable-lora --max-loras 16
```

### Curl to an OpenAI-compatible endpoint
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct-AWQ",
    "messages": [{"role":"user","content":"Hello"}],
    "temperature": 0.2,
    "max_tokens": 512,
    "stream": true
  }'
```

### Ray Serve
```python
import ray
from ray import serve

@serve.deployment(num_replicas=3, ray_actor_options={"num_gpus": 1})
class Embedder:
    def __init__(self):
        self.model = load_model()

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.05)
    async def embed(self, texts: list[str]):
        return self.model.encode(texts).tolist()

serve.run(Embedder.bind())
```

### FastAPI skeleton
```python
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(req: Request):
    return app.state.model.predict(req.features)
```

### Qdrant (Python client)
```python
from qdrant_client import QdrantClient
client = QdrantClient("localhost", port=6333)

client.create_collection(
    "docs",
    vectors_config={"size": 1024, "distance": "Cosine"}
)
client.upsert("docs", points=[...])
hits = client.search("docs", query_vector=q, limit=10, query_filter=flt)
```

---

## 18.5 Production patterns summary

### "Production LLM chatbot" stack
```
React/Next.js → FastAPI (async) → LangGraph (agent) → vLLM (LLM) + Qdrant (RAG)
                                                      ↓
                                                 Langfuse (trace)
                                                 LiteLLM (router)
                                                 Guardrails (safety)
                                                 Redis (cache)
```

### "Real-time ML inference" stack
```
Client → API Gateway → FastAPI on K8s (HPA) → ONNX model → response
                              ↓ (async)
                          Kafka (audit) → Snowflake (features) → Retraining
                              ↓
                          Prometheus + Grafana (metrics)
                          Datadog (drift)
                          MLflow (registry)
```

### "Fine-tune + deploy" stack
```
Data → Ray Data (clean) → Ray Train (QLoRA) → MLflow (register adapter)
                                                    ↓
                                                vLLM enable_lora (deploy)
                                                    ↓
                                                RAGAS eval → promote
```

---

## 18.6 The 10-minute pre-interview pep talk

You have 8 years of production ML experience across 4 companies in fintech, healthcare, and enterprise. You've shipped real-time APIs, RAG systems, agent systems, MLOps frameworks, drift monitoring, and fine-tuned classifiers. You know AWS deeply and Azure respectably. You use Claude, vLLM, LangChain, LangGraph day-to-day.

You are the person the JD describes.

Interviewers want to hire. They want you to be good. Give them the evidence.

### Your three throw-lines (use twice each during the interview)
1. "At TrueBalance, I shipped X — let me connect it to what you're asking about..."
2. "At ResMed, we hit this exact problem — here's how we solved it..."
3. "The trade-off here is X vs Y; for your context I'd pick Z because..."

### When stuck
> "Let me think for a moment. [10 seconds.] My intuition is X, but let me reason through whether that holds. [Reason aloud.] So the answer is Y."

### When you don't know
> "I haven't done this specific thing, but here's how I'd approach it..." Then list first principles.

### When you realize you were wrong
> "Actually, I want to revise what I said — let me correct that." Interviewers *love* this.

---

## 18.7 The 60-second elevator pitch (print and memorize)

> Sachin Singh. Senior ML Engineer with 8 years of experience productionizing ML and LLM systems across fintech, healthcare, and enterprise. Currently at TrueBalance — real-time XGBoost Lambda with p99 under 500 milliseconds across 3 VPC-isolated environments, and a Claude-powered developer platform unifying Jira, GitHub, Athena, Jenkins for the ML team. Before that, 2.5 years at ResMed — 8 models to production in 6 months, a RAG-based clinical chatbot, and Datadog drift dashboards that became the team standard. Sweet spot: bridging research to production — LLMOps with vLLM and Kubernetes, real-time inference, quantization, and the monitoring that keeps models healthy at scale. That's exactly the bridge Avrioc's JD describes, which is why this onsite Abu Dhabi role excites me.

---

## 18.8 Final reminders

- You are qualified. The JD is almost custom-written for you.
- Energy, warmth, curiosity > perfection.
- Ask clarifying questions before every non-trivial answer.
- Draw diagrams unsolicited.
- Be honest about what you don't know — senior engineers respect it.
- Close with "thank you for the time — I enjoyed the conversation."

Good luck. Go get the offer.

---

**End of pack.** Back to **[Chapter 00 — Master Index](00_index.md)**.
