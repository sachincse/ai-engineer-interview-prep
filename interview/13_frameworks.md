# Chapter 13 — Frameworks

> **Why this chapter matters more than most:** Every framework in this chapter is named explicitly in the Avrioc JD — FastAPI, LangChain, vLLM, Chainlit, Streamlit. The interviewer will treat this list as a checklist; if you can speak fluently about *why* each framework exists, *when* you'd reach for it, and *what its production gotchas are*, you signal you've actually shipped these things rather than just listed them on a resume. This chapter is structured to give you that fluency, with the production patterns first, the surface-level details second.

---

## 13.1 FastAPI — the standard Python API framework for ML

### Why FastAPI exists

Before FastAPI, the Python web framework world was bifurcated. Flask was simple but offered no input validation, no async support, no automatic documentation. Django was full-featured but heavyweight, ORM-coupled, and slow to spin up for a simple inference service. Sebastián Ramírez built FastAPI in 2018 to fill the gap: built on Starlette for async, Pydantic for validation, automatic OpenAPI docs, and explicit type hints driving the entire schema. Within two years it became the default Python framework for ML APIs because the things ML serving needs — typed input/output, async I/O for downstream LLM calls, automatic Swagger UI for clients — were exactly the things FastAPI made trivial.

### The mental model

```
   ┌─────────────┐   typed   ┌─────────────────┐    your    ┌──────────┐
   │   Client    │──────────▶│  Pydantic Model │──────────▶│  Handler │
   │  (HTTP)     │  (JSON)   │   (validation)  │  (Python) │ function │
   └─────────────┘           └─────────────────┘           └────┬─────┘
                                                                │
                                                                ▼
                                                         ┌──────────────┐
                                                         │  Response    │
                                                         │  Pydantic    │
                                                         │  (serialize) │
                                                         └──────────────┘
```

Every endpoint is a Python function. The function's argument types tell FastAPI how to validate the incoming request. The function's return type tells FastAPI how to serialize the response. The OpenAPI documentation at `/docs` is generated automatically from these types. This is the entire mental model — types drive everything.

### Sync versus async — when each wins

Async wins when your endpoint spends most of its time **waiting on I/O** — a database query, an HTTP call to another service, a cache lookup. The event loop can context-switch to a different request while the current one is waiting, which dramatically improves throughput for I/O-bound workloads. The catch: async only works if your downstream is async too. `httpx.AsyncClient` instead of `requests`. `asyncpg` instead of `psycopg2`. `motor` instead of `pymongo`. If you put a synchronous library inside an `async def` handler, you block the event loop, and every other request hitting that worker queues up behind the blocking call. The result is dramatically worse than just using sync handlers.

Sync handlers in FastAPI are not a downgrade. FastAPI runs sync handlers in a thread pool by design, which gives them concurrency for free up to the thread pool size. For CPU-bound work — like an in-process model inference — sync handlers are typically the right choice because async would just block the event loop anyway.

The decision tree:

```
                      Is your handler I/O-bound (waiting on DB / HTTP / disk)?
                         │
                ┌────────┴────────┐
                │                 │
              Yes                 No (CPU-bound, e.g. model inference)
                │                 │
   Is the downstream library      Use sync handler — FastAPI
   async-compatible?              runs it in a threadpool, which
                │                 gives you concurrency without
        ┌───────┴───────┐         blocking the event loop.
        │               │
       Yes              No
        │               │
   Use async         Either use sync, or async with
   def handler        run_in_threadpool() for the
                      blocking parts.
```

### A worked example — sync ML inference endpoint

Here's a production-shaped FastAPI service for a simple classifier. The point is not the model — it's the structure.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import numpy as np

app = FastAPI(title="Loan Withdrawal Predictor", version="1.4.2")

# Model loaded once at startup — not per request
model: xgb.Booster | None = None

@app.on_event("startup")
def load_model():
    global model
    model = xgb.Booster()
    model.load_model("/opt/model/xgb.bin")

class PredictRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    income: float = Field(ge=0, le=10_000_000)
    credit_score: int = Field(ge=300, le=850)
    loan_amount: float = Field(ge=100, le=1_000_000)

class PredictResponse(BaseModel):
    customer_id: str
    withdraw_probability: float = Field(ge=0, le=1)
    model_version: str

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded yet")
    features = np.array([[req.income, req.credit_score, req.loan_amount]])
    prob = float(model.predict(xgb.DMatrix(features))[0])
    return PredictResponse(
        customer_id=req.customer_id,
        withdraw_probability=prob,
        model_version="1.4.2",
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"status": "ready" if model is not None else "loading"}
```

What to notice. The model loads once at startup, not on every request — this is the single biggest performance trap for new ML engineers. Pydantic models on both request and response give automatic input validation and output type-checking with a 422 error returned automatically on bad input. The `Field` constraints — `ge`, `le`, `min_length` — give cheap, declarative input hardening. Two distinct endpoints `/health` and `/ready` separate "process is alive" from "process can serve traffic" — Kubernetes uses these very differently.

### Streaming endpoints for LLMs

LLM responses arrive token-by-token, and users expect to see them stream. The FastAPI pattern is `StreamingResponse` with an async generator:

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

@app.post("/chat")
async def chat(req: dict, request: Request):
    async def token_stream():
        async with httpx.AsyncClient(timeout=None) as upstream:
            async with upstream.stream(
                "POST", "http://vllm-service:8000/v1/completions",
                json={**req, "stream": True},
            ) as response:
                async for chunk in response.aiter_lines():
                    if await request.is_disconnected():
                        # client gone — stop pulling, free upstream resources
                        break
                    if chunk.startswith("data: "):
                        yield chunk + "\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
```

The trap that catches everyone the first time: returning a generator from a regular `JSONResponse` does not stream. The whole response is buffered and sent at once. You must use `StreamingResponse` and set `media_type="text/event-stream"` for proper SSE framing. The `request.is_disconnected()` check is critical — without it, your upstream LLM call keeps generating tokens that nobody is reading, wasting GPU time.

### Production checklist

A senior reviewer expects these in every FastAPI service:

1. **Pydantic v2 models on every endpoint** — validation, automatic 422 responses, OpenAPI docs.
2. **Structured JSON logs with request_id correlation** — use `structlog` and a middleware that injects the ID into the log context. The same `request_id` should propagate to every downstream call.
3. **OpenTelemetry tracing** — auto-instrumentation hooks into FastAPI, Starlette, httpx, asyncpg. Sends spans to Jaeger or Tempo.
4. **Prometheus metrics middleware** — histograms for latency, request counters, error rates per endpoint. The `prometheus-fastapi-instrumentator` package is the standard.
5. **Rate limiting** — `slowapi` for per-IP or per-token limits, or push it upstream to the API gateway.
6. **Process management** — Gunicorn as the master, Uvicorn workers as children. Workers count: `2 * CPU + 1` is the canonical formula for I/O-bound, drop to `CPU` for CPU-bound.
7. **Distinct `/health` and `/ready`** — `/health` returns 200 if process is alive, `/ready` returns 200 only if model is loaded and downstream dependencies are reachable. K8s uses readiness to gate traffic, liveness to restart.
8. **Request body size limits** — prevents memory exhaustion attacks. Default is 1MB which is too small for image uploads but too large for JSON-only endpoints. Tune.
9. **Graceful shutdown** — use the `lifespan` context manager so startup and shutdown events run correctly. Combine with Kubernetes `terminationGracePeriodSeconds` so in-flight requests can finish.
10. **Authentication and authorization** — typically `OAuth2PasswordBearer` with JWT, or upstream auth at the gateway with the gateway forwarding identity headers.

### FastAPI versus Flask versus Django for ML

| Concern | Flask | FastAPI | Django |
|---------|-------|---------|--------|
| Async support | No (Flask 2 added partial) | Yes, native | Yes, since 3.x |
| Input validation | Manual or Marshmallow | Pydantic, automatic | DRF serializers |
| OpenAPI docs | Manual | Automatic from types | DRF + drf-spectacular |
| Performance | Decent | Highest (Starlette) | Slowest |
| Learning curve | Easiest | Easy | Steepest |
| Ecosystem | Largest | Growing fast | Largest for full apps |
| Best for | Small services, quick prototypes | ML APIs, modern microservices | Full-stack apps with ORM |

For an ML serving role at Avrioc, FastAPI is the default. Mention Flask only if you want to contrast a legacy service migration story.

### Common FastAPI mistakes and gotchas

1. **Loading the model inside the handler.** Every request reloads from disk; latency is dominated by load time. Load at startup.
2. **Using `requests` inside an async handler.** Blocks the event loop. Use `httpx.AsyncClient`.
3. **Conflating `/health` and `/ready`.** Liveness vs readiness mean different things to Kubernetes; collapsing them causes either premature traffic routing or unnecessary restarts.
4. **Forgetting to set `response_model`** on endpoints. Without it, you lose output validation and the OpenAPI schema becomes incomplete.
5. **Returning raw dicts when you have Pydantic models.** Costs you typing and documentation. Always return the Pydantic instance.
6. **Default Uvicorn worker count of 1.** A single worker is a single process, which doesn't use multiple cores. Run Gunicorn with multiple Uvicorn workers in production.

> **How to say this in an interview:** "I use FastAPI as my default because Pydantic gives me input validation for free, async lets me chain LLM and DB calls efficiently, and the OpenAPI docs are auto-generated. The biggest pitfalls are model loading on every request, and mixing sync I/O libraries inside async handlers — both serialize the event loop."

---

## 13.2 LangChain and LangGraph — orchestrating LLM workflows

### Why LangChain exists

LangChain emerged in late 2022 as the first serious attempt to give Python developers a unified abstraction over LLM workflows. Before it, developers wrote raw OpenAI API calls, hand-built prompt templates, hand-built retrieval, hand-built memory. LangChain offered a "Chain" abstraction — a sequence of LLM-and-tool calls — and a "Tool" abstraction for external integrations, plus dozens of pre-built integrations with vector stores, document loaders, and APIs.

The framework has been controversial in production. Its abstractions move quickly between versions, breaking apps. Its layers can be hard to debug because they obscure the underlying LLM calls. But it remains the largest LLM ecosystem in Python by a wide margin, and for prototyping there's nothing faster.

### The mental model

```
   ┌──────┐    ┌────────────┐    ┌──────┐    ┌───────┐    ┌────────┐
   │ User │──▶│  Prompt    │──▶│ LLM  │──▶│ Parser│──▶│ Output │
   │ input│    │  Template  │    │ Call │    │       │    │        │
   └──────┘    └────────────┘    └──────┘    └───────┘    └────────┘
                                       │
                                       │ (optional: tool call)
                                       ▼
                                ┌──────────────┐
                                │  Tool        │ — search, code exec,
                                │  (function)  │   API call, retrieval
                                └──────────────┘
```

A LangChain "chain" is just this pipeline expressed as Python objects. LCEL — LangChain Expression Language — lets you compose chains with the pipe operator: `prompt | llm | parser`. The runnable interface gives every component a uniform `invoke`, `stream`, `batch`, and `astream` API, which is genuinely useful.

### When LangChain wins and when it loses

LangChain wins when you're prototyping and want twenty minutes from idea to working chatbot. It wins when you want pre-built integrations with a long-tail vector store or document format. It wins when your workflow is genuinely a linear chain — query → retrieve → format → answer.

LangChain loses when your workflow is stateful, when you need to pause and resume an agent, when you need to inspect intermediate state without the framework hiding it, or when you need precise control over the request shape. For production agent systems, most teams have moved to LangGraph.

### LangGraph — the production successor

LangGraph is the same team's answer to the production critique. It's a stateful graph framework with explicit nodes and edges, conditional routing between nodes based on state, persistence so you can pause and resume, and human-in-the-loop interruption for approval workflows.

```
   ┌──────────┐
   │  START   │
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ rewrite_ │  rewrite the user's query for retrieval
   │ query    │
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ retrieve │  fetch docs from vector store
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  grade_  │  LLM scores each doc for relevance
   │  docs    │
   └────┬─────┘
        │
   ┌────┴────────┐
   ▼             ▼
score < 0.7    score >= 0.7
   │             │
   ▼             ▼
┌──────────┐   ┌──────────┐
│  web_    │   │ generate │
│  search  │   │  answer  │
└────┬─────┘   └────┬─────┘
     │              │
     └──────┬───────┘
            ▼
         ┌──────┐
         │ END  │
         └──────┘
```

This is a self-correcting RAG flow. Each box is a function that takes the current graph state and returns updated state. Edges between boxes are conditional — they fire based on state values. The whole graph compiles to a checkpointable executor: at any node you can serialize state to a database (Postgres, Redis, SQLite) and resume execution later, possibly on a different machine.

A minimal LangGraph definition for the above flow:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class RAGState(TypedDict):
    question: str
    rewritten_query: str
    docs: list[str]
    relevance_score: float
    answer: str

def rewrite_query(state: RAGState) -> RAGState:
    # call LLM to rewrite question for retrieval
    return {"rewritten_query": rewrite_with_llm(state["question"])}

def retrieve(state: RAGState) -> RAGState:
    return {"docs": vector_store.search(state["rewritten_query"], k=5)}

def grade_docs(state: RAGState) -> RAGState:
    return {"relevance_score": llm_grader(state["question"], state["docs"])}

def web_search(state: RAGState) -> RAGState:
    return {"docs": state["docs"] + tavily_search(state["rewritten_query"])}

def generate(state: RAGState) -> RAGState:
    return {"answer": llm_generate(state["question"], state["docs"])}

def route_after_grade(state: RAGState) -> str:
    return "web_search" if state["relevance_score"] < 0.7 else "generate"

graph = StateGraph(RAGState)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_docs", grade_docs)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate)

graph.set_entry_point("rewrite_query")
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("retrieve", "grade_docs")
graph.add_conditional_edges("grade_docs", route_after_grade,
                             {"web_search": "web_search", "generate": "generate"})
graph.add_edge("web_search", "generate")
graph.add_edge("generate", END)

app = graph.compile()
```

What's powerful here is the explicit state machine. Every transition is visible. Every node is independently testable. Persistence is one config flag away — `graph.compile(checkpointer=PostgresCheckpointer(...))` and now you can pause an agent mid-execution and resume hours later.

### Resume tie-in

> Sachin's TrueBalance Claude-powered ML workspace integrates Jira, GitHub, AWS Athena, and Jenkins — this is exactly a LangGraph-shaped problem. Each external system is a tool node, the agent's reasoning is a state machine, and the workspace can pause and resume based on Jira ticket state changes. Frame it that way in the interview: "I used LangGraph because the workflow is multi-step, stateful, and triggered by external events — Jira tickets transitioning state. Each tool call is a graph node, the agent's working memory is graph state, and the checkpointer means we survive process restarts mid-task."

---

## 13.3 CrewAI — multi-agent orchestration

### Why CrewAI is different

LangChain and LangGraph treat the LLM as the central executor with tools as helpers. CrewAI inverts that: it treats agents as first-class entities with roles, goals, and backstories, and the orchestration is closer to "team of LLM personas collaborating" than "graph of LLM calls."

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Senior Researcher",
    goal="Uncover cutting-edge developments in AI",
    backstory="You're a respected researcher with 20 years in academia.",
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Content Writer",
    goal="Craft engaging blog posts",
    backstory="You write for technical audiences clearly.",
    verbose=True,
)

research_task = Task(
    description="Research the state of LLM agents in production",
    agent=researcher,
)

write_task = Task(
    description="Write a 1000-word blog post on the research findings",
    agent=writer,
    context=[research_task],
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
```

The mental model is "I'm hiring a team of personas." Each agent is a chat-tuned LLM with a persistent role identity. Tasks pass between agents as conversation. The orchestration is conversational, not programmatic.

### When CrewAI wins

CrewAI wins when the problem decomposes naturally into specialized agents — researcher, writer, editor — and the collaboration pattern is "team meeting" rather than "state machine." It's particularly fast for content workflows. It loses when you need precise control over the call graph, when you need to inspect intermediate state, when you need to handle errors deterministically — all the things LangGraph gives you.

For Avrioc specifically, I'd default to LangGraph for production agents and only mention CrewAI if asked about multi-agent patterns specifically.

---

## 13.4 vLLM — high-performance LLM serving

### Why vLLM exists

When ChatGPT shipped and everyone wanted to self-host LLMs, the available serving stacks — naive HuggingFace pipelines, basic Triton — were leaving sixty to eighty percent of GPU memory on the table because of KV-cache fragmentation, and they couldn't keep the GPU busy because they used static batching. vLLM, built at UC Berkeley by the Skypilot team, introduced two innovations that changed everything: PagedAttention for KV-cache management, and continuous batching for scheduling. Together they made vLLM two to four times more efficient than the alternatives, and it became the de-facto open-source LLM serving framework within a year.

### The architecture

```
   ┌──────────────────────────────────────────────────────────┐
   │                    vLLM Engine                           │
   │  ┌──────────┐    ┌──────────────┐    ┌────────────────┐  │
   │  │ Request  │──▶│  Scheduler   │──▶│  Model Runner  │  │
   │  │ Queue    │    │ (waiting,    │    │ (PagedAttn,    │  │
   │  │          │    │  running,    │    │  batched fwd   │  │
   │  │          │    │  swapped)    │    │  pass)         │  │
   │  └──────────┘    └──────────────┘    └────────────────┘  │
   │                          │                                │
   │                          ▼                                │
   │                   ┌──────────────┐                        │
   │                   │  KV Cache    │ (paged blocks,         │
   │                   │  Manager     │  shared prefixes,      │
   │                   │              │  swap to CPU on        │
   │                   │              │  pressure)             │
   │                   └──────────────┘                        │
   └──────────────────────────────────────────────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ OpenAI-compat│
                       │ API server   │  /v1/completions, /v1/chat
                       └──────────────┘
```

The scheduler is the brain. Every iteration — that is, every forward pass through the model — the scheduler looks at three queues. The running queue contains sequences currently generating. The waiting queue contains new requests that haven't started. The swapped queue contains requests whose KV cache was evicted to CPU memory under GPU memory pressure. After each forward pass, the scheduler can promote waiting requests into running, swap running requests out, or swap swapped requests back in. This iteration-level scheduling — also called continuous batching — is why vLLM keeps the GPU busy when sequences finish at very different times.

### The vLLM CLI flags that matter

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 8 \
  --quantization awq \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256
```

`--tensor-parallel-size 8` shards the model across 8 GPUs within a node, so each GPU holds 1/8 of the weights. NVLink's 600-900 GB/s aggregate bandwidth makes this fast. `--quantization awq` loads AWQ-quantized weights, halving memory and roughly doubling throughput at minimal quality cost. `--gpu-memory-utilization 0.9` tells vLLM to use 90% of GPU memory for weights and KV cache, leaving 10% headroom for CUDA context and activations. `--max-model-len 8192` caps the context window, which directly bounds the maximum KV cache size per request. `--enable-chunked-prefill` is the latency-smoothing flag — it interleaves prefill tokens with decode tokens so a long-context user joining mid-batch doesn't stall everyone else. `--max-num-batched-tokens` and `--max-num-seqs` cap batch size by tokens and by sequences respectively.

### vLLM versus alternatives — the comparison everyone asks about

| Framework | Best for | Strengths | Weaknesses |
|-----------|----------|-----------|------------|
| **vLLM** | Default, general-purpose | PagedAttention, continuous batching, broad model support, OpenAI-compatible API, healthy ecosystem | Some H100-specific optimizations come later than NVIDIA's own stack |
| **TGI (HuggingFace)** | HuggingFace-centric stacks | Tight HF ecosystem integration, Rust core | Slightly behind vLLM on raw throughput in 2024-2025 benchmarks |
| **SGLang** | Agent workloads with shared system prompts | RadixAttention for cross-request prefix caching | Newer, smaller ecosystem |
| **TensorRT-LLM** | NVIDIA hardware extreme performance | Best raw H100 throughput, FP8 native | Tight tooling, requires engine recompilation, NVIDIA-only |
| **Text Generation WebUI** | Local single-user development | Fast UI iteration, broad model support | Not for production scale |

For Avrioc, my default would be vLLM because the JD names it explicitly, the ecosystem support is broadest, and the operational maturity is highest. I'd mention SGLang for prefix-cache-heavy agent workloads if the conversation goes there.

### Production deployment of vLLM on Kubernetes

A trimmed but realistic Deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-70b
spec:
  replicas: 2
  selector:
    matchLabels: { app: vllm-llama3 }
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
  template:
    metadata:
      labels: { app: vllm-llama3 }
    spec:
      terminationGracePeriodSeconds: 120
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.6.3
          args:
            - "--model=/models/llama-3-70b-awq"
            - "--tensor-parallel-size=8"
            - "--quantization=awq"
            - "--gpu-memory-utilization=0.9"
            - "--max-model-len=8192"
            - "--enable-chunked-prefill"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: 8
          volumeMounts:
            - name: models
              mountPath: /models
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            periodSeconds: 30
          readinessProbe:
            httpGet: { path: /v1/models, port: 8000 }
            periodSeconds: 10
            failureThreshold: 30   # allow 5 minutes for cold load
          lifecycle:
            preStop:
              httpGet: { path: /drain, port: 8000 }
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: model-weights
```

The lines that matter most. `terminationGracePeriodSeconds: 120` plus the `preStop` drain hook means in-flight requests get up to two minutes to finish during a rolling update. The readiness probe has `failureThreshold: 30` because cold-loading a 70B model can take three to five minutes — without that threshold, the pod would be killed before it ever loaded. `nvidia.com/gpu: 8` requests all 8 GPUs in the node for tensor-parallel sharding. The `model-weights` PVC is shared across pods, populated by an init job from S3.

---

## 13.5 Chainlit — chat UI for LLM apps

### Why Chainlit exists

Once you have a LangChain or LangGraph backend, you need a UI. Streamlit is general-purpose but its widget model is a poor fit for streaming chat. Plain React requires you to write a lot of plumbing. Chainlit was built specifically for chat-shaped LLM apps — it gives you streaming token rendering, tool-call visualization, message-level customization, file uploads, and authentication, all wired in.

### A complete Chainlit chat app in 30 lines

```python
import chainlit as cl
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [
        SystemMessage(content="You are a helpful AI engineer assistant.")
    ])
    await cl.Message(content="Hi! What can I help with today?").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append(HumanMessage(content=message.content))

    llm = ChatAnthropic(model="claude-opus-4-7", streaming=True)

    msg = cl.Message(content="")
    async for chunk in llm.astream(history):
        await msg.stream_token(chunk.content)
    await msg.send()

    history.append(msg)
    cl.user_session.set("history", history)
```

What's notable. `cl.user_session` gives you per-user state without writing session middleware. `msg.stream_token` paints tokens as they arrive — this is what makes the UI feel fast. The whole thing runs with `chainlit run app.py`, no React knowledge required.

### Chainlit versus Streamlit for chat UIs

```
                            Chainlit                Streamlit
                            ─────────               ─────────
   Chat-first UI            ✔ native                ✘ rebuild every msg
   Streaming tokens         ✔ first-class           △ workaround needed
   Tool-call visualization  ✔ auto                  ✘ manual
   Multi-user sessions      ✔ built-in              △ session_state hacks
   Custom data widgets      △ limited               ✔ rich
   Auth                     ✔ OAuth + email built-in   △ third-party
```

Use Chainlit when the app is fundamentally chat. Use Streamlit when the app is a data dashboard with a chat sidecar.

---

## 13.6 Streamlit — data apps

### Why Streamlit exists

Streamlit was the first framework that let a data scientist turn a Python script into a web app without learning HTML, JavaScript, or React. The model is brutally simple: every widget interaction reruns the entire script from top to bottom. That's deeply weird the first time you see it, but it works because Streamlit caches expensive computations and persists state across reruns.

### The execution model

```
                  user moves slider
                         │
                         ▼
          ┌──────────────────────────────┐
          │  Streamlit reruns the entire │
          │  Python script from top to   │
          │  bottom                      │
          └────────────┬─────────────────┘
                       │
                       ▼
          ┌──────────────────────────────┐
          │  @st.cache_data results      │
          │  are replayed from cache     │
          │  (skip recompute)            │
          └────────────┬─────────────────┘
                       │
                       ▼
          ┌──────────────────────────────┐
          │  st.session_state values     │
          │  persist across reruns       │
          └────────────┬─────────────────┘
                       │
                       ▼
          ┌──────────────────────────────┐
          │  Widgets re-render with      │
          │  updated values              │
          └──────────────────────────────┘
```

### A worked example — drift dashboard

This is a Streamlit pattern Sachin can mention as a tie-in to his ResMed work:

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Model Drift Dashboard")

@st.cache_data(ttl=300)  # 5-minute cache
def load_drift_data(model_name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM drift_metrics WHERE model='{model_name}'",
                       conn)

@st.cache_resource
def get_db_connection():
    return create_engine(os.environ["DB_URL"])

conn = get_db_connection()

model = st.selectbox("Model", ["fraud_v1", "fraud_v2", "churn_v3"])
df = load_drift_data(model)

col1, col2 = st.columns(2)
with col1:
    st.metric("Latest PSI", f"{df['psi'].iloc[-1]:.3f}",
              delta=f"{df['psi'].diff().iloc[-1]:.3f}")
with col2:
    st.metric("Drift Status",
              "Stable" if df['psi'].iloc[-1] < 0.1 else "Drifting")

fig = px.line(df, x="date", y="psi", title=f"{model} PSI over time")
fig.add_hline(y=0.1, line_dash="dash", line_color="orange")
fig.add_hline(y=0.25, line_dash="dash", line_color="red")
st.plotly_chart(fig, use_container_width=True)
```

The two caching decorators are the production-critical bits. `@st.cache_data` memoizes a function's return value based on its arguments — perfect for SQL query results. `@st.cache_resource` is for expensive global objects like database connections that should be shared across reruns and across users. Without these decorators, every slider move re-queries the database and your dashboard grinds to a halt.

### Streamlit production deployment

The simplest path is Streamlit Community Cloud — push to GitHub, point the cloud at the repo, done. For production with custom auth, custom domains, and private data sources, deploy as a Docker container behind an authenticating reverse proxy. The container is just `pip install streamlit && streamlit run app.py --server.port=8080` plus your dependencies. Add a Cloudflare Access or AWS Cognito layer in front for SSO.

### Common Streamlit mistakes

1. **No `@st.cache_data`.** Every interaction re-runs every database query. Site is unusable past two users.
2. **Mutating cached return values.** Streamlit's cache returns a reference, not a copy. If you mutate it, the cached value changes too. Either copy explicitly or use `@st.cache_data(persist=True, hash_funcs=...)`.
3. **Putting secrets in the script.** Use `st.secrets` (which reads from a TOML config that doesn't get committed) or environment variables.
4. **Heavy work outside `@st.cache_data`.** A model load that takes 3 seconds runs every interaction. Wrap it.

> **How to say this in an interview:** "Streamlit's mental model is that the whole script reruns on every widget interaction. The two production-critical decorators are `@st.cache_data` for results and `@st.cache_resource` for connections — without them dashboards are unusable past a few users. For pure chat UI I'd reach for Chainlit instead; Streamlit shines when the app is a data dashboard with optional chat."

---

## 13.7 Putting it all together — the framework choice tree

When the interviewer asks "what would you build a chatbot with?" the wrong answer is "LangChain plus Streamlit." The right answer is a decision tree:

```
                         What's the workload shape?
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        │                         │                          │
   Single-shot Q&A           Stateful agent             Data dashboard
   over a corpus             with multi-step             with charts
        │                    reasoning                       │
        │                         │                          │
        ▼                         ▼                          ▼
   LangChain LCEL              LangGraph                  Streamlit
   chain + FastAPI             + FastAPI                  (with cache_data
   for the API                 + Chainlit                  decorators)
   + Chainlit for UI           for UI

   Backend:                    Backend:                   Backend:
   - vLLM for LLM              - vLLM (LLM)              - LLM optional
   - pgVector / Qdrant         - LangGraph state in PG    - SQL warehouse
     for retrieval             - tools as graph nodes
```

For Avrioc specifically — given they name FastAPI, vLLM, LangChain, Chainlit, Streamlit all in the JD — they're not asking which one you know. They're asking whether you can pick the right combination for a given problem. Speak in combinations: "I'd build the chat-shaped product as Chainlit + LangGraph + vLLM behind FastAPI; the analytical dashboard as Streamlit + SQL + cached widgets."

---

## 13.8 Interview questions with full narrative answers

### Q. Walk me through how you'd choose between LangChain and LangGraph.

The split is stateless versus stateful. LangChain's strength is single-shot pipelines — query goes in, retrieve, format, generate, answer comes out. The whole computation is a function call. LangGraph's strength is multi-step processes where the agent might pause, branch on intermediate results, or need to be resumed later. So for a simple RAG endpoint, LangChain is fine. For an agent that opens Jira tickets, runs Athena queries, and updates Jenkins jobs based on what it finds — that's stateful, branching, and possibly long-running, so LangGraph. The mental test I run: if I were to pause the workflow halfway through and serialize its state, would the downstream steps need that state to continue? If yes, LangGraph. If no, LangChain.

### Q. What's the difference between sync and async FastAPI handlers, and what's the most common mistake?

Sync handlers are run by FastAPI in a thread pool, so they get concurrency without blocking the event loop. Async handlers run on the event loop directly, which is great for I/O-bound work where you can yield while waiting on a database or HTTP call. The most common mistake is using a synchronous library inside an async handler — like `requests` instead of `httpx.AsyncClient`. That blocks the event loop and serializes every request that hits that worker, which is dramatically worse than just using a sync handler. The second most common mistake is loading the model inside the handler instead of at startup, which makes every request pay the load cost.

### Q. Explain PagedAttention in two sentences.

PagedAttention is vLLM's KV-cache management — it borrows the OS virtual-memory idea and applies it to GPU memory, breaking each request's KV cache into fixed-size pages that don't have to be contiguous. The result is that GPU memory fragmentation drops from sixty-to-eighty percent to under four percent, which means we fit two-to-four times more concurrent requests on the same hardware.

### Q. Why might you choose Chainlit over Streamlit for an LLM chatbot?

Chainlit is purpose-built for chat — streaming tokens, tool-call visualization, multi-turn message rendering, file uploads, and per-user sessions all work out of the box. Streamlit's execution model — re-running the whole script on every interaction — is a poor fit for chat because the rebuild discards the streaming state. You can hack Streamlit into a chat UI, but it's swimming upstream. The rule I follow: if the primary interaction is back-and-forth messages, use Chainlit. If the primary interaction is data exploration with a chat sidecar, use Streamlit.

### Q. What's the production checklist for a FastAPI service serving an LLM?

Pydantic v2 input validation, structured JSON logs with a request_id propagated through every downstream call, OpenTelemetry traces, Prometheus latency histograms, slowapi rate limiting, Gunicorn-managing-Uvicorn-workers process model with `2 * CPU + 1` workers tuned for the workload, distinct `/health` and `/ready` endpoints since Kubernetes uses them differently, request body size limits to prevent memory exhaustion, graceful shutdown via the lifespan context manager. For the LLM specifically: streaming responses via `StreamingResponse` with `request.is_disconnected()` checks so we free upstream calls when clients leave, and a circuit breaker on the upstream LLM service so cascading failures don't take everything down.

### Q. What does `--enable-chunked-prefill` do in vLLM and why does it matter?

It interleaves prefill compute with decode compute in the same batch. Without it, when a long-context user joins an in-flight chat batch, the prefill phase — which is compute-bound and proportional to context length — stalls all the decoding users waiting for their next token, causing TTFT spikes for them and ITL spikes everywhere. Chunked prefill breaks the long prefill into pieces small enough to interleave with decode tokens, smoothing out latency across the batch. For chat workloads where TTFT matters more than throughput, it's almost always worth enabling.

---

End of Chapter 13. Continue to **[Chapter 14 — Monitoring & Drift](14_monitoring_drift.md)** or back to **[Chapter 00 — Master Index](00_index.md)**.
