# Chapter 13 — Frameworks
## FastAPI, LangChain, LangGraph, CrewAI, vLLM, Chainlit, Streamlit

> Every single framework the JD mentions lives here. This is your pass/fail chapter for "productionizing LLM-based applications."

---

## 13.1 FastAPI — the standard Python API framework

### Why FastAPI for ML
- Async-first (for I/O-bound workloads, LLM calls, DB queries)
- Pydantic validation (clean schema for requests/responses)
- OpenAPI auto-docs (Swagger at `/docs`)
- Fast (Starlette + uvloop)
- Rich ecosystem (auth, middleware, background tasks)

### Basic structure
```python
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once
    app.state.model = load_model()
    app.state.vector_db = connect_vector_db()
    yield
    # Shutdown: close connections
    await app.state.vector_db.close()

app = FastAPI(lifespan=lifespan)

class Query(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

class Answer(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: float

@app.post("/rag", response_model=Answer)
async def rag_endpoint(q: Query, user=Depends(get_current_user)):
    chunks = await app.state.vector_db.search(q.text, q.top_k)
    answer = await app.state.model.generate(q.text, chunks)
    return Answer(**answer)
```

### Sync vs async
- `async def` for I/O-bound (DB, HTTP, LLM calls). Requires async-compatible libraries.
- `def` for CPU-bound (runs in a threadpool). Don't mix a blocking call inside `async def` — it stalls the event loop.

### Pydantic v2 — what changed
- 5-50× faster validation (Rust core)
- `model_config` instead of `Config` class
- `.model_dump()` instead of `.dict()`
- Stricter type coercion
- **Breaking changes** — migrate carefully.

### Dependency injection
```python
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.get("/users/{id}")
def read_user(id: int, db = Depends(get_db)):
    return db.query(User).get(id)
```

For app-lifetime singletons (models, LLM clients, Redis), load in `lifespan` and access via `request.app.state` — don't use `Depends` to reinstantiate heavy objects per request.

### Streaming responses (LLM)
```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def stream(q: Query):
    async def event_gen():
        async for tok in llm.stream(q.text):
            yield f"data: {json.dumps({'token': tok})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

Disable nginx proxy buffering (`proxy_buffering off`).

### Security
- API key / JWT via `Security()` dependencies
- Rate limiting (SlowAPI or gateway)
- Input schema validation (Pydantic)
- Output guardrails
- CORS (explicit allowlist in prod)

### Observability
- `prometheus-fastapi-instrumentator` for RED metrics
- OpenTelemetry for traces (export to Tempo / Jaeger)
- Structured logging (structlog / loguru)
- For LLM apps: wrap calls with spans; add token counts as span attrs

### Background tasks — when NOT to use
```python
@app.post("/send-email")
async def send_email(bt: BackgroundTasks):
    bt.add_task(send_email_fn, ...)
```

Runs in-process after the response. Fine for fire-and-forget logging. **Bad for anything that might fail, retry, or outlive the worker.** Use Celery/RQ/Arq with a broker for durable jobs.

---

## 13.2 LangChain — the building blocks

### What LangChain gives you
- Abstractions: `LLM`, `ChatModel`, `Embeddings`, `VectorStore`, `Retriever`, `OutputParser`
- **LCEL** (LangChain Expression Language) — declarative pipelines via `|` operator
- Integrations with 100+ providers
- Built-in RAG utilities (text splitters, chunkers)

### LCEL example — a simple RAG chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template("""
Use ONLY the context to answer.
Context: {context}
Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What does the compliance doc say about ...?")
```

### When LangChain shines
- Rapid prototyping
- Pre-built integrations
- Simple linear pipelines (LCEL is clean)

### When LangChain hurts
- Complex state machines (use LangGraph)
- Production reliability (heavy abstractions hide bugs)
- Opinionated interfaces churn between versions

---

## 13.3 LangGraph — stateful agents done right

### The mental model
A **graph** of nodes. Each node is a function that reads/writes a shared State. Edges define transitions. **Supports cycles, conditional branches, human-in-the-loop, checkpointing.**

```
     ┌─────────┐
     │  Start  │
     └────┬────┘
          ▼
    ┌──────────────┐
    │ classify     │
    │ (LLM call)   │
    └──────┬───────┘
           │
      ┌────┴─────┐
      │ decision │
      └─┬──┬────┘
        │  │
   order│  │query
        ▼  ▼
  ┌──────┐ ┌───────┐
  │process│ │ answer│
  └──┬───┘ └───┬───┘
     ▼         │
  ┌──────┐     │
  │verify│     │
  └──┬───┘     │
     ▼         ▼
    ┌─────────┐
    │   End   │
    └─────────┘
```

### Code structure
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    query: str
    intent: str
    result: str
    retries: int

def classify(state: State) -> State:
    state["intent"] = llm.classify(state["query"])
    return state

def answer(state: State) -> State:
    state["result"] = llm.answer(state["query"])
    return state

def route(state: State) -> str:
    return "answer" if state["intent"] == "query" else "process_order"

graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("answer", answer)
graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("answer", END)

app = graph.compile(checkpointer=PostgresSaver(conn))
```

### Checkpointers — the production feature
- **MemorySaver** — local, ephemeral
- **SqliteSaver** — single-node persistent
- **PostgresSaver** — production; enables resume-after-crash, multi-turn chat via thread_id, human-in-the-loop

### When to upgrade from LangChain to LangGraph
Any flow with a `while` or `if` loop over LLM calls. Linear chains → LCEL. Complex flows → LangGraph.

---

## 13.4 CrewAI — role-based multi-agent

### The approach
Declarative: define agents with goals + backstories, let them collaborate.

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Senior Researcher",
    goal="Find the most recent papers on X",
    backstory="You have a PhD in ML and...",
    tools=[search_tool, arxiv_tool],
)
writer = Agent(
    role="Technical Writer",
    goal="Summarize the papers for a business audience",
    tools=[],
)

t1 = Task(description="Find top 5 papers on vLLM", agent=researcher)
t2 = Task(description="Write a 500-word summary", agent=writer, context=[t1])

crew = Crew(agents=[researcher, writer], tasks=[t1, t2])
result = crew.kickoff()
```

### Where CrewAI wins
- Fast prototyping of "team" workflows
- "Researcher + Writer + Critic" patterns
- Internal tools, demos

### Where LangGraph wins
- Customer-facing agents (need determinism, observability)
- Complex state transitions
- Human-in-the-loop

**Rule of thumb (2026):** CrewAI for demos and internal tools; LangGraph for production.

---

## 13.5 vLLM — the production LLM server

### What vLLM is
High-throughput, memory-efficient inference engine for LLMs. Open-source, Python, CUDA-accelerated.

### Core features
- **PagedAttention** — KV-cache as virtual memory pages
- **Continuous batching** — 5-10× throughput vs static batching
- **Prefix caching** — shared prompts don't recompute KV
- **Tensor parallelism** — multi-GPU
- **Speculative decoding** — 2-3× latency via draft model
- **Quantization** — AWQ, GPTQ, FP8
- **Multi-LoRA** — one base + many adapters
- **OpenAI-compatible API** — drop-in replacement

### Launching vLLM
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct-AWQ \
  --quantization awq \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --enable-lora \
  --max-loras 16 \
  --max-model-len 8192
```

### Sizing a vLLM deployment
```
GPU memory = weights + KV cache + activations
Weights:     2B/param (FP16) or 0.5B/param (INT4)
KV cache:    2 * L * h * d_k * seq * bytes per req
Activations: ~10% overhead

Llama-3 70B FP16: ~140GB weights + 20GB KV (typical load)
                  → 2×H100-80GB with TP=2 or 4×A100-80GB with TP=4
```

### Tuning `--gpu-memory-utilization`
- Default 0.9
- Leave headroom for activation spikes
- Monitor eviction rate; raise if low

### When vLLM beats alternatives
- Production serving at scale (throughput)
- Multi-tenant LoRA
- OpenAI API compatibility (drop-in for existing apps)

### When TRT-LLM or SGLang wins
- Pure NVIDIA throughput at fixed hardware (TRT-LLM adds 20-30%)
- Agentic / tree-search workloads (SGLang RadixAttention)

---

## 13.6 Chainlit — LLM chat UI in 5 minutes

### What it is
A Python library that gives you a ChatGPT-like UI for any LLM pipeline. Built for **developers prototyping** — not polished product UIs.

### Minimal example
```python
import chainlit as cl
from langchain_core.runnables import RunnableConfig

@cl.on_chat_start
async def start():
    cl.user_session.set("chain", build_rag_chain())

@cl.on_message
async def on_message(msg: cl.Message):
    chain = cl.user_session.get("chain")
    async for token in chain.astream({"query": msg.content}):
        await cl.Message(content=token).send()
```

### Features
- Streaming responses
- Chat history / threads
- Auth (OAuth, API key)
- Dark mode, markdown, code blocks
- Step-by-step agent display (shows intermediate tool calls)
- Deployment: `chainlit run app.py`

### When Chainlit fits
- Demos to stakeholders
- Internal tools for Data Scientists and Analysts
- Rapid PoCs for hackathons

### When it doesn't
- Customer-facing polished product UI (use Next.js + backend API)
- Heavy custom UI requirements

---

## 13.7 Streamlit — dashboards + light apps

### What it is
Script-based dashboards in Python. Widget-driven, reruns top-to-bottom on interaction.

### Example
```python
import streamlit as st

st.title("Credit Risk Predictor")

income = st.number_input("Monthly income (INR)", min_value=10000)
loan = st.number_input("Loan amount", min_value=50000)

if st.button("Predict"):
    risk = model.predict([[income, loan]])[0]
    st.metric("Default probability", f"{risk:.1%}")
    st.bar_chart(risk_factors(income, loan))
```

### Key patterns
- `st.session_state` for persisting across reruns
- `@st.cache_resource` for heavy objects (models)
- `@st.cache_data` for data frames
- Deploy: Streamlit Cloud, Docker container, AWS ECS

### When Streamlit fits
- Internal analytics dashboards
- DS prototypes for business stakeholders
- ML model demo pages

### When it doesn't
- Customer-facing products (poor UX polish)
- Complex multi-page apps (use Plotly Dash or React)
- High concurrent users

---

## 13.8 LiteLLM — LLM gateway

### The pattern
One unified API for 100+ LLM providers; route/cache/track/budget at the gateway.

```python
from litellm import completion
response = completion(
    model="claude-opus-4-6",  # or "openai/gpt-5", "azure/gpt-5", "bedrock/anthropic.claude-..."
    messages=[{"role": "user", "content": "..."}]
)
```

### Proxy mode
Run `litellm --config config.yaml` as a service. Apps call the gateway's OpenAI-compatible endpoint; gateway handles routing, caching, budgets, fallback, cost tracking.

### Avrioc use case
Route cheap queries to Haiku, expensive to Opus. Track per-tenant cost. Cache common queries. Single config file controls all LLM traffic.

---

## 13.9 MCP — Model Context Protocol

### What it is
Open protocol (Anthropic, Nov 2024) standardizing how LLM clients connect to tools/data.

### Architecture
- **MCP server** exposes: **tools** (actions), **resources** (readable data), **prompts** (templates)
- Transports: stdio or HTTP/SSE
- Any MCP-compatible client (Claude Desktop, Cursor, your agent) uses them

### Why it matters
Avoid N×M integrations. Write a Jira MCP server once → every client can use it.

### For your resume
Your "ML workspace assistant integrating Jira, GitHub, Athena, Jenkins" is exactly this pattern, even if not explicitly labeled MCP. Be ready to pivot that story to MCP terminology.

---

## 13.10 Interview Q&A — Frameworks

**Q1. Sync vs async FastAPI endpoints — default?**
> `async def` for I/O-bound (DB, HTTP, LLM). `def` for CPU-bound (runs in threadpool). Never mix a blocking call inside async — stalls the event loop.

**Q2. Pydantic v2 — what changed for FastAPI?**
> Rust core: 5-50× faster validation. Breaking: `model_config` replaces `Config`, `.model_dump()` replaces `.dict()`, stricter type coercion. `TypeAdapter` for non-model validation.

**Q3. FastAPI dependency injection — when and when not?**
> Use for per-request resources (DB session, auth user). Don't use for app-lifetime heavy objects (models, LLM clients) — load in `lifespan`, access via `app.state`.

**Q4. When should you NOT use FastAPI background tasks?**
> BackgroundTasks run in-process after the response. Fine for fire-and-forget logging. Bad for durable jobs that might fail/retry — use Celery/RQ/Arq with a broker.

**Q5. How do you stream LLM responses in FastAPI?**
> `StreamingResponse` with an async generator emitting SSE (`data: {...}\n\n`). Async LLM client. Disable nginx `proxy_buffering`.

**Q6. [Gotcha] Your FastAPI app leaks memory under load. Where do you start?**
> (1) Unclosed httpx/DB clients created per-request (should be lifespan singletons). (2) Large responses held in middleware. (3) Pydantic models with unbounded lists accumulating. (4) Logging handlers retaining frames. Tools: tracemalloc, memray, py-spy dump.

**Q7. LangChain LCEL vs LangGraph — when each?**
> LCEL for linear pipelines (retriever → prompt → LLM → parser). LangGraph for stateful, cyclic, conditional flows. Rule: any `while` or `if` over LLM calls → LangGraph.

**Q8. LangGraph checkpointers — why and which?**
> Persist graph state after every node. Enable resume-after-crash, human-in-the-loop, time-travel debugging, multi-turn via thread_id. Production: PostgresSaver.

**Q9. CrewAI vs LangGraph — choose?**
> CrewAI: role-based, declarative, fast to prototype. LangGraph: imperative graph, full control, reliable production. CrewAI for demos/internal; LangGraph for customer-facing.

**Q10. What is MCP?**
> Model Context Protocol (Anthropic, 2024): standardized way for LLM clients to connect to tools/resources/prompts over stdio or HTTP/SSE. Avoids N×M integration sprawl.

**Q11. vLLM core features?**
> PagedAttention (paged KV-cache), continuous batching (5-10× throughput), prefix caching, tensor parallelism, speculative decoding, AWQ/GPTQ/FP8 quant, multi-LoRA, OpenAI-compatible API.

**Q12. How do you size a vLLM deployment?**
> GPU mem = weights + KV cache + activations. For Llama-70B FP16: ~140GB weights + ~20GB KV → 2×H100-80GB (TP=2) or 4×A100-80GB (TP=4). Use `--gpu-memory-utilization 0.9`.

**Q13. [Gotcha] vLLM throughput collapses under long-context traffic. Why?**
> KV-cache pressure. Long contexts = more cache blocks = fewer concurrent requests, vLLM preempts. Fixes: enable prefix caching, more TP for more aggregate VRAM, cap max_model_len, route long-context to dedicated replicas.

**Q14. vLLM vs TensorRT-LLM vs SGLang?**
> vLLM: breadth (models, quantizations, ease). TRT-LLM: pure NVIDIA throughput via kernel fusion + FP8. SGLang: agentic/tree-search (RadixAttention, better scheduling). Most teams pick vLLM; swap to TRT-LLM for 20-30% more on fixed hardware.

**Q15. When do you use Chainlit vs Streamlit vs a React frontend?**
> Chainlit: LLM chat UIs for prototypes/internal tools (streaming, threads, agent steps). Streamlit: dashboards + ML demos. React/Next.js: customer-facing polished products.

**Q16. Streamlit caching — `cache_resource` vs `cache_data`?**
> `@st.cache_resource` for singleton heavy objects (models, DB connections). `@st.cache_data` for dataframes / immutable returns (hash by args).

**Q17. LiteLLM gateway — value?**
> Unified OpenAI-compatible API across 100+ providers. Route, cache, budget, fallback, cost-track. One config changes routing without code changes. Essential for multi-model cost optimization.

**Q18. How do you implement tool-calling with LangChain?**
> `.bind_tools([...])` attaches schemas; model emits JSON tool calls. `ToolNode` or custom executor dispatches; results feed back in. For production, use LangGraph — more control.

**Q19. How do you test LangGraph agents?**
> (1) Unit test each node (pure functions given state). (2) Integration test the graph with mocked LLM calls. (3) Golden-set evaluation on end-to-end traces. (4) Load test with replayed traces. (5) Observability (LangSmith / Langfuse) for production debugging.

**Q20. [Gotcha] Your LangGraph agent loops forever. Fix?**
> Recursion limit: `.invoke(..., {"recursion_limit": 25})`. Inspect traces — usually the agent re-calls the same tool because it ignores the result. Mitigate: dedupe tool calls in a pre-model hook; add a "reflect" node forcing the model to answer or escalate after N tool calls.

**Q21. FastAPI + OpenTelemetry — setup?**
> `opentelemetry-instrumentation-fastapi` for auto-instrumentation. Export to an OTEL collector. Routes: metrics → Prometheus, traces → Tempo/Jaeger, logs → Loki. Wrap LLM calls with custom spans; include token counts as span attributes.

**Q22. How do you secure a FastAPI LLM endpoint?**
> Stack: API key or OAuth2/JWT via `Security()`, rate limiting (SlowAPI or gateway), input Pydantic validation, output guardrails, per-tenant budget, CORS allowlist, mTLS for internal. Auth key quota enforced before the LLM call.

**Q23. [Gotcha] CrewAI crew takes 2 minutes per run. Optimize?**
> (1) Parallelize tasks — `Crew(process=Process.hierarchical)` or let agents run concurrently. (2) Use cheaper models for non-critical agents. (3) Shorter system prompts. (4) Disable verbose. (5) Cache tool calls.

**Q24. How do you deploy a Streamlit app on AWS?**
> Containerize: Streamlit in Docker, expose port 8501. Deploy via ECS Fargate or EKS. Put behind an ALB; CloudFront for CDN. Streamlit's session affinity requires sticky sessions (target group stickiness).

**Q25. Chainlit for a multi-turn chat — state management?**
> `cl.user_session.set/get` for per-session state (tied to thread_id). Persist to DB in `@cl.on_message` for durability. Combine with LangGraph + PostgresSaver for full persistence.

---

## 13.11 Resume tie-ins

- **"FastAPI, Flask, Django"** — own FastAPI. Flask was earlier in your career; Django probably for web UIs. JD is about FastAPI — lean in.
- **"LangChain, LangGraph, CrewAI, vLLM, Streamlit, Chainlit"** — you have the stack the JD wants. Have concrete stories:
  - LangGraph story → your order-flow / voice-driven project (from existing doc) or your ML workspace assistant.
  - vLLM story → if you haven't deployed vLLM, study hard; prepare a sketch of how you'd deploy Llama-3-70B AWQ on vLLM with Ray Serve for a hypothetical Avrioc workload.
  - Chainlit/Streamlit → your ResMed dashboards probably used Streamlit; own that.

---

Continue to **[Chapter 14 — Monitoring & Drift](14_monitoring_drift.md)**.
