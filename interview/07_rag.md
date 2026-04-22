# Chapter 07 — Retrieval-Augmented Generation (RAG)
## From naive RAG to production-grade systems

> Your ResMed project was a RAG chatbot. This chapter is you turning that bullet into a 20-minute masterclass.

---

## 7.1 Why RAG exists

LLMs have two fundamental limitations:

1. **Frozen knowledge** — training cut-off means no awareness of recent events, internal docs, or proprietary data
2. **Hallucination** — generate plausible-sounding but wrong text when asked about unfamiliar topics

RAG fixes both: **retrieve relevant documents at query time, include them in the prompt as context**, and let the LLM ground its answer in retrieved facts (with citations).

---

## 7.2 Naive RAG — the baseline

```
┌─────────────────── Indexing (Offline) ────────────────────┐
│                                                            │
│   Docs  ──▶ Chunker ──▶ Embedding ──▶ Vector Store         │
│                          Model        (pgVector / Qdrant)  │
└────────────────────────────────────────────────────────────┘

┌─────────────────── Query (Online) ────────────────────────┐
│                                                            │
│   User ──▶ Query ──▶ Embedding ──▶ ANN Search ──▶ Top-K    │
│   Query              Model          (Vector DB)    Chunks  │
│                                                      │     │
│                      ┌───────────────────────────────┘     │
│                      ▼                                      │
│               Prompt: [System + Context + Query]           │
│                      │                                      │
│                      ▼                                      │
│                     LLM ──▶ Answer                         │
└────────────────────────────────────────────────────────────┘
```

### Naive RAG limitations (why you move past it)
- **Chunking is naive** — fixed-size splits break semantic units
- **Query and docs have different shapes** — cosine similarity mismatches
- **No reranking** — top-k by cosine ≠ top-k by actual relevance
- **No query rewriting** — poorly phrased questions = poor retrieval
- **Context stuffing** — irrelevant chunks dilute the LLM's attention
- **No evaluation** — you can't improve what you can't measure

---

## 7.3 The 5 levers of a good RAG system

| Lever | What it controls | Biggest wins |
|-------|-----------------|--------------|
| **1. Chunking** | Information unit in the index | Semantic / parent-document chunking |
| **2. Embedding model** | Semantic similarity quality | Domain-specific / instruction-tuned |
| **3. Retrieval strategy** | What shows up in top-K | Hybrid search (dense + sparse) |
| **4. Reranking** | Precision of top-N → top-K | Cross-encoder rerank |
| **5. Generation** | Final output quality | Citations, grounding, evaluation |

---

## 7.4 Chunking strategies

### Fixed-size
```
chunk_size = 512 tokens, overlap = 50
```
Simple, works for homogeneous text. Breaks semantic units.

### Recursive character splitter
Tries splitting at \n\n → \n → sentence → word → char. Default in LangChain. Better than fixed but still blind to structure.

### Sentence / paragraph
Split at sentence/paragraph boundaries. Respects semantics; variable chunk size.

### Semantic chunking
Compute embedding per sentence. Group consecutive sentences whose embeddings are close. Natural topic boundaries. Slow to index but high quality.

### Document-aware
Respect structural markers (markdown headers, code blocks, tables, HTML sections). For clinical reports at ResMed, the natural boundaries are "History of Present Illness," "Medications," etc.

### Parent-document retrieval
Embed SMALL chunks (256 tokens) for precise retrieval. At answer time, pass the PARENT chunk (2000 tokens) to the LLM for more context. Huge quality win.

### Contextual retrieval (Anthropic, 2024)
Before embedding a chunk, prepend an LLM-generated **context description** of that chunk within the full doc:
```
Chunk: "Deposited $5,000 into checking account."
Context: "This is from the Q3 2024 earnings call of Acme Corp."
Embed: "[Context] ... [Chunk] ..."
```
Improves retrieval precision by 30-50% on labeled eval sets.

### Chunk size heuristics
- General text: 512 tokens, 50 overlap
- Dense / code / tables: 256 tokens with parent-doc
- Very long docs / GraphRAG: larger (2000+) with summaries

---

## 7.5 Retrieval strategies — beyond cosine

### Hybrid search (dense + sparse)
Dense (semantic) + BM25 (lexical) with fusion:
- **Reciprocal Rank Fusion (RRF):** `score = Σ 1/(k + rank_i)`, k=60. Parameter-free; robust to score scale differences.
- **Weighted sum:** `final = w·cos + (1-w)·bm25_norm`. Requires normalization; tune w.

Wins on queries with rare entities, product SKUs, code tokens — anywhere exact match matters.

### HyDE (Hypothetical Document Embeddings)
Use LLM to generate a fake answer to the query; embed **that** (not the query) for retrieval. Closes query-doc asymmetry, especially on zero-shot domains. Adds LLM latency.

### Multi-query retrieval
Generate N paraphrases of the query with an LLM; union the retrieved sets. Improves recall when the original query is poorly phrased.

### Query rewriting
Use an LLM to reformulate the query. Crucial in chat RAG for:
- Pronoun resolution ("it" → "the last order")
- Context injection ("he" → "employee name from prior turn")
- Decomposition ("Compare A and B and recommend" → {query A features, query B features})

### Sub-question decomposition
Break multi-hop questions into sub-questions; retrieve per sub-question; synthesize. Used in LlamaIndex, LangGraph.

### Self-query retrieval
LLM extracts structured filters from the natural-language query: "Show me papers by Yann LeCun since 2020" → {author: "Yann LeCun", year > 2020}. Apply as metadata filter alongside vector search.

---

## 7.6 Reranking — cross-encoder as the precision booster

### Architecture recap
- **Bi-encoder** (for retrieval): encode q, d separately. Cosine sim. Fast, lossy.
- **Cross-encoder** (for reranking): encode [q, d] together. Full attention. Slow, precise.

### Pipeline
```
Query
  │
  ▼
Bi-encoder top-100 (candidates from millions)
  │
  ▼
Cross-encoder rerank → top-5 (precision)
  │
  ▼
LLM with top-5 as context
```

### Models (2026)
- **BGE-reranker-v2-m3** (open, multilingual, ~3B params)
- **Cohere Rerank 3** (managed, top quality)
- **Jina Reranker v2**

### Cost
~100-300 ms added latency. Non-negotiable for production quality.

### Why it helps so much
Bi-encoder compresses each doc into one vector. Cross-encoder uses full attention between query and doc tokens — far more information, far more accurate.

---

## 7.7 Contextual compression

After retrieval, run each chunk through a small LLM (or LongLLMLingua) to **extract only the query-relevant sentences** or drop entire chunks.

Reduces context tokens → lower cost, better LLM attention, higher faithfulness.

LangChain `ContextualCompressionRetriever` bundles this.

---

## 7.8 Advanced RAG — the 2026 patterns

### Self-RAG (Asai et al., 2023)
Train the LLM to emit reflection tokens: `[Retrieve]`, `[Relevant]`, `[Supported]`, `[Useful]`. Model decides when to retrieve and whether to use results. Requires fine-tuning.

### Corrective RAG (CRAG)
Lightweight retrieval evaluator (T5) scores retrieved docs as Correct/Ambiguous/Incorrect. Low confidence → fall back to web search. Production-friendly — no base-model fine-tuning.

### GraphRAG (Microsoft)
Build an entity-relationship graph via LLM extraction; cluster (Leiden algorithm); summarize communities. Shines on **global questions** ("what are the main themes across this corpus") and multi-hop reasoning.

- **Pros:** SOTA on "summarize the entire corpus" queries
- **Cons:** Indexing cost is substantial (LLM over every chunk); not worth it for FAQ-style bots
- **Use case:** Legal discovery, intelligence analysis, research literature review

### Agentic RAG
Wrap retrieval as a tool the LLM can call iteratively. The LLM decides: "I need more info on X, let me search again with a refined query." Implemented with LangGraph / AgentExecutor.

### Late chunking (Jina, 2024)
Embed the full document (with long-context embedder), then pool per-chunk from the full-doc hidden states. Captures cross-chunk context in each chunk's embedding. Solves the "chunk in isolation loses context" problem.

### Contextual Retrieval (Anthropic, 2024)
LLM generates a per-chunk context description before embedding. Combined with hybrid search and reranking: 67% reduction in retrieval failures on their eval.

---

## 7.9 RAG evaluation — RAGAS + golden set

### RAGAS framework — 4 metrics

| Metric | Measures | How computed |
|--------|----------|--------------|
| **Faithfulness** | Is answer grounded in context? | LLM-judge: claims in answer present in context? |
| **Answer Relevance** | Does answer address query? | Embed answer, embed query variations, cosine |
| **Context Precision** | Are retrieved chunks relevant? | LLM ranks chunks by relevance to answer |
| **Context Recall** | Is ground-truth in retrieved context? | Needs labeled reference |

### Other tools
- **TruLens** — trace-level debugging, feedback functions
- **Arize Phoenix** — embedding drift, RAG visualizations
- **DeepEval** — comprehensive LLM testing framework
- **LangSmith / Langfuse** — trace logging + eval

### The golden set
100-300 (query, ground-truth answer, relevant doc IDs) tuples — your source of truth. Build it once, run all changes against it, never ship without re-running.

---

## 7.10 Production RAG reliability checklist

```
☐ Hybrid search (dense + BM25 + RRF)
☐ Cross-encoder reranker on top-N
☐ Chunk strategy chosen for doc type (parent-doc for dense text)
☐ Query rewriting for chat (pronoun resolution)
☐ Citations in response (source doc IDs)
☐ Confidence threshold: below X → "I don't know"
☐ Output guardrails (PII redaction, hallucination filter)
☐ Latency budget: TTFT < 1s, E2E < 3s for chat
☐ Prefix caching (system prompt) for TTFT reduction
☐ Eval on golden set (RAGAS) gated in CI
☐ Online metrics: click-through, thumbs up/down, regenerate rate
☐ Prompt versioning + audit log
☐ Data residency compliance (UAE for Avrioc)
```

---

## 7.11 Debugging "my RAG gives wrong answers"

**Decompose into stages. Fix from the bottom up.**

### Stage 1: Retrieval recall
Question: is the correct chunk even in top-K?
- Build labeled (query, correct_chunk_id) set
- Measure recall@K
- If low: fix chunking (try semantic / parent-doc), swap to better embedding, add hybrid search

### Stage 2: Rerank precision
Question: after reranking top-100 → top-5, is the correct chunk in top-5?
- Measure nDCG@5 on labeled set
- If low: upgrade reranker (BGE-reranker-v2-m3, Cohere), tune top-N cap

### Stage 3: Generation faithfulness
Question: given correct context, does the LLM produce a correct answer?
- Measure RAGAS faithfulness
- If low: tighten prompt ("Answer ONLY from the context. If not there, say so."), lower temperature, switch to stronger model

**80% of "bad RAG" is fixed by Stage 1 + Stage 2 improvements** (chunking + hybrid + reranker).

---

## 7.12 The prompt template for production RAG

```
System:
You are a helpful assistant. Answer the user's question using ONLY the
information in the CONTEXT below. If the context does not contain the
answer, say "I don't have information on that." Always cite sources as
[doc_id].

CONTEXT:
[1] <chunk 1 text>   (source: report_123.pdf, page 4)
[2] <chunk 2 text>   (source: report_456.pdf, page 2)
...

QUESTION: <user question>

Answer:
```

Key elements:
1. **System prompt** with anti-hallucination instruction
2. **Numbered context chunks** with source metadata
3. **Explicit citation requirement**
4. **Fallback instruction** ("say 'I don't have information'")

---

## 7.13 Interview Q&A — RAG

**Q1. What is RAG and why?**
> RAG = Retrieval-Augmented Generation. Retrieve relevant docs at query time, inject them into the LLM prompt as context. Gives the LLM current/proprietary knowledge without retraining, reduces hallucination, enables citations.

**Q2. Naive RAG vs advanced RAG — what's changed in 2025-2026?**
> Naive RAG = cosine search → stuff context → generate. Advanced = query rewriting, hybrid search (dense + BM25), cross-encoder reranking, parent-doc / contextual chunking, RAGAS evaluation. Every production RAG in 2026 has all of these.

**Q3. Hybrid search — why, how to fuse?**
> BM25 + dense vector. Use RRF (reciprocal rank fusion) — parameter-free, robust to score scales. Or weighted sum after normalization. Wins on queries with rare entities, product codes, code tokens.

**Q4. HyDE — what and when?**
> Hypothetical Document Embeddings. Use an LLM to generate a fake answer, embed that, retrieve. Closes the query-doc asymmetry gap. Best for zero-shot domains without domain-tuned embeddings. Adds LLM latency.

**Q5. Multi-query vs query rewriting — difference?**
> Multi-query: generate N paraphrases, union retrieved sets (improves recall). Query rewriting: single better-formed query (pronoun resolution, context injection). In chat RAG, always rewrite first.

**Q6. Cross-encoder reranking — why is it "mandatory"?**
> Bi-encoder compresses each doc into one vector (lossy). Cross-encoder does full attention between query and doc tokens (much more info). 100-300ms cost for 30-70% nDCG@10 improvement on typical sets.

**Q7. Chunking — what's your default?**
> Start with 512 tokens, 50 overlap. Use parent-document retrieval (embed 256, return 2000) for dense text. Structural-aware splits for docs with headers (markdown, reports). Always eval against your golden set.

**Q8. Contextual compression — what and when?**
> After retrieval, pass each chunk through an LLM to extract query-relevant sentences or drop irrelevant chunks. Reduces token cost and improves LLM attention. LangChain ContextualCompressionRetriever.

**Q9. GraphRAG — when is it worth it?**
> Use for "global" questions across a corpus (themes, summaries) and multi-hop reasoning. Not for single-doc QA or FAQ bots. Heavy indexing cost (LLM over every chunk). Great for enterprise knowledge bases, legal discovery.

**Q10. Self-RAG vs Corrective RAG?**
> Self-RAG: fine-tune the LLM to emit reflection tokens ([Retrieve], [Relevant], [Supported]) — requires training. CRAG: lightweight T5 evaluator scores retrieved docs, triggers web-search fallback on low confidence — no LLM fine-tune. CRAG is more production-friendly.

**Q11. How do you evaluate a RAG system?**
> RAGAS framework: Faithfulness, Answer Relevance, Context Precision, Context Recall. Pair with a golden set (100-300 labeled (query, answer, chunk_ids)). Trace tools: TruLens, Phoenix, LangSmith, Langfuse.

**Q12. [Gotcha] Users complain RAG gives wrong answers even when the info is in the corpus. Debug?**
> Decompose: (1) Is the correct chunk in top-K retrieval? Fix chunking / embedding / hybrid. (2) Is it in top-N after rerank? Upgrade reranker. (3) Given correct context, is the LLM right? Fix prompt / lower T / stronger model. 80% of bugs are at Stage 1-2.

**Q13. Chunk size tradeoffs?**
> Small (256): precise retrieval, less context per chunk. Large (1024+): more context per chunk, less precise retrieval. Parent-doc pattern gets both — embed small, return parent. For dense content (code, tables), smaller chunks win.

**Q14. How do you reduce RAG latency?**
> Prefix caching (system prompt) for TTFT. Parallel retrieval + rerank. Smaller embedding / rerank models. Reduce top-K where acceptable. Stream the response. Pre-compute common queries (semantic cache).

**Q15. How do you handle "the answer is multi-document / multi-hop"?**
> (1) Sub-question decomposition: LLM breaks query into sub-queries; retrieve per sub-query. (2) Agentic RAG: LLM iteratively retrieves, reasons, re-retrieves. (3) GraphRAG for corpus-wide themes.

**Q16. [Gotcha] High cosine similarity on retrieved docs, but LLM's answers are wrong. Likely cause?**
> Symmetric model used for asymmetric task. Cosine = similarity, not query-answer relevance. Swap to E5/BGE (asymmetric, query/passage prefixes) and add a cross-encoder reranker.

**Q17. Contextual retrieval (Anthropic) — what does it add?**
> Before embedding each chunk, prepend an LLM-generated context description of that chunk within the full doc. ~30-50% retrieval precision lift on labeled sets. Indexing cost: extra LLM call per chunk.

**Q18. Prompt engineering for RAG — key elements?**
> Anti-hallucination instruction ("use ONLY the context"). Numbered chunks with source metadata. Explicit citation requirement. Fallback ("if not in context, say 'I don't know'"). Low temperature.

**Q19. How do you handle RAG across languages (e.g., English + Arabic for UAE)?**
> Multilingual embedding model (BGE-M3, Cohere Embed v3 multilingual). Consider per-language indices if corpora are clearly separated. Query translation optional. Tokenizer efficiency matters — Arabic can be 3-5× more tokens without a tuned tokenizer.

**Q20. Citations — how do you make sure the LLM cites correctly?**
> (1) Include source IDs in the context. (2) Instruct the LLM to cite as [id]. (3) Post-process: parse citations and verify they exist in your context. (4) Use a stronger model (citation accuracy correlates with model size). (5) For structured output, use JSON schema with citations as a list.

**Q21. [Gotcha] Your users now ask questions that didn't exist when you set up RAG. How do you adapt?**
> Monitor query distribution drift. Log failure cases (thumbs down, regenerate). Periodic re-indexing with new docs. Online evaluation — sample 1% of queries into a manual review queue. Retrain the query-rewriting step if patterns shift.

**Q22. Semantic cache — when does it help?**
> High-duplicate query workload (FAQ bot, customer support). Embed query, search a cache index, return cached answer if similarity ≥ threshold. Watch: cache poisoning if queries look similar but context changed (new docs).

**Q23. Long-context LLMs vs RAG — is RAG obsolete?**
> No. Long context (1M tokens) is expensive, slow, and has attention dilution (needle-in-haystack fails). RAG is still more efficient (pay for retrieved K tokens, not the whole corpus), auditable (citations), and updatable (reindex beats retrain). Hybrid: RAG for grounding + long context for the retrieved chunk set.

**Q24. Security — how do you prevent prompt injection in RAG?**
> Retrieved docs can contain adversarial instructions. Defenses: (1) treat retrieved content as data via separation tags (<document>...</document>), (2) input/output guardrails, (3) strong system prompt that ignores instructions from docs, (4) sandboxed tool-calling (if the LLM executes code), (5) rate-limit and monitor per-user.

**Q25. RAG-triad visualization?**
> Faithfulness, Answer Relevance, Context Precision — score each on 0-1 scale, plot as a triangle. One low score diagnoses where RAG is failing. TruLens ships this view.

---

## 7.14 Resume tie-ins — framing your ResMed story

**"GenAI-powered query routing system using LLMs and RAG"** — expand to:

> At ResMed, I built a RAG-based clinical chatbot answering questions over unstructured medical reports. The architecture had three layers:
>
> 1. **Indexing** — clinical reports were pre-processed with a section-aware chunker (respecting headers like "History of Present Illness"), embedded with [your choice of embedding model], stored in pgVector.
>
> 2. **Query routing** — an LLM classifier routed incoming queries into three lanes: factual queries went to RAG, analytical queries went to a code-generation path that wrote Python/SQL, and conversational queries went direct.
>
> 3. **RAG pipeline** — hybrid search (dense + BM25), cross-encoder rerank, parent-doc retrieval (embedded small, returned large), citations in responses, RAGAS evaluation on a golden set we built with the clinical team.
>
> The key insight was that "one-size-fits-all" RAG failed for analytical queries ("what's the average age of diabetic patients in this cohort?") — those need code, not retrieval. Splitting the problem cut hallucination and improved analyst time-to-answer significantly.

---

Continue to **[Chapter 08 — Vector Databases](08_vector_databases.md)**.
