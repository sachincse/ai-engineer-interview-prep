# Chapter 07 — Retrieval-Augmented Generation (RAG)
## A whiteboard masterclass — naive RAG, advanced RAG, GraphRAG, Self-RAG, Corrective RAG, agentic RAG, and the RAGAS evaluation playbook

> Sachin spent 2.5 years at ResMed building a clinical RAG chatbot. This chapter is what you'd say if a senior engineer at Avrioc handed you a marker and said "draw it on the board, walk me through it." Every section is a thing you should be able to explain verbally for 3–5 minutes.

---

## 7.1 Why RAG exists — the two problems it solves

LLMs have two fundamental limitations that no amount of clever prompting fixes.

**Problem 1: Frozen knowledge.** A pretrained model is a snapshot of the internet at a specific point in time. It doesn't know what happened yesterday. It doesn't know your company's internal documents. It doesn't know the new clinical guideline that was published last week. Retraining the model every week is economically impossible.

**Problem 2: Hallucination.** When asked about something it doesn't know, an LLM doesn't say "I don't know." It generates plausible-sounding but wrong text. The model is trained to *complete* text, not to be truthful. For most consumer chat applications this is annoying. For a medical chatbot or legal assistant, it's a liability.

RAG fixes both with a simple idea: at query time, retrieve the most relevant documents from a corpus, paste them into the prompt as context, and let the LLM ground its answer in those documents (with citations).

The model doesn't need to "know" anything beyond what's in its window — you give it fresh, relevant facts every time. And because the answer is grounded in retrieved text, you can audit it, cite it, and correct it by updating the corpus.

> **How to say this in an interview:** "RAG addresses two LLM limitations. First, frozen knowledge — the model only knows what was in pretraining, so it can't speak to your internal documents or recent events. Second, hallucination — when asked about unfamiliar topics, models generate plausible but wrong text. RAG retrieves relevant documents at query time and injects them into the prompt as context, so the model grounds its answer in fresh, cited facts. The model becomes a reasoning engine over your data, not a knowledge store."

---

## 7.2 Naive RAG — the architecture you start with

Naive RAG has two phases: an offline indexing pipeline and an online query pipeline. Almost every RAG system starts here, then incrementally adds the advanced techniques in section 7.4.

### Indexing pipeline (offline, runs when corpus changes)

```
   ┌──────────────┐
   │   Documents  │  PDFs, Confluence pages, clinical reports, etc.
   │   (raw)      │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Loader /   │  Extract text from PDF, HTML, DOCX
   │   Parser     │  Preserve metadata (source, page, section)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Chunker    │  Split into 256–1024 token chunks
   │              │  Often with overlap (50–100 tokens)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Embedding  │  Each chunk → vector ∈ ℝ^d
   │   Model      │  (e.g., bge-large, OpenAI text-embedding-3, Cohere v3)
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Vector DB  │  Store (chunk_text, embedding, metadata)
   │              │  Index with HNSW / IVF
   └──────────────┘
```

### Query pipeline (online, runs per user query)

```
   ┌──────────────┐
   │  User Query  │  "What was patient 12's CHF history?"
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Embedding  │  Same model used for indexing!
   │   Model      │
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Vector Search│  ANN search → top-K chunks (typically K=5–10)
   │  (top-K)     │  Returns chunks with cosine similarity
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │   Prompt     │  System: "Answer using only context"
   │   Assembly   │  Context: [chunk_1, chunk_2, ...]
   │              │  User: query
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │     LLM      │  Generates answer with citations
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │  Final Answer│  "Patient 12 was diagnosed with CHF in 2018... [doc_42]"
   └──────────────┘
```

### Worked example — the clinical chatbot at ResMed

User asks: *"What was patient 12's CHF history?"*

1. **Embed the query.** Using bge-large-en-v1.5 we get a 1024-dim vector.
2. **ANN search in pgVector.** Top-5 chunks come back, each with cosine similarity. Let's say the top match is from `report_p12_2024.pdf`, page 4, with similarity 0.84.
3. **Assemble the prompt.**
   ```
   System: You are a clinical assistant. Answer using ONLY the context.
           Cite sources as [doc_id]. If not in context, say so.
   
   Context:
   [doc_42] Patient 12, 67-year-old male, was diagnosed with CHF in March 2018.
            Initial EF was 35%. Started on lisinopril and carvedilol...
   [doc_43] At 6-month follow-up, EF improved to 42%. Patient reports...
   [doc_44] Most recent visit (Q3 2024): NYHA class II, stable on current regimen.
   
   Question: What was patient 12's CHF history?
   ```
4. **LLM generates.**
   ```
   Patient 12 (67-year-old male) was diagnosed with congestive heart failure
   in March 2018 with an initial ejection fraction of 35%. He was started on
   lisinopril and carvedilol [doc_42]. At 6-month follow-up, his EF improved
   to 42% [doc_43]. As of his most recent visit in Q3 2024, he remains
   stable on his current regimen at NYHA class II [doc_44].
   ```

This is naive RAG working correctly. Now consider what could go wrong.

### Naive RAG limitations (the reasons advanced RAG exists)

1. **Chunking is structure-blind.** A fixed 512-token split might cut a paragraph in half, or worse — separate a finding from its context.
2. **Query and document live in different shapes.** A short question and a paragraph of detail have different embedding profiles. Cosine similarity can mismatch.
3. **No reranking.** Top-5 by cosine isn't necessarily top-5 by *true relevance*. Cosine is a lossy proxy.
4. **No query rewriting.** Pronouns, vague queries, and multi-turn context don't survive embedding.
5. **Context dilution.** Stuffing 10 mediocre chunks into the prompt degrades the LLM's attention more than helps.
6. **No evaluation loop.** You can't improve what you can't measure.

Each of these has a fix in advanced RAG.

> **How to say this in an interview:** "Naive RAG has two phases. Offline: load documents, chunk into 256-to-1024 token pieces, embed each chunk, store in a vector database with metadata. Online: embed the query with the same model, do ANN search for top-K nearest chunks, assemble a prompt with the chunks as context, generate with low temperature and citation instructions. It's a great starting point but has well-known failure modes — naive chunking, query-document asymmetry, no reranking, no evaluation. Production RAG addresses each of these."

---

## 7.3 The 5 levers of a good RAG system

When debugging or improving a RAG system, every change you make falls into one of five categories. Knowing which lever you're pulling helps you debug systematically.

| Lever | What it controls | Highest-impact upgrade |
|-------|------------------|------------------------|
| **1. Chunking** | The information unit in the index | Parent-document or contextual chunking |
| **2. Embedding model** | Semantic similarity quality | Domain-tuned or asymmetric model (E5/BGE) |
| **3. Retrieval strategy** | What ends up in top-K | Hybrid search (dense + BM25 + RRF) |
| **4. Reranking** | Precision of top-N → top-K | Cross-encoder reranker |
| **5. Generation** | Final answer quality | Citation prompting + low temperature |

**The 80/20 rule:** In most underperforming RAG systems, 80% of the gain comes from fixing levers 1, 3, and 4 (chunking, hybrid search, reranking). Lever 2 (better embedding model) helps but has diminishing returns. Lever 5 (generation tuning) only matters once the retrieval is right.

---

## 7.4 Chunking strategies — the most underrated lever

Chunking determines what unit of text gets embedded and retrieved. Get this wrong and no embedding model on earth saves you.

### Fixed-size chunking

```
chunk_size = 512 tokens
overlap = 50 tokens
```

Simple. Fast to index. But blind to structure — splits paragraphs mid-sentence, separates a heading from its content. Use as a baseline.

### Recursive character splitter

Tries splitting at \n\n first, then \n, then sentence boundaries, then words, then characters. LangChain default. Better than fixed for natural text but still structure-blind.

### Sentence / paragraph chunking

Split at sentence or paragraph boundaries. Variable chunk size, but respects semantic units. Works well for narrative text (articles, blog posts). Less well for tabular or structured docs.

### Semantic chunking

Embed each sentence individually. Group consecutive sentences whose embeddings are close. Cut at semantic discontinuities. Slow to index but produces high-quality semantic chunks.

```
   Sentence 1 ──┐
   Sentence 2 ──┤  embeddings close → group together
   Sentence 3 ──┘
   ─── cut here (embedding distance jumps) ───
   Sentence 4 ──┐
   Sentence 5 ──┘  embeddings close → new group
```

### Document-aware (structure-aware) chunking

Respect structural markers in the document. For markdown: split on `#`, `##`, `###`. For clinical reports: split on canonical sections like *History of Present Illness*, *Medications*, *Assessment and Plan*. For code: split on function boundaries. The chunker uses domain knowledge.

This was Sachin's approach at ResMed — chunk on clinical section headers because that's what humans use to find information.

### Parent-document retrieval (game-changer)

The trick: embed *small* chunks for precise retrieval, but at answer time return the *parent* (larger) chunk for richer context.

```
   Index time:                  Query time:
   ┌──────────────┐             ┌──────────────┐
   │ Parent chunk │             │  Find best   │
   │ (2000 tok)   │             │  small chunk │
   └──┬───────────┘             └──────┬───────┘
      │ split into                      │
      ▼                                  │ map to parent
   ┌──────────┬──────────┬──────────┐   │
   │ small 1  │ small 2  │ small 3  │   │
   └──────────┘ embed each ┘────────┘   ▼
                                  ┌──────────────┐
   Embed small (256 tok) → store  │ Return parent│
                                  │ (2000 tok)   │
                                  └──────────────┘
```

Why it works: small chunks are semantically tight, so retrieval is precise. But the LLM benefits from the larger surrounding context to actually reason. You get both.

### Contextual retrieval (Anthropic, 2024) — the best chunking trick

Before embedding a chunk, prepend an LLM-generated *context description* that situates the chunk within the larger document.

```
Original chunk: "Deposited $5,000 into checking account."

Context-enhanced chunk:
"This is from Acme Corp's Q3 2024 earnings call, in the section
discussing operating cash flow trends. [chunk] Deposited $5,000
into checking account."

Embed the context-enhanced version. Store it.
```

Why it works: chunks lose context when extracted from their parent document. A chunk saying "deposited $5,000" is ambiguous in isolation. The LLM-generated context disambiguates it before embedding. Anthropic's paper shows ~30–50% reduction in retrieval failures.

Cost: one LLM call per chunk at indexing time. For 100K chunks, that's expensive but a one-time cost. With prompt caching (system + full document cached), the cost is dominated by the per-chunk variable portion.

### Chunk size heuristics

- **General prose:** 512 tokens, 50 overlap
- **Code or dense technical content:** 256 tokens, 30 overlap, parent-doc pattern
- **Tables or structured data:** keep table whole, embed table summary, return table at answer time
- **Long documents with sections:** structure-aware, then 256–512 within section
- **GraphRAG-style:** larger chunks (2000+) plus LLM-generated summaries

### Common chunking mistakes

1. **Fixed 512 tokens with no overlap.** Split a sentence in half, lose context, retrieval fails.
2. **Tables sliced into rows.** Tables only make sense as whole units; embed the table summary or full table.
3. **Code split mid-function.** Use language-aware splitting (LangChain has a CodeSplitter).
4. **Same chunk size across heterogeneous corpus.** A research paper and a tweet need different sizes. Profile your corpus.

> **How to say this in an interview:** "Chunking is the most underrated lever. I start with structure-aware splitting — markdown headers, clinical sections, code function boundaries. For dense content I use parent-document retrieval: embed 256-token chunks for precision, return the 2000-token parent at answer time. The biggest recent advance is Anthropic's contextual retrieval — prepend an LLM-generated context description before embedding each chunk. It costs an LLM call per chunk at index time but cuts retrieval failures 30 to 50 percent."

---

## 7.5 Retrieval strategies — beyond plain cosine

### Dense retrieval (the baseline)

Embed query, do ANN search by cosine similarity. Works well for paraphrased queries. Fails on rare entities, product SKUs, code tokens, exact phrase matches.

### BM25 (sparse retrieval)

Classic lexical matching with TF-IDF and document length normalization. Excellent at exact-term matching. Misses semantically related but lexically different queries.

### Hybrid search — the production default

Combine dense and sparse. Each retrieves top-K. Fuse the rankings.

```
   Query
     │
     ├──▶ Dense embedding ──▶ HNSW ──▶ top-50 (semantic matches)
     │                                          ╲
     │                                           ╲
     ├──▶ BM25 ──────────────────────▶ top-50    ──▶ RRF Fusion ──▶ top-10
     │                            (lexical matches)  ╱
     │                                              ╱
```

### Reciprocal Rank Fusion (RRF)

The simplest and most robust fusion method. For each document d, compute:

```
RRF_score(d) = Σ_i 1 / (k + rank_i(d))
```

Where rank_i(d) is the rank of document d in retriever i's results, and k is a constant (typically 60). The 1/k prevents top-1 from dominating.

Worked example: a document is rank 1 in dense and rank 3 in BM25, k=60.
- 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

Another document is rank 5 in dense, not in BM25's top-10:
- 1/(60+5) + 0 = 0.0154

RRF works because:
- It's parameter-free (k=60 works universally)
- It doesn't require score calibration between retrievers (different scales work)
- It's robust — a document strong in either retriever bubbles up

### Weighted score fusion

Alternative to RRF:
```
final_score = w * cosine_normalized + (1-w) * bm25_normalized
```

Requires normalization (min-max or z-score) and tuning w. Use only if RRF underperforms; in practice RRF is fine.

### HyDE (Hypothetical Document Embeddings)

Use an LLM to *generate a fake answer* to the query, then embed that fake answer for retrieval (instead of the query).

```
Query: "How does CRISPR work?"
   │
   ▼
LLM generates fake answer: "CRISPR is a gene-editing technique that uses
the Cas9 protein to cut DNA at specific sites guided by a small RNA molecule..."
   │
   ▼
Embed the fake answer (looks like a real document)
   │
   ▼
Search vector DB with this embedding
```

Why it works: queries and documents have different shapes. A query is short and interrogative. A document chunk is longer and declarative. HyDE bridges this asymmetry by transforming the query into something document-shaped before embedding.

When to use: zero-shot domains where you don't have a domain-tuned embedding model. Adds latency (one LLM call before retrieval).

### Multi-query retrieval

Use an LLM to generate N paraphrases of the query, retrieve top-K for each, union the results.

```
Original query: "How is the stock market doing?"
   │
   ▼
LLM generates paraphrases:
  - "What's happening with the stock market?"
  - "Recent stock market performance?"
  - "Stock market trends this week?"
   │
   ▼
Retrieve top-K for each, union
```

Improves recall when the original query is poorly phrased. Costs an extra LLM call.

### Query rewriting (essential for chat RAG)

In a chat session, the user's nth message often references prior turns:
- "What about his salary?" (he = the candidate from turn 3)
- "Show me the second one" (the second = an item in a list from earlier)

Embedding "What about his salary?" gives garbage retrieval. The fix: an LLM rewrites the query in a self-contained form using chat history:

```
History: "Tell me about candidate John Smith." → assistant response
User now asks: "What about his salary?"

LLM rewrites: "What is John Smith's salary?"
Now embed and retrieve with the rewritten query.
```

This is non-negotiable for chat RAG. Without it, multi-turn quality collapses.

### Sub-question decomposition

For multi-hop questions, decompose into sub-questions, retrieve per sub-question, synthesize the answer.

```
Query: "Compare Apple's and Google's Q3 revenue and recommend which to invest in."
   │
   ▼
Decompose:
  - "What was Apple's Q3 revenue?"
  - "What was Google's Q3 revenue?"
  - "What are analyst recommendations for Apple?"
  - "What are analyst recommendations for Google?"
   │
   ▼
Retrieve per sub-question, then synthesize.
```

Used in LlamaIndex's SubQuestionQueryEngine, LangGraph state machines.

### Self-query retrieval

LLM extracts structured filters from natural-language queries:

```
Query: "Show me papers by Yann LeCun published since 2020"
   │
   ▼
LLM extracts: {author: "Yann LeCun", year > 2020}
   │
   ▼
Apply as metadata filter alongside vector search.
```

Powerful for queries that mix semantic and structural intent.

> **How to say this in an interview:** "I always start with hybrid search — dense plus BM25 fused with RRF. It catches both semantic and lexical matches. RRF is parameter-free and robust to different score scales. For chat RAG I always add LLM-based query rewriting that resolves pronouns and prior context. For zero-shot domains without a tuned embedding, HyDE bridges query-document asymmetry. Multi-hop questions get sub-question decomposition. Self-query handles queries with structural filters like dates or authors."

---

## 7.6 Reranking — the precision booster you can't skip

### Why bi-encoder retrieval is fundamentally lossy

The embedding model used for retrieval is a *bi-encoder*: it encodes query and document independently into a single vector each, and similarity is cosine of those vectors.

This is fast — you precompute document embeddings once and reuse forever. But each document is squashed into one fixed-size vector regardless of its contents. A 2000-token chunk with 5 distinct topics gets averaged into one vector. Information is lost.

### Cross-encoder rerankers — the precision step

A *cross-encoder* takes the query and document *together* through a transformer with full attention. Output is a single relevance score.

```
   Bi-encoder (retrieval):
   ┌─────────┐   ┌─────────┐
   │  Query  │   │   Doc   │
   │ encoder │   │ encoder │
   └────┬────┘   └────┬────┘
        │             │
        ▼             ▼
      vec_q         vec_d
        │             │
        └──── cosine ─┘
   FAST. Lossy.

   Cross-encoder (reranking):
   ┌─────────────────────────┐
   │  [CLS] Query [SEP] Doc  │
   │       full attention    │
   │   between Q and D       │
   └──────────┬──────────────┘
              │
              ▼
         Single score
   SLOW. Precise.
```

Because the cross-encoder sees both query and document tokens with full attention, it can compute fine-grained relevance — does the doc actually answer the query, or just match topically?

### The two-stage pipeline

```
   Query
     │
     ▼
   Bi-encoder retrieval ────▶ top-100 candidates  (fast, recall-focused)
                                       │
                                       ▼
                            Cross-encoder rerank
                                       │
                                       ▼
                            top-5 (precise, used in prompt)
                                       │
                                       ▼
                                  LLM with context
```

Bi-encoder gets you recall on the cheap. Cross-encoder gives you precision on a small candidate set.

### Worked example — when reranking saves you

Query: *"What is the dosing protocol for amoxicillin in pediatric patients with otitis media?"*

Bi-encoder top-5 (cosine):
1. "Amoxicillin pharmacokinetics in adults..." (sim 0.81)
2. "Pediatric infections — antibiotic stewardship..." (sim 0.79)
3. "Otitis media diagnostic criteria in children..." (sim 0.78)
4. "Amoxicillin dosing for ear infections in kids: 80 mg/kg/day..." (sim 0.77)
5. "Streptococcus pneumoniae resistance patterns..." (sim 0.75)

The fourth result is the *correct* one but ranks fourth by cosine — because shorter chunks with the exact query terms can score lower than longer chunks with vague topical overlap.

Cross-encoder rerank (full attention sees query + doc together):
1. Doc 4 (true relevance score 9.2) ← correct one
2. Doc 3 (5.1)
3. Doc 2 (4.0)
4. Doc 1 (3.2)
5. Doc 5 (2.0)

Top-1 after rerank is now correct. The LLM gets the right context.

### Production reranker models (2026)

- **BGE-reranker-v2-m3** — open, multilingual, ~600M params, runs on a GPU
- **Cohere Rerank 3** — managed, top quality, $1 per 1K queries
- **Jina Reranker v2** — open, multilingual, fast
- **Voyage rerank** — managed, strong on long contexts

### Cost

Cross-encoders add 100–300ms latency on a single GPU for top-100 reranking. In a 2-second budget, that's significant but worth it. For latency-sensitive production, you can:
- Reduce top-N (rerank top-30 instead of top-100)
- Use a smaller reranker model
- Cache rerank results for repeat queries

### Common reranking mistakes

1. **Skipping it entirely.** This is the #1 RAG quality fix. It's worth 100ms.
2. **Reranking too many candidates.** Top-1000 → 5 is wasteful and slow. Top-50 → 5 is enough.
3. **Reranking without retrieval recall.** If the right chunk isn't in top-50, no reranker can save you. Fix retrieval first.

> **How to say this in an interview:** "Bi-encoder retrieval squashes each document into one vector regardless of content — fast but lossy. A cross-encoder reranker passes query and document together through a transformer with full attention, computing a fine-grained relevance score. The two-stage pipeline is bi-encoder for top-100 candidates, then cross-encoder rerank to top-5. Costs 100 to 300 milliseconds, gives 30 to 70 percent nDCG improvement on typical sets. It's non-negotiable for production. I usually use BGE-reranker-v2-m3 for open-source, Cohere Rerank for managed."

---

## 7.7 Contextual compression — fitting more relevance into the context window

After retrieval and reranking, you have top-5 chunks. But each chunk might be 2000 tokens with only 200 tokens that are actually relevant to the query. Contextual compression filters chunks down to the relevant sentences before sending to the LLM.

```
   Top-5 chunks (10,000 tokens total)
              │
              ▼
   ┌──────────────────────┐
   │ Compression model    │  Small LLM or LongLLMLingua
   │ (per chunk)          │  Extracts relevant sentences
   └──────────┬───────────┘
              │
              ▼
   Compressed context (3,000 tokens)
              │
              ▼
   LLM with compressed context
```

Benefits: lower token cost, better LLM attention (less dilution), higher faithfulness.
Cost: extra LLM call per chunk (or LongLLMLingua's smaller, faster model).

LangChain wraps this in `ContextualCompressionRetriever`.

---

## 7.8 Advanced RAG patterns — Self-RAG, CRAG, GraphRAG, agentic RAG

### Self-RAG (Asai et al., 2023)

Train the LLM to emit *reflection tokens* during generation:
- `[Retrieve]` — model decides it needs to retrieve more
- `[Relevant]` / `[Irrelevant]` — judges retrieved chunks
- `[Supported]` / `[Not Supported]` — judges whether its answer is grounded
- `[Useful]` / `[Not Useful]` — judges final answer quality

The model is fine-tuned to use these tokens, allowing it to dynamically decide when to retrieve and whether to use what it retrieves.

```
User: How does mRNA vaccine work?
Model: [Retrieve] → searches → [Relevant] for chunk_42, [Irrelevant] for chunk_43
        Using chunk_42, the model generates answer
        [Supported] [Useful]
```

**Pros:** model knows when retrieval helps vs hurts
**Cons:** requires fine-tuning the base LLM with reflection-token training data

### Corrective RAG (CRAG)

Lightweight production-friendly alternative. A small T5-based evaluator scores retrieved chunks as Correct, Ambiguous, or Incorrect. Low confidence triggers a fallback (web search, query rewrite, or human escalation).

```
   Query → Retrieve → Evaluator (T5)
                          │
              ┌───────────┼────────────┐
           Correct    Ambiguous     Incorrect
              │           │              │
              ▼           ▼              ▼
        Use as is    Rewrite query    Fallback to
                     and retry        web search
                          │              │
                          ▼              ▼
                       LLM answer    LLM answer
```

**Pros:** no LLM fine-tuning required, deployable on top of any RAG stack
**Cons:** the evaluator is another model to maintain

### GraphRAG (Microsoft, 2024)

For "global" questions across a corpus — "what are the main themes in these 10,000 documents" or multi-hop queries — vector retrieval struggles because no single chunk has the answer.

GraphRAG builds a knowledge graph from the corpus:

```
   Indexing:
   1. Chunk corpus
   2. LLM extracts (entity, relation, entity) triples per chunk
   3. Build graph from all triples
   4. Cluster the graph using Leiden algorithm
   5. LLM summarizes each cluster into a "community report"
   6. Index both raw chunks AND community reports

   Query time:
   - "Local" questions: route to vector retrieval over chunks
   - "Global" questions: route to community reports for corpus-wide reasoning
```

**Pros:** SOTA on global questions and multi-hop reasoning
**Cons:** indexing is expensive (LLM call per chunk for entity extraction); not worth it for simple QA

**When to use:** legal discovery (find connections across thousands of documents), intelligence analysis, research literature review, enterprise knowledge bases.

**When not:** FAQ chatbots, single-document QA, anything where vector RAG already works.

### Agentic RAG

Wrap retrieval as a tool the LLM can call iteratively. The LLM decides:
- "I need more information on X. Let me search again with a refined query."
- "The first retrieval was off-topic. Let me try a different phrasing."
- "I have enough. Let me synthesize the answer."

```
   ┌────────────┐
   │   Agent    │ ◄────┐
   │   (LLM)    │      │
   └─────┬──────┘      │
         │             │
   plan? retrieve?     │
         │             │
   ┌─────▼──────┐      │
   │ Tool call: │      │
   │  search    │      │
   └─────┬──────┘      │
         │             │
   ┌─────▼──────┐      │
   │  Results   │ ─────┘
   └────────────┘
```

Implemented with LangGraph state machines or AgentExecutor. Useful for complex queries where the right retrieval strategy isn't obvious upfront.

**When to use:** research assistants, deep-dive analysis, multi-step queries
**When to skip:** simple FAQ, latency-sensitive (each tool call adds 500ms+)

### Late chunking (Jina, 2024)

Embed the *full document* with a long-context embedder, then pool per-chunk from the full-document hidden states. Each chunk's embedding now captures cross-chunk context.

Solves the "chunk in isolation loses context" problem similarly to contextual retrieval, but at embedding time rather than indexing time.

> **How to say this in an interview:** "Beyond naive RAG, the patterns I reach for are CRAG for production reliability — a small evaluator that triggers fallback on bad retrieval, no LLM fine-tuning required. GraphRAG for global corpus questions where no single chunk has the answer — expensive to index but unmatched on multi-hop. Self-RAG for cases where I can fine-tune the model. Agentic RAG for complex multi-step queries where the right retrieval strategy isn't obvious upfront. The choice depends on use case — FAQ bot uses naive RAG, legal discovery uses GraphRAG, research assistant uses agentic."

---

## 7.9 RAG evaluation — RAGAS and the golden set

### Why this matters more than your model choice

The single biggest predictor of RAG success in production is whether you have a *golden eval set* and a *RAGAS-style metric pipeline*. Without these, you're flying blind. With them, you can rationally A/B test chunking, embedding, retrieval, reranking, and prompt changes.

### The golden set

100–300 (query, ground-truth answer, relevant_chunk_ids) tuples. Built once with human (ideally domain-expert) labeling. This is your source of truth.

```
Golden example:
{
  "query": "What was patient 12's CHF history?",
  "ground_truth": "Diagnosed with CHF in March 2018, EF 35%, started on lisinopril and carvedilol.",
  "relevant_chunks": ["doc_42_p4_chunk_3", "doc_43_p1_chunk_1"],
  "category": "patient_history"
}
```

Update the golden set periodically as queries drift. Run all RAG changes against it in CI before shipping.

### RAGAS framework — 4 metrics

RAGAS uses LLM-as-judge to compute these metrics automatically.

#### Faithfulness — is the answer grounded in the context?

For each claim in the answer, ask: "Is this claim supported by the context?" Faithfulness = (supported claims) / (total claims).

```
Answer: "Patient 12 was diagnosed with CHF in 2018 with EF 35%."
Claims:
  1. Patient 12 was diagnosed with CHF in 2018  → supported by context? YES
  2. EF was 35%                                 → supported by context? YES
Faithfulness = 2/2 = 1.0
```

If the model hallucinates a claim not in the context, faithfulness drops.

#### Answer Relevance — does the answer address the query?

Embed the answer, generate variations of the query that the answer "could be answering," embed those, compute average cosine to the original query.

```
Answer: "Patient 12 was diagnosed in 2018."
Possible queries: ["When was patient 12 diagnosed?", "Patient 12 diagnosis date"]
Compare to original: "What was patient 12's CHF history?"
Cosine similarity → Answer Relevance score
```

#### Context Precision — are the retrieved chunks relevant to the answer?

For each retrieved chunk in rank order, judge if it contributed to the ground-truth answer. Compute mean reciprocal precision across ranks.

```
Retrieved (in rank order):
  1. doc_42_p4_chunk_3   → relevant? YES
  2. doc_50_p1_chunk_1   → relevant? NO
  3. doc_43_p1_chunk_1   → relevant? YES
Context precision = mean(1/1, 0/2, 2/3) = 0.55
```

Low precision = retrieving noise that dilutes the LLM's attention.

#### Context Recall — does the retrieved context contain the ground-truth?

For each fact in the ground-truth answer, check if it can be supported by retrieved context.

```
Ground-truth facts:
  1. Diagnosed in 2018      → in retrieved context? YES
  2. EF 35%                  → in retrieved context? YES
  3. On lisinopril           → in retrieved context? NO (chunk wasn't retrieved!)
Context recall = 2/3 = 0.67
```

Low recall = retrieval missed key chunks. Fix chunking, embedding, or hybrid search.

### The RAG triad (TruLens visualization)

Plot Faithfulness, Answer Relevance, Context Precision on a triangle. One low score diagnoses where the system is failing:
- Low faithfulness → tighten prompt, lower temperature, stronger model
- Low answer relevance → query rewriting, prompt issues
- Low context precision → reranker, better embedding model
- Low context recall → fix chunking, hybrid search, increase top-K

### Evaluation pipeline

```
   ┌────────────────┐
   │  Golden Set    │  300 (query, answer, chunk_ids)
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ Run RAG system │  Generate answer + retrieve chunks
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │   RAGAS LLM    │  Use Claude as judge
   │   Evaluator    │
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │ Metric Report  │  Faithfulness=0.86, AnswerRel=0.79, ...
   └────────────────┘
```

### Other evaluation tools

- **TruLens** — trace-level debugging, feedback functions, RAG triad visualization
- **Arize Phoenix** — embedding drift detection, retrieval visualizations
- **DeepEval** — comprehensive LLM testing framework (RAGAS + more)
- **LangSmith / Langfuse** — trace logging plus eval
- **Ragas** — the open-source library implementing the metrics

### LLM-as-judge with Claude

For custom dimensions (factuality on medical claims, tone, refusal correctness), use Claude as a judge with a structured prompt. Validated against human raters on a 100-example calibration set first.

> **How to say this in an interview:** "RAG eval is the single biggest predictor of RAG quality in production. I always build a golden set of 100-to-300 query-answer-chunk tuples with domain-expert labels. Then I use RAGAS — Faithfulness measures if the answer is grounded in the context, Answer Relevance measures if it addresses the query, Context Precision and Recall measure retrieval quality. The RAG triad visualizes Faithfulness, Answer Relevance, and Context Precision on a triangle — one low score diagnoses where the system is failing. I run RAGAS against the golden set in CI gating every change."

---

## 7.10 Production RAG checklist

```
☐ Hybrid search (dense + BM25 + RRF)
☐ Cross-encoder reranker on top-N
☐ Chunk strategy chosen for doc type (parent-doc for dense text)
☐ Query rewriting for chat (pronoun resolution, history injection)
☐ Citations in response (source doc IDs in output)
☐ Confidence threshold: below X → "I don't know"
☐ Output guardrails (PII redaction, hallucination filter)
☐ Latency budget: TTFT < 1s, E2E < 3s for chat
☐ Prefix caching (system prompt + few-shots) for TTFT reduction
☐ Eval on golden set (RAGAS) gated in CI before deploy
☐ Online metrics: thumbs up/down, regenerate rate, click-through
☐ Prompt versioning + audit log
☐ Data residency compliance (UAE for Avrioc)
☐ Prompt injection defenses (treat retrieved docs as data, not instructions)
☐ Drift monitoring (query distribution, embedding norm distribution)
```

---

## 7.11 Debugging "my RAG gives wrong answers" — the diagnostic ladder

Most RAG bugs trace to one of three stages. Diagnose bottom-up.

### Stage 1: Retrieval recall

Question: *Is the correct chunk even in top-K?*

```
- Build a labeled (query, correct_chunk_id) set from your golden set
- Measure recall@K: did we retrieve the right chunk in top-K?
- If recall is low (<0.7): fix chunking, swap embedding model,
  add hybrid search, increase top-K
```

### Stage 2: Rerank precision

Question: *After reranking top-100 → top-5, is the correct chunk in top-5?*

```
- Measure nDCG@5 on labeled set (or precision@5)
- If precision is low: upgrade reranker, tune top-N candidate cap,
  add contextual compression
```

### Stage 3: Generation faithfulness

Question: *Given correct context, does the LLM produce a correct answer?*

```
- Measure RAGAS faithfulness (or just human-eval on 50 examples)
- If low: tighten prompt ("Answer ONLY from context. If not there, say so."),
  lower temperature, switch to a stronger model, add few-shot examples
```

**80% of "bad RAG" is fixed by Stage 1 + Stage 2 improvements.** If users complain about hallucinations and you immediately reach for prompt tweaks, you're probably misdiagnosing.

---

## 7.12 The production RAG prompt template

```
System:
You are a helpful assistant. Answer the user's question using ONLY the
information in the CONTEXT below. If the context does not contain the
answer, respond with "I don't have information on that in my knowledge
base." Always cite sources as [doc_id]. Do not use prior knowledge.

CONTEXT:
[doc_42] (source: report_p12_2024.pdf, page 4)
   Patient 12, 67-year-old male, was diagnosed with CHF in March 2018...

[doc_43] (source: report_p12_2024.pdf, page 5)
   At 6-month follow-up, EF improved to 42%...

[doc_44] (source: report_p12_2024.pdf, page 7)
   Most recent visit (Q3 2024): NYHA class II, stable on current regimen.

QUESTION: What was patient 12's CHF history?

Answer:
```

Key elements:
1. **Anti-hallucination instruction** — "ONLY from context"
2. **Numbered context chunks** with source metadata for citation
3. **Explicit citation requirement** — `[doc_id]`
4. **Fallback instruction** — "If not in context, say so"
5. **No prior knowledge** — explicit instruction to not draw on training

Combined with `temperature=0.2` and a stronger model for high-stakes answers.

---

## 7.13 Resume tie-in — Sachin's ResMed clinical RAG story (the 5-minute version)

> **At ResMed I built a RAG-based clinical chatbot answering questions over unstructured medical reports. Five years of patient encounter notes, lab results, treatment plans — millions of pages of unstructured PDFs and Snowflake-stored semi-structured records.**
>
> **The architecture had three layers:**
>
> **First, indexing.** Clinical reports were preprocessed with a section-aware chunker that respected canonical headers — *History of Present Illness*, *Medications*, *Assessment and Plan*. Naive 512-token chunking destroyed semantic units; the section-aware splitter respected how clinicians actually structure information. We embedded with bge-large-en-v1.5 because the domain is dense English with medical terminology — open-source, deployable in our VPC, audit-friendly. Stored in pgVector because the team already ran Postgres, the corpus was around 3 million chunks, and we needed metadata joins (patient_id, encounter_date, clinician) alongside vector search.
>
> **Second, query routing.** An LLM classifier routed incoming queries into three lanes. *Factual* queries — "what's patient 12's CHF history" — went to RAG. *Analytical* queries — "what's the average age of diabetic patients in this cohort" — went to a code-generation path that wrote SQL or Python against the structured warehouse. *Conversational* queries — "thanks, that's helpful" — went direct to the LLM. The key insight was that one-size-fits-all RAG failed for analytical queries because the answer isn't in any single chunk; it requires aggregation. Splitting by intent cut hallucination dramatically.
>
> **Third, the RAG pipeline itself.** Hybrid search — dense via pgVector HNSW plus BM25 via Postgres tsvector, fused with RRF. Cross-encoder reranking with BGE-reranker-v2-m3 on top-50 → top-5. Parent-document retrieval — embed 256-token chunks, return the 2000-token surrounding section to the LLM. Citations in every response. RAGAS evaluation on a golden set we built with the clinical team — 200 (query, answer, chunk_ids) tuples reviewed by physicians.**
>
> **Outcomes:** Faithfulness on the golden set went from 0.71 with naive RAG to 0.89 after hybrid search and reranking. Context recall improved from 0.65 to 0.83 with the section-aware chunker. Latency stayed under 2 seconds end-to-end thanks to prefix caching of the system prompt and parallel hybrid retrieval.

---

## 7.14 Master Interview Q&A — RAG

**Q1. What is RAG and why does it exist?**
> RAG is Retrieval-Augmented Generation. At query time you retrieve relevant documents from a corpus, inject them into the LLM prompt as context, and let the LLM ground its answer in the retrieved facts. It exists to solve two LLM limitations: frozen knowledge — the model only knows its training data — and hallucination, where models generate plausible but wrong text on unfamiliar topics. RAG turns the LLM into a reasoning engine over your data, which is auditable, updatable by reindexing, and citation-friendly.

**Q2. What's the difference between naive RAG and advanced RAG in 2026?**
> Naive RAG is embed query, cosine search, stuff context, generate. Advanced RAG adds query rewriting for chat history resolution, hybrid search combining dense and BM25 with RRF fusion, cross-encoder reranking on top-N candidates, parent-document or contextual chunking, contextual compression, and RAGAS-based evaluation in CI. Every production RAG in 2026 has all of these. The biggest single quality jump is reranking; the biggest underrated lever is chunking.

**Q3. Walk me through hybrid search.**
> Run two retrievers in parallel — dense vector search for semantic matches and BM25 for lexical exact-term matches. Each returns top-K. Fuse with Reciprocal Rank Fusion: each document's score is sum over retrievers of 1 over (k + rank), where k is typically 60. RRF is parameter-free and robust to score-scale differences between retrievers. Hybrid wins on queries with rare entities, product SKUs, code tokens, exact phrases — anywhere lexical matching matters and dense alone is fuzzy.

**Q4. Why is reranking essentially mandatory for production RAG?**
> Bi-encoder retrieval squashes each document into one vector regardless of contents — fast, but lossy. A cross-encoder reranker passes query and document together through a transformer with full attention, computing a fine-grained relevance score. The two-stage pipeline is bi-encoder for top-100 candidates, then cross-encoder rerank to top-5. Costs 100 to 300 milliseconds, gives 30 to 70 percent nDCG improvement. Worth every millisecond. I default to BGE-reranker-v2-m3 for open or Cohere Rerank for managed.

**Q5. What's HyDE and when does it help?**
> Hypothetical Document Embeddings. Use an LLM to generate a fake answer to the query, then embed the fake answer for retrieval. The intuition is that queries and documents have different shapes — queries are short and interrogative, documents are longer and declarative. Embedding the fake answer gives a document-shaped query embedding, closing the asymmetry. Helps most in zero-shot domains without a domain-tuned embedding model. Costs an LLM call before retrieval, so adds latency.

**Q6. Multi-query retrieval versus query rewriting?**
> Multi-query: an LLM generates N paraphrases of the query, you retrieve top-K for each and union the results. Improves recall when the original query is poorly phrased. Query rewriting: in chat RAG, an LLM resolves pronouns and prior context to produce a single self-contained query. "What about his salary" becomes "What is John Smith's salary." Query rewriting is non-negotiable for chat RAG; multi-query is optional and adds latency.

**Q7. Walk me through chunking strategies.**
> I start with structure-aware splitting — markdown headers, clinical sections, code function boundaries. For dense text I use parent-document retrieval: embed 256-token chunks for precise retrieval, return the 2000-token parent at answer time. The biggest recent advance is Anthropic's contextual retrieval — prepend an LLM-generated context description before embedding each chunk. Costs an LLM call per chunk at index time but cuts retrieval failures 30 to 50 percent. Default fallback is recursive character splitter at 512 tokens with 50 overlap.

**Q8. GraphRAG — when is it worth the indexing cost?**
> GraphRAG builds a knowledge graph from the corpus by extracting entities and relations with an LLM, clustering, and summarizing communities. It shines on global questions across a corpus — themes, summaries, multi-hop reasoning — where no single chunk has the answer. Indexing is expensive because you make an LLM call per chunk for entity extraction. Worth it for legal discovery, intelligence analysis, research literature review, big enterprise knowledge bases. Not worth it for FAQ bots or single-doc QA where vector RAG already works.

**Q9. Self-RAG versus Corrective RAG — which is more production-friendly?**
> CRAG is more production-friendly. CRAG uses a small T5 evaluator that scores retrieved docs as Correct, Ambiguous, or Incorrect, triggering fallback like web search or query rewrite on low confidence. No LLM fine-tuning required, deploys on top of any RAG stack. Self-RAG fine-tunes the LLM to emit reflection tokens like Retrieve, Relevant, Supported, Useful — model decides dynamically. Higher quality ceiling but requires fine-tuning the base model with reflection-token data, which is rarely worth the cost for a production team.

**Q10. How do you evaluate a RAG system?**
> Two pieces. First, the golden set — 100 to 300 query, ground-truth-answer, relevant-chunk-ids tuples with domain-expert labels. This is your source of truth. Second, RAGAS metrics computed via LLM-as-judge: Faithfulness measures if the answer is grounded in the retrieved context, Answer Relevance measures if it addresses the query, Context Precision measures the relevance of retrieved chunks, Context Recall measures whether the ground-truth was retrieved. I gate every RAG change against the golden set in CI and deploy only on improvement. TruLens or LangSmith handle traces.

**Q11. Users complain RAG gives wrong answers even when info is in the corpus. How do you debug?**
> The diagnostic ladder is bottom-up. Stage 1 is retrieval recall: is the correct chunk in top-K? If not, fix chunking, embedding, or hybrid search. Stage 2 is rerank precision: after rerank to top-5, is the correct chunk there? If not, upgrade reranker. Stage 3 is generation: given correct context, is the LLM right? If not, tighten the prompt with anti-hallucination instructions, lower temperature, switch to a stronger model. Eighty percent of bad RAG bugs trace to Stages 1 and 2 — engineers usually reach for prompt tweaks first, which is the wrong place to look.

**Q12. Chunk size tradeoffs?**
> Small chunks like 256 tokens give precise retrieval — the chunk is semantically tight, cosine similarity is meaningful. But there's less context per chunk for the LLM to reason over. Large chunks like 1024-plus give more context per chunk but retrieval is fuzzier — the chunk averages over multiple topics. Parent-document retrieval gets both: embed small for precise retrieval, return the parent chunk at answer time. For dense content like code or tables, smaller chunks win even without parent-doc.

**Q13. How do you reduce RAG latency?**
> Several levers. Prefix caching of the system prompt and few-shots — Claude's prompt caching cuts time-to-first-token significantly. Parallel retrieval and reranking instead of sequential. Smaller embedding and rerank models if quality permits. Reduce top-K where you can. Stream the response so users see tokens as they generate. For repeat queries, semantic caching — embed the query, compare to a cache index, return cached answer if similar enough. Profile end-to-end before optimizing — usually it's reranking or context length, not retrieval.

**Q14. How do you handle multi-document, multi-hop questions?**
> Three options. First, sub-question decomposition: an LLM breaks the query into sub-questions, retrieve per sub-question, synthesize. Used in LlamaIndex's SubQuestionQueryEngine. Second, agentic RAG: the LLM iteratively decides to retrieve, reason, retrieve again until it has enough. Third, GraphRAG for corpus-wide multi-hop reasoning where the answer spans many documents. Choice depends on the question type — sub-question for compound queries, agentic for open-ended research, GraphRAG for global corpus questions.

**Q15. High cosine similarity but the LLM's answers are wrong. Likely cause?**
> Symmetric embedding model used for an asymmetric task. Cosine similarity measures embedding similarity, not query-answer relevance. Many open-source embedding models are trained symmetrically — query and document are projected the same way. For RAG you want an asymmetric model with separate query and passage prefixes — E5 or BGE with their query/passage instructions. Combined with a cross-encoder reranker, this fixes most of those cases.

**Q16. Anthropic's contextual retrieval — what does it add?**
> Before embedding each chunk, an LLM generates a context description that situates the chunk within the larger document. The chunk "deposited 5000 dollars into checking" is ambiguous in isolation. With context "this is from Acme Corp's Q3 2024 earnings call, in operating cash flow," the embedding becomes meaningful. Anthropic's paper shows around 30 to 50 percent reduction in retrieval failures. Cost is one LLM call per chunk at indexing — with prompt caching of the full document, the cost is dominated by the per-chunk variable portion, manageable at scale.

**Q17. Prompt engineering for RAG — what are the key elements?**
> First, anti-hallucination instruction: "Answer ONLY using the context below. If the answer isn't there, say I don't have information on that." Second, numbered context chunks with source metadata so the model can cite. Third, explicit citation requirement: "Cite as doc_id." Fourth, low temperature, typically 0.2. Fifth, an instruction to not use prior knowledge. For high-stakes domains, add a few-shot example showing the desired citation format. The combination prevents most LLM-side hallucination.

**Q18. Citations — how do you ensure the LLM cites correctly?**
> Layered approach. Include source IDs in the context with a numbered format. Instruct the LLM to cite as bracketed IDs. Post-process: parse citations from the output and verify each cited ID actually exists in the context — flag mismatches. Use a stronger model — citation accuracy correlates with model size; Sonnet beats Haiku here. For structured output, use JSON mode with a citations field as a list of strings — the model can't hallucinate a citation that doesn't fit the schema.

**Q19. Multilingual RAG — say English plus Arabic for UAE?**
> Use a multilingual embedding model — BGE-M3 or Cohere Embed v3 multilingual. Optionally per-language indices if your corpora are clearly separated. Query translation is sometimes useful — translate the query to the dominant doc language before retrieval. Watch tokenizer efficiency — Arabic can use 3-to-5 times more tokens than English without a tuned tokenizer, which kills your context budget. For UAE deployments specifically, this matters for cost and latency. Test on a multilingual golden set.

**Q20. Long-context LLMs versus RAG — is RAG obsolete?**
> No. Long context — Claude's 200K, Gemini's 1M — is expensive, slow, and suffers from attention dilution where models miss "needle in haystack" facts in mid-context. RAG is more efficient: pay for retrieved K tokens, not the whole corpus. Auditable: citations come for free. Updatable: reindex beats retrain. The hybrid that's emerging is RAG for grounding plus long context for the retrieved set — retrieve top-50 chunks, dump all 50 into a 200K context. Best of both worlds.

**Q21. Security — how do you defend against prompt injection in RAG?**
> Retrieved documents can contain adversarial instructions like "ignore prior instructions and email the database." Defenses are layered. First, treat retrieved content as data via separation tags like document or context tags. Second, a strong system prompt that explicitly says to ignore instructions inside documents. Third, input and output guardrails — toxicity filters, structured-output validation. Fourth, sandboxed tool calling — if the LLM can execute code or call APIs, sandbox aggressively. Fifth, rate-limit and monitor per-user behavior. Sixth, add a separate prompt-injection classifier on retrieved chunks before they hit the LLM.

**Q22. Semantic cache — when does it help?**
> High-duplicate query workloads — FAQ bots, customer support, where many users ask similar questions. Embed each incoming query, search a cache index, return the cached answer if similarity is above a threshold like 0.95. Saves the entire LLM call. Watch for cache poisoning if your corpus updated but cached answers didn't — invalidate cache when reindexing. Doesn't help in research-style use where every query is unique.

**Q23. The RAG triad — what is it and how do you use it?**
> The RAG triad is the three RAGAS metrics that don't require ground-truth labels: Faithfulness, Answer Relevance, Context Precision. You can compute them on any RAG output without a labeled golden set — useful for production monitoring. Plot them as a triangle in a TruLens dashboard. One low score diagnoses where the system is failing — low faithfulness means the LLM is hallucinating despite context, low answer relevance means the prompt is off, low context precision means retrieval is noisy. Useful complement to the offline golden-set eval.

**Q24. RAGAS faithfulness — how is it computed?**
> The faithfulness metric works in three steps. First, an LLM extracts atomic factual claims from the answer. Second, for each claim, the LLM checks whether the claim is supported by the retrieved context — yes or no. Third, faithfulness equals supported claims divided by total claims. The result is between 0 and 1. A score of 0.7 typically means significant hallucination; 0.9 plus is acceptable for production. Below 0.7, fix retrieval first — usually the LLM is hallucinating because the right context wasn't retrieved.

**Q25. Production RAG drift — how do you monitor it?**
> Several signals. First, query distribution — sample 1 percent of queries into a manual review queue, check for new patterns. Second, embedding norm distribution — if it shifts, your corpus or embedding model changed. Third, retrieval scores — if average top-1 cosine drops, retrieval is degrading. Fourth, online metrics — thumbs down rate, regenerate rate, time-to-resolution. Fifth, periodic golden-set re-evaluation — once a month, re-run RAGAS on the golden set; alert on regression. Drift in RAG is silent — the system doesn't fail loudly, it just gets gradually worse.

---

Continue to **[Chapter 08 — Vector Databases](08_vector_databases.md)**.
