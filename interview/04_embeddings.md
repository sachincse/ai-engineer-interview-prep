# Chapter 04 — Embedding Models
## How they're created, what makes them work, and what to pick

> "Explain how an embedding model is created" — almost certain interview question for RAG-heavy roles. This chapter goes from the contrastive loss to production choice (BGE vs E5 vs OpenAI vs Cohere).

---

## 4.1 What is an embedding model?

A function **f(text) → vector ∈ ℝ^d** that maps semantically similar texts to nearby points in a d-dimensional space.

```
f("I love NLP")          → [0.12, -0.43, 0.77, ..., -0.11]     (d=768)
f("Natural language is   → [0.11, -0.41, 0.75, ..., -0.12]     (nearby)
    my passion")
f("The boiler is        → [-0.82, 0.33, -0.10, ..., 0.45]      (far)
    broken")
```

Similarity is then measured with **cosine similarity** (or dot product / Euclidean):

```
sim(a, b) = (a · b) / (||a|| · ||b||)
```

---

## 4.2 Architecture — an encoder with a pooling head

Modern embedding models are almost all **encoder-only transformers** (BERT-family) fine-tuned with a contrastive objective.

```
   Text: "How do I reset my password?"
         │
    ┌────▼─────┐
    │ Tokenizer│ → [CLS] how do I reset my password ? [SEP]
    └────┬─────┘
         │
    ┌────▼─────────────────┐
    │ Transformer Encoder  │   ← BERT / RoBERTa / DeBERTa / modern retriever
    │   (bidirectional     │
    │    self-attention)   │
    └────┬─────────────────┘
         │ hidden states H ∈ ℝ^(n × d)
         │
    ┌────▼─────┐
    │  Pooling │   ← mean / CLS / weighted / last-token
    └────┬─────┘
         │
    ┌────▼─────┐
    │ Normalize│   ← L2 norm → unit vector (needed for cosine)
    └────┬─────┘
         ▼
     Embedding ∈ ℝ^d
```

### Pooling choices

- **[CLS] token** (BERT classic) — use the first special-token's hidden state. Works if you *fine-tuned* the model, less great off-the-shelf.
- **Mean pooling** (SBERT default) — average all token hidden states, usually weighted by attention mask. Most robust.
- **Last-token pooling** (for decoder-only embeddings like GTE-Qwen, NV-Embed) — the last token aggregated info causally.
- **Weighted / learned pooling** — attention-based mean with learned weights.

---

## 4.3 Contrastive Learning — the core training recipe

### The intuition

Push **similar** texts together; pull **dissimilar** apart. No absolute labels needed — just "these two are related, these aren't."

### InfoNCE loss

Given a query q, one positive p⁺, and N negatives {p_i}:

```
L = - log [ exp(sim(q, p⁺) / τ)  /  Σᵢ exp(sim(q, pᵢ) / τ) ]
```

- sim = cosine similarity (or dot product after L2 norm)
- τ = **temperature** (typically 0.01-0.05)
- Sum over **in-batch negatives** + optional hard negatives

It's literally a softmax cross-entropy where the "classes" are "which candidate is the positive?"

### The temperature knob
- Small τ (0.01) → sharp contrasts, learns fine distinctions but unstable
- Large τ (0.5) → soft contrasts, under-powered gradient
- Sweet spot: τ ≈ 0.02-0.05

### In-batch negatives
With batch size B, each of the B queries has (B-1) "free" negatives — the positives of other queries in the batch. **Bigger batch = more negatives = better training.** Which is why embedding training uses huge batches (2K-32K).

### Hard negatives
Random negatives are usually trivially separable (no gradient signal). **Hard negatives** are semantically similar-but-wrong candidates. Mining strategies:

- **BM25 top-k** for a query (lexically similar, semantically may miss)
- **Previous-model top-k** (self-supervised mining — model-as-labeler)
- **ANCE** — async hard-negative refresh from the current model
- **Triplet mining** — semi-hard (positives closer than negative, but within margin)

---

## 4.4 The multi-stage training recipe (BGE, E5, GTE — 2024-2026 SOTA)

```
Stage 1: Weakly-Supervised Contrastive Pretraining
  - Billions of pairs from web: (title, body), (Q, A), (comment, reply)
  - High-throughput, low-quality filter
  - Big batch (16K+ negatives)
  - Goal: teach model "semantic closeness" broadly

Stage 2: Supervised Contrastive Fine-tuning
  - Curated: MS-MARCO, NQ, HotpotQA, FEVER, TriviaQA, etc.
  - Hard negatives mined (BM25 + previous-model)
  - Smaller batch, smaller LR
  - Goal: tight retrieval performance

Stage 3 (optional): Instruction Tuning
  - Prepend task prefix: "Represent this query for retrieval:"
  - Same model serves retrieval, classification, clustering, STS
  - Goal: one model, many tasks (MTEB winner style)

Stage 4 (optional): Distillation / Matryoshka
  - Distill teacher into smaller student for latency
  - Train with Matryoshka loss so truncated dims still work
```

---

## 4.5 Bi-Encoder vs Cross-Encoder vs ColBERT

### Bi-encoder (SBERT, BGE, E5)
```
f(q) = vector_q      │  f(d) = vector_d      │  score = cos(vector_q, vector_d)
```
- Encodes query and doc **independently**
- Pre-encode corpus; at query time, one forward pass + ANN
- **Fast, scalable** — used for retrieval

### Cross-encoder (BGE-reranker, Cohere Rerank)
```
g([q; d]) → score
```
- Encodes query AND doc **together** with full attention
- Much higher quality, **100-1000× slower**
- Used for **reranking** top-50 → top-5

### ColBERT (late interaction)
```
Per-token embeddings for q (q_i) and d (d_j)
score = Σ_i max_j (q_i · d_j)      ← MaxSim
```
- Keeps per-token vectors instead of one pooled vector
- Higher accuracy than bi-encoder, much cheaper than cross-encoder
- **Expensive storage** (one vector per token, not per doc)
- Used when recall matters and storage is acceptable

### Standard production pipeline

```
Query ──▶ Bi-encoder top-100 (fast, high recall)
              │
              ▼
         Cross-encoder rerank → top-5 (high precision)
              │
              ▼
           LLM answer generation
```

---

## 4.6 Matryoshka Representation Learning (MRL)

### The problem
You train a 1024-d embedding. Storage cost at 100M vectors = 400GB. You want to cheaply truncate to 256-d but plain truncation destroys retrieval.

### The solution
Train the embedding so the **first 64, 128, 256, 512, 1024 dims are each a valid embedding** — via a multi-resolution contrastive loss:

```
L_MRL = Σ_d ∈ {64, 128, 256, 512, 1024}  L_InfoNCE(truncate(emb, d))
```

- Each smaller prefix is forced to carry retrieval-relevant signal
- At inference, pick the dimension that fits your storage budget

### Used in
- OpenAI `text-embedding-3-small` / `text-embedding-3-large`
- Nomic Embed v1
- BGE-M3 (multi-granularity)

### Why MRL beats PCA on a non-MRL embedding
PCA maximizes variance — not retrieval utility. MRL is *trained* for graceful truncation. On BEIR, MRL@256 often beats PCA@256 by 5-10 nDCG points.

---

## 4.7 Instruction-tuned embeddings

The MTEB leaderboard (Massive Text Embedding Benchmark) is dominated by models that prepend a task instruction:

```
For retrieval:
    q = "query: How do I reset my password?"
    d = "passage: To reset your password, click..."

For clustering:
    t = "Represent the sentence for clustering: ..."
```

**Why this works:** One model learns to be task-conditional — retrieval asymmetry (query vs passage), STS symmetry, clustering, classification — all from one checkpoint.

### Families
- **E5** (Microsoft): `"query: "`, `"passage: "`
- **BGE** (BAAI): `"Represent this sentence for searching relevant passages: "`
- **GTE** (Alibaba): simpler, general prefix
- **Nomic / Jina / Mxbai**: similar instruction style

---

## 4.8 BGE-M3 — the Swiss Army knife

**BGE-M3** (BAAI, 2024) is one model that emits **three kinds of embeddings** in a single forward pass:

1. **Dense vector** (1024-d) for semantic ANN
2. **Sparse lexical** weights (like SPLADE) for BM25-style term matching
3. **Multi-vector** (ColBERT-style) for late interaction

Combined via a learned weighted score. SOTA on multilingual retrieval. **Serious option for Avrioc's UAE deployment** since it handles English + Arabic + 100+ languages.

---

## 4.9 The MTEB Leaderboard — what's SOTA in 2026

(Moving target — check the live leaderboard before interviews.)

| Tier | Model | Dim | Strengths |
|------|-------|-----|-----------|
| Frontier open | BGE-M3, E5-Mistral-7B, NV-Embed-v2 | 1024-4096 | Top MTEB, multilingual |
| Balanced | BGE-large-en-v1.5, BGE-M3, GTE-large | 1024 | Quality/cost |
| Fast | BGE-small-en-v1.5, all-MiniLM-L6-v2 | 384 | Real-time, edge |
| Closed | OpenAI text-embedding-3-large, Cohere Embed v3, Voyage AI | 256-3072 | Managed, strong baseline |

### Picking one — decision rules

| Constraint | Recommendation |
|-----------|----------------|
| Need Arabic + English (UAE) | **BGE-M3** or **Cohere Embed v3 multilingual** |
| Must stay on-prem (data residency) | BGE / E5 / GTE (Apache-2) |
| Latency-critical | BGE-small or all-MiniLM-L6-v2 |
| Best quality, money-no-object | NV-Embed-v2 or OpenAI text-embedding-3-large |
| Want truncatable | OpenAI text-embedding-3 or BGE-M3 |
| Need hybrid (sparse+dense) | BGE-M3 (built-in) or SPLADE |

---

## 4.10 Evaluation — how do you know an embedding is good?

### Intrinsic
- **STS-B** — sentence-pair similarity correlation with human ratings
- **MTEB** — comprehensive: retrieval, classification, clustering, reranking, STS, summarization

### Extrinsic (retrieval)
- **Recall@k** — did the right doc make top-k?
- **nDCG@k** — ranking quality
- **MRR** — position of first relevant
- **Hit@k** — any relevant in top-k?

### Domain-specific evaluation matters
MTEB is general; your data may be medical, legal, or financial. **Always build a labeled in-domain eval set** (100-500 (query, relevant doc) pairs) before picking a model. A top-MTEB model can lose to a smaller domain-specific one on your corpus.

---

## 4.11 Interview Q&A — Embeddings

**Q1. How is a modern embedding model trained?**
> Two-stage contrastive. Stage 1 — weakly-supervised on billions of (query, positive) pairs mined from web. Stage 2 — supervised fine-tuning on MS-MARCO / NQ with mined hard negatives. Optional: instruction tuning (prefix-based), distillation, Matryoshka.

**Q2. Explain InfoNCE loss.**
> Softmax cross-entropy over (positive + many negatives). `L = -log(exp(sim(q,p+)/τ) / Σ exp(sim(q,pi)/τ))`. Temperature τ controls contrast sharpness. Uses in-batch negatives + mined hard negatives.

**Q3. Why hard negatives are critical.**
> Random negatives are trivially separable (near-zero gradient). Hard negatives — semantically similar but not relevant — force fine-grained discrimination. BM25 top-k, previous-model top-k, and ANCE are standard mining approaches.

**Q4. Bi-encoder vs Cross-encoder — when each?**
> Bi-encoder encodes independently, pre-indexable, millisecond latency per query. Cross-encoder encodes (q, d) jointly — 100-1000× slower but far more accurate. Standard: bi-encoder top-100, cross-encoder rerank top-5.

**Q5. What is ColBERT's MaxSim?**
> Per-token embeddings for query and doc. Score = Σ_i max_j (q_i · d_j) — each query token finds its best-matching doc token; sum across query. Fine-grained term-level relevance at a fraction of cross-encoder cost but heavy storage.

**Q6. Matryoshka embeddings — what and why?**
> Train one model whose first d' ≤ d dims are themselves a valid embedding, for multiple d' simultaneously. Lets you truncate to 256-d for cheap storage with minimal accuracy loss — MRL@256 beats PCA@256 on non-MRL embeddings by 5-10 nDCG.

**Q7. What is instruction-tuned embedding?**
> Model prepends a task prefix ("query: …", "passage: …", "Represent for clustering: …") so one checkpoint serves retrieval, classification, clustering, STS. BGE, E5, GTE all do this.

**Q8. BGE-M3 — what makes it special?**
> One model emits three outputs in a single pass: dense vector, sparse lexical weights (SPLADE-style), and multi-vector (ColBERT-style). Combined via learned weights. SOTA multilingual, including Arabic.

**Q9. [Gotcha] Cosine similarity 0.92 retrievals but RAG answers are wrong. Why?**
> Cosine measures *similarity*, not *answer relevance*. A question and its answer are often asymmetric — symmetric STS models pull paraphrases, miss query-passage relationships. Fix: asymmetric retrieval model (E5 query/passage, BGE), and add a cross-encoder reranker.

**Q10. What's the difference between symmetric and asymmetric retrieval?**
> Symmetric: query and doc share the same encoder/prompt (STS, paraphrase mining). Asymmetric: queries are short questions, docs are long passages — encode with different instructions. Most real-world retrieval is asymmetric.

**Q11. How do you evaluate a new embedding model on your data?**
> Build a 100-500 labeled (query, relevant doc) set. Measure recall@10, nDCG@10 on it. Compare against baseline (BM25 + current model). Check MTEB as a general sanity, but in-domain eval is the ground truth.

**Q12. [Gotcha] What's the risk of choosing a high-dim embedding?**
> Linear storage cost in dim, quadratic cost for some indices (HNSW), slower similarity computations. A 3072-d OpenAI model at 100M vectors ≈ 1.2 TB before any index overhead. Matryoshka or simple truncation-trained models mitigate.

**Q13. Hybrid search (dense + sparse) — why and how?**
> Dense captures semantics; sparse (BM25, SPLADE) captures exact term matches (product codes, rare names). Fuse scores with RRF (reciprocal rank fusion) or weighted sum. Wins on mixed-intent queries. BGE-M3 bakes this in.

**Q14. [Gotcha] Can you fine-tune an embedding model on your data?**
> Yes, with contrastive fine-tuning. Needs (query, positive) pairs; hard negatives are a huge lever. Libraries: sentence-transformers, InfiNity, Nomic's API. Risk: overfitting on a small set and losing generalization — usually better to try a domain-specific public model first.

**Q15. RAG is hallucinating even with high-cosine retrieved docs. Debug checklist?**
> (1) Retrieval: are the *right* docs actually in top-10? Check against labeled set. (2) Embedding symmetry: using a symmetric model for an asymmetric task? Swap to E5/BGE. (3) Missing BM25: add hybrid search. (4) Reranker: are you reranking with a cross-encoder? (5) Chunking: is the right info even in one chunk, or split across chunks? (6) Prompt: is the LLM ignoring the context?

**Q16. What is SPLADE?**
> SParse Lexical AnD Expansion — learns a sparse token-weight representation that functions like BM25 but with query/doc expansion. Uses MLM + a sparsity regularizer. Combines well with dense retrieval in hybrid search.

**Q17. Vector normalization — does it matter?**
> If you're using cosine similarity, you MUST L2-normalize. If dot product, normalization makes it equivalent to cosine; without it, vector magnitudes influence scores (usually not what you want). Most embedding libs normalize by default — check.

**Q18. Can decoder-only LLMs be used as embedding models?**
> Yes. Mount a last-token pool + contrastive fine-tune. NV-Embed-v2, GTE-Qwen, E5-Mistral-7B, Linq-Embed-Mistral — all decoder-only. Quality is SOTA on MTEB but inference is expensive (7B params vs 110M for BGE-large).

**Q19. How do embedding models handle code vs natural language?**
> Code-specific models (CodeBERT, GraphCodeBERT, UnixCoder, Jina Code) are trained on code corpora with AST-aware objectives. For mixed text-and-code retrieval, general multilingual models (BGE-M3, Voyage-code-2) often suffice.

**Q20. Explain asymmetric search in one line.**
> Queries and documents have different shapes — a short question and a long paragraph — so you encode them with different prompts, not the same prompt.

---

## 4.12 Resume tie-ins

- **"RAG-based knowledge-base chatbot using LLMs and RAG"** — be precise on your embedding choice. "We evaluated text-embedding-3-small and BGE-large on a 500-pair labeled medical QA set; chose X because..."
- **"NER-based lender entity extraction"** — though a classification task, you likely fine-tuned a BERT-family encoder. Same architecture family as embedding models; connect them.
- **"Snowflake feature store"** — for real-time embedding retrieval you'd want an online feature store + vector DB. Be ready to discuss the latency budget for the full retrieve-embed-search chain.

---

Continue to **[Chapter 05 — LLM Parameter Tuning](05_parameter_tuning.md)**.
