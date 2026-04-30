# Chapter 27 — RAG Evaluation Deep Dive

> **Why this chapter exists:** RAG evaluation is the single most-asked deep technical topic in modern LLM interviews. Every senior interviewer at an LLM-shop wants to hear how you would *measure* a RAG system, not just build one. The reason: RAG systems fail in many distinct ways (bad retrieval, bad reranking, hallucinated answers, irrelevant answers), and each failure mode needs a different metric to catch. If you can explain Precision@k, Recall@k, MRR, NDCG, and the four RAGAS metrics with the math worked through on a whiteboard, you're already in the top tier of candidates.
>
> **How to read this chapter:** Section by section. Each metric has a plain-English mental model, then the formula, then a worked numeric example you can recreate at a whiteboard, then the interview-style narrative answer. Master one metric before moving on.

---

## 27.1 The two halves of RAG evaluation

A RAG pipeline has two distinct stages, and each needs its own evaluation:

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                      RAG PIPELINE                                │
   │                                                                  │
   │   ┌────────────┐    ┌────────────┐    ┌────────────┐             │
   │   │   Query    │───▶│ Retrieval  │───▶│  Generation│             │
   │   │            │    │  (top-k    │    │  (LLM uses │             │
   │   │            │    │   docs)    │    │   docs to  │             │
   │   │            │    │            │    │   answer)  │             │
   │   └────────────┘    └────────────┘    └────────────┘             │
   │                          │                  │                    │
   │                          ▼                  ▼                    │
   │                  ┌─────────────────┐ ┌─────────────────┐         │
   │                  │   Retrieval     │ │   Generation    │         │
   │                  │   metrics       │ │   metrics       │         │
   │                  │                 │ │                 │         │
   │                  │  Precision@k    │ │  Faithfulness   │         │
   │                  │  Recall@k       │ │  Answer rel.    │         │
   │                  │  MRR            │ │  BERTScore      │         │
   │                  │  NDCG           │ │  ROUGE / BLEU   │         │
   │                  │  Hit Rate       │ │  LLM-as-judge   │         │
   │                  └─────────────────┘ └─────────────────┘         │
   └──────────────────────────────────────────────────────────────────┘
```

**Retrieval metrics** answer: "Did we find the right documents?"
**Generation metrics** answer: "Given the documents, did we produce a good answer?"

You need both. A perfect retrieval feeding a hallucinating LLM still gives a bad answer. Perfect generation over wrong documents still gives a bad answer. Each metric isolates one failure mode.

---

## 27.2 Setting up — what you need to evaluate RAG

Before any metric, you need a **golden evaluation set**. This is a list of questions, each with:

1. The question text
2. The set of documents that *should* be retrieved (the "relevant documents") — usually labeled by domain experts
3. A reference answer the LLM should produce (optional but valuable)

Building this set is the hardest part of RAG evaluation. For ResMed's clinical chatbot, we worked with domain experts to label 200 questions with their relevant clinical sections. For most teams, 50-200 well-curated questions beats 10,000 noisy ones.

Once you have the golden set, every evaluation run measures: for each question, run the RAG pipeline, capture the retrieved documents and the generated answer, and compare against the golden labels.

---

## 27.3 Retrieval metric 1 — Precision@k

### The plain-English mental model

Imagine you ask "what causes asthma?" and your retriever returns the top 5 documents. Of those 5, 3 are actually about asthma causes and 2 are off-topic. Your **Precision@5** is 3/5 = 0.6. That's it — Precision@k is the fraction of retrieved documents that are actually relevant.

### The formula

```
                            number of relevant documents in top-k
   Precision@k  =  ─────────────────────────────────────────────
                                        k
```

Note: the denominator is **k** (the number of documents you retrieved), not the total number of relevant documents in the corpus.

### Worked example

Suppose for a query, the corpus has 10 truly-relevant documents. Your retriever returns the top 5, and you check which are relevant:

```
   Rank  Document     Relevant?
   ────  ──────────   ─────────
    1    doc_42         ✓
    2    doc_113        ✓
    3    doc_7          ✗
    4    doc_201        ✓
    5    doc_56         ✗
```

Out of 5 retrieved, 3 are relevant.

```
   Precision@5  =  3 / 5  =  0.60
```

If you computed Precision@3, you'd consider only the first three rows: 2 of 3 are relevant, so Precision@3 = 0.67.

### Why Precision@k matters

Precision@k tells you whether your retriever is **wasting context window**. If you retrieve 10 documents but only 3 are relevant, you're stuffing 70% noise into the LLM's prompt. The LLM has to do work to ignore the noise, which can degrade answers. High Precision@k means a clean prompt.

### What to fix when Precision@k is low

- Better embedding model (the documents you retrieve are not actually similar)
- Add a reranker (cross-encoder scoring of (query, doc) pairs to push relevant docs to the top)
- Hybrid search (combine BM25 keyword match with dense embedding match — RRF fusion)
- Tighter chunking (a chunk that mixes relevant and irrelevant content gets retrieved for the wrong reasons)

### Interview narrative for Precision@k

> "Precision@k tells me what fraction of the documents I retrieved are actually relevant. If I retrieve 5 documents and 3 are relevant, my Precision@5 is 0.6. The metric matters because it bounds the noise in the LLM's context window — if precision is low, the LLM is sifting through irrelevant documents to find the answer, which often hurts faithfulness. Low Precision@k usually means my reranker is missing or my embedding model isn't tuned to the domain. The fix is typically a cross-encoder reranker that scores query-document pairs jointly and pushes the most relevant to the top."

---

## 27.4 Retrieval metric 2 — Recall@k

### The plain-English mental model

Same query, "what causes asthma?" If there are 10 truly-relevant documents in the corpus and your top-5 retrieval finds 3 of them, your **Recall@5** is 3/10 = 0.3. Recall@k is the fraction of all relevant documents that you managed to retrieve.

### The formula

```
                          number of relevant documents in top-k
   Recall@k   =  ─────────────────────────────────────────────────
                       total number of relevant documents in corpus
```

The denominator is the *total* number of relevant documents that exist for this query, regardless of where they rank.

### Worked example

Same scenario as before — corpus has 10 truly-relevant documents, your top-5 has 3 relevant.

```
   Recall@5  =  3 / 10  =  0.30
```

Now imagine you increased k to 20 and found 7 of the 10 relevant documents in your top-20:

```
   Recall@20  =  7 / 10  =  0.70
```

So Recall increases (or stays flat) as k grows — by definition, retrieving more documents can only catch more relevant ones, never fewer.

### Why Recall@k matters

Recall@k tells you whether the **answer is even possible**. If the relevant documents aren't in your top-k, the LLM cannot ground its answer on them. The LLM might hallucinate something plausible, but it can't actually use the source. So Recall@k is a hard ceiling on RAG quality — no amount of fancy prompting fixes a retrieval that didn't include the relevant document.

### Precision vs Recall — the trade-off

These metrics pull in opposite directions. As k increases:

- **Recall@k** goes up (more documents caught)
- **Precision@k** typically goes down (more noise mixed in)

So you choose k based on your latency budget and your context window. Too small a k → low recall. Too large a k → low precision and a bloated prompt.

The way I think about this in production: **set k by what your reranker can handle**. Retrieve 50 with the embedding model (high recall), rerank to top-5 (high precision), feed those 5 to the LLM. You get the best of both.

### What to fix when Recall@k is low

- Larger k (cheap, sometimes the right answer)
- Hybrid search: dense retrieval often misses keyword matches that BM25 catches
- Query rewriting: rewrite the query into multiple paraphrases and retrieve for each
- Larger embedding model
- Longer chunks (so a relevant fact doesn't get split across chunks that fail to retrieve)

### Interview narrative for Recall@k

> "Recall@k is the fraction of all relevant documents in the corpus that I retrieved in my top-k. If there are 10 documents about asthma in the corpus and my top-5 finds 3 of them, Recall@5 is 0.3. This metric matters because it's a hard ceiling on RAG quality — if the relevant document isn't in the top-k, the LLM can't ground its answer on it, period. Low recall typically means dense retrieval alone is missing keyword matches; the fix is hybrid search combining BM25 with dense embeddings via Reciprocal Rank Fusion. I usually set k as high as my reranker can handle, retrieve broadly, then rerank tightly — that captures recall and precision together."

---

## 27.5 Retrieval metric 3 — Mean Reciprocal Rank (MRR)

### The plain-English mental model

Suppose you retrieve 5 documents and the *first* relevant document appears at rank 3. Your reciprocal rank for this query is 1/3 = 0.33. If the first relevant was at rank 1, RR = 1/1 = 1. If it was at rank 5, RR = 1/5 = 0.2. MRR is the mean of reciprocal ranks across all queries in your eval set.

### The formula

```
                  1     |Q|     1
   MRR   =      ───── × Σ    ─────────
                 |Q|    i=1   rank_i
```

where `|Q|` is the number of queries and `rank_i` is the position of the first relevant document for query i.

### Worked example

Suppose we have 4 queries and these are the positions of the first relevant document for each:

```
   Query 1:  first relevant at rank 1   →  RR = 1/1 = 1.00
   Query 2:  first relevant at rank 3   →  RR = 1/3 ≈ 0.33
   Query 3:  first relevant at rank 2   →  RR = 1/2 = 0.50
   Query 4:  no relevant in top-k       →  RR = 0
```

```
   MRR  =  (1.00 + 0.33 + 0.50 + 0) / 4
        =  1.83 / 4
        =  0.46
```

### Why MRR matters

MRR captures **how high the first useful document ranks**. It rewards you for surfacing the most relevant document early. This matters in two scenarios:

1. When the user only sees the top-1 result (search snippets, "did you mean").
2. When the LLM tends to anchor on the first document in its context — many models do this.

MRR is binary in spirit: it asks "where's the first hit?" It doesn't care about subsequent relevant documents. That's both its strength (sharp focus on first-hit) and its weakness (insensitive to whether your second and third hits are also relevant).

### Limitation

MRR only counts the first relevant document. If your top-5 has relevant docs at ranks 1, 2, and 3, MRR is 1.0. If your top-5 has relevant docs only at rank 1 (and the rest are noise), MRR is also 1.0. The metric can't distinguish those cases. For that, you need NDCG.

### Interview narrative for MRR

> "MRR — Mean Reciprocal Rank — measures how high the first relevant document ranks in my retrieval, averaged across all eval queries. If for one query the first relevant is at rank 3, the reciprocal rank is 1/3. I average reciprocal ranks across all queries to get MRR. The metric matters when I care about top-1 quality — for example, when the LLM tends to anchor on the first document in context, or for search snippet UX. Its limitation is that it only counts the first hit, so it doesn't reward me for ranking subsequent relevant documents well. For that I'd use NDCG."

---

## 27.6 Retrieval metric 4 — Normalized Discounted Cumulative Gain (NDCG)

### The plain-English mental model

NDCG is the most sophisticated retrieval metric. It accounts for two things at once: whether each retrieved document is relevant, *and* how high it appears in the ranking. Higher-ranked relevant documents contribute more to the score; lower-ranked relevant documents contribute less, with the discount being logarithmic. The score is normalized against the *ideal* ranking so it stays in [0, 1].

### The formulas

```
   First, the gain — how relevant is each document?
   You assign each document a relevance score, often:
     0 = not relevant
     1 = somewhat relevant
     2 = highly relevant

   Then, Discounted Cumulative Gain at k:

                k    rel_i
   DCG@k   =   Σ   ────────────
              i=1   log_2(i+1)

   The ideal DCG (IDCG) is DCG computed for the perfect ranking
   — the same relevant documents but sorted by relevance descending.

   Finally:

                       DCG@k
   NDCG@k   =   ─────────────────
                       IDCG@k
```

The discount factor `log_2(i+1)` is what makes higher ranks count more. At rank 1, the discount is log_2(2) = 1, so rel_1 contributes its full value. At rank 2, log_2(3) ≈ 1.58, so it's discounted. At rank 10, log_2(11) ≈ 3.46, so it's heavily discounted.

### Worked example

Suppose for a query you retrieve 5 documents with these graded relevance scores:

```
   Rank  Document    Relevance (0/1/2)
   ────  ─────────   ─────────────────
    1    doc_42        2  (highly relevant)
    2    doc_113       0  (not relevant)
    3    doc_7         1  (somewhat relevant)
    4    doc_201       2  (highly relevant)
    5    doc_56        0  (not relevant)
```

Compute DCG@5:

```
   DCG@5 = 2/log_2(2)  +  0/log_2(3)  +  1/log_2(4)  +  2/log_2(5)  +  0/log_2(6)
         = 2/1.000     +  0/1.585     +  1/2.000     +  2/2.322     +  0/2.585
         = 2.000       +  0           +  0.500       +  0.861       +  0
         = 3.361
```

Now compute IDCG@5 — what would the perfect ranking look like? The same documents sorted by relevance descending: 2, 2, 1, 0, 0.

```
   IDCG@5 = 2/log_2(2)  +  2/log_2(3)  +  1/log_2(4)  +  0/log_2(5)  +  0/log_2(6)
          = 2.000        +  1.262        +  0.500        +  0            +  0
          = 3.762
```

Therefore:

```
   NDCG@5 = 3.361 / 3.762 = 0.893
```

So our ranking is 89.3% of the way to ideal. The penalty came from putting `doc_113` (irrelevant) at rank 2, displacing what should have been a relevant document.

### Why NDCG matters

NDCG is the **most informative single retrieval metric** because:

1. It uses **graded relevance** (not just binary) — distinguishes "highly relevant" from "kind of relevant."
2. It uses **rank position** — rewards putting the most relevant document at the top.
3. It's **normalized** — stays in [0, 1] regardless of how many relevant documents exist for the query.

For production RAG systems, NDCG@5 or NDCG@10 are my preferred top-line retrieval metrics.

### Limitation

NDCG requires graded relevance labels. If you only have binary relevance (relevant / not relevant), NDCG simplifies but loses some of its value. In that binary case, MRR or Recall@k is often easier to interpret.

### Interview narrative for NDCG

> "NDCG — Normalized Discounted Cumulative Gain — is the most complete single retrieval metric. It rewards two things at once: whether each retrieved document is relevant, with a graded score, and how high it appears in the ranking, with a logarithmic discount. The numerator, DCG, sums each document's relevance divided by log-base-2 of its rank-plus-one. The denominator, IDCG, is the same calculation for the perfect ranking. Dividing keeps the score in zero to one. I use NDCG@5 or NDCG@10 as my top-line retrieval metric in production RAG because it captures both ranking quality and graded relevance simultaneously. Its only limitation is that it requires graded labels — if you only have binary relevance, MRR is often easier to interpret."

---

## 27.7 Retrieval metric 5 — Hit Rate (a.k.a. Recall@1 or Recall@k binary)

### The plain-English mental model

Did we retrieve **at least one** relevant document in the top-k? That's it. Yes/no, averaged across queries.

### The formula

```
                     number of queries with ≥1 relevant in top-k
   Hit Rate@k   =   ──────────────────────────────────────────────
                                total number of queries
```

### Worked example

Out of 100 eval queries, you retrieve top-5 for each and check if at least one of those 5 is relevant. Suppose 78 queries have at least one relevant in the top-5.

```
   Hit Rate@5  =  78 / 100  =  0.78
```

### When Hit Rate is the right metric

Hit Rate is useful as a **first-pass health check**. If your Hit Rate@5 is below 0.7, your retrieval is fundamentally broken — most queries can't find any relevant document. That's a "go fix retrieval before doing anything else" signal.

It's a coarse metric. It doesn't distinguish "found one of three relevant" from "found three of three relevant." For finer signals, use Precision@k, Recall@k, or NDCG.

### Interview narrative for Hit Rate

> "Hit Rate@k is the simplest retrieval metric — what fraction of queries find at least one relevant document in the top-k? I use it as a first-pass health check. If Hit Rate@5 is below 0.7, retrieval is fundamentally broken and I need to fix that before tuning anything else. It's coarse — doesn't distinguish 'one of three relevant' from 'three of three relevant' — so I move to Precision and Recall once Hit Rate is healthy."

---

## 27.8 Generation metric 1 — Faithfulness

### The plain-English mental model

Faithfulness asks: **is the LLM's answer actually supported by the retrieved context, or did it hallucinate?** A faithful answer makes only claims that are derivable from the context. An unfaithful answer asserts things the context doesn't support.

This is the most important generation metric for RAG. The whole point of retrieval is to ground the LLM, and faithfulness measures whether that grounding actually happened.

### How it's computed (RAGAS approach)

The standard RAGAS faithfulness metric uses an LLM as a judge in two steps:

1. **Decompose the answer into atomic claims.** Use an LLM to break the answer into a list of standalone statements.
2. **Verify each claim against the context.** For each claim, ask an LLM: "is this claim supported by the context? yes/no."

```
                    number of claims supported by the context
   Faithfulness  =  ──────────────────────────────────────────
                              total number of claims
```

### Worked example

Question: "What are the side effects of Drug X?"
Context: "Drug X may cause headache, nausea, and dizziness. Patients should consult a doctor if symptoms persist."
Answer: "Drug X causes headache, nausea, dizziness, and severe liver damage."

Decompose the answer into claims:
1. Drug X causes headache. → Supported by context. ✓
2. Drug X causes nausea. → Supported by context. ✓
3. Drug X causes dizziness. → Supported by context. ✓
4. Drug X causes severe liver damage. → NOT supported by context. ✗

```
   Faithfulness  =  3 / 4  =  0.75
```

The metric correctly catches that the LLM hallucinated "severe liver damage" — that's not in the context, even though the rest of the answer is grounded.

### Why faithfulness matters

For high-stakes domains — clinical, legal, financial — an unfaithful answer is dangerous. The user trusts the model's output as if it came from the source documents, but it actually contains fabrications. Faithfulness directly measures hallucination rate and is the metric I'd auto-rollback an LLM deployment on.

### What to fix when faithfulness is low

- Stronger prompting: "Answer using only the provided context. If the answer is not in the context, say 'I don't have enough information.'"
- Add a citations requirement: "Quote the specific sentences from the context that support each claim in your answer."
- Stronger LLM (Claude Opus over Haiku, GPT-4 over GPT-3.5)
- Constrained generation: tools like Guardrails-AI or Outlines that force the LLM to only output claims that match the context

### Interview narrative for Faithfulness

> "Faithfulness measures whether the LLM's answer is actually supported by the retrieved context — it's the direct measure of hallucination in RAG. The standard RAGAS approach decomposes the answer into atomic claims using an LLM, then verifies each claim against the context using another LLM call. The score is the fraction of claims that are supported. So if the LLM produces four claims and three are grounded but one is fabricated, faithfulness is 0.75. This is the metric I'd watch most carefully in clinical or legal RAG, where hallucinations can be dangerous. The fix when faithfulness drops is usually a prompt that forces the LLM to cite specific context sentences, or a stronger model."

---

## 27.9 Generation metric 2 — Answer Relevance

### The plain-English mental model

Answer relevance asks: **does the answer actually address the user's question?** A faithful answer can still miss the point — for example, it might be perfectly grounded in the context but answer a tangentially-related question.

### How it's computed (RAGAS approach)

RAGAS measures answer relevance by reverse-engineering — generate questions from the answer and compare them to the original question:

1. **Use an LLM to generate N candidate questions** that the given answer would naturally answer.
2. **Embed the original question and each candidate question.**
3. **Compute mean cosine similarity** between the original question and the candidates.

```
                              1    N
   Answer Relevance  =  ──── × Σ  cosine(q_orig, q_cand_i)
                          N    i=1
```

### Worked example

Question: "What are the side effects of Drug X?"
Answer: "Drug X is manufactured by PharmaCorp and was approved by the FDA in 2018."

Generate candidate questions from the answer:
1. "Who manufactures Drug X?"
2. "When was Drug X approved?"
3. "Which agency approved Drug X?"

Compute cosine similarity of each to "What are the side effects of Drug X?":

```
   sim(orig, q1) ≈ 0.32
   sim(orig, q2) ≈ 0.28
   sim(orig, q3) ≈ 0.30

   Answer Relevance = (0.32 + 0.28 + 0.30) / 3 = 0.30
```

Low score — the answer is faithful (every claim is presumably true) but irrelevant to the question asked. The user wanted side effects; the LLM provided manufacturing details.

### Why answer relevance matters

It catches the failure mode where the LLM avoids the question. This happens when:

- The retrieved context doesn't actually contain the answer, so the LLM substitutes related-but-different information.
- The prompt is over-conservative ("I cannot answer that") and hedges.
- The LLM misinterprets the question.

Faithfulness and Answer Relevance together cover most generation failure modes: faithfulness catches hallucinated content; answer relevance catches off-topic content.

### Interview narrative for Answer Relevance

> "Answer relevance measures whether the answer actually addresses the user's question, regardless of whether it's faithful. The RAGAS approach is clever — it reverse-engineers the metric by using an LLM to generate candidate questions that the answer would naturally answer, then computes cosine similarity between the original question and those candidates in embedding space. If the answer talks about manufacturing details when the user asked about side effects, the candidate questions will look like 'who makes the drug?' which has low similarity to 'what are the side effects?' so the metric correctly flags low relevance. Together with faithfulness it covers most generation failure modes — faithfulness catches hallucinations, answer relevance catches off-topic answers."

---

## 27.10 Generation metric 3 — Context Precision

### The plain-English mental model

Context precision asks: **of the retrieved chunks that the LLM saw, how high in the ranking are the actually-useful ones?** This is similar to retrieval Precision@k but evaluated from the LLM's perspective using a reference answer.

### How it's computed (RAGAS approach)

For each retrieved chunk in order:

1. **Use an LLM judge** to determine whether the chunk is useful for answering the question, given a reference answer.
2. **Compute weighted precision** that gives more weight to relevant chunks at higher ranks.

```
                            k
                            Σ   (Precision@i × v_i)
                           i=1
   Context Precision  =  ──────────────────────────
                          number of relevant chunks
```

where `v_i` is 1 if chunk i is useful and 0 otherwise.

### Worked example

Suppose your retrieval returns 5 chunks. The judge LLM marks them like this:

```
   Rank  Chunk      Useful for answering?
   ────  ──────     ─────────────────────
    1    chunk_42     ✓
    2    chunk_113    ✗
    3    chunk_7      ✓
    4    chunk_201    ✗
    5    chunk_56     ✗
```

Compute Precision@i for each rank where v_i = 1:

```
   Rank 1:  Precision@1 = 1/1 = 1.00,  v_1 = 1  →  contribution 1.00
   Rank 3:  Precision@3 = 2/3 ≈ 0.67,  v_3 = 1  →  contribution 0.67
```

Sum: 1.00 + 0.67 = 1.67. Divide by number of relevant chunks (2):

```
   Context Precision  =  1.67 / 2  =  0.835
```

### Why context precision matters

This metric tells you whether your **reranker is doing its job**. If relevant chunks are buried at rank 5 while irrelevant ones are at rank 1, the LLM has to wade through noise to find the answer, which often hurts faithfulness. High context precision = relevant chunks at the top.

### What to fix when context precision is low

- Add or improve the reranker (cross-encoder over top-k retrieval)
- Better embedding model
- Hybrid search

### Interview narrative for Context Precision

> "Context precision measures how high the actually-useful chunks rank in the retrieved set, evaluated by an LLM judge against a reference answer. Unlike retrieval Precision@k, which uses static labels, context precision uses the LLM judge to decide chunk usefulness in the specific context of the question. The formula weighs higher-ranked relevant chunks more heavily. Low context precision means relevant chunks are buried — the reranker isn't doing its job. The fix is usually adding or strengthening a cross-encoder reranker that scores query-chunk pairs jointly."

---

## 27.11 Generation metric 4 — Context Recall

### The plain-English mental model

Context recall asks: **of all the information in the reference answer, how much was supported by the retrieved chunks?** It's the LLM-based equivalent of retrieval Recall@k but computed against an actual answer rather than document labels.

### How it's computed (RAGAS approach)

1. **Decompose the reference answer into atomic claims.**
2. **For each claim, check whether any of the retrieved chunks supports it.**

```
                          number of claims supported by retrieved context
   Context Recall  =  ────────────────────────────────────────────────────
                              total number of claims in reference answer
```

### Worked example

Reference answer: "Drug X causes headache and nausea. It's contraindicated for pregnant women."

Decompose into claims:
1. Drug X causes headache.
2. Drug X causes nausea.
3. Drug X is contraindicated for pregnant women.

Retrieved context: "Drug X may cause headache, nausea, and dizziness." (no mention of pregnancy contraindication)

Check each claim:

```
   Claim 1 (headache):       supported by context  ✓
   Claim 2 (nausea):         supported by context  ✓
   Claim 3 (pregnancy):      NOT supported          ✗

   Context Recall = 2 / 3 = 0.67
```

The retrieval missed the pregnancy contraindication chunk, so the LLM literally couldn't have answered fully.

### Why context recall matters

Context recall is a hard ceiling on RAG quality from the answer side. If a claim isn't in the retrieved context, the LLM either has to hallucinate it or omit it. Either way, your final answer will be incomplete or unfaithful. Context recall identifies which claims are missing from retrieval.

### What to fix when context recall is low

- Larger k (more chunks retrieved)
- Hybrid search (catch keyword matches that dense retrieval misses)
- Query rewriting / expansion
- Larger or finer chunks (a fact split across chunk boundaries can fail to retrieve)

### Interview narrative for Context Recall

> "Context recall measures how much of the reference answer is actually supported by the retrieved context. RAGAS decomposes the reference answer into atomic claims, then for each claim checks whether any retrieved chunk supports it. The score is the fraction of claims that are supported. If it's low, the retriever is missing relevant content — the LLM either has to hallucinate it or leave it out. Together, context precision and context recall give me a complete picture of retrieval quality at the chunk level, evaluated in the specific context of each question."

---

## 27.12 The four RAGAS metrics together — a decision tree

Once you have all four RAGAS metrics, you can diagnose RAG failures cleanly:

```
                    RAG quality is bad, what's broken?
                                  │
                ┌─────────────────┴──────────────────┐
                ▼                                    ▼
        Faithfulness low?                    Answer relevance low?
                │                                    │
        Yes: hallucinations                Yes: off-topic answers
        Fix: stronger prompt /             Fix: better prompt /
        cite sentences /                   stronger LLM /
        stronger LLM                       check the question is
                                            answerable from context
                │                                    │
                ▼                                    ▼
        Context precision low?              Context recall low?
                │                                    │
        Yes: relevant chunks                Yes: relevant chunks
        ranked too low                      missing entirely
        Fix: add reranker /                 Fix: increase k /
        better embeddings                   hybrid search /
                                            query rewriting
```

This decomposition is powerful because **each failure mode has a different fix**. You don't randomly throw techniques at low overall quality; you measure where the failure is and target it.

---

## 27.13 Other generation metrics worth knowing

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram overlap between the generated answer and a reference answer. Used heavily in machine translation, less so in RAG. Limitation: BLEU rewards surface-form similarity, not meaning. A paraphrased correct answer scores low. Use sparingly.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Variants: ROUGE-N (n-gram overlap), ROUGE-L (longest common subsequence), ROUGE-S (skip-bigram). Used in summarization. Same limitation as BLEU — surface-form, not semantic.

### BERTScore

Computes similarity between generated and reference answers in BERT embedding space. Better at semantic similarity than BLEU/ROUGE because embeddings capture meaning. Three variants: BERTScore-P (precision), BERTScore-R (recall), BERTScore-F1 (harmonic mean). Use this when you have a reference answer and need a single semantic-similarity number.

### LLM-as-Judge

Have a strong LLM (Claude Opus, GPT-4) score the generated answer against a reference or against the question. Common configurations:

- **Pairwise**: "which of these two answers is better, A or B?" (lowest variance)
- **Pointwise**: "rate this answer 1-5 on relevance, accuracy, completeness." (higher variance, anchor with rubrics)
- **Reference-based**: compare to a gold reference answer.
- **Reference-free**: judge the answer alone against the question (faithfulness-style).

Best practices: use a stronger judge than the system you're evaluating. Use rubrics. Sample size matters — for low-variance results, judge 100+ items. Anchor with reference answers when possible.

---

## 27.14 Online evaluation — what you measure in production

Offline evaluation requires a labeled golden set. Online evaluation runs in production traffic, with real users. You need both.

### User feedback signals

- **Thumbs up / thumbs down**: simple, biased toward extremes (only motivated users click), but the most direct signal you can get.
- **Implicit dwell time**: did the user read the answer or close immediately? Useful but noisy.
- **Follow-up questions**: did the user ask a clarifying question, suggesting the original answer was incomplete?
- **Click-through on citations**: are users actually verifying the cited sources?

### LLM-as-judge in production

Sample, say, 1% of traffic and run an LLM-as-judge offline (the next day) against captured (question, retrieved-context, answer) tuples. Track faithfulness, answer relevance, and context metrics over time. Alert on regressions.

### Drift monitoring on the eval distribution

Embed all production queries. Track the distribution over time. If the embedding distribution shifts significantly (PSI > 0.25), your offline eval set is no longer representative — refresh it.

---

## 27.15 Building a golden eval set for RAG

This is the unsung craft of RAG evaluation. Here's the practical pipeline:

```
   ┌──────────────────────┐
   │  Sample real queries │   from production logs (anonymized)
   │  from production     │   or user research
   │  (or generate with   │
   │  LLM if pre-launch)  │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Domain experts label│   for each query, mark which
   │  the corpus           │   documents are relevant
   │                      │   ideally with graded scores (0/1/2)
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Domain experts write│   the gold answer they'd expect
   │  reference answers   │   the LLM to produce
   │                      │
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Quality assurance   │   another expert reviews 10% of
   │  pass on labels      │   labels for inter-rater agreement
   └──────────┬───────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Versioned in Git    │   golden set as code, reviewed
   │  alongside the code  │   in PRs, never silently mutated
   └──────────────────────┘
```

The volume target depends on your domain: 50-200 well-curated queries usually beat 10,000 noisy ones. The reason is signal-to-noise — a few well-chosen edge cases tell you more than thousands of random queries.

### Resume tie-in box

> At ResMed for the clinical chatbot we curated a 200-question golden set with clinical domain experts. Each question had:
> - The question text (drawn from real clinical workflow questions)
> - Labeled relevant clinical sections with graded relevance (0/1/2)
> - A reference answer the chatbot should have produced
>
> We ran the eval pipeline nightly in CI, comparing every PR's RAG output against the golden set on Faithfulness, Answer Relevance, Context Precision, Context Recall. A regression on any metric blocked the PR. The single biggest quality win — moving from naive token-based chunking to section-aware chunking — was driven by Context Precision going from 0.62 to 0.81 on the same eval set.

---

## 27.16 The complete evaluation pipeline diagram

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                  GOLDEN EVAL SET                                     │
   │  (questions, relevant docs with grades, reference answers)           │
   └──────────────────────────────────────┬───────────────────────────────┘
                                          │
                ┌─────────────────────────┴───────────────────────────┐
                ▼                                                     ▼
   ┌────────────────────────────┐                    ┌────────────────────────────┐
   │   RETRIEVAL EVAL           │                    │   GENERATION EVAL          │
   │   (per query)              │                    │   (per query)              │
   │                            │                    │                            │
   │   Run retriever            │                    │   Run full RAG pipeline    │
   │   Get top-k docs           │                    │   Get answer + context     │
   │                            │                    │                            │
   │   Compute:                 │                    │   Compute (LLM-as-judge):  │
   │     • Hit Rate@k           │                    │     • Faithfulness         │
   │     • Precision@k          │                    │     • Answer Relevance     │
   │     • Recall@k             │                    │     • Context Precision    │
   │     • MRR                  │                    │     • Context Recall       │
   │     • NDCG@k               │                    │                            │
   │                            │                    │   Optional:                │
   │                            │                    │     • BERTScore            │
   │                            │                    │     • LLM-as-judge         │
   │                            │                    │       pairwise win-rate    │
   └──────────────┬─────────────┘                    └──────────────┬─────────────┘
                  │                                                 │
                  └─────────────┬───────────────────────────────────┘
                                ▼
                ┌────────────────────────────────────┐
                │   AGGREGATE METRICS DASHBOARD      │
                │   Track every metric over time,    │
                │   alert on regressions, gate PRs   │
                │   on threshold violations.         │
                └────────────────────────────────────┘
```

---

## 27.17 Interview Q&A — full narrative answers

### Q1. How do you evaluate a RAG system end-to-end?

I split it into two halves: retrieval evaluation and generation evaluation, because RAG fails in distinct ways and each failure needs a different metric. For retrieval, I look at Precision@k, Recall@k, MRR, and NDCG. For generation, I use the four RAGAS metrics — faithfulness, answer relevance, context precision, context recall. Each catches a specific failure mode. Faithfulness catches hallucination; answer relevance catches off-topic answers; context precision catches "relevant chunks ranked too low"; context recall catches "relevant chunks missing entirely." I run all of these against a curated golden eval set, ideally 100-200 questions labeled by domain experts. In production I supplement with online thumbs-up-thumbs-down and LLM-as-judge on a sample of real traffic. The goal is that when overall RAG quality drops, the metric decomposition tells me exactly where to look.

### Q2. Walk me through Precision@k.

Precision@k is the fraction of the top-k retrieved documents that are actually relevant. Numerator: number of relevant documents in the top-k. Denominator: k. So if I retrieve 5 documents and 3 are relevant, my Precision@5 is 0.6. The metric matters because it bounds the noise in the LLM's context window — if precision is low, the LLM is wading through irrelevant content to find the answer, which often hurts faithfulness. Low Precision@k typically means the reranker is missing or the embedding model isn't tuned to the domain. The fix is usually a cross-encoder reranker that scores query-document pairs jointly and pushes the most relevant to the top.

### Q3. Walk me through Recall@k.

Recall@k is the fraction of all relevant documents in the corpus that are present in my top-k retrieval. Numerator: number of relevant documents in the top-k. Denominator: total number of relevant documents in the corpus for that query. So if there are 10 relevant documents and my top-5 finds 3 of them, my Recall@5 is 0.3. The metric matters because it's a hard ceiling on RAG quality — if a relevant document isn't in the top-k, the LLM literally cannot ground its answer on it. It will either hallucinate or omit. Low recall typically means dense retrieval alone is missing keyword matches; the fix is hybrid search combining BM25 and dense embeddings via Reciprocal Rank Fusion. In production I usually retrieve a large k from the embedding model for high recall, then rerank tightly to a small k for high precision.

### Q4. Why is there a tradeoff between Precision@k and Recall@k?

As k grows, you retrieve more documents. By definition, retrieving more can only catch more relevant documents, never fewer — so Recall@k goes up. But you also pull in more irrelevant documents, so the fraction that are relevant — Precision@k — goes down. So the two metrics pull in opposite directions: high recall pushes toward large k, high precision pushes toward small k. The way to capture both: retrieve broadly with a cheap model (high k, high recall), then rerank tightly with an expensive cross-encoder (small k, high precision). That two-stage pattern is the production standard.

### Q5. Explain MRR with an example.

MRR — Mean Reciprocal Rank — measures how high the first relevant document ranks, averaged across all eval queries. For each query, take the rank position of the first relevant document, take the reciprocal — so rank 1 gives 1.0, rank 2 gives 0.5, rank 3 gives 0.33. If there's no relevant document in the top-k, the reciprocal rank is 0. Average across all queries to get MRR. Suppose I have four queries with first relevant at ranks 1, 3, 2, and not-found. RR values are 1.0, 0.33, 0.5, 0. MRR is their mean: 1.83 / 4 = 0.46. The metric matters when I care about top-1 quality, like search snippets or LLMs that anchor on the first document. Its limitation is that it only counts the first hit — for finer ranking signal I'd use NDCG.

### Q6. Walk me through NDCG.

NDCG — Normalized Discounted Cumulative Gain — is the most complete single retrieval metric. It captures both relevance and rank position. The first piece is DCG, Discounted Cumulative Gain: sum over the top-k of `relevance / log_2(rank + 1)`. So a relevant document at rank 1 contributes its full relevance, while one at rank 10 is heavily discounted. The second piece is IDCG, the DCG of the perfect ranking — same documents but sorted by relevance descending. NDCG is DCG divided by IDCG, which keeps the score in zero to one. So if my ranking is perfect, NDCG = 1; if it's terrible, NDCG approaches 0. Worked example: suppose I retrieve five documents with graded relevance 2, 0, 1, 2, 0. DCG = 2/1 + 0/1.585 + 1/2 + 2/2.322 + 0/2.585 ≈ 3.36. IDCG with sorted relevance 2, 2, 1, 0, 0 ≈ 3.76. NDCG@5 = 3.36 / 3.76 = 0.89. So my ranking is about 89% of ideal. NDCG is what I'd reach for as my top-line retrieval metric in production.

### Q7. Explain Faithfulness — the most important RAGAS metric.

Faithfulness is the direct measure of hallucination in RAG. The standard RAGAS approach decomposes the answer into atomic claims using an LLM, then verifies each claim against the retrieved context using another LLM call. The score is the fraction of claims supported. So if the LLM produces four claims and three are grounded in the context but one is fabricated, faithfulness is 0.75. This is the metric I'd watch most carefully in clinical or legal RAG, where hallucinations are dangerous. The fix when faithfulness drops is usually a prompt that forces the LLM to cite specific context sentences, or moving to a stronger LLM that's better at staying grounded. I'd auto-rollback an LLM deployment on a faithfulness regression.

### Q8. How is Answer Relevance computed?

The RAGAS approach is a clever reverse-engineering. Step one: use an LLM to generate N candidate questions that the given answer would naturally answer. Step two: embed the original user question and each candidate question. Step three: compute the mean cosine similarity between the original and candidates. The intuition: if the answer is on-topic, the candidate questions should look very similar to the original question; if the answer is off-topic, the candidate questions look different. So an answer about manufacturing details when the user asked about side effects produces candidate questions like "who makes the drug?" — low similarity to "what are the side effects?" — flagging low answer relevance.

### Q9. What's the difference between Context Precision and Context Recall?

Both are RAGAS metrics about retrieval quality, but evaluated from the LLM's answering perspective rather than against static document labels. Context Precision asks: of the chunks the LLM saw, how high in the ranking are the actually-useful ones? It rewards relevant chunks at the top of the retrieved list. Context Recall asks: of all the claims in the reference answer, how many are supported by any retrieved chunk? It rewards completeness of the retrieval. So if relevant chunks are at the top, context precision is high. If all the necessary information is somewhere in the retrieved set (even if buried), context recall is high. Together they decompose retrieval failures: low context precision means the reranker is broken; low context recall means retrieval is missing chunks entirely.

### Q10. How would you build a golden eval set for RAG from scratch?

Five steps. First, sample real queries from production logs if you have any, or generate with an LLM seeded by domain experts if you're pre-launch. Second, have domain experts label which documents in the corpus are relevant to each query, ideally with graded scores like 0/1/2 rather than binary, because that enables NDCG. Third, have those same experts write reference answers — the gold-standard response the RAG should produce. Fourth, do a quality assurance pass where another expert reviews 10% of the labels for inter-rater agreement — if agreement is below 80% your labeling guidelines are too vague. Fifth, version the golden set in Git alongside the code, never silently mutate it, and wire it into CI so every PR runs against the same eval. Volume target: 50-200 well-curated queries usually beat 10,000 noisy ones. The signal-to-noise ratio matters more than the raw count.

### Q11. What's LLM-as-judge and what are its limitations?

LLM-as-judge is using a strong LLM to score the output of a system. Common configurations: pairwise comparisons (which of these two answers is better?) which are lowest-variance; pointwise scoring (rate this 1-5 on accuracy) which is higher-variance and needs rubrics; reference-based (compare to a gold answer) versus reference-free (judge the answer alone). Best practices: use a stronger judge than the system being evaluated, use detailed rubrics, sample at least 100 items for low-variance results, anchor with reference answers when possible. Limitations: the judge has its own biases, position bias in pairwise comparisons (often slight preference for the first option, mitigate by randomizing), prompt sensitivity, cost (judging 1000 queries with Claude Opus isn't free). I'd never rely on LLM-as-judge alone — combine it with human spot-checks and structural metrics like RAGAS.

### Q12. How do you connect online and offline RAG evaluation?

Offline evaluation is the lab — fixed eval set, deterministic metrics, run in CI. Online evaluation is the field — real users, real distributions, noisier signals. Both are necessary. The bridge is: monitor that the embedding distribution of production queries matches the embedding distribution of the eval set. If production queries drift away from eval coverage (PSI > 0.25 in embedding space), the offline eval is no longer representative and I need to refresh it with sampled production queries. I also run LLM-as-judge on a sample of production traffic — say 1% — and compare those scores to offline scores. Significant divergence means either the eval set is out-of-date or the offline eval is over-fitting to specific question types.

### Q13. What's the single most common mistake in RAG evaluation?

Eval-set rot. Teams build a golden set early, the system improves, the team starts gaming the golden set instead of measuring genuine quality. Or production query distributions shift and the eval set stops matching reality. The fix: refresh the eval set quarterly with production-sampled queries, and track separately which questions are "challenging" (system gets wrong) versus "easy" (system gets right). If 90% of your eval is easy questions, your metric numbers are inflated and you're not learning. The principle: an eval set is a living dataset that needs maintenance, not a fixture you build once.

---

## 27.18 Cheatsheet — formulas in one place

```
   PRECISION@k     =  (relevant in top-k) / k

   RECALL@k        =  (relevant in top-k) / (total relevant in corpus)

   MRR             =  (1/|Q|) × Σ (1 / rank_of_first_relevant_i)

   DCG@k           =  Σ_i  rel_i / log_2(i + 1)
   IDCG@k          =  DCG@k computed for the perfect ranking
   NDCG@k          =  DCG@k / IDCG@k

   HIT RATE@k      =  (queries with ≥1 relevant in top-k) / |Q|

   FAITHFULNESS    =  (claims supported by context) / (total claims in answer)

   ANSWER RELEV.   =  mean cosine(orig_q, generated_question_from_answer_i)

   CONTEXT PREC.   =  Σ_i (Precision@i × v_i) / (relevant chunks)
                      where v_i = 1 if chunk_i useful else 0

   CONTEXT RECALL  =  (claims supported by retrieved context) /
                      (total claims in reference answer)
```

---

## 27.19 The interview-ready closing summary

> "RAG evaluation has two halves. For retrieval I use Precision@k for context-window cleanliness, Recall@k as a hard ceiling on what's even possible, MRR when first-hit matters, and NDCG as my top-line ranked-retrieval metric. For generation I use the four RAGAS metrics — faithfulness for hallucination, answer relevance for on-topic, context precision for chunk ranking, context recall for chunk completeness. Each metric isolates a specific failure mode, so when overall RAG quality drops, the metric decomposition tells me exactly where to look. I run all of these against a curated golden set in CI on every PR, and supplement with LLM-as-judge on a 1% sample of production traffic. The principle: don't tune RAG without measuring; don't optimize one metric without checking the others."

---

End of Chapter 27. Continue back to **[Chapter 00 — Master Index](00_index.md)**.
