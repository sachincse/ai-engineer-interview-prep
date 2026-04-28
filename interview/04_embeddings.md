# Chapter 04 — Embedding Models

> **Why this chapter matters:** Almost every RAG-flavored interview question routes through embeddings. "How does an embedding model get trained?" "Why does cosine similarity work?" "When would you choose BGE over OpenAI?" These are not trivia — they test whether you understand the layer that makes semantic search work. Sachin's ResMed RAG chatbot lived or died on embedding quality, and his pgVector usage in production is the natural tie-in throughout this chapter.

---

## 4.1 What an embedding model actually is

### The plain-English mental model

An embedding model is a function that takes a piece of text and returns a list of numbers — a vector. The magic is that this function is trained so that **texts with similar meaning produce nearby vectors**. "I love NLP" and "Natural language is my passion" come out as vectors with high cosine similarity, even though they share zero words in common. "I love NLP" and "I hate this code" come out as vectors that are far apart, even though they share three out of four words.

That property — **semantic proximity in vector space** — is the foundation of every RAG system, every semantic search, every clustering of documents. Without it, you can only do exact keyword search. With it, you can answer "find me documents about cycling training plans" with a corpus that never uses the word "cycling" but talks about "indoor riding" and "FTP intervals."

### The block diagram

```
   ┌─────────────────────┐
   │  "I love NLP"       │   raw text
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Tokenizer          │   split into tokens (subwords)
   │  (BPE / WordPiece)  │   ["I", "love", "NL", "P"]
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Token Embeddings   │   each token → 768-dim vector
   │  (lookup matrix)    │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Transformer        │   contextualize tokens through
   │  Encoder            │   self-attention layers
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Pooling            │   reduce sequence to one vector
   │  ([CLS], mean,      │   (most common: mean pooling
   │   or last-token)    │    of last hidden state)
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Optional:          │
   │  Normalize to       │   so cosine = dot product
   │  unit length        │
   └──────────┬──────────┘
              │
              ▼
       [0.12, -0.43, 0.77, ..., -0.11]   final embedding (e.g. d=768)
```

The transformer encoder gives you contextualized token representations — the vector for "bank" depends on whether the surrounding text is about rivers or finance. Pooling collapses the sequence into a single vector. Mean pooling is the most common — average the last hidden states across all tokens. Some models use the [CLS] token's representation; some use the last token's; the choice depends on how the model was trained.

### Why cosine similarity?

Once you have unit-normalized vectors, the cosine of the angle between them is just their dot product. Cosine ranges from -1 (opposite meaning) through 0 (unrelated) to 1 (identical meaning). It works because semantically similar texts produce vectors pointing in similar directions in the high-dimensional space. We don't care about magnitude — we care about direction. Two texts can both be "long" or both be "short" and still mean very different things, so we throw away magnitude and keep only direction.

The math: cosine_similarity(A, B) = (A · B) / (||A|| · ||B||). For unit vectors, the denominator is 1, so cosine equals dot product. This is why production systems normalize embeddings on the way into the vector store — once normalized, you can use the fastest possible distance metric: dot product.

---

## 4.2 How embedding models get trained — the contrastive learning story

### The core idea

You can't train an embedding model with a regression loss because there's no "correct vector" for a text. Instead, we use **contrastive learning**: train the model so that similar pairs get close together and dissimilar pairs get pushed apart. The model only ever sees pairs (or triplets) — it never sees absolute targets.

```
   Anchor:    "I love NLP"
   Positive:  "Natural language is my passion"   ← should be CLOSE
   Negative:  "Database indexing strategies"     ← should be FAR
```

The training objective pulls anchor-positive pairs together and pushes anchor-negative pairs apart, all simultaneously across batches of millions of triplets.

### Triplet loss — the simplest contrastive loss

For an anchor a, positive p, and negative n, the triplet loss is:

```
L = max(0, ||a - p||² - ||a - n||² + margin)
```

In plain English: penalize the model whenever the distance from anchor to positive is not at least `margin` smaller than the distance from anchor to negative. The margin (typically 0.2 to 1.0) prevents the trivial solution where everything collapses to the same point.

A worked numeric example. Suppose `||a - p||² = 0.3` and `||a - n||² = 0.5`, with margin = 0.5. Then loss = max(0, 0.3 - 0.5 + 0.5) = 0.3. The model gets penalized because the positive (distance 0.3) and negative (distance 0.5) aren't separated by the full margin yet. As training progresses, the model adjusts to push the negative further away.

### InfoNCE loss — the modern standard

Triplet loss uses one positive and one negative per anchor. InfoNCE — Information Noise Contrastive Estimation — uses one positive and many negatives, treating the problem as a classification task: among (positive + N negatives), which one is the true positive?

```
L = -log( exp(sim(a, p) / τ) / Σᵢ exp(sim(a, xᵢ) / τ) )
```

where the denominator sums over the positive and all negatives, and τ is a temperature parameter (typically 0.05 to 0.1). The intuition: maximize the probability the model assigns to the true positive among all candidates. Lower temperature makes the loss sharper — the model has to be very confident about the positive.

### Where do training pairs come from?

This is the secret sauce. The big embedding models — BGE, E5, OpenAI's text-embedding-3 — are trained on massive datasets of pairs gathered from the open web. Common sources:

- **(query, document) pairs from search engines** — the document the user clicked is the positive.
- **(question, answer) pairs** from forums like StackOverflow and Reddit.
- **(title, abstract) pairs** from scientific papers.
- **(paragraph_a, paragraph_b)** pairs from the same document — semantically related by document context.
- **Cross-lingual pairs** — same text in different languages — for multilingual embeddings.
- **Synthetically generated pairs** — an LLM rewrites a passage in a different style; the original and rewrite are a positive pair.

The biggest models are trained on hundreds of millions to billions of such pairs, often with multi-stage pipelines (weak supervision pretraining → high-quality supervised fine-tuning → task-specific instruction tuning).

### Hard negative mining — the quality lever

Random negatives are too easy. If your anchor is about cycling and your random negative is about quantum physics, the model learns nothing — they were already obviously different. The technique that improves embedding models most is **hard negative mining**: for each anchor, find negatives that are *almost* relevant but not quite. For an anchor "best indoor cycling trainer," a hard negative might be "best running treadmill review" — same shape of query, different domain. Hard negatives force the model to learn fine-grained distinctions.

In practice, hard negatives come from:
- **In-batch negatives**: other examples in the same training batch.
- **Mined negatives**: documents that were retrieved by the current model but labeled as not relevant.
- **Adversarial negatives**: paraphrases generated by an LLM that flip the meaning.

---

## 4.3 The model landscape — what to use when

### The major open-source families

```
   ┌──────────────────────────────────────────────────────────────┐
   │             Open-source embedding model families             │
   ├──────────────────────────────────────────────────────────────┤
   │                                                              │
   │  Sentence-BERT (SBERT) — UKP Lab, the original family.       │
   │   Pre-2023 era. Models like all-MiniLM-L6-v2, all-mpnet.     │
   │   Still good baselines.                                      │
   │                                                              │
   │  BGE (BAAI General Embedding) — Beijing Academy of AI.       │
   │   Strong English + Chinese performance. bge-large-en-v1.5    │
   │   was state of the art for much of 2023-2024.                │
   │                                                              │
   │  E5 (Embedding from bidirEctional Encoder rEpresentations)   │
   │   — Microsoft. e5-large-v2, multilingual-e5-large.           │
   │   Strong general-purpose performance.                        │
   │                                                              │
   │  Nomic / Jina — embeddings designed for long context         │
   │   (8K+ tokens) and multimodal (text+image).                  │
   │                                                              │
   │  Voyage / Cohere — commercial, often best-in-class on        │
   │   benchmark leaderboards but you pay per token.              │
   │                                                              │
   │  OpenAI text-embedding-3 — small (1536d) and large (3072d).  │
   │   Strong, consistent, expensive. Good default for hosted     │
   │   workloads.                                                 │
   └──────────────────────────────────────────────────────────────┘
```

### The choice matrix

| Concern | Recommendation |
|---------|----------------|
| English-only, self-hosted, max quality | `bge-large-en-v1.5` or `e5-large-v2` |
| Multilingual (Arabic, Hindi, etc.) | `multilingual-e5-large` or `bge-m3` |
| Hosted API, no infra to manage | OpenAI `text-embedding-3-large` or Voyage |
| Long context (4K-32K tokens) | `nomic-embed-text-v1.5` or Jina embeddings v3 |
| Tight latency budget, low-resource | `all-MiniLM-L6-v2` (384d, very fast) |
| Code search | `voyage-code-2`, OpenAI embedding 3 (with code prompts) |
| Domain-specific (clinical, legal, finance) | Fine-tune a base model on domain pairs — see §4.5 |

### Dimensions and storage

The dimension d is a critical decision because storage scales linearly with d. For one million documents:

| Model | d | Storage (FP32) | Storage (FP16) |
|-------|---|----------------|----------------|
| MiniLM | 384 | 1.5 GB | 0.75 GB |
| BGE-large | 1024 | 4 GB | 2 GB |
| E5-large | 1024 | 4 GB | 2 GB |
| OpenAI text-embedding-3-large | 3072 | 12 GB | 6 GB |

With product quantization (Chapter 8) you can compress 8-32x further, but you pay in recall. The dimensional trade-off: higher d generally improves retrieval quality up to a point, but indexing and querying become slower. Pick d to match your corpus size and latency budget.

---

## 4.4 Matryoshka representation learning — the variable-dimension trick

### The motivation

Suppose you trained a 1024-dim embedding model, but your production system only has memory for 256 dims. Naively truncating to the first 256 dims usually destroys quality, because the model never learned that the first 256 dims should be self-sufficient.

Matryoshka Representation Learning, named after Russian nesting dolls, trains the model so that **truncating to any prefix of the vector still yields a good embedding**. The first 64 dims encode the coarsest semantic distinctions, the next 64 add finer granularity, and so on up to the full dimension.

### How it works

During training, the loss is computed at multiple dimensions simultaneously — typically [64, 128, 256, 512, 768, 1024] — and the gradients flow back through all of them. The model learns that the first k dims, for any k in that list, must be a usable embedding. The result: at inference time you can truncate the vector to fit your memory budget without retraining.

```
   Full embedding:     [0.12, -0.43, 0.77, 0.31, -0.05, ..., -0.11]   1024 dims
                       ←──────64──────→
                       ←──────────128──────────→
                       ←──────────────256──────────────→
                       ←──────────────────────512──────────────────→
                       ←──────────────────────────────1024──────────────────────────────→

   Each prefix is a self-sufficient embedding with monotonically improving quality.
```

OpenAI's `text-embedding-3-large` is Matryoshka-trained — you can truncate to 256 dims with a 30 percent quality drop instead of the 80 percent drop you'd get with a non-Matryoshka model.

### Why this matters in production

1. **Tiered storage**: store full 1024 dims in cold storage for re-ranking, but only top-256 in hot Redis cache for first-pass retrieval.
2. **Adaptive precision**: high-traffic users get 256d retrieval; premium users get 1024d retrieval, both using the same index.
3. **Quantization**: combining Matryoshka truncation with PQ compression gives you 50-100x storage savings with manageable quality loss.

---

## 4.5 Fine-tuning embedding models for your domain

### When fine-tuning is worth it

Off-the-shelf embeddings are great for general English. They struggle on:

- **Specialized vocabulary**: clinical terms (ResMed!), legal citations, financial instruments.
- **Domain-specific similarity**: in clinical context, "stage III" and "stage IV" cancer should be similar but distinct; a generic model collapses them as "cancer staging."
- **Style/tone matching**: matching customer support tickets across paraphrases.

Sachin's ResMed RAG chatbot was a textbook fine-tuning use case. Generic embeddings on raw clinical text gave context precision around 60%; even simple section-aware chunking moved that to 80%. A domain-fine-tuned embedding could plausibly push it past 90%.

### The fine-tuning pipeline

```
   ┌────────────────┐
   │  Base model    │   e.g. bge-large-en-v1.5
   │  (pretrained)  │
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │  Domain pairs  │   (query, relevant_doc) pairs from
   │  collection    │   logs, expert labels, synthetic data
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │  Hard negative │   for each query, mine negatives from
   │  mining        │   the corpus that look-similar-but-aren't
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │  Contrastive   │   InfoNCE loss with mined negatives,
   │  fine-tuning   │   typically 1-3 epochs at small LR
   └───────┬────────┘
           │
           ▼
   ┌────────────────┐
   │  Eval on held- │   measure retrieval@k and MRR on a
   │  out queries   │   golden eval set BEFORE deploying
   └────────────────┘
```

Tooling: `sentence-transformers` library has great fine-tuning utilities. For larger fine-tunes, MTEB benchmark scripts and the HuggingFace ecosystem.

A key pitfall: fine-tuning on too-small a dataset overfits and breaks the general-domain capability. Always measure both domain quality and general-domain regression before deploying.

---

## 4.6 ColBERT — late interaction for high-precision retrieval

### Why ColBERT is different

Standard "bi-encoder" embedding models reduce a document to **one vector**. ColBERT (Contextualized Late Interaction over BERT) keeps **one vector per token**. So a 100-token document produces a (100 × 128) tensor, not a 768-dim vector.

```
   Standard bi-encoder:
   "The cat sat"  →  [0.1, 0.5, ..., -0.2]   one 768-dim vector

   ColBERT:
   "The cat sat"  →  ┌─────────────────────┐
                     │ The   [0.1, ..., 0.4] │
                     │ cat   [0.3, ...,-0.1] │
                     │ sat   [0.2, ..., 0.5] │   100×128 matrix
                     │ ...                   │
                     └─────────────────────┘
```

At query time, ColBERT computes a **MaxSim** score: for each query token, find the most similar document token, sum up those max similarities. This "late interaction" gives much finer-grained matching — you can find a document because of one specific token-level match, not just because the document gist is similar.

### The trade-off

ColBERT is dramatically more accurate for fine-grained retrieval — typically 10-20% better than bi-encoders on TREC and MS MARCO benchmarks. It's also dramatically more expensive in storage (100x more vectors per document) and slower at query time (MaxSim across all tokens). For most production systems the bi-encoder + reranker combo gets you 90% of ColBERT's quality at 1% of the cost.

When ColBERT wins: small, high-precision corpora where re-ranker latency is unacceptable. Large legal databases. Enterprise search where every millisecond of recall matters.

---

## 4.7 Embedding model gotchas in production

### 1. Embedding model versioning is real

If you upgrade your embedding model from BGE-v1 to BGE-v1.5, **all your stored embeddings are now stale** — they're in the old model's vector space, and queries from the new model will retrieve garbage. You must re-embed your entire corpus on every model upgrade. For 100M documents, that's a non-trivial batch job.

The mitigation: version your embeddings explicitly in the database. `embeddings_bge_v1`, `embeddings_bge_v1_5`. Run new and old versions in parallel during transition. Use feature flags to switch query traffic.

### 2. Asymmetric query/document embeddings

Some models — notably E5 and BGE — were trained with explicit prefixes: queries get prepended with `"query: "` and documents with `"passage: "`. Forgetting these prefixes at inference time degrades retrieval quality silently by 10-20%. Always check the model card.

```python
# CORRECT for E5
query_emb = model.encode("query: best indoor trainer")
doc_emb = model.encode("passage: The MyWhoosh smart trainer ...")

# WRONG — same model, no prefixes — will silently underperform
query_emb = model.encode("best indoor trainer")
doc_emb = model.encode("The MyWhoosh smart trainer ...")
```

### 3. Tokenization mismatches in non-English text

Generic English-trained embedding models have shrunken vocabularies for non-Latin scripts. Arabic, Chinese, and Korean queries get split into many more tokens, often hitting the model's max-token limit (typically 512) on short text. For Avrioc's Comera and Labaiik (multilingual UAE products), `multilingual-e5-large` or `bge-m3` is the right choice.

### 4. Mean pooling vs CLS pooling

The model card tells you which pooling strategy was used during training. Using the wrong one at inference time silently degrades quality. Sentence-Transformers handles this automatically via `model.encode()` — if you're calling the underlying transformer directly with `model(input)`, you have to apply the correct pooling yourself.

### 5. Batch normalization quirks

Embedding models trained with InfoNCE-style losses are normalized to unit length by design. If you store unnormalized embeddings, your "cosine similarity" computation becomes meaningless. Always normalize at the boundary — at write time before storing, and at query time before search.

---

## 4.8 Resume tie-in box

> Sachin's ResMed RAG-powered clinical chatbot used pgVector for storage and retrieval. The narrative for the interview should hit these points:
>
> 1. **The choice of pgVector**: "We were already on Postgres for transactional data, so adding pgVector kept the operational surface area small — no new infra to maintain."
> 2. **The embedding model**: "Started with a generic open-source bi-encoder, eventually moved to a domain-aware fine-tune trained on (query, retrieved_clinical_section) pairs from production logs."
> 3. **Section-aware chunking**: "The single biggest quality win was respecting clinical section boundaries — Findings, Impressions, Recommendations — rather than naive token-based chunking. Pushed context precision from ~60% to over 80%."
> 4. **The retrieval pipeline**: "pgVector for first-pass retrieval (top-20), then a cross-encoder reranker for top-5 final selection. Faithfulness and answer-relevance metrics in RAGAS dictated when to add stages."
> 5. **Versioning**: "We versioned embeddings in the database explicitly because we knew model upgrades would mean re-embedding the whole corpus."

---

## 4.9 Interview Q&A with full narrative answers

### Q. How does an embedding model get trained?

The training is contrastive. The model never sees a "correct embedding" for a text — instead, it sees pairs of texts and learns to push similar pairs together while pulling dissimilar pairs apart in the vector space. The training data is enormous — hundreds of millions of (query, document) pairs harvested from search logs, (question, answer) pairs from QA forums, (title, abstract) pairs from scientific papers, and so on. The loss function is typically InfoNCE, which treats each batch as a classification problem: among one positive and many negatives, identify the true positive. Quality comes from hard negative mining, where the model is shown negatives that are almost-but-not-quite relevant, forcing it to learn fine-grained distinctions. Modern models like BGE and E5 use multi-stage pipelines — weak supervision pretraining followed by high-quality supervised fine-tuning followed by instruction tuning for specific tasks like retrieval versus classification.

### Q. Why does cosine similarity work for embeddings?

Because embedding models are trained so that semantically similar texts produce vectors pointing in similar directions in the high-dimensional space. We don't care about magnitude — two texts can both be long or both short and still mean very different things — we care only about direction. Cosine similarity measures exactly that, the cosine of the angle between two vectors. If we normalize the vectors to unit length first, cosine similarity becomes simply the dot product, which is the fastest possible distance metric on modern GPUs. Production systems normalize embeddings on the way into the vector store specifically so they can use dot product at query time.

### Q. When would you fine-tune an embedding model versus using off-the-shelf?

Off-the-shelf models are great for general English content. They struggle on three specific problem shapes. First, specialized vocabulary — clinical text, legal citations, finance — where domain terms aren't well represented in general training data. Second, domain-specific similarity — in clinical context, "stage III" and "stage IV" cancer should be related but distinct, but a generic model often collapses them as "cancer staging." Third, style or tone matching, where you want to match across paraphrases that share semantic content but differ in surface form. The fine-tuning recipe is contrastive on domain pairs from production logs, with hard negative mining from the corpus, evaluated against a held-out set so you catch general-domain regressions. At ResMed, we considered domain fine-tuning seriously — the chunking improvements got us most of the way there, but a domain-tuned embedding would have been the next step.

### Q. What is Matryoshka representation learning?

Matryoshka, named after Russian nesting dolls, is a training technique that teaches an embedding model to be useful at multiple dimensions simultaneously. During training, the loss is computed at several truncation lengths — say 64, 128, 256, 512, 1024 dimensions — and gradients flow back through all of them. The result is that you can truncate the vector to any of those prefix lengths at inference time without retraining, and the truncated version is still a usable embedding. OpenAI's text-embedding-3 is Matryoshka-trained, which is why you can configure `dimensions=256` on their API and lose only 30% quality instead of the 80% you'd get from naive truncation. In production this enables tiered storage strategies — hot Redis cache holds 256-dim embeddings for first-pass retrieval, cold storage holds the full 1024-dim for reranking.

### Q. How do you decide between bi-encoders and ColBERT?

Bi-encoders give you one vector per document and one vector per query, then compare them with cosine similarity — fast, scalable to billions of documents, but they reduce the document to a single point. ColBERT keeps one vector per token, then computes MaxSim at query time — for each query token, find the most similar document token, sum the max similarities. ColBERT is much more accurate for fine-grained matching — typically 10-20% better on standard benchmarks — but it costs 100x more in storage and is slower at query time. The pragmatic production choice is bi-encoder for first-pass retrieval (top-20 from millions of documents) plus a cross-encoder reranker for the top-5 (does the slow but accurate matching only on a handful of candidates). That bi-encoder + reranker stack gets you most of ColBERT's quality at 1% of its cost.

### Q. What would you check first if RAG retrieval quality drops in production?

Five things in order. First, has the corpus changed — are you indexing new documents that weren't part of your eval set? Second, has the query distribution changed — is there a new product launch sending queries the model wasn't tuned for? Third, has the embedding model itself been silently upgraded — vendors push minor version bumps that change the vector space, breaking previously-stored embeddings. Fourth, is there a tokenization issue with non-English queries — Arabic queries hitting English-trained models can lose meaning fast. Fifth, has the chunking changed upstream — even small changes to chunk boundaries can shift retrieval quality by 10%. The production hygiene that catches all five is offline RAGAS evaluation on a frozen golden set, run nightly, alerting on any metric regression.

---

End of Chapter 04. Continue to **[Chapter 05 — Parameter Tuning](05_parameter_tuning.md)** or back to **[Chapter 00 — Master Index](00_index.md)**.
