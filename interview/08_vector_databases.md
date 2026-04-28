# Chapter 08 — Vector Databases & Search Indexes
## HNSW, IVF, IVF-PQ explained from first principles, plus pgVector vs Pinecone vs Qdrant vs Weaviate vs Milvus vs FAISS

> Sachin used pgVector at ResMed for the clinical RAG. This chapter is the depth behind that bullet — the algorithms, the cost models, the tradeoffs, and how to defend the architecture choice in front of a senior interviewer.

---

## 8.1 Why exact search doesn't scale — the math problem RAG hits at 1M vectors

You have N documents, each represented as a vector in d dimensions (typically 768, 1024, or 1536). A query arrives as another d-dim vector. You want the top-10 most similar documents under cosine similarity.

The exact algorithm: compute similarity between the query and every document, sort, take the top-10.

```
Cost per query = N × d × 8 bytes (FP64)  for the dot product
              + O(N log N) for the sort
```

Worked example with N=100M, d=1024:

```
Memory accessed per query: 100M × 1024 × 4 bytes (FP32) = 400 GB
Compute per query:         100M × 1024 = 102 billion multiply-adds
On a modern CPU:           ~10 seconds per query (memory-bandwidth bound)
On a GPU:                  ~2 seconds per query
At 100 QPS:                impossible without 100x parallel hardware
```

This is why exact search is dead at production scale. We trade *exactness* for *log-scale latency* using Approximate Nearest Neighbor (ANN) indexes.

### The ANN bargain

ANN indexes accept some recall loss (typically 0.95 to 0.99 instead of 1.0) in exchange for orders of magnitude faster queries. A 100M vector ANN search runs in 2–10 ms with recall@10 around 0.95. That's 1000× faster than exact, with 5% miss rate.

For RAG, this is essentially free — the LLM doesn't notice the difference between rank-1 and rank-2 in retrieval, especially after reranking. ANN's approximation is below the noise floor of downstream tasks.

> **How to say this in an interview:** "Exact search is O(N times d) per query — 100 million vectors at 1024 dimensions means 400 GB of memory bandwidth per query, which is impossibly slow at production rates. ANN indexes trade a few percent of recall for log-scale query latency. The recall loss is usually well below the noise floor of downstream LLM tasks, so it's effectively free in practice. The art is choosing the right ANN algorithm for your scale and memory budget."

---

## 8.2 The three ANN index families — HNSW, IVF, PQ

Every modern vector database is built on one or more of these algorithms (or a hybrid). Understanding them matters because tuning them is a real production responsibility.

### 8.2.1 HNSW — Hierarchical Navigable Small World

HNSW is the most popular ANN index for the 1M to 100M scale. It builds a multi-layer graph where the top layers are sparse (long-distance jumps) and the bottom layer is dense (short-distance refinement).

#### The picture

```
Layer 2 (sparse, long edges):
        A ────────────── B ────────────── C
        │                │                │
        ▼                ▼                ▼
Layer 1 (intermediate):
        A ─── B ─── C ─── D ─── E ─── F
        │     │     │     │     │     │
        ▼     ▼     ▼     ▼     ▼     ▼
Layer 0 (dense, all vectors):
        A ─ B ─ C ─ D ─ E ─ F ─ G ─ H ─ I ─ J ─ K ...
        (every vector connected to its nearest neighbors)
```

Each vector is randomly assigned a maximum layer L during insertion, with probability decaying geometrically — so layer 0 has all vectors, layer 1 has ~1/M-th, layer 2 has 1/M^2, etc.

#### How search works

```
1. Start at top layer at an entry point.
2. Greedy descent: at the current layer, jump to the neighbor closest
   to the query, repeat until no neighbor is closer.
3. Drop to the next layer at the current node, repeat greedy descent.
4. At layer 0, do a beam search with width ef_search:
   maintain a candidate set of size ef_search, expand best, return top-K.
```

#### Worked search trace

Imagine searching for query Q in a graph with 1M vectors.

```
Layer 2 (5 nodes): start at random node, greedy hop to closest. ~3 hops.
Layer 1 (50 nodes): drop down to current node's layer-1 self, greedy hop. ~5 hops.
Layer 0 (1M nodes): drop down, beam search with ef_search=64. ~50 distance computations.

Total distance computations: ~60. Versus 1M for exact. 16,000x speedup.
Recall: typically 0.95 at this setting.
```

The hierarchy is the magic — top layers give long-distance jumps that get you in the right neighborhood fast; bottom layer does fine-grained refinement.

#### Key parameters

| Parameter | What it controls | Typical | Effect of increase |
|-----------|------------------|---------|--------------------|
| **M** | Max neighbors per node | 16–32 | ↑ recall, ↑ memory, ↑ build time |
| **ef_construction** | Candidate list size during build | 100–500 | ↑ recall, ↑ build time |
| **ef_search** | Candidate list size during query | 64–512 | ↑ recall, ↑ latency |

Tuning rule of thumb: start with M=16, ef_construction=200, ef_search=64. If recall is low, bump ef_search to 128, then 256. If still low, bump M to 32 (rebuild required). The biggest single recall lever is ef_search.

#### Memory cost

```
Per vector: d × 4 bytes (FP32 vector) + M × 8 bytes per layer (graph edges)
For d=1024, M=16, average 1.5 layers per vector:
  4096 + 16 × 8 × 1.5 = 4288 bytes ≈ 4.2 KB per vector
At 100M vectors: ~420 GB. Must fit in RAM for low-latency.
```

This is why HNSW alone struggles past ~100M vectors — memory becomes prohibitive.

#### When to use HNSW

- 1M to 100M vectors with sufficient RAM
- When recall and latency both matter and budget allows
- Default for most modern vector DBs (pgVector 0.5+, Qdrant, Weaviate, Milvus)

> **How to say this in an interview:** "HNSW is a multi-layer graph. The top layers are sparse with long-distance edges; the bottom layer has all vectors with short edges. Search is greedy descent — at each layer, hop to the neighbor closest to the query, drop down a layer when no closer neighbor exists, and beam-search the bottom layer with width ef_search. The hierarchy gives log-scale lookup. Key parameters are M for max neighbors per node — 16 to 32 is typical — ef_construction for build-time candidate list, and ef_search for query-time beam width. ef_search is the main runtime knob trading recall for latency."

### 8.2.2 IVF — Inverted File Index

IVF predates HNSW and uses a different idea: cluster the vectors first, then only search within the most relevant clusters.

#### The picture

```
Step 1: Cluster all vectors into nlist centroids using k-means.

       ┌─────────────────────────────────┐
       │   ●                             │
       │ ● ●  cluster 1     cluster 2 ●  │
       │   ●               ●●●●●  ●      │
       │                                  │
       │              ●●●                │
       │  cluster 3  ●●●●●               │
       │             ●●●                 │
       └─────────────────────────────────┘

Step 2: Each vector is assigned to its nearest centroid.

       cluster 1: [v_1, v_5, v_8, v_42, ...]
       cluster 2: [v_3, v_19, v_77, ...]
       cluster 3: [v_2, v_4, v_11, ...]

Step 3: At query time:
       - Compute distance from query to all nlist centroids
       - Pick top nprobe nearest clusters
       - Search exhaustively within those clusters
       - Return top-K overall
```

#### Worked example

100M vectors, nlist = √100M ≈ 10,000 (rule of thumb).

- Each cluster contains ~10K vectors on average.
- nprobe = 10 → search 10 clusters → 100K vectors examined per query.
- Distance to centroids: 10K computations.
- Total: ~110K distance computations vs 100M exact. 900× speedup.
- Recall: typically 0.85–0.95 depending on nprobe.

#### Key parameters

| Parameter | What it controls | Typical | Effect |
|-----------|------------------|---------|--------|
| **nlist** | Number of clusters | √N rule of thumb | ↑ → finer clusters, more centroid distance computations |
| **nprobe** | Clusters searched per query | 1–50 | ↑ recall, ↑ latency, linear |

#### Memory cost

```
Per vector: d × 4 bytes + small list bookkeeping
At 100M, d=1024: 400 GB raw + ~1 GB centroid storage
Lower memory than HNSW (no graph), but recall is lower at same latency.
```

#### When to use IVF

- 10M to 1B vectors
- Memory-constrained (no graph overhead)
- Combined with PQ for billion-scale (next section)

### 8.2.3 PQ — Product Quantization

PQ is a *compression* technique, not a standalone index. It dramatically reduces memory by representing each vector with a few bytes instead of kilobytes.

#### The intuition

A 1024-dim vector at FP32 is 4 KB. If you have 1 billion vectors, that's 4 TB of RAM — prohibitive. PQ compresses each vector down to typically 32–64 bytes, a 64–128× reduction with modest recall loss.

#### How it works

```
Step 1: Split each vector into m sub-vectors.

  1024-dim vector  ──split──▶  16 sub-vectors of 64 dims each

Step 2: For each sub-vector position, train a codebook of K=256 centroids.

  Position 0: train k-means with 256 centroids on all position-0 sub-vectors.
  Position 1: same.
  ...
  Position 15: same.

Step 3: Encode each vector as 16 codebook indices.

  Original: [v0, v1, ..., v15]   (each 64 floats)
  Encoded:  [c0, c1, ..., c15]   (each is a byte: index into 256-entry codebook)
  Storage:  16 bytes per vector!

Step 4: At query time, distance lookup.
  - Split query similarly into 16 sub-vectors.
  - For each position, precompute distances from query sub-vector to all 256 codebook entries → 16 lookup tables.
  - For each indexed vector, distance = sum of lookups across 16 positions.
```

This is fast because the inner loop is 16 byte-indexed table lookups + sums — extremely cache-friendly.

#### Compression worked example

```
Original: 1024 × 4 bytes = 4096 bytes (4 KB) per vector
PQ with m=16, 8 bits each: 16 bytes per vector

Compression ratio: 4096 / 16 = 256x
Recall loss: ~5-10% on most benchmarks
```

#### IVF-PQ — the production combo for billion-scale

IVF (cluster routing) + PQ (compression) gives you the best of both: fast cluster filtering plus compressed storage.

```
1B vectors, d=1024:
  Raw:             4 TB
  IVF-PQ (m=64):   ~75 GB    (50× compression)
  Search latency:  10-50 ms
  Recall:          ~0.85-0.92
```

This is how Pinecone, Weaviate, and Milvus handle billion-scale.

#### Key parameters

| Parameter | What it controls | Typical |
|-----------|------------------|---------|
| **m** | Number of sub-vectors | 8, 16, 32, 64 |
| **nbits** | Bits per sub-codebook | 8 (256 centroids) |

> **How to say this in an interview:** "Product quantization splits each vector into m sub-vectors, trains a 256-centroid codebook per position, and represents each sub-vector by its codebook index — one byte per position. A 1024-dim vector becomes 16 bytes if m=16. That's 256x compression. At query time, you precompute distance lookup tables from the query sub-vectors to each codebook, and per-document distance is a sum of 16 byte-indexed lookups — cache-friendly. Combined with IVF for cluster routing, IVF-PQ scales to billions of vectors with 10-millisecond latency. Recall drops 5 to 10 percent versus exact, which is fine for RAG."

### 8.2.4 The decision matrix

| Scale | Memory budget | Best choice |
|-------|---------------|-------------|
| < 1M | Any | HNSW (M=16, ef=128) — overkill works |
| 1M – 10M | Plenty | **HNSW (default)** |
| 10M – 100M | Medium | HNSW with binary quantization, or IVF-PQ |
| 100M – 1B | Limited | **IVF-PQ** |
| 1B+ | Tight | IVF-PQ + DiskANN (disk-resident graph) |

---

## 8.3 Quantization — beyond PQ

PQ is one form of quantization. Two other production-relevant forms:

### Scalar quantization (FP32 → INT8)

Simplest possible quantization. Each FP32 component becomes an INT8 (8-bit) approximation with a scaling factor stored per-block.

```
Compression: 4× (FP32 → INT8)
Recall loss: minor, often <1%
Compute: faster (8-bit SIMD)
Standard in most modern vector DBs.
```

### Binary quantization

Each dimension becomes a single bit (sign of the value).

```
Compression: 32× (FP32 → 1 bit)
Recall loss: 5-10% on typical benchmarks
Recovery: rescore top-K with full-precision vectors → near-original recall
Use case: billion-scale on commodity hardware
```

Worked example: 100M vectors at 1024-dim FP32 = 400 GB. Same with binary = 12.5 GB. Plus an HNSW graph on the binary-quantized index, you get a fast first-pass with full-precision rescoring on top-100. Net: hundreds-of-millions scale on a single machine.

Native in Qdrant, Weaviate, and Milvus.

### Matryoshka embeddings + binary

Modern embedding models like OpenAI's text-embedding-3 and Cohere v3 support Matryoshka Representation Learning — the embedding works at multiple dimensions. You can truncate a 1536-dim vector to 256 dims with minimal quality loss. Combined with binary quantization: 256 dims × 1 bit = 32 bytes per vector. 100M vectors = 3.2 GB total index. Fits on a laptop.

---

## 8.4 Distance metrics — cosine, dot, L2

| Metric | Formula | When |
|--------|---------|------|
| **Cosine similarity** | (a·b) / (‖a‖ × ‖b‖) | Default for normalized embeddings |
| **Dot product** | a·b | If embeddings are pre-normalized, equivalent to cosine but cheaper |
| **L2 (Euclidean)** | ‖a − b‖₂ | Rare for embeddings |
| **Inner product (negative)** | −a·b | For unnormalized dense embeddings |

### Practical notes

- Most modern embedding models output L2-normalized vectors. If yours doesn't, normalize them yourself before indexing.
- Cosine and dot product are equivalent for normalized vectors. Use dot product (cheaper computation).
- Make sure your indexing and querying use the *same* metric. Mismatch silently destroys recall.

---

## 8.5 Vector database product comparison (2026 snapshot)

This is the table the interviewer most wants to see. Be ready to explain your choices.

|  | **pgVector** | **Qdrant** | **Pinecone** | **Weaviate** | **Milvus** | **FAISS** |
|---|--------------|-----------|--------------|---------------|------------|-----------|
| **Type** | Postgres extension | Standalone DB | Managed SaaS | Standalone DB | Standalone DB | Library |
| **Self-host** | Yes | Yes | No (hybrid coming) | Yes | Yes | Yes |
| **Managed cloud** | Supabase, RDS | Qdrant Cloud | Native | Weaviate Cloud | Zilliz Cloud | No |
| **Practical scale** | ~5M | Billions | Billions | Hundreds of M | Billions | Billions (embedded) |
| **Hybrid search** | Via extensions (tsvector + vec) | Native | Yes | Native | Native | No |
| **Filters** | SQL WHERE | Payload index | Metadata filter | GraphQL filters | Boolean filter | Weak |
| **HNSW** | Yes (0.5+) | Yes | Yes (behind API) | Yes | Yes | Yes |
| **IVF / PQ** | No | No | Yes | No (HNSW only) | Yes | Yes |
| **Binary quantization** | No | Yes | Yes | Yes | Yes | Yes |
| **Multi-vector / ColBERT** | No | Yes | Limited | Yes | Yes | Yes |
| **Language** | C (Postgres ext) | Rust | Closed | Go | Go | C++ / Python |
| **CRUD / streaming** | Yes (Postgres) | Yes | Yes | Yes | Yes | No |
| **Best for** | Already on Postgres | Production default for self-host | Zero-ops managed | Schema-first apps | Hyperscale | Embed in service |

### Strengths and when to use each

**pgVector** — A Postgres extension. You get ACID, backups, replication, joins with relational data, all the operational maturity of Postgres. Quality and scale used to be embarrassing; with pgvector 0.5+ adding HNSW it's now competitive up to ~5M vectors. **Use when:** you already run Postgres, your corpus is small to medium, you need SQL joins with metadata.

**Qdrant** — Rust-built, fast, self-hostable, native hybrid search and binary quantization. Excellent payload filtering with dynamic query optimization. **Use when:** you want a production-grade self-hosted vector DB without the ops complexity of Milvus.

**Pinecone** — Fully managed SaaS. Zero ops. Strong filters, hybrid search, multi-region. Closed-source, vendor lock-in. **Use when:** you want zero ops and accept SaaS lock-in. Watch data residency for regulated industries.

**Weaviate** — Modular standalone DB with built-in vectorizers (text2vec, multi2vec) and GraphQL API. Good if you want vectorization built into the DB. **Use when:** you want an opinionated stack with built-in modules.

**Milvus** — Distributed vector DB built for hyperscale. Cloud-native, supports IVF, IVF-PQ, HNSW, scales to billions. Heavier ops. **Use when:** you're at hundreds of millions to billions of vectors with serious infra team.

**FAISS** — Facebook's library, the gold standard for embedded ANN. No persistence, no replication, no CRUD — it's a library, not a database. **Use when:** you want maximum performance embedded in your service, with static or batch-rebuilt indices.

> **How to say this in an interview:** "I pick based on three axes — scale, ops capacity, and existing infrastructure. For under 5 million vectors and an existing Postgres deployment, pgVector with HNSW is great — you get ACID, backups, and SQL joins for free. For self-hosted production at 5 to 100 million, Qdrant is my default — Rust performance, native hybrid search, payload filtering. For billion-scale, Milvus or Pinecone. For embedded use cases or static indices, FAISS as a library. Avoid Pinecone if data residency is a hard constraint, since it's SaaS-only."

---

## 8.6 pgVector deep-dive — Sachin's actual choice at ResMed

Since you used pgVector at ResMed, expect detailed questions. Here's the depth.

### Why pgVector at ResMed

```
Constraints:
  - Corpus: ~3 million chunks (clinical reports)
  - Team already runs Postgres for the operational DB
  - Need SQL joins with patient metadata, encounter dates, clinician
  - On-prem deployment (data residency for clinical data)
  - Audit / backup / compliance requirements

pgVector wins because:
  - Already running Postgres → zero new infra
  - SQL joins with relational metadata → no separate query path
  - Backup/restore/replication via existing Postgres tooling
  - HNSW since 0.5 → competitive ANN performance
  - Audit logs come for free
```

### HNSW index in pgVector

```sql
-- Create the index
CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- Set ef_search per query session
SET hnsw.ef_search = 100;

-- Query
SELECT chunk_id, content, embedding <=> $1 AS distance
FROM document_chunks
WHERE patient_id = 12 AND encounter_date >= '2024-01-01'
ORDER BY embedding <=> $1
LIMIT 10;
```

`<=>` is cosine distance. `<->` is L2. `<#>` is negative inner product.

### Hybrid search in pgVector

You can do dense + sparse hybrid using Postgres's full-text search via tsvector:

```sql
-- Add full-text search column
ALTER TABLE document_chunks
ADD COLUMN content_fts tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX ON document_chunks USING GIN (content_fts);

-- Hybrid query with RRF in SQL
WITH dense AS (
  SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS rank
  FROM document_chunks
  ORDER BY embedding <=> $1
  LIMIT 50
),
sparse AS (
  SELECT chunk_id, ROW_NUMBER() OVER (ORDER BY ts_rank(content_fts, to_tsquery($2)) DESC) AS rank
  FROM document_chunks
  WHERE content_fts @@ to_tsquery($2)
  LIMIT 50
),
fused AS (
  SELECT chunk_id, SUM(1.0 / (60 + rank)) AS rrf_score
  FROM (SELECT * FROM dense UNION ALL SELECT * FROM sparse) AS combined
  GROUP BY chunk_id
)
SELECT * FROM fused ORDER BY rrf_score DESC LIMIT 10;
```

Pure SQL hybrid search with RRF fusion. No external services.

### When pgVector breaks down

- **Past ~10M vectors** — HNSW index size and rebuild time become painful
- **High write throughput** — Postgres MVCC + HNSW updates can lock; bulk inserts are fine but small frequent updates degrade
- **Pure vector throughput** — Qdrant/Milvus are 2–10× faster on raw QPS at scale
- **Distributed scale** — Postgres scales vertically; for horizontal scale you need partitioning logic

### When to migrate from pgVector

Signals: index rebuild >2 hours, query p99 >500ms, corpus growing past 10M, hybrid search getting unwieldy in SQL.

> **How to say this in an interview:** "At ResMed I used pgVector with HNSW because the team already ran Postgres, the corpus was 3 million chunks, we needed metadata joins with patient and encounter data, and the on-prem requirement for clinical data was a hard constraint. The combination of ACID, backups, replication, and HNSW indexing in one stack was unbeatable. I'd migrate to Qdrant past around 10 million vectors when index rebuilds and hybrid SQL get painful. The switching cost is real but pgVector is the right starting point for any team already on Postgres."

---

## 8.7 Hybrid search at the database level

Most production RAG uses hybrid search — dense + sparse — because each catches what the other misses. The implementation depends on the DB:

### Built-in hybrid (Qdrant, Weaviate, Milvus)

```python
# Qdrant example
results = client.query_points(
    collection_name="docs",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=50),
        Prefetch(query=sparse_vector, using="sparse", limit=50),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
)
```

The DB handles fusion internally. Faster than client-side fusion (no round trips).

### Application-level fusion (pgVector, FAISS)

Run dense and sparse separately, fuse with RRF in app code. Worked example:

```python
def hybrid_retrieve(query, top_k=10):
    dense_results = pgvector_search(query, top_k=50)
    sparse_results = bm25_search(query, top_k=50)
    
    # RRF fusion
    scores = defaultdict(float)
    for rank, doc_id in enumerate(dense_results, 1):
        scores[doc_id] += 1 / (60 + rank)
    for rank, doc_id in enumerate(sparse_results, 1):
        scores[doc_id] += 1 / (60 + rank)
    
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
```

### Sparse vector representations

Modern hybrid uses *learned sparse* vectors instead of pure BM25. SPLADE and BGE-M3 produce sparse vectors with ~30K non-zero terms — better than BM25, indexable in vector DBs that support sparse.

---

## 8.8 Metadata filtering — the easy thing to get wrong

Most production RAG queries have filters:
```
"Find chunks similar to query, but only from patient 12, encounter after 2024-01-01"
```

How the DB combines vector search and filters dramatically affects performance.

### Pre-filter vs post-filter

**Pre-filter:** apply filter before vector search. SQL `WHERE` clause is the canonical example.
- Pro: tight result set, no wasted work
- Con: HNSW may degrade to linear scan on low-selectivity filters (filter eliminates 99% of vectors → graph navigation breaks down)

**Post-filter:** vector search first, then filter results.
- Pro: HNSW operates as designed
- Con: may return fewer than K results if filter is selective (have to over-fetch then trim)

### Production-friendly: dynamic optimizer

Qdrant's payload index plus query optimizer dynamically picks pre vs post based on filter selectivity. Milvus has similar logic. pgVector and Pinecone are more rigid.

### Worked example

100M vector index. Query has filter `tenant_id = 'small_tenant'` (selectivity 0.001).

- Pure pre-filter: 100K vectors after filter → linear scan, ~10ms (fine)
- Pure post-filter: HNSW returns top-100, filter to 1 result, miss recall (bad)
- Selective: build a separate HNSW per major tenant, OR use payload-aware HNSW

**The test:** measure for your workload. Profile filter selectivity distribution and benchmark p99 latency.

---

## 8.9 Multi-tenancy patterns

If you're building a SaaS product on top of RAG, you have to isolate tenants:

### Pattern 1: Shared index + tenant filter

```
Single HNSW index. Each vector has tenant_id metadata.
Every query filters by tenant_id.
```

**Pros:** simple, one index to manage, works at low to medium tenant counts.
**Cons:** filter performance degrades on per-tenant pre-filter, leak risk if filter bypass bug.

### Pattern 2: Per-tenant collection / namespace

```
One collection per tenant. Tenant-scoped index, tenant-scoped queries.
```

**Pros:** clean isolation, no leak risk, filter performance is irrelevant.
**Cons:** overhead per tenant (~MB of bookkeeping), doesn't scale to thousands of tenants efficiently.

Native support: Qdrant (collections), Pinecone (namespaces), Milvus (collections).

### Pattern 3: Per-tenant database

Full DB-level isolation for high-security tenants. Maximum cost, maximum isolation.

### Default: shared index with strict access control

For most products, shared index + tenant filter + middleware-enforced access control is the right default. Add per-tenant collections only for the largest or highest-security tenants.

---

## 8.10 Index parameters tuning — a worked example

You have an HNSW index with recall@10 of 0.85 on your test set. Target is 0.95. What do you tune?

### Step 1: try ef_search (cheapest)

```
Current: ef_search = 64 → recall 0.85, latency 5ms
Try:     ef_search = 128 → recall 0.91, latency 9ms
Try:     ef_search = 256 → recall 0.94, latency 17ms
Try:     ef_search = 512 → recall 0.95, latency 35ms
```

ef_search trades latency for recall. Pick the highest you can afford.

### Step 2: try M (requires rebuild)

```
Current: M = 16, ef_construction = 200 → recall 0.95 at ef_search 512
Try:     M = 32, ef_construction = 400 → recall 0.96 at ef_search 256
```

Higher M means denser graph, better recall at lower ef_search. Cost: more memory (graph storage) and longer build time.

### Step 3: if recall plateaus

If you can't get past 0.92 regardless of HNSW params, the embedding model is the bottleneck. No amount of index tuning fixes a bad embedding. Consider:
- Better embedding model (BGE-large vs MiniLM)
- Domain-tuned embedding (fine-tune on your corpus)
- Asymmetric model with query/passage prefixes

> **How to say this in an interview:** "When recall is below target, I tune in this order. First, ef_search at query time — cheapest, no rebuild, trades latency for recall linearly. Bump from 64 to 128 to 256, measure both axes, pick the highest you can afford. Second, M — requires rebuild but gives a better recall-latency Pareto frontier. Third, if recall plateaus around 0.92 regardless, the bottleneck is the embedding model, not the index. No HNSW tuning fixes a bad embedding."

---

## 8.11 Cost modeling — the numbers to memorize

### Storage at FP32

```
1024-dim FP32 vector:           4 KB
100M vectors:                   400 GB
1B vectors:                     4 TB

With HNSW graph overhead (~1.3-1.5×):
100M:                           ~520-600 GB
```

### Storage with compression

```
Scalar quant (INT8):            1 KB/vector → 100M = 100 GB
Binary quant (1 bit):           128 B/vector → 100M = 12.5 GB
PQ (m=64, 8-bit):               64 B/vector → 100M = 6.4 GB
Matryoshka 256 + binary:        32 B/vector → 100M = 3.2 GB
```

### Memory (HNSW must fit)

```
100M FP32 + HNSW graph: ~600 GB → multi-machine or quantize
100M binary + HNSW graph: ~20 GB → fits in commodity RAM
1B PQ-compressed: ~75 GB → fits in mid-range box
```

### Throughput (single instance)

```
HNSW on decent CPU at 1M scale:  10K-50K QPS
HNSW at 100M scale:              1K-5K QPS
IVF-PQ at 1B scale:              100-1K QPS
Scale horizontally with shards + replicas.
```

### Latency targets

```
p50 retrieval:        < 20 ms
p99 retrieval:        < 100 ms
p99 with reranking:   < 300 ms
End-to-end RAG p99:   < 3000 ms
```

---

## 8.12 Index maintenance — building, updating, deleting

### Build time

```
HNSW build: O(N × ef_construction × log N)
For 10M vectors with ef_construction=200: ~30-60 min on a good machine
For 100M: ~5-10 hours (parallelize across cores)
For 1B: dedicated build cluster, ~hours to a day
```

### Incremental inserts

- **HNSW:** native incremental support. New vectors are inserted into the graph cheaply.
- **IVF:** inserts go to existing clusters; if cluster distribution shifts, periodic rebuild needed.
- **IVF-PQ:** same as IVF; codebooks may drift with new data, periodic retraining.

### Deletion

Most ANN indexes don't truly delete — they tombstone (mark as deleted but don't remove from the graph). Why: removing nodes from a graph is expensive and can break navigability.

**Production pattern:** soft delete + periodic rebuild. Deletes accumulate; once they exceed ~10–20% of the index, rebuild to reclaim space and improve performance.

### Blue-green index rebuild

For zero-downtime updates:

```
1. Build new index in parallel (V2)
2. Run both indices in shadow mode (V2 receives reads, results compared)
3. Cut over traffic to V2
4. Decommission V1
```

Most production teams do this weekly or monthly depending on insert/delete rate.

---

## 8.13 The "1B vectors at scale" architecture

Interview question: how would you design vector retrieval for 1 billion vectors?

```
   ┌──────────────────────────────────────────────────────────┐
   │                    Application Layer                      │
   │                                                            │
   │   Query Router ──▶ shards by tenant_id (consistent hash)  │
   └─────────┬─────────────────┬─────────────────┬──────────────┘
             │                 │                 │
        ┌────▼─────┐      ┌────▼─────┐     ┌────▼─────┐
        │ Shard 1  │      │ Shard 2  │ ... │ Shard N  │
        │          │      │          │     │          │
        │ IVF-PQ + │      │ IVF-PQ + │     │ IVF-PQ + │
        │ HNSW on  │      │ HNSW on  │     │ HNSW on  │
        │ centroids│      │ centroids│     │ centroids│
        │          │      │          │     │          │
        │ ~100M    │      │ ~100M    │     │ ~100M    │
        │ vecs    │      │ vecs    │     │ vecs    │
        └──────────┘      └──────────┘     └──────────┘
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                       Aggregator (top-K from each shard,
                       merge to global top-K)
                               │
                               ▼
                          Reranker
                               │
                               ▼
                            LLM
```

Key design choices:
- **Sharding by tenant_id** — single-tenant queries hit one shard
- **IVF-PQ** for compression at billion-scale
- **HNSW on the centroids** for fast cluster routing
- **Replicas per shard** for read throughput and HA
- **Async eventually-consistent inserts** — bulk-loaded, daily rebuild

This is roughly Pinecone's architecture, and what Milvus exposes more transparently.

---

## 8.14 Resume tie-ins

> **Resume tie-in (ResMed pgVector):** "I used pgVector with HNSW for the clinical chatbot at ResMed. The corpus was about 3 million chunks. Existing Postgres deployment, on-prem requirement for clinical data residency, and the need to JOIN with patient encounter metadata made pgVector the obvious choice. We did hybrid search via pgVector for dense and Postgres tsvector for BM25, fusing with RRF in SQL. HNSW M=16, ef_construction=200, and we tuned ef_search per session — 100 was enough for our recall target. The ops simplicity of one Postgres-shaped thing was huge for a small team."

> **Resume tie-in (Snowflake feature store at ResMed):** "Snowflake was the feature store for tabular real-time inference, not vectors. Important to be precise: Snowflake stored patient feature vectors for our risk models — engagement, treatment adherence, demographic features — accessed at sub-second latency through a Snowflake-to-SageMaker integration. pgVector was the *separate* store for the RAG side. Different stores, different workloads. Mixing them up is a common interview trap."

> **Resume tie-in (TrueBalance — XGBoost p99 < 500ms):** "Different system — that's a real-time XGBoost on Lambda for credit risk. But the latency-vs-quality tradeoff framing transfers to vector search. For LLM serving and RAG, the equivalent levers are HNSW ef_search (recall vs latency), reranker top-N (precision vs latency), and quantization (memory vs recall). Same engineering muscle."

---

## 8.15 Master Interview Q&A — Vector Databases

**Q1. Walk me through HNSW, IVF, and IVF-PQ — when each?**
> HNSW is a hierarchical graph — top layers are sparse for long jumps, bottom layer has all vectors with short edges. Search is greedy descent through layers. Best recall-latency tradeoff up to about 100 million vectors. Memory-heavy because it stores the full vectors plus the graph. IVF clusters vectors into nlist groups via k-means and searches the nprobe nearest clusters per query — lower memory than HNSW, slightly lower recall at same latency. IVF-PQ adds product quantization on top — each vector becomes 16 to 64 bytes via codebook indices, scales to billions with around 50x compression. My defaults: HNSW for under 10 million, HNSW with binary quant for 10 to 100 million, IVF-PQ for billion plus.

**Q2. Explain product quantization.**
> Split each vector into m sub-vectors. For each sub-vector position, train a codebook of 256 centroids using k-means. Encode each vector as m codebook indices — m bytes total. A 1024-dim FP32 vector at 4 KB becomes 16 bytes if m=16, a 256x compression. At query time, you precompute distance lookup tables from the query sub-vectors to each codebook, and per-document distance is a sum of m byte-indexed table lookups, which is extremely cache-friendly. Recall drops 5 to 10 percent versus exact, which is fine for RAG. Production at billion-scale uses IVF-PQ — IVF for cluster routing, PQ for compression.

**Q3. When is pgVector enough?**
> Under 5 million vectors, an existing Postgres deployment, need for SQL joins with relational metadata, and on-prem or specific-region requirements. You inherit ACID, backups, replication, audit logs, the entire Postgres operational maturity for free. With pgVector 0.5 plus and HNSW it's competitive on recall and latency in that scale. It breaks down past around 10 million vectors when index rebuilds and hybrid SQL queries get painful, and that's when I'd migrate to Qdrant.

**Q4. Pinecone versus Qdrant versus Weaviate versus Milvus — which would you pick for a UAE fintech?**
> First, data residency. UAE PDPL means I want either self-hosted or a SaaS with verified UAE region. That nudges toward Qdrant or Milvus self-hosted. Qdrant is my default for self-hosted production — Rust performance, native hybrid search, payload filtering with dynamic optimizer, simple ops. Milvus if I'm scaling to billions and have an infra team to handle the distributed deployment. Pinecone is only viable if it has a UAE region — fastest to ship but SaaS-only is risky for regulated industries. Weaviate is fine but doesn't differentiate enough for fintech.

**Q5. FAISS versus a vector database — when which?**
> FAISS is a library, not a database. No persistence, no replication, no CRUD, weak filtering. Use FAISS when you want maximum performance embedded in your service with static or batch-rebuilt indices, like a recommendation engine that rebuilds nightly. Use a real vector DB when you need real-time inserts, metadata filtering, multi-tenancy, replication, HA. FAISS is the engine inside many vector DBs — it's foundational but rarely the right top-level choice for RAG.

**Q6. Your Qdrant HNSW recall at 10 is 0.85 and you need 0.95. What do you tune?**
> Walk through it in order. First, ef_search — bump from 64 to 128 to 256 to 512, that's the cheapest knob, query-time only. Each step trades latency for recall roughly linearly. Second, M — requires rebuild, but higher M like 32 or 48 gives a better Pareto frontier of recall versus latency. Third, ef_construction — more candidates during build improves graph quality. Fourth, if recall plateaus around 0.9 to 0.92 regardless of HNSW params, the bottleneck is the embedding model — try a stronger one or domain-tune yours.

**Q7. HNSW M parameter — what does it control?**
> M is the maximum number of neighbors per node in each layer. Higher M means a denser graph — better recall, more memory, longer build time. Memory cost is roughly M times 8 bytes per node per layer. Typical values are 16 to 32. I bump to 48 or 64 only with serious RAM headroom. The classic tradeoff: cheap recall comes from raising ef_search at query time; the recall ceiling for a given latency budget comes from raising M at build time.

**Q8. Binary quantization — what's the tradeoff?**
> One bit per dimension, sign of the value. 32x memory reduction. Recall loss is around 5 to 10 percent on most benchmarks. The recovery trick: rescore the top-100 binary-retrieved candidates with full-precision vectors. That brings recall close to original at minimal cost. Native in Qdrant, Weaviate, Milvus. Combined with Matryoshka embeddings — truncate to 256 dims then binarize — you can fit 100 million vectors in around 3 GB. Game-changer for cost-sensitive deployments.

**Q9. Matryoshka embeddings for storage — how?**
> Matryoshka Representation Learning trains an embedding model where the first 64 dimensions are themselves a useful embedding, the first 128 are a better one, the first 256 even better, and so on — each prefix is independently trained to be a valid embedding. So you can truncate a 1536-dim embedding to 256 dims with minimal quality loss. Combined with binary quantization, you compress 1536 floats to 256 bits — 32 bytes per vector. Often beats PCA on non-MRL embeddings by 5 to 10 nDCG points. OpenAI's text-embedding-3 and Cohere v3 support this natively.

**Q10. Pre-filter versus post-filter on metadata — which wins?**
> Depends on filter selectivity. High-selectivity filters that eliminate 99 percent of the corpus break HNSW navigation — pre-filter ends up doing linear scan. Low-selectivity filters work fine with pre-filter. Post-filter does HNSW search first then applies the filter — clean from HNSW's perspective but may return fewer than K results, so you over-fetch. Production-friendly DBs like Qdrant have a payload index plus dynamic query optimizer that picks based on selectivity. With pgVector you have to think carefully and sometimes manually decide. Always profile your filter selectivity distribution.

**Q11. Implement hybrid search — how?**
> Two retrievers in parallel — dense via HNSW for semantic and BM25 or learned sparse via inverted index for lexical. Each returns top-50. Fuse with Reciprocal Rank Fusion: each document's score is sum over retrievers of 1 over (60 plus rank). Parameter-free, robust to score-scale differences. Some DBs — Qdrant, Weaviate, Milvus — have built-in hybrid with internal fusion, no client round trips. With pgVector I do it in pure SQL using tsvector for sparse and the vector column for dense, fusing in a CTE.

**Q12. Distance metrics — cosine versus dot versus L2?**
> Most modern embedding models output L2-normalized vectors. For normalized vectors, cosine similarity equals dot product, so use dot product because it's cheaper computationally. L2 Euclidean is rarely used for embeddings — only for unnormalized representations like raw learned features. If your model outputs unnormalized vectors, normalize them yourself before indexing. The biggest gotcha is mismatched metrics between indexing and querying — silently destroys recall.

**Q13. Cost of 100 million vectors at 1024 dimensions FP32?**
> Raw storage is 100 million times 1024 times 4 bytes equals 400 gigabytes. With HNSW graph overhead of about 1.3 to 1.5x, the index hits 520 to 600 gigabytes. With scalar INT8 quantization, 100 GB. With binary quantization, about 12.5 GB. With PQ at m equals 64, around 6.4 GB. Memory-side, HNSW must fit in RAM for low-latency, so for 100 million FP32 you need a multi-machine sharded deployment — or you quantize. With binary quantization plus rescoring, 100 million fits in 20 GB on a commodity box.

**Q14. Multi-tenancy — shared index or per-tenant?**
> Default is shared index with tenant_id metadata filter and middleware-enforced access control. Simple, scales to thousands of tenants cheaply, one index to manage. Add per-tenant collections — Qdrant collections, Pinecone namespaces — only for the largest tenants that need filter-performance isolation, or for tenants with unique schema. Per-tenant database for the highest-security tenants where regulatory requirements demand full isolation. Most products land at shared with strict access control plus per-tenant collections for the top 5 percent.

**Q15. Index rebuild strategy?**
> HNSW supports incremental adds cheaply, but accumulated deletions are tombstoned, not removed. Once tombstones exceed 10 to 20 percent of the index, rebuild to reclaim space and recover navigability. IVF-PQ has cluster drift as new data arrives — schedule periodic codebook retraining and rebuild, weekly or monthly. Production pattern is blue-green: build the new index in parallel, run shadow reads for validation, cut over traffic, decommission the old one. Zero downtime.

**Q16. How would you architect 1 billion vectors?**
> Sharded IVF-PQ. Roughly 100 million per shard, sharded by tenant_id with consistent hashing. Within each shard, IVF-PQ with HNSW on the centroids for fast cluster routing. Replicas per shard for read throughput and HA. SSD-backed for the IVF-PQ tables since they're large. Aggregator service merges top-K per shard into global top-K. Bulk async writes, daily index rebuilds, blue-green cutover. Expect p99 around 50 to 200 milliseconds end-to-end. This is essentially Pinecone's architecture under the hood.

**Q17. ColBERT versus single-vector retrieval — what's the storage cost?**
> ColBERT is multi-vector retrieval — one embedding per token, not one per document. A 500-token document becomes 500 vectors instead of 1 — 500x storage. Late-interaction matching at query time gives much better recall, especially on long documents. Worth it when retrieval quality is critical and you can afford the storage. Newer variants like ColBERT-PLAID compress aggressively. Use ColBERT for high-stakes RAG where you can afford 10 to 100x storage; use single-vector for everything else.

**Q18. Cold-start and warm-up for vector DBs?**
> HNSW must load the full graph plus vectors into RAM for low-latency queries. Cold start can take 30 to 90 seconds for a large index. Solution: keep warm replicas always running, so a cold instance never serves traffic. Qdrant has memmap mode for partial loading. Milvus has memory-efficient modes that page in on demand. For deployments with elastic scaling, pre-warm new instances before adding to the load balancer.

**Q19. How do you benchmark a vector DB?**
> ann-benchmarks on GitHub is the standard tool. It runs a fixed dataset and query set against your DB, measuring recall at k, queries per second at a recall target, index build time, and memory footprint. Standard datasets are SIFT, GIST, GloVe, Deep1B. The catch: your data isn't those datasets. Always benchmark on your own workload — your embedding model's distribution, your typical filter patterns, your query length distribution. Public benchmarks are useful for ballparking; production decisions need your-data benchmarks.

**Q20. What CI/CD tests do you have on a vector DB?**
> Five layers. First, index builds successfully on test data — basic smoke test. Second, known queries return expected document IDs — sanity check. Third, latency regression test — p99 retrieval under budget. Fourth, recall on golden set above threshold. Fifth, migration test — old index format upgrades to new, no data loss. Plus production monitoring: latency p50/p99, recall sampled from periodic golden-set eval, query distribution drift, memory usage, disk usage if PQ-quantized. Treat the vector DB like any other production data store.

---

Continue to **[Chapter 09 — Model Optimization](09_model_optimization.md)**.
