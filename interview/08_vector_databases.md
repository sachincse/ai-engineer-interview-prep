# Chapter 08 — Vector Databases & Search Indexes
## HNSW, IVF, PQ — and pgVector vs Qdrant vs Pinecone vs Milvus vs Weaviate

> Every RAG system has a vector DB. This chapter covers the indexes, the products, and the picks for UAE deployments.

---

## 8.1 The problem: nearest-neighbor at scale

You have 100M embeddings of 1024 dims. A new query embedding arrives. Find the top-10 nearest neighbors under cosine similarity.

**Brute force:** 100M · 1024 · 8 bytes = 800 GB per query. ~seconds on a big machine. **Not viable.**

**Approximate Nearest Neighbor (ANN) indexes** trade exactness for speed, aiming for recall ≥ 0.9 at millisecond-range latency.

---

## 8.2 The three index families

### 8.2.1 HNSW — Hierarchical Navigable Small World

```
Layer 2:     A ────────── B       ← sparse top layer (long jumps)
             │
Layer 1:     A ── B ── C ── D     ← intermediate
             │    │    │    │
Layer 0:     all vectors connected via short edges
```

- Build a multi-layer graph; top layers are sparse, bottom is dense
- Search: start at top, greedy descent, refine at each layer
- **Best recall/latency tradeoff** up to ~10M vectors
- Memory-heavy — stores full vectors + graph

**Params:**
- `M` — max connections per node (16-64); ↑ = better recall, more memory
- `ef_construct` — candidates during build (100-500)
- `ef_search` — candidates during query (64-512); ↑ = recall up, latency up

### 8.2.2 IVF — Inverted File

```
1. Cluster all vectors into K centroids (k-means)
2. At query: compute distance to centroids, pick nearest M clusters
3. Search exhaustively within those clusters
```

- Lower memory than HNSW
- Scales to 100M+ vectors
- Slightly lower recall at same latency

**Params:**
- `nlist` — number of clusters (√N rule of thumb)
- `nprobe` — clusters to search (1-50); ↑ = recall up

### 8.2.3 PQ — Product Quantization

```
Split each vector into sub-vectors; replace each with a codebook index.
1024-d vector → 16 sub-vectors of 64d → 16 bytes (one byte per codebook lookup)
```

- **Compresses vectors 64-100×** with some accuracy loss
- Used as **IVF-PQ** for billion-scale
- Distance computed via lookup tables

**Params:**
- `m` — number of subquantizers
- `nbits` — bits per subquantizer (usually 8)

### 8.2.4 Decision matrix

| Scale | Memory budget | Best choice |
|-------|--------------|-------------|
| <1M | Any | HNSW (ef=128, M=16) |
| 1M-10M | High | **HNSW** (default) |
| 10M-100M | Medium | IVF-Flat or HNSW |
| 100M-1B | Limited | IVF-PQ |
| 1B+ | Tight | IVF-PQ + disk-based (DiskANN) |

---

## 8.3 Distance metrics

| Metric | Formula | When |
|--------|---------|------|
| **Cosine** | 1 - (a·b)/(‖a‖‖b‖) | Default for normalized embeddings |
| **Dot product** | a·b | Embeddings pre-normalized → equivalent to cosine |
| **L2 (Euclidean)** | ‖a-b‖₂ | Non-normalized; rarely used for embeddings |
| **Inner product** | -a·b | Recommended for unnormalized dense embeddings |

For cosine / dot product, always L2-normalize your embeddings if the model doesn't.

---

## 8.4 Binary / scalar quantization

### Scalar (FP32 → INT8)
~4× memory reduction. Minor accuracy loss. Cheap.

### Binary quantization
Each dim → 1 bit (sign). 32× memory, ~5% recall loss. Supported in Qdrant, Weaviate.
Enables billion-scale indices on commodity hardware.

### Matryoshka + binary
Combine: truncate via MRL to 256 dims, then binary → 256 bits = 32 bytes per vector. 100M vectors = 3.2 GB index.

---

## 8.5 Product comparison (2026 snapshot)

| | **pgVector** | **Qdrant** | **Pinecone** | **Weaviate** | **Milvus** | **FAISS** |
|--|-------------|-----------|--------------|--------------|------------|-----------|
| Type | Postgres ext | Standalone | Managed SaaS | Standalone | Standalone | Library |
| Self-host | ✅ | ✅ | ❌ (hybrid coming) | ✅ | ✅ | ✅ |
| Managed cloud | Supabase, RDS | Qdrant Cloud | ✅ | Weaviate Cloud | Zilliz | ❌ |
| Scale | ~5M | Billions | Billions | Hundreds of M | Billions | Billions (embedded) |
| Hybrid search | via extensions | ✅ native | ✅ | ✅ native | ✅ | ❌ |
| Filters | SQL | ✅ | ✅ | ✅ | ✅ | weak |
| HNSW | ✅ (0.5+) | ✅ | (behind API) | ✅ | ✅ | ✅ |
| Binary quant | – | ✅ | – | ✅ | ✅ | – |
| Language | C (Postgres ext) | Rust | – | Go | Go | C++ |
| Best for | "We already have Postgres" | Production default | Zero-ops managed | Built-in modules | Hyperscale | Embed in your service |

---

## 8.6 Choosing for Avrioc (UAE) — the real-world answer

**Likely constraints:**
- Data residency — UAE PDPL, Abu Dhabi-specific
- Egress restrictions
- Cost-consciousness (startup/scale-up)
- Skill fit with team

**Pragmatic recommendations:**

1. **MVP / <5M vectors** — **pgVector on AWS RDS me-central-1 (Bahrain)** or Azure UAE region. You probably already run Postgres; zero new infra.
2. **Production / 5M-100M** — **Qdrant self-hosted on EKS** in me-central-1. Rust perf, simple ops, built-in hybrid, binary quant, payload filters.
3. **Hyperscale (>100M)** — **Milvus self-hosted** with IVF-PQ, if you have the ops budget.
4. **Zero-ops, accept SaaS** — **Pinecone regional pods** (check me-central availability) or **Qdrant Cloud**.

Avoid Pinecone US-only if UAE residency is a hard constraint.

---

## 8.7 Metadata filtering — easy to underestimate

```sql
-- pgVector example
SELECT doc_id, embedding <-> $1 AS distance
FROM docs
WHERE tenant_id = 'acme' AND created_at > '2025-01-01'
ORDER BY embedding <-> $1
LIMIT 10;
```

**Important:**
- Pre-filter vs post-filter affects recall dramatically
- HNSW with pre-filter can degrade to linear scan at low-selectivity filters
- Qdrant's payload index + dynamic query optimizer handles this gracefully
- Pinecone's filters are fast but limit compositions

---

## 8.8 Multi-tenancy patterns

### 1. Shared index + tenant filter
Single index, `tenant_id` as metadata filter. Simple, cheap at low tenant counts. Leak risk if filter bypass.

### 2. Per-tenant collections
Separate collection per tenant (Qdrant collections, Milvus collections). Clean isolation. Overhead per tenant.

### 3. Per-tenant DB
Full isolation for high-security tenants. Max cost.

For Avrioc's likely architecture (one product, many users): shared index with tenant filter + robust access control.

---

## 8.9 Index building & maintenance

### Build time
- HNSW: O(N · ef_construct · log N)
- IVF: O(N · nprobe)
- PQ: O(N · codebook_training) — usually fast

For 10M vectors, HNSW builds in ~30-60 min on a good machine.

### Incremental inserts
- HNSW supports incremental adds cheaply
- IVF-PQ can struggle if cluster distribution shifts — periodic rebuild needed
- Qdrant / Milvus handle incremental well

### Deletion
- HNSW: soft delete + periodic rebuild (tombstones)
- IVF-PQ: same pattern

---

## 8.10 Cost modeling

### Storage
- FP32 embeddings at 1024 dim: 4 KB/vector
- 100M vectors: 400 GB
- With 1.3-1.5× index overhead: 500-600 GB
- Binary quantized: ~3 GB

### Memory (HNSW)
- Must fit in RAM for <100ms latency
- 100M FP32 × 1024 + graph = ~500 GB RAM → prohibitive
- Solution: binary quantization OR IVF-PQ OR disk-based (DiskANN)

### Throughput
- HNSW on decent CPU: 1000-5000 QPS per instance
- Scale horizontally with shards + replicas

---

## 8.11 Interview Q&A — Vector DBs

**Q1. HNSW vs IVF vs PQ — when each?**
> HNSW: up to 10M, best recall/latency, memory-heavy. IVF: 10M-100M, lower memory, slightly lower recall. IVF-PQ: 100M+, compresses via product quantization, scales to billions.

**Q2. pgVector — when is it enough?**
> <5M vectors, already using Postgres, need SQL joins + metadata filtering. You get ACID, backups, familiar ops for free. Breaks down past ~10M where ANN tuning gets painful.

**Q3. Pinecone vs Qdrant vs Weaviate vs Milvus — pick for UAE fintech?**
> Qdrant or Milvus self-hosted. Both support on-prem, payload filtering, hybrid search, data residency (UAE PDPL). Qdrant has simpler ops; Milvus scales horizontally better. Pinecone is fastest to ship but SaaS-only — verify data residency.

**Q4. FAISS vs vector DB?**
> FAISS is a library — no persistence, replication, filters, CRUD. Use embedded in your service for static indices with max perf. Use a vector DB when you need real-time inserts, filters, HA.

**Q5. [Gotcha] Qdrant HNSW recall@10 is 0.85. Target 0.95. What do you tune?**
> Increase `ef_construct` (128→256, build-time) and `ef_search` (64→256, query-time) — trade latency for recall. Increase `m` (16→32) if memory allows — biggest recall lever but rebuilds. If recall plateaus at 0.9 regardless, embedding model is bottleneck.

**Q6. HNSW M parameter?**
> Max connections per node. Higher M = better recall, more memory (M · 8 bytes per node per layer). Typical 16-32; go higher (48-64) only with RAM to spare.

**Q7. Binary quantization — trade-offs?**
> 1 bit per dim → 32× memory reduction. ~5% recall loss on most benchmarks. Combined with rescoring (re-rank top-N with full-precision vectors) recovers recall nearly entirely. Native in Qdrant, Weaviate.

**Q8. Matryoshka embeddings for storage — how?**
> Train (or use a model trained with) MRL. Truncate the 1024-d embedding to 256 dims. 4× storage reduction. Often beats PCA on non-MRL embeddings by 5-10 nDCG points.

**Q9. Pre-filter vs post-filter — which?**
> Pre-filter (SQL-WHERE before ANN): narrows candidates, but with strict filters on HNSW can degrade to linear scan. Post-filter: ANN first, filter results — may leave too few. Qdrant dynamically picks. Measure for your workload.

**Q10. Hybrid search — how to implement?**
> Dense index (HNSW) + sparse index (BM25). Query both, fuse scores via RRF. Some DBs (Weaviate, Qdrant, Milvus) have built-in hybrid. Or build separately and fuse in application code.

**Q11. Distance metric — cosine vs dot vs L2?**
> Normalized embeddings: cosine = dot product. Use dot product (cheaper). L2 (Euclidean) is uncommon for embeddings but sometimes better for learned representations without normalization.

**Q12. Cost of 100M vectors at 1024 dim FP32?**
> Raw: 400 GB. With HNSW graph + index overhead: ~600 GB. Binary quantized: ~3 GB. Plan memory (HNSW must fit in RAM) accordingly.

**Q13. Multi-tenancy — shared or per-tenant index?**
> Shared + filter: simple, cheap at scale, leak risk. Per-tenant collection: clean isolation, overhead. Per-tenant DB: max isolation, max cost. Default: shared with strict filter enforcement.

**Q14. Index rebuild strategy?**
> HNSW: soft delete + periodic rebuild; supports incremental adds. IVF-PQ: cluster drift → periodic rebuild (weekly/monthly). Production: blue-green rebuild (new index built, traffic cut over).

**Q15. How would you architect RAG storage for 1B vectors?**
> IVF-PQ or DiskANN (disk-resident graph). Sharding by tenant or hash. SSD-backed. Binary quantization for hot path. Expect p95 latency ~50-200 ms per query.

**Q16. [Gotcha] Your latency is fine on 10M vectors, but recall is collapsing on a specific user segment. Why?**
> Likely a metadata filter that collapses candidates below index's recall regime. Check filter selectivity; consider per-segment indexing or switching to pre-filter-aware index (Qdrant's).

**Q17. ColBERT vs single-vector retrieval — storage impact?**
> ColBERT: one vector per token. Typical doc (500 tokens) = 500× storage of a single-vector index. Use only where recall matters and you can afford 10-100× storage.

**Q18. Warm-up and cold-start for vector DB?**
> HNSW: must load full graph + vectors into RAM. Cold start can take 30-90s for large indices. Solution: keep always-warm replicas; use Qdrant's memmap or Milvus's memory-efficient modes.

**Q19. How do you benchmark a vector DB?**
> ann-benchmarks (Github: erikbern/ann-benchmarks) is the standard. Measure: recall@k, QPS at recall target, index build time, memory footprint. Benchmark on YOUR data — public datasets don't predict your recall curve.

**Q20. Vector DB in CI/CD — what tests?**
> (1) Index builds successfully on test data. (2) Known queries return expected IDs. (3) Latency regression test (< budget). (4) Recall on golden set (≥ threshold). (5) Migration test (old index → new index schema).

---

## 8.12 Resume tie-ins

- **"Databases & Feature Stores: ... pgVector ..."** — be specific: "I used pgVector 0.5 with HNSW indexes for the ResMed clinical chatbot because the team already ran Postgres, the corpus was ~3M chunks, and we needed metadata joins with SQL."
- **"Snowflake feature store"** — not a vector DB but an online feature store. Be ready to distinguish: Snowflake = tabular features for real-time inference; vector DB = semantic similarity. Different stores, different latencies.

---

Continue to **[Chapter 09 — Model Optimization](09_model_optimization.md)**.
