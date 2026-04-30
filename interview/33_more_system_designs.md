# Chapter 33 — More System Design Cases (LLM, RAG, Recommendations)

> **Why this chapter exists:** System design rounds are the highest-variance part of an interview — the question can be almost anything in your domain. The dedicated system-design chapters (16, 28, 30, 31) cover the canonical heavy hitters; this chapter adds six more cases that round out the catalog. Each design is presented as you'd narrate it at a whiteboard: clarifying questions, capacity math, architecture diagram, technical depth on the hard parts, evaluation, and follow-up questions. Treat this chapter as a quarry — when an interviewer asks "design X," scan the table of contents and find the closest match, then adapt.

**The six designs in this chapter:**
1. LLM-powered customer support agent (multi-channel, ticket-aware)
2. E-commerce product recommendation system at scale
3. Enterprise RAG over heterogeneous data (docs, Slack, email, code)
4. Code review copilot
5. Multi-modal product search (text + image)
6. Real-time fraud detection with LLM-generated explanations

---

## 33.1 Design — LLM-powered customer support agent

### The problem

"Build a customer support agent for a SaaS company. It handles email, chat, and Slack. It can answer policy questions from a knowledge base, look up customer account state, escalate to humans when needed, and learn from agent feedback."

### Clarifying questions

1. Volume? (~10K tickets/day, mix of email, chat, Slack.)
2. Tier of customers? (Free vs paid — different SLAs and what the agent can offer.)
3. Account-level actions? (Can the agent issue refunds, change plans, reset passwords? Assume read + suggest, not write.)
4. Multilingual? (English, Spanish, French primary; auto-detect.)
5. Latency targets? (Chat: real-time response in 5s. Email: minutes is fine. Slack: <30s.)
6. Privacy? (Customer data is sensitive but not HIPAA. Standard SOC 2.)

### Architecture

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  CHANNELS                                                          │
   │  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
   │  │  Email   │  │  Chat    │  │  Slack   │                          │
   │  │ (SES)    │  │(WebSock.)│  │ (events) │                          │
   │  └─────┬────┘  └─────┬────┘  └─────┬────┘                          │
   └────────┼─────────────┼─────────────┼─────────────────────────────────┘
            │             │             │
            ▼             ▼             ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  Channel Adapter (FastAPI) — normalizes to common Ticket schema     │
   │  Loads / creates Ticket; assigns ticket_id; routes to orchestrator  │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  Orchestrator Agent (LangGraph)                                    │
   │   state: customer_id, ticket_id, history, user_msg                 │
   │                                                                    │
   │   nodes:                                                           │
   │     classify_intent  → policy / account / billing / abuse / smalltalk│
   │     load_customer_ctx → fetch tier, plan, recent activity          │
   │     retrieve_kb       → if policy: RAG over KB                     │
   │     query_account     → if account: tools to query systems         │
   │     compose_response  → final LLM call with all context            │
   │     escalate_if_needed → flag for human if confidence low or       │
   │                          trigger words (cancel, fraud, lawyer, ...) │
   │                                                                    │
   │   edges: classify_intent fans out conditionally                    │
   └────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────┬───────┴───────┬─────────────────┐
        ▼                 ▼               ▼                 ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
   │ KB Vector│    │ Customer │    │ Billing  │    │ Human Agent  │
   │ Store    │    │ DB       │    │ System   │    │ Queue (Zendesk│
   │(pgVector)│    │ (Postgres│    │ (Stripe  │    │  / Intercom) │
   │          │    │  /Snowflk)│    │  via API)│    │              │
   └──────────┘    └──────────┘    └──────────┘    └──────────────┘

   ┌────────────────────────────────────────────────────────────────────┐
   │  FEEDBACK LANE                                                     │
   │  Every agent message → thumbs-up/down from customer                │
   │  Every escalated ticket → human resolution → labeled feedback      │
   │  Nightly: train intent classifier, refine retrieval, audit         │
   └────────────────────────────────────────────────────────────────────┘
```

### The hard parts

**Confidence-aware escalation.** The agent isn't omniscient. The orchestrator measures confidence at multiple points: retrieval confidence (top-1 cosine), generation confidence (model log-probability if exposed, otherwise an LLM-as-judge), policy match (was the retrieved KB chunk classified as authoritative). Below threshold or on trigger words ("cancel my account", "this is fraud"), escalate immediately and queue for human pickup with full context.

**Customer context injection without prompt bloat.** Instead of dumping the customer's entire profile into the prompt, the orchestrator injects only the relevant slice based on intent. Billing question → plan tier, payment method, last 3 invoices. Policy question → just the customer's tier (so refunds policy is correct). Account question → recent actions only.

**Multi-turn ticket continuity.** Tickets span days. The conversation history is the ticket's transcript, not a single chat. The agent loads the full ticket history on every turn and treats it as a long-running conversation — with summarization-fallback for tickets over 50 turns.

**Hand-off to humans without losing context.** When escalating, the agent generates a structured handoff package: customer's question, agent's attempted answer, retrieved KB chunks, customer context slice, conversation history. The human picks up with full context, not a cold start.

### Evaluation

- Intent classifier accuracy (offline, golden set).
- Faithfulness of RAG answers (LLM-as-judge on production sample).
- Escalation precision/recall — did we escalate when we should have? Did we mis-escalate?
- Customer satisfaction (CSAT) on agent-resolved tickets.
- Resolution time (agent-handled vs human-handled).
- Cost per ticket — token spend / count.

### Follow-ups

**"How do you handle abusive customers?"** Detect via toxicity classifier on the user message. The agent has a hardened system prompt to remain calm and decline abusive requests, while still offering escalation. Repeat abuse triggers automatic escalation with a flag for the human to handle (or dismiss).

**"What if the KB is wrong?"** Customer feedback (thumbs-down + explanation) feeds a review queue. The KB is treated as living content, with versioned articles and a process for updating. The agent's responses cite KB articles by URL+version so wrong answers are traceable to specific content.

**"How do you prevent the agent from making unauthorized refunds?"** The agent doesn't have refund-write permissions — only read. When the agent decides a refund is appropriate, it generates a structured refund proposal that's queued for human approval. The human clicks approve, the action executes via the billing system. Structural separation of propose vs execute.

---

## 33.2 Design — E-commerce product recommendation system at scale

### The problem

"Build a recommendation system for an e-commerce platform with 10 million users and 5 million products. Show personalized recommendations on the homepage, product pages, and in the cart."

### Clarifying questions

1. Latency? (Homepage: <100ms p95. Real-time on every page load.)
2. Coldstart for new users? (Yes — many anonymous browsers. Use trending + intent signals.)
3. Coldstart for new products? (Yes — must surface new SKUs. Content-based as fallback.)
4. Personalization signals? (Click history, purchase history, search queries, time-of-day, location.)
5. Business constraints? (Promote certain categories; avoid showing out-of-stock; respect brand-blocking by user.)

### Capacity math

```
   Users:           10M
   Products:        5M
   Sessions/day:    3M (rough 30% DAU)
   Recommendations per session: 5 surfaces × 20 items = 100 items
   Total recs/day:  300M
   Latency target:  <100ms p95
```

This is a **two-stage recommender** problem. Single-stage on 5M products at 100ms is impossible.

### Architecture

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                          STAGE 1 — RETRIEVAL                       │
   │  Goal: 5M products → ~500 candidates per (user, surface)           │
   │  Latency target: 30ms                                              │
   │                                                                    │
   │  Three retrievers run in parallel and union results:               │
   │   1. Collaborative filtering (matrix factorization / two-tower)    │
   │      → user_embedding · product_embedding via vector ANN           │
   │   2. Content-based: similar to recently-viewed/purchased products  │
   │   3. Trending / fresh products (per category, per time window)     │
   │                                                                    │
   │  Output: ~500 candidate products with retrieval scores              │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                          STAGE 2 — RANKING                         │
   │  Goal: 500 candidates → final 20 ranked items                       │
   │  Latency target: 50ms                                              │
   │                                                                    │
   │  XGBoost or Neural Ranker, features per (user, product, context):  │
   │   - User: tier, recency-weighted purchase categories, dwell time   │
   │   - Product: price, popularity, freshness, in-stock, category      │
   │   - Interaction: predicted CTR, price-affinity match                │
   │   - Context: time of day, surface type, device                     │
   │                                                                    │
   │  Output: ranked list, calibrated CTR and conversion probabilities  │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                          STAGE 3 — POLICY                          │
   │  Apply business rules: deduplicate categories, ensure diversity,   │
   │  filter out-of-stock, apply brand-block, inject promoted items.    │
   │  Latency target: 5ms                                               │
   └────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────┐
   │                          OFFLINE PIPELINES                         │
   │  Daily: train two-tower model on click data, refresh user/product  │
   │         embeddings, compute trending scores.                        │
   │  Hourly: refresh feature store (Redis online + Snowflake offline). │
   │  Real-time: stream click events to Kafka → update user embedding   │
   │             via lightweight retrain (ALS update).                  │
   └────────────────────────────────────────────────────────────────────┘
```

### The hard parts

**Two-tower model architecture.** A two-tower model has a "user tower" and a "product tower" that map their inputs to a shared embedding space. The dot product (or cosine) between user_embedding and product_embedding approximates relevance. Crucially, the towers are independent at inference: precompute all 5M product embeddings, store in vector index. At query time, embed the user once, ANN-search against the product index → top-k retrieval in <30ms.

**Coldstart.** New user (no clicks): use demographic + context features (location, device, referrer) to embed into a "default-user" zone, fall back to trending products for their region. New product (no clicks): use content embedding (title, description, image features) to place it in the product tower's space, eligible for retrieval immediately.

**Diversity.** A naive ranker shows 20 nearly-identical products. Apply diversity constraints: at most 2 from the same brand, at most 5 from the same category, ensure price variety. Implement as a re-ranker over the top-50 ranked items (not the top-20) using a determinantal point process or simple greedy diversification.

**Real-time signals.** A user just clicked a product — that signal should affect the next page's recommendations. Stream clicks via Kafka to a lightweight user-embedding updater that bumps the user's embedding toward the clicked product's neighborhood. Total propagation latency: <500ms.

### Evaluation

- Offline: NDCG@20, AUC of CTR predictions on held-out clicks.
- Online: A/B test win-rate, CTR lift, conversion lift, GMV (gross merchandise value) lift.
- Diversity: average pairwise similarity of recommended items (lower is more diverse).
- Coverage: fraction of catalog ever recommended.
- Equity: long-tail product impressions vs head products.

### Follow-ups

**"How do you handle re-ranking for promotions?"** Promotions are policy-layer concerns, not model concerns. The ranker outputs unbiased CTR predictions; the policy layer boosts promoted items by a small multiplicative factor (e.g., 1.2x) and ensures at least 1 promoted item appears in the top-20.

**"How do you prevent the model from getting stuck recommending what users already bought?"** Feature engineering — include "last_purchased_category" as a signal that the ranker can learn to discount appropriately. Also explicit business rule: never recommend a product the user has already purchased within the last 30 days unless it's a consumable.

**"What if a user is shopping for someone else (gift)?"** Hard problem. Detect via signals: searches for "gift", purchases that don't match the user's history, queries for products outside the user's typical price range or category. When detected, switch the recommendation tower to a "gift mode" that uses recipient-context (if specified) or popular-gifts in the searched category.

---

## 33.3 Design — Enterprise RAG over heterogeneous data

### The problem

"An enterprise has knowledge scattered across Confluence pages, Slack channels, email archives, GitHub repos, and Google Drive. Build a unified RAG system that lets employees ask questions and get answers grounded in this data."

### Clarifying questions

1. Volume? (~5M Confluence pages, ~50M Slack messages, ~100M emails, ~10K GitHub repos, ~5M Drive docs.)
2. Permissions? (Critical. Each source has its own ACLs — must be respected at retrieval time.)
3. Freshness? (Emails: hourly. Confluence: daily. GitHub: on-merge. Slack: real-time.)
4. Multilingual? (English primary, Japanese and German for some teams.)
5. Latency? (5-10s acceptable for substantive questions.)

### Architecture

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                          INGESTION LAYER                           │
   │   Per-source connector (Confluence API, Slack API, Gmail API,      │
   │   GitHub API, Drive API). Each emits documents with:               │
   │     content, metadata (source, url, author, ts), acl_groups        │
   │                                                                    │
   │   Connectors push to Kafka → ingestion workers → chunking          │
   │   → embedding (parallelized across many GPU workers)               │
   │   → vector store (per-source index OR unified index with           │
   │   source-type metadata)                                            │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                          RETRIEVAL LAYER                           │
   │   Per query:                                                       │
   │   1. Get the user's identity and ACL groups (from SSO/AD)          │
   │   2. Query expansion: rewrite into multiple subqueries (HyDE)      │
   │   3. Parallel hybrid search across sources:                        │
   │      - Each source's index returns its top-K                       │
   │      - Filter by ACL: drop chunks the user can't access            │
   │   4. Reciprocal Rank Fusion across source results                  │
   │   5. Cross-encoder rerank top-30 → top-5                           │
   │   6. Source-aware reranking: boost authoritative sources           │
   │      (Confluence > random Slack message)                           │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                          GENERATION LAYER                          │
   │   LLM with rich citations: each claim links back to the source     │
   │   document/page/message URL. Markdown rendering preserves source    │
   │   formatting (e.g., code blocks from GitHub).                      │
   │   Faithfulness check post-generation.                              │
   └────────────────────────────────────────────────────────────────────┘
```

### The hard parts

**Permissions / ACL enforcement.** This is the biggest pitfall in enterprise RAG. Naive systems index everything and rely on UI-side filtering, which is fundamentally insecure (the LLM has already seen the chunks). Correct design: every chunk in the vector store is tagged with `acl_groups` (the user groups that can access it). Retrieval filters by user's group memberships *before* returning chunks. An efficient implementation uses post-filtering for HNSW (filter after retrieval) or pre-filtering with metadata indexes (filter during retrieval). Whichever you choose, the LLM never sees a chunk the user can't access.

**Source heterogeneity.** Confluence pages are long-form; Slack messages are short and conversational; emails have threading; GitHub has code. Each needs its own chunking and embedding strategy. Slack: chunk per thread, not per message — context matters. Email: chunk per thread, with subject in metadata. Code: chunk per function or per file with language-aware splits. Confluence: section-aware chunking respecting headings.

**Authority weighting.** A formal Confluence page about company policy is more authoritative than a random Slack message that mentions the policy. The reranker incorporates a `source_authority_score` feature (manually configured per source), so retrieval favors authoritative sources for policy-shaped questions. For "what's the team currently working on" questions, recent Slack might score higher.

**Freshness.** Stale answers are worse than no answer. Track `last_updated` per chunk, decay relevance for old content, and run drift monitoring on the indexed corpus to detect coverage gaps.

### Evaluation

- Per-source NDCG@5 on a curated golden set (questions whose answers should come from each source).
- Cross-source: when the answer requires synthesizing across sources, did the system retrieve from all relevant sources?
- ACL leakage tests: simulated probing with restricted users.
- Citation accuracy: do the cited URLs actually contain the cited claim?

### Follow-ups

**"What if Slack content is needed but the message is in a private channel the user isn't in?"** The chunk is tagged with the channel's membership at index time. If the user isn't a member, the chunk is filtered out at retrieval. The user gets an answer using only sources they have access to, which may be incomplete — the system should disclose: "Some context may be in channels you don't have access to."

**"How do you keep the index fresh?"** Push-based for high-priority sources (Slack via webhooks, GitHub via webhooks), pull-based for the rest (Confluence daily diff scan, Email hourly polling). Median lag end-to-end: 5 minutes.

---

## 33.4 Design — Code review copilot

### The problem

"Build a code review assistant that watches GitHub PRs, comments inline with suggestions, flags potential bugs, and ensures style conformance. Integrate with the team's existing PR review process — assistive, not replacement."

### Clarifying questions

1. Repo size? (~5M lines of code across 200 repos, mostly Python and TypeScript.)
2. PR volume? (~500 PRs per day across the org.)
3. Review style? (Inline comments per file, plus a top-level summary.)
4. Existing tooling? (GitHub Actions, ESLint, Black, type checkers — already running.)

### Architecture

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  GitHub Webhook → SQS → Reviewer Lambda                            │
   │  Triggered on: pull_request opened/synchronize                      │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  Reviewer Lambda                                                   │
   │   1. Fetch PR diff + changed file paths                            │
   │   2. For each changed file:                                        │
   │      - Fetch full file content (not just diff)                     │
   │      - Build context: file content + relevant repo context         │
   │        (related modules, tests, recent changes)                    │
   │      - Call Reviewer LLM (Claude Sonnet) with structured output:   │
   │        list of (line_number, severity, comment)                    │
   │   3. For top-level summary:                                        │
   │      - Aggregate per-file findings                                 │
   │      - Run a second LLM pass over the diff for arch concerns       │
   │   4. Filter: skip duplicate of existing static-analysis findings   │
   │   5. Post comments via GitHub API                                  │
   │   6. Log every comment for feedback collection                     │
   └────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────┐
   │  FEEDBACK LANE                                                     │
   │   GitHub reactions (👍/👎) on bot comments → labeled training data │
   │   Resolved comments without action → likely false positive         │
   │   Quarterly: fine-tune prompts; review systematic false positives  │
   └────────────────────────────────────────────────────────────────────┘
```

### The hard parts

**Context window economy.** A file might be 2K lines; the diff is 30 lines; the model needs to see the change in context but you can't blow the budget. Strategy: include the diff plus a window of 50 lines above and below each change, plus a brief "imports" snapshot. For larger context needs, the model can request more via tool calls.

**Avoiding noise.** A bad code-review bot is worse than no bot. Filtering layers: (1) skip findings that overlap with what static analyzers already report. (2) confidence threshold — only post if the LLM is confident. (3) per-author calibration — some authors get noisier reviews than others; tune. (4) explicit "not actionable" filter via a small classifier before posting.

**Repo context.** Just the changed file isn't enough — the model often needs to see related code (the function being changed plus its callers, the test file, the type definitions). Use a code-graph (built nightly) to fetch related symbols. Pull these into context as additional files.

**Auto-fix vs comment.** For high-confidence simple fixes (unused imports, formatting), the bot can push a follow-up commit with the fix. For anything substantive, comment only — humans decide.

### Evaluation

- Comment helpfulness rate (👍 / total) per category (security, perf, style, bug).
- False positive rate (resolved comments without code change).
- True positive depth: did the bot catch real bugs that humans confirmed?
- Time-to-merge impact: do PRs with bot review merge faster (false) or slower (negative) compared to without.

### Follow-ups

**"What stops the bot from leaking secrets in its review?"** Pre-input filter: scan the diff for likely secrets (regex + entropy detection on string literals) and refuse to send those chunks to the LLM. The bot can still comment on the structure but doesn't see the actual secret values.

**"How do you handle multi-language repos?"** Per-language reviewers — the system selects the right prompt set based on file extension. Python files get Python-specific prompts (typing, async, common pitfalls); TypeScript gets TS-specific (types, async, immutability).

---

## 33.5 Design — Multi-modal product search (text + image)

### The problem

"Build product search for an e-commerce platform that accepts both text queries ('blue running shoes') and image uploads ('I want shoes that look like this'). Return ranked products."

### Clarifying questions

1. Catalog size? (5M products, each with title, description, multiple images.)
2. Latency? (<200ms p95 for a smooth UX.)
3. Image input? (User uploads or pastes URL; expect 1-5MB JPEGs.)
4. Hybrid (text + image together)? (Yes — "running shoes that look like this image".)

### Architecture

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  CLIENT: text query + optional image                               │
   └────────────────────────┬───────────────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  Search API (FastAPI)                                              │
   │   1. Embed query:                                                  │
   │      - Text only: text-embedding (multilingual)                    │
   │      - Image only: CLIP image embedding                            │
   │      - Both: weighted sum of text + image embeddings               │
   │   2. ANN search against unified product index (CLIP-aligned)       │
   │   3. Hybrid with BM25 over titles for keyword precision            │
   │   4. RRF fusion → top-100                                          │
   │   5. Reranker (cross-encoder for text+image) → top-50              │
   │   6. Filter: in-stock, allowed regions, price filters              │
   │   7. Return top-20 with images and snippets                        │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  PRODUCT INDEX (vector + BM25)                                     │
   │   Each product has:                                                │
   │     - title and description (for BM25)                             │
   │     - text_embedding (from title+description, multilingual)        │
   │     - image_embedding (from primary image, CLIP)                   │
   │     - both stored in same vector space (CLIP-aligned text encoder) │
   │     - metadata: category, price, brand, in_stock, region           │
   └────────────────────────────────────────────────────────────────────┘
```

### The hard parts

**Aligning text and image embeddings.** Use CLIP or a CLIP-style model whose text and image encoders project into a shared space. Cosine similarity between a text query and a product image is meaningful in this space. Without alignment, you'd need separate text and image indices and complex fusion logic.

**Multi-image products.** A product has 5 images (different angles, colors). Index multiple embeddings per product (one per image), keep `product_id` as a metadata field, dedupe in retrieval (return each product once, using its best-matching image's score). Optionally, compute a "primary image embedding" per product as an average of all its images for a single canonical representation.

**Combining text and image queries.** Naive: average the text and image embeddings. Better: weighted average tuned on validation data — typically image gets weight 0.3-0.5 when both are present (because text is usually more specific). Best: a learned fusion model that takes both embeddings and outputs a single query embedding.

**Negative cases.** "Show me running shoes that look like this image but NOT in red." Negation in queries is hard for embedding models. Pattern: parse explicit constraints from the text ("NOT red") via a small LLM, apply as metadata filter post-retrieval.

### Evaluation

- Retrieval NDCG@20 on a curated set of (query, gold-relevant-products) pairs.
- Click-through rate by rank in production.
- Image-specific tests: did similar-image queries surface visually similar products?
- Hybrid query tests: do mixed text+image queries beat text-only?

### Follow-ups

**"What if the user's image is a face or other PII?"** Detect via a fast face/PII classifier; reject the upload with a polite message: "We can't search using that image. Try a product photo."

**"How do you handle a query with words in a non-English language?"** Multilingual text encoder (multilingual CLIP variants like multilingual-CLIP or mCLIP). Same vector space as the English encoder, so retrieval works across languages.

---

## 33.6 Design — Real-time fraud detection with LLM-generated explanations

### The problem

"Build a real-time fraud detection system for credit-card transactions. Decisions in <100ms, with explanations that can be shown to fraud analysts and (eventually) to customers."

### Clarifying questions

1. Volume? (10K transactions/second peak.)
2. Latency? (Hard 100ms p99 — can't slow down checkout.)
3. False positive cost? (Very high — a false flag annoys a real customer.)
4. False negative cost? (Higher in dollars — fraud loss.)
5. Explanation depth? (For analysts: full reasoning. For customers: high-level "we flagged this because...".)

### Architecture

```
   ┌────────────────────────────────────────────────────────────────────┐
   │  Transaction event from payment processor                           │
   │  → Kinesis / Kafka                                                 │
   └────────────────────────┬───────────────────────────────────────────┘
                            │
                            ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  Real-time scorer (Lambda or Fargate, target p99 < 100ms)          │
   │   1. Fetch features:                                               │
   │      - Online: Redis-fronted feature store (last hour activity)    │
   │      - Offline: pre-computed cards, cached at the edge             │
   │   2. Run XGBoost model → fraud probability                         │
   │   3. Below threshold (~0.7): approve                               │
   │   4. Above threshold: hold for additional checks (rules engine)    │
   │   5. Above hard threshold (~0.95): block immediately               │
   │   6. Log decision + features used                                  │
   └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  Async explanation generator (separate from scoring path)          │
   │   For every flagged transaction (held or blocked):                 │
   │   1. Compute SHAP values per feature                               │
   │   2. Pick top-5 contributing features                              │
   │   3. Call LLM with structured prompt:                              │
   │      "Generate a 2-sentence explanation for why this transaction   │
   │       was flagged, citing these top features and their values."    │
   │   4. Store explanation alongside the transaction                   │
   │   5. Push to fraud analyst queue                                   │
   └────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────┐
   │  FEEDBACK LANE                                                     │
   │   Analyst marks transaction as: confirmed_fraud / false_positive   │
   │   Customer disputes: false_positive (chargeback)                   │
   │   Daily: retrain XGBoost with labels                               │
   │   Drift monitoring on feature distributions                        │
   └────────────────────────────────────────────────────────────────────┘
```

### The hard parts

**Latency budget.** XGBoost inference is fast (~5ms), but feature fetching is the bottleneck. Online features must be in Redis, accessible in <10ms. Snowflake-backed offline features must be pre-computed and pushed to Redis on a schedule. Anything that requires a synchronous Snowflake call breaks the budget.

**LLM explanations are async.** The LLM call takes 500-2000ms — well beyond the scoring budget. The architecture splits these: synchronous fast path produces the decision; asynchronous slow path generates the explanation later. Analysts get the explanation in 1-2 seconds; the customer-facing "transaction held" notification doesn't include an LLM-generated explanation until the analyst confirms.

**SHAP at scale.** TreeSHAP is O(features × depth × leaves). For a tree ensemble, this is 1-5ms — fine for the async explanation path, too slow for the sync scoring path. Compute SHAP only for held/blocked transactions (a small fraction of total volume).

**Calibration.** XGBoost outputs probabilities that aren't well-calibrated. For threshold-based decisions, calibrate using isotonic regression on a held-out set so that "0.85" really means 85% historical fraud rate.

### Evaluation

- Precision and recall on labeled fraud cases.
- False positive rate (annoyed customers / total approved transactions).
- Latency p50, p95, p99.
- LLM explanation usefulness (analyst rating, sample 1%).
- Drift: PSI per feature over time, alert on > 0.25.

### Follow-ups

**"What if the LLM hallucinates an explanation that doesn't match the actual SHAP values?"** Deterministic post-check: parse out feature names mentioned in the LLM's explanation, verify each appears in the top-10 SHAP contributors. If a feature is mentioned that's not in SHAP, regenerate with a stricter prompt: "Use ONLY these features." Three failures → fall back to a templated explanation listing the SHAP features.

**"How do you handle adversarial users probing the model?"** Don't expose SHAP values or detailed explanations to end users — only to internal analysts. Customer-facing explanations are high-level templates ("activity unusual for your account") that don't reveal feature importances.

---

## 33.7 Generic patterns that show up across all of these designs

Reading the six designs above together, you can extract recurring patterns:

```
   PATTERN 1 — Two-stage retrieval and ranking
     Coarse retrieval (fast, broad) → fine reranking (slower, precise)
     Used in: recommendation, RAG, search, fraud (in a sense)

   PATTERN 2 — Async explanation/insight
     Make the synchronous decision fast; generate explanations async
     Used in: fraud, code review, customer support escalation

   PATTERN 3 — Confidence-aware routing
     Multiple paths; route by confidence or intent
     Used in: customer support, DAWN, search results

   PATTERN 4 — Permissions-aware retrieval
     ACL filters baked into vector store metadata
     Used in: enterprise RAG, DAWN authenticated path

   PATTERN 5 — Real-time + batch hybrid pipeline
     Real-time signals augment batch-trained models
     Used in: recommendations, fraud, RAG over Slack

   PATTERN 6 — Feedback lane
     Every system has a labeled-feedback pipeline that improves quality
     Used in: every design above

   PATTERN 7 — Structural security
     Don't rely on prompts for security; use structural constraints
     (whitelisted tools, scoped IAM, ACL filters, propose-vs-execute)
     Used in: DAWN, customer support, fraud, code review

   PATTERN 8 — Cost-aware model cascade
     Use cheap models for simple work, expensive models only when needed
     Used in: customer support, DAWN, code review
```

When a system design question lands, one of these patterns is usually the right opener. "I'd structure this as a two-stage retrieval and ranking pipeline" sets up the rest of the answer cleanly.

---

## 33.8 The cheatsheet

```
   QUESTION ARCHETYPES TO EXPECT
     - Customer-facing chat / support     → §33.1 patterns
     - Recommendation / personalization  → §33.2 patterns
     - Enterprise knowledge / RAG        → §33.3 patterns
     - Code copilot / dev tools          → §33.4 patterns
     - Multi-modal search / commerce     → §33.5 patterns
     - Real-time prediction + explain    → §33.6 patterns

   THE TEMPLATE FOR ANY DESIGN ANSWER
     1. Clarify (3-5 questions, then summarize)
     2. Capacity math (scale, latency, storage)
     3. Architecture diagram (sketched, every box explained)
     4. Hard parts (3-5, with concrete trade-offs)
     5. Evaluation (offline + online + drift)
     6. Follow-ups (anticipate 5-10)

   THE CARDINAL RULE
     Don't draw fancy unless you can explain every box in under 30 seconds.
     A simpler diagram with clear narration beats a complex diagram with
     hand-waving every time.
```

---

End of Chapter 33. Continue back to **[Chapter 00 — Master Index](00_index.md)**.
