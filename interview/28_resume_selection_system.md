# Chapter 28 — System Design: HR Resume Selection System (RAG-Style)

> **Why this chapter exists:** This is exactly the kind of question Avrioc could ask in the system-design round. It's tractable enough to design in 45 minutes, deep enough to test every layer of your knowledge — embeddings, vector search, ranking, evaluation, scaling, fairness. This chapter walks through the full design as you'd narrate it at a whiteboard, from requirements gathering through deployment, with the trade-offs and follow-up questions an interviewer would push on.
>
> **The problem statement (assume the interviewer just gave you this):** "Design a system for HR teams. Candidates submit roughly 1,000 resumes per day for a particular Job Description, totaling around 30,000 resumes per month. When an HR person opens the portal, they should see the top 50 or top 100 resumes that match a given JD, ranked by relevance."

---

## 28.1 Step 1 — Clarifying questions you ask before sketching

Never start drawing diagrams without clarifying. The clarifying questions themselves earn points; they signal product thinking. Ask 3–5 questions, then summarize what you heard.

1. **Multiple JDs?** Is there one JD per system or many? (Assume many — different roles, different teams.)
2. **Latency target when HR opens the portal?** Sub-second? A few seconds? (Assume sub-second so we cache rankings.)
3. **Real-time or batch ranking?** Are rankings computed live when HR opens the page, or pre-computed offline? (Assume hybrid — pre-compute, refresh on demand.)
4. **Languages and formats?** Are resumes only English, only PDF, or multi-language and multi-format (PDF, DOCX, image scans)? (Assume English + multilingual variants, mostly PDF, some DOCX, occasional image scans.)
5. **Recency requirements?** Are 6-month-old resumes still relevant or should we filter? (Assume 6-month sliding window of "active candidates.")
6. **Bias and fairness constraints?** Are we required by policy to anonymize protected attributes (name, gender, age, photo) before scoring? (Assume yes — required for both compliance and quality. This is huge for design.)
7. **Recruiter feedback loop?** Can the system learn from recruiter actions (shortlist, reject, hire)? (Assume yes — recruiter signals are the gold-standard feedback.)
8. **Volume scaling?** Could 1,000/day become 10,000/day? (Design for 10x — costs little extra now, painful retrofit later.)

**Then summarize back:** "So we're building a multi-JD, sub-second-ranking, anonymized RAG-style resume scoring system, hybrid pre-compute plus on-demand refresh, English-plus-multilingual with mixed input formats, with recruiter feedback as a long-term improvement signal. Designed to scale to roughly 10K resumes/day. Sound right?"

That summary alone separates senior candidates from juniors.

---

## 28.2 Step 2 — Capacity math

Always do this on the whiteboard. It anchors every later decision.

```
   Volume:       1,000 resumes/day → 30,000/month → ~360,000/year
   Scale-out:    design for 10,000 resumes/day = 300,000/month
   Active set:   6-month sliding window → 300K × 6 = 1.8M resumes
   Embedding d:  768 (BGE-large) or 1024 (E5-large)
   Storage:      1.8M × 1024 floats × 4 bytes = ~7.4 GB raw FP32
                 With FP16 + PQ: ~1 GB — easily fits in memory
   Index size:   HNSW with M=16, efConstruction=200 → ~3 GB on disk

   Query rate:   HR users (say 100 active at peak) opening the portal
                 → assume 1 query per HR per minute when active
                 → ~100 QPS peak for "show top-100 for this JD"

   Compute:      Each query: embedding lookup (cached) + ANN search
                 → ~50ms p99 with HNSW on warm index
                 Reranker (cross-encoder over top-100): ~200ms p99
                 → total p99 ~250ms before LLM-based explanations
```

This sizing tells me: this is a **small-to-medium system**, not a huge one. We don't need a multi-region distributed vector store. A single-node Postgres+pgVector or Qdrant instance handles 1.8M vectors comfortably.

---

## 28.3 Step 3 — High-level architecture

Sketch this on the whiteboard. Every box gets explained.

```
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                              INGESTION PATH                              │
   │                                                                          │
   │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
   │  │ Candidate│───▶│  Upload  │───▶│   S3     │───▶│  Parser  │            │
   │  │ portal   │    │  API     │    │ (raw     │    │  Lambda  │            │
   │  │ (web)    │    │ (FastAPI)│    │  resumes)│    │ (PDF→txt)│            │
   │  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘            │
   │                                                       │                  │
   │                                                       ▼                  │
   │                                       ┌────────────────────────┐         │
   │                                       │  Anonymizer + Section  │         │
   │                                       │  Extractor (LLM-based) │         │
   │                                       │  removes PII, splits   │         │
   │                                       │  into sections         │         │
   │                                       └────────────┬───────────┘         │
   │                                                    │                     │
   │         ┌──────────────────┐                       ▼                     │
   │         │  Postgres        │◀──────────┬───────────────────────┐         │
   │         │  (resume_meta:   │           │                       │         │
   │         │  id, parsed_text,│           ▼                       ▼         │
   │         │  sections,       │  ┌──────────────┐         ┌──────────────┐  │
   │         │  skills, exp,    │  │  Embedding   │         │ Skills       │  │
   │         │  upload_ts)      │  │  Model       │         │ Extractor    │  │
   │         └────────┬─────────┘  │  (BGE/E5)    │         │ (NER + skill │  │
   │                  │            └──────┬───────┘         │  taxonomy)   │  │
   │                  │                   │                 └──────┬───────┘  │
   │                  ▼                   ▼                        ▼          │
   │         ┌──────────────────────────────────────────────────────┐         │
   │         │              pgVector (or Qdrant)                    │         │
   │         │   per-section vectors + skill tag arrays             │         │
   │         │   indexed by HNSW + GIN on skills array              │         │
   │         └──────────────────────────────────────────────────────┘         │
   └──────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────────┐
   │                              QUERY PATH                                  │
   │                                                                          │
   │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────────────┐      │
   │  │  HR      │───▶│  Portal  │───▶│  Cache   │──▶│  Ranking       │      │
   │  │  user    │    │  API     │    │  (Redis: │    │  Service       │      │
   │  │  (web)   │    │ (FastAPI)│    │  jd_id → │    │  (FastAPI)     │      │
   │  └──────────┘    └──────────┘    │  top-100)│    └────────┬───────┘      │
   │                                  └──────────┘             │              │
   │                                                           ▼              │
   │                                                  ┌────────────────┐      │
   │                                                  │ Stage 1: ANN   │      │
   │                                                  │ retrieve top-K │      │
   │                                                  │ (K=500) from   │      │
   │                                                  │ pgVector       │      │
   │                                                  └────────┬───────┘      │
   │                                                           │              │
   │                                                           ▼              │
   │                                                  ┌────────────────┐      │
   │                                                  │ Stage 2: hybrid│      │
   │                                                  │ filter — must- │      │
   │                                                  │ have skills,   │      │
   │                                                  │ experience cut │      │
   │                                                  │ (drops to ~200)│      │
   │                                                  └────────┬───────┘      │
   │                                                           │              │
   │                                                           ▼              │
   │                                                  ┌────────────────┐      │
   │                                                  │ Stage 3: cross-│      │
   │                                                  │ encoder rerank │      │
   │                                                  │ (BGE-reranker  │      │
   │                                                  │  or Cohere)    │      │
   │                                                  │ top-100 final  │      │
   │                                                  └────────┬───────┘      │
   │                                                           │              │
   │                                                           ▼              │
   │                                                  ┌────────────────┐      │
   │                                                  │ Optional Stage │      │
   │                                                  │ 4: LLM-gen     │      │
   │                                                  │ explanation    │      │
   │                                                  │ for top-10     │      │
   │                                                  └────────────────┘      │
   └──────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────────────────┐
   │                            FEEDBACK / EVAL LANE                          │
   │                                                                          │
   │   HR actions (shortlist, reject, hire) → event log →                     │
   │      → nightly eval job: NDCG@100 against shortlist labels               │
   │      → drift monitoring on JD and resume embedding distributions         │
   │      → fine-tune the reranker quarterly on accumulated feedback          │
   └──────────────────────────────────────────────────────────────────────────┘
```

This is the picture. Every box is explained next.

---

## 28.4 Step 4 — Walking through each component

### 4.1 The ingestion path

**Upload API.** Candidates submit resumes through a web portal. The API is FastAPI behind an API Gateway with rate limiting (~10 uploads/IP/minute to prevent spam). Files go directly to S3 with a presigned URL — never through your application server, because that wastes bandwidth.

**Parser Lambda.** S3 PutObject triggers a Lambda. The Lambda:
1. Detects file type (PDF, DOCX, image).
2. PDF → text via `pdfplumber` or AWS Textract for complex layouts.
3. DOCX → text via `python-docx`.
4. Image scans → AWS Textract OCR.
5. Falls back to a stronger OCR (Tesseract + pre-processing) if Textract confidence is low.

If parsing fails (corrupted file, encrypted PDF), we send the candidate an email asking for a clean version. Don't silently drop resumes — losing one strong candidate is worse than processing a few junk files.

**Anonymizer + Section Extractor.** This is the most important and most often-skipped step. We pass parsed text through an LLM (Claude Haiku for cost) with a structured-output prompt that:
1. **Removes PII**: name, photo, address, phone, email, gender markers, age, ethnicity hints.
2. **Extracts sections**: summary, skills, experience (with companies and durations), education, projects, certifications.
3. **Returns structured JSON** with each section.

The reasons for anonymization are twofold. First, fairness: scoring against name or photo introduces bias proven to hurt diversity. Second, quality: the model focuses on actual qualifications instead of pattern-matching name origins. Some jurisdictions (UK, parts of EU) require this.

**Skills Extractor.** Separate from the LLM call, we run a fine-tuned NER model trained on a skills taxonomy (15-20K canonical skills like "Python", "AWS Lambda", "PyTorch", "Kubernetes"). NER lets us match canonical skills despite spelling variations ("k8s" → "Kubernetes", "py-torch" → "PyTorch"). The output is a normalized skill array stored alongside the resume.

**Why a separate skills extractor?** Two reasons. First, hard filters like "must have AWS" or "min 5 years Python" are clean over a normalized array, not over free text. Second, recruiters often search structured queries — "Find me senior Python engineers with Kubernetes" — which is fast over GIN-indexed arrays and slow over free text.

**Embedding Model.** For each section (summary, skills concat, each experience block), embed using a strong open-source bi-encoder like BGE-large-en-v1.5 or multilingual-e5-large for non-English resumes. We embed sections separately rather than the whole resume because:
1. A senior candidate's most-recent experience is more relevant than their first job.
2. JD-side queries usually have different sub-aspects (skills required, experience needed, education).
3. We can later weight sections differently (recent experience > old experience).

### 4.2 The storage layer

**Postgres for metadata.** Every resume has a row: `resume_id`, `candidate_anon_id`, `parsed_text`, `sections` (JSONB), `skills` (text[]), `experience_years` (numeric), `upload_ts`, `is_active` (bool, true if within 6-month window). Indexes: GIN on `skills`, B-tree on `upload_ts`, partial index on `is_active`.

**pgVector for embeddings.** Per-section vectors stored in a `resume_vectors` table: `resume_id`, `section_type` (summary/skills/experience), `embedding` (vector(1024)). HNSW index for ANN search, with operator class `vector_cosine_ops`.

Why pgVector over a dedicated vector DB? Three reasons specific to this system. First, scale — 1.8M vectors fits comfortably in pgVector with HNSW. Second, the metadata and the vectors live in one database, so we get transactional consistency between resume metadata updates and vector updates. Third, hybrid search — combining structured filters (skills, experience years) with vector search is a single SQL query, not a multi-system orchestration. If we ever scaled past 50M vectors I'd revisit and consider Qdrant or Milvus.

### 4.3 The query path — the heart of the design

**Cache layer.** When an HR user opens "JD #1234, top 100," we first check Redis: `jd_1234_top100`. If present and fresh (TTL 6 hours), return immediately. This makes the common case sub-50ms.

**Cache invalidation.** Trickier. We invalidate when:
1. The JD itself is updated (recruiter edits requirements).
2. New resumes have been ingested in the last hour (incremental — not on every upload, batched).
3. Recruiter actions on the current ranking (shortlist, reject) — we want fresh rankings on next open.

We use a TTL of 6 hours plus event-driven invalidation. If staleness becomes an issue we shorten the TTL.

**Stage 1 — ANN retrieval (top-500).** When the cache misses, we embed the JD (cached itself, since JDs are reused) and run an HNSW similarity search against the resume vectors. We retrieve top-500 (broad funnel) using a weighted combination of section similarities — recent experience weighted higher than education, for example. The query in pgVector looks like:

```sql
SELECT r.resume_id,
       1 - (rv.embedding <=> :jd_embedding) AS sim_score
FROM resume_vectors rv
JOIN resume_meta r ON r.resume_id = rv.resume_id
WHERE r.is_active = true
  AND rv.section_type = 'summary'
ORDER BY rv.embedding <=> :jd_embedding
LIMIT 500;
```

**Stage 2 — Hybrid filter.** We apply hard requirements from the JD: must-have skills, minimum years of experience, location constraints if any. This is a SQL filter on the metadata, not a learned model:

```sql
SELECT r.resume_id, r.sim_score
FROM stage1_results r
WHERE r.skills @> :must_have_skills
  AND r.experience_years >= :min_years
ORDER BY r.sim_score DESC
LIMIT 200;
```

The funnel narrows from 500 to ~200 here, depending on how strict the filter is.

**Stage 3 — Cross-encoder reranking.** Bi-encoder embeddings (Stage 1) are fast but coarse. A cross-encoder reranker scores (JD, resume) pairs jointly with much higher precision. We use BGE-reranker-large or Cohere's reranker, served via vLLM or as an API call. We rerank the 200 candidates and keep top-100.

The reranker is the single biggest quality lever in this design. Without it, embedding-only retrieval routinely surfaces resumes that are surface-similar but actually irrelevant. With it, we typically see a 15–25 NDCG@100 lift.

**Stage 4 (optional) — LLM-generated explanations for top-10.** For the top 10 resumes, we ask an LLM (Claude Haiku for speed) to produce a 2-sentence explanation: "This candidate matches because they have 6 years of Python and 3 years of AWS Lambda experience, but they lack the Kubernetes background mentioned in the JD." HR loves this — it makes the system feel transparent and helps them shortlist faster.

We cap explanations at top-10 because each LLM call costs cents and we don't want to pay for explanations on resumes the recruiter will never read.

### 4.4 Putting it all together — total latency budget

```
   Cache hit:     5ms  (Redis)
   Cache miss:
     - JD embedding (cached):       2ms
     - Stage 1 ANN search (top-500): 50ms
     - Stage 2 hybrid filter:        10ms
     - Stage 3 cross-encoder rerank: 200ms (200 pairs × ~1ms each)
     - Stage 4 LLM explanations:    400ms (parallel calls for 10)
                                    ────
     - Total p99 cold:               ~700ms (under 1s)
```

The cache hit path is dominant. We get sub-second p99 even on cold hits, sub-50ms on warm.

---

## 28.5 Step 5 — Challenges and how to resolve each

### Challenge 1 — Resume format chaos

Resumes come in PDFs (digital and scanned), DOCX, even image-only formats. Even within "PDF," layouts vary wildly — single-column, two-column, with embedded tables, with graphics, with non-Unicode fonts.

**Resolution:** A multi-tier parsing pipeline:
1. Try `pdfplumber` first (fast, handles most digital PDFs).
2. If extracted text is below a threshold (say <500 chars for a 1-page resume), it's probably a scan — fall back to AWS Textract.
3. If Textract confidence is low, fall back to Tesseract with image pre-processing (deskew, denoise, threshold).
4. If all parsers fail, log the resume to a manual-review queue and email the candidate for a re-upload.

We track parser success rates per format in Datadog. Any drop alerts the team.

### Challenge 2 — Bias and fairness

A naive system trained on historical hiring data inherits historical biases. Names, photos, school prestige all leak protected attributes. Beyond ethical issues, biased rankings damage recruiting outcomes (less diverse candidate pool) and can violate employment law in certain jurisdictions.

**Resolution:** Several layers of mitigation.
1. **Anonymization at ingest.** PII removed before anything else touches the resume.
2. **No demographic features in the model.** Skills, experience, education topic — yes. Names, photos, addresses — never.
3. **Fairness audits.** Quarterly evaluation of ranking outcomes across demographic groups (we still know the demographics from a separate compliance dataset, just not used for ranking). Look for disparate impact: are similarly-qualified candidates from underrepresented groups being ranked similarly? If not, investigate.
4. **Counterfactual testing.** For a sample of resumes, swap demographic markers and verify the ranking is invariant.
5. **Reranker training data.** Make sure the recruiter-feedback data we fine-tune on isn't itself biased — for example, recruiters might historically have shortlisted only specific schools, which the model would learn. We weight feedback by recruiter, not aggregate, to dampen this.

### Challenge 3 — Cold-start for new JDs

The first time an HR posts a brand-new JD, there's no cache, no recruiter feedback, no pre-computed top-100. The pipeline has to run cold and may give weaker rankings until we accumulate feedback.

**Resolution:**
1. The Stage 1+2+3 pipeline already gives a reasonable cold-start ranking with no historical data. The bi-encoder + reranker combination doesn't need JD-specific training.
2. Within the first day, recruiter shortlist actions provide signal we can use to dynamically reweight the ranking — promote resumes similar to shortlisted ones, demote those similar to rejected ones, in real time per JD. This is a small per-JD model on top of the global ranker.
3. After accumulating enough recruiter feedback (say 1000 actions), we retrain the reranker quarterly with the new signal.

### Challenge 4 — Multilingual resumes

Some candidates submit in Arabic, Hindi, Spanish. The English-trained embedding model treats these as garbage.

**Resolution:** Use `multilingual-e5-large` or `bge-m3` as the embedding model — they're trained on dozens of languages and produce embeddings in a shared space, so an Arabic resume can match an English JD if the skills and experience align. Alternatively, machine-translate the resume to English at ingest time using a translation API (more brittle, but allows monolingual embedding model). For Avrioc specifically — given Comera's MENA focus and Labaiik's UAE customer base — multilingual-from-day-one is the right choice.

### Challenge 5 — Stale resumes

A 6-month-old resume may have outdated employment status (the candidate has since taken a job elsewhere). Ranking them as "active" wastes recruiter time and frustrates candidates who get spammed by recruiters about positions they've moved past.

**Resolution:**
1. **6-month sliding window.** Resumes older than 6 months are filtered out by default unless the candidate explicitly refreshes them.
2. **Refresh nudges.** Email candidates whose resumes are 5 months old, asking them to confirm or update.
3. **Implicit signals.** If a candidate stops opening recruiter emails, their resume's "active score" is lowered.

### Challenge 6 — Duplicate resumes

Candidates submit slightly different versions of their resume to multiple roles. The system shouldn't show 5 near-duplicates.

**Resolution:** Deduplication at ingest time by hashing the structured-extraction output. If two extracted JSON payloads (anonymized, structured) are >95% identical by content, we keep only the most recent and link the older versions to the same `candidate_anon_id`.

### Challenge 7 — Skill taxonomy drift

The skill taxonomy (15-20K skills) needs to keep up with the industry. "Llama-3" wasn't a skill in 2023; "vLLM" wasn't in 2022. The NER skill extractor needs ongoing updates.

**Resolution:**
1. **Quarterly taxonomy refresh.** Mine top trending tokens from incoming JDs and resumes; add new ones as canonical skills.
2. **Aliases and abbreviations.** Maintain a mapping (k8s ↔ Kubernetes, py-torch ↔ PyTorch).
3. **Long-tail skills.** When a skill appears <10 times, don't add it to the taxonomy — keep it in a separate "raw skills" array for matching but don't standardize.

### Challenge 8 — Recruiter feedback data quality

Recruiters' shortlist/reject decisions are noisy. They may shortlist for non-qualification reasons (interviewer availability, rushing through). They may reject for reasons not visible to the system (failed phone screen).

**Resolution:**
1. Weight shortlist actions higher than reject actions (rejects are noisier).
2. Use recruiter-specific calibration — some recruiters shortlist 80%, others 20%, normalize.
3. Use multi-stage signals: shortlist → phone screen passed → onsite passed → offer extended → accepted. Each later stage is stronger signal.
4. Don't overfit to feedback. The reranker should still be informed primarily by JD-resume content match; feedback is a small adjustment, not the entire signal.

### Challenge 9 — Update latency and consistency

When a resume is uploaded, how quickly does it become rankable? When a JD is updated, when do existing rankings reflect the change?

**Resolution:**
1. **New resumes:** ingestion pipeline is async. Embedding takes ~5 seconds per resume in batched mode, so a new resume is rankable within minutes. Acceptable for HR — they're not refreshing the page every second.
2. **JD updates:** invalidate the JD's cache entry on update. Next portal-open re-runs the ranking from scratch.
3. **Inconsistency window:** if a JD update fires while a ranking is in flight, the in-flight ranking might be against the old JD. We accept this — it's bounded to seconds. If we needed strict consistency, we'd version the JD and tag every cache entry with the version.

### Challenge 10 — Rare skills, niche JDs

For very specialized roles (say "Senior Reinforcement Learning Engineer with PPO experience"), the vector search may struggle because the corpus has few qualified candidates and the embedding may not capture the specific specialization.

**Resolution:**
1. **Domain-specific reranker fine-tuning** for high-volume specialized roles. After enough recruiter feedback, train a per-role-family reranker.
2. **Skill-strict mode.** Allow recruiters to mark certain skills as "must have, exact match." The system filters strictly on those before ranking.
3. **Active learning loop.** When a resume is shortlisted for a niche JD, similar resumes get a relevance boost in future rankings.

---

## 28.6 Step 6 — How you evaluate this system

Evaluation has three layers, mirroring the offline-online pattern from Chapter 27.

### Offline evaluation — golden eval set

Build a curated set of (JD, resume, gold-relevance) triples. For each JD, recruiters label 50-100 resumes with graded relevance: 0 (irrelevant), 1 (somewhat), 2 (clearly qualified), 3 (top candidate).

Run the full ranking pipeline on this golden set. Compute:

- **NDCG@100** — primary metric. Should be > 0.7 for a healthy system.
- **Precision@10** — for the top resumes shown, are most actually qualified?
- **Recall@100** — of the truly qualified candidates, what fraction did we surface in top-100?
- **Hit Rate@10** — fraction of JDs where at least one truly-qualified candidate appears in top-10.

### Recruiter-action online evaluation

Once in production, instrument every recruiter action: open JD ranking, view resume, shortlist, reject, schedule interview, hire. Log every event with `(jd_id, resume_id, ranking_position, action, timestamp)`.

From this data, compute:

- **Click-through rate at rank** — fraction of resumes at each rank that are opened.
- **Shortlist rate at rank** — fraction shortlisted.
- **Hire rate at rank** — the gold standard, but with long delay (weeks-to-months).

A well-ranked system has shortlist rates that decay smoothly with rank — top-10 shortlisted at 30-50%, ranks 50-100 at 5-10%. A flat decay means the model isn't learning what recruiters value; a steep decay means we're overfitting.

### A/B testing for ranker changes

When we change anything — embedding model, reranker, weights — we A/B test. Half of HR users get the new ranker, half the old. Compare:

- Time-to-shortlist (faster is better)
- Number-of-resumes-viewed-per-shortlist (lower is better — recruiter spends less time)
- Long-term: time-to-hire and offer-acceptance rate

A/B tests run for at least 2 weeks because hiring cycles are long. Don't conclude on day 3.

### Bias / fairness audit

Quarterly: evaluate ranking quality stratified by demographic groups (using the compliance dataset, not the model's view). The shortlist rate should be roughly equal across qualified candidates from different groups. Disparate impact triggers an investigation.

### Drift monitoring

- **JD embedding drift:** monitor the distribution of incoming JDs. A new business launching new role types shifts the distribution; alert and refresh eval set.
- **Resume embedding drift:** new tools and technologies appear in resumes (the "vLLM" example earlier). Alert when new clusters emerge.
- **Recruiter behavior drift:** a new recruiter joining changes the shortlist patterns. Per-recruiter calibration handles most of this; bulk drift alerts on regime change.

---

## 28.7 Step 7 — Tech stack summary (tied to Avrioc-relevant tools)

| Layer | Choice | Why |
|-------|--------|-----|
| Storage (metadata) | Postgres | ACID, GIN indexes for skill arrays, transactional consistency with vectors |
| Storage (vectors) | pgVector | 1.8M scale fits comfortably, single DB to operate, hybrid SQL queries |
| Object storage | S3 | Raw resume artifacts, lifecycle rules to archive >6 months |
| Compute (parsing) | Lambda | Bursty workload, varying file sizes, stateless |
| Compute (embedding) | EC2 GPU (g5.xlarge) or SageMaker | Batch embedding of new resumes; 30-50 RPS achievable |
| Compute (reranker) | Same GPU pool | Cross-encoder inference, cached results |
| Compute (LLM explanations) | Claude API or vLLM-served Llama | Top-10 only, low volume |
| Cache | Redis | jd_id → top-100 cached rankings, 6h TTL |
| Application | FastAPI | Standard Python ML API choice |
| Orchestration | Airflow | Daily eval, weekly fairness audit, quarterly reranker fine-tuning |
| Vector index | HNSW (M=16, efConstruction=200, efSearch=64) | Good recall/latency balance at this scale |
| Embedding model | BGE-large-en-v1.5 + multilingual-e5-large | English baseline + multilingual support |
| Reranker | BGE-reranker-large or Cohere | Either open-source or hosted |
| Anonymizer | Claude Haiku via API | Cost-effective LLM for structured extraction |
| Skills extractor | Fine-tuned DistilBERT for NER | Fast, accurate against custom taxonomy |
| Monitoring | Datadog + Prometheus + Grafana | Application metrics, drift, eval scores |
| CI/CD | GitHub Actions + Terraform | DAG tests, Docker builds, infra-as-code |

---

## 28.8 The interview-ready narration (the way you'd say it out loud)

> "Let me clarify a few things before designing. (Asks clarifying questions.) OK, so we're building a multi-JD, sub-second-ranking, anonymized resume scoring system at roughly 1K resumes per day — let's design for 10K to be safe.
>
> Quick capacity math: 10K resumes per day, 6-month sliding window means about 1.8 million active resumes. With BGE-large embeddings at 1024 dimensions, that's about 7 GB raw, much less with PQ — easily fits in pgVector on a single-node Postgres. So this is a small-to-medium system, not huge.
>
> The architecture splits into ingestion, query, and feedback. (Sketches the diagram.)
>
> On ingestion: candidates upload to S3 via a presigned URL, a Lambda parses PDF or DOCX or image scans with a multi-tier fallback, the parsed text goes through an LLM-based anonymizer that strips PII and returns structured JSON with sections — summary, skills, experience, education. The anonymization is critical for both fairness and quality. A NER skills extractor produces a normalized skill array, and a bi-encoder embedding model produces per-section vectors. Everything lands in Postgres plus pgVector.
>
> On the query side: when HR opens the portal we first check Redis for `jd_id, top-100`. Cache hit is sub-50ms. On cache miss, we run a four-stage funnel — Stage 1 is HNSW ANN retrieval to top-500, Stage 2 is hybrid filtering on must-have skills and experience years to top-200, Stage 3 is cross-encoder reranking with BGE-reranker to top-100, and an optional Stage 4 generates LLM explanations for the top-10. Total p99 on cold is around 700ms; cache hits are sub-50ms.
>
> On the feedback lane: every recruiter action — shortlist, reject, hire — is logged. Nightly we compute NDCG@100 against the shortlist labels and run drift monitoring on JD and resume embeddings. Quarterly we fine-tune the reranker on accumulated feedback.
>
> The biggest challenges are bias mitigation through anonymization and counterfactual testing, multilingual support via multilingual-e5, parsing robustness with multi-tier fallback, cold-start for new JDs handled by the bi-encoder-plus-reranker design that doesn't need per-JD training, and feedback noise handled by weighting later stages of the hiring funnel higher.
>
> Evaluation: offline NDCG@100 on a recruiter-curated golden set, online click-through-and-shortlist rates by rank, A/B tests over two-week windows when changing rankers, plus quarterly fairness audits.
>
> Stack: pgVector for storage simplicity at this scale, FastAPI for the application layer, Airflow for daily eval and quarterly retraining, Datadog for drift monitoring."

That's about 4 minutes of narration — perfect for the opening of a 45-minute system design discussion. The interviewer will then pull on specific threads, and you respond from the depth in the rest of this chapter.

---

## 28.9 Likely interviewer follow-up questions

### Q1. Why pgVector and not Qdrant or Pinecone?

At 1.8M vectors, pgVector handles the workload comfortably with HNSW. The win is operational: metadata and vectors live in the same Postgres database, so we get ACID consistency between resume updates and vector updates, and hybrid queries (skill filter + vector search) become a single SQL query rather than orchestrating across two systems. Specialized vector DBs win at much higher scale — say 50M+ vectors — where Postgres connection limits and HNSW build times become bottlenecks. For this scale, the simpler architecture wins. If we scaled to 50M vectors I'd revisit and probably move to Qdrant.

### Q2. How would you handle a brand-new JD that has no candidates?

Two paths. First, the ranking pipeline already works cold — bi-encoder embedding plus cross-encoder reranker doesn't need JD-specific training, it generalizes from the embedding space. So even a totally new JD gets reasonable rankings immediately. Second, recruiter feedback in the first few hours dynamically reweights — if the recruiter shortlists a couple of resumes, similar resumes get promoted in real time. After accumulating ~1000 feedback events across all JDs of the same role family, we incorporate that into the next reranker fine-tune.

### Q3. What happens when 100 HR users open the portal at the same time on a Monday morning?

Cache layer absorbs most of it. A single JD is usually being looked at by multiple recruiters, so cache hit rate stays high. For cache misses, the FastAPI ranking service is stateless and scales horizontally — KEDA on Kubernetes scales replicas based on Prometheus queue depth, not CPU. The bottleneck would be the cross-encoder reranker; we keep a pool of GPU-backed reranker pods and queue requests with a 200-ms batch wait. At sustained 100 QPS we'd want maybe 3-4 GPU pods, which is cheap.

### Q4. How do you ensure the reranker doesn't get worse with each retraining?

Three guards. First, every reranker candidate is evaluated against the frozen golden set offline; if NDCG@100 regresses below the production baseline, we don't deploy. Second, we A/B test new rerankers in production for at least two weeks before rolling fully. Third, we monitor shortlist rate by rank in production after rollout — a steep drop in click-through at top ranks is a real-time alert that something's off. Auto-rollback on guardrail breach.

### Q5. What if a candidate is mismatched in the data — say their resume parses but they listed a skill they don't actually have?

The system ranks based on stated qualifications; we can't verify them. The phone screen and interview stages are the verification. What we can do is alert recruiters when ranking confidence is borderline — a resume that ranks 80th-100th has a much higher false-positive rate than one ranked 1st-10th. The UI can show this confidence band so recruiters allocate verification effort accordingly.

### Q6. How do you prevent the system from being gamed — candidates stuffing keywords from JDs into their resumes?

Three mitigations. First, the system uses semantic matching, not keyword matching, so naive keyword stuffing doesn't help much — the embeddings reflect actual context. Second, the reranker stage uses a cross-encoder that scores joint relevance, which is much harder to game. Third, we monitor for resumes with anomalous skill-density (e.g., a resume claiming 50 distinct skills) and downweight them. Long-term, recruiter feedback teaches the system that gamed resumes don't pan out.

### Q7. How do you scale to 100K resumes per day?

10x scale. Three changes. First, vector storage at 18M vectors becomes painful for pgVector on a single node — migrate to Qdrant or Milvus with sharding. Second, the embedding pipeline needs a dedicated GPU cluster (Ray Serve LLM with autoscaling). Third, the metadata DB might benefit from read replicas for the query path while the write path goes to the primary. The architecture shape stays the same; we're just swapping single-node components for clustered ones.

### Q8. What are your ranker's hyperparameters and how would you tune them?

Three layers. At the embedding stage: section weights (summary=0.2, recent_experience=0.5, skills=0.2, education=0.1) — tune by grid search against the golden set's NDCG. At the ANN stage: HNSW M, efConstruction, efSearch — efSearch is the at-query lever, higher means better recall but slower; tune to hit the latency budget. At the reranker stage: temperature in cross-encoder scoring, must-have-skill weighting, recency boosting — these are usually small lookup-table parameters tuned by recruiter feedback. Fine-tuning the reranker itself happens quarterly with accumulated feedback.

### Q9. Walk me through what happens when an HR user marks a resume as "unfair to skip" — meaning they think the system should have ranked it higher.

Log the event with `(jd_id, resume_id, current_rank, recruiter_id, "promote")`. In the short term, the cache for that JD is invalidated, and on the next portal open the resume's score gets a recruiter-specific boost from a small per-recruiter calibration model. Over the medium term, this feedback enters the reranker training data — paired with the JD as a positive example. The recruiter sees the resume promoted in the next ranking refresh, which closes the feedback loop and signals the system is responsive to their input.

### Q10. What's the business metric you'd ultimately optimize for?

Time-to-hire and quality-of-hire are the only metrics that really matter. Time-to-hire is straightforward — calendar days from JD-posted to offer-accepted. Quality-of-hire requires longer feedback — 6-12 month employee performance ratings, retention. We'd track both, with NDCG and shortlist rates as proxies in the short term but business metrics as the ground truth. If our ranking improvements don't translate to faster hiring or better hires, we're just optimizing surrogates.

---

## 28.10 The cheatsheet — what to remember from this chapter

```
   Capacity:          1K/day → 10K/day design, 6-month window ≈ 1.8M active
   Storage:           pgVector at this scale; Qdrant if 10x growth
   Architecture:      ingestion + query + feedback lanes
   Anonymizer:        REMOVE PII before any scoring (fairness + quality)
   Skills:            normalized array via fine-tuned NER
   Embedding:         per-section, weighted (recent experience > education)
   Query funnel:      cache → ANN top-500 → filter to 200 → rerank to 100
                      → optional LLM explanation for top-10
   Latency:           sub-50ms cache hit, ~700ms cold p99
   Multilingual:      multilingual-e5 or bge-m3 from day one
   Bias controls:     anonymize, audit quarterly, counterfactual test
   Cold-start:        bi-encoder + reranker generalizes; recruiter feedback
                      re-weights in real time per JD
   Eval:              offline NDCG@100 + online shortlist-rate-by-rank +
                      A/B tests for ranker changes + fairness audit
   Business metric:   time-to-hire, quality-of-hire — NDCG is a proxy
```

---

End of Chapter 28. Continue to **[Chapter 29 — Ensemble Models](29_ensemble_models.md)** or back to **[Chapter 00 — Master Index](00_index.md)**.
