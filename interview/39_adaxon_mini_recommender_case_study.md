# Chapter 39 — Case Study: Adaxon Mini-Recommender (Take-Home Walk-Through, End-to-End)

> **Why this chapter exists:** This is a complete walk-through of a real take-home submission for a Senior ML Engineer role (MobUpps, AI Department) — a containerised app-recommendation service. In interview rounds the candidate will be asked to defend every architectural decision, explain every line, and modify the code live. This chapter is the speaking script: the problem, the data, the model, the service, the tests, the operational story, the feature engineering, the failure modes, and the questions an interviewer is most likely to ask.
>
> **The repo:** [github.com/sachincse/adaxon-mini-recommender](https://github.com/sachincse/adaxon-mini-recommender) (public).
> **Source brief:** *Adaxon Mini-Recommender — Take-Home Assignment for Senior ML Engineer role*, MobUpps AI Department. Time budget 6–8 hours of focused work. Submission as a private Git repo. Evaluation criteria: engineering quality, production thinking, clarity of reasoning — not model accuracy.

---

## 39.1 The assignment in one paragraph

Build an HTTP service that, given a `user_id`, returns a ranked list of `n` app recommendations. Data is provided as three CSVs (users, apps, interactions); the catalogue has 500 apps and 10 000 users, with ~490 000 user-app interactions over a 90-day window. The data is intentionally messy (orphan foreign keys, duplicate rows, malformed timestamps, negative durations, blocked-app references). Wrap the service in a Docker container that starts cleanly from a single `docker build` + `docker run`. Provide tests, an evaluate script (optional to use), and a README following an exact section structure (Architecture decisions, Data quality findings, Tradeoffs made, Testing strategy, Production considerations, two-weeks roadmap, time-spent breakdown).

The brief is explicit on three points:
1. *"A simple, well-justified baseline is preferred over a complex model with poor reasoning."*
2. *"A recall of 0.05 with strong reasoning beats a recall of 0.30 with a black-box model and a thin README."*
3. *"Do not invest time in over-engineering. A clean simple system beats a clever complex one."*

This frames the entire submission: every decision should be defensible in terms of engineering hygiene and production thinking, not in terms of pushing offline metrics.

---

## 39.2 The end-to-end architecture (one diagram)

```
   ┌─────────────────┐       ┌─────────────────┐        ┌────────────────────┐
   │ Partner mobile  │──────▶│  Load balancer  │───────▶│  FastAPI container │
   │     /  web      │       │ / reverse proxy │        │                    │
   └─────────────────┘       └─────────────────┘        │  ┌──────────────┐  │
                                                        │  │ middleware:  │  │
                                                        │  │ access log   │  │
                                                        │  │ + req_id     │  │
                                                        │  └──────┬───────┘  │
                                                        │         │          │
                                                        │  ┌──────▼───────┐  │
                                                        │  │ /health      │  │
                                                        │  │ /recommend   │  │
                                                        │  └──────┬───────┘  │
                                                        │         │          │
                                                        │  ┌──────▼───────┐  │
   ┌─────────────────┐                                  │  │ Item-CF      │  │
   │  data/ volume   │── mounted read-only ─────┐       │  │ recommender  │  │
   │ apps.csv        │                          │       │  │  (in mem)    │  │
   │ users.csv       │                          ▼       │  └──────┬───────┘  │
   │ interactions    │                ┌─────────────────┤         │          │
   └─────────────────┘                │ Data layer:     │  ┌──────▼───────┐  │
                                      │ load + clean +  ├─▶│ Sparse U +   │  │
                                      │ DataQuality-    │  │ dense S      │  │
                                      │ Report          │  │ (cosine sim) │  │
                                      └─────────────────┘  └──────────────┘  │
                                            (boot)         └────────────────────┘
                                                                 │
                                                                 ▼
                                                       stdout (JSON lines)
                                                       → Loki / ELK / Datadog
```

**Three load-bearing properties:**
1. **The data layer runs once, at boot.** From that point on, requests are pure CPU on in-memory data structures. There is no per-request DB query, no per-request file I/O, and no per-request lock contention.
2. **The container is the unit of deployment.** Data lives outside the image (mounted at runtime), so the same image runs in dev, staging, and prod against different datasets without rebuild.
3. **Observability is a single structured log line per request.** Anything else (Prometheus, traces) is layered on top in production; the minimal viable observability ships in the first version.

---

## 39.3 The data layer

### 39.3.1 The schemas (what the CSVs actually contain)

`apps.csv` — 500 rows, columns include: `app_id`, `bundle_id`, `app_name`, `platform`, `status` (active / paused / **blocked**), `category` (21 distinct), `subcategory_1..4`, `size_mb`, `age_rating` (`4+ / 9+ / 12+ / 17+`), `avg_rating` (float 0–5), `total_ratings` (int).

`users.csv` — 10 000 rows, columns include: `user_id`, `country_code`, `language`, `platform` (ios / android), `device_model`, `os_version`, `signup_date`, `age_bucket` (`18-24` / `25-34` / `35-44` / `45-54` / `55+`).

`interactions.csv` — ~490 k rows, columns include: `user_id`, `app_id`, `event_time` (ISO 8601), `event_name` (`af_view`, `af_click`, `af_install`, `af_open`, `af_purchase`, `af_complete_registration`, `af_level_achieved`), `interaction_type` (the normalised form: `view` / `click` / `install` / `open`), `duration_seconds`.

### 39.3.2 The data quality findings

Running `pandas` diagnostics against the raw CSVs produces these counts. Every count surfaces in the startup log line via a `DataQualityReport` dataclass — silent drops are unacceptable.

| Issue | Count | Decision | Rationale |
|---|---:|---|---|
| Interactions referencing a `user_id` not in `users.csv` | 245 | **Drop** | Orphan rows: cannot apply any user-side filter or feature without a profile. |
| Interactions referencing an `app_id` not in `apps.csv` | 199 | **Drop** | Cannot filter by status or apply content features without catalogue metadata. |
| Interactions on apps with `status == blocked` | 64 678 (16% of all rows) | **Drop** | **Most user-visible failure mode.** Recommending a blocked app to a real user is the worst regression possible. The single most important line of code in `data.py`. |
| Duplicate `(user_id, app_id, event_time, type)` rows | 1 316 | **Drop** | Inflates weights, skews similarities. Treats the natural key as a unique key. |
| Malformed `event_time` | 493 | **Drop** | `pd.to_datetime(..., errors='coerce')` then `dropna` on the timestamp. |
| `duration_seconds < 0` | 325 | **Clip to 0** | The row itself (the *fact* the interaction happened) is still informative; only the duration field is broken. Clipping preserves the implicit-feedback signal. |
| `duration_seconds is null` | 6 542 | **Keep + flag** | Duration is not used as a feature today. Flagging it in the report leaves the option open. |
| `apps.csv` rows with `status == paused` | 55 | **Keep + flag** | Paused apps may return; staying in the catalogue is the right default. |
| `users.csv` rows missing `country_code` | 30 | **Keep + flag** | Country is not used by the current model; flagged in case a content-blend feature lights it up later. |
| `event_time` in the future | many | **Keep** | Synthetic dataset artefact; in production this would be a clock-skew alert and the affected partition would be quarantined. |

**Cleaning order matters.** The pipeline filters in this exact sequence: blocked apps from the catalogue → orphan FKs → duplicates → bad timestamps → unknown event types → negative-duration clip. Reordering can produce different drop counts (e.g. counting blocked-app interactions under "unknown app_id" if you filtered the catalogue late).

### 39.3.3 The `DataQualityReport` pattern

```python
@dataclass
class DataQualityReport:
    apps_total: int = 0
    apps_blocked_filtered: int = 0
    apps_paused_kept: int = 0
    interactions_total: int = 0
    interactions_blocked_app_dropped: int = 0
    interactions_unknown_user_dropped: int = 0
    interactions_unknown_app_dropped: int = 0
    interactions_duplicate_dropped: int = 0
    interactions_bad_timestamp_dropped: int = 0
    interactions_negative_duration_clipped: int = 0
    interactions_missing_duration_kept: int = 0
    interactions_unknown_event_dropped: int = 0
    interactions_final: int = 0
    issues: list[str] = field(default_factory=list)
```

The report is logged at startup as a structured field:

```json
{"ts":"2026-06-01T...","level":"INFO","logger":"src.recommender.data",
 "msg":"data_cleaned","report":{"apps_total":500,"apps_blocked_filtered":66,
 "interactions_total":490382,"interactions_blocked_app_dropped":79916, ...}}
```

A reviewer can grep one line and see every rule's contribution.

---

## 39.4 The model — Item-based collaborative filtering

### 39.4.1 The model in one paragraph

Build a sparse user × item matrix `U` where `U[u, i] = Σ event_weight(t)` for every interaction `t` of user `u` on item `i`. Event weights are `view=1, click=2, open=3, install=5`. L2-normalise each column of `U`, then compute the item-item cosine similarity matrix `S = U_norm^T · U_norm` (dense float32, ~1 MB at 379 items). For inference, the score for user `u`'s row is `s = u · S`; mask the items already in the user's history (set their scores to `-inf`), then `argpartition` the top-N indices.

### 39.4.2 Why this model and not something fancier

The brief explicitly prefers a simple baseline with strong reasoning. Even setting that aside, the data scale doesn't justify a heavier model:

| Option | Why rejected |
|---|---|
| **Popularity-only baseline** | The brief includes this as a baseline; outperformed by item-CF on offline metrics, no per-user signal. |
| **Matrix factorisation (ALS / BPR)** | Adds a heavy dep (`implicit` or `lightfm`), needs hyperparameter tuning, marginal quality lift at 500 items. Operational cost > benefit. |
| **Two-tower neural network** | Requires GPU at inference (otherwise too slow), introduces training-loop failure modes, and is wildly disproportionate for 500 × 10 000 data. |
| **Content-only similarity** | Throws away the strongest available signal (behaviour). Useful as a *blend* on top of CF (see § 39.7 Feature Engineering), not as the primary model. |
| **Item-cosine CF (chosen)** | Interpretable ("users who use X also use Y"), no heavy deps, sub-millisecond inference, deterministic training in ~6 s on the full dataset. |

### 39.4.3 The implementation, annotated

```python
class ItemCoocRecommender:
    def fit(self, interactions, catalogue_app_ids, item_features=None):
        # 1. Filter to catalogue apps only (blocked apps already removed upstream)
        df = interactions[interactions["app_id"].isin(set(catalogue_app_ids))].copy()

        # 2. Map event type → numeric weight
        df["weight"] = df["interaction_type"].map(EVENT_WEIGHTS).fillna(1.0)

        # 3. (Optional) time decay — multiplies weight by 0.5 ** (age / half_life)
        if self.time_decay_half_life_days > 0:
            ev = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
            if ev.notna().all():
                age = (ev.max() - ev).dt.total_seconds() / 86400
                df["weight"] *= 0.5 ** (age / self.time_decay_half_life_days)

        # 4. Aggregate per (user, app); sum weights across multiple events
        per_ui = df.groupby(["user_id","app_id"])["weight"].sum().reset_index()

        # 5. Drop low-support items (noise reduction)
        support = per_ui.groupby("app_id")["user_id"].nunique()
        kept = support[support >= self.min_item_support].index
        per_ui = per_ui[per_ui["app_id"].isin(set(kept))]

        # 6. Build sparse U (CSR)
        # ... assign indices, scipy.sparse.csr_matrix construction ...

        # 7. Compute item-item cosine sim
        col_norms = np.sqrt((U.multiply(U)).sum(axis=0)).ravel()
        col_norms[col_norms == 0] = 1.0
        U_norm = U.multiply(1.0 / col_norms)
        S_cf = (U_norm.T @ U_norm).toarray().astype(np.float32)
        np.fill_diagonal(S_cf, 0.0)   # never recommend an item as similar to itself

        # 8. (Optional) blend in content-based similarity
        if self.content_blend_alpha > 0:
            S_content = self._content_similarity(item_features)
            self._item_sim = ((1 - α) * S_cf + α * S_content).astype(np.float32)
        else:
            self._item_sim = S_cf

        # 9. Build popularity list for cold start
        self._popularity = (
            df.groupby("app_id")["user_id"].nunique().sort_values(ascending=False)
            .index.tolist()
        )
```

```python
def recommend(self, user_id: str, n: int) -> list[str]:
    u_idx = self._user_index.get(user_id)
    if u_idx is None:
        return self._popularity[:n]            # unknown user → popularity

    user_row = self._user_item.getrow(u_idx)
    if user_row.nnz == 0:
        return self._popularity[:n]            # known user, no usable history

    scores = (user_row @ self._item_sim).ravel()
    scores[user_row.indices] = -np.inf         # mask already-seen items

    top_unsorted = np.argpartition(-scores, n - 1)[:n]   # O(n), not O(N log N)
    top = top_unsorted[np.argsort(-scores[top_unsorted])]

    recs = [self._items[i] for i in top if scores[i] > 0]
    if len(recs) < n:                           # top up from popularity if needed
        # ... add popular items not in history and not already in recs ...
    return recs
```

**Two implementation notes worth mentioning in interview:**
- **`argpartition` not `argsort`.** Sorting all 379 scores is wasteful when only the top 10 matter. `argpartition(-scores, n-1)[:n]` is O(n), then we sort just those n. The diff at 379 items is small; at 50 000+ items it matters a lot.
- **Mask history with `-inf`, don't slice.** Setting indices to `-inf` keeps the score vector intact for `argpartition`. Slicing the score vector requires reindexing back to item ids.

### 39.4.4 Cold start handling

```
            ┌─────────────────────┐
            │ recommend(user, n)  │
            └──────────┬──────────┘
                       │
              ┌────────▼─────────┐
              │ user in index?   │
              └────┬────────┬────┘
              No   │        │  Yes
                   ▼        ▼
       popularity[:n]   user_row.nnz > 0?
       source: popularity      │   │
                          No   │   │  Yes
                               ▼   ▼
                       popularity[:n]      scores = u · S
                       source: popularity  mask history
                                           argpartition top-n
                                           top up w/ popularity if needed
                                           source: model
```

The choice to *fall back* rather than 404 is intentional: a calling app needs *something* to render, and a popularity slate is the safest default for a new-user experience. The response includes a `source` field (`"model"` or `"popularity"`) so the caller can render the right header ("recommended for you" vs "popular near you") and product can A/B test cold-start strategies.

---

## 39.5 The HTTP service

### 39.5.1 The contract

```
GET /health
  → 200 {"status":"ok","model_fitted":true,"n_items":379,"n_users":9934}
  → 503 {"detail":"model not ready"}   (during cold boot, before fit completes)

GET /recommend?user_id=<str>&n=<int>
  → 200 {"user_id":"u_000123","n":10,
         "recommendations":["a_0487","a_0315",...],
         "source":"model"}             (personalised — known user)
  → 200 {"user_id":"u_brand_new","n":5,
         "recommendations":[...],"source":"popularity"}
                                       (cold-start fallback, NEVER 404)
  → 422 {"detail":[{"type":"missing","loc":["query","user_id"], ...}]}
                                       (input validation)
```

The brief mandates only `{user_id, recommendations: [...]}`. The extras (`n`, `source`) are documented diagnostics — `source` in particular is the lever for cold-start A/B testing in production.

### 39.5.2 The lifespan + middleware setup

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(SETTINGS.log_level)
    log.info("service_starting", extra={"data_dir": SETTINGS.data_dir})
    app.state.app_state = build_state()    # load + clean + fit, ~6 s
    log.info("service_ready", extra={"stats": app.state.app_state.model.stats()})
    yield
    log.info("service_stopping")

@app.middleware("http")
async def access_log(request, call_next):
    req_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - start) * 1000
        log.exception("request_error",
                      extra={"request_id": req_id, "method": request.method,
                             "path": request.url.path, "latency_ms": round(latency_ms, 2)})
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    log.info("request",
             extra={"request_id": req_id, "method": request.method,
                    "path": request.url.path, "query": str(request.url.query),
                    "status": response.status_code, "latency_ms": round(latency_ms, 2)})
    response.headers["x-request-id"] = req_id
    return response
```

The middleware does four things, in this order:
1. Assigns or echoes a `request_id` (propagates correlation across systems).
2. Times the request.
3. Catches exceptions — turns crashes into log lines, not process kills.
4. Emits exactly one structured JSON line per request.

### 39.5.3 Input validation

Pydantic handles it for free via `Query(..., min_length=1, max_length=128, ge=1, le=MAX_N)`:

```python
@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    request: Request,
    user_id: str = Query(..., min_length=1, max_length=128),
    n: int     = Query(SETTINGS.default_n, ge=1, le=SETTINGS.max_n),
) -> RecommendResponse:
    ...
```

Invalid input → 422 with a structured Pydantic error body that includes `type`, `loc`, and `msg`. A caller can machine-parse the `loc` field to highlight the failing form field.

---

## 39.6 The test suite — philosophy over coverage

15 tests, none of which pin numerical model output (those would break the moment weights change). Each test guards a **contract** the service depends on at runtime:

```
tests/
├── conftest.py         ← tiny synthetic fixture (5 apps, 4 users, ~15 rows)
├── test_data.py        ← 3 tests
│   ├── test_blocked_apps_never_leak_into_cleaned_data    ★ load-bearing
│   ├── test_cleaning_removes_each_garbage_class
│   └── test_paused_apps_kept_but_flagged
├── test_model.py       ← 3 tests
│   ├── test_unknown_user_returns_popularity_and_does_not_crash
│   ├── test_known_user_never_gets_their_own_history_back
│   └── test_recommend_count_is_honoured_and_bounded
├── test_service.py     ← 5 tests via FastAPI TestClient
│   ├── test_health_returns_ok_and_model_stats
│   ├── test_recommend_happy_path
│   ├── test_recommend_unknown_user_uses_popularity_source
│   ├── test_recommend_validates_input
│   └── test_request_id_round_trips
└── test_features.py    ← 4 tests
    ├── test_time_decay_lowers_old_interaction_weight
    ├── test_content_blend_changes_similarity_matrix
    ├── test_content_blend_alpha_validated
    └── test_content_blend_requires_item_features
```

**Test deliberately starred:** `test_blocked_apps_never_leak_into_cleaned_data`. A blocked app appearing in real users' recommendations is the highest user-visible failure mode this service has. It's the test that protects the data layer from a small refactor regressing the most important cleaning rule.

**Three things deliberately NOT tested:**
1. **The FastAPI lifespan hook** (CSV reading) — exercised end-to-end by `docker run`; unit-testing it would couple to the filesystem.
2. **Numerical similarity values** — brittle; would need rewriting any time weights change.
3. **Recall@K / NDCG@K thresholds** — they're a metric, not a contract; pinning them in tests gives false stability.

---

## 39.7 Feature engineering — what was added and why (opt-in)

The baseline is item-CF. Two optional features sit on top, both default off so the baseline behaviour is preserved unless explicitly enabled via environment variable.

### 39.7.1 Time-decay weighting

`weight *= 0.5 ** (age_in_days / half_life)`

A click 90 days ago tells you less about today's preferences than a click last week. The reference point is `max(event_time across the cleaned slice)`, not `now()` — this keeps recommendations stable when the cron-job refit runs at midnight vs noon.

**Defensive behaviour:** if `event_time` is missing or has any unparseable values for *any* surviving row, decay is skipped entirely. A silent partial decay on dirty data is worse than no decay.

### 39.7.2 Content-based item similarity blend

Build an item-feature matrix `F` from:
- one-hot `category` (cardinality ≈ 21)
- `avg_rating / 5` (normalised quality)
- `log1p(total_ratings) / log1p(max)` (log-scaled popularity)
- `size_mb / 500` (normalised footprint)
- `age_rating` mapped to ordinal `{4+:0, 9+:1, 12+:2, 17+:3}` then `/3`

L2-normalise each row, compute `S_content = F · F^T`, blend with the CF similarity:

```
S_final = (1 - α) * S_cf + α * S_content
```

`α = 0` (default) preserves baseline. `α = 0.15` is conservative — enough to differentiate tail items where CF signal is weak, small enough to preserve CF dominance on items with substantial co-occurrence.

### 39.7.3 Worked comparison: baseline vs FE-enabled (user `u_000123`)

| Step | Baseline weight | FE weight (decay) | Notes |
|---|---:|---:|---|
| `a_0031` (3 events, most recent 2026-04-23) | 4.00 | 1.94 | Three events including an old click → decayed |
| `a_0178` (1 click, 2026-04-16) | 2.00 | 1.24 | One mid-recency click |
| `a_0088` (1 view, 2026-04-19) | 1.00 | 0.68 | One mid-recency view |
| `a_0245` (1 view, 2026-04-09) | 1.00 | 0.49 | Older view |
| `a_0457` (1 view, 2026-04-03) | 1.00 | 0.41 | Oldest view |

Top-10 recommendations:

| rank | Baseline | FE-enabled | Notes |
|---:|---|---|---|
| 1–5  | `a_0487, a_0315, a_0278, a_0275, a_0136` | `a_0487, a_0315, a_0136, a_0278, a_0292` | First-pick stable; some swaps within top 5 |
| 6    | `a_0029` | `a_0216` | **New in FE: Shopping / Coupons (adjacent to user's Shopping history)** |
| 7–9  | `a_0292, a_0160, a_0104` | `a_0160, a_0029, a_0104` | Re-ranked |
| 10   | `a_0150` (Travel — outlier) | `a_0185` (Sports / **Fitness Tracking** — matches user's strongest signal) | Category-coherent substitution |

8 of 10 overlap. The two changes are both improvements: Travel and Games outliers replaced by category-coherent picks.

### 39.7.4 Features deliberately NOT added

- **Duration-weighted boost** — a 1-second `open` is not 1/100 the value of a 100-second `open`; calibrating that saturation function is out of scope for the time budget.
- **User-feature vector for cold-start** — country / age cohort could refine the popularity fallback, but doing it responsibly requires an A/B framework. Listed in the "two-weeks roadmap".
- **BPR / negative-sampling** — typically outperforms raw cosine, but requires a training loop and hyperparameter optimisation. Over-engineering for this exercise.

---

## 39.8 Production considerations

### 39.8.1 What to monitor and alert on

| Metric | Source | Alert threshold (suggested) |
|---|---|---|
| P50 / P95 / P99 request latency | structured access log | P95 > 50 ms for 5 min |
| `/recommend` 5xx rate | access log status | > 1% for 5 min |
| `/recommend` empty-recommendation rate | structured field (would add) | > 1% — model gave up on real users |
| Cold-start fallback rate (`source=popularity`) | structured field | > expected baseline + 5 pp — your traffic distribution shifted, or upstream is sending garbage user_ids |
| Model age (time since last refit) | startup log | > 48 h — refit pipeline stalled |
| Container memory usage | runtime | > 75% of limit — dataset growing |

### 39.8.2 Scaling

- **10× traffic:** add replicas. The service is stateless after startup; rolling deploys are cheap (~6 s boot). No tuning needed.
- **100× traffic:** two bottlenecks bite:
  1. **Memory per replica** — every replica holds the model. Solution: move the user × item and item × item matrices into a shared store (Redis or S3 with mmap), refactor the service to a thin scoring layer, push training into a sidecar CronJob.
  2. **Boot time** — reading 80 MB of CSV per replica per restart is wasteful at fleet scale. Move to a pre-built parquet artifact (or the fitted similarity matrix) in object storage. Reduces boot time by an order of magnitude.

Both changes are deliberately *not* made today because the traffic doesn't justify them. Listed in the "two-weeks roadmap".

### 39.8.3 Most likely failure mode

A schema change in upstream data (a new `interaction_type` value, a renamed column) is the most likely thing to break this service in the wild. Today's data layer counts unknown event types in the `DataQualityReport`. The detection plan:

1. **Alert on `interactions_unknown_event_dropped > threshold`** — fires when upstream introduces a new event type.
2. **CI check** that runs `scripts/evaluate.py` against a frozen mini-fixture; fails on a recall-percentage delta beyond ±10%. Catches model regressions that structured logs alone wouldn't.

---

## 39.9 The "two more weeks" roadmap (what's next, in priority order)

1. **Content-based blending for cold items (~3 days)** — fully implemented in this submission as opt-in feature engineering. Next step is to grid-search `(α, half_life)` against the offline harness and ship a sensible default.
2. **Online evaluation harness (~2 days)** — wire the access log into a small batch job that joins `recommend` responses with subsequent `install` / `open` events, computes online CTR / conversion per slot. The only metric that matters in production is "did the user engage with the slot we served them". This unlocks A/B testing for any future model change.
3. **Model artifact pipeline (~2 days)** — train in a CronJob, persist `(U, S, popularity, user_index, item_index)` as a versioned parquet/npz tuple in object storage, have the service mmap from S3 at boot. Cuts cold-start time and lets multiple replicas share the artifact.

---

## 39.10 Things the interviewer will probably ask

### Q1. "Why item-CF and not matrix factorisation?"
Item-cosine CF at 500 items is sub-millisecond inference with no heavy dependency and full interpretability. MF would add a `lightfm` or `implicit` dependency, a hyperparameter surface (regularisation, factors), and a quality lift that is not measurable at this scale. Operationally it's a worse trade. At 50 000+ items it would be the right next step.

### Q2. "Walk me through what happens for an unknown user_id."
- `recommend(user_id, n)` is called.
- `self._user_index.get(user_id)` returns `None`.
- Method returns `self._popularity[:n]` — the top-N most-used apps by unique-user count.
- The HTTP handler sees the user_id wasn't in the index and sets `source = "popularity"` in the response.
- Result: 200 OK, never 404, never 500. Calling app gets a renderable slate.

### Q3. "Why do you mask the user's history with `-inf` rather than dropping those indices?"
Two reasons. First, `argpartition(-scores, n-1)` operates on the full vector — setting masked entries to `-inf` lets them rank last without changing the vector's shape. Second, the score vector indices are item indices; dropping entries would require maintaining a separate reverse mapping. `-inf` is O(1) and keeps the index semantics intact.

### Q4. "How does time decay interact with dirty event_time data?"
Decay is *skipped entirely* if any surviving row has an unparseable `event_time`. The reasoning: a silent partial decay on a dirty column is worse than no decay — you'd get inconsistent weights across users for no auditable reason. The skip is logged so it's debuggable.

### Q5. "What's the most painful production bug this service could have, and how would you detect it?"
A blocked app appearing in real users' recommendations. Detection happens at two layers: (1) the cleaning step filters blocked apps both from the catalogue *and* from the interaction stream before fitting; (2) `test_blocked_apps_never_leak_into_cleaned_data` asserts the cleaned interactions frame is free of blocked-app references. The combination means a regression has to break both the production filter AND get past the test before it ships.

### Q6. "What about cold-start *items* (a new app with no users yet)?"
Not handled by pure CF — a new item with zero rows in `U` has zero cosine similarity to everything. This is exactly the case the content-blend feature was designed for. With `CONTENT_BLEND_ALPHA = 0.15`, a new item still gets a non-zero similarity contribution from its category / rating / size, so it can be surfaced. This is the #1 entry on the two-weeks roadmap.

### Q7. "How would you A/B test a model change?"
Two-step setup:
- **Online harness** — join `recommend` responses (via `request_id`) with subsequent `af_install` / `af_open` events from the interaction stream. Compute CTR / conversion per slot per variant.
- **Routing layer** — split user IDs into buckets (hash mod N), serve variant A or B based on bucket. Keep variants stateless so flipping a single env var changes the model.

The `source` field in the response is already the lever for the simplest possible A/B test (model vs popularity).

### Q8. "Why dense similarity rather than sparse top-k?"
At 379 items the dense `S` matrix is ~575 KB. Sparse top-k would save memory but add lookup complexity. The break-even is somewhere in the thousands of items; well above today's scale.

### Q9. "What's your refit cadence?"
Daily, via a CronJob in the production picture. Inside the container, the fit runs once at startup. In the current single-container deployment, refit = redeploy. In the production picture (§ 39.8), the CronJob persists an artifact and replicas pick it up on next restart.

### Q10. "What did you spend more time on, and what did you cut?"
**Invested time in:** the cleaning report (audit trail), structured access logging with request-id propagation, cold-start path, the worked-example documentation. Those are the things that determine whether the service survives contact with on-call.
**Cut corners on:** model tuning (single weight scheme, no grid search), authentication (assume the gateway handles it), Prometheus metrics export (logged but not exported), online evaluation (no harness yet).

---

## 39.11 Repo layout (for orientation in the interview)

```
adaxon-mini-recommender/
├── README.md            ← the 8-section README the brief mandates
├── TESTING.md           ← reviewer runbook (clone → run → verify)
├── SMOKE_TEST.md        ← PowerShell + bash quick-reference for hitting the service
├── Dockerfile           ← slim-python, non-root, single-stage
├── Makefile             ← build / run / test / eval / shell / clean
├── requirements.txt     ← 8 pinned dependencies
├── docs/
│   ├── EXPLAINER.md     ← deep technical reference (mermaid diagrams + worked example)
│   ├── architecture.drawio
│   ├── screenshots/     ← captured pytest, /health, /recommend, Swagger UI PNGs
│   └── assignment/      ← the original brief PDF
├── src/recommender/
│   ├── config.py        ← env-var driven Settings dataclass
│   ├── data.py          ← load + clean + DataQualityReport
│   ├── model.py         ← ItemCoocRecommender (fit + recommend + content blend)
│   ├── service.py       ← FastAPI app, middleware, lifespan
│   └── logging_setup.py ← JSON formatter
├── scripts/
│   ├── evaluate.py            ← Recall@10 / NDCG@10 harness
│   ├── worked_example.py      ← prints the u_000123 walk-through
│   └── capture_screenshots.py ← regenerates the docs/screenshots PNGs
└── tests/               ← 15 tests across 4 files
```

---

## 39.12 The numbers worth remembering

| Number | What it means |
|---:|---|
| 500 | Source apps in `apps.csv` |
| 66 | Apps with `status=blocked` (removed from catalogue) |
| 379 | Apps remaining after blocked + low-support filter (the recommendable catalogue) |
| 10 000 | Source users in `users.csv` |
| 9 934 | Users with usable interactions after cleaning |
| 490 382 | Raw interactions |
| 407 850 | Interactions remaining after all cleaning rules |
| 79 916 | Interactions on blocked apps — the biggest single drop |
| 1 633 | Exact duplicate interactions removed |
| 263 301 | NNZ in the sparse user-item matrix |
| ~6 s | End-to-end startup time (load + clean + fit) on a laptop |
| < 2 ms | P95 `/recommend` latency once warm |
| 15 / 15 | Tests passing |
| 0.1015 | Recall@10 (offline; baseline mode) |
| 0.1066 | NDCG@10 (offline; baseline mode) |

---

## 39.13 Closing thought

This exercise rewards *clarity of reasoning* over modelling acrobatics. The single highest-leverage action is to make every decision visible and defensible: every cleaning drop logged, every cold-start fallback annotated with a `source` field, every architectural alternative listed with its rejection rationale. The README is the most important deliverable not because the brief says so, but because in production every shortcut becomes a future incident — and the only thing that helps the next engineer is the audit trail you left behind.
