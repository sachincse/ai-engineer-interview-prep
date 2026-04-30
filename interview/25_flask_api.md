# Chapter 25 — Flask API

> **Why this chapter exists:** Flask appears in the "Frameworks & APIs" line of your resume. It's also the framework most ML engineers used before FastAPI took over around 2020-2022, so an interviewer may ask "you mention Flask — what would you choose Flask over FastAPI for in 2026?" That question separates candidates who can rattle off framework names from those who actually understand the tradeoffs. This chapter gives you that fluency in about 30 minutes of reading.

---

## 25.1 What Flask is — and the historical context

Flask is a "micro" Python web framework written by Armin Ronacher, first released in 2010. The "micro" doesn't mean limited; it means **unopinionated**. Flask gives you a routing layer, a request/response cycle, and Jinja2 templating, and asks you to bring your own everything else — database layer, validation, authentication, async support, OpenAPI docs. For a decade it was the default Python web framework, the safe choice that every team knew.

The historical importance for ML engineers: from roughly 2014 to 2020, **every production ML model served from Python was served via Flask**. Spark Hosting, AWS SageMaker's old "BYO container" path, Heroku ML demos, Databricks model serving, all leaned on Flask. So most ML engineers with 5+ years of experience have shipped at least one Flask service.

The reason Flask is no longer the default: FastAPI arrived in 2018 and gave you Pydantic validation, native async, automatic OpenAPI docs, and Starlette's performance, all without giving up Flask's simplicity. By 2022 most new Python ML APIs were FastAPI, with Flask remaining for legacy services and very simple use cases.

### Where Flask still wins in 2026

- **Legacy services** that are stable and don't need migration churn
- **Small internal tools** where the simplicity of Flask outweighs the value of FastAPI's auto-docs
- **Sync-only workloads** where the entire downstream stack is synchronous and async would add no benefit
- **Very strict dependency floors** where bringing in Pydantic 2.x and Starlette is a problem

---

## 25.2 The mental model — the request/response cycle

```
   ┌─────────────┐    HTTP    ┌──────────────┐    Python    ┌──────────┐
   │   Client    │──────────▶│  WSGI Server │──────────────▶│  Flask   │
   │  (browser,  │           │  (Gunicorn   │               │   App    │
   │   curl,     │           │   / uWSGI)   │               │          │
   │   service)  │           └──────────────┘               └────┬─────┘
   └─────────────┘                                                │
                                                                  ▼
                                                       ┌─────────────────┐
                                                       │  Route handler  │
                                                       │  (your function)│
                                                       └────────┬────────┘
                                                                │
                                              ┌─────────────────┼─────────────────┐
                                              ▼                 ▼                 ▼
                                       ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
                                       │  request     │  │  Jinja      │  │  Response    │
                                       │  context     │  │  template   │  │  object      │
                                       │  (per req)   │  │  (optional) │  │  (JSON/HTML) │
                                       └──────────────┘  └─────────────┘  └──────────────┘
```

Three things to internalize:

1. **WSGI is the protocol.** Flask is a WSGI app, which means it implements the Python "Web Server Gateway Interface" — a sync-callable interface where each request is a function call from the server to your app. This is fundamentally different from ASGI (which FastAPI uses), where requests are async. WSGI is per-process synchronous; ASGI is event-loop async.

2. **The request context.** Inside any view function, you can access `request` (current HTTP request), `g` (per-request global), and `session` (signed-cookie session). These are thread-local, so each request has its own copy.

3. **Routing is decorator-based.** `@app.route("/predict", methods=["POST"])` registers a function as the handler for that URL. There's no separate routing config file.

---

## 25.3 A complete production-shaped Flask ML service

This is the kind of code an interviewer would expect you to recognize and discuss. The example serves an XGBoost model — close to what you ran at TrueBalance, but with Flask.

```python
import os
import logging
from flask import Flask, request, jsonify, g
from werkzeug.exceptions import HTTPException
import xgboost as xgb
import numpy as np
import time
import uuid

app = Flask(__name__)

# Load model once at import time — NOT inside the request handler
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/model/xgb.bin")
model = xgb.Booster()
model.load_model(MODEL_PATH)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.4.2")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)


@app.before_request
def attach_request_id():
    """Inject a request ID for log correlation."""
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.start_time = time.time()


@app.after_request
def log_response(response):
    duration_ms = int((time.time() - g.start_time) * 1000)
    logger.info(
        f"request_id={g.request_id} method={request.method} path={request.path} "
        f"status={response.status_code} duration_ms={duration_ms}"
    )
    response.headers["X-Request-ID"] = g.request_id
    return response


@app.errorhandler(Exception)
def handle_unexpected(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    logger.exception(f"request_id={g.request_id} unhandled: {e}")
    return jsonify({"error": "Internal Server Error", "request_id": g.request_id}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ready", methods=["GET"])
def ready():
    if model is None:
        return jsonify({"status": "loading"}), 503
    return jsonify({"status": "ready"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON"}), 400

    # Manual validation — Flask has no Pydantic equivalent built in
    required = ["customer_id", "income", "credit_score", "loan_amount"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = np.array([[
            float(payload["income"]),
            int(payload["credit_score"]),
            float(payload["loan_amount"]),
        ]])
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Bad field types: {e}"}), 400

    prob = float(model.predict(xgb.DMatrix(features))[0])

    return jsonify({
        "customer_id": payload["customer_id"],
        "withdraw_probability": prob,
        "model_version": MODEL_VERSION,
        "request_id": g.request_id,
    })


if __name__ == "__main__":
    # Dev only — production runs via Gunicorn
    app.run(host="0.0.0.0", port=8080, debug=False)
```

What to point at when discussing this in an interview:

- **Model loaded at import time, not in the handler.** This is the single biggest mistake junior engineers make. Loading per-request means cold-load latency on every call.
- **Manual validation.** Flask has no Pydantic equivalent, so you write the field-checking yourself. Annoying but explicit. (FastAPI would replace 15 lines of validation with a 5-line Pydantic model.)
- **Structured logging with request_id.** Every log line is correlatable. The `before_request` / `after_request` hooks are Flask's way to inject middleware-like behavior.
- **Distinct `/health` and `/ready`.** Same Kubernetes pattern as everywhere else — health = process alive, ready = model loaded.
- **No `app.run()` in production.** That's Werkzeug's dev server, single-threaded, slow. Production uses Gunicorn or uWSGI as a WSGI server.

---

## 25.4 Production deployment — Gunicorn

In production, Flask runs behind a WSGI server. The standard is Gunicorn:

```bash
gunicorn \
  --workers 4 \                    # 2*CPU + 1 typical
  --threads 2 \                    # threads per worker (only if sync I/O blocks)
  --worker-class sync \            # sync (default), gthread, gevent, eventlet
  --timeout 30 \                   # worker SIGKILL after N seconds
  --keep-alive 5 \                 # keepalive seconds for HTTP connections
  --max-requests 1000 \            # restart workers after N requests (memory leak safety)
  --max-requests-jitter 100 \      # randomize restart timing
  --bind 0.0.0.0:8080 \
  --access-logfile - \
  --error-logfile - \
  app:app                          # module:flask_app_object
```

The flags worth knowing:

- **`--workers`**: number of worker processes. Each worker is a separate Python interpreter. For CPU-bound ML, 1 worker per GPU; for I/O-bound, `2*CPU + 1`.
- **`--worker-class`**: `sync` (default, one request per worker at a time), `gthread` (thread pool inside each worker), `gevent` or `eventlet` (greenlet-based async, monkey-patches stdlib). For ML inference, `sync` or `gthread` is typical.
- **`--max-requests` with jitter**: restart workers periodically to free memory leaks. Good hygiene; shouldn't be necessary, but is.
- **`--timeout`**: kill worker if request takes longer. For ML inference, set this generously (e.g., 60-120s) since first-call cold paths can be slow.

### Gunicorn vs uWSGI vs nginx

- **Gunicorn**: simplest, pure-Python, the default choice in 2026.
- **uWSGI**: older, faster in some benchmarks, much more configurable. Used in heavily-tuned Python deployments. Configuration is notoriously complex.
- **nginx**: reverse proxy, often sits in front of Gunicorn. Handles TLS termination, static files, load balancing, rate limiting.

The standard production stack: nginx (reverse proxy) → Gunicorn (WSGI server) → Flask (app).

---

## 25.5 Flask vs FastAPI — the comparison interviewers ask

| Concern | Flask | FastAPI |
|---------|-------|---------|
| Protocol | WSGI (sync) | ASGI (async-first) |
| Validation | Manual or `marshmallow` add-on | Pydantic, automatic |
| OpenAPI docs | Manual or `flask-restx` add-on | Automatic from type hints |
| Performance | Lower (sync, no event loop) | Higher (Starlette, async) |
| Async support | Limited (Flask 2+ added partial async) | First-class native |
| Learning curve | Easiest | Easy (slightly steeper because of types) |
| Ecosystem | Vast (15+ years of plugins) | Growing fast |
| Best for | Legacy, simple sync services | New ML APIs, async-heavy services |

The interview-winning sentence:

> "Flask was the right call for a long time and is still the right call for legacy services and very simple sync-only use cases. For new ML APIs in 2026 I default to FastAPI because Pydantic gives me input validation for free, async lets me chain LLM and DB calls efficiently, and OpenAPI docs are auto-generated. The transition cost from Flask to FastAPI is low — both have similar mental models — but the productivity gain on a new project is real."

### When you'd actually pick Flask over FastAPI in 2026

1. **Stable legacy service** that's not worth migrating.
2. **Heavily customized middleware stack** that depends on WSGI hooks (Flask's `before_request`, `after_request`, plus the Flask-* ecosystem like Flask-Login, Flask-SQLAlchemy).
3. **Strict dependency floor** — if you can't bring in Pydantic 2.x, FastAPI is off the table.
4. **Very small team that already knows Flask cold** and doesn't want to learn a new framework.
5. **Internal admin tools with Jinja templates** — Flask's HTML rendering story is slightly more polished than FastAPI's.

---

## 25.6 Flask 2.x async — the partial story

Flask 2.0 (2021) added async support. You can write `async def` view functions, and Flask will run them in a thread pool via `asgiref`. So you *can* call async libraries from Flask:

```python
@app.route("/external")
async def call_external():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.example.com/data")
    return r.json()
```

But — and this is the catch — Flask remains a WSGI app. The async function runs in a sync thread that's ad-hoc spawning an event loop per request. You don't get the event-loop concurrency benefits that ASGI gives FastAPI. So Flask async is a convenience for using async libraries, not a way to scale concurrency.

If you genuinely need async-native concurrency, you should:
1. Keep Flask if the rest of the stack is sync.
2. Migrate to Quart (Flask's API but ASGI-native) for a low-friction async migration.
3. Migrate to FastAPI for a fresh start.

---

## 25.7 Common Flask middleware and patterns

### Blueprints — modular routing

Blueprints split routes across multiple Python modules, useful for medium and large apps:

```python
# users.py
from flask import Blueprint
users_bp = Blueprint("users", __name__, url_prefix="/users")

@users_bp.route("/<int:user_id>")
def get_user(user_id):
    return jsonify({"id": user_id})

# app.py
from users import users_bp
app.register_blueprint(users_bp)
```

### Flask-Login for authentication

```python
from flask_login import LoginManager, login_required, current_user

login_manager = LoginManager(app)

@app.route("/admin")
@login_required
def admin():
    return f"Hi {current_user.name}"
```

### Flask-SQLAlchemy for ORM

```python
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, unique=True)

@app.route("/users")
def list_users():
    return jsonify([{"id": u.id, "email": u.email} for u in User.query.all()])
```

### Marshmallow for validation

The closest Flask gets to Pydantic:

```python
from marshmallow import Schema, fields, ValidationError

class PredictRequestSchema(Schema):
    customer_id = fields.Str(required=True)
    income = fields.Float(required=True, validate=lambda x: x >= 0)
    credit_score = fields.Int(required=True, validate=lambda x: 300 <= x <= 850)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = PredictRequestSchema().load(request.get_json())
    except ValidationError as err:
        return jsonify({"error": err.messages}), 400
    # ... use data
```

This is the legacy ML pattern. It works but is more verbose than Pydantic v2.

---

## 25.8 Testing Flask apps

Flask has a first-class test client built in:

```python
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json == {"status": "ok"}

def test_predict(client):
    r = client.post("/predict", json={
        "customer_id": "abc",
        "income": 50000.0,
        "credit_score": 700,
        "loan_amount": 10000.0,
    })
    assert r.status_code == 200
    assert "withdraw_probability" in r.json

def test_predict_missing_field(client):
    r = client.post("/predict", json={"customer_id": "abc"})
    assert r.status_code == 400
    assert "Missing fields" in r.json["error"]
```

The test client doesn't actually start an HTTP server — it calls the WSGI app directly. Tests run fast and deterministically.

---

## 25.9 Common Flask gotchas and anti-patterns

1. **Loading the model inside the handler.** Same trap as FastAPI. Load at module level.
2. **Using `app.run()` in production.** That's the dev server. Use Gunicorn or uWSGI.
3. **`debug=True` in production.** Exposes the Werkzeug debugger console — a remote code execution vulnerability if reachable from the internet.
4. **Forgetting `silent=True` in `request.get_json`.** Without it, malformed JSON raises an exception that crashes the request with a 500 instead of returning a clean 400.
5. **Storing model inference state in `g`.** `g` is per-request, but more importantly, sharing model state across workers won't work — each Gunicorn worker has its own Python process.
6. **Not using a connection pool for DB calls.** Each request opening a fresh DB connection is a resource leak. Use Flask-SQLAlchemy's pooling or a libraries like `psycopg-pool`.
7. **Using Flask sessions for sensitive state.** Flask sessions are signed but not encrypted by default. Don't store secrets in `session`.
8. **Mixing async libraries with sync Flask.** Without `async def` view functions and Quart/asgiref, blocking calls inside Flask handlers stall workers.
9. **Returning raw Python objects from views.** Flask serializes only basic types. Use `jsonify` or convert to a dict.
10. **Ignoring `--max-requests` for memory leaks.** Long-running Flask workers accumulate memory; restart them periodically.

---

## 25.10 Resume tie-in box

> Sachin's resume lists Flask alongside FastAPI under "Frameworks & APIs." The natural narrative when asked about Flask:
>
> *"I've shipped Flask services in production — the ResMed early ML platform had several Flask endpoints before we migrated to FastAPI as the team standardized. The migration was straightforward because the mental models are similar, and the productivity gain on FastAPI was clear: Pydantic validation alone removed about 30% of our request-handler code. For new services I default to FastAPI now. I'd reach for Flask in 2026 only for stable legacy services or very simple sync-only internal tools."*

---

## 25.11 Interview Q&A — full narrative answers

### Q1. What's the main difference between Flask and FastAPI?

The protocol they implement. Flask is a WSGI app — synchronous, request-per-worker — while FastAPI is built on Starlette and implements ASGI, which is async-native and event-loop-driven. That single difference cascades into everything else. Flask handles concurrency by spawning multiple sync workers (Gunicorn does this); FastAPI handles concurrency by yielding the event loop during I/O waits, so one worker can serve many concurrent requests as long as they're I/O-bound. The other major difference is built-in validation: FastAPI uses Pydantic to validate every request automatically and to generate OpenAPI docs from your type hints; Flask asks you to bring your own validation library. For new ML services in 2026 I default to FastAPI; I'd pick Flask only for legacy maintenance or very simple sync-only services.

### Q2. How do you serve a model with Flask?

Load the model at module scope, expose a `/predict` route that accepts JSON, validate the input fields, run inference, and return JSON. Crucially, the model must be loaded once at import time — not inside the request handler — otherwise every request pays the cold-load cost. In production you run Flask behind Gunicorn with multiple worker processes; each worker has its own copy of the model in memory. Add `/health` and `/ready` endpoints so Kubernetes can correctly distinguish process liveness from readiness to serve traffic. Wire structured logging with a request ID for correlation, and you have a production-shaped service.

### Q3. Why doesn't Flask scale well for high concurrency?

Because it's WSGI — each worker handles one request at a time. To serve 100 concurrent requests you need either 100 workers or a worker class like `gthread` that uses thread pools per worker. With sync workers and CPU-bound ML inference, you're capped at workers × throughput-per-worker. FastAPI, being async-native, can interleave many concurrent requests on a single worker as long as they're I/O-bound — for example, an endpoint that mostly waits on an LLM call can serve hundreds of concurrent requests on one worker. For pure CPU-bound work, both frameworks have the same bottleneck (Python's GIL), but Flask amplifies it.

### Q4. What's a Blueprint in Flask?

A Blueprint is Flask's mechanism for splitting an app's routes across multiple Python modules. You define a Blueprint with its own URL prefix, register routes on it, and then attach it to the main app at startup. Useful for organizing larger apps into logical sections — a `users_bp`, an `admin_bp`, an `api_v1_bp`. Each Blueprint can have its own URL prefix, error handlers, and middleware. It's organizational rather than functional — the runtime behavior is identical to having all routes on one app.

### Q5. How does Flask handle request context?

Through thread-local proxies. Inside any view function, you access `request`, `g`, and `session` as if they were global variables, but Flask resolves them per-request using thread-local storage. So one request's `request.json` is never confused with another's. The `g` object specifically is per-request scratch space — you can attach things to it in a `before_request` handler and read them in the view. The downside is that thread-locals are quirky in async or multi-process contexts, which is part of why FastAPI moved away from this pattern.

### Q6. When would you actually choose Flask over FastAPI in 2026?

Three real scenarios. First, maintaining a stable legacy service that works fine — migration churn isn't worth the marginal upside. Second, very simple internal tools where the framework overhead doesn't matter and the team already knows Flask cold. Third, sync-only workloads where the entire downstream stack is sync — there's no async benefit to capture, and the simplicity of Flask wins. For anything new and async-heavy, especially ML services that chain LLM calls, FastAPI is the right choice.

### Q7. How do you deploy Flask in production?

Behind a WSGI server, never `app.run()`. The standard stack is nginx as reverse proxy in front of Gunicorn workers running the Flask app. Gunicorn handles process management, worker pool sizing, and HTTP serving; nginx handles TLS termination, static file caching, rate limiting, and load balancing. Configure Gunicorn with `--workers 2*CPU + 1` for I/O-bound workloads, set `--timeout` generously for ML inference, and enable `--max-requests` with jitter to recycle workers periodically as a memory-leak safety net. In Kubernetes, you'd run this as a Deployment with proper liveness and readiness probes pointing at `/health` and `/ready`.

### Q8. What's the model-loading anti-pattern, and why does it bite people?

Loading the model inside the request handler instead of at module scope. The naive pattern looks innocent: `def predict(): model = load_model(); ...`. But every request pays the load cost — for an XGBoost model maybe 50ms, for a transformer maybe seconds. Latency goes up, throughput collapses, and the GPU/CPU is dominated by repeated loads instead of inference. The fix is one line: load the model at module level, before any handlers are defined. The reason it bites people is that it works fine on the dev server with one worker doing one request — the slowdown only shows up under concurrency. So engineers ship and the problem appears in load testing or production.

### Q9. How would you migrate a Flask service to FastAPI?

The mental models are similar enough that the migration is mostly mechanical. Replace `@app.route` decorators with FastAPI's `@app.get` / `@app.post` etc. Replace `request.get_json()` plus manual validation with a Pydantic model as the function parameter. Replace `jsonify` returns with the Pydantic response model — automatic. Replace `before_request` / `after_request` hooks with FastAPI middleware or dependencies. Replace Gunicorn-with-sync-worker-class with Gunicorn-with-Uvicorn-workers, or just Uvicorn directly. The handful of things that don't translate cleanly: heavy use of Flask-specific extensions (Flask-Login, Flask-SQLAlchemy) — those have FastAPI equivalents but with different APIs, so you'd port custom code. Total effort for a typical small-to-medium ML service: a day or two.

### Q10. What does `app.errorhandler` do?

It registers a function as the handler for a specific exception class or HTTP status code. So `@app.errorhandler(404)` lets you customize the 404 response, and `@app.errorhandler(Exception)` lets you catch every uncaught exception and render a clean JSON error response. The pattern matters for ML APIs because by default Flask renders an HTML traceback for 500 errors — useful in dev, terrible in production. A global `@app.errorhandler(Exception)` that logs the exception with the request ID and returns a sanitized JSON response is the production-hygiene starting point.

---

## 25.12 The morning-of cheatsheet for Flask

```
   Protocol:           WSGI (sync), one request per worker
   Production server:  Gunicorn behind nginx
   Workers:            2*CPU + 1 for I/O-bound, 1/GPU for ML inference
   Model loading:      At module scope, NEVER in handler
   Validation:         Manual, or Marshmallow add-on
   Health/ready:       Always distinct endpoints
   Request context:    request, g, session — thread-local
   Routing:            @app.route("/path", methods=[...])
   Async support:      Flask 2.0+ partial; not async-native
   Testing:            app.test_client() — fast, deterministic
   Migration to FastAPI: usually 1-2 days for small/medium services
```

---

End of Chapter 25. Continue to **[Chapter 26 — Resume Skills Crib Sheet](26_resume_skills_crib.md)** or back to **[Chapter 00 — Master Index](00_index.md)**.
