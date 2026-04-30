# Chapter 38 — Lambda Cold-Start Mitigation for ML Workloads

> **Why this chapter exists:** "Real-time XGBoost Lambda with p99 < 500ms across 3 environments" is on Sachin's resume, and any senior interviewer who reads that bullet will pull on it: "How did you handle cold starts? What's your strategy if you wanted to deploy a transformer-class model the same way?" This is also a recurring topic at any company shipping ML on serverless — Avrioc, Upvest, and most fintechs use Lambda for at least some ML serving. Master this chapter and you can speak fluently to any "production serverless ML" question.

---

## 38.1 The cold-start anatomy — what you're actually optimizing

Before any mitigation, understand the three phases. Each has a different cost and a different lever.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                  LAMBDA CONTAINER COLD START                         │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                      │
   │   PHASE 1                  PHASE 2              PHASE 3              │
   │   CONTAINER PULL           INIT                 INVOKE               │
   │   ──────────────           ────                 ──────               │
   │                                                                      │
   │   AWS pulls image          /init runs:         Your handler         │
   │   from ECR, decompresses   - Python imports    runs the request.    │
   │   layers, mounts them.     - Module-level      Inference happens    │
   │   Lambda has aggressive      code              here.                │
   │   caching but on a fresh   - Model load                             │
   │   runtime you pay 200ms    - Connection pool                        │
   │   to 3 seconds here.       setup                                    │
   │                                                                      │
   │   Optimized by:            Optimized by:       Optimized by:        │
   │   - Smaller image          - Lazy imports      - Profile and remove │
   │   - Multi-stage builds     - Faster model      - Avoid synchronous  │
   │   - Layer cache reuse        format (ONNX,        I/O on hot path   │
   │   - Tighter pruning          INT8)             - Use cached state   │
   │                            - Larger memory       (Redis, in-memory) │
   │                            - EFS-shared model                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The mistake junior engineers make: they obsess over Phase 3 micro-optimizations while Phase 1 + Phase 2 dominate the cold-start latency. Always measure where your time actually goes — `INIT_DURATION` from CloudWatch is your starting point, then `Duration` minus `INIT_DURATION` is the invoke time.

### A real cold-start budget breakdown

For Sachin's TrueBalance XGBoost Lambda (real shapes):

```
   Phase 1 (container pull):       ~800ms (first invocation in a fresh env)
                                    ~50-100ms (cached layers, subsequent containers)
   Phase 2 (init):                  ~300ms
     - Python imports (xgboost,    ~200ms
       numpy, pandas, redis,
       snowflake-connector,
       pydantic, structlog)
     - Model load from /opt:        ~30ms
     - Redis pool setup:            ~20ms
     - boto3 clients:               ~50ms
   Phase 3 (invoke, warm path):     ~50-100ms
     - Feature fetch (Redis cache): ~10ms
     - XGBoost predict:             ~5ms
     - Response build:              ~5ms

   Total cold:                       ~1150ms
   Total warm:                        ~50-100ms
   p99 cold rate (without PC):      ~1-3% of invocations
```

Without mitigation, ~1% of invocations hit ~1.2s — already breaking the 500ms p99 SLA. With mitigation, every request is ~50-100ms.

---

## 38.2 Where each model class hurts

Different model classes have different cold-start profiles. Treat them differently.

### Class 1 — Tabular models (XGBoost, LightGBM, sklearn, RF)

```
   Model size:           5-50 MB
   Init dominator:       Heavy Python deps (xgboost + numpy + pandas)
   Cold start typical:   1-3 seconds
   Right for Lambda:     YES — fits naturally
```

**Strategy:** focus on container image size and module-level loading. PC for SLA.

### Class 2 — Encoder transformers (BERT, DistilBERT, RoBERTa for NER/classification)

```
   Model size:           70-500 MB depending on quantization
   Init dominator:       Model load via from_pretrained (huge)
   Cold start typical:   3-8 seconds
   Right for Lambda:     MAYBE — needs aggressive optimization
```

**Strategy:** quantize to INT8, switch to ONNX Runtime, mount from EFS. PC essential. Or move to ECS Fargate.

### Class 3 — Decoder LLMs (Llama, Mistral, GPT-class)

```
   Model size:           1-140 GB
   Init dominator:       Impossible — too big for Lambda
   Cold start typical:   N/A (won't fit anyway)
   Right for Lambda:     NO — use vLLM on EKS or Bedrock
```

**Strategy:** don't deploy LLMs on Lambda. Use Bedrock for inference, EKS with vLLM for self-hosted. Lambda becomes the orchestration layer that *calls* the LLM, not the layer that *runs* it.

### Class 4 — Computer vision (YOLO, ResNet, simple CNNs)

```
   Model size:           20-200 MB
   Init dominator:       Model load + image preprocessing libs (Pillow, cv2)
   Cold start typical:   2-5 seconds
   Right for Lambda:     YES with optimization
```

**Strategy:** ONNX Runtime, INT8 where possible, move to GPU only if latency demands it (Lambda has no GPU; switch to SageMaker Async or ECS with GPU).

This chapter focuses on Classes 1 and 2 (Sachin's two real production cases). The principles generalize.

---

## 38.3 Mitigation 1 — Provisioned Concurrency

This is your single biggest lever. Do this first.

### What PC does

Provisioned Concurrency tells Lambda to keep N instances permanently initialized — already past the cold-start phase, ready to invoke. Requests that arrive when PC instances are available skip Phase 1 and Phase 2 entirely.

```
   Without PC:
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Request 1│────▶│ COLD     │────▶│ Response │  1200ms
   └──────────┘     │ Lambda   │     │          │
                    └──────────┘     └──────────┘
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Request 2│────▶│ WARM     │────▶│ Response │  50ms
   └──────────┘     │ (reused) │     │          │
                    └──────────┘     └──────────┘

   With PC=5:
   ┌──────────┐     ┌────────────┐     ┌──────────┐
   │ Request 1│────▶│ ALWAYS WARM│────▶│ Response │  50ms
   └──────────┘     │ (PC pool)  │     │          │
                    └────────────┘     └──────────┘
   ┌──────────┐     ┌────────────┐     ┌──────────┐
   │ Request 2│────▶│ ALWAYS WARM│────▶│ Response │  50ms
   └──────────┘     └────────────┘     └──────────┘
```

### Setting PC up

```bash
# 1. Publish a version of your function
aws lambda publish-version --function-name withdrawal-predictor

# 2. Point an alias at that version
aws lambda create-alias \
    --function-name withdrawal-predictor \
    --name prod \
    --function-version 47

# 3. Configure PC on the alias
aws lambda put-provisioned-concurrency-config \
    --function-name withdrawal-predictor \
    --qualifier prod \
    --provisioned-concurrent-executions 5
```

PC only applies to **published versions accessed via alias** — not `$LATEST`. This is a frequent gotcha (see §38.10).

### Sizing PC

PC cost is real. Right-sizing matters.

```
   Rule of thumb:
   PC = peak concurrent requests during business hours

   Where to find this:
   - CloudWatch Metrics: ConcurrentExecutions metric, peak value
   - Tighter: percentile chart of ConcurrentExecutions over time
   - Conservative: PC = p99 of ConcurrentExecutions

   For TrueBalance withdrawal predictor:
   - Average concurrent: 2
   - p99 concurrent during business hours: 5
   - Set PC = 5

   When PC capacity exceeded:
   - Requests overflow to on-demand Lambdas (cold start hits ~1%)
   - Or you can configure to throttle (return 429)
```

### Auto-scaling PC by schedule

Steady-state PC is wasteful for businesses with diurnal patterns. Use Application Auto Scaling:

```python
# CloudFormation / Terraform / boto3
import boto3
appas = boto3.client("application-autoscaling")

# Register the scalable target
appas.register_scalable_target(
    ServiceNamespace="lambda",
    ResourceId="function:withdrawal-predictor:prod",
    ScalableDimension="lambda:function:ProvisionedConcurrency",
    MinCapacity=2,    # never go below 2
    MaxCapacity=10,   # cap at 10 to control cost
)

# Scheduled scaling: business hours up, off-hours down
appas.put_scheduled_action(
    ServiceNamespace="lambda",
    ScalableDimension="lambda:function:ProvisionedConcurrency",
    ResourceId="function:withdrawal-predictor:prod",
    ScheduledActionName="ScaleUpBusinessHours",
    Schedule="cron(0 9 ? * MON-FRI *)",
    ScalableTargetAction={"MinCapacity": 5, "MaxCapacity": 10},
)
appas.put_scheduled_action(
    ServiceNamespace="lambda",
    ScalableDimension="lambda:function:ProvisionedConcurrency",
    ResourceId="function:withdrawal-predictor:prod",
    ScheduledActionName="ScaleDownAfterHours",
    Schedule="cron(0 22 ? * MON-FRI *)",
    ScalableTargetAction={"MinCapacity": 1, "MaxCapacity": 3},
)
```

Typical savings: 40-60% on PC cost vs always-on max.

### PC cost math

```
   PC pricing (us-east-1, 2026):
     - Memory size × duration × $0.000004646 per GB-second
     - Plus normal invocation cost

   Example for TrueBalance Lambda (3GB memory, PC=5, always on):
     5 instances × 3 GB × 3600 s/hr × 24 hr × 30 days
     = 38,880,000 GB-seconds/month
     × $0.000004646
     = $180/month per environment

   Three environments (dev/staging/prod): $540/month total

   With auto-scaling (avg 3 instances):
     $108/month per env, $324/month total
```

For a fintech serving real customer traffic, this is cheap insurance against SLA breach. Always do the math vs the cost of one missed SLA.

---

## 38.4 Mitigation 2 — Container image optimization

The image you ship determines Phase 1 latency. Smaller image = faster pull = faster cold start.

### Multi-stage Dockerfile pattern

```dockerfile
# Stage 1: builder — install all heavy deps with build tooling
FROM public.ecr.aws/lambda/python:3.11 AS builder

# Build deps
RUN yum install -y gcc gcc-c++ python3-devel

# Install with --target so we can copy a clean tree
RUN pip install --target /opt/python --no-cache-dir \
    xgboost==2.0.3 \
    numpy==1.26.0 \
    pandas==2.1.0 \
    redis==5.0.1 \
    snowflake-connector-python==3.5.0 \
    pydantic==2.5.0 \
    structlog==23.2.0

# Stage 2: runtime — minimal, no build tools
FROM public.ecr.aws/lambda/python:3.11

# Copy only what we need
COPY --from=builder /opt/python /opt/python
ENV PYTHONPATH=/opt/python

# Application code
COPY app/ ${LAMBDA_TASK_ROOT}/

# Model artifact
COPY models/xgb.json /opt/models/xgb.json

CMD ["app.handler.lambda_handler"]
```

Why it matters: builder stage is discarded, only the final stage ships to ECR. Without multi-stage, you ship gcc and Python headers (~200 MB extra) you'll never use at runtime.

### What to remove from the runtime image

```
   ✗ Build tools (gcc, make, headers)
   ✗ Pip cache directories
   ✗ Test files in dependencies
   ✗ Documentation files in dependencies
   ✗ __pycache__ directories
   ✗ .pyc files older than the source
   ✗ Optional dependencies you don't use (e.g. xgboost has [scikit-learn] extras)
   ✗ Anything in /tmp at build time
```

### Layer ordering for cache reuse

Lambda caches container layers across deployments. Order them from least-to-most frequently changing:

```dockerfile
# Stable layers first (rarely change → cache hits often)
FROM public.ecr.aws/lambda/python:3.11
RUN pip install --no-cache-dir xgboost numpy pandas    # dependencies change rarely
COPY models/xgb.json /opt/models/xgb.json              # model changes occasionally

# Volatile layer last (changes every deploy)
COPY app/ ${LAMBDA_TASK_ROOT}/                         # code changes every deploy
```

When you redeploy, only the last layer is re-pulled because the dependency layers are cached. Cold-start improvement after first cache hit: ~500ms saved.

### Image size targets

```
   Class 1 (XGBoost-style):  aim for <500 MB
   Class 2 (DistilBERT):     aim for <800 MB after quantization
   Class 4 (CNN):            aim for <1 GB
```

Above 1 GB, container pull dominates and cold-start is painful. Below 500 MB, you're optimizing inside noise.

---

## 38.5 Mitigation 3 — Module-level (init phase) loading

Anything you put at module scope runs **once per container**, not once per request. This is free amortization across thousands of warm invocations.

### The right pattern

```python
# handler.py — module level (Phase 2 INIT)
import os
import xgboost as xgb
import redis
import boto3
import structlog

# 1. Logger (cheap, always at module level)
logger = structlog.get_logger()

# 2. Model (loaded ONCE per container, reused thousands of times)
_model = xgb.Booster()
_model.load_model("/opt/models/xgb.json")
_MODEL_VERSION = "1.4.2"

# 3. Connection pools (reused across invocations on same container)
_redis_pool = redis.ConnectionPool(
    host=os.environ["REDIS_HOST"],
    port=6379,
    max_connections=10,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=1,
)

# 4. boto3 clients (each client init is ~50ms)
_secrets = boto3.client("secretsmanager")
_dynamodb = boto3.resource("dynamodb")

# 5. Optional: pre-fetch immutable config from Secrets Manager
_SECRET_CACHE = {}

def _get_secret(name: str) -> str:
    if name not in _SECRET_CACHE:
        _SECRET_CACHE[name] = _secrets.get_secret_value(SecretId=name)["SecretString"]
    return _SECRET_CACHE[name]

# === Per-invocation handler (Phase 3 INVOKE) ===
def lambda_handler(event, context):
    customer_id = event["customer_id"]
    request_id = context.aws_request_id

    redis_client = redis.Redis(connection_pool=_redis_pool)
    features_json = redis_client.get(f"features:{customer_id}")

    if features_json is None:
        logger.warning("feature_cache_miss", customer_id=customer_id)
        features = _fallback_features()  # circuit breaker
    else:
        features = json.loads(features_json)

    score = float(_model.predict(xgb.DMatrix([features]))[0])

    return {
        "request_id": request_id,
        "customer_id": customer_id,
        "withdraw_probability": score,
        "model_version": _MODEL_VERSION,
    }
```

### The wrong pattern (don't do this)

```python
# Anti-pattern — model load INSIDE handler
def lambda_handler(event, context):
    model = xgb.Booster()              # ← runs every invocation, 30ms wasted
    model.load_model("/opt/xgb.json")  # ← every warm call pays this
    redis_client = redis.Redis(...)    # ← TCP connect every time, 5ms wasted
    secret = boto3.client("secretsmanager").get_secret_value(...)  # ← 100ms wasted
    # ... actual work
```

Every warm invocation pays the full init cost. p99 explodes. This is the #1 mistake junior engineers make.

### When to lazy-load instead

For rarely-used optional features, lazy-load to keep cold-start fast:

```python
_optional_model = None  # not loaded at init

def _get_optional_model():
    global _optional_model
    if _optional_model is None:
        _optional_model = expensive_load()  # one-time, only if needed
    return _optional_model

def lambda_handler(event, context):
    if event.get("use_optional", False):
        model = _get_optional_model()
        # ... use it
    # otherwise skip
```

The pattern: eager-load anything used by >5% of requests, lazy-load anything used by <5%.

---

## 38.6 Mitigation 4 — Memory sizing (buy CPU with memory)

Lambda CPU allocation scales linearly with memory. More memory = more vCPU = faster cold start *and* faster invocations.

```
   Memory       vCPU equiv    XGBoost cold start    NER cold start
   ─────────    ──────────    ──────────────────    ──────────────
   512 MB       0.3           ~3 seconds            ~7 seconds
   1.0 GB       0.6           ~2 seconds            ~5 seconds
   1.8 GB       1.0           ~1.5 seconds          ~4 seconds
   3.0 GB       2.0           ~1.0 seconds          ~2.5 seconds
   5.0 GB       3.4           ~0.8 seconds          ~1.8 seconds
   10 GB        6.0           ~0.7 seconds          ~1.5 seconds
```

### The counterintuitive cost optimization

More memory often *lowers* your bill, because the function finishes faster and you're billed per GB-second.

```
   Example math for XGBoost predictor:
   1 GB × 1.5 s = 1.5 GB-seconds at $0.0000166667/GB-s = $0.000025 per invocation
   3 GB × 0.8 s = 2.4 GB-seconds at $0.0000166667/GB-s = $0.000040 per invocation

   Wait, 3GB is more expensive per invocation. But:
   - 3 GB latency hits SLA → no missed SLA penalties
   - 3 GB has fewer cold starts perceived (faster init)
   - If you have a strict latency SLA, the 3 GB is the only option
```

The tool to use: `aws-lambda-power-tuning`. It runs your function at multiple memory sizes and outputs cost vs latency vs power-tuning recommendation.

```bash
# AWS Lambda Power Tuning state machine
aws stepfunctions start-execution \
  --state-machine-arn arn:aws:states:us-east-1:...:stateMachine:powerTuningStateMachine \
  --input '{
    "lambdaARN": "arn:aws:lambda:us-east-1:...:function:withdrawal-predictor",
    "powerValues": [512, 1024, 1536, 2048, 3008, 4096, 6144, 10240],
    "num": 50,
    "payload": {"customer_id": "test_customer"},
    "strategy": "balanced"
  }'
```

Output gives you a graph: pick the inflection point where latency stops dropping but cost starts rising fast.

---

## 38.7 Mitigation 5 — Model format optimization (the NER game-changer)

For Class 2 models (transformers), the model load itself dominates cold-start. Three big wins, in order of impact.

### Win 1 — Quantize to INT8

```python
# One-time conversion (run before deploying, NOT in Lambda)
from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# Load and export to ONNX
model = ORTModelForTokenClassification.from_pretrained(
    "your-distilbert-ner",
    export=True,
)
model.save_pretrained("./ner_onnx_fp32")

# Quantize FP32 → INT8
quantizer = ORTQuantizer.from_pretrained("./ner_onnx_fp32")
config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
quantizer.quantize(save_dir="./ner_int8", quantization_config=config)
```

Results for DistilBERT NER:

```
   Model size:       265 MB → 70 MB (4× smaller)
   Cold load time:   3 s → 600 ms (5× faster)
   Inference time:   80 ms → 35 ms (2.3× faster)
   F1 quality:       0.87 → 0.864 (~0.7 pt drop, often acceptable)
```

### Win 2 — Use ONNX Runtime instead of PyTorch

```python
# Bad: PyTorch-based loading (slow)
from transformers import AutoModelForTokenClassification, AutoTokenizer
_model = AutoModelForTokenClassification.from_pretrained("./ner_int8")  # ~3s
_tokenizer = AutoTokenizer.from_pretrained("./ner_int8")

# Good: ONNX Runtime (fast)
import onnxruntime as ort
_session = ort.InferenceSession(
    "./ner_int8/model.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions(),
)
_tokenizer = AutoTokenizer.from_pretrained("./ner_int8")  # tokenizer is small, fast
```

ONNX Runtime initializes faster than PyTorch and is typically 3-5× faster on CPU inference.

### Win 3 — Distill to a smaller student model

If quantization isn't enough, train a smaller student model:

```python
# Conceptual: train a 6-layer DistilBERT on outputs of your 12-layer teacher
from transformers import DistilBertForTokenClassification, Trainer

# Initialize a smaller architecture
student_config = DistilBertConfig(
    n_layers=6,            # half the teacher's layers
    dim=256,               # smaller hidden size
    n_heads=8,
    num_labels=NUM_NER_TAGS,
)
student = DistilBertForTokenClassification(student_config)

# Train with distillation loss (KL divergence on teacher's logits + standard CE)
# Standard distillation training loop...
```

A well-trained student model is ~30 MB and loads in ~150 ms. Inference 2-3× faster. F1 typically 1-2 points below the teacher.

---

## 38.8 Mitigation 6 — Where the model lives

Three storage strategies for the model artifact. Pick based on size and traffic.

### Strategy A — In container image

```dockerfile
# Embed in image
COPY models/xgb.json /opt/models/xgb.json
```

**Pros:** simplest, single artifact, atomic deploys.
**Cons:** image bloat, Phase 1 (pull) cost grows with model size.

**Use for:** Class 1 models (XGBoost, sklearn). Anything <100 MB.

### Strategy B — S3 download at init

```python
import boto3

_s3 = boto3.client("s3")

# Init phase — runs once
_local_path = "/tmp/xgb.json"
if not os.path.exists(_local_path):
    _s3.download_file("ml-models", "xgb_v1.4.2.json", _local_path)

_model = xgb.Booster()
_model.load_model(_local_path)
```

**Pros:** decouples model versioning from code deploys, smaller container image.
**Cons:** adds 100-500ms to Phase 2 per cold start, every cold container does the download.

**Use for:** rarely. The S3 fetch overhead usually outweighs the deploy decoupling benefit.

### Strategy C — EFS mount (best for big models)

```dockerfile
# Container doesn't ship the model
# Instead, configure the Lambda to mount EFS
```

```python
# Init phase — model lives at the EFS mount point
_session = ort.InferenceSession(
    "/mnt/ner/model.onnx",  # EFS-mounted path
    providers=["CPUExecutionProvider"],
)
```

```bash
# Lambda configuration
aws lambda update-function-configuration \
  --function-name ner-predictor \
  --file-system-configs Arn=arn:aws:elasticfilesystem:...:access-point/...,LocalMountPath=/mnt/ner
```

**Pros:**
- Container image stays small (just code + ONNX Runtime, ~300 MB)
- Multiple Lambda containers share OS page cache for the model (subsequent cold starts in same AZ are warm reads from page cache)
- Model versioning by changing the mount path or symlink
- Atomic model swaps without redeploying code

**Cons:**
- EFS mount adds 100-200ms to Phase 1 (one-time per cold container)
- EFS costs (cheap, but not zero)
- Lambda must be in a VPC with the EFS access point

**Use for:** Class 2 models (BERT family) and anything >200 MB.

### Comparison table

```
   Strategy          Cold start        Image size        Model swap
   ──────────────    ──────────────    ──────────────    ──────────────
   A: In container   Best small (1s)   Big with model    Redeploy code
                     Bad big (3-5s)
   B: S3 download    +0.1-0.5s         Small             Update S3
   C: EFS mount      Best after        Smallest          Update EFS
                     warm (<0.5s)
                     +0.2s first time
```

---

## 38.9 Mitigation 7 — Connection pooling and reuse

Network connections are surprisingly expensive. TCP handshake, TLS negotiation, auth — easily 100-500ms on a cold connection. Pool everything at module scope.

### Redis pooling

```python
# Bad — new connection per invocation
def lambda_handler(event, context):
    r = redis.Redis(host=REDIS_HOST, port=6379)  # ~50ms TCP+ping
    return r.get(...)

# Good — module-level pool, connections reused
_redis_pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=6379,
    max_connections=10,
    socket_keepalive=True,
)

def lambda_handler(event, context):
    r = redis.Redis(connection_pool=_redis_pool)  # ~0ms (pooled)
    return r.get(...)
```

### Snowflake pooling

```python
import snowflake.connector
from snowflake.connector.connection import SnowflakeConnection

# Module-level connection
_snowflake_conn: SnowflakeConnection | None = None

def _get_snowflake() -> SnowflakeConnection:
    global _snowflake_conn
    if _snowflake_conn is None or _snowflake_conn.is_closed():
        _snowflake_conn = snowflake.connector.connect(
            user=SF_USER,
            password=SF_PASSWORD,
            account=SF_ACCOUNT,
            warehouse=SF_WAREHOUSE,
        )
    return _snowflake_conn
```

### Connection-storm protection during PC scaling

When PC scales up, all new instances connect simultaneously. With 10 PC instances × 10 connections = 100 connections to Redis appearing at once. This can hammer Redis.

Mitigations:
- Use `socket_connect_timeout` to fail fast if Redis is overwhelmed
- Set `max_connections` per instance lower than naive
- Stagger PC scaling with auto-scaling cooldowns
- Use Redis Cluster mode that distributes load

---

## 38.10 Common gotchas — and how to avoid them

### 1. PC and `$LATEST` mismatch

PC only applies to **published versions accessed via alias**. Deploys that update `$LATEST` don't benefit from existing PC.

```
   Wrong workflow:                    Right workflow:
   ─────────────                       ─────────────
   1. Deploy code → $LATEST           1. Deploy code → $LATEST
   2. PC is on alias 'prod'            2. Publish version: $LATEST → v48
      pointing to v47                  3. Update alias 'prod' to v48
   3. New code is in $LATEST,         4. PC migrates from v47 to v48
      not v47 → not warm               5. v47 PC drains, v48 PC stays warm
```

Use `aws lambda update-alias --routing-config` to do gradual rollouts that respect PC.

### 2. Image cache invalidation on deploy

Every deploy invalidates Lambda's container image cache. The first invocation post-deploy is a cold cold start.

**Mitigation:** trigger N synthetic invocations after each deploy to warm the cache.

```bash
# Post-deploy hook
for i in {1..10}; do
  aws lambda invoke --function-name withdrawal-predictor:prod \
    --payload '{"warmup": true}' /tmp/out.json &
done
wait
```

### 3. `init` timeout (10 seconds hard limit)

If your INIT phase exceeds 10 seconds, Lambda fails the cold start. For huge models, this happens. Mitigations:
- Quantize and optimize (Mitigation 7)
- Move to EFS (Mitigation 8) so the load is from local-mounted storage, not from S3
- Move off Lambda — at this point ECS Fargate is simpler

### 4. PC + reserved concurrency conflicts

Reserved concurrency caps total concurrency *including* PC. If `reserved=5` and `PC=10`, you've broken something.

```bash
# Sane defaults
reserved_concurrency = 100   # cap total
provisioned_concurrency = 10  # warm pool, must be ≤ reserved
```

### 5. Connection storms during PC scale-up

When PC scales 0→10 instances, all connect to Redis at once. Use connection limits and staggered scaling.

### 6. Module-level imports and side effects

Some libraries do work at import time. Lazy-import where possible:

```python
# Bad — pandas takes ~200ms to import even if you don't use it for this request
import pandas as pd

def lambda_handler(event, context):
    if needs_pandas(event):
        df = pd.DataFrame(...)
    # Otherwise, you wasted 200ms

# Good — lazy import
def lambda_handler(event, context):
    if needs_pandas(event):
        import pandas as pd  # only imported when needed
        df = pd.DataFrame(...)
```

For big optional libraries (TensorFlow, PyTorch), lazy-import saves seconds.

### 7. Default `max_workers` for thread pools

Lambda containers are sized for one concurrent request. ThreadPoolExecutor defaults to (CPU count × 5) workers, which can be too many or too few. Set explicitly:

```python
_executor = ThreadPoolExecutor(max_workers=4)  # tune to your invoke pattern
```

### 8. Logging at module level

```python
# Subtle bug — logger config runs every import
logging.basicConfig(level=logging.INFO)  # ← every cold start

# Better — configure once with idempotency
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
```

### 9. Forgetting to monitor INIT_DURATION separately

Standard Duration metric mixes warm and cold. Look at INIT_DURATION as a separate dimension:

```sql
-- CloudWatch Logs Insights
fields @timestamp, @initDuration, @duration
| filter @type = "REPORT"
| stats avg(@initDuration), max(@initDuration), avg(@duration) by bin(5m)
```

### 10. Container limit: 10 GB memory, 15-min timeout

Sometimes you hit a wall. If your model needs >10 GB or your inference >15 min, Lambda is wrong. Switch to ECS Fargate or SageMaker.

---

## 38.11 The recommended architectures

### For XGBoost-style (Class 1) — Lambda is the right answer

```
   ┌────────────┐   ┌─────────────────┐   ┌──────────────────┐
   │API Gateway │──▶│ Lambda Container│──▶│ Redis (online    │
   │            │   │ - PC=5 with     │   │  feature cache)  │
   │            │   │   schedule-     │   └──────────────────┘
   │            │   │   based scaling │              │ (miss)
   │            │   │ - 3GB memory    │              ▼
   │            │   │ - module-level  │   ┌──────────────────┐
   │            │   │   model + pool  │   │ Snowflake        │
   │            │   │ - circuit       │   │ (offline, async  │
   │            │   │   breaker on    │   │  refresher cron) │
   │            │   │   feature miss  │   └──────────────────┘
   └────────────┘   └─────────────────┘
```

p99 budget: ~50-100ms warm, never cold (PC). Total monthly cost ~$200/env at moderate traffic.

### For NER/transformer-style (Class 2) — Lambda with optimization OR Fargate

```
   ┌────────────┐   ┌─────────────────┐   ┌──────────────────┐
   │SQS         │──▶│ Lambda Container│──▶│ EFS Mount        │
   │(async      │   │ - PC=2 baseline │   │ /mnt/ner_int8/   │
   │ batched)   │   │ - 5GB memory    │   │ (ONNX INT8       │
   │            │   │ - ONNX Runtime  │   │  model, shared   │
   │            │   │ - INT8 model    │   │  page cache)     │
   │            │   │ - lazy tokenizer│   └──────────────────┘
   │            │   │   warm at init  │
   └────────────┘   └─────────────────┘
```

**Decision rule for NER:** if QPS > 10 sustained, move to ECS Fargate.

```
   Lambda PC vs Fargate cost crossover (3 GB memory, NER workload):

   Lambda PC=3 always: ~$108/month
   Fargate 1 task (3GB) always: ~$30/month  ← much cheaper

   Switch when sustained traffic justifies always-on capacity.
   Lambda wins for spiky / low-traffic. Fargate wins for steady traffic.
```

### When to leave Lambda entirely

```
   Move to Fargate / ECS / EKS when:
   ✓ Cold start can't get under your SLA even with all mitigations
   ✓ Sustained QPS > 10-20 (PC cost > Fargate)
   ✓ Model too big for Lambda's 10 GB memory limit
   ✓ Need GPU (Lambda has none)
   ✓ Inference > 15 min
   ✓ Need always-on background processing

   Stay on Lambda when:
   ✓ Spiky / unpredictable traffic
   ✓ Sub-second SLA met with PC
   ✓ Small models, fast inference
   ✓ Want simpler ops
```

---

## 38.12 Observability — what to track

Cold-start observability lets you optimize, not guess. Track these metrics.

### CloudWatch metrics worth alarming

```
   ConcurrentExecutions          → know your peak (sizes PC)
   ProvisionedConcurrencyUtilization → 80% utilization = scale up PC
   ProvisionedConcurrencyInvocations → vs Invocations, ratio of warm
   ProvisionedConcurrencySpilloverInvocations → requests that didn't get warm
   InitDuration (custom metric)  → cold start init phase time
   Duration                      → end-to-end (mix of warm/cold)
   Errors                        → any non-2xx response
```

### Log Insights queries

```sql
-- p99 init duration over time
fields @initDuration
| filter @type = "REPORT" and ispresent(@initDuration)
| stats pct(@initDuration, 99) by bin(5m)

-- Cold vs warm invocations
fields @timestamp, @initDuration, @duration
| filter @type = "REPORT"
| stats
    count() as total,
    sum(case ispresent(@initDuration) when 1 then 1 else 0 end) as cold,
    avg(@duration) as avg_duration,
    avg(@duration) - avg(@initDuration) as avg_invoke_only
  by bin(5m)
```

### X-Ray tracing

Enable X-Ray to see every downstream call inside the Lambda. Critical for finding "why is this 200ms slower than I expected?" — usually a forgotten synchronous call to Snowflake or unwrapped DNS lookup.

### Custom metrics

Embed custom metrics in your handler for things CloudWatch doesn't show:

```python
def lambda_handler(event, context):
    cache_status = "hit" if redis_features else "miss"
    print(json.dumps({
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": "WithdrawalPredictor",
                "Dimensions": [["env", "model_version"]],
                "Metrics": [
                    {"Name": "FeatureCacheStatus", "Unit": "Count"},
                    {"Name": "ModelInferenceTime", "Unit": "Milliseconds"},
                ],
            }],
            "env": "prod",
            "model_version": _MODEL_VERSION,
            "FeatureCacheStatus": 1 if cache_status == "hit" else 0,
            "ModelInferenceTime": inference_ms,
        },
    }))
```

EMF (Embedded Metric Format) costs nothing extra and gives you per-invocation telemetry.

---

## 38.13 Resume tie-in — the interview answer

When an interviewer pulls on the TrueBalance bullet:

> "I solved cold-start with three layered strategies: provisioned concurrency for the SLA guarantee, container image optimization for fast Phase 1, and aggressive module-level loading for fast Phase 2.
>
> PC sized at five instances with auto-scaling by schedule — high during business hours, low at night, dropping cost about 50% versus always-on max. Multi-stage Dockerfile with the runtime image stripped of build tools, layer ordering optimized for Lambda's cache reuse across deploys. Model loaded once at module scope, Redis connection pool created once, boto3 clients created once — every warm invocation skips all of that.
>
> Memory sized at 3 GB because Lambda's CPU scales with memory, and Power Tuning showed that was the cost-vs-latency inflection point. p99 stayed under 500ms across all three environments.
>
> If we'd moved to a transformer-class model — say for the NER lender-id work — I'd have layered ONNX Runtime with INT8 quantization to shrink the model 4×, mounted it from EFS so multiple containers shared the page cache, and reconsidered ECS Fargate for sustained traffic above ~10 QPS where always-on is cheaper than PC."

That's a 90-second answer that hits the three biggest mitigations, shows the cost consciousness, and signals you've thought beyond the current model class.

---

## 38.14 Cheatsheet

```
   THE THREE PHASES
     1. Container pull   — image size, multi-stage builds, layer cache
     2. Init             — module-level loads, lazy imports, ONNX over PT
     3. Invoke           — connection pools, circuit breakers, no I/O

   TOP MITIGATIONS BY IMPACT
     1. Provisioned Concurrency (50-90% of cold-start problem solved)
     2. Container image optimization (-30-50% Phase 1)
     3. Module-level loading (-100% on warm, -10% on cold)
     4. Memory sizing 3GB+ (Lambda CPU scales with memory)
     5. ONNX + INT8 for transformers (-70% model load time)
     6. EFS mount for big models (page cache reuse across containers)
     7. Connection pooling (-50-200ms per invocation on warm)

   GOTCHAS
     - PC is on alias, not $LATEST
     - Deploy invalidates image cache (warm with synthetic invocations)
     - INIT 10-second hard limit (very large models break)
     - Reserved + PC interaction (PC must be ≤ reserved)
     - PC scale-up storms downstream (use connection limits)
     - Module-level imports run on every cold start (lazy when optional)

   WHEN TO LEAVE LAMBDA
     - Sustained QPS > 10-20 (Fargate cheaper than PC)
     - Model > 10GB or inference > 15 min
     - Need GPU
     - Cold start can't meet SLA even with all mitigations

   COST FORMULA
     PC cost = instances × memory_GB × seconds × $0.000004646
     Right-size: PC = p99 ConcurrentExecutions during peak
     Auto-scale: schedule-based saves 40-60%

   THE INTERVIEW ANSWER
     Three layers: PC (SLA guarantee) + image optimization (faster pull)
     + module-level loading (free on warm). Memory at 3GB for power
     tuning sweet spot. For transformers: ONNX + INT8 + EFS, or move to
     Fargate above ~10 QPS sustained.
```

---

End of Chapter 38. Continue back to **[Chapter 00 — Master Index](00_index.md)**.
