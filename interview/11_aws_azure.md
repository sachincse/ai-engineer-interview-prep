# Chapter 11 — AWS & Azure for AI

## Cloud skills for a polyglot AI engineer — narrative deep-dive

> Avrioc JD: "Working across AWS/Azure to deploy and scale AI solutions." Your resume is AWS-heavy (SageMaker, Lambda, VPC, ECR) with solid Azure (Databricks, Data Factory, ML Studio). This chapter gives you the language, the diagrams, and the worked examples to verbally walk through any cloud deep-dive at the whiteboard.

---

## 11.0 The big picture — why two clouds for one ML system?

Before we open up service-by-service, let's set the stage. Most production ML companies in 2026 are not single-cloud. They have one **primary** (where most workloads live) and one **secondary** (for specific tools, regional data residency, or compliance). For Avrioc — UAE-based, ad-tech-adjacent — AWS me-central-1 (Bahrain) is probably primary, with Azure pulled in for Databricks-based analytics or for Microsoft-stack customers.

The mental model I always start with: **the cloud is just primitives.** Compute, storage, network, identity, and managed services on top. Every ML platform you'll ever build is some assembly of those five buckets. AWS and Azure name them differently, but the shape is the same — once you've internalized the AWS pattern, Azure is a vocabulary swap.

```
   ┌─────────────────────── The Five Primitives ─────────────────────────┐
   │                                                                       │
   │   COMPUTE      STORAGE       NETWORK       IDENTITY      MANAGED      │
   │   ───────      ───────       ───────       ────────      ───────      │
   │   EC2 / VM     S3 / Blob     VPC / VNet    IAM / AAD     SageMaker    │
   │   Lambda       EBS / Disk    Subnets       Roles         Bedrock      │
   │   ECS / AKS    EFS / Files   Security      Secrets       Databricks   │
   │   EKS                        Groups        KMS           OpenAI       │
   │                                                                       │
   │   Everything else is a recombination of these five.                   │
   └───────────────────────────────────────────────────────────────────────┘
```

> **How to say this in an interview:** "When I evaluate a cloud architecture, I think in five layers — compute, storage, network, identity, managed services. AWS and Azure use different names but solve the same problems. So when a stakeholder says 'we need to move from AWS to Azure,' my first conversation is which of those five layers actually need to move versus which can stay abstracted behind a gateway."

---

## 11.1 AWS for ML — the full landscape diagram

This is the diagram I draw first on every AWS deep-dive call. It anchors every follow-up question.

```
   ┌────────────────────── AWS AI/ML Reference Architecture ─────────────────────┐
   │                                                                              │
   │   ┌── Data Layer ──────────────────────────────────────────────────────┐    │
   │   │  S3 (lake)  Glue (catalog)  Athena (SQL)  Lake Formation (gov)    │    │
   │   │  Kinesis (streams)  MSK (Kafka)  RDS / Aurora / DynamoDB          │    │
   │   └────────────────────────────────────────────────────────────────────┘    │
   │                                  │                                            │
   │   ┌── Training Layer ────────────▼───────────────────────────────────────┐   │
   │   │  SageMaker Training Jobs (managed PyTorch/TF/HF)                    │   │
   │   │  SageMaker Processing Jobs (Spark, sklearn preprocess)              │   │
   │   │  SageMaker Pipelines (DAG of training steps)                        │   │
   │   │  EC2 P5/P4d (DIY training on H100/A100)                             │   │
   │   │  EKS + Ray Train (custom distributed training)                      │   │
   │   └────────────────────────────────────────────────────────────────────┘    │
   │                                  │                                            │
   │   ┌── Registry & CI/CD ──────────▼───────────────────────────────────────┐   │
   │   │  SageMaker Model Registry  ECR (containers)  S3 (artifacts)         │   │
   │   │  CodeCommit → CodeBuild → CodePipeline → CodeDeploy                 │   │
   │   │  EventBridge (event triggers) Step Functions (orchestration)        │   │
   │   └────────────────────────────────────────────────────────────────────┘    │
   │                                  │                                            │
   │   ┌── Inference Layer ───────────▼───────────────────────────────────────┐   │
   │   │  SageMaker Endpoint (real-time / async / serverless / batch)        │   │
   │   │  Lambda (lightweight, container image up to 10 GB)                  │   │
   │   │  EKS + KServe / Ray Serve / vLLM (GPU LLM serving)                  │   │
   │   │  Bedrock (managed Claude/Nova/Llama API)                            │   │
   │   └────────────────────────────────────────────────────────────────────┘    │
   │                                  │                                            │
   │   ┌── Observability & Governance ▼───────────────────────────────────────┐   │
   │   │  CloudWatch (metrics+logs+alarms)  X-Ray (tracing)                  │   │
   │   │  SageMaker Model Monitor (drift)  CloudTrail (audit)                │   │
   │   │  IAM / KMS / Secrets Manager / WAF                                  │   │
   │   └────────────────────────────────────────────────────────────────────┘    │
   │                                                                              │
   │   Networking spine: VPC + subnets + SG + VPC endpoints + Route53 + ALB      │
   └──────────────────────────────────────────────────────────────────────────────┘
```

> **How to say this in an interview:** "AWS for ML breaks into five layers — data, training, registry/CI-CD, inference, and governance. Underneath them is the VPC networking spine. When I architect, I trace a single feature request through all five — for example, a fraud-score request lands at an ALB, hits a Lambda or SageMaker endpoint that pulled its model from ECR via S3, ran in a VPC private subnet, fetched features from RDS through a VPC endpoint, and emitted CloudWatch metrics that an EventBridge rule watches for retraining triggers."

---

## 11.2 SageMaker — what it actually is

### Why SageMaker exists (the problem it solves)

Before SageMaker, training a model in AWS meant: spin up EC2, install CUDA + PyTorch + your code, copy data from S3, run, save artifacts back, tear down. Every team built their own version of this script. Failure recovery, spot interruption, multi-GPU coordination, hyperparameter sweeps, distributed training — every one of these was rebuilt across teams. **SageMaker is the managed assembly of those primitives**: you hand it a Docker image and a training script, it handles EC2 lifecycle, distributed training coordination, S3 IO, checkpoints, and metrics.

### Mental model with analogy

SageMaker is like a **catering service for ML compute**. You don't rent the kitchen (EC2), you don't buy the ingredients (storage), you just hand the chef a recipe (your container + script) and a guest count (instance count + type) and they deliver. You pay a small premium over EC2 for the convenience and reliability.

### SageMaker's real surface area

```
   ┌────────────────────── SageMaker (zoomed in) ────────────────────────┐
   │                                                                      │
   │   STUDIO              ── notebook IDE; entry point for everything   │
   │                                                                      │
   │   PROCESSING JOBS     ── one-shot Spark / sklearn / Ray batch       │
   │   TRAINING JOBS       ── managed training with metrics, checkpoint  │
   │   HPO TUNING JOBS     ── Bayesian / random HP search                │
   │                                                                      │
   │   MODEL REGISTRY      ── versioned models with approval workflow    │
   │   MODEL CARDS         ── governance metadata                        │
   │                                                                      │
   │   ENDPOINTS                                                          │
   │     ├─ Real-time   (always-on, sub-second SLA)                      │
   │     ├─ Async       (SQS-backed, scale-to-zero, up to 60 min)        │
   │     ├─ Serverless  (CPU-only, cold-start, sporadic)                 │
   │     └─ Batch       (transform S3 → S3, no endpoint)                 │
   │                                                                      │
   │   Endpoint variants                                                  │
   │     ├─ Single-model        (one container)                          │
   │     ├─ Multi-model (MME)   (many small models, same container)      │
   │     ├─ Multi-container (MCE) (up to 15 hetero containers, A/B)      │
   │     └─ Inference Pipeline   (preprocess → model → postprocess)      │
   │                                                                      │
   │   PIPELINES           ── DAG of steps (preprocess → train → eval)   │
   │   FEATURE STORE       ── online + offline feature serving           │
   │   MODEL MONITOR       ── data + concept drift detection             │
   │   CLARIFY             ── bias + explainability                      │
   │   JUMPSTART           ── pre-trained models marketplace             │
   └─────────────────────────────────────────────────────────────────────┘
```

### SageMaker pricing — the senior-engineer-level view

You pay for **instance-hours plus storage plus data transfer**. Three pricing surprises that bite teams:

1. **Endpoint-hours bill while idle.** If you provision an `ml.g5.2xlarge` real-time endpoint and it serves zero traffic for a weekend, you still pay ~$1.21/hr × 48hr = ~$58. Multiply by environments and models — easily five figures a month wasted. Use Async or Serverless for low-traffic.
2. **Notebook instances bill while attached, even if your kernel is idle.** Always set lifecycle scripts to auto-stop after N idle hours.
3. **Multi-AZ doubles your floor.** If you require high availability, `min_instances=2` across AZs is your floor cost.

### Worked example — sizing a SageMaker real-time endpoint

Suppose Avrioc asks: "We need to serve a fraud-detection XGBoost model. 200 requests/sec peak, 50/sec average. p99 < 200 ms. Cost-conscious."

```
   Step 1 — Single-instance throughput
     Profile a 100-tree XGBoost on ml.c6i.large (CPU-only): ~800 RPS at p99 ~120 ms.
   
   Step 2 — Required instance count
     Peak 200 RPS / 800 RPS-per-instance = 0.25 instances → 1 instance is enough.
     But for HA across AZs: minimum 2 instances.
   
   Step 3 — Auto-scaling policy
     target_tracking on InvocationsPerInstance, target=400 (50% headroom).
     min=2, max=6.
   
   Step 4 — Cost
     ml.c6i.large = $0.085/hr × 2 instances × 730 hr/month = $124/month base.
     During peak, add ~1 instance for ~10 hours/day × 30 = 300 hr × $0.085 = $26.
     Total ~$150/month. Compare to Bedrock at $X per 1M requests if relevant.
```

### SageMaker endpoint variants — when each (the table you draw)

| Variant | Cold-start | Latency | Use case | Pricing |
|---------|------------|---------|----------|---------|
| Real-time | None (always on) | <100 ms | User-facing chatbot, fraud-score | Per instance-hour |
| Async | Few minutes (scale-to-zero) | Seconds-minutes | Document processing, video transcribe | Per instance-hour while active |
| Serverless | 5–30 s (Lambda-style) | 100 ms – 1 s | Internal tools, sporadic | Per request + memory-time |
| Batch Transform | N/A (job-based) | Minutes-hours | Nightly scoring | Per instance-hour for job duration |
| MME (multi-model) | Per-model load 1-10 s | <500 ms hot | Many tenant-specific small models | Per instance-hour |
| MCE (multi-container) | None | <100 ms | A/B testing, ensembles, chains | Per instance-hour |
| Inference Pipeline | None | <500 ms (sum) | preprocess → model → postprocess | Per instance-hour |

### Common SageMaker mistakes (senior-level gotchas)

1. **Forgetting to enable data capture.** Without it, you cannot run Model Monitor — you have no record of what predictions went out. Always enable on day one with reasonable sampling (100% for low QPS, 5–10% for high).
2. **Hardcoding instance type in Pipeline definitions.** Makes promotion across dev/staging/prod painful. Parameterize via `ParameterString`.
3. **Treating the Model Registry as a write-only log.** It's an approval workflow — your CI should require `ModelApprovalStatus=Approved` before deploying to prod.

### SageMaker Q&A

**Q1. Walk me through SageMaker training job lifecycle.**
> The SDK posts a CreateTrainingJob API call with my container URI in ECR, my hyperparameters, my input/output S3 channels, and instance config. SageMaker provisions the requested instance, mounts S3 input as a local directory or pipe, pulls my image, runs the entry point script with hyperparameters as env vars or CLI args, captures stdout/stderr to CloudWatch Logs, monitors for the metric regex I provided, and on exit copies `/opt/ml/model` back to my S3 output prefix. If it's a spot job, SageMaker handles interruption and resume from the last checkpoint. The whole lifecycle is observable through CloudWatch and the Studio UI.

**Q2. When would you choose Async over Real-time endpoint?**
> Async is the right choice when individual inferences are slow — minutes for OCR on a long PDF, video transcription, or large-context LLM completions. The endpoint scales to zero when the SQS queue is empty, so you don't pay for idle GPU. The trade-off is latency from the user's perspective — they get an SNS notification or poll the output S3, not a synchronous response. For Avrioc, I'd use Async for any workload where the user accepts a "we'll email you when ready" UX.

**Q3. What's the difference between MME and MCE — give a concrete use case for each.**
> MME, Multi-Model Endpoint, is built for the case where you have hundreds or thousands of homogeneous small models — say, one fraud model per merchant. They share one container; SageMaker loads them on-demand from S3 and caches them. At ResMed I've used MME for tenant-specific recommendation models. MCE, Multi-Container Endpoint, supports up to 15 heterogeneous containers — different runtimes, different frameworks. It's perfect for A/B testing two model versions or chaining preprocess in TensorFlow + main model in PyTorch + postprocess in sklearn. At ResMed we used MCE for a multi-container clinical pipeline.

**Q4. How do you do canary deployments with SageMaker endpoints?**
> Two ways. Production Variants on a single endpoint — define two ProductionVariants with weight 90/10 between them, traffic splits accordingly, observe metrics, then shift weights gradually. Or via Endpoint Configuration updates — create a new EndpointConfig pointing at the new model, then UpdateEndpoint with deployment config that includes a CanaryTrafficShifting block. CloudWatch alarms can trigger an automatic rollback if error rate spikes. I prefer Production Variants for fast iteration and EndpointConfig for clean GitOps.

**Q5. SageMaker Pipelines vs Step Functions — when each?**
> SageMaker Pipelines is purpose-built for ML — it understands TrainingStep, ProcessingStep, RegisterModelStep natively, integrates with the Model Registry, gives lineage tracking. It's the right tool for the training-to-registration loop. Step Functions is general-purpose orchestration — better when your workflow spans beyond ML, e.g., trigger a SageMaker job + push results to a CRM + send an email. At ResMed I used SageMaker Pipelines for the model training DAG and Step Functions to kick off Pipelines on data arrival events.

**Q6. [Gotcha] Your endpoint's p99 spikes every few minutes. Debug?**
> First I'd pull CloudWatch ModelLatency at p99 and OverheadLatency. If OverheadLatency is the culprit, it's SageMaker-side — usually auto-scaling thrashing or container restarts; check ScalingActivities and instance health. If ModelLatency, it's your container — likely a cold tokenizer load on a new instance, garbage collection pause in Python, or KV-cache fragmentation if it's an LLM. I'd enable data capture, sample the slow requests, and inspect input distributions — a long-tail of unusually long inputs can cause this. Triton Inference Server replaces TorchServe for tighter Python-free latency.

---

## 11.3 Lambda for ML — the Sachin TrueBalance story

### Why Lambda exists

Lambda is **serverless compute**: you upload code (or a container image up to 10 GB), AWS runs it on demand, scales to thousands of concurrent invocations, and you pay only for compute-milliseconds. No instances to manage. The trade-off is constraints: 15-minute max runtime, 10 GB RAM ceiling, no GPU, and **cold start** — the first invocation on a fresh instance pays an init penalty.

### Mental model with analogy

**Lambda is like calling Uber for compute** — pay per trip, you don't own the car. The cold start is waiting for the driver to arrive. For frequent rides (high QPS), you can pre-book (provisioned concurrency); for occasional rides, you accept the wait.

### Lambda's surface area for ML

```
   ┌─────────────────────── Lambda for ML ───────────────────────┐
   │                                                              │
   │   Triggers:                                                  │
   │     API Gateway ──┐                                          │
   │     Function URL ─┤                                          │
   │     ALB ──────────┤                                          │
   │     S3 event ─────┼──▶  λ Function                           │
   │     SQS / SNS ────┤      ├─ Runtime: Python 3.12 / container │
   │     EventBridge ──┘      ├─ RAM: 128 MB – 10 GB              │
   │                          ├─ Ephemeral /tmp: 512 MB – 10 GB   │
   │   Outputs:               ├─ Max duration: 15 min             │
   │     Return value         ├─ Concurrency: 1000 default        │
   │     S3 / DynamoDB        └─ VPC: optional ENI attachment     │
   │     SNS / SQS                                                │
   │                                                              │
   │   Cold start sources:                                        │
   │     1. Init container / runtime (~200-500ms)                 │
   │     2. ENI attachment if VPC (~adds 200-1000ms first run)    │
   │     3. Your code init (model load, DB connect)               │
   └──────────────────────────────────────────────────────────────┘
```

### Worked example — your TrueBalance XGBoost Lambda

Let me reconstruct the architecture for the whiteboard. This is the kind of story Avrioc will want to hear.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                  TrueBalance Withdraw-Predictor on Lambda            │
   │                                                                       │
   │   Mobile app                                                          │
   │      │ HTTPS                                                          │
   │      ▼                                                                │
   │   ┌──────────────┐                                                    │
   │   │ API Gateway  │ ── auth via Cognito JWT, rate-limit per user      │
   │   └──────┬───────┘                                                    │
   │          │                                                            │
   │          ▼                                                            │
   │   ┌──────────────┐                                                    │
   │   │   Lambda     │ ── container image, 1.5 GB, 1769 MB memory        │
   │   │   (in VPC)   │ ── provisioned concurrency: 5 (warm)              │
   │   └──┬─────┬─────┘                                                    │
   │      │     │                                                          │
   │      │     │   VPC private subnet                                     │
   │      │     │                                                          │
   │      ▼     ▼                                                          │
   │   ┌─────────┐  ┌──────────────┐                                       │
   │   │ Redis   │  │ Snowflake    │ ── feature store: real-time          │
   │   │(Elastic │  │ via private  │     transactions in Redis;            │
   │   │ Cache)  │  │ link         │     historical aggregates pre-        │
   │   └─────────┘  └──────────────┘     computed in Snowflake             │
   │                                                                       │
   │   Side-paths via VPC endpoints:                                       │
   │     S3 (Gateway) ── model.pkl pulled at cold start                    │
   │     Secrets Manager (Interface) ── Snowflake creds                    │
   │     CloudWatch Logs (Interface) ── structured logs                    │
   │                                                                       │
   │   Observability:                                                      │
   │     CloudWatch custom metric: Latency p99 per env per model           │
   │     X-Ray tracing: Lambda → Redis → Snowflake spans                   │
   │     Alarm: p99 > 500ms for 3 consecutive minutes                      │
   │                                                                       │
   │   Three-environment isolation:                                        │
   │     dev / staging / prod = three separate VPCs (non-overlapping CIDR) │
   │     three ECR repos, three KMS CMKs, three IAM role chains            │
   │     Terraform module instantiated three times                         │
   └──────────────────────────────────────────────────────────────────────┘
```

### Three-minute architecture narrative (memorize this)

> "At TrueBalance I built a real-time withdraw-prediction service on Lambda with p99 under 500 milliseconds. The mobile app calls API Gateway, which authenticates via Cognito and forwards to a Lambda deployed as a container image — about 1.5 GB with the XGBoost runtime and our trained model embedded. The Lambda is attached to a private subnet in our prod VPC. On invocation it reads real-time signals from Redis ElastiCache — last hour transactions, recent device fingerprints — and historical aggregates from Snowflake over a private link. The XGBoost predict runs in roughly 30 milliseconds, and we emit a CloudWatch custom metric with the latency tagged by model name and environment. Provisioned concurrency keeps five warm instances so cold starts don't blow our SLA, and the whole stack — VPC, Lambda, ECR repo, KMS key, Secrets Manager — is repeated dev/staging/prod via a Terraform module with three workspaces. We caught drift via a daily SageMaker Model Monitor job that read the data-capture S3 prefix."

### Why Lambda for that workload, not SageMaker?

This is a question you'll get. The honest answer:

- Sustained QPS was below 1 per second — at SageMaker pricing the floor cost (one always-on `ml.c6i.large` × 2 AZs) is more than Lambda's per-request cost
- The model was small (XGBoost, ~50 MB) — Lambda's 10 GB image limit is plenty
- Strong AWS-native integration (Cognito, API Gateway, IAM) made Lambda fit the broader app stack
- 15-min timeout was irrelevant — predictions take 30 ms

If sustained QPS had been above ~10/sec, SageMaker Real-time would have won.

### Lambda cold start optimization (senior-level checklist)

1. **Container image basics**: slim base, multi-stage build, prune dev deps, `--platform linux/amd64` consistency.
2. **Provisioned concurrency**: pre-warmed instances. Cost: pay for warm hours. Tune via target tracking on `ProvisionedConcurrencyUtilization`.
3. **SnapStart** (Java; Python in preview): pre-initialized runtime snapshot, eliminates init-time.
4. **VPC ENI cost**: first invocation on a cold ENI adds ~200-1000 ms historically. AWS now caches ENIs ("Hyperplane"); still measure.
5. **Lazy-load**: imports of pandas, numpy, sklearn dominate cold start. If a function path doesn't need them, defer the import.
6. **Init outside the handler**: model load, DB connection — do at module scope so they persist across warm invocations.

### Lambda Q&A

**Q1. Lambda vs SageMaker — break-even rule of thumb?**
> Below roughly one sustained request per second, Lambda is cheaper because SageMaker's floor is an instance-hour. Above ten sustained requests per second on a single model, SageMaker Real-time wins on cost-per-request and gives you GPU options. Between one and ten, it depends — SageMaker Serverless is a good middle ground if your model is CPU-only and under 6 GB. For my TrueBalance XGBoost case, sustained QPS was well below one, so Lambda was the right call.

**Q2. Cold start is killing your SLA. What's your playbook?**
> First I measure where the time is going — is it ENI attachment, runtime init, container pull, or my model load? CloudWatch Logs Insights can give me INIT_REPORT durations. Then I attack the biggest one. ENI attach: I check if AWS Hyperplane caching is helping (it usually is by 2026); for Python 3.12 SnapStart is now GA and gives near-zero init. Container pull: minimize image size; multi-stage build; prune. Model load: smaller artifact via quantization or move to S3 + lazy load. Last resort: provisioned concurrency for predictable warmth.

**Q3. Your Lambda is in a VPC. What do you absolutely need?**
> A VPC endpoint for every AWS service the Lambda talks to that I don't want flowing through a NAT Gateway. At minimum: S3 Gateway endpoint for any artifact pulls, Secrets Manager Interface endpoint for credentials, CloudWatch Logs Interface endpoint for logging. If the Lambda calls Bedrock or DynamoDB, those each need their own endpoint. Forgetting Logs is a classic — your function runs but no logs appear and you can't debug.

**Q4. Why not put the model file inside Redis or DynamoDB?**
> Two reasons. First, Redis has memory cost — a 50 MB model duplicated across replicas adds up. Second, model loading is a one-time cost per container instance — Lambda already caches across warm invocations. The right pattern is S3 for model artifact, in-process load at module init, in-memory inference. Redis is for *features* (per-user state) not for *model weights*.

**Q5. How do you do per-environment isolation for Lambda?**
> Three separate VPCs with non-overlapping CIDR ranges — say 10.10.0.0/16 dev, 10.20.0.0/16 staging, 10.30.0.0/16 prod. Each environment has its own ECR repository, its own KMS customer-managed key, its own Secrets Manager secrets, its own IAM execution role. Terraform module accepts an `env` variable and instantiates everything three times via three workspaces or three state files. Cross-environment access is forbidden — no Transit Gateway between them — so a bug in dev cannot accidentally hit prod data.

---

## 11.4 ECR — your container image lifecycle

### Why ECR matters

Every Lambda container, every SageMaker training job, every ECS task pulls its image from ECR. If your ECR strategy is sloppy — overwritten tags, no scanning, no lifecycle policy — you'll spend money on storage you don't need and ship vulnerable images to production.

### Mental model

ECR is **git for Docker images**. Each repository is a project; each tag is a commit. Image digests are the immutable SHAs; tags are mutable pointers. Treat tags like git branches — `latest` is a symlink that moves; `git-abc123` is a permanent commit reference.

### ECR best-practice checklist

```
   ┌────────────────── ECR for Production ML ──────────────────┐
   │                                                            │
   │   Repository naming:                                       │
   │     ml/<service>/<env>     e.g., ml/fraud-api/prod         │
   │                                                            │
   │   Tagging:                                                 │
   │     IMMUTABLE — set repo to immutable, no overwrites       │
   │     TAG with: git SHA + semver + env                       │
   │     e.g., fraud-api:v1.2.3-abc1234-prod                    │
   │                                                            │
   │   Lifecycle policy:                                        │
   │     Untagged > 7 days  ── delete                           │
   │     Tagged old > 30 days, keep last 10 ── delete           │
   │     ML images are 5-15 GB; without policy storage explodes │
   │                                                            │
   │   Scanning:                                                │
   │     Enhanced Scanning (Inspector) — continuous CVE         │
   │     Block deploy if HIGH/CRITICAL via CodePipeline gate    │
   │                                                            │
   │   Pull-through cache:                                      │
   │     Mirror Docker Hub / quay.io / nvcr.io                  │
   │     Faster pulls, availability if upstream is down         │
   │                                                            │
   │   Cross-region replication:                                │
   │     For multi-region deploys; replicate prod images        │
   │                                                            │
   │   Access control:                                          │
   │     IAM role per env, no shared root                       │
   │     Repository policies for cross-account                  │
   └────────────────────────────────────────────────────────────┘
```

### Common ECR mistakes

1. **Mutable tags in production.** Someone re-pushes `:latest` and a working pod crashes on next pull. Always use immutable tags + git SHA.
2. **No lifecycle policy.** ML images are huge. A team of 5 engineers shipping daily can rack up terabytes of unreferenced layers. Always set a 7-day untagged purge.
3. **Skipping scan-on-push.** A CVE in your CUDA base image gets picked up automatically by Inspector — but only if scanning is enabled. Make it default in your Terraform module.

### ECR Q&A

**Q1. Why do you prefer immutable tags?**
> A mutable tag is a moving target. If two pods in the same Deployment pull at slightly different times, they could end up running different bits, which is a debugging nightmare. With immutable tags — typically tied to git SHA — the tag forever points to a specific image digest. Rolling back is just deploying a previous tag. It's the same reason we don't reuse git commit SHAs.

**Q2. How do you handle ECR storage costs at scale?**
> Two levers. First, lifecycle policies — auto-delete untagged after 7 days, keep last 10 tagged versions per repo, archive stale repos. Second, cross-region replication only for repos that need it; don't replicate dev or experimental images. For Avrioc-scale, I'd also use ECR Pull-Through Cache for Docker Hub and nvcr.io to avoid pulling the same NVIDIA base image 100 times.

---

## 11.5 EC2, EBS, EFS — when SageMaker isn't enough

### Why you'd skip SageMaker

SageMaker is great for the 90% case but loses on:
- Bleeding-edge frameworks (vLLM nightly, Ray dev branch) that aren't in the SageMaker AMI
- Multi-tenant control where you want exact instance pinning
- Cost-sensitive long-running workloads where the SageMaker premium hurts
- Complex networking where you need raw VPC primitives

For those, you go to **EC2 directly** and put your own orchestrator on top (often Kubernetes; see Chapter 12).

### EC2 instance families for ML

```
   ┌────────────────────── EC2 ML Instance Cheat Sheet ──────────────────────┐
   │                                                                          │
   │   FAMILY    GPU                  VRAM    USE CASE                       │
   │   ─────     ───                  ────    ────────                       │
   │   g4dn      NVIDIA T4 (1-8)       16GB   Cheap inference, small models  │
   │   g5        NVIDIA A10G (1-8)     24GB   Mid-range inference, RAG       │
   │   g6        NVIDIA L4 (1-8)       24GB   New gen, better $/throughput   │
   │   p3        NVIDIA V100           16GB   Legacy training                │
   │   p4d       NVIDIA A100 (8)       40GB   Big training, 70B+ models      │
   │   p4de      NVIDIA A100 (8)       80GB   Bigger training                │
   │   p5        NVIDIA H100 (8)       80GB   State-of-art training+serving  │
   │   p5e       NVIDIA H100 (8)       141GB  Even bigger context            │
   │   inf1      AWS Inferentia 1      —      Cost-optimized inference (TF)  │
   │   inf2      AWS Inferentia 2      —      LLM inference, OpenAI-compat   │
   │   trn1      AWS Trainium          —      Training cost-optimized        │
   │                                                                          │
   │   Pricing (us-east-1 on-demand, approximate):                           │
   │     g5.2xlarge      $1.21/hr                                             │
   │     g5.12xlarge     $5.67/hr  (4× A10G)                                  │
   │     p4d.24xlarge   $32.77/hr  (8× A100-40GB)                             │
   │     p5.48xlarge   $98.32/hr   (8× H100-80GB)                             │
   │   Spot: 50-90% discount; can be interrupted with 2-min warning.         │
   └──────────────────────────────────────────────────────────────────────────┘
```

### EBS — block storage for an instance

EBS is your local SSD. Two types matter for ML:

- **gp3** — general-purpose SSD, 3,000 baseline IOPS, 125 MB/s throughput. Default for most workloads.
- **io2** — provisioned IOPS, up to 256,000 IOPS, multi-attach. Use for high-throughput training that hits the disk hard.

Worked example: A PyTorch DataLoader with 16 workers reading random 1 KB samples can spike to 50,000+ IOPS during epoch start. gp3 will throttle at 3,000 — your epoch first batch takes 30 seconds instead of 2. Either provision higher IOPS gp3 (up to 16,000) or move to io2.

### EFS — shared filesystem

EFS is **NFS as a service**. Multiple EC2 instances mount the same filesystem; ideal for:
- Shared training datasets across distributed workers
- Shared model artifact directory across an inference fleet
- Notebook home directories shared across a team

Trade-off: latency is higher than EBS (~1-5 ms vs <1 ms for EBS). Don't use EFS for hot inference paths; use it for shared static state.

### Worked example — provisioning a Ray training cluster on EC2

Suppose you need to fine-tune Llama-3 70B with LoRA. You'd choose:
- 4× p4d.24xlarge instances (4 nodes × 8 A100 = 32 A100 GPUs total)
- gp3 boot volume per node (200 GB)
- EFS mounted at `/mnt/dataset` for shared training data
- VPC private subnet with security group allowing intra-cluster NCCL traffic on TCP 1024-65535
- IAM instance role with S3 read for dataset and S3 write for checkpoints
- Ray head node bootstrap script + ray worker auto-join

Cost back-of-envelope: 4 × $32.77/hr × 6 hrs training = ~$786 per fine-tune run. Spot would cut to ~$236.

---

## 11.6 VPC patterns for ML

### Why VPC matters

Public-internet ML endpoints expose model artifacts and inference API to the world. Regulated workloads (healthcare, finance, GDPR, UAE PDPL) require **VPC isolation**: traffic stays inside AWS, no public IPs, no internet egress unless explicitly allowed. Plus, VPC is where you implement multi-tenant separation.

### Mental model with analogy

Think of VPC as **your private floor in a skyscraper**. AWS owns the building (the data center). Your VPC is your floor — you control which doors exist (subnets), who has keycards (security groups, NACLs), which utilities are piped in (VPC endpoints), and whether there's an elevator to the lobby (NAT Gateway / IGW).

### Reference VPC for ML

```
   ┌──────────────────────── ML VPC Reference ─────────────────────────────┐
   │                                                                        │
   │   VPC: 10.20.0.0/16 (prod)                                             │
   │                                                                        │
   │   ┌── AZ-1 (me-central-1a) ──┐    ┌── AZ-2 (me-central-1b) ──┐        │
   │   │                            │    │                            │        │
   │   │ Public subnet              │    │ Public subnet              │        │
   │   │   10.20.1.0/24             │    │   10.20.2.0/24             │        │
   │   │     ALB                    │    │     ALB                    │        │
   │   │     NAT Gateway            │    │     NAT Gateway            │        │
   │   │                            │    │                            │        │
   │   │ Private subnet (app)       │    │ Private subnet (app)       │        │
   │   │   10.20.10.0/24            │    │   10.20.11.0/24            │        │
   │   │     Lambda ENIs            │    │     Lambda ENIs            │        │
   │   │     EKS app pods           │    │     EKS app pods           │        │
   │   │                            │    │                            │        │
   │   │ Private subnet (data)      │    │ Private subnet (data)      │        │
   │   │   10.20.20.0/24            │    │   10.20.21.0/24            │        │
   │   │     RDS / Aurora           │    │     RDS replica            │        │
   │   │     ElastiCache            │    │     ElastiCache replica    │        │
   │   │                            │    │                            │        │
   │   │ Private subnet (gpu)       │    │ Private subnet (gpu)       │        │
   │   │   10.20.30.0/24            │    │   10.20.31.0/24            │        │
   │   │     EKS GPU node group     │    │     EKS GPU node group     │        │
   │   │     vLLM pods              │    │     vLLM pods              │        │
   │   │                            │    │                            │        │
   │   └────────────────────────────┘    └────────────────────────────┘        │
   │                                                                        │
   │   VPC Endpoints (no NAT cost):                                         │
   │     com.amazonaws.me-central-1.s3              (Gateway)               │
   │     com.amazonaws.me-central-1.dynamodb        (Gateway)               │
   │     com.amazonaws.me-central-1.ecr.api         (Interface)             │
   │     com.amazonaws.me-central-1.ecr.dkr         (Interface)             │
   │     com.amazonaws.me-central-1.secretsmanager  (Interface)             │
   │     com.amazonaws.me-central-1.logs            (Interface)             │
   │     com.amazonaws.me-central-1.bedrock-runtime (Interface)             │
   │     com.amazonaws.me-central-1.sagemaker.*     (Interface, multiple)   │
   │                                                                        │
   │   Why VPC endpoints?                                                   │
   │     1. Traffic to AWS services stays on AWS backbone — no internet     │
   │     2. No NAT Gateway data-processing fees (NAT is $0.045/GB!)         │
   │     3. Security: scope per-endpoint policy (only this VPC can use)     │
   └────────────────────────────────────────────────────────────────────────┘
```

### NAT Gateway costs — the senior-engineer landmine

NAT Gateway charges $0.045 per GB processed. An ML training job that downloads 500 GB of training data from S3 pays $22.50 just for the NAT processing — *if* it goes through NAT instead of an S3 Gateway endpoint. With a Gateway endpoint, that cost is **zero**. Multiply by retraining frequency, multi-AZ, multi-environment, and missing endpoints can cost five-figures monthly.

### VPC Q&A

**Q1. Why does a SageMaker training job in a private subnet sometimes hang forever?**
> Almost always missing VPC endpoints. The training container starts, tries to pull additional dependencies or download a HuggingFace base model, and the request goes to a public DNS that the private subnet can't reach. Without a NAT Gateway or a VPC endpoint to the destination, the connection times out. The fix is always: enumerate every external dependency and either add a VPC endpoint or whitelist the route through NAT. The mysterious-hang is the #1 SageMaker support ticket.

**Q2. How do you do multi-tenant isolation for ML?**
> Two patterns. Hard isolation: separate VPC per tenant with non-overlapping CIDRs and Transit Gateway only for explicitly-shared services. This is for high-trust regulated tenants (healthcare). Soft isolation: shared VPC with per-tenant Security Groups, Network Policies in EKS, IAM roles per tenant, KMS CMK per tenant for encryption. This is for SaaS where 100+ tenants make per-VPC unworkable. At Avrioc scale I'd start with soft isolation and graduate specific high-trust customers to dedicated VPCs.

**Q3. Why do you put RDS in a private subnet but not the same subnet as Lambda?**
> Subnet separation is defense-in-depth. The application subnet is reachable from the load balancer and may host code that's exposed to the internet. The data subnet should be reachable only from the application subnet, never from the internet. If a Lambda gets compromised, the blast radius is limited — the attacker can't pivot to admin tools sitting somewhere else. Security Groups enforce the rule: data SG allows ingress only from app SG.

---

## 11.7 CloudWatch — metrics, logs, alarms

### Why CloudWatch matters for ML

Every model in production needs three things observed: **request rate, error rate, latency**, plus model-specific metrics like prediction confidence distribution, drift signals, token usage. CloudWatch is the AWS-native answer; teams often supplement with Prometheus/Grafana on EKS, but CloudWatch is the default everywhere AWS-managed services live.

### Mental model

CloudWatch is **three products with one name**: Metrics (numerical time-series, 1-min granularity by default), Logs (text + structured logs), and Alarms (metric thresholds that trigger SNS / EventBridge). Internally they're separate services with separate APIs.

### Custom metrics for ML

The CloudWatch metric every senior ML engineer publishes:

```python
import time
from datetime import datetime
import boto3

cw = boto3.client('cloudwatch')

def emit_inference_metric(model_name, env, latency_ms, success):
    cw.put_metric_data(
        Namespace='ML/Inference',
        MetricData=[
            {
                'MetricName': 'LatencyMs',
                'Dimensions': [
                    {'Name': 'ModelName', 'Value': model_name},
                    {'Name': 'Env',       'Value': env},
                ],
                'Value': latency_ms,
                'Unit': 'Milliseconds',
                'Timestamp': datetime.utcnow(),
            },
            {
                'MetricName': 'Invocations',
                'Dimensions': [
                    {'Name': 'ModelName', 'Value': model_name},
                    {'Name': 'Env',       'Value': env},
                    {'Name': 'Success',   'Value': str(success)},
                ],
                'Value': 1,
                'Unit': 'Count',
            },
        ]
    )
```

**Walkthrough:** `Namespace` is the bucket — `ML/Inference` for all your model metrics. `Dimensions` are tags — model name and env let you slice metrics across hundreds of models without exploding the namespace. Publishing on every invocation costs $0.30 per million metric data points; for high-QPS, batch via the Embedded Metric Format (EMF) instead — log JSON to CloudWatch Logs with a special `_aws.CloudWatchMetrics` block and CloudWatch auto-extracts metrics for free.

### Log Insights queries (the senior-engineer skill)

```sql
fields @timestamp, model_name, latency_ms, error
| filter env = "prod"
| stats pct(latency_ms, 99) by model_name, bin(1m)
| sort @timestamp desc
```

This gives you per-model p99 over time. The `pct()` function does percentile in-place. Use Log Insights when CloudWatch metrics don't exist yet for the dimension you need.

### Alarms that matter for ML

```
   ALARM                                THRESHOLD              ACTION
   ─────                                ─────────              ──────
   Endpoint p99 latency                 > SLO × 1.2 for 5min   PagerDuty
   Endpoint 5xx rate                    > 1% for 3min          PagerDuty + auto-rollback
   Invocations dropped to 0             0 for 10min            Slack (upstream issue?)
   Drift score (Model Monitor)          > 0.3                  Slack + retrain trigger
   GPU utilization                      < 20% for 30min        Cost review (oversized?)
   Cost anomaly                         > 2x baseline for 1hr  Email + Slack
```

### CloudWatch Q&A

**Q1. Why use Embedded Metric Format instead of put-metric-data?**
> Cost and atomicity. PutMetricData is $0.30 per million points; at high QPS, that adds up. EMF lets you log a single JSON line per request and CloudWatch auto-extracts metrics from it for free — you only pay for the log storage. Plus, the metric and the log line are atomic — you don't risk metric being emitted but log being lost or vice versa. EMF is what I'd default to for any new high-QPS service.

**Q2. How do you alert on prediction drift?**
> SageMaker Model Monitor runs scheduled jobs comparing live captured data to a baseline. It emits CloudWatch metrics like `feature_baseline_drift_<feature_name>`. I set CloudWatch Alarms on those metrics — threshold around 0.3 for KS-statistic-style drift, alarmed for 24 hours of consecutive breach to avoid false alarms. The alarm sends to SNS which fans out to PagerDuty for prod and Slack for everything else. For LLM apps where Model Monitor doesn't apply, I track input embedding distribution drift via a custom metric.

---

## 11.8 EventBridge — event bus for MLOps

### Why EventBridge

EventBridge is **the cron-and-events service**. Two faces:
- **Schedule** — replace Cron / Lambda Schedule. "Every Tuesday 2am, retrain."
- **Event Bus** — listen to AWS service events ("S3 put in this bucket," "SageMaker job changed state to Failed") and route them.

It glues your MLOps DAG together without you writing polling code.

### Worked example — full retraining pipeline

```
   ┌──────────── EventBridge-Driven Retraining Pipeline ──────────────┐
   │                                                                    │
   │   New labeled data lands in s3://prod-ml/labels/                   │
   │                          │                                          │
   │                          ▼                                          │
   │   EventBridge rule (S3 PutObject prefix=labels/)                   │
   │                          │                                          │
   │                          ▼                                          │
   │   Step Functions state machine                                     │
   │     ├─ Lambda: validate file                                       │
   │     ├─ Glue: aggregate to feature parquet                          │
   │     ├─ SageMaker Processing: Great Expectations validation         │
   │     ├─ SageMaker Training Job: re-train XGBoost                    │
   │     ├─ SageMaker Processing: evaluate on holdout                   │
   │     ├─ Choice: gain > threshold? --> RegisterModel                 │
   │     └─ SNS notify Slack                                            │
   │                          │                                          │
   │                          ▼                                          │
   │   Manual approval gate (CodePipeline)                              │
   │                          │                                          │
   │                          ▼                                          │
   │   CodePipeline → CodeBuild → CodeDeploy                            │
   │     Canary 5% → 50% → 100% on SageMaker endpoint                   │
   │                                                                     │
   │   Side branch:                                                      │
   │     EventBridge rule (SageMaker job state = Failed)                │
   │     → Lambda: post incident in Slack                               │
   └────────────────────────────────────────────────────────────────────┘
```

### EventBridge Q&A

**Q1. EventBridge vs SQS?**
> EventBridge is for *routing events* — one event can fan out to many targets, with filtering rules. SQS is for *durable queues* — one consumer (or consumer group) processing messages with at-least-once delivery. They're complementary: EventBridge often delivers *into* SQS for reliable async processing. For ML, EventBridge for triggering pipelines on data arrival; SQS for backpressure on async inference.

**Q2. How do you trigger retraining on data drift?**
> SageMaker Model Monitor publishes CloudWatch metrics. CloudWatch Alarm fires on threshold breach, sends to SNS or directly to EventBridge via the alarm-state-change rule. EventBridge starts a Step Functions execution that runs the retraining DAG — preprocess, train, evaluate, register if better. If you want fully automatic retrain-and-deploy, the deploy stage uses CodePipeline with a manual approval gate for safety.

---

## 11.9 CodePipeline + CodeBuild — end-to-end MLOps CI/CD

### Why an MLOps CI/CD distinct from app CI/CD?

ML deployments have stages app deployments don't: model training, evaluation against benchmarks, registry approval, canary against shadow traffic. A standard GitHub Actions workflow that builds an image and pushes it isn't enough. CodePipeline + CodeBuild is the AWS-native answer; same pattern works in GitHub Actions or GitLab CI with AWS API calls.

### Reference pipeline

```
   ┌──────────────────── ML CI/CD Reference ────────────────────────┐
   │                                                                  │
   │   STAGE 1 — Source                                               │
   │     CodeCommit / GitHub webhook                                  │
   │                                                                  │
   │   STAGE 2 — Build                                                │
   │     CodeBuild: lint, unit tests, build container, push to ECR   │
   │                                                                  │
   │   STAGE 3 — Train                                                │
   │     CodeBuild: launch SageMaker Training Job, wait, fetch       │
   │     metrics, fail if eval-AUC < threshold                        │
   │                                                                  │
   │   STAGE 4 — Register                                             │
   │     Lambda: register in SageMaker Model Registry,                │
   │     ModelApprovalStatus=PendingManualApproval                    │
   │                                                                  │
   │   STAGE 5 — Manual Approval                                      │
   │     CodePipeline approval action (notify via SNS)                │
   │                                                                  │
   │   STAGE 6 — Deploy Staging                                       │
   │     CodeDeploy: blue-green to staging endpoint                   │
   │     Run smoke tests                                              │
   │                                                                  │
   │   STAGE 7 — Deploy Prod (Canary)                                 │
   │     CodeDeploy: 10% canary for 1 hour, watch alarms              │
   │     Auto-promote on success, auto-rollback on alarm              │
   │                                                                  │
   │   STAGE 8 — Notify                                               │
   │     Lambda: post deployment to Slack with model card link        │
   └──────────────────────────────────────────────────────────────────┘
```

### Worked example — promotion across environments

Same image, three deploys: dev → staging → prod. Each environment has its own SageMaker endpoint, Model Registry approval status, alarms. The ECR image digest stays the same — what changes is config (instance type, replica count, KMS key, env vars). This is **artifact promotion**, opposite of "rebuild for prod" which can drift.

### CodePipeline Q&A

**Q1. Why have a Manual Approval gate in your ML pipeline?**
> Two reasons. First, model behavior changes are riskier than code changes — a regression can degrade user experience for everyone. Second, regulated industries often require sign-off (an approver attests to fairness, bias, drift). The manual gate is cheap insurance. For experimental models, I drop the gate; for production models, I keep it but auto-approve in dev and staging environments.

**Q2. Difference between CodeBuild and CodePipeline?**
> CodeBuild is the *worker*: it runs a buildspec.yaml inside a Docker container, executes commands, produces artifacts. CodePipeline is the *orchestrator*: it sequences stages, manages approvals, watches outputs, triggers next stage. CodeBuild can run standalone; CodePipeline almost always invokes CodeBuild as a stage.

---

## 11.10 S3 patterns for ML

### Why S3 design matters

S3 is unlimited, durable, and cheap, but bad prefix design and bad request patterns can murder your training-job throughput or balloon your costs.

### Prefix design for parallel reads

S3 partitions hot prefixes automatically, but you can help by **avoiding sequential prefixes** that all hash to the same partition. Instead of `s3://bucket/2024-04-28/00001.parquet`, use `s3://bucket/<sha-prefix>/2024-04-28/00001.parquet` so reads spread across partitions.

### Multi-part upload

Files > 100 MB should always upload as multi-part. Lets you parallelize and resume on failure. Boto3's TransferManager does this automatically; just check the `multipart_threshold` config.

### Lifecycle to Glacier

Old training data and old model artifacts to S3 Glacier after 90 days. Restore takes hours but cuts storage cost ~80%. For Avrioc-scale, this is real money.

### S3 Express One Zone

For ultra-low-latency reads (<10 ms), **S3 Express One Zone** offers single-AZ buckets with consistent single-digit ms read latency. Use for hot training-data shuffle loops where standard S3's variable latency tanks GPU utilization.

### S3 Q&A

**Q1. Why does my training job's first epoch take 10× longer than the rest?**
> Cold S3 data + DataLoader prefetch ramping up. First reads pay full TCP/SSL handshake + S3 partition warming. Subsequent reads benefit from local OS page cache, S3 partition heat, and DataLoader prefetch buffer. Mitigations: warm-cache pass at job start, FastFile or Pipe mode in SageMaker, or move hot data to S3 Express One Zone.

**Q2. How do you prevent accidental data exfiltration via S3?**
> Bucket-level deny on PutObject from outside your VPC (using `aws:SourceVpce` condition). S3 Block Public Access on every prod bucket. KMS-CMK encryption with key policy scoped to your account. CloudTrail data events logged to a separate audit account. Macie or Recognition to scan for PII in buckets quarterly.

---

## 11.11 Azure — the parallel universe

Azure follows the same five primitives but with different names and slightly different defaults. Let's anchor with the diagram.

```
   ┌────────────────────── Azure AI/ML Reference ─────────────────────────┐
   │                                                                        │
   │   ┌── Data Layer ──────────────────────────────────────────────────┐  │
   │   │  ADLS Gen2 (lake)  Synapse  Data Explorer  Azure SQL/Cosmos    │  │
   │   │  Event Hubs (Kafka-compatible)  Service Bus  Stream Analytics  │  │
   │   └───────────────────────────────────────────────────────────────┘  │
   │                                  │                                     │
   │   ┌── Training Layer ────────────▼───────────────────────────────┐    │
   │   │  Azure ML Studio (managed compute, jobs, pipelines)          │    │
   │   │  Azure Databricks (Spark, MLflow, Delta, Unity Catalog)      │    │
   │   │  AKS + Ray Train (custom)                                    │    │
   │   └─────────────────────────────────────────────────────────────┘    │
   │                                  │                                     │
   │   ┌── Inference Layer ───────────▼───────────────────────────────┐    │
   │   │  Azure ML Managed Endpoints (online + batch)                 │    │
   │   │  Azure Functions (serverless, container support)             │    │
   │   │  AKS + KServe / vLLM                                         │    │
   │   │  Azure OpenAI Service (managed GPT, multimodal)              │    │
   │   └─────────────────────────────────────────────────────────────┘    │
   │                                  │                                     │
   │   ┌── Observability & Governance ▼────────────────────────────────┐   │
   │   │  Azure Monitor (metrics+logs+alerts)  App Insights (tracing)  │   │
   │   │  Microsoft Purview (catalog+lineage)  Defender (security)     │   │
   │   │  Azure AD / Managed Identity / Key Vault                      │   │
   │   └──────────────────────────────────────────────────────────────┘   │
   │                                                                        │
   │   Networking: VNet + Subnets + NSG + Private Endpoints + Front Door   │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 11.12 Azure Databricks — your Tiger Analytics / Mars story

### Why Databricks exists

Spark on bare clusters is painful — node provisioning, library management, multi-tenancy, notebook UX. Databricks is **Spark as a managed product** with notebook IDE, governed catalog, MLflow bundled, Delta Lake bundled, and SQL warehouse on top. It's available on all three clouds; on Azure it integrates deeply with AAD, Storage, and Synapse.

### Mental model

Databricks is **a fully-loaded Spark IDE with serverless options**. Think of it as: "I want to do Spark + ML + SQL + dashboards on top of object storage, with one identity, one catalog, one MLflow." That's Databricks.

### Core Databricks concepts

```
   ┌──────────────────── Databricks Surface Area ────────────────────┐
   │                                                                  │
   │   WORKSPACE — UI for notebooks, jobs, repos, dashboards         │
   │                                                                  │
   │   COMPUTE                                                        │
   │     All-purpose cluster (interactive notebooks)                 │
   │     Job cluster (one-shot, ephemeral, cheap)                    │
   │     Serverless SQL warehouse (Photon engine)                    │
   │     ML cluster (preinstalled scikit, torch, MLflow)             │
   │                                                                  │
   │   DATA                                                           │
   │     DBFS (filesystem abstraction over Blob/ADLS/S3)             │
   │     Unity Catalog (3-level namespace: catalog.schema.table)     │
   │     Delta Lake (ACID parquet with time travel)                  │
   │     Hive Metastore (legacy, replaced by Unity)                  │
   │                                                                  │
   │   ML                                                             │
   │     MLflow (experiments, models, registry)                      │
   │     Feature Store (online + offline)                            │
   │     Model Serving (managed endpoints)                           │
   │     Mosaic AI (vector search, fine-tuning)                      │
   │                                                                  │
   │   ORCHESTRATION                                                  │
   │     Workflows (Databricks-native DAGs)                          │
   │     Delta Live Tables (declarative pipelines)                   │
   │                                                                  │
   │   GOVERNANCE                                                     │
   │     Unity Catalog (RBAC, lineage, audit)                        │
   │     Personal Access Tokens / Service Principals                 │
   │     Cluster policies (cost guardrails)                          │
   └──────────────────────────────────────────────────────────────────┘
```

### Delta Lake — the killer feature

Delta Lake is **ACID transactions on Parquet**. Concretely: a `_delta_log` directory next to your parquet files holds JSON commits. Readers see a consistent snapshot. Writers don't conflict via optimistic concurrency control. You get:
- ACID writes (atomic upserts via MERGE)
- Time travel: `SELECT * FROM table VERSION AS OF 5`
- Schema enforcement and evolution
- OPTIMIZE + ZORDER for fast filtered reads
- Streaming source AND sink

### Worked example — Mars-style data quality on Databricks

The story you want to tell about Tiger Analytics' Mars project:

```
   ┌──────── Mars Daily Demand Forecast Pipeline ─────────────┐
   │                                                            │
   │   Source: hourly POS data → Event Hubs                    │
   │                          │                                  │
   │                          ▼                                  │
   │   Auto Loader (Databricks streaming)                       │
   │   → bronze.pos_raw (Delta)                                  │
   │                          │                                  │
   │                          ▼                                  │
   │   Silver job (every 6h)                                     │
   │     Deequ data-quality checks:                              │
   │       - completeness > 0.99 on store_id, sku                │
   │       - is_unique on (date, store, sku)                     │
   │       - inrange on units_sold (0, 10000)                    │
   │       - statistical: column_mean drift < 2σ vs prior week   │
   │     Writes silver.pos_clean (Delta)                         │
   │     On failure → Azure Monitor alert + Slack                │
   │                          │                                  │
   │                          ▼                                  │
   │   Gold job (daily 2am)                                     │
   │     Feature engineering, joins with weather, holidays      │
   │     Writes gold.demand_features                             │
   │                          │                                  │
   │                          ▼                                  │
   │   Forecast training (weekly)                               │
   │     LightGBM per store-cluster                              │
   │     MLflow logs experiment, registers model                 │
   │     Champion-challenger via mlflow.evaluate()               │
   │                          │                                  │
   │                          ▼                                  │
   │   Model Serving (Databricks-managed endpoint)              │
   │   OR exported to Azure ML for production                    │
   └────────────────────────────────────────────────────────────┘
```

### Common Databricks mistakes

1. **All-purpose clusters left running.** A 16-node ML cluster at $5/hr × 24hr × team-of-10 = $36k/month wasted. Auto-terminate after 30 min idle, default to job clusters.
2. **Hive Metastore in 2026.** Migrate to Unity Catalog. Hive is DBR < 11; Unity gives you fine-grained access, lineage, and cross-workspace federation.
3. **Stale Delta tables not OPTIMIZE'd.** Tons of small files = slow reads. Run OPTIMIZE weekly with ZORDER on filter columns.

### Databricks Q&A

**Q1. Walk me through Unity Catalog vs Hive Metastore.**
> Hive Metastore is per-workspace, lacks fine-grained governance, and was the default in older Databricks. Unity Catalog is account-level — one catalog spans many workspaces. Three-level namespace: catalog.schema.table. Centralized RBAC at the column level, automatic lineage tracking from notebook to dashboard, audit logs, and federation across clouds. Migrating is non-trivial — you redefine table locations and reapply permissions — but for any 2026 Databricks rollout, Unity is the only sane choice.

**Q2. Delta Lake vs Iceberg vs Hudi?**
> All three give ACID on object-store parquet. Delta is Databricks-native, deepest integration, best on Databricks. Iceberg is more open (Snowflake, Trino, Spark, Flink all read), better for multi-engine architectures. Hudi has the strongest streaming-update story (record-level upserts via merge-on-read). At Tiger I used Delta because it was Databricks-native. If I were architecting from scratch with multi-engine reads, I'd default to Iceberg.

**Q3. How do you do MLflow on Databricks?**
> MLflow is bundled. Inside a notebook, `mlflow.start_run()` auto-tags experiment with notebook ID, user, cluster. `mlflow.log_param/metric/artifact` populate the run. `mlflow.<flavor>.log_model()` registers in the workspace registry. Promote via the UI or API to Staging/Production. Databricks Model Serving picks up registered models for managed endpoints. Lineage flows into Unity Catalog automatically.

**Q4. When do you choose Job cluster over All-purpose cluster?**
> Always for production. Job clusters are ephemeral — provisioned per run, terminated on completion, you pay only for the runtime. All-purpose clusters are interactive — for development, idle time is the killer cost. My rule: every notebook in source control runs on a job cluster via Workflows. Interactive work in PR review uses a personal all-purpose cluster with auto-terminate at 30 minutes.

---

## 11.13 Azure Data Factory — managed ETL orchestration

### Why ADF vs Airflow?

Both orchestrate. ADF is **Azure-native, code-light, click-to-build** — drag-and-drop pipelines with built-in connectors (200+) for SaaS, databases, files. Airflow is **code-first, open-source, multi-cloud, more flexible** — you write Python DAGs, you control everything.

For the Tiger Analytics Mars story: ADF was the orchestrator gluing Salesforce, S3, Azure SQL, Databricks, and the dashboard refresh. The team had limited Python depth — ADF's GUI fit the team. For a Python-deep team building a custom platform, Airflow would have won.

### Mental model

ADF is **SSIS in the cloud** for the Microsoft generation, plus IPaaS connectors. It pairs well with Databricks (a "Databricks Notebook" activity is first-class).

### Worked example — ADF pipeline activity sequence

```
   ┌───────────── ADF Pipeline: Daily Mars Refresh ─────────────┐
   │                                                              │
   │   1. Lookup activity: get yesterday's date partition         │
   │   2. Copy activity: SFTP source → ADLS Gen2 raw zone        │
   │   3. Validation activity: row count > threshold              │
   │   4. Databricks Notebook activity: silver-to-gold transform │
   │   5. Stored Procedure activity: refresh data warehouse      │
   │   6. Web activity: trigger PowerBI dataset refresh           │
   │   7. If activity: failure → email via Logic Apps            │
   └──────────────────────────────────────────────────────────────┘
```

Trigger: Schedule (daily 2am) or Tumbling Window (per-hour with watermark) or Event (storage blob created).

### ADF Q&A

**Q1. ADF mapping data flows vs Databricks?**
> Mapping Data Flows are ADF-native Spark transforms via GUI. Databricks is a separate compute. For light transforms in a non-Spark-shop, mapping data flows are convenient — no separate cluster needed. For heavy transforms or where the team already lives in Databricks, invoking Databricks notebooks from ADF is the cleaner pattern. At Tiger we did Databricks; in a Microsoft-shop with no Databricks license, mapping data flows are reasonable.

---

## 11.14 Azure ML Studio

### What it is

Azure's parallel to SageMaker. Workspace (the container), Compute (clusters, instances, attached AKS), Datasets, Experiments, Models, Endpoints, Pipelines.

### Endpoints — managed online and batch

```
   Managed Online Endpoint
     ── always-on, sub-second SLA
     ── traffic split via Deployments (blue/green)
     ── auto-scaling
     ── Equivalent to SageMaker Real-time

   Batch Endpoint
     ── async batch scoring on a dataset
     ── Equivalent to SageMaker Batch Transform

   Kubernetes Online Endpoint
     ── deploy on attached AKS for full control
     ── Equivalent to "BYO K8s on AWS"
```

### Compute targets

- **Compute Instance** — single VM for dev (notebooks).
- **Compute Cluster** — autoscaling pool of VMs for training jobs.
- **Attached compute** — your existing AKS cluster, Databricks workspace, or HDInsight, registered for AzureML to dispatch jobs to.

### MLOps in Azure ML

```
   GitHub Actions / Azure DevOps
        │
        ▼
   Build + push to ACR (Azure Container Registry)
        │
        ▼
   az ml job create --file train.yaml  ── runs training
        │
        ▼
   az ml model register                  ── registers in workspace
        │
        ▼
   az ml online-deployment create        ── deploys to endpoint
        │
        ▼
   Traffic shift via az ml online-endpoint update --traffic
```

The whole pipeline is YAML + CLI v2, GitOps-friendly.

### Azure ML Q&A

**Q1. Azure ML vs Databricks for training — when each?**
> Azure ML for tabular training, scikit/XGBoost/PyTorch, MLOps lifecycle (registry, deployments, monitoring). Databricks for big-data preprocessing on Spark, especially when data sits in Delta. Common hybrid: Databricks does feature engineering on Delta Lake; Azure ML pulls features as a Dataset and runs training. Both have MLflow for tracking — pick one as the source of truth.

---

## 11.15 Azure Storage tiers and ADLS Gen2 vs Blob

### The tiers

```
   Hot      — active data, $0.018/GB/month, lowest read cost
   Cool     — infrequent (>30d), $0.01/GB/month, higher read cost
   Cold     — rare (>90d), $0.0036/GB/month, even higher read cost
   Archive  — backup (>180d), $0.002/GB/month, hours to rehydrate
```

ML pattern: hot for current training data and serving artifacts; cool for older logs and historical data still queryable; archive for compliance backups.

### ADLS Gen2 vs Blob

Same underlying Storage Account, but **ADLS Gen2 adds hierarchical namespace** — true directory semantics, POSIX-like ACLs on directories, and renames are O(1) instead of O(n). For analytics (Spark, Databricks, Synapse) ADLS Gen2 is required. For object storage (web assets, backups) plain Blob is fine. Default for any new ML workspace: ADLS Gen2.

---

## 11.16 Azure AD vs IAM — identity comparison

### The mental model

Both AWS IAM and Azure Active Directory (AAD, now "Microsoft Entra ID") are identity systems. They solve the same problem with different vocabulary.

```
   AWS IAM                         Azure AD / RBAC
   ────────                        ───────────────
   IAM User                        AAD User
   IAM Group                       AAD Group
   IAM Role                        Service Principal / Managed Identity
   IAM Policy                      RBAC Role + Role Assignment
   STS AssumeRole                  AAD Token via OAuth
   Resource policies               RBAC scope (subscription/RG/resource)
   Permission boundary             Conditional Access + custom roles
   Federation                      AAD federation with external IdP
```

### Managed Identity — Azure's killer feature

Instead of storing credentials, an Azure resource (VM, Function, Container) is assigned a **Managed Identity** — an AAD identity AAD trusts. Code uses `DefaultAzureCredential()`; the SDK fetches tokens from the Instance Metadata Service. Equivalent to AWS IAM Roles for EC2 / Lambda. The pattern eliminates secrets from your code.

### Azure AD Q&A

**Q1. How do you handle cross-cloud identity?**
> Federation. AAD can federate with AWS via SAML — users sign in via AAD, get an STS token. AWS Cognito can federate with AAD for app users. For service-to-service, OIDC: AAD issues tokens that AWS IAM can trust via OIDC provider configuration; reverse via AWS STS issuing tokens AAD trusts. The result: one identity (the user or service) with claims that both clouds honor. The setup is a project in itself; for Avrioc-scale I'd put a gateway like Auth0 in front and federate to both clouds.

---

## 11.17 AWS vs Azure — feature comparison for ML

| Capability | AWS | Azure |
|------------|-----|-------|
| Managed training | SageMaker Training Jobs | Azure ML Jobs |
| Managed inference (real-time) | SageMaker Endpoint | Azure ML Online Endpoint |
| Managed inference (batch) | Batch Transform | Azure ML Batch Endpoint |
| Foundation-model API | Bedrock | Azure OpenAI |
| Notebook IDE | SageMaker Studio | Azure ML Studio |
| Managed Spark / lakehouse | Glue, EMR | Databricks (3rd party but deep) |
| Object storage | S3 | Blob / ADLS Gen2 |
| Block storage | EBS | Managed Disks |
| Shared filesystem | EFS / FSx | Azure Files / NetApp |
| Container registry | ECR | ACR |
| Kubernetes | EKS | AKS |
| Serverless function | Lambda | Functions |
| Event bus / orchestration | EventBridge / Step Functions | Event Grid / Logic Apps / ADF |
| CI/CD | CodePipeline / CodeBuild | Azure DevOps / GitHub Actions |
| Identity | IAM | AAD / Entra ID |
| Secrets | Secrets Manager | Key Vault |
| Observability | CloudWatch / X-Ray | Azure Monitor / App Insights |
| Data catalog | Glue / Lake Formation | Purview |
| Network | VPC / SG / VPC Endpoints | VNet / NSG / Private Endpoints |

---

## 11.18 Multi-cloud patterns and gotchas

### When multi-cloud is real

- **Regional data residency** — UAE customers on Azure UAE-North + AWS me-central-1.
- **Vendor cost arbitrage** — Spot pricing varies; some workloads cheaper on the other cloud.
- **Compliance separation** — keep regulated data in one cloud, public-facing in another.
- **M&A reality** — you bought a company; their stack is on the other cloud and migration isn't worth it.

### Patterns

- **LLM gateway abstraction**: LiteLLM or custom gateway routes to Bedrock or Azure OpenAI based on cost/latency/availability.
- **Storage gateway**: present S3 API; backed by either S3 or Azure Blob via translation.
- **Terraform multi-provider**: same module structure across providers; abstracted resources where possible.
- **Identity federation**: as above — AAD or Okta as the primary IdP; both clouds federate.

### Gotchas

1. **Egress costs.** Cross-cloud bandwidth is expensive. Don't put hot inference in one cloud and the model artifact in the other.
2. **DNS coherence.** Make sure both clouds use the same DNS strategy; a private DNS zone in one cloud doesn't auto-resolve in the other.
3. **Monitoring fragmentation.** CloudWatch and Azure Monitor don't talk. Use Datadog, New Relic, or Grafana with both as datasources.
4. **Skills tax.** Engineers who know both well are rare; budget for it.

---

## 11.19 Resume tie-in narratives

> **How to say the TrueBalance story in an interview:**
> "At TrueBalance I built a real-time withdraw-prediction service on AWS Lambda hitting p99 under 500 milliseconds. Container image with XGBoost, deployed in three VPC-isolated environments via a Terraform module. Lambda fetched real-time features from Redis ElastiCache and historical aggregates from Snowflake over a private link. Provisioned concurrency kept five warm instances. CloudWatch custom metrics gave us per-model p99 visibility, and a SageMaker Model Monitor job ran daily on the data-capture S3 prefix to catch drift. The whole pipeline — training in SageMaker, registration in Model Registry, deploy via CodePipeline with canary — was triggered by EventBridge rules on new labeled data."

> **How to say the ResMed story in an interview:**
> "At ResMed I shipped eight models to production in six months on AWS. The pattern was SageMaker Training Jobs for training, Model Registry for versioning, Multi-Container Endpoints for the clinical pipeline that needed preprocess in TensorFlow + main model in PyTorch + postprocess in sklearn. Multi-Model Endpoints for tenant-specific recommendations. EventBridge triggered retraining on new labeled data; Step Functions orchestrated the DAG; CodePipeline deployed with manual approval and canary."

> **How to say the Tiger Analytics Mars story:**
> "At Tiger Analytics on the Mars project I built the data quality and forecasting layer on Azure Databricks. Auto Loader ingested POS data into bronze Delta tables. A scheduled silver job ran Deequ checks — completeness, uniqueness, statistical drift on column means — and alerted Slack on failure. Gold tables held demand features, and a weekly LightGBM training job logged to MLflow with champion-challenger evaluation. Azure Data Factory orchestrated the daily refresh that fed PowerBI dashboards."

---

## 11.20 Comprehensive Q&A — Cloud Mastery

**Q1. Walk me through the AWS ML stack you'd recommend for Avrioc.**
> Primary region me-central-1 in Bahrain for data residency. EKS cluster with GPU node groups (g5 for embedding, p5 for LLM serving) plus a CPU pool for the gateway and orchestration. SageMaker for model training and the model registry; we deploy from there to either SageMaker endpoints for tabular models or to vLLM-on-EKS for LLMs. Bedrock as the managed LLM API for low-volume use cases. S3 for the data lake with Lake Formation governance, RDS Aurora for transactional, Redshift Serverless for analytics. EventBridge + Step Functions for orchestration. CodePipeline for CI/CD with manual approval to prod. Datadog or Grafana on top of CloudWatch for unified observability. The whole thing in three VPCs (dev/staging/prod) with VPC endpoints to avoid NAT costs.

**Q2. Lambda vs SageMaker — break-even rule of thumb?**
> Below one sustained request per second, Lambda is cheaper. Above ten, SageMaker Real-time. Between, it depends on whether you can use SageMaker Serverless (CPU-only, <6 GB) or need GPU. For my TrueBalance XGBoost case, sustained QPS was below one, so Lambda was the right call. If we'd grown to 100 QPS, the migration target would have been SageMaker Real-time on `ml.c6i.large` with auto-scaling.

**Q3. SageMaker endpoint variants — tell me about MME.**
> Multi-Model Endpoints serve many homogeneous models from one container. Models live in S3; SageMaker loads on-demand and caches in instance memory. Cache eviction is LRU; cold model takes 1-10 seconds to load. Use case: tenant-specific small models — fraud rules per merchant, recommendation models per region. Trade-off: all models share the same container so they must use the same framework version. For mixed frameworks I'd use Multi-Container Endpoints instead.

**Q4. VPC isolation gotchas for SageMaker?**
> Set NetworkIsolation=True on the training job to prevent any internet egress. Then add VPC endpoints for everything you need: S3 Gateway endpoint for model artifacts and data, ECR Interface endpoints (api and dkr) for image pulls, CloudWatch Logs for logging, Secrets Manager for credentials, and STS for role assumption. Missing the S3 Gateway is the number one mysterious-hang issue. Also: security group must allow self-referencing for distributed training NCCL traffic.

**Q5. ECR best practices for ML?**
> Immutable tags pinned to git SHA. Lifecycle policy to delete untagged after 7 days and keep the last 10 tagged versions per repo — ML images are 5-15 GB so this matters fast. Enhanced Scanning with Inspector for continuous CVE detection. Pull-through cache for nvcr.io and Docker Hub to avoid pulling huge base images repeatedly. Cross-region replication only for prod images that need it. IAM scoped per environment, not a shared root role.

**Q6. Tell me about your TrueBalance Lambda VPC story.**
> Three-environment VPC isolation with non-overlapping CIDRs — 10.10/16 dev, 10.20/16 staging, 10.30/16 prod. Each environment has its own ECR repo, KMS key, Secrets Manager secrets, and IAM execution role. Terraform module accepts an env variable and is instantiated three times via three workspaces. The Lambda lives in private subnets across two AZs in each VPC. VPC endpoints for S3, ECR, CloudWatch Logs, Secrets Manager. No Transit Gateway between environments — a bug in dev cannot reach prod data. Snowflake access is via PrivateLink to a per-environment Snowflake account.

**Q7. [Gotcha] SageMaker training job hangs at "Downloading input data" forever. Diagnose.**
> Almost always missing S3 VPC Gateway endpoint when the training job runs in VPC isolation mode. The container starts, tries to fetch data from S3, hits a non-routable IP because no NAT and no endpoint, and waits. CloudWatch Logs will show the wait. Fix: add the S3 Gateway endpoint to the route tables of the subnets the training job runs in. Secondary suspects: missing ECR endpoint (can't pull image), missing STS endpoint (can't assume the role).

**Q8. Bedrock vs self-hosting on EKS — when each?**
> Bedrock for: low-volume use cases, need for Claude or Nova specifically, no infrastructure team capacity, fast time-to-market. Per-token pricing means costs scale linearly with usage. Self-host vLLM on EKS for: sustained high throughput where per-token pricing exceeds instance pricing — usually around 1M tokens/hour. Open-source models you've fine-tuned. Multi-tenant LoRA where you need PagedAttention. Strict data residency where Bedrock isn't available regionally. The break-even shifts as Bedrock adds volume discounts; check current pricing.

**Q9. Provisioned concurrency for Lambda — when worth it?**
> When the cold-start penalty hurts your SLA. For my TrueBalance XGBoost case, p99 budget was 500 ms; cold start could be 800 ms, blowing it. Provisioned concurrency keeps N instances warm — pay for warm hours but zero cold start. Tune via target tracking on `ProvisionedConcurrencyUtilization`. If utilization stays below 70%, you're over-provisioned. For occasional traffic where SLA is loose, regular Lambda + SnapStart is cheaper.

**Q10. EventBridge schedule vs Lambda Schedule vs CloudWatch Events?**
> EventBridge IS the new CloudWatch Events — same service, rebranded. Lambda Schedule is a convenience that creates an EventBridge rule under the hood. Always use EventBridge directly: it gives you Schedule expressions, event pattern matching, multiple targets per rule, and dead-letter queues. For ML retraining triggered by S3 PutObject, EventBridge is the answer.

**Q11. CodePipeline manual approval — why have it?**
> Two reasons. Model changes are riskier than code — a regression hurts every user. Regulated industries require sign-off. The manual gate is cheap insurance. I auto-approve in dev and staging, manual approve in prod. The approver gets a Slack message with the model card link, evaluation metrics, drift report, and an Approve/Reject button.

**Q12. Azure Databricks core concepts — explain to me as if I'm a junior.**
> The workspace is your IDE, like a giant Jupyter notebook environment with collaboration. Behind it sits compute — clusters that run Spark, ephemeral or always-on, your choice. Storage is Delta Lake on top of cloud blob — Parquet files plus a JSON transaction log that gives you ACID, time travel, and schema evolution. Unity Catalog is the governance layer — three-level namespace, RBAC at column level, lineage tracking. MLflow is bundled for experiment tracking and registry. Workflows orchestrate jobs into DAGs. The pitch is: one place for data + ML + SQL with one identity and one catalog.

**Q13. What's Unity Catalog and why does it matter?**
> Unity Catalog is account-level governance for Databricks — replaces the per-workspace Hive Metastore. Catalog → Schema → Table is the three-level namespace. RBAC is fine-grained, including column-level masking. Lineage is automatic from notebook to dashboard. Cross-workspace federation lets a single catalog serve many teams. Audit logs go to a system table. For any 2026 Databricks deployment, Unity is non-negotiable; the migration from Hive is a project but worth it.

**Q14. Azure ML Online Endpoint — how does the deployment object work?**
> An Endpoint is the URL; Deployments are the model versions behind it. You create a deployment with a model, environment, scoring script, instance type, replica count. Multiple deployments can sit behind one endpoint. Traffic split is managed via `az ml online-endpoint update --traffic blue=90,green=10`. To do blue-green: create green at 0%, validate, shift to 100%, retire blue. Equivalent to SageMaker ProductionVariants.

**Q15. Azure OpenAI vs Bedrock — which would you choose?**
> Depends on the model lineup and the org. Azure OpenAI gives you GPT-4o, GPT-4 Turbo, embedding models — the OpenAI catalog hosted in Azure with enterprise compliance. Bedrock gives you Claude, Nova, Llama, Mistral, Cohere — broader open and proprietary lineup. For a UAE customer wanting Claude or Llama, Bedrock me-central-1 is the answer. For a Microsoft-shop wanting GPT-4o, Azure OpenAI in UAE-North. Most enterprises run both via a gateway like LiteLLM that routes by model family.

**Q16. [Gotcha] Lambda cold start is 4 seconds, breaking your SLA. Options in order?**
> First, profile where the time is going via INIT_REPORT in CloudWatch Logs. If it's runtime init, switch to Python 3.12 + SnapStart for near-zero. If it's container pull, slim the image — multi-stage build, prune dev deps, smaller base. If it's model load at module init, lazy-load and cache after first invocation. If none of those get you there, provisioned concurrency. As a last resort, switch to SageMaker Real-time with min instance count.

**Q17. Spot instances for ML training — when and how?**
> 50-90% discount, but interrupted with two-minute warning. Always for fault-tolerant batch training where you checkpoint every 15-30 minutes. SageMaker Managed Spot Training handles interruption and resume automatically — you just set `use_spot_instances=True` and `max_wait` greater than `max_run`. Never for real-time inference — the interruption tail risk is unacceptable. For training a 70B model, spot saves five-figures per run; checkpoint interval is the key knob.

**Q18. Multi-region ML deployment — pattern?**
> Source of truth: Model Registry in one region (primary). Artifacts replicated cross-region via S3 Cross-Region Replication. Endpoints replicated in each region with the same EndpointConfig and model. Route53 latency-based routing or active-active failover. CI/CD deploys to all regions in parallel; rollback rolls back all. CloudWatch in each region; cross-region dashboard via cross-account or cross-region metrics. For Avrioc me-central-1 + UAE-North dual-cloud, the pattern is similar but with translation between cloud-native services.

**Q19. KMS strategy for ML?**
> Customer-managed CMKs per environment per data class. So: prod-pii, prod-public, staging-pii, staging-public, dev-shared. Key policy scopes per service principal. Rotation enabled, automatic 1-year. SageMaker encrypts volume, EBS, S3 outputs, Notebook with the matching CMK. KMS data events to CloudTrail to a separate audit account. The cost is real — $1/key/month plus per-API-call — so don't go crazier than necessary.

**Q20. ADLS Gen2 vs S3 — feature parity?**
> Both are object storage with strong consistency. ADLS Gen2 adds true hierarchical namespace — directories are real, renames are O(1), POSIX-like ACLs on files and directories. S3 has flat namespace; "directories" are prefixes. For analytics workloads (Spark, Databricks, Synapse), ADLS Gen2 is more efficient and required by some integrations. For object storage (web assets, backups), S3 is simpler. From a feature-set standpoint they're comparable; the default for any Azure ML workspace is ADLS Gen2.

**Q21. Avrioc UAE-specific recommendation?**
> Primary cloud AWS me-central-1 (Bahrain) for data residency and Bedrock with Claude. Secondary cloud Azure UAE-North if Microsoft-shop integration matters. EKS for compute heavy lifting; SageMaker for training and tabular endpoints; vLLM on EKS for self-hosted LLMs; Bedrock for managed Claude API. Datadog for observability across both clouds. Auth via AAD or Okta as the IdP, federated to both. CloudFront in front for global delivery. Lambda for lightweight integrations. The whole thing in dev/staging/prod three-VPC topology with a shared transit gateway only for shared services like DNS and observability.

**Q22. Cost monitoring for ML — how do you set it up?**
> AWS Cost Explorer plus Budgets on tag-based slices: `service`, `env`, `model`, `tenant`. CloudWatch Logs Insights to correlate cost spikes with Lambda invocations or SageMaker job durations. For LLMs, token-tracking via the LiteLLM gateway with Prometheus metrics — cost per tenant, per model, per day. Anomaly detection alarms on unexpected cost spikes. Monthly review meeting with finance to flag oversized resources. At ResMed we cut costs 30% by killing idle SageMaker endpoints and migrating dev to Spot.

**Q23. Terraform vs CloudFormation vs CDK?**
> Terraform: multi-cloud, mature ecosystem, separate state management. CloudFormation: AWS-only, no external state, native rollback. CDK: programmatic — you write TypeScript or Python that synthesizes CloudFormation. For multi-cloud (AWS + Azure), Terraform wins. For AWS-only with a TypeScript team, CDK is more ergonomic. For AWS-only with simple needs, CloudFormation is fine. My resume lists Terraform — that's my default for portability.

**Q24. [Gotcha] Your training job runs fine in dev but fails in prod with permission errors.**
> Three suspects. First, IAM role differences — dev role might be over-permissive, prod might be missing a specific permission. Compare roles side by side; the diff is usually obvious. Second, KMS key policies — prod uses a different CMK and the SageMaker execution role isn't in the key policy. Third, S3 bucket policies — prod buckets often have explicit deny conditions (like `aws:SourceVpce`) that dev doesn't. Fix: replicate the IAM, KMS, and bucket policies from a known-good environment via Terraform.

**Q25. How do you migrate from Azure to AWS or vice versa?**
> Phased lift-and-shift first, then gradually rearchitect. Phase one: identify boundary — usually around storage and identity. Phase two: replicate storage (S3 to Blob via DataSync, or vice versa with AzCopy). Phase three: rebuild identity layer — federate via Okta or maintain dual IdP with mappings. Phase four: migrate compute services one by one — Lambda to Functions, SageMaker to Azure ML, etc. Phase five: cutover — DNS shift, monitoring, decommission. Total project for a real ML stack: 6-18 months. The hardest part is data egress costs; budget for it.

---

## 11.21 Resume tie-in summary

- **TrueBalance XGBoost Lambda (p99 < 500 ms, 3-env VPC):** the gold-standard AWS narrative. Be ready to whiteboard VPC layout, Terraform module structure, CloudWatch custom metrics, provisioned concurrency strategy, container optimization steps. This is your strongest AWS story.
- **TrueBalance Claude-powered ML workspace (LLM + Jira + GitHub + Athena + Jenkins):** frame as "I architected an MCP-style integration with Claude as the orchestrator and tools for each external system. Bedrock for LLM, Lambda for tool execution, EventBridge for scheduled syncs, Athena for ad-hoc queries against the data lake."
- **ResMed 8 models in 6 months:** SageMaker Pipelines + Multi-Container Endpoints + Model Registry + EventBridge + CodePipeline. The "MLOps platform" story.
- **Tiger Analytics Mars on Azure Databricks:** Auto Loader → bronze/silver/gold Delta with Deequ checks → MLflow champion-challenger → ADF orchestration → PowerBI. The Azure-native story.

---

Continue to **[Chapter 12 — Kubernetes, Ray, Docker](12_kubernetes_ray.md)**.
