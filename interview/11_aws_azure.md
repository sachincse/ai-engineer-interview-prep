# Chapter 11 — AWS & Azure for AI
## Cloud skills for a polyglot AI engineer

> JD: "Working across AWS/Azure to deploy and scale AI solutions." Your resume is AWS-heavy (SageMaker, Lambda, VPC, ECR) with solid Azure (Databricks, DataFactory, ML Studio). This chapter depth-checks both.

---

## 11.1 AWS — the ML-relevant services

```
┌─────────────────────── AWS AI/ML stack ─────────────────────────┐
│                                                                   │
│  Managed AI services:                                             │
│    Bedrock    (foundation models API)                             │
│    Comprehend (NLP)                                               │
│    Textract   (OCR)                                               │
│    Rekognition (vision)                                           │
│                                                                   │
│  SageMaker suite:                                                 │
│    Studio            (notebook IDE)                               │
│    Training Jobs     (managed training)                           │
│    Endpoints         (real-time, async, serverless, batch)        │
│    Feature Store                                                  │
│    Model Registry                                                 │
│    Pipelines         (CICD for ML)                                │
│    Model Monitor     (drift detection)                            │
│    JumpStart         (pre-trained models)                         │
│                                                                   │
│  Core infra:                                                      │
│    Lambda     — serverless compute, container images              │
│    ECR        — container registry                                │
│    EC2 / ECS  — compute (GPU instances: g4dn, g5, g6, p5)         │
│    EFS / EBS / S3 — storage                                       │
│    VPC / SG / IAM — networking & access                           │
│    CloudWatch — metrics, logs, alarms                             │
│    CodeBuild / CodePipeline / CodeDeploy — CICD                   │
│    EventBridge — event-driven triggers                            │
│    Athena / Glue — serverless analytics                           │
└───────────────────────────────────────────────────────────────────┘
```

---

## 11.2 SageMaker Endpoints — the four modes

### 1. Real-time endpoint
- Always-on, sub-second latency
- Pay for instance hours
- Auto-scaling by invocations-per-instance or target tracking
- **Use:** chatbots, user-facing inference, strict SLAs
- Instance families: ml.g4dn (T4), ml.g5 (A10G), ml.g6 (L4), ml.p5 (H100)

### 2. Async endpoint
- Up to 60-minute inference
- SQS-backed queue, scale-to-zero
- **Use:** large-doc processing, batch-ish workloads without pegging compute

### 3. Serverless endpoint
- Cold start 5-30s
- Pay per request
- Max 6 GB memory (CPU-only by default)
- **Use:** sporadic traffic, internal tools

### 4. Batch Transform
- No endpoint — batch-process S3 objects
- **Use:** nightly scoring, bulk featurization

### Advanced endpoint types
- **Multi-Model Endpoint (MME)** — load many homogeneous models on demand from S3; one container
- **Multi-Container Endpoint (MCE)** — up to 15 heterogeneous containers; invokable by target; used for A/B, ensembles, chains
- **Inference Pipeline** — linear chain of containers (preprocess → model → postprocess)

### Decision matrix

| Need | Pick |
|------|------|
| <100ms latency, steady traffic | Real-time |
| <1 req/sec, acceptable cold start | Serverless |
| Long-running inference (up to 60min) | Async |
| Offline nightly scoring | Batch Transform |
| Many tenant-specific models (thousands) | Multi-Model (MME) |
| A/B testing, ensembles, chained preprocess | Multi-Container (MCE) |

---

## 11.3 Lambda for ML

- Container images up to 10 GB
- 10 GB RAM, 15-min max
- **Use:** small models (<1 GB), spiky traffic, S3-triggered pipelines
- **Bad for:** GPU workloads (Lambda has no GPU), models >2 GB (cold start suffers)

### Your TrueBalance story
XGBoost with p99 <500 ms on Lambda — this is exactly the sweet spot. Talk about:
- Container image optimization (base image choice, layer caching)
- Feature fetch from Redis/Snowflake (talk about connection pooling)
- CloudWatch custom metrics for p99 tracking
- Provisioned concurrency if cold starts hurt
- VPC configuration (private subnets, VPC endpoints for S3 + Secrets Manager)

### Lambda vs SageMaker break-even
- <1 req/sec sustained → Lambda is cheaper
- Beyond that → SageMaker serverless or real-time wins

---

## 11.4 Bedrock — AWS's LLM API

- Foundation models: **Claude** (Anthropic), Nova (Amazon), Llama (Meta), Mistral, Cohere, AI21
- **Available in me-central-1 (Bahrain)** — useful for UAE data residency
- Private VPC endpoints
- Guardrails (AWS-native)
- Knowledge Bases (managed RAG)
- Agents (tool-calling orchestration)
- Batch inference (50% cheaper)

### Bedrock vs self-host on SageMaker
| Criterion | Bedrock | Self-host on SageMaker |
|-----------|---------|-----------------------|
| Ops | Zero | Full |
| Pricing | Per-token | Per-instance-hour |
| Quality | Claude Opus available | Open models only |
| Data residency | me-central-1 or regional | Any AWS region |
| Break-even | Low-medium volume | ~1M tokens/hour+ |

---

## 11.5 VPC isolation for ML workloads

### Why it matters
- Regulated data (medical, financial, UAE PDPL)
- Zero-egress requirements
- Multi-tenant SaaS separation

### Core pattern
```
  ┌─ Private subnet (AZ-1) ─┐    ┌─ Private subnet (AZ-2) ─┐
  │                         │    │                         │
  │  Lambda (ENI here)      │    │  Lambda replica         │
  │                         │    │                         │
  │  SageMaker endpoint     │    │  SageMaker endpoint     │
  │                         │    │                         │
  └─────────────┬───────────┘    └───────────┬─────────────┘
                │                            │
                └─────────────┬──────────────┘
                              │
                     ┌────────▼────────────┐
                     │  VPC Endpoints       │
                     │  - S3 (Gateway)      │
                     │  - ECR (Interface)   │
                     │  - Secrets Manager   │
                     │  - CloudWatch Logs   │
                     │  - Bedrock (optional)│
                     └──────────────────────┘
```

### Must-have VPC endpoints for ML
- **S3 (Gateway)** — model artifacts, data. Missing this = stuck jobs.
- **ECR (Interface)** — container images
- **Secrets Manager (Interface)** — API keys, DB creds
- **CloudWatch Logs (Interface)** — logging
- **Bedrock (Interface)** — if using LLM API from private subnet

### VPC gotchas
- **Missing S3 endpoint** = training jobs hang (#1 cause of mysterious failures)
- **NAT Gateway costs** — can spike with ML download traffic
- **DNS resolution** — VPC endpoints need DNS on in the VPC
- **KMS keys** — scope to specific services/roles
- **Cross-account** — use PrivateLink

### Your 3-env VPC isolation story
At TrueBalance you had 3-env VPC isolation for the XGBoost Lambda. Be ready to draw:
- 3 VPCs (dev/staging/prod) with non-overlapping CIDRs
- Transit gateway or peering only where needed (probably none between envs)
- Terraform modules for repeatable provisioning
- Per-env KMS CMKs, Secrets Manager secrets, ECR repos

---

## 11.6 ECR — container registry for ML

### Key practices
- **Immutable tags** — don't overwrite tags; use git SHA
- **Lifecycle policies** — delete untagged images (ML images are 5-15 GB)
- **Image scanning** — enable Enhanced Scanning (Inspector) for CVEs
- **Pull-through cache** — cache public images (HuggingFace, PyTorch) for speed + availability
- **Cross-region replication** — if multi-region deploys
- **IAM-scoped access** — roles per env, not a shared root

---

## 11.7 EventBridge + CodePipeline for MLOps

The ResMed pattern:
```
Data arrives S3 → EventBridge rule → Step Functions
                                        │
                                        ├─▶ Data validation (Great Expectations)
                                        ├─▶ Training job (SageMaker)
                                        ├─▶ Eval job
                                        ├─▶ Register model (MLflow / SageMaker Registry)
                                        └─▶ Notify (SNS)

Manual approval
                                        │
CodePipeline → CodeBuild → CodeDeploy → SageMaker endpoint update
                                        │
                                        └─▶ Canary → full rollout
```

---

## 11.8 AWS Lambda deep optimizations

### Cold start reduction
- **Container image** optimizations: slim base, `--platform linux/amd64`, multi-stage build
- **Provisioned concurrency** — pre-warmed instances for zero cold start
- **SnapStart** (Java/Python) — pre-initialized snapshots
- **Small deployment package** — <50 MB zipped
- **Avoid large imports at cold start** — lazy-load

### Latency optimizations
- Connection pooling (to DB, downstream services) using global variables
- Caching (in-memory) for feature lookups or config
- Async I/O via asyncio
- Streaming response via Lambda Function URLs (for LLM streaming)

---

## 11.9 CloudWatch — metrics, logs, alarms

### Custom metrics (the key for your p99 Lambda story)
```python
cloudwatch.put_metric_data(
    Namespace='MLModels',
    MetricData=[{
        'MetricName': 'InferenceLatencyMs',
        'Value': latency_ms,
        'Unit': 'Milliseconds',
        'Dimensions': [
            {'Name': 'ModelName', 'Value': 'withdraw_predictor'},
            {'Name': 'Env', 'Value': 'prod'},
        ]
    }]
)
```

### Alarms
- p99 latency > SLO
- Error rate > threshold
- Invocations dropped to 0 (upstream issue)
- Drift metric out of bounds

---

## 11.10 Azure — the ML-relevant services

```
┌──────────────── Azure AI/ML stack ─────────────────┐
│                                                     │
│  Azure AI Foundry / Azure OpenAI Service           │
│    - GPT, Claude, Llama, Mistral via API           │
│                                                     │
│  Azure Machine Learning (AML)                       │
│    - Workspace                                      │
│    - Training jobs                                  │
│    - Endpoints (real-time + batch)                  │
│    - Model registry                                 │
│    - Pipelines                                      │
│    - Data assets                                    │
│                                                     │
│  Azure Databricks                                   │
│    - Notebooks                                      │
│    - Spark jobs                                     │
│    - MLflow (bundled)                               │
│    - Delta Lake                                     │
│    - Unity Catalog                                  │
│                                                     │
│  Azure Data Factory (ADF)                           │
│    - Managed ETL/ELT                                │
│    - Pipelines with triggers                        │
│                                                     │
│  Core infra:                                        │
│    Azure Functions — serverless                     │
│    AKS              — Kubernetes                    │
│    Blob Storage     — S3 equivalent                 │
│    ACR              — container registry            │
│    Key Vault        — secrets                       │
│    Azure Monitor    — metrics + logs                │
└─────────────────────────────────────────────────────┘
```

---

## 11.11 Azure Databricks (your Tiger Analytics / ResMed experience)

### Core concepts
- **Workspace** — collaborative notebook environment
- **Clusters** — Spark compute (ephemeral or persistent)
- **Jobs** — scheduled notebook / Python runs
- **DBFS** — abstraction over Blob / ADLS
- **Unity Catalog** — centralized data governance
- **Delta Lake** — ACID transactions on parquet
- **MLflow** (bundled) — experiment tracking
- **Feature Store** (Databricks FS)
- **Model Serving** (managed endpoints)

### Your Mars / ResMed wins
- "Implemented Deequ data quality checks on Databricks for Mars" — talk about:
  - Deequ constraints (completeness, uniqueness, min/max, statistical tests)
  - Running as a scheduled Databricks job
  - Publishing failures to Azure Monitor / Slack
- "Orchestrated pipelines using Azure Data Factory" — ADF pipelines with Databricks activities
- "Snowflake feature store" (ResMed) — even though Snowflake (not Azure), the *pattern* is Databricks-adjacent

---

## 11.12 Azure ML Studio

### Training
- Define `AmlCompute` (cluster) and `ScriptRunConfig` (code + env)
- Submit via SDK v2 (Python)
- Logs / metrics → Azure ML workspace

### Endpoints
- **Managed online endpoints** — real-time
- **Batch endpoints** — async scoring
- Deployment: traffic split for A/B; blue-green via deployment objects

---

## 11.13 Multi-cloud patterns

### Why multi-cloud for AI
- Regional data residency (UAE has Azure and AWS regions)
- Vendor cost arbitrage (spot instance pricing)
- Feature parity across tenants
- Regulatory separation

### Common patterns
- **Models on AWS, analytics on Azure** — SageMaker endpoints; Azure ML for heavy Spark analytics
- **Gateway abstraction** — LiteLLM / custom gateway routes LLM calls across Bedrock / Azure OpenAI
- **Terraform modules** provider-abstracted (where sensible)

### Gotchas
- Identity federation (AWS STS ↔ Azure AD) is a meaningful workstream
- Network egress costs across clouds
- Monitoring unification (e.g., Datadog across both)

---

## 11.14 Interview Q&A — Cloud

**Q1. SageMaker endpoint types — when each?**
> Real-time: <100ms, steady traffic, always-on. Async: long (<60min) inference, SQS-backed, scale-to-zero. Serverless: sporadic traffic, CPU-only, cold-start prone. Batch Transform: offline S3 scoring, no endpoint.

**Q2. MME vs MCE?**
> MME (Multi-Model Endpoint): many homogeneous models sharing one container; load from S3 on demand. Good for thousands of small tenant-specific models. MCE (Multi-Container): up to 15 heterogeneous containers invokable by target; good for A/B, ensembles, preprocess→model chains.

**Q3. Lambda vs SageMaker for inference?**
> Lambda: small models (<1GB quantized), spiky traffic, tight AWS integration. Break-even: <1 req/sec sustained → Lambda; beyond → SageMaker serverless or real-time.

**Q4. Bedrock vs self-hosting?**
> Bedrock: zero ops, per-token pricing, access to Claude/Llama/Mistral. Self-host: control quantization/batching (vLLM), cheaper at sustained QPS. Break-even ~1M tokens/hour. For UAE, verify me-central-1 availability.

**Q5. VPC isolation gotchas for SageMaker?**
> NetworkIsolation=True to block egress. Need S3 VPC endpoint (Gateway) for model artifacts. ECR, Secrets Manager, CloudWatch Logs Interface endpoints. Missing S3 endpoint = #1 cause of mysteriously hanging jobs.

**Q6. ECR best practices for ML?**
> Immutable tags (git SHA), lifecycle delete of untagged images (they're huge), Enhanced Scanning (Inspector CVE), pull-through cache for public ECR, cross-region replication for multi-region deploys.

**Q7. [Gotcha] SageMaker endpoint has p99 spikes every few minutes. Debug?**
> (1) Auto-scaling thrashing — check scaling events. (2) Cold tokenizer/model loads on new instances — pre-warm. (3) GC pauses in Python — try Triton / TorchServe. (4) KV cache fragmentation with variable batch sizes. (5) Noisy neighbor — pin to dedicated. Always enable Model Monitor + data capture.

**Q8. Provisioned concurrency for Lambda?**
> Pre-initialized Lambda instances — zero cold start. Costs money when idle. Use for ML Lambdas with strict p99 (your TrueBalance case). Tune via target tracking of `ProvisionedConcurrencyUtilization`.

**Q9. What's the role of EventBridge in MLOps?**
> Event-driven triggers: new data in S3 → start training pipeline. Model registered → notify Slack. Schedule-based retraining. Decouples producers from consumers.

**Q10. Azure Databricks core concepts?**
> Workspace (notebooks), clusters (Spark compute), jobs (scheduled runs), DBFS (storage abstraction), Unity Catalog (governance), Delta Lake (ACID parquet), bundled MLflow + Feature Store + Model Serving.

**Q11. What is Unity Catalog?**
> Centralized governance across Databricks workspaces — tables, ML models, volumes (unstructured), permissions (RBAC), lineage. Replaced per-workspace Hive metastores.

**Q12. Azure ML Studio endpoint types?**
> Managed online endpoints (real-time, traffic split for A/B) and batch endpoints (async scoring). Deployment via SDK v2 with ScriptRunConfig and environment dependencies.

**Q13. Azure OpenAI vs Bedrock?**
> Azure OpenAI: GPT family (OpenAI hosted in Azure), enterprise compliance. Bedrock: Claude/Llama/Nova/Mistral, AWS-native. Most enterprises run both via a gateway (LiteLLM, Portkey) to route by model family and cost.

**Q14. [Gotcha] Lambda cold start is 4 seconds — too slow for your SLA. Options?**
> (1) Provisioned concurrency. (2) SnapStart (Python/Java). (3) Slim container image (prune deps). (4) Lazy-load heavy imports. (5) If none work, switch to SageMaker real-time endpoint with min 1 instance.

**Q15. Airflow for ML — gotchas?**
> Airflow is task orchestration, not data streaming. Use Airflow for retraining pipelines, feature ingestion jobs, batch scoring. Don't use Airflow for real-time serving. Monitor DAG run duration; alert on stale feature data.

**Q16. How do you handle secrets in ML pipelines?**
> Secrets Manager (AWS) or Key Vault (Azure). Never in code, env files, or SageMaker Processing arguments. IAM role grants read at runtime. Rotate regularly.

**Q17. Cost monitoring for ML?**
> AWS Cost Explorer + CloudWatch Logs Insights (tag-based). Azure Cost Management + Log Analytics. For LLMs, token tracking via gateway (LiteLLM). Dashboards: cost per model, per tenant, per environment. Alert on budget breach.

**Q18. Terraform vs CloudFormation vs CDK?**
> Terraform: multi-cloud, big ecosystem, state file management. CloudFormation: AWS-native, no external state. CDK: programmatic (TypeScript, Python) generating CloudFormation. For ML on AWS + Azure, Terraform wins. Your resume lists Terraform — good.

**Q19. KMS for SageMaker?**
> Customer-managed CMKs for volume, EBS, S3, SageMaker Notebook, endpoints. Separate CMKs per environment + data sensitivity class. Rotation on schedule.

**Q20. Multi-region ML deploy pattern?**
> Model registry = single source of truth. Artifacts replicated to regional S3. Endpoints replicated across regions. Route53 latency-based routing or failover. CI/CD deploys to all regions simultaneously via parallel stages.

**Q21. How do you handle cross-AZ failover for SageMaker?**
> Endpoints auto-deploy across AZs when min_instances ≥ 2 in multi-AZ subnets. Alarm + Route53 health checks for cross-region fallback if needed.

**Q22. [Gotcha] Your training job hangs on VPC-isolated SageMaker. What's wrong?**
> Missing VPC endpoint (usually S3 or ECR). Check security-group egress to required endpoints. Check VPC route tables. Enable CloudWatch Logs to see where it hangs (dataset download, image pull, etc.).

**Q23. Spot instances for training?**
> Huge cost savings (50-90%). Checkpoint frequently (every 15-30 min) so interruption doesn't lose progress. SageMaker Managed Spot Training handles interruption + resume automatically. Not appropriate for real-time inference.

**Q24. ADLS Gen2 vs Blob storage?**
> Both use the same storage account. ADLS Gen2 adds hierarchical namespace (directory semantics), ACLs on files/directories, POSIX-like permissions. Better for big-data analytics. Default for Databricks.

**Q25. Avrioc UAE — cloud recommendation?**
> Primary: AWS me-central-1 (Bahrain) for Bedrock + SageMaker. Secondary: Azure UAE North / Central for redundancy or Azure-specific workloads. Self-hosted LLMs via vLLM on EKS for data-residency use cases. Bedrock for managed inference. Unified observability via Datadog / Langfuse.

---

## 11.15 Resume tie-ins

- **"Built real-time XGBoost Lambda (p99 < 500 ms, 3-env VPC-isolated)"** — this is a gold-standard AWS story. Be ready to whiteboard: VPC layout, Terraform modules, CloudWatch custom metrics, provisioned concurrency, container optimization.
- **"8 models to production in 6 months" (ResMed)** — SageMaker Pipelines + CodePipeline + MLflow + EventBridge. Draw the pipeline.
- **"Deequ data quality on Databricks" (Tiger)** — Azure Databricks-native. Have a concrete constraint example.
- **"Snowflake feature store"** — not AWS/Azure native but you integrated it. Discuss the online read path latency.

---

Continue to **[Chapter 12 — Kubernetes, Ray, Docker](12_kubernetes_ray.md)**.
