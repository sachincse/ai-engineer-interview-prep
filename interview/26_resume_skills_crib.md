# Chapter 26 — Resume Skills Crib Sheet

> **Why this chapter exists:** Your resume lists ~70 distinct tools and concepts in the Technical Skills section. An interviewer can pick any one and ask "you mentioned X — tell me about it." This chapter gives you a tight 2–4 sentence narrative for each, so you're never caught silent on a tool you legitimately know but haven't articulated recently. Read this morning of the interview if you want a quick skill-list refresh.
>
> **How to use:** Use Ctrl-F to find a specific skill if asked about one. The narrative for each is calibrated for "20-30 seconds of fluent talk" — enough to show you know it, short enough not to bore the interviewer. If they pull deeper, you've got the deep chapters.

---

## 26.1 Programming Languages

### Python
The default language for ML in 2026 and the language I write most production code in. Strengths: vast ML ecosystem (NumPy, PyTorch, scikit-learn, Pandas), readable, fast to prototype. Weaknesses: GIL limits true parallelism, async-vs-sync complexity, type system is optional. I lean on type hints and Pydantic to compensate. Used Python in every role I've held — XGBoost Lambda at TrueBalance, RAG pipelines at ResMed, SageMaker workflows at Tiger Analytics.

### PySpark
Python API for Apache Spark. The right tool when data doesn't fit on one machine — terabyte-scale ETL, distributed feature engineering, large-scale joins. The mental model: write Pandas-shaped code that compiles to a distributed execution plan across a cluster. Used at Tiger Analytics for Mars's large-scale data quality pipelines on Databricks. The catch is the learning curve around lazy evaluation, partitioning, and shuffles — getting these wrong turns a 10-minute job into a 6-hour job.

### Java
Used at Sopra Steria for backend services and during my coding-championship win in 2019. Strong for enterprise systems with strict typing requirements. Less common in modern ML stacks except for streaming infrastructure (Kafka, Flink) and Android. I keep enough fluency to read it; don't claim to author large Java codebases anymore.

### C
Used during my B.Tech program for systems-level work. Foundation for understanding performance-critical sections of ML libraries — most NumPy and PyTorch operations bottom out in C/C++. Comfortable reading C; not actively writing it day-to-day. Useful for understanding why FlashAttention or Triton kernels matter.

### SQL
The single highest-leverage skill in any ML role. Used heavily across Snowflake at TrueBalance and ResMed, BigQuery and Postgres on side projects. Strengths I leverage: window functions for time-series features, CTEs for readable multi-stage transformations, MERGE for idempotent writes, EXPLAIN for query optimization. The principle I follow: push compute into the database (set-based) before pulling rows into Python (row-based). Orders of magnitude difference in performance.

### Shell Scripting
Bash for one-off automation, glue, and CI/CD pipelines. Comfortable with basic loops, pipes, awk, sed, find, xargs. I don't write large bash programs — past 100 lines I switch to Python — but I can ship a working deployment script or log-parsing one-liner without thinking.

---

## 26.2 Machine Learning & LLMs

### GenAI
Generative AI — the umbrella term for models that produce content (text, images, code) rather than classifying or predicting numerically. In production GenAI work, "GenAI" usually means LLM-based systems with prompt engineering, RAG, or agent orchestration. Built the Claude-powered ML workspace at TrueBalance and the RAG-based clinical chatbot at ResMed.

### LLM (Large Language Model)
Transformer-based language models with billions of parameters trained on internet-scale text — Claude, GPT, Llama, Mistral, Qwen. The two-phase mental model: pretraining on raw text (next-token prediction), then alignment via SFT plus RLHF or DPO. Used Claude at TrueBalance for the developer-platform agent, and integrated open-source LLMs via vLLM for self-hosted serving research.

### RAG (Retrieval-Augmented Generation)
Pattern where an LLM is grounded in documents retrieved at query time rather than relying on its weights alone. Pipeline: chunk corpus, embed chunks, store in vector DB, on query embed and retrieve top-k, stuff into prompt, generate answer with citations. Built RAG at ResMed for the clinical knowledge-base chatbot — pgVector for storage, section-aware chunking, LLM-based query routing across factual/analytical/conversational query types.

### NER (Named Entity Recognition)
NLP task of extracting structured entities — names, locations, organizations, dates — from unstructured text. Modern approach: fine-tune a transformer encoder like BERT or DistilBERT with token-level classification. Used at TrueBalance to lift production lender-identification accuracy from 29.7% to 68.0% by layering a NER-based extractor over a 2K-term keyword expansion plus BANK-lender boosting heuristic.

### XGBoost
Gradient-boosted decision trees, the workhorse for tabular ML. Strengths: handles missing values, robust to outliers, fast inference, well-understood. The XGBoost Lambda at TrueBalance — p99 < 500ms across three environments — is a production XGBoost story. Tuning levers I reach for: max_depth, learning_rate, n_estimators, regularization (alpha, lambda), early stopping on validation loss.

### LGBM (LightGBM)
Microsoft's gradient-boosting library, similar to XGBoost but faster on large datasets via histogram-based splitting and leaf-wise growth. The choice between XGBoost and LightGBM is often empirical — try both, ship whichever validates better. LightGBM tends to win on data with many features and many rows.

### Random Forest
Ensemble of decision trees with bagging — train each tree on a bootstrap sample with random feature subsets, average their predictions. Less powerful than XGBoost typically, but more interpretable and harder to overfit. Useful as a baseline before reaching for boosting.

### SVM (Support Vector Machine)
Classifier that finds the maximum-margin hyperplane separating classes; with kernels (RBF, polynomial) handles non-linear boundaries. More common pre-2015; rarely the right choice today for tabular ML compared to gradient boosting. I know it; haven't shipped one in years.

### ARIMA (AutoRegressive Integrated Moving Average)
Classical time-series forecasting model — predicts future values from past values via autoregression, integration (differencing for stationarity), and moving average of errors. Used at Sopra Steria for server anomaly detection on Prometheus metrics. Modern deep learning forecasters (Prophet, NeuralProphet, deep state-space models) often beat ARIMA but require more tuning.

### RNN / LSTM
Recurrent neural networks with hidden state that propagates across timesteps. LSTM — Long Short-Term Memory — adds gating to handle long-range dependencies. Largely superseded by transformers post-2018 for NLP, but still useful for short-sequence time-series and some embedded use cases. Used at Sopra Steria for time-series anomaly detection.

### CNN (Convolutional Neural Network)
Neural network with convolutional layers that exploit spatial locality — the standard architecture for computer vision pre-vision-transformers. ResNet, VGG, EfficientNet are the famous families. Used at Sopra Steria for the digital ID verification system (face matching plus document classification).

### YOLO (You Only Look Once)
Real-time object detection — single-stage detector that predicts bounding boxes and classes in one forward pass, much faster than two-stage detectors like Faster R-CNN. Used at Sopra Steria for ID card detection in the verification pipeline.

### OCR (Optical Character Recognition)
Extracting text from images. Tools I've used: Tesseract for general-purpose OCR, AWS Textract for structured document OCR, easyOCR for multi-language. Used at Sopra Steria as the second stage of ID verification — detect the document with YOLO, OCR the text fields, validate against expected formats.

### BERT (Bidirectional Encoder Representations from Transformers)
The seminal encoder-only transformer (Google, 2018). Pre-trained on masked language modeling, fine-tuned for downstream tasks like classification, NER, and sentence embeddings. Modern variants: DistilBERT (smaller, faster), RoBERTa (better trained), DeBERTa (improved attention). The NER work at TrueBalance was BERT-family fine-tuned for token classification.

### NLP (Natural Language Processing)
Umbrella for the discipline of teaching computers to understand and generate human language. Modern NLP is essentially "transformers and LLMs" — the rest (tokenizers, pre-trained embeddings, classification heads) is plumbing. The NER work and the RAG chatbot at ResMed are both NLP-flavored.

### XAI (Explainable AI)
Techniques for explaining why a model made a specific prediction. SHAP (Shapley values) is the production standard for tabular models — gives you per-feature contribution per prediction. LIME for local linear explanations. For deep learning: integrated gradients, attention maps. Used SHAP at Sopra Steria for the loan-risk ensemble model, where regulators needed explanations for individual decisions.

### KMeans
Unsupervised clustering algorithm that partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids to the cluster mean. Useful for customer segmentation, anomaly detection (flag points far from any cluster), and as a feature engineering step. Limitations: requires choosing K (use elbow method or silhouette score), assumes spherical clusters, sensitive to scale.

### Multi-Objective Optimization
Optimizing for multiple competing goals simultaneously — typical example is route planning that balances total distance, time windows, and vehicle capacity. Used at Sopra Steria with OR-Tools (Google's optimization library) to reduce coverage time for 300 logistics locations from 7 days to 5. The two main approaches: scalarize (weighted sum of objectives) or compute the Pareto frontier and pick from it.

### Quantization
Compressing model weights from FP16/FP32 to lower precision (INT8, INT4, FP8) to reduce memory and increase throughput. Standard formats: GPTQ and AWQ for INT4, SmoothQuant for handling activation outliers, FP8 for H100 native inference. A 70B model is 140GB in FP16, 70GB in INT8, 35GB in INT4 — quantization is what lets you serve large models on smaller hardware.

### Pruning
Removing weights from a model to reduce its size. Magnitude pruning zeroes out weights below a threshold; structured pruning removes entire heads or layers. Combined with fine-tuning to recover accuracy. Less commonly used than quantization in 2026 because quantization gives a similar memory win without the accuracy cost.

### Distillation
Training a small "student" model to mimic a large "teacher" model. The student learns from the teacher's soft probability outputs (which carry more information than hard labels), often combined with a temperature parameter to smooth them. DistilBERT (40% smaller, 60% faster, 97% of BERT's performance) is the canonical example. I use distillation when I need a deployable model and a large reference model is available.

---

## 26.3 Cloud — AWS

### SageMaker
AWS's managed ML platform. Components I've used: training jobs (managed compute for model training), processing jobs (managed compute for arbitrary scripts, used for preprocessing), endpoints (real-time inference — single, multi-model, multi-container variants), pipelines (DAG of jobs), model registry (versioned model artifacts). The IHS MLOps platform at ResMed was SageMaker-centric, with multi-container endpoints consolidating eight low-traffic models on shared compute.

### Lambda
AWS's serverless compute. Container image format up to 10GB, 15-minute timeout, no GPU. Best for CPU-bound, low-latency, intermittent inference workloads. The TrueBalance XGBoost Lambda is the canonical use case — p99 < 500ms, three-env VPC isolation, container image, provisioned concurrency to absorb cold starts.

### CodeBuild
AWS's managed build service. Triggered by source events (CodeCommit, GitHub, S3), runs builds in Docker containers, produces artifacts. I use it as the build stage in CodePipeline workflows for SageMaker model containers — `buildspec.yml` defines build commands.

### CodePipelines
AWS's CI/CD orchestration. Triggers on source changes, runs through stages (build, test, deploy) with manual approvals as gates. Used for SageMaker pipeline promotion from dev → staging → prod. Less feature-rich than GitHub Actions or Jenkins but tightly integrated with AWS resources via IAM.

### ECR (Elastic Container Registry)
Private Docker registry on AWS. Image versioning, lifecycle policies (delete images older than X), scan-on-push for CVEs, immutable tags. Used to publish SageMaker custom containers, Lambda container images, and EKS workloads.

### EC2
Virtual machines on AWS. Instance families that matter: c-series for CPU-bound, r-series for memory-heavy, g- and p-series for GPU, inf for AWS Inferentia. Used directly less in modern stacks (Lambda, ECS, EKS abstract it away), but understanding instance types is critical for capacity planning.

### EFS (Elastic File System)
Managed shared filesystem on AWS, NFS-compatible, mountable from many EC2/Lambda instances simultaneously. Used at TrueBalance for the EFS-shared state in the Claude-powered ML workspace — multiple EC2 instances need the same git worktrees and tooling. Slower than local SSD but enables cross-instance sharing.

### EBS (Elastic Block Storage)
Block storage volumes attached to single EC2 instances. Variants: gp3 (default general-purpose), io2 (high-IOPS), st1 (throughput-optimized), sc1 (cold storage). Used the 3-method EBS lifecycle pattern at TrueBalance for the workspace ML environment provisioning.

### CloudWatch
AWS's monitoring service. Components: CloudWatch Metrics (time-series numerical data), CloudWatch Logs (log aggregation), CloudWatch Alarms (alerts on metric thresholds), CloudWatch Insights (SQL-like log queries). Used for p99 latency instrumentation on the TrueBalance Lambda and for log-based debugging across all AWS services.

### EventBridge
AWS's event bus / cron scheduler. Triggers Lambda or other AWS services on cron expressions, S3 events, SageMaker job state changes, custom events. Used for the cache-refresher cron in the TrueBalance Lambda, and for SageMaker training job notifications on the IHS platform.

### VPC (Virtual Private Cloud)
AWS's network isolation primitive. Components: subnets (public for internet-facing, private for internal), route tables, security groups (firewall rules), NAT gateways (allow private subnets to reach the internet outbound). Three-environment VPC isolation at TrueBalance separated dev/staging/prod with no cross-VPC traffic.

### S3
AWS's object storage. Foundation for almost every AWS data workflow — model artifacts, training data, logs, backups. Patterns I use: prefix design for parallel reads (datestamp prefixes), multipart upload for files > 100MB, lifecycle rules to archive to Glacier after N days, server-side encryption with KMS, S3 Express One Zone for low-latency hot reads.

---

## 26.4 Cloud — Azure

### Databricks
Azure's managed Apache Spark + Delta Lake + MLflow platform. Used at Tiger Analytics for the Mars project — large-scale data quality with the Deequ library, distributed feature engineering. Strengths: tight Spark integration, Delta Lake gives ACID over parquet, notebooks with clusters spin up on demand. The pricing model can spike if clusters are misconfigured.

### Data Factory
Azure's ETL orchestration service. UI-driven pipelines with a JSON definition under the hood. Used at Tiger Analytics for batch data movement and pipeline orchestration before we shifted some workloads to Airflow. Less Pythonic than Airflow but tightly integrated with Azure Storage and Databricks.

### Machine Learning Studio
Azure's managed ML platform — the rough equivalent of SageMaker. Components: compute targets, environments (Python dependency specs), pipelines, registered models, real-time and batch endpoints. I've used the basics; less day-to-day fluency than I have in SageMaker.

### Azure Storage
Azure's blob storage, equivalent to S3. Tiers: Hot (frequent access, expensive), Cool (infrequent, cheaper), Archive (rare, cheapest, slow retrieval). ADLS Gen2 layers a hierarchical filesystem on top of blob storage for big-data workloads.

### Virtual Machine
Azure's VMs, equivalent to EC2. Used for ad-hoc compute when serverless or managed services don't fit. Less common in modern Azure stacks where Azure Container Apps and AKS are preferred.

### Azure AD (Azure Active Directory)
Azure's identity and access management. Used for SSO, service principals (the equivalent of AWS IAM roles), and managed identities. Worked with managed identities on Databricks workspaces to avoid storing credentials.

---

## 26.5 Containerization & Orchestration

### Docker
Container runtime — packages applications with their dependencies into reproducible images. The base layer of every modern deployment. Multi-stage builds for smaller images, BuildKit cache mounts for faster builds, .dockerignore to keep build contexts lean. I containerize every ML service I ship.

### Kubernetes
Container orchestrator — schedules and manages containerized workloads across a cluster. Core abstractions: Pod (one or more containers), Deployment (rolling updates over Pods), Service (stable network endpoint), Ingress (HTTP routing). Used for the Claude-powered ML workspace's tooling layer at TrueBalance and for SageMaker endpoints' underlying compute at ResMed (managed by AWS, but the abstractions are K8s-equivalent).

### OpenShift
Red Hat's enterprise Kubernetes distribution. Adds ImageStreams, Routes (Kubernetes Ingress equivalent), built-in CI/CD via OpenShift Pipelines (Tekton), and stricter security defaults. Used at Sopra Steria for some enterprise workloads. Mostly K8s under the hood with extra opinions.

### Docker Compose
Tool for defining multi-container apps in a YAML file (`docker-compose.yml`) and running them locally with one command. Used for local dev environments where I need a Postgres + Redis + my-service stack running together. Not for production — that's K8s territory.

### NVIDIA Container Toolkit
The Docker plugin that exposes GPUs to containers. Without it, `docker run` doesn't see the GPU. Setup is install on the host, set the runtime to `nvidia`, and pass `--gpus all`. Required infrastructure for any ML container that uses GPU.

### Ray
Distributed Python compute framework. Components: Ray Core (tasks and actors), Ray Serve (model serving with composition), Ray Train (distributed training), Ray Tune (hyperparameter search), Ray Data (distributed datasets). The Avrioc JD names Ray explicitly. I'd reach for Ray Serve when composing multi-stage AI pipelines (RAG with embedder + retriever + reranker + generator) where each stage needs different resources and autoscaling.

---

## 26.6 Frameworks & APIs

### FastAPI
Modern Python web framework — async-native, Pydantic validation, automatic OpenAPI docs, Starlette under the hood. The default for new ML APIs in 2026. Detailed in [Chapter 13](13_frameworks.md).

### Flask
The classic Python web framework — sync (WSGI), unopinionated, vast ecosystem. Used widely pre-2020 for ML serving; now mostly legacy. Detailed in [Chapter 25](25_flask_api.md).

### Django
Heavyweight Python framework with built-in ORM, admin interface, auth, templating. Strong for full-stack apps (think Pinterest, Instagram early days), less common for pure ML serving. I know it but rarely reach for it in ML contexts.

### LangChain
LLM workflow framework — chains, agents, tools, integrations with hundreds of vector stores and APIs. Strong for prototyping; production teams often migrate stateful workflows to LangGraph. Detailed in [Chapter 13](13_frameworks.md).

### LangGraph
Stateful graph framework for LLM workflows. Explicit nodes and edges, conditional routing, persistence, human-in-the-loop. The production successor to LangChain agents. The TrueBalance Claude-powered ML workspace uses LangGraph for its agent state machine.

### CrewAI
Multi-agent orchestration framework with role-based agents and task passing. Useful when the problem decomposes into specialist agents (researcher, writer, editor). Detailed in [Chapter 13](13_frameworks.md).

### vLLM
High-performance LLM serving framework. Innovations: PagedAttention (KV-cache management), continuous batching (iteration-level scheduling). The default open-source LLM serving stack in 2026. Detailed in [Chapter 13](13_frameworks.md).

### Streamlit
Data app framework — turns Python scripts into web apps with reactive widgets. Strong for dashboards and internal tools. The execution model (rerun the whole script on every interaction) is unique. Detailed in [Chapter 13](13_frameworks.md).

### Chainlit
Chat-first UI framework for LLM apps. Streaming tokens, tool-call visualization, multi-turn message rendering, auth, file uploads — all built-in. Use over Streamlit when the primary interaction is chat. Detailed in [Chapter 13](13_frameworks.md).

---

## 26.7 Automation & CI/CD

### Terraform (IaC)
HashiCorp's infrastructure-as-code tool — declarative configuration for cloud resources, with state tracking and plan/apply workflow. Used for the TrueBalance VPC + Lambda + Redis cluster across three environments. Strengths: cloud-agnostic (AWS, Azure, GCP, K8s), strong community modules, deterministic plans. Watch out for: state file management (S3 backend with DynamoDB locking is the standard), drift between code and reality.

### CloudFormation
AWS's native IaC service. Less popular than Terraform in multi-cloud teams, but tightly integrated with AWS — you can describe SageMaker pipelines, Lambda functions, and Step Functions in YAML and they map cleanly. Used at ResMed for some legacy resource stacks.

### Apache Airflow
Workflow orchestrator for data pipelines. Detailed in [Chapter 24](24_apache_airflow.md). Used at ResMed for IHS preprocessing orchestration.

### GitHub Actions
GitHub-native CI/CD. YAML-defined workflows triggered by push, PR, or schedule. Used for testing, linting, building Docker images, deploying to staging, running scheduled jobs. Strengths: first-class GitHub integration, large action marketplace, free for public repos. Used for testing Airflow DAGs and building model containers.

---

## 26.8 Observability & Monitoring

### Prometheus
Time-series metrics database with a pull model — scrapes metrics from instrumented services on a schedule. Standard for Kubernetes monitoring. PromQL is the query language. Used at Sopra Steria for server metrics, paired with Grafana.

### Grafana
Visualization tool for time-series metrics. Connects to Prometheus, Datadog, Snowflake, Elasticsearch, and dozens of others. Build dashboards with PromQL queries and alerts. The standard infrastructure dashboard tool.

### Datadog
Commercial APM (Application Performance Monitoring) platform. Logs, metrics, traces, and dashboards in one place. Used heavily at ResMed for the drift dashboard utility — pushed model drift metrics from Snowflake-based jobs to Datadog as custom metrics, auto-generated dashboards per model. Pricing scales with cardinality of tags, which can spike if you over-tag.

### TrueFoundry
ML platform for deploying and managing models — abstracts away K8s and serving infrastructure. Used briefly for one project at TrueBalance. Strengths: fast deployments, good observability, multi-cloud support. Less mature than Databricks or SageMaker for end-to-end MLOps.

### Metabase
Open-source BI / analytics tool — connects to data warehouses (Snowflake, BigQuery, Postgres) and lets non-technical users build dashboards via SQL or a visual builder. Used at ResMed for some non-ML business dashboards.

---

## 26.9 Databases & Feature Stores

### PostgreSQL
The default open-source relational database. ACID, rich SQL, strong ecosystem. Variants I leverage: pgVector (vector search extension), pg_partman (table partitioning), pg_stat_statements (query performance analysis). Used at multiple roles for transactional and analytical workloads.

### Snowflake
Cloud data warehouse. Decoupled compute and storage, near-infinite scaling, column-store performance. Used heavily at TrueBalance and ResMed for feature stores, analytical queries, and as the base for Datadog drift metrics. Pricing model: per-second compute via warehouse credits — auto-suspend warehouses to save cost.

### Oracle
Legacy enterprise relational database. Used at Sopra Steria for ETL within Oracle Apps via PL/SQL. Strong for transactional OLTP; in ML contexts now mostly legacy.

### MySQL
Open-source relational database. Used for some smaller services. Less common than Postgres in modern Python stacks, but widely deployed.

### SQLite
Embedded zero-config relational database — a single file, no server. Used for local dev, small CLI tools, Python packaging tests. Great for "I just need a database for this script" use cases.

### pgVector
Postgres extension that adds vector data types and similarity search (HNSW, IVFFlat indexes). Used at ResMed for the RAG clinical chatbot — kept the vector store in Postgres alongside transactional data, no separate vector DB to operate. Production-ready up to ~10M vectors; specialized vector DBs win at higher scale.

### Redis
In-memory key-value store with pub-sub, streams, and pub-sub. Use cases: cache, session store, rate limiting, lightweight queue. Used at TrueBalance as the online feature cache for the XGBoost Lambda — Redis-fronted Snowflake feature store, sub-millisecond reads.

---

## 26.10 Dashboarding

### Plotly
Python charting library — interactive HTML charts (zoom, hover, pan). Used for ad-hoc analysis and inside Streamlit/Dash apps. The express API (`plotly.express`) is how I'd reach for it most often.

### Streamlit
Already covered above under Frameworks.

---

## 26.11 Functional Domains (less likely to be probed deeply, but be ready)

### LLMOps
Operations practices for LLM-based applications — prompt versioning, eval frameworks, observability, guardrails, A/B testing. The newer cousin of MLOps. Tools: LangFuse, Helicone, Arize Phoenix, Guardrails-AI.

### MLOps
End-to-end ML lifecycle operations — data pipelines, training, deployment, monitoring, retraining. The practice that makes ML work in production. Detailed in [Chapter 10](10_mlops_llmops.md).

### Predictive Modeling
Building models that predict an outcome (regression or classification). The TrueBalance loan-withdrawal prediction is the canonical example — XGBoost predicting probability of withdrawal post-approval.

### Real-Time Inference
Serving predictions with sub-second latency. The TrueBalance Lambda — p99 < 500ms — is real-time inference. Patterns: in-process inference, dedicated model server (vLLM, Triton), feature caching, model warm-up.

### Model Optimization
Reducing model size or latency without sacrificing accuracy. Techniques: quantization (INT8/INT4), pruning, distillation, KV-cache management, speculative decoding. Detailed in [Chapter 09](09_model_optimization.md).

### Computer Vision
Models that process images or video. Worked on YOLO + CNN + OCR for ID verification at Sopra Steria. Modern shift: vision transformers (ViT, CLIP) replacing CNNs for many tasks; vision-language models (Llava, Qwen-VL) for multi-modal.

### NLP
Already covered.

### Drift Monitoring
Detecting when production data has shifted from training data. Techniques: PSI, KS test, embedding drift, prediction-vs-ground-truth deltas. Detailed in [Chapter 14](14_monitoring_drift.md). The Datadog drift dashboard at ResMed is the signature project.

### AI Solutions Architecture
End-to-end design of AI systems — choosing components, defining data flows, capacity planning, cost analysis. The system-design discipline applied to AI. Detailed in [Chapter 16](16_system_design.md).

### Scalable System Design
Designing systems that scale gracefully under load. Patterns: caching, partitioning, asynchronous processing, horizontal scaling, queueing, circuit breakers. The TrueBalance Lambda architecture (Lambda + Redis + Snowflake fallback + Terraform multi-env) is a small example.

---

## 26.12 The 3 things to do if asked about a tool you barely know

It happens. Someone picks an obscure entry on your skills list. The honest play wins:

1. **Acknowledge the truth.** "I've used X briefly on one project, but it's not in my daily toolkit. Here's what I know..." beats faking depth.
2. **Anchor what you do know.** Even with a tool you've used briefly, you usually know what category it's in and what problem it solves. Lead with that.
3. **Pivot to the related thing you know cold.** "I haven't done much with Y, but the closest tool I've used in production is Z, where I built..."

Interviewers respect honesty. Faking knowledge gets caught fast and damages credibility. Saying "I'd need to come up to speed on that" is almost always fine.

---

## 26.13 Read order on the morning

If you have 10 minutes:

1. **§26.1 Programming Languages** — Python, SQL, PySpark answers
2. **§26.5 Containerization** — Docker, K8s, Ray (Avrioc JD-named)
3. **§26.6 Frameworks** — FastAPI, vLLM, LangChain (Avrioc JD-named)
4. **§26.7 Automation** — Terraform, Airflow, GitHub Actions
5. **§26.8 Observability** — Datadog (your signature drift story)

Skip the rest unless you have specific concerns.

---

End of Chapter 26. Continue to **[Chapter 27 — RAG Evaluation Deep Dive](27_rag_evaluation.md)** or back to **[Chapter 00 — Master Index](00_index.md)**.
