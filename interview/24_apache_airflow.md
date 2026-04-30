# Chapter 24 — Apache Airflow End-to-End

> **Why this chapter matters:** Sachin's resume bullet says "Automated preprocessing tasks using Apache Airflow for seamless job orchestration" — that single line is enough for an Avrioc interviewer to spend 15 minutes pulling on. Airflow is also the most common workflow orchestrator in MLOps shops, so even if the interviewer doesn't pull on your bullet directly, they may ask "what would you use to orchestrate a feature pipeline?" expecting Airflow as the lead answer. This chapter gives you depth across architecture, operators, scheduling semantics, gotchas, and interview Q&A. Read in order if learning; jump to §24.18 (interview Q&A) and §24.19 (live coding) if drilling for an interview.

---

## 24.1 What Airflow actually is

### The plain-English definition

Apache Airflow is a workflow orchestrator. You describe a pipeline as a directed acyclic graph (DAG) of tasks in Python, point Airflow at it, and Airflow takes care of scheduling task runs, executing them in dependency order, retrying failures, propagating success/failure signals, and showing the whole thing in a UI. It does not move data itself — it tells other systems (Snowflake, SageMaker, S3, Kubernetes) to move data, then waits for them to finish and decides what to run next.

The mental analogy I use: Airflow is the **conductor** of an orchestra. It doesn't play any instrument. Its job is to know who plays when, watch for cues, and stop the next section if someone misses theirs.

### Why orchestration matters

Without an orchestrator, ML pipelines become a tangle of cron jobs, bash scripts, and "did the upstream finish?" Slack messages. Airflow gives you four things cron cannot:

1. **Dependency management** — task B runs only if task A succeeded, automatically.
2. **Retries with backoff** — a transient S3 hiccup doesn't kill the pipeline.
3. **Backfill** — re-run yesterday's pipeline for any past date with one CLI command.
4. **Visibility** — a UI that shows every task run, every log, every failure, every duration.

> **How to say this in an interview:** "Airflow's role in our IHS platform was orchestration, not compute. The heavy lifting happened in Snowflake and SageMaker; Airflow's job was to know what should run when, retry transient failures, alert on persistent ones, and give the team a UI to debug from. The principle I followed: don't put compute in Airflow workers — put it in the system designed for that compute, and use Airflow as the conductor."

---

## 24.2 The architecture — every component and what it does

### The block diagram

```
   ┌────────────────────────────────────────────────────────────────────┐
   │                          AIRFLOW DEPLOYMENT                        │
   ├────────────────────────────────────────────────────────────────────┤
   │                                                                    │
   │   ┌──────────────┐       ┌──────────────┐       ┌─────────────┐    │
   │   │  Webserver   │◀─────▶│  Metadata DB │◀─────▶│  Scheduler  │    │
   │   │   (UI, REST) │       │  (Postgres)  │       │             │    │
   │   └──────────────┘       └──────────────┘       └──────┬──────┘    │
   │                                  ▲                     │           │
   │                                  │                     │           │
   │                                  │                     ▼           │
   │                          ┌───────┴──────┐       ┌─────────────┐    │
   │                          │  DAGs Folder │       │  Executor   │    │
   │                          │ (Python files│       │             │    │
   │                          │  parsed by   │       └──────┬──────┘    │
   │                          │  scheduler)  │              │           │
   │                          └──────────────┘              ▼           │
   │                                                  ┌─────────────┐   │
   │                                                  │   Workers   │   │
   │                                                  │ (Celery /   │   │
   │                                                  │  K8s / Local)│  │
   │                                                  └─────────────┘   │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

### Each component, what it does, and why it exists

**Scheduler.** The brain. It continuously parses DAG files in the DAGs folder, computes which task instances should run right now based on schedules and upstream state, and pushes ready-to-run tasks to the executor. The scheduler also handles retries, SLA misses, and timeouts. In Airflow 2.x there can be multiple schedulers running in HA mode, sharing work via row-level locks on the metadata DB.

**Metadata DB.** Usually Postgres in production (MySQL works; SQLite is dev-only). Stores DAG definitions parsed from Python, every task instance ever run with its state, XComs, connections, variables, users, permissions. This DB is the source of truth — if you lose it, you lose history. **Back it up.**

**Webserver.** Flask app that gives you the UI and REST API. Read-mostly — it queries the metadata DB to render task graphs, logs, durations. The webserver does not execute any tasks; it's just a viewer with some control buttons.

**Executor.** The component that decides where and how task instances run. Five common executors:

- **SequentialExecutor**: runs one task at a time, in-process. Dev only, with SQLite.
- **LocalExecutor**: runs many tasks in parallel as subprocesses on the scheduler machine. Fine for small deployments.
- **CeleryExecutor**: distributes tasks to a pool of Celery workers via a message broker (Redis or RabbitMQ). Production standard for years.
- **KubernetesExecutor**: spawns a Kubernetes pod per task. Each task gets its own isolated environment. Heavier startup but cleanly isolated.
- **CeleryKubernetesExecutor**: hybrid — most tasks go to Celery workers (low overhead), specific high-resource tasks go to K8s pods.

**Workers.** The processes that actually run task code. They pull tasks from the executor, execute them, and report status back to the metadata DB.

**DAGs folder.** A directory the scheduler watches. Every Python file is parsed; any DAG objects defined at module level are registered. By default the scheduler re-parses every 30 seconds, which is why heavy top-level code in a DAG file is a sin (more on that in §24.16).

---

## 24.3 The four core abstractions

Every Airflow user must internalize these four concepts. They show up in every interview question.

### 24.3.1 DAG — the workflow

A DAG is a Python object representing the pipeline. It has a unique `dag_id`, a schedule (cron expression, timedelta, or `@daily`-style preset), a start date, and tasks. Tasks within a DAG have dependencies expressed via `>>` and `<<` operators or `set_upstream` / `set_downstream` calls.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="ihs_feature_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="0 2 * * *",            # daily at 02:00 UTC
    catchup=False,                    # don't backfill from start_date
    default_args={
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
        "owner": "sachin",
    },
    tags=["ihs", "features"],
) as dag:
    extract = PythonOperator(task_id="extract", python_callable=extract_fn)
    transform = PythonOperator(task_id="transform", python_callable=transform_fn)
    load = PythonOperator(task_id="load", python_callable=load_fn)

    extract >> transform >> load
```

### 24.3.2 Task — a single unit of work in a DAG

A task is an instance of an operator with a specific configuration. The operator defines *what kind* of work happens; the task is *the specific thing* in this DAG. When you write `extract = PythonOperator(task_id="extract", ...)`, `extract` is a task. The Python class `PythonOperator` is the operator.

### 24.3.3 Operator — the type of work a task performs

Operators are reusable templates. Hundreds ship with Airflow's "providers" (Airflow's plugin system) — `S3KeySensor`, `SnowflakeOperator`, `PythonOperator`, `KubernetesPodOperator`, etc. You can write your own by subclassing `BaseOperator`.

### 24.3.4 Task Instance — a specific run of a task

A task instance is a task at a specific execution date. If `extract` is the task and `2026-04-30` is the execution date, the task instance is `extract-2026-04-30`. Each task instance has its own state (queued, running, success, failed, skipped, retry, etc.) tracked in the metadata DB.

```
   DAG = "ihs_feature_pipeline"
        │
        ├── Task = "extract"  (PythonOperator instance)
        │       ├── TaskInstance "extract-2026-04-29"  state=success
        │       ├── TaskInstance "extract-2026-04-30"  state=running
        │       └── ...
        │
        ├── Task = "transform"
        │       └── TaskInstance "transform-2026-04-29"  state=success
        │
        └── ...
```

---

## 24.4 The full operator catalog — categorized

There are hundreds of operators. Don't memorize the list — internalize the categories and a few exemplars per category, then you can reason about any operator.

### Category 1 — Action operators (do work)

These operators execute work directly.

| Operator | What it does | When to use |
|----------|--------------|-------------|
| `PythonOperator` | Calls a Python function | Custom logic that doesn't fit a specialized operator |
| `BashOperator` | Runs a shell command | Quick scripts, file moves, calling CLIs |
| `SQLExecuteQueryOperator` | Runs SQL on a connection | Generic SQL execution |
| `SnowflakeOperator` | Runs SQL on Snowflake | Snowflake transforms (deprecated; use `SnowflakeSqlApiOperator` in newer versions) |
| `PostgresOperator` | Runs SQL on Postgres | Postgres transforms |
| `BigQueryOperator` | Runs SQL on BigQuery | GCP data warehouse work |
| `EmailOperator` | Sends an email | Notification (also see `email_on_failure`) |
| `KubernetesPodOperator` | Spawns a K8s pod, runs a container | Arbitrary container execution, isolation |
| `DockerOperator` | Runs a Docker container locally | Local containerized tasks |
| `SparkSubmitOperator` | Submits a Spark job | Spark workloads |
| `DatabricksRunNowOperator` | Triggers a Databricks job | Databricks-managed jobs |
| `DbtCloudRunJobOperator` | Triggers a dbt Cloud job | dbt transformations |
| `SageMakerProcessingOperator` | Starts a SageMaker Processing Job | Heavy preprocessing on AWS |
| `SageMakerTrainingOperator` | Starts a SageMaker Training Job | Model training |
| `SageMakerEndpointOperator` | Deploys / updates an endpoint | Model deployment |
| `EcsRunTaskOperator` | Runs an ECS task | AWS containerized work |
| `EmrCreateJobFlowOperator` | Spins up an EMR cluster | Spark on EMR |

### Category 2 — Sensors (wait for a condition)

Sensors block a task slot until a condition becomes true. They poll, then either succeed (condition met), fail (timeout), or get retried.

| Sensor | What it watches |
|--------|-----------------|
| `S3KeySensor` | A specific key (or prefix) appearing in S3 |
| `S3KeysUnchangedSensor` | A set of keys staying unchanged for N seconds (file is fully written) |
| `FileSensor` | A file existing on a local filesystem |
| `ExternalTaskSensor` | A specific task in another DAG completing |
| `TimeSensor` / `TimeDeltaSensor` | Wall-clock time reaching a specific point |
| `SqlSensor` | A SQL query returning a non-zero result |
| `HttpSensor` | An HTTP endpoint returning a specific response |
| `SnowflakePartitionSensor` | A Snowflake partition existing |
| `BigQueryTableExistenceSensor` | A BigQuery table existing |
| `KubernetesPodSensor` | A pod reaching a specific state |

**Critical sensor concept — modes.** Sensors have two execution modes:

- `mode="poke"` (default) — the task holds a worker slot the whole time it polls. Cheap to implement, expensive at scale.
- `mode="reschedule"` — the task releases its worker slot between polls. Slot is given back; sensor re-runs at the next poke interval.

For long waits (>5 minutes), always use `reschedule` mode. Otherwise you pin worker capacity to do nothing.

**Even better — deferrable operators.** Airflow 2.2+ introduced "deferrable" operators that yield control to a separate process called the `triggerer`. The task takes zero worker capacity while waiting. `S3KeySensorAsync`, `ExternalTaskSensorAsync`, `TimeDeltaSensorAsync` are examples. For long waits these are strictly better than reschedule mode.

### Category 3 — Transfer operators (move data between systems)

| Operator | Source → Destination |
|----------|---------------------|
| `S3ToSnowflakeOperator` | S3 → Snowflake (via COPY INTO) |
| `S3ToRedshiftOperator` | S3 → Redshift |
| `MySqlToS3Operator` | MySQL → S3 |
| `PostgresToGCSOperator` | Postgres → GCS |
| `GCSToBigQueryOperator` | GCS → BigQuery |
| `S3ToGCSOperator` | S3 → GCS |
| `LocalFilesystemToS3Operator` | Local file → S3 |

Transfer operators exist because the "move data from X to Y" pattern is so common that doing it yourself in Python is wasteful — these operators handle authentication, chunking, retry, and idempotency.

### Category 4 — Control flow operators (orchestrate without doing real work)

These don't do work — they shape the DAG.

| Operator | Purpose |
|----------|---------|
| `EmptyOperator` (formerly `DummyOperator`) | Placeholder, useful as a join/fan-in point |
| `BranchPythonOperator` | Choose which downstream branch to execute |
| `ShortCircuitOperator` | Skip all downstream tasks if a condition is false |
| `LatestOnlyOperator` | Skip downstream tasks unless this is the latest scheduled run |
| `TriggerDagRunOperator` | Trigger another DAG |

### Category 5 — Operators by Avrioc-relevant providers

If you're at Avrioc with the JD's stack, here's what you'd reach for:

```
   AWS provider:        S3KeySensor, S3ToSnowflakeOperator,
                        SageMakerProcessing/Training/EndpointOperator,
                        EcsRunTaskOperator, LambdaInvokeFunctionOperator,
                        AthenaOperator, GlueJobOperator

   Snowflake provider:  SnowflakeOperator, SnowflakeSqlApiOperator,
                        SnowflakePartitionSensor, S3ToSnowflakeOperator

   Kubernetes provider: KubernetesPodOperator, KubernetesPodSensor,
                        KubernetesJobOperator

   HTTP provider:       SimpleHttpOperator, HttpSensor

   Generic:             PythonOperator, BashOperator, BranchPythonOperator
```

---

## 24.5 The TaskFlow API — modern Airflow 2.x

The classic operator API is verbose. Airflow 2.0 introduced the TaskFlow API, which lets you write tasks as decorated Python functions and infer XCom data flow automatically.

### Classic style

```python
from airflow.operators.python import PythonOperator

def extract():
    return [1, 2, 3]

def transform(**context):
    data = context["ti"].xcom_pull(task_ids="extract")
    return [x * 2 for x in data]

def load(**context):
    data = context["ti"].xcom_pull(task_ids="transform")
    print(f"Loaded {data}")

with DAG(...) as dag:
    e = PythonOperator(task_id="extract", python_callable=extract)
    t = PythonOperator(task_id="transform", python_callable=transform)
    l = PythonOperator(task_id="load", python_callable=load)
    e >> t >> l
```

### TaskFlow style

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2026, 1, 1), schedule="@daily", catchup=False)
def etl_pipeline():
    @task
    def extract():
        return [1, 2, 3]

    @task
    def transform(data):
        return [x * 2 for x in data]

    @task
    def load(data):
        print(f"Loaded {data}")

    load(transform(extract()))

dag = etl_pipeline()
```

The TaskFlow version is shorter, more Pythonic, and the data flow (XCom) is implicit in the function call graph rather than stringly-typed `xcom_pull` calls. **For new DAGs in 2.x, default to TaskFlow** unless you specifically need a non-Python operator (in which case you mix — TaskFlow for Python tasks, classic operators for the rest).

---

## 24.6 Scheduling — execution_date, data intervals, and the trickiest concept in Airflow

This is the concept that confuses everyone the first time. Airflow's scheduling is **interval-based**, not point-in-time.

### The mental model

When a DAG with `schedule="@daily"` runs, the **execution_date** is the *start* of the data interval, not the time the task ran. So a DAG run with `execution_date=2026-04-30` is processing data from `2026-04-30 00:00` to `2026-05-01 00:00`, and it runs **at the end of that interval** — `2026-05-01 00:00`.

```
                          DAG schedule = "@daily"

   ┌─────────────────────────┬─────────────────────────┬──────────────
   │  data interval start =  │  data interval start =  │
   │  execution_date =       │  execution_date =       │
   │  2026-04-29 00:00       │  2026-04-30 00:00       │
   ├─────────────────────────┼─────────────────────────┼──────────────
   │                         │                         │
   │      data interval      │      data interval      │
   │       (one day)         │       (one day)         │
   │                         │                         │
   ├─────────────────────────┼─────────────────────────┼──────────────
                  ▲                          ▲
                  │                          │
            DAG run fires:             DAG run fires:
            2026-04-30 00:00           2026-05-01 00:00
            (right after the           (right after the
             interval ends)             interval ends)
```

So if you trigger the DAG at midnight April 30, the `execution_date` of the run is **April 29**. This is correct: the run is processing the day that just ended.

### Why this matters

If you write `WHERE event_date = '{{ ds }}'` in your SQL, you get the data for `execution_date`, which is the day the run is processing — not the day the run actually fires. That's almost always what you want.

### `ds` and `data_interval_start` / `data_interval_end`

Airflow exposes these as Jinja templates:

- `{{ ds }}` — `execution_date` formatted as `YYYY-MM-DD` (e.g. `2026-04-29`)
- `{{ ds_nodash }}` — same without dashes (e.g. `20260429`)
- `{{ data_interval_start }}` — the start of the data interval (a `datetime`)
- `{{ data_interval_end }}` — the end of the data interval
- `{{ ts }}` — execution_date as ISO timestamp
- `{{ next_ds }}` — the *next* execution date, which equals `data_interval_end` formatted as `YYYY-MM-DD`

The 2.2+ rename: in modern Airflow, prefer `data_interval_start` / `data_interval_end` because the term "execution_date" is misleading.

### Catchup vs. no-catchup

`catchup=True` (default) means: if the DAG's start_date is in the past and the DAG hasn't run for that period, Airflow will *backfill* every missed interval, one after another, until it catches up to now. This is useful for new DAGs replacing old pipelines. It's also dangerous — turning on a new DAG with a 6-month-old start date can spawn 180 simultaneous runs.

Best practice for new DAGs: `catchup=False` plus an explicit `start_date` close to "now." Backfill manually if you need to.

---

## 24.7 Templating — Jinja and macros

Every operator field that's marked as templated (most string fields) supports Jinja. This is how you parameterize SQL, S3 paths, file names, etc., per execution date.

```python
SnowflakeOperator(
    task_id="daily_aggregate",
    sql="""
        INSERT INTO daily_metrics
        SELECT date_trunc('day', event_ts) AS day,
               COUNT(*) AS events
        FROM events
        WHERE event_ts >= '{{ data_interval_start }}'
          AND event_ts <  '{{ data_interval_end }}'
        GROUP BY 1
    """,
)
```

### Common macros you'll use

```
   {{ ds }}                  execution_date YYYY-MM-DD
   {{ ds_nodash }}           execution_date YYYYMMDD
   {{ ts }}                  execution_date ISO timestamp
   {{ run_id }}              unique ID for this DAG run
   {{ dag.dag_id }}          DAG ID
   {{ task.task_id }}        task ID
   {{ var.value.my_var }}    Airflow Variable value
   {{ conn.snowflake_default.password }}  Connection field
   {{ macros.ds_add(ds, 7) }}             ds + 7 days
   {{ macros.ds_format(ds, '%Y-%m-%d', '%Y/%m/%d') }}
   {{ ti.xcom_pull(task_ids='upstream') }}  pull XCom inline
```

### The big rule

You can render Jinja in a templated field. You **cannot** render Jinja in arbitrary Python code in your DAG file. So this works:

```python
SnowflakeOperator(sql="SELECT * FROM events WHERE day = '{{ ds }}'")  # ✓
```

This does not — `ds` is undefined here, the DAG file is parsed at scheduler time:

```python
date = "{{ ds }}"
print(f"Today is {date}")  # ✗ — string is the literal "{{ ds }}", not the date
```

If you need to do logic on `ds`, do it inside a `PythonOperator` callable (where `context["ds"]` is real) or use the TaskFlow API.

---

## 24.8 XCom — passing data between tasks

XCom (cross-communication) is how tasks pass small data between each other.

### How it works

- A task's return value is stored as an XCom in the metadata DB, keyed by `(dag_id, task_id, execution_date, key)`.
- Downstream tasks pull XCom by `task_ids` (and optionally `key`).
- Default XCom backend is the metadata DB. You can swap in S3/GCS for larger payloads (custom XCom backend).

### TaskFlow style is implicit

```python
@task
def extract():
    return {"rows": 1234}    # automatically stored as XCom

@task
def transform(stats):
    return stats["rows"] * 2  # `stats` arg is the XCom from extract

stats = extract()
result = transform(stats)
```

### Classic style is explicit

```python
def transform(**context):
    data = context["ti"].xcom_pull(task_ids="extract")
    return data * 2
```

### The cardinal XCom rule

**Never put large data in XCom.** XCom is in the metadata DB. A 100MB return value will choke Postgres and hurt scheduler performance for everyone. Rules of thumb:

- **<1KB**: anything (counters, small dicts, status flags)
- **<1MB**: small JSON, list of file paths
- **>1MB**: don't put it in XCom — write to S3 / Snowflake and pass the path/key

```python
# Bad
@task
def extract():
    return huge_dataframe.to_dict()  # MB or GB ends up in metadata DB

# Good
@task
def extract():
    huge_dataframe.to_parquet(f"s3://my-bucket/{uuid.uuid4()}.parquet")
    return f"s3://my-bucket/{uuid.uuid4()}.parquet"  # tiny path
```

---

## 24.9 Connections, Hooks, and Secrets

### Connections

A Connection in Airflow is a named credential bundle — host, port, login, password, schema, extra JSON. Stored encrypted in the metadata DB, identified by `conn_id`.

```python
SnowflakeOperator(
    task_id="...",
    snowflake_conn_id="snowflake_ihs",   # references a Connection
)
```

Where do connections live? Defined via:
1. The Airflow UI (Admin → Connections)
2. Environment variables (e.g. `AIRFLOW_CONN_SNOWFLAKE_IHS=snowflake://...`)
3. A secrets backend — AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager (production-grade)

In a real production system, **never store credentials in the metadata DB**. Use a secrets backend so credentials live in Vault/Secrets Manager and Airflow fetches them at runtime.

### Hooks

A Hook is the Python class that talks to an external system. `S3Hook`, `SnowflakeHook`, `PostgresHook`. You use Hooks inside a `PythonOperator` when you need programmatic access to a system rather than the declarative shape of an operator.

```python
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

@task
def list_keys():
    hook = S3Hook(aws_conn_id="aws_default")
    return hook.list_keys(bucket_name="my-bucket", prefix="incoming/")
```

Operators are mostly thin wrappers around Hooks. Knowing this distinction lets you reach for a Hook when no operator fits your need.

### Variables

Variables are key-value pairs (strings or JSON) stored in the metadata DB, used for runtime configuration that's not a credential. Things like a feature-flag toggle, a list of accounts to process, a slack channel to alert. Accessed in code via `Variable.get("my_key")` or in templates via `{{ var.value.my_key }}` / `{{ var.json.my_key }}`.

---

## 24.10 Branching, trigger rules, and skipping

### Branching with BranchPythonOperator

```python
from airflow.operators.python import BranchPythonOperator

def choose_branch(**context):
    is_weekend = context["data_interval_start"].weekday() >= 5
    return "weekend_path" if is_weekend else "weekday_path"

branch = BranchPythonOperator(task_id="branch", python_callable=choose_branch)
weekend_path = EmptyOperator(task_id="weekend_path")
weekday_path = EmptyOperator(task_id="weekday_path")
join = EmptyOperator(task_id="join", trigger_rule="none_failed_min_one_success")

branch >> [weekend_path, weekday_path] >> join
```

The not-chosen branch gets state `skipped`. The downstream `join` task needs `trigger_rule="none_failed_min_one_success"` because the default rule (`all_success`) would treat the skipped branch as a failure and skip the join too.

### Trigger rules — every senior should know these

```
   all_success                 (default)  All upstream succeeded
   all_failed                              All upstream failed
   all_done                                All upstream finished, regardless of state
   one_success                             At least one upstream succeeded
   one_failed                              At least one upstream failed
   none_failed                             No upstream failed (success or skipped OK)
   none_failed_min_one_success             ↑ AND at least one succeeded (use after branching)
   none_skipped                            No upstream was skipped
   always                                  Always run, no matter what
```

### ShortCircuitOperator

Skip *all* downstream tasks if a condition is false. Useful for "if there's no new data, don't bother running the rest."

```python
def has_new_data(**context):
    return s3_hook.list_keys(bucket="...", prefix=f"incoming/{context['ds']}/") != []

short = ShortCircuitOperator(task_id="check_data", python_callable=has_new_data)
short >> downstream_task
```

If `has_new_data` returns False, every task downstream of `short` is skipped.

---

## 24.11 TaskGroups — organizing complex DAGs

For DAGs with 30+ tasks, a flat layout becomes unreadable. TaskGroups visually nest tasks in the UI and code.

```python
from airflow.utils.task_group import TaskGroup

with DAG(...) as dag:
    with TaskGroup(group_id="extract") as extract_group:
        extract_users = PythonOperator(...)
        extract_orders = PythonOperator(...)
        extract_inventory = PythonOperator(...)

    with TaskGroup(group_id="transform") as transform_group:
        clean_users = PythonOperator(...)
        join_data = PythonOperator(...)

    extract_group >> transform_group >> load_to_warehouse
```

In the UI, `extract` and `transform` show as collapsible group nodes. Inside, you see the individual tasks. This is purely organizational — execution semantics are unchanged.

(SubDAGs are the older, deprecated way to do this. Don't use SubDAGs in new code; they have terrible scheduler ergonomics.)

---

## 24.12 Dynamic task mapping — Airflow 2.3+

Airflow 2.3 added the ability to expand a single task into N parallel task instances at runtime, where N is determined by upstream output. This was the missing primitive for fan-out workloads.

```python
@task
def list_files():
    return ["file1.parquet", "file2.parquet", "file3.parquet"]

@task
def process_file(filename: str):
    print(f"Processing {filename}")

files = list_files()
process_file.expand(filename=files)   # creates 3 task instances at runtime
```

Each call to `process_file` runs as its own task instance with a `map_index` (0, 1, 2). This replaces the awkward "create N tasks at parse time using a Python loop" pattern, which broke when N varied between runs.

---

## 24.13 Datasets and data-aware scheduling — Airflow 2.4+

Traditional Airflow schedules on time. Airflow 2.4 added **datasets**: schedule a DAG to run when a specific dataset is updated by an upstream DAG.

```python
from airflow import Dataset

raw_data = Dataset("s3://my-bucket/raw/")
features = Dataset("s3://my-bucket/features/")

# DAG 1 produces raw_data
@dag(schedule="@daily", start_date=...)
def ingest():
    @task(outlets=[raw_data])
    def write_raw(): ...
    write_raw()

# DAG 2 runs whenever raw_data is updated
@dag(schedule=[raw_data], start_date=...)
def transform():
    @task(outlets=[features])
    def build_features(): ...
    build_features()

# DAG 3 runs whenever features is updated
@dag(schedule=[features], start_date=...)
def train():
    ...
```

This replaces `ExternalTaskSensor` for many use cases — the dependency is a property of the data, not a coupling between DAGs. For an MLOps platform with shared feature pipelines this is genuinely useful.

---

## 24.14 Executors compared — and how to choose

```
   ┌───────────────────────────────────────────────────────────────────┐
   │  Executor              When to use                                │
   ├───────────────────────────────────────────────────────────────────┤
   │  Sequential            Local dev only. SQLite.                    │
   │                                                                   │
   │  Local                 Single machine, small parallelism.         │
   │                                                                   │
   │  Celery                Distributed workers, low task overhead.    │
   │                        Production standard for years. Needs       │
   │                        Redis or RabbitMQ as broker.               │
   │                                                                   │
   │  Kubernetes            Each task = one pod. Cleanly isolated      │
   │                        environments per task. Heavier startup     │
   │                        (~5-30s per task) but no shared deps.      │
   │                                                                   │
   │  CeleryKubernetes      Mix: short tasks → Celery (low overhead),  │
   │                        heavy/isolated tasks → K8s (per-pod).      │
   └───────────────────────────────────────────────────────────────────┘
```

**The trade-off summary:** Celery for low overhead, Kubernetes for isolation and resource flexibility. For a multi-team Airflow with diverse Python dependencies per team, KubernetesExecutor avoids the dependency hell. For a single-team setup with consistent deps, CeleryExecutor is faster per task.

---

## 24.15 Deployment options

```
   ┌─────────────────────────────────────────────────────────────────┐
   │  Self-hosted              Docker Compose, Helm chart, EKS/GKE.  │
   │                           Maximum control, maximum operational  │
   │                           burden.                               │
   │                                                                 │
   │  AWS MWAA                 Managed Workflows for Apache Airflow. │
   │                           AWS handles infra; you bring DAGs.    │
   │                           VPC integration, slow upgrades.       │
   │                                                                 │
   │  GCP Cloud Composer       GCP's managed Airflow. Tight GCP      │
   │                           integration. Pricing can spike.       │
   │                                                                 │
   │  Astronomer (Astro)       Commercial managed Airflow. Best      │
   │                           developer experience, REST API,       │
   │                           native CI/CD.                         │
   └─────────────────────────────────────────────────────────────────┘
```

For Avrioc's likely setup, MWAA on AWS would be the path of least resistance. For multi-cloud or AWS-heavy with Snowflake elsewhere, Astronomer earns the cost. For full control, self-host on EKS with the official Helm chart.

---

## 24.16 Best practices and anti-patterns

These are the things that separate someone who's used Airflow from someone who's run Airflow in production. Mention any of these unprompted in an interview and you signal real depth.

### DO

1. **Idempotent tasks.** Re-running a task with the same inputs should produce the same outputs and not corrupt state. Use `INSERT OVERWRITE` or `MERGE` patterns in SQL. Use `if exists, delete then write` in Python.

2. **Use `data_interval_start` / `data_interval_end`** instead of `datetime.now()` inside DAGs. Otherwise backfills produce wrong data, and the DAG drifts.

3. **Keep DAG files lightweight.** The scheduler re-parses them every 30 seconds. Heavy imports, network calls, or DB queries at module level slow down the entire scheduler. Wrap such code inside task functions.

4. **Use connections and variables** for configuration. Never hard-code credentials, hosts, paths.

5. **Set `catchup=False`** unless you specifically want backfills.

6. **Set retries explicitly.** Default is 0. Make it 2-3 with exponential backoff.

7. **Use `mode="reschedule"` or deferrable sensors** for any long wait.

8. **Set SLA on critical tasks** so you get alerted when a task takes longer than expected.

9. **Use TaskFlow API for new DAGs.** The XCom plumbing is implicit and the code is shorter.

10. **Tag your DAGs** for organization in the UI (`tags=["mlops", "ihs", "daily"]`).

### DON'T

1. **Don't use `datetime.now()` in a DAG file.** It's evaluated every parse — your `start_date` will keep drifting.

2. **Don't put large data in XCom.** Write to S3/Snowflake and pass paths.

3. **Don't run heavy compute inside Airflow workers.** Workers are orchestrators. Push compute to Snowflake, SageMaker, Spark, K8s pods.

4. **Don't use SubDAGs.** Use TaskGroups instead.

5. **Don't fan out by writing a Python `for` loop that creates N operators at parse time.** This was a common pre-2.3 pattern; use dynamic task mapping instead.

6. **Don't depend on side effects between tasks via the local filesystem.** Workers may run on different machines. Use S3/GCS for any inter-task data.

7. **Don't share Python imports inside operator constructors.** Heavy module-level imports run on every parse.

8. **Don't set `depends_on_past=True` on long pipelines.** A single old failure blocks every future run.

9. **Don't trigger DAG runs from within tasks** unless you really need to. `TriggerDagRunOperator` is fine; `BashOperator` calling `airflow dags trigger` is a bad pattern.

10. **Don't use the metadata DB as application state.** It's for Airflow's own bookkeeping.

---

## 24.17 Testing DAGs

Airflow tests come in three layers.

### Layer 1 — DAG integrity

Every DAG file must parse cleanly. CI should run:

```python
import pytest
from airflow.models import DagBag

@pytest.fixture
def dagbag():
    return DagBag(dag_folder="dags/", include_examples=False)

def test_dag_loaded(dagbag):
    assert dagbag.import_errors == {}, f"Errors: {dagbag.import_errors}"

def test_dag_count(dagbag):
    assert len(dagbag.dags) > 0
```

### Layer 2 — Unit tests for callables

Anything in a `PythonOperator` is just a Python function — test it like any function.

```python
def test_quality_check():
    df = pd.DataFrame({"patient_id": [1, 2, None]})
    with pytest.raises(ValueError):
        quality_check_fn(df)
```

### Layer 3 — Integration tests

Run a task or whole DAG locally against test data:

```bash
airflow dags test ihs_feature_pipeline 2026-04-30
airflow tasks test ihs_feature_pipeline extract 2026-04-30
```

`airflow dags test` runs the whole DAG synchronously without using the scheduler. `airflow tasks test` runs one task in isolation. Both ignore retries and dependencies — perfect for local debugging.

---

## 24.18 Interview Q&A — full narrative answers

### Q1. What is Apache Airflow at a high level?

Airflow is a workflow orchestrator. You describe a pipeline as a DAG of tasks in Python, and Airflow's scheduler runs those tasks in dependency order, retries on failure, alerts when things break, and exposes everything in a UI. Airflow does not move data itself — it tells other systems like Snowflake or SageMaker to move data, and waits for them. The mental model I use: Airflow is the conductor of an orchestra. It doesn't play any instrument. Its job is to know who plays when, watch for cues, and stop the next section if someone misses theirs.

### Q2. Walk me through Airflow's architecture.

There are five components. The scheduler continuously parses DAG files and decides which task instances should run now based on schedules and upstream state. The metadata database — usually Postgres — stores DAG definitions, every task instance ever run, XComs, connections, variables. The webserver gives you the UI and REST API by querying the metadata DB; it doesn't execute tasks. The executor is the dispatch layer that decides where tasks run — Celery for distributed workers, Kubernetes for one pod per task. Workers are the processes that actually execute task code. The DAGs folder is a directory the scheduler watches for Python files. The core thing to understand is that the scheduler and metadata DB are the source of truth; everything else is replaceable.

### Q3. What's the difference between a DAG, a task, an operator, and a task instance?

A DAG is the workflow — the whole pipeline as a graph. An operator is a template for what kind of work a task does — `PythonOperator`, `SnowflakeOperator`, `S3KeySensor`. A task is an instance of an operator with specific configuration in a specific DAG — when I write `extract = PythonOperator(task_id="extract", python_callable=extract_fn)`, that's a task. A task instance is a specific run of a task at a specific execution date — `extract` running on `2026-04-29` is one task instance, on `2026-04-30` is another. Each task instance has its own state — queued, running, success, failed, skipped — tracked in the metadata DB.

### Q4. What kinds of operators are there, and which have you used?

Operators split into three families. Action operators do work — `PythonOperator`, `SnowflakeOperator`, `KubernetesPodOperator`. Sensors wait for a condition — `S3KeySensor`, `ExternalTaskSensor`, `SqlSensor`. Transfer operators move data between systems — `S3ToSnowflakeOperator`, `S3ToRedshiftOperator`. There's also a control-flow category — `EmptyOperator`, `BranchPythonOperator`, `ShortCircuitOperator` — that shapes the DAG without doing real work. At ResMed for the IHS feature pipeline I used `S3KeySensor` to wait for raw clinical data, `SnowflakeOperator` for set-based transforms, `PythonOperator` for custom logic and quality checks, `SageMakerProcessingOperator` for heavy ML work, and `BranchPythonOperator` for model-specific paths. The principle was: each operator type at the right layer — Airflow does dependency and retries, the heavy compute lives in Snowflake or SageMaker, and Python is reserved for glue.

### Q5. What's the difference between poke mode and reschedule mode for sensors?

In poke mode, the default, the sensor task holds a worker slot the entire time it's waiting. The sensor wakes up every poke interval, checks the condition, and goes back to sleep — but the slot is occupied throughout. For long waits, this pins worker capacity to do nothing. Reschedule mode releases the worker slot between checks; the task instance gets recreated at each check interval. For any wait longer than a few minutes, reschedule mode is the right choice. Even better in modern Airflow are deferrable operators, which yield to a separate process called the triggerer — those take zero worker capacity during the wait. I'd reach for deferrable sensors first today, reschedule second, poke only for sub-minute waits.

### Q6. What happens if the metadata DB goes down?

The scheduler can't read or write task state, the webserver can't render anything, and workers can't pull new tasks. Existing in-flight tasks may continue if they've already pulled their work, but the moment they try to update state, they fail. The metadata DB is single-point-of-failure for Airflow, which is why you back it up religiously and run it in HA mode in production. AWS RDS multi-AZ Postgres is the standard choice. Losing the DB without a backup means losing every task history — DAG definitions you can rebuild from code, but XComs, run history, manual marks like clearing or pausing — those are gone.

### Q7. How does Airflow's scheduling work — what's `execution_date`?

This is the trickiest concept in Airflow. The execution_date is not when the task ran — it's the start of the data interval the run is processing. So a daily DAG with `execution_date=2026-04-29` is processing data from April 29 midnight to April 30 midnight, and it actually fires at the end of that interval — April 30 midnight. The interval-based model is correct because most pipelines process "yesterday's data," but it confuses everyone the first time. Airflow 2.2 renamed the concept to `data_interval_start` and `data_interval_end` to make the semantics clearer. In templates I prefer `{{ data_interval_start }}` over `{{ ds }}` for new code, because it's unambiguous.

### Q8. How do tasks pass data to each other?

Through XCom — cross-communication. A task's return value is automatically stored as an XCom in the metadata DB, keyed by DAG ID, task ID, execution date. Downstream tasks pull XCom either explicitly with `ti.xcom_pull(task_ids="upstream")`, or implicitly via the TaskFlow API, which infers the data flow from Python function signatures. The cardinal rule: never put large data in XCom. The default backend is the metadata DB, and large XComs choke Postgres. For anything over a few hundred kilobytes, write to S3 or Snowflake and pass the path through XCom instead. You can also configure a custom XCom backend that stores values in S3, but I prefer being explicit.

### Q9. What's the difference between the classic operator API and the TaskFlow API?

Classic uses `PythonOperator(task_id=..., python_callable=fn)` and explicit `xcom_pull` calls. TaskFlow uses the `@task` decorator on a function, and the data flow is inferred from function calls — `transform(extract())` automatically wires an XCom from `extract` to `transform`. TaskFlow is shorter, more Pythonic, and the dependency graph reads like normal Python code. For new DAGs I default to TaskFlow unless I need a non-Python operator like `SnowflakeOperator`, in which case I mix — TaskFlow for Python tasks, classic for the rest.

### Q10. What are connections and how should you manage credentials?

A connection is a named credential bundle — host, port, login, password, schema, extra JSON — stored encrypted in the metadata DB and identified by `conn_id`. Operators reference connections by ID rather than embedding credentials. In production, never store credentials in the metadata DB; use a secrets backend like AWS Secrets Manager, HashiCorp Vault, or GCP Secret Manager. Airflow will fetch credentials from the backend at task runtime, so credentials live in the secrets manager and Airflow's metadata DB only stores connection IDs.

### Q11. What's `catchup`, and when do you turn it off?

`catchup=True` means: if your DAG's start_date is in the past and the DAG hasn't run for that period, Airflow will backfill every missed interval, one run per interval, until it catches up to now. This is useful for new DAGs replacing old pipelines where you legitimately want history rebuilt. It's also dangerous — turning on a new DAG with a 6-month-old start_date can spawn 180 simultaneous runs that hammer downstream systems. For new DAGs, I default to `catchup=False` plus an explicit `start_date` close to now, and I backfill manually with `airflow dags backfill` if I need history.

### Q12. What are trigger rules?

Trigger rules control when a task runs based on the state of its upstream tasks. The default is `all_success` — run only if every upstream task succeeded. Other rules: `all_failed`, `all_done` (regardless of state), `one_success`, `one_failed`, `none_failed` (success or skipped is fine), `none_failed_min_one_success` (the right rule after a branch, where one branch is skipped), `always`. Trigger rules become important after `BranchPythonOperator` because the not-chosen branch's tasks are skipped, and the default `all_success` rule would skip the join task downstream too. The fix is `trigger_rule="none_failed_min_one_success"` on the join.

### Q13. How does dynamic task mapping work?

Pre-2.3, fanning out N parallel tasks where N depended on runtime data was awkward — you'd write a Python loop in the DAG file that created N operators, but if N varied between runs you got DAG schema changes that broke history. Airflow 2.3 added `expand`, which creates N task instances at runtime based on the output of an upstream task. So `process_file.expand(filename=list_files())` produces one task instance per file. Each instance gets a `map_index` (0, 1, 2, ...), and they run in parallel up to your concurrency limit. This is the right primitive for fan-out — I'd use it for things like "process each S3 prefix" or "train one model per region."

### Q14. What are datasets and data-aware scheduling?

Airflow 2.4 added datasets, which let one DAG schedule based on another DAG's output rather than time. You declare a dataset like `Dataset("s3://my-bucket/features/")`, mark a producing task with `outlets=[dataset]`, and a consuming DAG with `schedule=[dataset]` runs whenever that dataset is updated. This replaces `ExternalTaskSensor` for many use cases — the dependency is a property of the data, not a coupling between DAGs. For a multi-team MLOps platform where the feature DAG is shared but each team has their own training DAG, datasets give you clean ownership and explicit data contracts.

### Q15. What's the difference between Celery and Kubernetes executors?

CeleryExecutor uses a pool of long-running Celery workers connected to a Redis or RabbitMQ broker. Tasks have low overhead — a few hundred milliseconds — because the worker process is already running. The downside is that all tasks share the worker's Python environment, so dependency conflicts are real. KubernetesExecutor spawns a new pod per task, with whatever container image the task specifies. Each task is fully isolated; one team's PyTorch 2.0 doesn't conflict with another team's TensorFlow 2.18. The downside is per-task overhead — pod startup is 5 to 30 seconds. The CeleryKubernetesExecutor combines them: most tasks go to Celery for low overhead, specific tasks marked with the `kubernetes` queue go to K8s pods for isolation. That hybrid is what I'd default to for a multi-team production deployment.

### Q16. How do you make tasks idempotent?

Idempotency means running a task twice with the same inputs produces the same outputs without corrupting state. For SQL, use `INSERT OVERWRITE` or `MERGE` patterns instead of plain `INSERT`. For S3 writes, use deterministic key paths keyed on `data_interval_start` so a re-run overwrites the same key rather than appending. For external API calls, design the API to accept an idempotency key — the task ID plus execution date is a natural choice. The reason idempotency matters: Airflow retries failed tasks, sometimes you'll backfill, sometimes you'll clear and re-run — every one of those re-executes the task, and if the task isn't idempotent, you get duplicate writes, double-counting, or corrupted state.

### Q17. What's the worst Airflow gotcha you've encountered?

Top of my list: someone uses `datetime.now()` inside a DAG file at module level, instead of using `data_interval_start` or `ds`. The DAG file gets parsed every 30 seconds by the scheduler, and `datetime.now()` evaluates at parse time, so the DAG's start_date or schedule keeps drifting. Backfills produce wrong data because the wrong "now" is captured. The fix is always: use Airflow's templated variables or read them from the task context inside a callable. Second worst: heavy module-level imports or DB queries in DAG files — they slow down the entire scheduler, not just that DAG. Third: setting `depends_on_past=True` on a long pipeline and then having one historical run fail forever — every future run is blocked because the past failure is unresolved.

### Q18. How would you monitor and alert on an Airflow pipeline?

Three layers. First, task-level — set `email_on_failure=True` and `retries=3` per task, plus `on_failure_callback` for custom alerting like Slack. Second, SLA-based — set an `sla` on critical tasks so Airflow alerts when a task takes longer than expected. Third, system-level — instrument the Airflow scheduler and webserver with Prometheus, alert on scheduler heartbeat lag, task queue depth, and DAG processing time. For a production deployment I'd also push DAG-level success/failure events to PagerDuty and have a Grafana dashboard with task duration histograms. The principle is: don't rely on the Airflow UI as your only monitoring; integrate alerting into the same paging system as the rest of your infrastructure.

### Q19. How do you test DAGs?

Three layers. DAG integrity tests in CI — load every DAG file via `DagBag` and assert no import errors. Unit tests for the Python callables — anything in a `PythonOperator` is just a function and can be tested directly with pytest. Integration tests via `airflow dags test <dag> <date>`, which runs the whole DAG synchronously without the scheduler, ignoring retries and waiting — perfect for local validation. For a production team I'd also run nightly integration runs in a staging Airflow against test data, separate from production, and gate production deploys on those runs passing.

### Q20. Why Airflow over alternatives like Step Functions, Prefect, or Dagster?

Airflow's strength is maturity and ecosystem — hundreds of operators, broad community, every cloud has managed Airflow, every data team has someone who knows it. Its weakness is the metadata DB single-point-of-failure, the awkward scheduling semantics, and the parse-everything-every-30-seconds scheduler. Step Functions are excellent for AWS-native workflows but lock you in and use JSON state machines that are painful for complex flows. Prefect 2.0 fixed many Airflow pain points — better local dev experience, async-native, dynamic by design — but the ecosystem is smaller. Dagster has the cleanest software-engineering story, with strong typing, asset-based scheduling, and observability built in — for a greenfield ML platform today I'd genuinely consider Dagster. At ResMed we were already on Airflow when I joined, and the migration cost wasn't justified by the upside. I'd evaluate the same call differently for new projects.

### Q21. How would you orchestrate a feature pipeline with hundreds of features across a dozen models?

I'd shape it as three layers. Bottom layer: a shared "raw to features" DAG that produces a versioned feature dataset, declared as an Airflow Dataset (Airflow 2.4+). It's daily, idempotent, lands features in Snowflake plus an S3 mirror. Middle layer: per-model DAGs that schedule on the feature Dataset — they trigger automatically when the feature dataset is updated, run model-specific training and validation, and write predictions. Top layer: a meta-DAG that aggregates all models' freshness signals and alerts if any model's predictions are stale. The key principles: shared compute lives in Snowflake (set-based transforms are free in column stores), per-model logic stays in per-model DAGs (clean ownership), and inter-DAG coupling uses Datasets, not ExternalTaskSensors. That's basically the IHS architecture I built at ResMed, abstracted.

### Q22. How does retry work in Airflow?

Set `retries` on a task or in `default_args`. When a task fails, Airflow waits `retry_delay`, then retries. With `retry_exponential_backoff=True`, each retry waits longer than the last, up to `max_retry_delay`. Retries are only for transient failures — bad logic doesn't get better with retries. Use `retry_delay=timedelta(minutes=5)` and `retries=3` as a reasonable default for I/O-shaped tasks. For tasks calling out to external APIs that have their own quotas or backoff requirements, you may need much longer delays. The retry counter is per task instance — a task that retried twice and then succeeded shows as success, not as success-after-retries, in the UI but the history is in the logs.

### Q23. What's a Hook and when do you use one?

A Hook is the Python class that talks to an external system — `S3Hook`, `SnowflakeHook`, `PostgresHook`. Operators are mostly thin wrappers around Hooks. You reach for a Hook directly inside a `PythonOperator` when no operator fits your need — for example, when you need to programmatically list files, do a custom transformation, or inspect a result. Hooks read connection details from the connection ID you pass them, so you don't have to manage credentials. The mental model: operators are declarative, Hooks are imperative. When you need imperative access to an external system, that's a Hook.

### Q24. How do you do a backfill?

`airflow dags backfill -s 2026-01-01 -e 2026-01-31 my_dag`. Airflow runs every interval between the start and end dates as a separate DAG run. Concurrency is controlled by `--max-active-runs` and the DAG's `max_active_runs` setting. For long backfills, set `--max-active-runs=4` or so to avoid hammering downstream systems. Production backfills should always go through a CI/CD-controlled mechanism, not someone's terminal, because backfills can rewrite a lot of data fast. At ResMed we wrote a wrapper script that took a date range, pre-flighted Snowflake to estimate row counts, and got approval before running.

### Q25. What's `depends_on_past` and when do you use it?

`depends_on_past=True` means a task instance only runs if the same task succeeded in the previous DAG run. Useful for serial pipelines where each day's output depends on yesterday's output — for example, a running balance calculation that updates yesterday's row. The risk is that a single old failure blocks every future run forever, because the chain is broken. Don't set it casually. If you genuinely need it, also set `wait_for_downstream` if you need a stricter chain, and have an alerting plan for the case where past failures need explicit clearing.

### Q26. Walk me through the Airflow setup at ResMed.

We ran self-hosted Airflow on AWS using the official Helm chart on EKS, with KubernetesExecutor for task isolation since multiple data science teams shared the cluster. Postgres on RDS multi-AZ for the metadata DB. Secrets came from AWS Secrets Manager. DAGs were in a Git repo, synced into the Airflow scheduler pod via a sidecar that ran `git pull` every 60 seconds. CI tested every PR by spinning up a temporary Airflow, parsing all DAGs, running unit tests on callables, and running `airflow dags test` for any DAG that changed. The IHS-specific pattern was a shared raw-to-features DAG that produced a Snowflake feature dataset, plus per-model DAGs that scheduled off that dataset using `ExternalTaskSensor` (Datasets came in 2.4 after we'd already wired the architecture). Operators were a mix of `S3KeySensor`, `SnowflakeOperator`, `PythonOperator` for glue, and `SageMakerProcessingOperator` for heavy ML work. Drift dashboards used a separate Airflow DAG that ran daily, computed PSI and KS metrics, and pushed to Datadog.

### Q27. How would you scale Airflow to handle 10,000 DAGs?

The bottleneck at scale is the scheduler — it parses every DAG file and updates state for every task instance. Mitigations: enable multiple schedulers (Airflow 2.0+ supports HA scheduler), increase `parsing_processes` and `min_file_process_interval`, partition DAGs into multiple Airflow deployments by team or domain, push old DAG runs out of the metadata DB via the cleanup CLI. For 10K DAGs I'd seriously consider sharding into multiple Airflow deployments rather than one monster. The KubernetesExecutor scales nearly linearly in worker capacity (each task gets a pod), so the limit becomes scheduler throughput and metadata DB load. Postgres connection pooling via PgBouncer is non-negotiable at this scale.

### Q28. What's the difference between `schedule="@daily"` and `schedule="0 0 * * *"`?

Functionally identical — both mean "run every day at midnight UTC." Airflow exposes preset names like `@daily`, `@hourly`, `@weekly`, `@monthly` as syntactic sugar over cron expressions. There's also `schedule=timedelta(days=1)` which means "run every 24 hours from the start_date" — subtly different from `@daily`, because it's tied to a delta from the start, not to wall-clock midnight. For most ML pipelines I want wall-clock alignment ("data through midnight UTC"), so I prefer `@daily` or the explicit cron `0 0 * * *`.

### Q29. How would you trigger an Airflow DAG from outside Airflow?

Three options. First, the REST API — `POST /api/v1/dags/<dag_id>/dagRuns`. Best for programmatic triggering from another service. Second, the CLI — `airflow dags trigger <dag_id>`. Best for interactive or shell-script triggers. Third, datasets — if the trigger is "an upstream dataset got updated," declare an Airflow Dataset and let the scheduler do it. For event-driven triggers like "S3 file arrived," I'd rather use S3 → SNS → Lambda → Airflow REST API than poll with a sensor, because the lag is minutes not seconds.

### Q30. What's the worst-case failure mode of an Airflow deployment, and how do you mitigate?

The worst case is silent data corruption: a task succeeds but produces wrong output, retries don't catch it because the failure isn't a Python exception, and the bad data flows downstream. Mitigations: every meaningful task should validate its output — row counts, schema checks, freshness checks — and fail loudly. Use Great Expectations or dbt tests on data outputs. Use SQL constraints where possible. Add a separate "validate" task downstream of every "extract/transform" task. The principle: trust nothing about your tasks succeeding silently. Failure that fails loud is debugged in hours; failure that's silent is debugged in weeks, after someone notices the model's accuracy dropped.

---

## 24.19 Live coding scenarios

If they ask you to write or extend a DAG live, here are the patterns to drill.

### Scenario 1 — Daily ETL with quality gate

Build a DAG that waits for an S3 key, runs SQL on Snowflake, validates the output, and triggers a downstream training job.

```python
from airflow.decorators import dag, task
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import pandas as pd

@dag(
    dag_id="ihs_daily_etl",
    start_date=datetime(2026, 4, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={"retries": 2, "owner": "sachin"},
    tags=["ihs", "etl"],
)
def ihs_daily_etl():
    wait_data = S3KeySensor(
        task_id="wait_for_data",
        bucket_name="ihs-raw",
        bucket_key="incoming/{{ ds }}/data.parquet",
        mode="reschedule",
        poke_interval=300,
        timeout=60 * 60 * 6,
    )

    transform = SnowflakeOperator(
        task_id="build_features",
        sql="""
            INSERT OVERWRITE INTO features_daily
            SELECT patient_id,
                   AVG(metric) OVER (PARTITION BY patient_id ORDER BY ts ROWS 6 PRECEDING) AS avg_7d
            FROM events
            WHERE ts >= '{{ data_interval_start }}'
              AND ts <  '{{ data_interval_end }}'
        """,
        snowflake_conn_id="snowflake_ihs",
    )

    @task
    def quality_check():
        from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
        hook = SnowflakeHook(snowflake_conn_id="snowflake_ihs")
        result = hook.get_first("SELECT COUNT(*) FROM features_daily WHERE date = '{{ ds }}'")
        if result[0] < 1000:
            raise ValueError(f"Too few rows: {result[0]}")
        return result[0]

    done = EmptyOperator(task_id="done")

    wait_data >> transform >> quality_check() >> done

dag = ihs_daily_etl()
```

### Scenario 2 — Dynamic fan-out per region

```python
@task
def list_regions():
    return ["us-east-1", "eu-west-1", "ap-southeast-1"]

@task
def process_region(region: str, **context):
    print(f"Processing {region} for {context['ds']}")

regions = list_regions()
process_region.expand(region=regions)
```

This produces three task instances at runtime, each with its own logs and retry behavior.

### Scenario 3 — Branch on data freshness

```python
from airflow.operators.python import BranchPythonOperator

def choose_path(**context):
    return "run_full_backfill" if context["ti"].xcom_pull(task_ids="check_freshness") else "skip_backfill"

check = PythonOperator(task_id="check_freshness", python_callable=is_stale_fn)
branch = BranchPythonOperator(task_id="branch", python_callable=choose_path)
backfill = BashOperator(task_id="run_full_backfill", bash_command="...")
skip = EmptyOperator(task_id="skip_backfill")
join = EmptyOperator(task_id="join", trigger_rule="none_failed_min_one_success")

check >> branch >> [backfill, skip] >> join
```

---

## 24.20 The morning-of cheatsheet for Airflow

If you only have two minutes to refresh:

```
   Operators:        Action (Python, Bash, SQL), Sensors, Transfer, Control flow
   Sensors mode:     poke (holds slot), reschedule (releases), deferrable (best)
   XCom rule:        Small data only; large data → S3 + path in XCom
   execution_date:   Start of data interval, NOT when task ran
   ds = "{{ ds }}":  Templated date string YYYY-MM-DD
   catchup:          Default True, almost always set False on new DAGs
   Idempotency:      INSERT OVERWRITE / MERGE / deterministic keys
   Retries:          Default 0, set 2-3 with retry_delay
   Trigger rule:     none_failed_min_one_success after BranchPythonOperator
   Executors:        Celery (low overhead), K8s (isolated), CeleryK8s (hybrid)
   Dynamic mapping:  .expand() for runtime fan-out (Airflow 2.3+)
   Datasets:         schedule=[Dataset("s3://...")] for data-aware (2.4+)
```

The interview-winning sentence to keep ready: "Airflow is the conductor — orchestration only. Heavy compute lives in Snowflake or SageMaker. The principle is each operator type at the right layer."

---

End of Chapter 24. Continue back to **[Chapter 00 — Master Index](00_index.md)** to navigate other chapters.
