# Chapter 32 — Claude Deep Dive: ML Workspace, Tool Integrations, and Solution Architecture Certification

> **Why this chapter exists:** Your TrueBalance bullet — "Architected an AI-powered ML workspace assistant on Claude, integrating Jira, GitHub, Athena, and Jenkins" — is one of the strongest items on your resume, and it's exactly the kind of project an Avrioc interviewer will dig into. This chapter covers the architecture of that system in depth, plus the Claude API and Claude-specific patterns that distinguish you from candidates who only know OpenAI. It also doubles as preparation for the Anthropic Claude Solution Architecture certification, which you mentioned you're preparing for. The mix of practical implementation and certification-grade theory makes this one of the longest and most important chapters in the pack.

---

## 32.1 The mental model — what a Claude-powered ML workspace actually is

The ML workspace at TrueBalance is best understood as a **conversational developer platform** sitting in front of the team's tooling. Instead of every ML engineer manually opening Jira to find their ticket, then GitHub to clone a repo, then SSHing into an EC2 to provision a GPU, then running an Athena query, then triggering a Jenkins build — they talk to Claude in a chat interface and Claude does all of those steps via tool calls. The workspace is not "a Claude wrapper"; it's a coordinated system where Claude is the conductor and the integrations are the instruments.

The block diagram:

```
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                          ENGINEER (chat UI)                              │
   │      "I'm picking up TICKET-1234. Spin up a GPU, clone the repo, run     │
   │       last week's training pipeline against fraud-v3 features"           │
   └──────────────────────────────┬───────────────────────────────────────────┘
                                  │  WebSocket (streaming)
                                  ▼
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                       Workspace Orchestrator (FastAPI)                   │
   │   - Loads conversation context from Postgres                             │
   │   - Calls Claude with the engineer's message + tool definitions          │
   │   - Receives Claude's response: text or tool_use blocks                  │
   │   - Executes tool_use blocks against the appropriate integration         │
   │   - Appends tool_result blocks to the conversation                       │
   │   - Loops until Claude returns text without tool_use                     │
   │   - Streams final text + intermediate tool-progress events to UI         │
   └────┬───┬───┬───┬───┬───────────────────────────────────────────────────┬─┘
        │   │   │   │   │                                                   │
        ▼   ▼   ▼   ▼   ▼                                                   ▼
   ┌──────┐┌──────┐┌──────┐┌──────────┐┌──────────┐                ┌──────────────┐
   │Jira  ││GitHub││Athena││Jenkins   ││Slack     │                │ Claude API   │
   │Tools ││Tools ││Tools ││Tools     ││Tools     │                │ (Sonnet 4 /  │
   │      ││      ││      ││          ││          │                │  Opus 4.7)   │
   │get_  ││clone_││run_  ││trigger_  ││post_     │                │              │
   │ticket││repo  ││query ││build     ││message   │                │ tool_use +   │
   │tx_   ││create││get_  ││get_      ││get_      │                │ vision +     │
   │status││PR    ││result││logs      ││thread    │                │ caching      │
   └──────┘└──────┘└──────┘└──────────┘└──────────┘                └──────────────┘

   ┌──────────────────────────────────────────────────────────────────────────┐
   │                       INFRASTRUCTURE LAYER                               │
   │                                                                          │
   │   EC2 GPU/CPU Provisioner Lambda (boto3 EC2 + EFS-shared state)          │
   │   Git Worktree Manager (per-ticket isolation, 3-method EBS lifecycle)    │
   │   AWS IAM (per-engineer least-privilege role)                            │
   │   CloudWatch Logs / X-Ray traces                                         │
   └──────────────────────────────────────────────────────────────────────────┘
```

The workspace is fundamentally a **multi-tool agent**: every external system is exposed to Claude as a set of tools it can call, and Claude decides when to call which based on the engineer's intent.

---

## 32.2 The Claude API — what's different from OpenAI

If you're coming from the OpenAI world, the Claude API has the same conceptual shape (messages, tools, streaming) but distinct field names and conventions. Knowing the differences cold is a strong interview signal.

### Core request shape

```python
import anthropic

client = anthropic.Anthropic(api_key="...")

response = client.messages.create(
    model="claude-sonnet-4-6",   # or claude-opus-4-7, claude-haiku-4-5
    max_tokens=2048,
    system="You are an ML workspace assistant. Be concise.",
    messages=[
        {"role": "user", "content": "Pull the latest fraud model metrics from Athena."}
    ],
    tools=[...],
    tool_choice={"type": "auto"},
)
```

Note: `system` is a **separate top-level parameter**, not a message in the messages list. This is one of the cleanest differences from OpenAI, where the system prompt is the first message. Claude treats the system prompt as a distinct, privileged context — better for prompt-injection resistance.

### Messages, content blocks, and the response shape

OpenAI's content is a string. Claude's content is a **list of blocks** — text, image, tool_use, tool_result, document. This makes multi-modal and tool-heavy conversations more structured.

```python
# A simple text-only assistant message
{
    "role": "assistant",
    "content": [{"type": "text", "text": "Here is the result..."}]
}

# An assistant message with a tool call
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Let me query Athena for that."},
        {
            "type": "tool_use",
            "id": "toolu_01abc",
            "name": "athena_query",
            "input": {"sql": "SELECT * FROM fraud_metrics WHERE month = '2026-04'"}
        }
    ]
}

# A user message that returns tool results
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01abc",
            "content": json.dumps({"rows": 1234, "auc": 0.92})
        }
    ]
}
```

Three things to notice. First, `tool_use_id` is the link between the call and its result (analogous to OpenAI's `tool_call_id`). Second, `tool_result` blocks live inside a `user` message, not a separate `tool` role like OpenAI uses. Third, you can include text and tool_use in the same assistant message — Claude often narrates "I'll check Athena" before emitting the actual `tool_use` block, which is great for streaming UX.

### Tool definitions

```python
tools = [
    {
        "name": "athena_query",
        "description": (
            "Run a SQL query against the AWS Athena data warehouse. "
            "Use this to fetch ML metrics, training data summaries, "
            "feature distribution stats. The query must be read-only "
            "(SELECT statements only)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query. Must start with SELECT."
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to return. Default 100, max 10000.",
                    "default": 100
                }
            },
            "required": ["sql"]
        }
    },
    {
        "name": "create_github_pr",
        "description": "Create a pull request on a GitHub repo.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name format"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "branch": {"type": "string"},
                "base": {"type": "string", "default": "main"}
            },
            "required": ["repo", "title", "body", "branch"]
        }
    }
]
```

The schema field is **`input_schema`**, not `parameters` (OpenAI). The model uses the `description` to decide *when* to call the tool, and `input_schema` (a JSON Schema) to decide *how* to fill in arguments.

### tool_choice options

| Value | Behavior |
|-------|----------|
| `{"type": "auto"}` | Model decides whether to call a tool |
| `{"type": "any"}` | Model is forced to call at least one tool |
| `{"type": "tool", "name": "X"}` | Model is forced to call this specific tool |
| `{"type": "auto", "disable_parallel_tool_use": true}` | Auto, but only one tool per turn |

The default is `auto`. For a workspace assistant, `auto` is right. For a strict SQL agent that must always query, `any` is right.

### Streaming

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    system="...",
    messages=[...],
    tools=[...],
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            block_type = event.content_block.type
            # text or tool_use
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                # New text token — push to UI
                push_to_websocket({"type": "token", "content": event.delta.text})
            elif event.delta.type == "input_json_delta":
                # Tool input is being streamed too — assemble it
                assemble_tool_input(event.delta.partial_json)
        elif event.type == "message_stop":
            # Done
            break
```

Streaming events are richer than OpenAI's. You see `content_block_start`, deltas (both text and tool input JSON), and stop events. The implementation in the workspace assembles each block as it streams.

### The full tool-calling loop with Claude

```python
def run_turn(conversation, user_message):
    conversation.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=conversation,
            tools=TOOLS,
        )
        conversation.append({
            "role": "assistant",
            "content": [block.model_dump() for block in response.content]
        })

        if response.stop_reason == "tool_use":
            # Execute every tool_use block in this assistant message
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })
            conversation.append({"role": "user", "content": tool_results})
            continue   # loop back

        # stop_reason was "end_turn" — final text response
        return response
```

`stop_reason` matters more in Claude than in OpenAI. The values are `end_turn` (model is done), `tool_use` (model wants to call tools), `max_tokens` (hit the budget), `stop_sequence` (hit a configured stop string), `pause_turn` (long-running tool). Always check `stop_reason` in production loops.

---

## 32.3 Building the workspace — tool design for each integration

The hardest engineering work in the workspace was not Claude integration — it was **tool design**. Each external system needs to be exposed as a small set of clean tools whose descriptions teach Claude when to use them.

### Tool design principles I followed

1. **Few, well-named tools beat many tools.** Aim for 5-15 tools total. Beyond that, Claude starts confusing them. If you have 50 endpoints, group them into 5-10 higher-level tools.
2. **Tool names should be verbs.** `get_jira_ticket` not `jira`, `create_github_pr` not `pr`.
3. **Descriptions teach Claude when to use the tool.** Include trigger conditions, anti-conditions, and limits. "Use this when the user asks about Jira tickets. Do NOT use it for Confluence pages."
4. **Argument names should be self-explanatory.** Claude's input filling improves dramatically when args are named like `ticket_id` not `id`.
5. **Return structured JSON, not free text.** Claude reasons better over structured data. Always include status fields like `status: "ok"` or `error: "..."` for the model to act on.
6. **Idempotency where possible.** If Claude retries a tool call, it shouldn't double-create.

### Jira tools

```python
{
    "name": "get_jira_ticket",
    "description": (
        "Fetch a Jira ticket by ID. Returns title, description, status, "
        "assignee, comments. Use this when the engineer mentions a ticket "
        "ID like ML-1234 or asks 'what's on my plate'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticket_id": {"type": "string", "description": "e.g. ML-1234"}
        },
        "required": ["ticket_id"],
    },
}
```

Other Jira tools: `list_my_tickets()`, `update_ticket_status(ticket_id, status)`, `add_comment(ticket_id, comment)`. Each is a thin wrapper over the Jira REST API with the engineer's identity carried via OAuth.

### GitHub tools

```python
{
    "name": "clone_or_checkout_branch",
    "description": (
        "Clone a repo (or fetch if already cloned) into a per-ticket "
        "git worktree, on the branch tied to the given ticket. Sets up "
        "isolation so concurrent tickets don't collide. Returns the "
        "worktree path. Use this when the engineer says they're starting "
        "work on a ticket."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "ticket_id": {"type": "string"}
        },
        "required": ["repo", "ticket_id"]
    }
}
```

This tool encapsulates the "3-method EBS lifecycle" referenced in the resume. Behind the scenes it: (1) checks if the per-ticket worktree exists, (2) creates an EBS volume tagged with the ticket ID if not, (3) attaches and mounts it, (4) creates the git worktree on the right branch. The complexity is hidden; Claude just calls the tool.

Other GitHub tools: `create_pr(...)`, `list_recent_commits(...)`, `read_file(repo, path, ref)`. Reading files is heavily used — Claude often wants to look at code before suggesting changes.

### Athena tools

```python
{
    "name": "run_athena_query",
    "description": (
        "Run a SQL query against Athena. Read-only — must start with "
        "SELECT. Returns the rows as a list of dicts. For large result "
        "sets, the query times out at 60 seconds; if you need more "
        "rows, narrow the query first. Athena has tables for: "
        "fraud_metrics, model_metrics, feature_distributions, "
        "experiment_runs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {"type": "string"},
            "database": {"type": "string", "default": "ml_metrics"},
        },
        "required": ["sql"]
    }
}
```

Important: the description tells Claude what tables exist. This avoids Claude making up table names. For larger schemas, the workspace also has a `describe_tables(database)` tool that Claude can call first to discover schema.

Athena is read-only. There is no tool for write operations, by design — the workspace shouldn't accidentally mutate data warehouse tables.

### Jenkins tools

```python
{
    "name": "trigger_jenkins_build",
    "description": (
        "Trigger a Jenkins job. Returns a build ID immediately; the "
        "build runs asynchronously. Use poll_build_status to wait for "
        "completion. Most common jobs: 'fraud-train', 'fraud-validate', "
        "'feature-pipeline-refresh'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "job_name": {"type": "string"},
            "parameters": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["job_name"]
    }
}
```

Paired with `poll_build_status(build_id)` and `get_build_logs(build_id, last_n_lines)`. The async pattern means Claude can trigger a 30-minute training run and continue helping with other tasks, polling periodically.

### Slack tools

```python
{
    "name": "post_slack_message",
    "description": (
        "Post a message to a Slack channel or thread. Use this when "
        "the engineer asks to notify the team, or when a long-running "
        "task completes and the engineer wants to be pinged."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {"type": "string"},
            "message": {"type": "string"},
            "thread_ts": {"type": "string", "description": "Optional: reply in a thread"}
        },
        "required": ["channel", "message"]
    }
}
```

Slack tools: `post_slack_message`, `read_slack_thread`, `get_user_dms`. These were used heavily for "notify me when training finishes" patterns.

### Infrastructure provisioning tools

```python
{
    "name": "provision_compute",
    "description": (
        "Spin up an EC2 instance for ML work. CPU instances are cheap "
        "and start in 30 seconds; GPU instances are expensive and take "
        "2-3 minutes. Always specify the instance_type. The instance "
        "auto-terminates after 4 hours of idle to save costs."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "instance_type": {
                "type": "string",
                "enum": ["c6i.xlarge", "c6i.4xlarge", "g5.xlarge", "g5.4xlarge", "p4d.24xlarge"]
            },
            "ticket_id": {"type": "string"},
            "purpose": {"type": "string", "description": "What you'll do with it"}
        },
        "required": ["instance_type", "ticket_id", "purpose"]
    }
}
```

The `purpose` argument is the security-and-audit lever: Claude must articulate what it's provisioning a GPU for, which is logged and reviewable. This deters "burn cluster cycles by accident."

### The system prompt that ties it all together

```
You are the ML Workspace Assistant for the TrueBalance ML team. You
help engineers manage their work across Jira, GitHub, Athena, Jenkins,
and AWS infrastructure.

Available tools (you must use these — do NOT make assumptions about
state):
  - Jira: get_jira_ticket, list_my_tickets, update_ticket_status, add_comment
  - GitHub: clone_or_checkout_branch, create_pr, list_recent_commits, read_file
  - Athena: run_athena_query, describe_tables
  - Jenkins: trigger_jenkins_build, poll_build_status, get_build_logs
  - Slack: post_slack_message, read_slack_thread
  - Infra: provision_compute, terminate_instance, list_my_instances

WORKING PRINCIPLES:
1. Confirm before destructive actions. ALWAYS ask before terminating
   an instance, deleting data, or merging a PR.
2. Be specific. If the engineer says "spin up a GPU", ask which
   instance type unless prior conversation makes it clear.
3. State your reasoning. When you decide to call a tool, briefly say why.
4. Surface costs. If you're about to provision expensive compute,
   mention the hourly cost.
5. Don't fabricate. If you're unsure of a Jira ticket ID, ask.
6. Cite tool results. Quote concrete data from tool outputs in your
   response so the engineer can verify.

You will receive engineer messages and use the tools to fulfill their
intent. Often the right answer requires multiple tool calls — chain
them. After each tool call, briefly summarize what you got back before
deciding the next action.
```

This prompt is the result of months of iteration. The "working principles" section is what makes the assistant feel professional rather than chaotic.

---

## 32.4 Multi-step agent patterns at the workspace

The workspace operates in three modes that the system prompt enables Claude to handle:

### Mode 1 — Single-step (most common)

Engineer: "What's the AUC on fraud-v3 from last Friday?"

Claude calls `run_athena_query` once with a SQL query, returns the answer.

### Mode 2 — Multi-step chain

Engineer: "I'm picking up ML-1234. Spin up a GPU and clone the repo on the right branch."

Claude does:
1. `get_jira_ticket("ML-1234")` → gets the ticket details, sees the linked repo
2. `provision_compute(instance_type="g5.xlarge", ticket_id="ML-1234", purpose="...")` → spins up GPU
3. `clone_or_checkout_branch(repo="...", ticket_id="ML-1234")` → clones repo
4. Returns confirmation with the instance IP and worktree path

This is a 3-tool chain Claude orchestrates without the engineer specifying each step.

### Mode 3 — Long-running with check-ins

Engineer: "Trigger the fraud-train job and ping me when it's done."

Claude does:
1. `trigger_jenkins_build("fraud-train")` → returns build ID immediately
2. `post_slack_message(...)` → "I'll watch this build and ping you when it finishes"
3. **Async background task**: the workspace orchestrator polls `poll_build_status(build_id)` every minute
4. When done: posts another Slack message with the result

The async pattern requires the orchestrator to manage long-running tasks outside Claude's direct control. Claude initiates and reports; the orchestrator monitors.

---

## 32.5 Conversation memory and context window management

The workspace has long-running conversations — engineers can chat across days. Conversation history is stored in Postgres keyed by user_id and session_id (separate from the WebSocket connection). Trimming strategies:

### Sliding window with importance preservation

Keep the last 30 turns by default, but always keep the system prompt and any messages that contain "decision points" — confirmations, configuration changes, key conclusions. The orchestrator marks important messages (using a small classifier or a flag set explicitly when a tool call is destructive).

### Summarization

When conversation exceeds 50 turns, the orchestrator runs a summarization pass: a small Claude Haiku call that produces a 200-word summary of older turns, replacing them with a single system-augmenting message. This is the production-grade approach for long-running engineer sessions.

### Prompt caching — Claude's killer feature

Claude supports **prompt caching**, which is critical for the workspace. Long system prompts plus tool definitions are reused across every turn. With caching enabled, the cached portion is charged at 10% of the input token rate.

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    system=[
        {
            "type": "text",
            "text": LARGE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"}  # cache for ~5 minutes
        }
    ],
    tools=[...],   # tools are also cached
    messages=[...],
)
```

For the workspace, the system prompt is ~2K tokens and the tool definitions are ~3K tokens. With 50 turns per session, caching saves about 90% of input cost on subsequent turns. This is one of the most important optimizations to know — the certification asks about it directly.

---

## 32.6 Vision and PDF — Claude's multi-modal capabilities

Claude can take images and PDFs as input alongside text. The workspace leverages this:

### Image input

```python
{
    "role": "user",
    "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_encoded_image
            }
        },
        {"type": "text", "text": "What's wrong with this loss curve?"}
    ]
}
```

Engineers paste training-loss screenshots; Claude diagnoses ("loss is plateauing around epoch 12, suggests learning rate too low" or "training loss diverged at step 5K, classic LR-too-high or batch-norm-issue"). Claude's vision is genuinely strong on plots and charts.

### PDF input

Claude 4.x can read PDFs directly:

```python
{
    "role": "user",
    "content": [
        {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64_pdf
            }
        },
        {"type": "text", "text": "Summarize the model card and tell me the dataset size"}
    ]
}
```

PDFs go directly into the context, no separate parsing step. Used for ingesting research papers and documentation that engineers want to discuss.

---

## 32.7 The Anthropic stack — what to know for the certification

The Solution Architecture certification covers more than just the API. Key areas:

### Model family (claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5)

| Model | Use case | Cost (per 1M input/output tokens, approx) |
|-------|----------|-------------------------------------------|
| **Opus 4.7** | Complex reasoning, best quality, agentic workflows requiring multiple steps | $15 / $75 |
| **Sonnet 4.6** | Default for most production work — fast and capable | $3 / $15 |
| **Haiku 4.5** | High-volume, low-latency, cost-sensitive (intent classification, simple tools) | $1 / $5 |

The "smart choice" for a system: route by complexity. Use Haiku for intent classification, Sonnet for the bulk of conversational work, Opus for hard reasoning steps. This is the standard "model cascade" pattern and it can cut bills by 60-80%.

### Where Claude runs

- **Anthropic API** — direct, lowest latency, full feature support including caching and computer use.
- **AWS Bedrock** — Anthropic's offerings via AWS. Region-pinned, BAA covers HIPAA. Slightly fewer features than direct API; lags behind on newest models by weeks.
- **GCP Vertex AI** — same shape as Bedrock for GCP customers.
- **Azure** — Claude is now on Azure too, similar tradeoffs.

For Avrioc on AWS with a healthcare/finance angle, Bedrock is the natural choice. For maximum capability and speed, direct API.

### The Anthropic Console / Workbench

The Workbench is Anthropic's dev playground at console.anthropic.com. It's where you:
- Test prompts interactively
- Compare outputs across models
- Generate code snippets in Python, TypeScript, etc.
- Debug tool calling without writing code
- Manage API keys and usage

Mention this in the certification — it's the "where do you start" answer.

### Pricing levers

| Lever | What it does | Savings |
|-------|--------------|---------|
| **Prompt caching** | Cache system prompts and tools across calls | Up to 90% on input tokens |
| **Model cascade** | Use cheaper model for simpler tasks | 60-80% with smart routing |
| **Batch API** | 50% discount on async batched requests | 50% (24h SLA) |
| **Output token control** | `max_tokens` limits how much the model can ramble | Variable |
| **Stop sequences** | Stop generation early when a marker is hit | Variable |

The certification will ask "how do you reduce cost on a high-volume application?" — the answer is "prompt caching + model cascade + batch API where async-acceptable."

### Constitutional AI — what makes Claude different

Anthropic trains Claude with **Constitutional AI** — a framework where the model is taught to follow explicit principles (be helpful, be honest, avoid harm) via self-critique loops in training. The implication for solution architects: Claude refuses unsafe requests by default, not because of guardrails layered on top but because of training signal. You still build guardrails, but you start from a safer baseline.

### Computer use (newest capability)

Claude 4.x can drive a computer — take screenshots, click, type. For now it's an "early access" feature. For the certification, you should know:
- It requires a sandboxed environment (you don't let it loose on production).
- It's slower and more expensive than text tools.
- Use cases: legacy app automation, browser-based agents, QA testing.
- Most workspace integrations (like ours) prefer typed tools over computer use because they're faster and more deterministic.

### MCP (Model Context Protocol)

MCP is Anthropic's open protocol for connecting LLMs to tools and data sources. It's like USB for AI — an MCP server exposes a set of tools and resources, an MCP client (like Claude Desktop or our workspace) consumes them.

For the workspace, the integrations could in theory be MCP servers — one for Jira, one for GitHub, etc. — and Claude would discover them dynamically. We didn't go that route at TrueBalance because we wanted tighter control over tool descriptions and security. But for the certification, know that:
- MCP enables tool reuse across LLM apps
- It's an open standard, not Anthropic-only
- Servers can run locally or remotely (HTTP, stdio, WebSocket transports)

---

## 32.8 Best practices for production Claude apps

These are the patterns the certification grades you on.

### 1. System prompts as configuration

Treat the system prompt like config, not code. Version it, A/B test it, store it in a registry separate from the application code. Roll out changes through canary deployments, just like model updates.

### 2. Tool definitions are part of the prompt budget

Every tool definition is sent on every API call. With 15 tools at ~200 tokens each, that's 3000 tokens per call. For high-volume systems, prune unused tools per intent, or use a two-stage pattern: a smaller LLM picks the relevant 5 tools first, then Claude runs with just those.

### 3. Always check `stop_reason`

`stop_reason == "end_turn"` means Claude finished. `tool_use` means it wants tools. `max_tokens` means you cut it off — bad for production, increase the budget. `pause_turn` is rare but happens with long-running workflows; you handle it like tool_use.

### 4. Stream from day one

Streaming is not a luxury — it's UX. Without it, conversational apps feel slow. With Claude's streaming, you get token-by-token events plus `content_block_start` / `content_block_stop` markers for structure.

### 5. Aggressive prompt caching

If your system prompt > 1024 tokens, cache it. Cost savings are real, latency improves modestly. Cache the tool definitions too — they don't change between turns.

### 6. Logging with structure

Log every API call with: model, system prompt hash, tool count, message count, token usage (input cached, input non-cached, output), latency, stop_reason. Aggregate in LangFuse or Helicone. Alert on token spikes (potential prompt-bloat or runaway loops).

### 7. Safety: defense in depth

- Hardened system prompt
- Pre-input filter (Anthropic's moderation endpoint)
- Output filter (regex + classifier)
- Tool argument validation
- Rate limiting per user
- Audit logs for sensitive tool uses

### 8. Multi-region for high availability

For mission-critical apps, deploy across multiple regions (us-east-1 + us-west-2 + eu-west-1 typical). Detect Anthropic API errors and fail over to Bedrock, or vice versa. Mention this in the certification — it's a common scenario question.

### 9. Cost monitoring

Per-user / per-feature token budgets. Alert when daily spend exceeds threshold. Most production Claude apps eventually need this; build it before you're surprised by the bill.

### 10. Don't put PII in prompts unnecessarily

For HIPAA, PCI, or GDPR contexts, minimize PHI in the prompt. Use IDs and lookups via tools rather than embedding the data in context. Anthropic's BAA covers PHI on Bedrock, but the principle of least exposure still applies.

---

## 32.9 Scenario-based certification questions with detailed answers

### Scenario 1 — Cost reduction for a high-volume chatbot

**Question:** "A customer's chatbot has a 5K-token system prompt and processes 100K conversations/day. Current monthly bill is $50K. Walk me through reducing cost."

**Answer:** First lever, prompt caching — the 5K system prompt repeats on every turn. With ephemeral cache, the cached tokens are charged at 10% of input rate. If the average conversation has 8 turns, caching saves ~90% on the system prompt portion across the full conversation, dropping that portion of the bill by ~$30K. Second lever, model cascade — most conversations probably don't need Sonnet for every turn. Route the first turn (intent classification, simple acknowledgment) to Haiku, only escalate to Sonnet for substantive responses. Realistically 50-70% of turns can stay on Haiku, saving another $5-10K. Third lever, batching for async use cases — if any of the 100K daily conversations are async (email summaries, reports, after-hours analysis), shift them to the Batch API for 50% discount. Fourth lever, audit max_tokens — if you set max_tokens=4096 but average response is 300 tokens, you're not paying for the headroom but you may be allowing unnecessarily long responses. Tighten to 1024 unless you genuinely need more. Combined: a $50K bill drops to $10-15K.

### Scenario 2 — Multi-region failover

**Question:** "Design a high-availability deployment of a Claude-powered service. Anthropic API has occasional regional issues."

**Answer:** Multi-region with fallback to Bedrock. Primary stack: Anthropic API direct in us-east-1 for lowest latency. Secondary: Bedrock in us-east-1 for the same model (claude-sonnet-4-6). Tertiary: Anthropic API in us-west-2 as cross-region backup. Application code uses a circuit breaker per provider — if the Anthropic primary fails 3 times in 60 seconds, route to Bedrock for the next 5 minutes. If both us-east endpoints fail, fall to us-west. Health checks run continuously against a small `messages.create` call. For the certification: emphasize that Bedrock and Anthropic API have slightly different feature support — caching works on both, but newer features may only be on Anthropic direct first. Plan for feature parity in the failover path or accept degraded capability during failover.

### Scenario 3 — Migrating from OpenAI to Claude

**Question:** "A customer has a working app on OpenAI gpt-4-turbo with function calling. They want to migrate to Claude. What's involved?"

**Answer:** The conceptual model is the same — messages, tools, streaming — but the API shapes differ. Concrete migration steps. First, shift the system prompt: OpenAI puts it in the messages list as `role: "system"`; Claude takes it as a separate top-level `system` parameter. Second, restructure tool definitions: OpenAI uses `parameters` for the JSON schema, Claude uses `input_schema`. Third, restructure tool result messages: OpenAI uses a separate `role: "tool"` message with `tool_call_id`; Claude uses `tool_result` content blocks inside a `user` message with `tool_use_id`. Fourth, handle response content shape: OpenAI's content is a string; Claude's content is a list of blocks. Fifth, check `stop_reason` instead of `finish_reason` — values are similar but spelled differently. Sixth, take advantage of Claude-specific features: prompt caching for cost, vision for multi-modal inputs, the system prompt as a privileged context for prompt-injection resistance. Total effort for a typical app: 1-2 weeks including testing and parallel-running. Use LangChain or LiteLLM as an adapter if you need to stay portable across both providers.

### Scenario 4 — Choosing between Claude direct API and Bedrock

**Question:** "A healthcare customer wants to build a clinical assistant. They're undecided between Anthropic API and Bedrock. What's your recommendation?"

**Answer:** Bedrock for healthcare. Three reasons. First, AWS BAA — Bedrock invocations are covered by AWS's BAA, which simplifies HIPAA compliance for the customer. Anthropic also offers BAA for direct API but it requires a separate agreement and Anthropic-specific compliance review. Second, region pinning — Bedrock guarantees data stays within the chosen AWS region, important for jurisdictions with strict data residency. Third, IAM integration — Bedrock invocations are authenticated via IAM, which fits naturally into the customer's existing AWS access controls. The trade-offs to flag: Bedrock typically lags the direct API by 1-4 weeks on new models. Computer use and other early-access features may not be on Bedrock yet. If the customer needs the absolute latest capabilities, direct API plus a custom BAA is the path. For most healthcare customers, Bedrock's compliance posture wins.

### Scenario 5 — Tool-calling for a regulated workflow

**Question:** "We're building a financial advisor agent that can place trades. How do we make this safe?"

**Answer:** Multiple layers. First, structurally: don't give Claude direct trade-execution permission. Define a `propose_trade(...)` tool that creates a draft, plus a separate `approve_and_execute_trade(draft_id, user_confirmation)` tool that requires explicit user confirmation collected outside the LLM context. Claude can propose; only a human can execute. Second, hard limits: per-user dollar caps enforced at the trade-API level, not in Claude's prompt. Third, audit logs: every proposed trade, every approval, every execution logged immutably. Fourth, compliance review: a parallel monitoring system runs every Claude conversation through a compliance classifier (was financial advice given that requires a licensed advisor?). Flag suspicious patterns. Fifth, model choice: Opus or Sonnet, never Haiku for high-stakes reasoning. Sixth, prompt-injection defense: hardened system prompts, pre-input filters, output checks for prohibited content. The over-arching principle: the LLM proposes, humans execute, every step is logged.

### Scenario 6 — Long-running agent with checkpointing

**Question:** "Design an agent that performs a complex 30-step research task. It might take 20 minutes. Users want to see progress."

**Answer:** Three architectural pieces. First, run the agent in LangGraph (Chapter 13) with a DynamoDB checkpointer — every node transition saves state, so a crashed worker resumes cleanly. Second, expose progress via WebSocket: the agent emits structured events (`step_started`, `tool_called`, `step_complete`) that flow to the user's UI in real time. Third, run on a long-lived compute target — Lambda's 15-minute limit is a hard ceiling, so use ECS Fargate or EKS for the agent runner. Fourth, support pause-and-resume: the user can stop the agent mid-flight, the state is checkpointed, and resuming starts from the last successful step. Finally, observability: every step emits a LangFuse trace span, so you have full causality on every tool call and LLM response. For the certification: emphasize that long-running agents need explicit state management — you can't rely on in-memory state inside a Lambda.

### Scenario 7 — Multi-tenant SaaS with strict isolation

**Question:** "We're building a Claude-powered SaaS where each tenant has their own knowledge base. How do we keep them isolated?"

**Answer:** Layered isolation. Application layer: every Claude call is parameterized with the tenant's ID, which is used to look up the correct vector store, the correct system prompt extension, the correct tool whitelist. Vector layer: separate index per tenant, or a single index with tenant_id as a metadata filter that's enforced server-side (never trust client-supplied tenant filters). Database layer: row-level security in Postgres or per-tenant table prefixes. Compute layer: in critical scenarios, separate Lambda functions or separate Bedrock IAM roles per tenant — overkill for most, necessary for healthcare or finance. Observability: tenant_id propagates through every log line and trace span. Cost allocation: per-tenant token usage tracked for chargeback. Don't trust the LLM to maintain tenant isolation — the LLM doesn't know what tenant means. Isolation is structural, not prompt-based.

### Scenario 8 — Handling rate limits gracefully

**Question:** "Our app is hitting Anthropic's rate limits during peak hours. What do we do?"

**Answer:** Several levers. First, request a higher tier — paid tiers have much higher RPM and TPM limits. Second, implement client-side queue with backoff: when you hit a 429, retry with exponential backoff and jitter. Third, route by model: if you're on Opus, Sonnet has separate (often higher) limits. Cascade to a smaller model when limits are hit. Fourth, batch where possible: the Batch API has separate limits and 50% pricing. Fifth, distribute across providers: split traffic between Anthropic direct and Bedrock for organic load balancing. Sixth, application-level rate limiting: don't let one user dominate. Seventh, cache aggressively: prompt caching reduces input tokens, which is the bottleneck on TPM. The certification answer: "tier upgrade + intelligent backoff + multi-provider routing + caching."

### Scenario 9 — Migrating to a newer Claude model

**Question:** "Anthropic just released Claude Sonnet 4.7. We're on Sonnet 4.6. What's the migration plan?"

**Answer:** Five steps. First, regression-test against the existing eval set — does the new model perform equal-or-better on intent classification, faithfulness, code generation, etc.? Run on offline fixtures, never on first-touch prod traffic. Second, prompt-test — newer models may interpret system prompts subtly differently. Have a small spot-check set of prompts that have caused issues historically. Third, cost-test — token efficiency varies by model; track tokens-per-conversation. Fourth, canary deploy — route 5% of traffic to the new model for a week, monitor LLM-as-judge scores, refusal rates, latency, error rates. Roll back automatically on guardrail breach. Fifth, full rollout once metrics stabilize. The certification answer emphasizes evaluation, not blind upgrade. Newer doesn't always mean better for your specific use case.

### Scenario 10 — Building an enterprise RAG with citations

**Question:** "An enterprise customer has 10 million documents and wants a RAG chatbot with strong citations. Walk me through the design with Claude."

**Answer:** Standard RAG plus Claude-specific optimizations. Ingest pipeline: chunking with semantic boundaries, embed with a strong model (BGE-large or voyage-large-2), store in a vector DB with metadata for source URL, last_updated, document_class. Query pipeline: hybrid search (BM25 + dense + RRF) → cross-encoder rerank → top-5 chunks → Claude with citation requirements in system prompt → output. The Claude-specific optimizations: First, use prompt caching aggressively. The system prompt that defines citation behavior is a stable ~3K tokens — cache it. Tool definitions for any tools (search refinement, follow-up retrieval) — cache them too. Second, leverage Claude's structured output: the response prompt requires `[citation_id]` markers in-line plus a `citations` JSON block at the end with source URLs. Third, use vision for any documents with figures or tables — Claude can read PDFs directly. Fourth, faithfulness check: post-process the response, verify each citation marker maps to a real retrieved chunk. Fifth, sliding-detail responses: default to 200-word answers with "tell me more" expansion. For 10M documents, the indexing and retrieval (Chapter 7, 8) are the heavy work. Claude is the natural choice for the answer LLM because it's strongest on faithful citation.

---

## 32.10 Solution Architecture certification — likely exam topics

Based on Anthropic's published curriculum and similar AWS-style architecture certifications, expect questions on:

### Topic 1 — Choosing the right model

- When to use Opus vs Sonnet vs Haiku
- The model cascade pattern (Haiku → Sonnet → Opus by complexity)
- Cost-quality tradeoffs

### Topic 2 — API mechanics

- Message structure (content blocks)
- Tool calling format (tool_use, tool_result, tool_use_id)
- Streaming events
- stop_reason values and handling
- System prompt vs user message
- max_tokens, temperature, top_p

### Topic 3 — Prompt engineering

- Clear, structured system prompts
- Few-shot examples
- XML-style structuring (Claude responds particularly well to XML tags in prompts)
- Chain-of-thought prompting
- Output format constraints

### Topic 4 — Cost optimization

- Prompt caching (ephemeral and 1-hour)
- Model cascade
- Batch API
- Streaming for early termination

### Topic 5 — Tool use and agents

- Tool design principles
- Multi-step agent patterns
- Long-running agents with checkpoints
- Computer use (early access)
- MCP (Model Context Protocol)

### Topic 6 — Multi-modal

- Vision (image input)
- PDF documents
- When to use vision vs OCR

### Topic 7 — Deployment options

- Anthropic API
- AWS Bedrock
- GCP Vertex
- Azure
- Migration considerations

### Topic 8 — Security and safety

- Constitutional AI
- Prompt injection defense
- Guardrails
- PII handling
- BAA / HIPAA
- Audit logging

### Topic 9 — Observability

- LangFuse, Helicone, custom tracing
- Token usage tracking
- Per-user cost attribution
- Alert design

### Topic 10 — Edge cases and failure modes

- Rate limits
- Multi-region failover
- Long context handling
- Hallucination detection
- Disagreement between model versions

---

## 32.11 Resume tie-in — what to say about the workspace in the interview

> "The Claude-powered ML workspace at TrueBalance was a multi-tool agent. I exposed Jira, GitHub, Athena, Jenkins, Slack, and our AWS infrastructure as a small set of well-described tools — about a dozen total — and Claude orchestrated work across them based on engineer intent. The hard engineering wasn't Claude integration; it was tool design. I learned that few well-named tools beat many tools, that descriptions teach Claude when to call, that idempotency matters because Claude retries. I cached the system prompt and tool definitions aggressively — that alone cut our API bill by about 80%. I used Sonnet for most turns, Haiku for intent classification, escalated to Opus only for hard reasoning. Sessions persisted across days using Postgres-backed conversation memory with sliding-window-plus-summarization for length management. The killer feature for engineers was async tool patterns — they could ask Claude to trigger a 30-minute training run, then go back to other work, and Claude would ping them on Slack when it finished."

That's the elevator pitch. It hits architecture, tool design, cost optimization, and operational maturity in 90 seconds.

---

## 32.12 Cheatsheet

```
   CLAUDE API KEY DIFFERENCES VS OPENAI
     - System prompt: top-level parameter, not in messages list
     - Content: list of blocks, not a string
     - Tool result: tool_result block in user message, with tool_use_id
     - Tool schema: input_schema, not parameters
     - Stop reason: stop_reason (end_turn / tool_use / max_tokens / ...)
     - Streaming: content_block events with deltas

   MODEL FAMILY (2026)
     Opus 4.7      — best, most expensive, complex reasoning
     Sonnet 4.6    — default for production
     Haiku 4.5     — fast and cheap, simple tasks

   COST LEVERS
     1. Prompt caching (90% savings on cached input)
     2. Model cascade (route by complexity)
     3. Batch API (50% off for async)
     4. max_tokens tightening
     5. Stop sequences

   DEPLOYMENT OPTIONS
     - Anthropic API: most features, fastest
     - AWS Bedrock: BAA, region pinning, IAM
     - GCP Vertex: similar to Bedrock for GCP
     - Azure: similar to Bedrock for Azure

   TOOL DESIGN PRINCIPLES
     1. Few well-named tools (5-15)
     2. Verbs in tool names
     3. Descriptions teach when to use
     4. Self-explanatory args
     5. Structured JSON returns
     6. Idempotency

   PROMPT CACHING
     - Add cache_control: {type: "ephemeral"} to a content block
     - Cache valid for ~5 minutes (ephemeral) or 1 hour (1h)
     - Cached input charged at 10% rate

   STOP REASONS TO HANDLE
     - end_turn — done
     - tool_use — execute tools, loop
     - max_tokens — increase budget
     - pause_turn — long-running tool
     - stop_sequence — hit configured stop string
```

---

End of Chapter 32. Continue to **[Chapter 33 — More LLM/RAG/Recommender System Designs](33_more_system_designs.md)** or back to **[Chapter 00 — Master Index](00_index.md)**.
