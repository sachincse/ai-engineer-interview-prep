# Chapter 35 — n8n Workflow Automation + AI Consulting Patterns

> **Why this chapter exists:** Modern Applied AI roles increasingly mix two skill sets: **building automations** with low-code workflow tools (n8n, Zapier, Make.com) for high-leverage business problems, and **consulting** with non-technical teams to identify where AI helps and where it doesn't. Upvest's Applied AI JD names both explicitly — "design and implement end-to-end workflows using tools such as n8n" and "act as an in-house AI consultant, running discovery sessions, creating playbooks, coaching colleagues on prompt engineering." This chapter prepares you for both surfaces.

---

## 35.1 What n8n is and where it fits

### The plain-English mental model

n8n (pronounced "n-eight-n," shorthand for "nodemation") is an **open-source workflow automation platform** with a visual drag-and-drop builder. You connect nodes — each node represents a step (read from Slack, query a database, call an LLM, send an email) — into a workflow that runs on a schedule, on a webhook, or manually.

The big differentiator from Zapier and Make.com: n8n is **self-hostable** and the source is open. For BaFin/FCA-regulated companies like Upvest, that matters — you can run n8n on your own infrastructure, keep all data inside your perimeter, and audit every workflow through your own logs. Zapier and Make.com require sending data to their hosted services, which is a non-starter for most regulated finance companies.

### When to use n8n versus a custom application

```
   Use n8n when:
     - The workflow is a sequence of API calls / data transformations
     - Non-engineers need to read or modify the workflow
     - You want to deploy a useful automation in hours, not weeks
     - The workflow's logic fits "if X happens, do Y, then Z"

   Use custom code when:
     - The logic has complex branching or state management
     - Performance matters (n8n adds overhead per node)
     - You need fine-grained error handling
     - The workflow is a core system, not a glue layer
```

A pragmatic principle that the Upvest JD calls out explicitly: **"make pragmatic decisions on when a simple automation is more effective than a complex agent-based solution."** Translation: don't reach for LLM agents when an n8n workflow with a hardcoded rule will do. Save the agent complexity for problems that genuinely need it.

---

## 35.2 The n8n architecture

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │                   n8n SELF-HOSTED DEPLOYMENT                         │
   │                                                                      │
   │   ┌────────────┐   ┌────────────┐   ┌────────────┐                   │
   │   │   Web UI   │   │  Editor    │   │  REST API  │                   │
   │   │  (browser) │   │ (workflow  │   │  (trigger  │                   │
   │   │            │   │   builder) │   │   exec)    │                   │
   │   └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                   │
   │         │                │                 │                         │
   │         └────────────────┴─────────────────┘                         │
   │                          │                                           │
   │                          ▼                                           │
   │                   ┌────────────┐                                     │
   │                   │  n8n core  │                                     │
   │                   │  (Node.js) │                                     │
   │                   └─────┬──────┘                                     │
   │                         │                                            │
   │       ┌─────────────────┼─────────────────┐                          │
   │       ▼                 ▼                 ▼                          │
   │   ┌────────┐      ┌──────────┐     ┌──────────┐                      │
   │   │Postgres│      │  Redis   │     │ Workflow │                      │
   │   │(meta + │      │ (queue)  │     │ Workers  │                      │
   │   │history)│      │          │     │ (Node.js)│                      │
   │   └────────┘      └──────────┘     └──────────┘                      │
   │                                          │                           │
   │                                          ▼                           │
   │                                   ┌──────────────┐                   │
   │                                   │ External APIs│                   │
   │                                   │ (Slack, DB,  │                   │
   │                                   │  HTTP, LLM)  │                   │
   │                                   └──────────────┘                   │
   └──────────────────────────────────────────────────────────────────────┘
```

The pieces:

- **Web UI**: visual workflow editor, accessed in the browser.
- **n8n core**: Node.js runtime that parses workflow definitions and dispatches execution.
- **Postgres**: stores workflows, credentials, execution history.
- **Redis**: queue between core and workers. For high throughput.
- **Workers**: separate Node.js processes that execute workflow nodes. Scale these horizontally for high volume.
- **External APIs**: Slack, GitHub, your databases, LLM providers — anything with an HTTP API.

Deployment in a regulated environment is straightforward: drop the n8n Docker image into your Kubernetes cluster, point it at an internal Postgres, configure SSO for the editor. All data stays inside your VPC.

---

## 35.3 Building workflows in n8n — the nodes you'll actually use

### Trigger nodes (how a workflow starts)

| Trigger | Use case |
|---------|----------|
| **Schedule** | Cron-like — run every hour, every Monday, etc. |
| **Webhook** | HTTP endpoint — external systems POST to this to trigger |
| **Email Trigger** | When an email arrives in a watched mailbox |
| **Slack Trigger** | When a message hits a channel or a slash command fires |
| **GitHub Trigger** | When a PR opens, an issue is created, etc. |
| **Manual** | Click a button to run, useful for testing |

### Action nodes (the work)

| Action | What it does |
|--------|--------------|
| **HTTP Request** | Generic — call any REST API |
| **Postgres / MySQL / Mongo** | Run queries against databases |
| **Slack** | Post messages, read channels |
| **Email** | Send via SMTP or SES |
| **Google Sheets / Airtable** | Read/write spreadsheet data |
| **OpenAI / Anthropic / HuggingFace** | LLM calls — chat completions, embeddings |
| **AI Agent** | A higher-level node that runs an LLM agent with multiple tool calls |
| **Code** | Run JavaScript or Python (in self-hosted n8n) |
| **Function** | Transform data with a small JS expression |
| **IF** | Conditional branching |
| **Switch** | Multi-way branching |
| **Merge** | Combine multiple branches |
| **Wait** | Pause for N minutes or until a condition |

### A worked example — Slack-to-LLM helpdesk

The kind of workflow Upvest's Applied AI engineer would build in week one:

```
   ┌──────────────┐
   │ Slack        │  Trigger: when a message in #help-it contains
   │ Trigger      │  the bot mention or matches a question pattern
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ Postgres     │  Action: query the IT knowledge base for the
   │ (KB lookup)  │  question text, return top-3 matching articles
   └──────┬───────┘
          │
          ▼
   ┌──────────────┐
   │ IF           │  Branch: did we find any matches?
   └──┬────────┬──┘
      │        │
   yes│        │no
      ▼        ▼
   ┌──────┐ ┌──────────┐
   │ LLM  │ │ Slack    │  No match → escalate to human via DM to IT lead
   │(Claude│ │ (post DM)│
   │ Sonnet)│└──────────┘
   │ Synth │
   │ answer│
   └──┬───┘
      │
      ▼
   ┌──────────────┐
   │ Slack        │  Action: reply in thread with the answer + KB links
   │ (reply)      │
   └──────────────┘
```

The whole workflow is about 7 nodes, builds in 30 minutes, and now answers IT questions automatically while learning when to escalate. That's the pattern Upvest wants — "lightweight integrations when needed, expand connectivity as new use cases evolve."

### LLM nodes deep-dive

The LLM nodes in n8n have two flavors:

1. **Direct LLM calls**: send a prompt, get a response. Like calling OpenAI directly. Use for simple text generation, classification, summarization.

2. **AI Agent node**: wraps an agentic loop with configurable tools. You provide tool nodes (Postgres, HTTP, Slack), the agent node uses them in a multi-step reasoning loop. Use for problems where the answer requires multiple steps, branching, or dynamic decision making.

The pragmatic decision rule from the JD: "make pragmatic decisions on when a simple automation is more effective than a complex agent-based solution." For "answer IT helpdesk questions" — simple LLM call after a KB lookup is fine. For "investigate this customer complaint, gather context from CRM and Slack and Jira, propose a resolution" — agent node makes sense.

---

## 35.4 n8n vs alternatives

| Tool | Self-hostable | LLM-native | Best for |
|------|---------------|------------|----------|
| **n8n** | Yes (open source) | Yes — native AI nodes | Regulated finance, technical teams, long-running workflows |
| **Zapier** | No (SaaS only) | Yes (limited) | Simple consumer/SMB workflows, non-technical users |
| **Make.com** | No (SaaS only) | Yes | Mid-complexity, visual-first |
| **Pipedream** | No (SaaS) but code-friendly | Yes | Developer-flavored automation |
| **Apache Airflow** | Yes | No (general-purpose) | Data engineering, batch pipelines |
| **Temporal** | Yes | Some integrations | Long-running stateful workflows in code |

For Upvest's use case — regulated, on-premises, internal automation across teams — n8n is exactly the right pick. Self-hosted means data residency is preserved; the visual editor means non-engineers can read and tweak workflows; the AI nodes plug right into Claude.

---

## 35.5 The AI consulting half of the role

Upvest's JD splits the role into "build" and "consult." The consulting half is what makes the role unusual — most AI engineering jobs are pure build. This is where you, in the interview, demonstrate that you've done both.

### Discovery sessions — how to find AI opportunities

A discovery session is a 60-90 minute structured conversation with a team to surface AI use cases. The pattern I'd use:

```
   1. WHAT WE DO (5 min)
      Brief from the team: what's the team's job, what tools do they use,
      what's the volume of work?

   2. PAINS AND BOTTLENECKS (20 min)
      Open question: what takes longer than it should? What do you wish
      was automated? What manual steps repeat?
      Probe: how often, how long, how many people, what's the cost?

   3. CURRENT WORKFLOW WALKTHROUGH (20 min)
      Walk through one typical task end-to-end. Have the team show their
      screen if possible. Note every tool switch, every copy-paste, every
      email or Slack thread.

   4. AI HYPOTHESES (20 min)
      Propose 3-5 places AI could help, with specific bets:
      - "What if when X happens, AI drafts Y for human review?"
      - "What if AI did the classification step here, then human acts?"
      - "What if AI answered the first 80% of cases automatically?"

   5. PRIORITIZE AND COMMIT (15 min)
      Rank by impact × feasibility. Pick ONE to prototype in the next
      two weeks. Identify success metric and stakeholders.
```

The output of a discovery session is a single-page summary: the team, the pain, the hypothesis, the metric, the timeline. Don't try to solve everything; get one win shipped.

### Common AI opportunity patterns

These are the patterns that show up everywhere. When you walk into a new team, look for these shapes:

```
   PATTERN 1 — Triage / classification
     "We get 200 emails / tickets / messages a day. Most are simple,
      some are urgent. We waste time triaging."
     → AI classifier routes by urgency / category, humans handle the
       hard ones.

   PATTERN 2 — Drafting and templating
     "We write similar reports / responses / messages over and over."
     → AI drafts from a template; humans review and edit.

   PATTERN 3 — Summarization
     "We have meeting transcripts / Slack threads / emails to read."
     → AI generates structured summaries on a schedule.

   PATTERN 4 — Question answering over docs
     "Half our questions could be answered by reading the docs."
     → RAG over docs in a chat interface (Slack bot, internal portal).

   PATTERN 5 — Data enrichment
     "We have unstructured data we manually categorize / tag."
     → LLM extracts structured fields from unstructured inputs.

   PATTERN 6 — Automated investigation
     "When alert X fires, an engineer manually checks Y, Z, W."
     → Agent runs the investigation steps, posts findings in Slack.

   PATTERN 7 — Translation / localization
     "We translate content for multi-region teams."
     → LLM translates with style guides; humans review.

   PATTERN 8 — Code review / linting beyond rules
     "Our reviewers spot the same patterns over and over."
     → AI bot pre-comments before the human reviewer.
```

Memorize these. When a team describes their pain, you map it to one of these patterns within 60 seconds, and you've already won credibility.

### Build vs buy decisions

Once a use case is identified, decide:

```
   USE OFF-THE-SHELF (don't build) when:
     - The use case is generic (general AI assistant, transcription)
     - Quality matters more than fit-to-system (Claude is better than
       a custom LLM)
     - The vendor has good security / compliance posture

   USE n8n (lightweight build) when:
     - The workflow is a sequence of well-understood steps
     - Non-engineers will read or modify it
     - Total complexity fits in 10-20 nodes

   BUILD CUSTOM CODE when:
     - Performance matters (sub-second latency)
     - Logic is complex enough that visual workflow becomes spaghetti
     - The system is core, not glue
     - You need to integrate with non-standard internal APIs

   BUILD AN MCP SERVER when:
     - Multiple AI applications will share this integration
     - The integration is reusable across teams
     - It's central to the company's AI strategy
```

A senior signal: prefer the leftmost option that works. Don't custom-build what you can off-the-shelf, don't agent-ize what you can simply automate.

---

## 35.6 Internal playbooks — what to write

The Upvest JD names "creating internal playbooks and best practices" as a deliverable. These are short documents that codify how the company uses AI. Examples:

### Playbook 1 — How to write a good system prompt

A 2-page guide for non-engineers:
- The four parts of a system prompt: persona, task, constraints, output format
- How to add examples (few-shot)
- How to encode safety rules
- How to test and iterate
- Common mistakes (vague tasks, contradicting constraints, no output format)

### Playbook 2 — When to use which LLM

A decision matrix:
- Haiku for simple classification, fast responses
- Sonnet for most production work
- Opus for complex reasoning
- GPT-4o for parity comparisons
- Open-source self-hosted (Llama) when data can't leave the perimeter

### Playbook 3 — How to integrate AI safely with internal data

For engineers integrating AI into their team's workflow:
- Never include PII in prompts unnecessarily
- Use MCP for reusable integrations
- Validate AI outputs before acting on them
- Log every AI action for audit
- Get sign-off from compliance for new data sources

### Playbook 4 — Prompt engineering by team type

Customized one-pagers per team:
- Sales: lead qualification prompts, email drafting prompts
- Support: response drafts, escalation classification
- Engineering: code review prompts, debug-helper prompts
- Operations: meeting summary prompts, status report prompts

These playbooks are how the AI consultant scales. Instead of consulting one team at a time forever, you produce reusable artifacts.

---

## 35.7 Coaching colleagues on prompt engineering

The third part of the consultant role — actually teaching humans to be good at AI. The pattern that works:

### The 4-week coaching arc

```
   Week 1: BASICS
     - 30-min session: anatomy of a prompt
     - Hands-on: write 3 prompts for their actual work
     - Homework: rewrite one prompt they're already using

   Week 2: ITERATION
     - 30-min session: how to iterate on a failing prompt
     - Demo: take a vague prompt and refine it through 3 iterations
     - Homework: improve one prompt 50% on a measurable metric

   Week 3: ADVANCED
     - 30-min session: few-shot, chain-of-thought, structured output
     - Hands-on: rewrite a complex prompt using these techniques
     - Homework: build one production-ready prompt

   Week 4: INTEGRATION
     - 30-min session: how to put a prompt into n8n / a script / Claude
     - Hands-on: deploy one of their prompts as an n8n workflow
     - Homework: ship one production AI feature end-to-end
```

By week 4, the colleague is autonomous. They build their own AI features with light supervision. Multiply this across 290 employees and you've created a 290-person AI-augmented organization, which is the actual goal of the role.

### Common mistakes to coach away

When reviewing colleagues' prompts, watch for these:

1. **Vague task description.** "Help me with this email" → "Draft a professional reply that thanks them for the offer and asks for clarification on point X."
2. **Contradictory constraints.** "Be detailed but concise" — pick one.
3. **No output format.** Always specify: "Return in the format: ..."
4. **No examples.** For classification, always show 2-3 examples in the prompt.
5. **Forgetting to test edge cases.** Always test with empty input, weird input, hostile input.
6. **Treating the LLM as deterministic.** Set temperature; expect variation; sample multiple responses for important tasks.
7. **No iteration loop.** Run, evaluate, refine. Most colleagues use the first prompt that "works."
8. **PII in prompts unnecessarily.** Coach the redaction habit.

---

## 35.8 ROI estimation — the conversation business owners want

When you propose an AI project, the business owner asks "what's the ROI?" Here's the framework:

```
   Estimate the BEFORE-AI cost:
     - People-hours per week × hourly rate
     - Error rate × cost per error
     - Time-to-resolution × volume

   Estimate the AFTER-AI cost:
     - LLM tokens × cost per million
     - Engineering time to build (one-time)
     - Maintenance time per month
     - Residual human hours on edge cases

   ROI = (Before - After) / Before, or as months-to-payback

   Be conservative. Always include:
     - Edge cases that need human handling (typically 10-30% remain)
     - Time for AI errors and rework
     - The cost of model upgrades / prompt drift over time
```

A worked example: customer support workflow processing 1000 tickets/day. Before-AI: 3 engineers × 8 hours × 5 days/week × $100/hour = $12K/week. After-AI: 1000 tickets × ~$0.05 LLM cost = $50/day = $250/week, plus 1 engineer for the 20% edge cases at $4K/week. Total After: $4.25K/week. ROI: 65% reduction, payback period for build cost: ~2 months.

These numbers — even rough ones — make the conversation business-credible.

---

## 35.9 Stakeholder management

The hidden hard part of the consultant role: dealing with humans.

### Setting expectations

- AI is not magic. It's good at some things, mediocre at others, terrible at a few. Be specific about each in your proposals.
- AI will make mistakes. Plan for human review on anything important.
- AI projects take 2-4 weeks of iteration. Don't promise overnight success.
- AI changes monthly. Budget for ongoing iteration, not one-off delivery.

### Handling skeptics

Common objections and responses:
- "AI is unreliable" → "Yes, that's why we have human review on the path. Here's the failure rate from our pilot: 3% requires escalation, the other 97% is faster than the manual process."
- "What if it leaks customer data?" → "We use [self-hosted / data-residency / BAA] to keep data inside our perimeter. Here's the audit log of every AI call."
- "Won't this replace people?" → "It changes what they do. Less triage, more strategic work. Here's how we're investing in upskilling."
- "Hallucinations are dangerous in fintech" → "Yes. That's why we use AI for drafting, not deciding. Every AI output gets a human review before action."

### Handling enthusiasts

The opposite problem: someone who wants to use AI for everything. Pull them back:
- "What's the simplest version of this?" — often a SQL query plus a Slack notification, no AI needed.
- "Could we ship this without AI in two days?" — if yes, do that, then add AI later if needed.
- "What's the failure mode if AI gets this wrong?" — if catastrophic, AI isn't the right tool yet.

---

## 35.10 Interview Q&A

### Q1. How do you decide between an n8n workflow and a custom-built application?

n8n is right when the logic is "API call A, transform, API call B, branch on result, API call C" — a sequence of well-understood steps. It's right when non-engineers need to read or modify the workflow. It's right when you want to ship in hours, not weeks. Custom code is right when performance matters (n8n adds overhead per node), when the logic has complex stateful branching that becomes spaghetti in a visual editor, or when the system is core to the product, not glue between systems. The pragmatic principle from the Upvest JD applies: prefer the simpler tool that solves the problem.

### Q2. Walk me through how you'd run a discovery session with a team.

Five sections in 60-90 minutes. First, the team briefs me on what they do and what tools they use. Second, an open conversation about pains and bottlenecks — what takes longer than it should, what's repetitive, what's frustrating. Third, walk through one typical task end-to-end, ideally with screen-share. Fourth, I propose 3-5 hypotheses where AI could help, each with a specific bet — "what if AI drafted the response and humans reviewed?" Fifth, we prioritize by impact times feasibility and commit to one to prototype in the next two weeks. The output is a one-page summary: team, pain, hypothesis, success metric, timeline. Don't try to solve everything; ship one win first.

### Q3. How do you decide whether to build an agent or a simple automation?

Default to simple automation. If "if X happens, do Y, then Z" works, build that — it's deterministic, debuggable, cheap. Reach for an agent when the workflow needs dynamic decisions: "investigate this alert, gather context from multiple systems, propose a resolution." If the steps are unknown until runtime, an agent is justified. The sign you've over-built: a 10-step deterministic workflow wrapped in agent decoration. The sign you've under-built: a hardcoded workflow that doesn't handle the long tail and gets escalated to humans constantly. Pragmatic test: write the simplest version first, ship, see where it breaks, decide if an agent is needed for those failure modes.

### Q4. Tell me about a successful AI consultation you've done.

Use the structure: situation, observation, hypothesis, prototype, result. Example shape: "The compliance team was spending 4 hours per week summarizing meeting transcripts for the regulator's quarterly report. I sat with them, watched the manual process, hypothesized that an LLM with a structured-output prompt could draft the summary in the right format. I built a prototype in n8n in two days that took the transcript, called Claude with their template, and posted the draft in Slack for review. After two weeks of iteration the draft quality was good enough that compliance review took 30 minutes per week instead of 4 hours. Net win: 14 hours per month freed up, plus more consistent format."

### Q5. How do you measure the success of an AI project?

Three layers. First, the business metric — time saved, errors reduced, cost cut, revenue up. This is what stakeholders actually care about. Second, the AI quality metric — accuracy, faithfulness, escalation rate. This tells you if the AI is doing its part. Third, the adoption metric — how many people actually use the feature, how often. The third is the silent killer of AI projects: a great-quality AI feature nobody uses is a failure. Track all three from day one. For each, define the threshold for "we should keep this" and the threshold for "we should kill this."

### Q6. What's an internal playbook you'd write for non-technical colleagues?

The first one I'd write: "How to write a good system prompt." Two pages. Section 1: the four parts of a prompt — persona, task, constraints, output format. Section 2: how to add examples for few-shot learning. Section 3: how to encode safety rules ("never X", "always cite Y"). Section 4: how to test and iterate, including the temperature setting. Section 5: common mistakes — vague tasks, contradicting constraints, no output format, no examples. Each section has a short example with the bad version and the improved version. The playbook is a teaching artifact; once 290 colleagues have read it, the org's average prompt quality improves significantly without me being involved in every prompt.

### Q7. How do you handle a stakeholder who's skeptical of AI?

Listen first. The skepticism usually has a real basis — a previous bad experience, a compliance worry, a concern about job displacement. Address the actual concern. If it's reliability: "Yes, AI makes mistakes. That's why this design has human review on every output. Here's the failure rate from our pilot." If it's data privacy: "We're self-hosting Claude through Bedrock. Here's the data flow diagram showing nothing leaves our perimeter." If it's job displacement: "It changes the work, not the headcount. Here's how the team's role evolves." Skeptics, properly addressed, often become the strongest champions because they've already thought through the failure modes.

### Q8. How do you handle a stakeholder who wants to use AI for everything?

Push back gently with the simplest-version question. "What's the simplest version of this without AI?" Often it's a SQL query plus a Slack notification. Build that first. If the gap between the simple version and the full vision is genuinely useful, that's the case for AI. If the simple version solves 80% of the value, that's the right scope. The mistake to avoid: AI-washing — adding LLM calls to a system that doesn't need them, just because LLM-everything is exciting. The cost is real complexity, real budget, real maintenance. A senior signal in interview: framing AI as a tool for specific problems, not a universal solution.

### Q9. What's your approach to coaching colleagues on prompt engineering?

A 4-week structured arc. Week 1: basics — anatomy of a prompt, hands-on with their actual work, homework to rewrite an existing prompt. Week 2: iteration — how to refine a failing prompt, demo of three iterations on a real example. Week 3: advanced techniques — few-shot, chain-of-thought, structured output. Week 4: integration — how to put their prompt into n8n or a script. By the end, the colleague can build their own AI features. Multiply this across the org and you've turned a 290-person team into a 290-person AI-augmented team. That's the leverage of the consultant role.

### Q10. How do you think about ROI on AI projects?

Conservative estimation in three columns. Before-AI cost: people-hours times rate, plus error costs and time-to-resolution. After-AI cost: LLM tokens times price, plus build effort, plus maintenance, plus residual human time on edge cases. ROI is the delta. Always include the unglamorous costs: 10-30% of cases will still need humans, AI errors create rework, model upgrades require re-tuning prompts. Be conservative — an AI project that promises 90% labor reduction and delivers 60% is failed against expectations; one that promises 50% and delivers 60% is a hit. Set expectations low, beat them slightly.

---

## 35.11 Cheatsheet

```
   n8n FUNDAMENTALS
     - Open-source, self-hostable, visual workflow builder
     - Trigger node + action nodes connected in a graph
     - Postgres for state, Redis for queue, workers for execution
     - LLM nodes (direct + AI Agent) integrate Claude/GPT natively
     - Self-hosted = data residency for regulated companies

   PRAGMATIC DECISION RULE
     Simple automation > complex agent (unless you really need the agent)
     Off-the-shelf > custom build (unless fit-to-system matters)
     Lightweight > heavyweight (always)

   DISCOVERY SESSION (60-90 min)
     1. What we do (5)
     2. Pains and bottlenecks (20)
     3. Workflow walkthrough (20)
     4. AI hypotheses (20)
     5. Prioritize and commit (15)
     Output: one-page summary with metric and timeline

   8 COMMON AI OPPORTUNITY PATTERNS
     Triage / classification
     Drafting and templating
     Summarization
     Question answering over docs
     Data enrichment
     Automated investigation
     Translation / localization
     Code review / linting beyond rules

   PLAYBOOKS TO WRITE FIRST
     1. How to write a good system prompt
     2. When to use which LLM
     3. How to integrate AI safely with internal data
     4. Prompt engineering by team type

   COACHING ARC (4 weeks)
     1. Basics
     2. Iteration
     3. Advanced techniques
     4. Integration

   ROI THREE COLUMNS
     Before-AI cost (people-hours × rate)
     After-AI cost (tokens + build + maintenance + residual humans)
     Delta + payback period (be conservative)

   STAKEHOLDER POSTURE
     With skeptics: address the actual concern with data
     With enthusiasts: ask "what's the simplest version?"
     Always: prefer leftmost-option-that-works
```

---

End of Chapter 35. Continue to **[Chapter 36 — Fintech AI Compliance](36_fintech_ai_compliance.md)** or back to **[Chapter 00 — Master Index](00_index.md)**.
