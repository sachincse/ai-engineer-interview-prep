# Chapter 37 — Upvest: Company Intelligence and Applied AI Role Prep

> **Why this chapter exists:** This is the company-specific layer for the Upvest Applied AI role in Berlin. It pairs with the generic technical chapters (34 — MCP, 35 — n8n + Consulting, 36 — Fintech Compliance) to give you a complete picture of what Upvest does, who interviews you, and how to position yourself. Treat this chapter as the playbook you read before the Upvest interview specifically; the rest of the library is the technical foundation.

---

## 37.1 Upvest in one paragraph

Upvest is a Berlin-headquartered fintech that provides **investment infrastructure as an API**. They are the layer that lets neo-banks, retail brokers, wealth management apps, and embedded finance products offer investment services to their own customers without having to build the regulatory, custody, and execution infrastructure from scratch. Think of them as "Stripe for investing" — they handle the regulated, hard-to-build core (a BaFin-licensed brokerage, a custody bank, MiFID II compliance, settlement, corporate actions) and expose it through clean APIs that other companies build on top of. As of 2026, they have ~290 employees, are BaFin and FCA regulated, and serve multiple high-profile B2B customers across Europe.

**Why this matters for the Applied AI role:** Upvest is itself a B2B platform serving other developers. Their internal AI strategy isn't about adding AI to a consumer product — it's about making 290 internal employees more productive by integrating AI tools deeply with internal systems. The role is closer to "internal CTO of AI" than "build AI features for users."

---

## 37.2 The Upvest product portfolio

What they build, in plain terms:

| Product / Capability | What it does |
|----------------------|--------------|
| **Brokerage Infrastructure** | Clients place orders through Upvest's API; Upvest routes to execution venues; settlement happens via Upvest's connections to clearing houses. |
| **Custody Bank** | Upvest holds securities and cash on behalf of end-customers in legally-segregated accounts, with full BaFin compliance. |
| **Investment Tax** | Tax reporting for German and other EU customers — typically the most painful part of investment services for B2B clients to build themselves. |
| **Crypto Investments** | Compliant access to cryptocurrencies through their licensed entity. |
| **Pension & Retirement** | Long-term savings products that meet regulatory requirements. |
| **Corporate Actions** | Handling dividends, stock splits, mergers — all the "after you bought the stock, now what?" complexity. |

**What this tells you for the interview:** Upvest's products are deeply technical and regulated. The AI you'd build doesn't directly touch customers — it serves the engineers, operations specialists, compliance team, and client-facing teams who build, run, and support these products.

---

## 37.3 Likely AI use cases at Upvest (your interview material)

Based on the product portfolio and the JD, here are the AI opportunities most likely on Upvest's roadmap. Use these as concrete examples in the interview.

### Engineering productivity

- **AI code review** integrated with their GitHub: pre-comment on PRs, catch common Python/TypeScript issues, ensure consistency across the codebase.
- **Documentation copilot**: when an engineer writes a new API endpoint, AI generates draft API docs for review.
- **Internal Q&A bot**: RAG over their Confluence + Slack + GitHub to answer "where is the code for X?" or "what's our policy on Y?".

### Operations productivity

- **Customer support draft generator**: B2B customer asks a technical question, AI drafts a response with code examples for the support engineer to review.
- **Incident summary**: when an incident happens, AI compiles a draft postmortem from Slack threads, PagerDuty events, and Datadog logs.
- **Reconciliation triage**: when settlement reconciliation finds discrepancies, AI categorizes them by likely cause and suggests next steps.

### Compliance productivity

- **Regulatory letter drafting**: BaFin or FCA queries get an AI-drafted response that the compliance team reviews.
- **Policy update analysis**: when a regulation changes, AI summarizes what's new and which internal policies need review.
- **Suspicious activity report (SAR) triage**: AI scores transactions for AML risk, surfacing the most concerning for human review.
- **KYC document validation**: AI extracts structured data from KYC documents (passports, utility bills) for the operations team to verify.

### Client-facing team productivity

- **B2B onboarding assistant**: when a new client integrates Upvest's API, AI guides them through common patterns and answers questions about their integration.
- **Sales enablement**: AI-drafted personalized outreach to prospects, RAG over case studies and product docs.
- **Account health digests**: weekly AI-generated summaries of each client's API usage, growth, and any issues.

### Data and analytics productivity

- **Natural language to SQL**: the data team gets ad-hoc requests; AI translates "show me weekly trade volume by client for March" into the right Snowflake query.
- **Anomaly explanation**: when a metric spikes, AI investigates by querying related tables and proposes explanations.

**The interview move:** when asked "what would you build first?", pick ONE of these and propose a 90-day plan. Don't try to boil the ocean.

---

## 37.4 The role analysis — what Upvest is actually hiring for

Reading the JD carefully:

```
   "Take ownership of AI implementation across Upvest"
   → Senior IC role, you OWN the AI strategy and implementation

   "Identify where AI can create the most impact"
   → Discovery and prioritization across 290+ people

   "Support the selection and integration of the right tools"
   → Vendor evaluation, build-vs-buy decisions

   "Deliver solutions that teams actively use"
   → Adoption matters more than technical novelty

   "Hands-on senior individual contributor"
   → No reports; influence through delivery, not hierarchy

   "Direct influence on how 290+ people work every day"
   → High-leverage; one good integration affects hundreds
```

The role is one part **builder** (MCP servers, n8n workflows, custom integrations), one part **consultant** (discovery sessions, playbooks, coaching), and one part **strategist** (where to invest, what to skip). The senior IC framing means you should be ready to talk about all three.

### What Upvest is NOT hiring for

- **Not** a research scientist — this is implementation, not LLM training.
- **Not** a manager — no team, you're an IC.
- **Not** a single-product builder — you're working horizontally across teams.
- **Not** a generalist software engineer — they want AI specialization.

---

## 37.5 Likely interview process

Based on Upvest's company size and German tech-company patterns, expect:

```
   1. Recruiter screen (30 min)
      → Why Upvest, salary expectations, work authorization,
        notice period

   2. Hiring manager interview (60 min)
      → Your background, your AI work, why this role,
        what you'd build first

   3. Technical deep dive (60-90 min)
      → MCP, LLM integration, agent architectures, prompt
        engineering, security/compliance

   4. System design (60 min)
      → Design an AI system for one of their use cases
        (likely customer support or internal Q&A)

   5. Cultural / values interview (45 min)
      → Working with non-technical teams, handling skeptics,
        operating in a regulated environment

   6. Final / executive round (30-45 min)
      → Senior leadership; check on strategic thinking and fit
```

For each of these:

### The recruiter screen

Be ready with:
- Why Upvest (see §37.7)
- Why Berlin (relocation? remote? hybrid?)
- Salary expectation in EUR
- Notice period
- Visa status (relevant for relocation)

### The hiring manager interview

The hiring manager (likely the Head of Engineering or CTO) wants to understand: are you actually senior enough to own this? Do you have judgment about when to use AI and when not to? Can you work with non-engineers?

Have ready:
- A 90-second tell-me-about-yourself with the AI consultant angle prominent
- 2-3 stories about AI projects where you decided NOT to use AI / agents
- 2-3 stories about working with non-technical teams
- Your "what I'd build first at Upvest" answer

### The technical deep dive

This is the chapter-34-and-32 territory. Expected topics:

- **MCP architecture** — you'll definitely be asked about this. Cover client/server/host, transports, tools/resources/prompts. Have ready: when to use MCP vs traditional tool calling.
- **LLM integration patterns** — how do you connect Claude to internal systems? What's your DPA strategy? How do you handle PII?
- **Tool design** — given a hypothetical (Upvest's compliance tool, say), design 5-7 tools. Names, descriptions, schemas, security model.
- **Prompt engineering** — show fluency. How do you handle prompt injection? How do you test prompt quality?
- **Cost / latency tradeoffs** — Sonnet vs Haiku vs Opus, prompt caching, batching.

### The system design

Likely prompt: "Design an internal Q&A bot for Upvest." Or: "Design AI-augmented compliance review." Use the framework from Chapter 33: clarify, capacity-math, architecture, hard parts, evaluation, follow-ups. Mention MCP servers as integration points; mention BaFin/FCA-aware design choices.

### The cultural / values interview

Topics:
- Coaching colleagues on AI tools (the consultant skill)
- Operating in regulated environments (you're not slowed down by compliance, you incorporate it from the start)
- Working with skeptical stakeholders
- Pragmatism (when NOT to use AI)

### The final round

Strategic thinking. They want to know: would you be a leader on this team in 12 months? What's your 1-year vision for AI at Upvest?

---

## 37.6 The 12 questions Upvest will most likely ask

Drilled with answers calibrated for the role.

### Q1. Why are you interested in this role at Upvest?

Three reasons. First, the role itself: a senior IC owning AI strategy across the entire company, with explicit budget for both build and consult. That's unusual — most roles are pure build, and the leverage of also coaching colleagues to use AI well multiplies impact. Second, the regulatory environment: I want to do AI work that has to satisfy real constraints — BaFin and FCA aren't optional, GDPR isn't optional. That forces engineering rigor I find rewarding. Third, the platform shape: Upvest is itself B2B infrastructure, which means internal AI work is highly leveraged — improving 290 employees' productivity affects every B2B customer they serve.

### Q2. Walk me through a project where you decided NOT to use AI.

Show pragmatism. Example: "At [previous role], a team asked me to build an AI agent to triage their support emails. After watching their workflow, I realized 95% of the emails fit five categories that a hardcoded routing rule would handle correctly. The remaining 5% genuinely needed human judgment. I built the routing rule in two days instead of a four-week AI agent project. The team was happier with the simple solution because it was instantly debuggable, and it freed engineering budget for problems that actually needed AI." This story signals the pragmatism the JD asks for.

### Q3. How would you start at Upvest?

A 30-60-90-day plan. Days 1-30: listen. Sit with each team — engineering, operations, compliance, sales, customer success. Run discovery sessions, identify top-5 high-leverage use cases. No code yet. Days 30-60: ship one win. Pick the simplest high-leverage use case from discovery, build it end-to-end, deploy with a target team. Document the rollout. Days 60-90: scale the pattern. Take the win and turn it into a reusable template — an MCP server other teams can use, or an n8n workflow other teams can clone. By day 90, I'd have one production AI feature, one playbook published, and a roadmap for the next quarter. The principle: ship one meaningful thing in the first quarter; don't try to revolutionize everything.

### Q4. Tell me about MCP and why it's relevant to Upvest.

(Use Chapter 34 §34.11 Q1 verbatim, then add:) For Upvest specifically, MCP is the right primitive for an internal AI strategy. Each internal system gets its own MCP server: GitHub for code, Slack for comms, Postgres for data, custom servers for the brokerage and custody platforms. Engineers can use these from Claude Desktop personally. A central internal AI portal can use them for all employees. Custom agents can use them for automated workflows. One server, many consumers — that's the leverage. Critically, the servers run on Upvest's infrastructure with their own scoped credentials, which solves the data residency and audit requirements BaFin cares about.

### Q5. How do you handle data privacy with LLMs in a BaFin-regulated firm?

Multiple layers. First, vendor selection: Claude on AWS Bedrock in the EU-Frankfurt region keeps data in the EU and is covered by AWS's DPA. Second, data minimization in prompts: don't include personal data unless necessary; when needed, fetch via tool calls where the tool layer can apply ACL and redaction. Third, prompt-injection defense: hardened system prompts, pre-input filters, output filters that catch leaked PII. Fourth, audit logging: every AI call logged with who-asked-what-when, immutable for the regulatory retention period. Fifth, GDPR rights: deletion pipelines, subject access request handling, automated-decision-making safeguards (Article 22). For specifically high-risk AI under the EU AI Act — credit scoring, KYC — I'd add formal model cards, conformity assessment, and human-in-the-loop on decisions.

### Q6. How would you decide whether to build a custom MCP server, use an off-the-shelf one, or use n8n?

Decision tree. If multiple AI applications need this integration and it's high-leverage — build a custom MCP server, even though it takes more engineering. If a community-maintained MCP server already exists for the system (GitHub, Slack, Postgres) — use it; don't reinvent. If the use case is a single workflow that fits "if X happens, do Y, then Z" — n8n is faster to ship and easier for non-engineers to read. If the use case is a single ML application with tightly-coupled tools — direct in-process tool calling can be the simplest. The pragmatic principle: leftmost option that works.

### Q7. Tell me about coaching a non-technical colleague on AI.

Use a real story. Example: "At ResMed, I worked with a clinical operations specialist who was using ChatGPT for free-text query reformulation but getting inconsistent results. Over four 30-minute sessions, I taught her the structure of system prompts, how to add few-shot examples, and how to iterate on a failing prompt. By session four she had built a structured prompt that improved her output quality measurably and saved her about 5 hours per week. She then taught two other people in her team. That's the multiplier effect of coaching — one investment, three people upgraded, and a pattern that could be documented for the rest of the team."

### Q8. How would you measure the success of your role at Upvest?

Three layers. Output: how many AI features shipped, how many MCP servers deployed, how many playbooks published. This is the easiest to measure but the least important. Adoption: how many internal users actively use AI tools, how often, sustained over time. This is the truest measure — features nobody uses are failed. Business impact: hours saved across the organization, errors reduced, decision quality improved. This is the hardest to measure but what stakeholders care about. I'd commit to quarterly metrics on all three, with concrete targets agreed with my manager and stakeholders.

### Q9. What would you change about how AI is being used in industry today?

The thing I'd change: less agent-everything, more "boring AI." Most useful AI work isn't a multi-step autonomous agent — it's a single LLM call wired into a well-defined workflow. The industry overweights agentic complexity because it's exciting; it underweights the simpler patterns that actually move the needle. At Upvest specifically, I'd default to lightweight automations and reserve agent complexity for genuinely needs-it problems. The pragmatic principle scales: ship the simple thing, learn what's needed, then add complexity if measured impact justifies it.

### Q10. How would you handle a stakeholder who's skeptical about AI?

Listen first to the actual concern. Reliability concerns: address with concrete failure rates from a pilot. Compliance concerns: walk through the specific design choices that satisfy BaFin/GDPR (data residency, audit logs, human review). Job displacement concerns: discuss how the work changes rather than headcount changing. Skeptics, properly addressed, often become the strongest champions because they've already mapped out the failure modes that enthusiasts ignore. The opposite challenge — handling enthusiasts who want AI everywhere — is just as important; for those, ask "what's the simplest version of this without AI?" If the simple version solves 80% of the value, that's the right scope.

### Q11. What's the role of the EU AI Act in your work?

The EU AI Act creates a tiered set of obligations based on risk level. For Upvest, AI used in credit scoring, suitability assessment, KYC verification — all high-risk under the Act. The implications: I document AI systems thoroughly (model cards, intended use, limitations), I ensure human oversight on consequential decisions, I build in transparency so customers know when AI was involved, I monitor for accuracy and fairness, and for systems classified as high-risk I'd participate in conformity assessment. For lower-risk AI like internal productivity tools, the obligations are lighter — mostly transparency that users are interacting with AI. The Act doesn't slow me down; it shapes how I design from day one.

### Q12. Do you have any questions for us?

Three questions. First: "What's the biggest internal pain point that you'd want the Applied AI hire to address in their first 90 days?" — this surfaces what they actually care about and lets you respond intelligently. Second: "How is the AI strategy at Upvest balanced between internal productivity and customer-facing features?" — signals you understand the dual nature of the role. Third: "How does the AI function fit with the existing engineering and compliance teams — is there a senior leadership owner, and how do decisions get made?" — checks the org-political reality you'd be operating in.

---

## 37.7 Why Upvest, why Berlin, why now (the personal answers)

These are the questions that don't have right answers, only honest ones. Have your version of each ready.

### Why Upvest specifically?

Three real reasons:
1. The role shape — senior IC owning AI strategy, with both build and consult components — is rare and matches my career interests.
2. Regulated fintech is a place where AI engineering rigor matters and is rewarded. The BaFin/FCA constraint is a feature, not a bug.
3. B2B platform with 290 employees is a sweet spot — large enough to have real internal complexity, small enough that one person can have outsized influence.

### Why Berlin?

Honest answer about the city:
- European tech hub with a deep talent pool
- Reasonable cost of living vs San Francisco / London
- Regulatory clarity (German engineering culture aligns with my approach)
- Lifestyle considerations: walkable, public transit, work-life balance

### Why now?

- The role is the natural next step from your current work
- The MCP / Claude ecosystem is where the AI tooling industry is going; being on the frontier of that is timely
- Personal life situation aligns with relocation

### The compensation conversation

Berlin senior AI engineering compensation in 2026:
- Base salary range: €100K–€140K for senior IC, €140K–€180K for staff/principal
- Equity/RSU: smaller vs SF, sometimes a meaningful cash bonus instead
- Benefits: 28–30 days vacation, health insurance, public pension contributions
- Total comp typical for this role: €120K–€160K base + 10–20% bonus

Have your number in EUR ready, framed as "I'm targeting around €X based on my research; flexible based on the full package."

---

## 37.8 The 60-second tell-me-about-yourself for Upvest

Tighten this version of your standard pitch with the consultant angle prominent:

> "I'm Sachin — eight years productionizing ML and LLM systems across fintech, healthcare, and enterprise. At TrueBalance today I own a real-time XGBoost Lambda for credit-risk prediction with p99 under 500ms, plus a Claude-powered developer platform that integrates Jira, GitHub, Athena, and Jenkins for our ML team. Before that, two and a half years at ResMed building their MLOps platform that hosted eight models in six months, plus a RAG-based clinical chatbot. The thread across all of it is making AI useful in production environments with real constraints — regulated data, latency budgets, audit requirements. The Upvest role appeals to me because it combines what I'm strong at — building production AI integrations — with what I want to grow into — formally consulting across teams to identify high-leverage AI opportunities. The MCP-centric architecture and the BaFin-regulated environment are exactly where I want to do the next chapter of my work."

Memorize the first ten and last ten words. Improvise the middle.

---

## 37.9 The three questions to ask Upvest at the end

(Repeated from §37.6 Q12, calling out that these go at the end of the technical or hiring-manager interview.)

1. "What's the biggest internal pain point that you'd want the Applied AI hire to address in their first 90 days?"
2. "How is the AI strategy at Upvest balanced between internal productivity and customer-facing features?"
3. "How does the AI function fit with the existing engineering and compliance teams — is there a senior leadership owner, and how do decisions get made?"

These signal three things: outcome-orientation, strategic awareness, and organizational savvy. Save salary/visa/relocation questions for the recruiter or HR round, not the technical or hiring-manager rounds.

---

## 37.10 The cheatsheet — what to carry into the interview

```
   UPVEST IS:
     B2B fintech infrastructure ("Stripe for investing")
     Berlin HQ, 290+ employees, BaFin + FCA regulated
     Products: brokerage, custody, tax, crypto, pensions, corporate actions

   THE ROLE IS:
     Senior IC owning AI strategy across the company
     Build + Consult + Strategist split
     Hands-on, no reports, high leverage across 290 people

   JD KEYWORDS TO HIT
     - MCP (Model Context Protocol) — definitely asked
     - n8n / lightweight automation
     - "in-house AI consultant"
     - BaFin / FCA / data privacy / compliance / audit
     - Pragmatic decisions (not over-engineering)
     - Coaching / playbooks / discovery sessions

   30-60-90 DAY PLAN
     30: Discovery sessions across all teams; pick top-5 use cases
     60: Ship ONE meaningful AI feature with a target team
     90: Turn the win into a reusable pattern; quarterly roadmap

   THE THREE INTERVIEW SUPER-MOVES
     1. Cite their products by name (brokerage, custody, tax)
     2. Map technical depth to their constraints (BaFin, MiFID, GDPR)
     3. Show pragmatism — tell at least one story of NOT using AI

   TOPICS YOU MUST OWN
     - MCP architecture and tradeoffs
     - LLM tool design principles
     - Prompt engineering and injection defense
     - Workflow automation patterns (n8n)
     - Fintech compliance basics (Chapter 36)
     - Discovery session structure
     - ROI estimation and stakeholder management

   THREE CLOSING QUESTIONS
     1. What's the biggest pain you want addressed in 90 days?
     2. How is AI balanced between internal and customer-facing?
     3. How does AI fit with engineering + compliance org?
```

---

End of Chapter 37. Continue back to **[Chapter 00 — Master Index](00_index.md)** to navigate other chapters.
