# Architecture Decisions & Rationale

This document serves as the living **Architecture Decision Record (ADR)** for the AgentForge project. It details the core technical choices, trade-offs, and design rationales made during the development of this competitive intelligence pipeline.

---

## 1. Orchestration: Why LangGraph instead of CrewAI or AutoGen?

* **Context:** We needed a framework to manage a multi-turn, state-sharing, five-agent collaborative pipeline.
* **Decision:** Selected **LangGraph** as the core orchestration framework.
* **Rationale:**
  * **Deterministic State Flow:** Alternative frameworks like CrewAI and AutoGen manage state implicitly through conversation history or autonomous agent-to-agent delegation. While great for simple chats, this is fragile for complex, multi-stage pipelines. LangGraph utilizes a first-class `StateGraph` with a typed schema, ensuring exactly how variables flow through nodes.
  * **Routing Control:** LangGraph allows conditional edges with deterministic routing functions. We can audit, trace, and inspect exactly why the Supervisor routed a task to the Analyst vs. the Writer.
  * **First-Class Human-in-the-Loop:** LangGraph's checkpoint-based architecture enables clean interrupt-and-resume. The graph pauses naturally at the approval gate without holding system processes or server threads hostage.

---

## 2. Observability: Why Langfuse Cloud instead of Self-Hosted?

* **Context:** We needed high-fidelity trace logging and step-by-step LLM call metrics.
* **Decision:** Migrated from a local self-hosted Langfuse + PostgreSQL stack to **Langfuse Cloud**.
* **Rationale:**
  * **Resource Efficiency:** Running a PostgreSQL database and a Next.js (Langfuse) application locally consumes 1.5 GB to 2.0 GB of RAM and continuous CPU cycles. Using the cloud hosting keeps our local development environment extremely lightweight.
  * **Resolving Local Conflicts:** Local setups often suffer from port clashes (like port `3000` being already occupied by other applications).
  * **Zero Maintenance & Cost:** Langfuse Cloud features a highly generous free tier (50,000 traces/month), eliminating database backups, schema updates, or local connection diagnostics.

---

## 3. Web Search: Why a Multi-Engine Resilient Fallback System?

* **Context:** The pipeline requires real-time information retrieval from the web, which can be prone to API rate limits, transient network issues, or lack of developer API keys.
* **Decision:** Implemented a three-tiered dynamic fallback search tool (`src/tools/web_search.py`).
* **Rationale:**
  * **Exa (1st Choice):** Selected as primary because Exa's neural/semantic search algorithm is custom-tailored for LLMs rather than humans, returning far cleaner text for prompt insertion.
  * **Tavily (2nd Choice):** Used as the next premium fallback, optimized specifically for RAG and agent workflows.
  * **DuckDuckGo (Zero-Config Fallback):** Used if no keys are provided (or during rate limits/outages). It is **100% free and keyless**, making local testing out-of-the-box frictionless without requiring third-party accounts.

---

## 4. Packaging: Why `pyproject.toml` instead of standard `requirements.txt`?

* **Context:** We needed to define package configurations, dev dependencies, and the install environment.
* **Decision:** Adopted standard Python packaging (`pyproject.toml` with `pip install -e ".[dev]"`).
* **Rationale:**
  * **Editable Local Importability:** Allows installing the local codebase as a packaged module. You can import modules cleanly (e.g. `import src`) and changes in your `src/` directory are immediately live without needing to rebuild or reinstall the package.
  * **Unified Meta-Config:** Instead of separating dependencies across multiple flat files (`requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `ruff.toml`), `pyproject.toml` houses all metadata, dependencies, linters (`ruff`), and testing utilities (`pytest`) in a single modern standard (PEP 517/621).

---

## 5. Security: Why a Dedicated MCP Workspace Container?

* **Context:** The Writer and Researcher agents need filesystem access to read and write outputs.
* **Decision:** Locked directory scope into a standalone **Model Context Protocol (MCP) server container** (`mcp_server/server.py`).
* **Rationale:**
  * **Security Sandboxing:** Giving LLMs direct write permissions on the host system is highly dangerous. By spinning up the MCP server in a isolated Docker container with a bounded volume `/workspace`, the LLM has absolutely no capability to inspect the host machine or read sensitive environment configurations outside the sandbox.
  * **Path-Traversal Protection:** The MCP server strictly validates relative paths to prevent security escapes (e.g. `../../etc/passwd`).

---

## 6. Budgeting: Why Per-Agent Token Budgets?

* **Context:** Large Language Models can produce highly wordy responses, quickly blowing past context limits and racking up massive API bills.
* **Decision:** Configured strict per-agent token budgets via a `TokenBudgetGuard` middleware.
* **Rationale:**
  * **Supervisor (4,000 tokens):** Kept tight because the supervisor only makes brief routing decisions. Giving it a large context wastes input tokens.
  * **Researcher (8,000 tokens):** Set higher to allow feeding in extensive raw web search fragments.
  * **Writer (10,000 tokens):** Provided the largest budget to give it enough runway to output complete, 1,000+ word markdown reports without truncation.

---

## 7. Pipeline Topology: Why 5 Agents (Supervisor + 4 Specialists)?

* **Context:** Designing the organizational structure of the pipeline.
* **Decision:** Settled on a 5-agent "Hub-and-Spoke" topology consisting of **1 Supervisor (router) and 4 Specialists (Researcher, Analyst, Writer, Reviewer)**.
* **Rationale:**
  * **Why not a single agent? (The "Monolith" trap):** Asking a single LLM call to do the research, perform the strategic analysis, write a 1,500-word business-standard report, and audit its own quality leads to **cognitive overload**. The model skips details, halluncinates facts to save tokens, and produces generic content. By breaking it into specialized roles, each agent focuses on **one singular task**, maximizing output quality.
  * **Why not more agents (e.g., 10 or 15)? (The "Over-Engineering" trap):** Every additional agent increases execution latency, API token costs, and compounding graph routing errors. A 5-agent setup provides the perfect equilibrium—isolating distinct skills (gathering, analyzing, writing, auditing) while keeping execution times under 60 seconds and costs under $0.08 per task.
  * **How they communicate (Hub-and-Spoke state-sharing):** 
    Rather than letting agents spam each other with unstructured chat history (which pollutes context windows), they communicate via a **centralized, typed state schema (`PipelineState`)** managed by LangGraph:
    1. The **Researcher** writes findings to `state.research_findings`.
    2. The **Analyst** reads those findings, processes them, and writes structured categories to `state.analysis_results`.
    3. The **Writer** reads the analysis categories and drafts `state.draft_report`.
    4. The **Reviewer** reads the draft and appends structured comments to `state.review_feedback`.
    5. The **Supervisor** reads these state parameters at every step to decide what node to trigger next.
    
    This structured data-flow prevents memory pollution, keeps API inputs highly optimized, and allows any node to be completely inspectable and hot-swappable.

