# 💸 Financier — AI Finance Assistant

> Democratizing financial literacy through an agentic, multi-agent AI.

Financier is a **multi-agent, retrieval-augmented conversational finance
educator**. A router classifies each message and hands it to a specialist
agent — Q&A, Portfolio, Market, or Goal — orchestrated with **LangGraph** and
powered by **Amazon Bedrock** (Amazon Nova by default; Claude, Gemini, and an
offline-mock fallback). It ships with a **Streamlit** multi-tab UI, a curated
finance knowledge base (RAG), live market data, and deterministic
portfolio/goal math.

It is built to **run out of the box with no credentials** (an offline demo
mode) and to scale up to real models on your own AWS account.

---

## ✨ Features

- **💬 Chat** — ask anything about investing; the router sends it to the right
  specialist and shows which agent answered and from which sources.
- **📊 Portfolio** — enter holdings and get allocation, a 0–100 diversification
  score, a risk level, concentration, and gain/loss — with charts and a
  plain-English explanation.
- **📈 Market** — live quotes and a 6-month trend chart for any ticker
  (yfinance, with an Alpha Vantage fallback).
- **🎯 Goals** — savings/retirement projections (compound growth), an
  "on track?" status, the required monthly contribution, and a projection
  chart.
- **🧠 Grounded & safe** — every substantive answer cites its sources and
  carries a clear "education, not advice" disclaimer.

---

## 🏗 Architecture

```
                         ┌──────────────────────────┐
   user message ───────► │   Router agent (LLM)     │  intent classification
                         └──────────┬───────────────┘
              ┌─────────────┬───────┼────────┬─────────────┐
              ▼             ▼       ▼        ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
        │   Q&A    │ │  Market  │ │Portfol.│ │  Goal  │ │Smalltalk │
        │  (RAG)   │ │ (quotes) │ │ (math) │ │ (math) │ │          │
        └────┬─────┘ └────┬─────┘ └───┬────┘ └───┬────┘ └────┬─────┘
             └────────────┴───────────┴──────────┴───────────┘
                                   ▼
                           ┌──────────────┐
                           │   Finalize   │  guarantee a reply + disclaimer
                           └──────┬───────┘
                                  ▼
                         response + sources + trace
```

A **LangGraph `StateGraph`** drives the flow; a single typed `AgentState` is
threaded through every node.

| Agent | Role |
|-------|------|
| **Router** | Classifies the query into one intent (structured output). |
| **Finance Q&A** | Answers concept questions grounded in the RAG knowledge base, with source attribution. |
| **Market Analysis** | Extracts tickers, fetches live quotes + 6-month history, narrates the move. |
| **Portfolio Analysis** | Computes allocation, diversification, risk, and gains, then explains them. |
| **Goal Planning** | Projects savings goals (compound growth), reports "on track?" and the required monthly contribution. |
| **Smalltalk** | Greetings / "what can you do" handling. |

### Why these technologies

| Concern | Choice | Notes / alternatives |
|--------|--------|----------------------|
| Orchestration | **LangGraph** | Typed `StateGraph` with conditional routing. |
| LLM | **Amazon Bedrock (Amazon Nova)** | A provider abstraction also supports Claude (Bedrock or direct), Gemini, and an offline mock. |
| RAG | **FAISS + sentence-transformers**, TF-IDF fallback | Auto-detects: semantic if the extras are installed, else lexical. |
| Market data | **yfinance** (+ Alpha Vantage) | No key needed for yfinance; deterministic mock keeps it offline-safe. |
| Structured output | **Pydantic** | Provider-agnostic JSON coercion + validation. |
| UI | **Streamlit** | Multi-tab. |
| Memory | **JSON profile store** | Optional **DynamoDB** backend. |

---

## 📁 Project structure

```
ai_finance_assistant/
├── config.yaml              # all runtime config (no secrets)
├── requirements.txt         # core deps (app runs on these, incl. boto3)
├── requirements-extras.txt  # FAISS / sentence-transformers / gemini
├── run.py                   # launcher + one-off CLI
├── Dockerfile
├── src/
│   ├── core/                # config, shared state, logging
│   ├── llm/                 # provider abstraction (bedrock/anthropic/gemini/mock)
│   ├── data/
│   │   ├── market_data.py   # quotes/history + caching + fallbacks
│   │   └── knowledge/       # curated finance-education docs (RAG corpus)
│   ├── rag/                 # chunking, retrievers, knowledge base
│   ├── finance/             # portfolio + projection math (pure, tested)
│   ├── agents/              # router + 4 specialist agents
│   ├── workflow/            # LangGraph graph
│   ├── memory/              # user-profile persistence
│   └── web_app/             # Streamlit UI
├── tests/                   # unit + workflow tests (offline)
├── data/                    # sample portfolio + saved profiles (gitignored)
└── docs/                    # architecture + deployment guides
```

---

## 🚀 Quickstart

```powershell
cd ai_finance_assistant

# 1. Create and activate a virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1            # Windows
# python3 -m venv .venv && source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
# (optional) semantic RAG + Gemini:  pip install -r requirements-extras.txt

# 3. Run the web app
streamlit run src/web_app/streamlit_app.py
```

With **no credentials**, Financier runs in **demo mode** (a built-in mock LLM +
mock market data) so the whole pipeline works for a walkthrough. Add AWS
credentials (below) for real model answers.

One-off CLI (no Streamlit):

```powershell
python run.py "What is dollar-cost averaging?"
```

---

## ☁️ Run on AWS Bedrock (primary backend)

Financier is configured to run on **Amazon Bedrock** using **Amazon Nova**
models by default (`llm.provider: bedrock` in `config.yaml`). Nova models are
Amazon's own and are enabled instantly with no third-party approval.

**1. Install the AWS CLI** (v2.13.23+) and configure credentials:

```bash
aws configure                 # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
aws sts get-caller-identity   # verify it works
```

Credentials from `aws configure` (in `~/.aws/credentials`) are picked up
automatically — you don't need to put keys in `.env`.

**2. Enable Claude model access** in the AWS Console →
**Bedrock → Model access** → request access to the **Amazon Nova** models (Nova Micro and Nova Lite) in
your region. Availability varies by region (`us-east-1` and `us-west-2` are
common). Anthropic Claude models now require separate approval, so Nova is
the simplest path. See `aws bedrock list-foundation-models --by-provider amazon
--query "modelSummaries[*].modelId"`.

**3. (Optional) Pick models / region** in `config.yaml`:

```yaml
llm:
  provider: bedrock
  bedrock:
    engine: converse
    region: us-east-1
    models:
      router:  "us.amazon.nova-micro-v1:0"
      default: "us.amazon.nova-lite-v1:0"
```

The defaults use **Amazon Nova**: the router uses **Nova Micro** and the
agents use **Nova Lite** -- both inexpensive and enabled instantly. If a `us.`
inference-profile id is rejected in your region, try the bare id (e.g.
`amazon.nova-lite-v1:0`). To use Claude instead, set `bedrock.engine: anthropic`
with Claude model ids (requires separate Anthropic approval).

**4. IAM** — the principal needs `bedrock:InvokeModel` (and
`bedrock:InvokeModelWithResponseStream`) on the chosen model's resources, plus
`bedrock:ListFoundationModels`. The AWS managed policy `AmazonBedrockFullAccess`
covers this for getting started.

**5. Run it.** The sidebar will show `LLM provider: bedrock`.

> **Cost:** Amazon Nova is very cheap -- Nova Micro about $0.035 / $0.14 and
> Nova Lite about $0.06 / $0.24 per 1M input/output tokens, so a chat session
> costs a fraction of a cent. Enable
> [Bedrock invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html)
> if you want an audit trail.

### Other backends

- **Anthropic direct API:** set `ANTHROPIC_API_KEY` and `llm.provider: anthropic`.
- **Google Gemini:** `pip install -r requirements-extras.txt`, set
  `GOOGLE_API_KEY`, and `llm.provider: gemini`.
- **Mock (offline):** `llm.provider: mock` — no dependencies or keys.

`llm.provider: auto` tries Anthropic → Bedrock → Gemini → mock based on what
credentials are present.

---

## 🐳 Docker

```bash
docker build -t financier .
docker run -p 8501:8501 --env-file .env financier     # http://localhost:8501
```

## 🗄 Optional DynamoDB profile store

Set `memory.backend: dynamodb` in `config.yaml` and create a table (default
`financier-profiles`) with a `user_id` string partition key. Falls back to the
local JSON store if DynamoDB or credentials are unavailable.

---

## 🧪 Testing

```bash
pip install pytest
pytest
```

Tests run **offline** (mock provider, deterministic market data) and cover
routing, portfolio math, goal projections, market-data fallbacks, RAG
retrieval, and an end-to-end workflow turn. The code is linted with `ruff`.

---

## ⚙️ Configuration reference

| Variable (`.env`) | Enables |
|----|----|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | Claude via **Bedrock** (or just use `aws configure`). |
| `ANTHROPIC_API_KEY` | Claude via the direct Anthropic API. |
| `GOOGLE_API_KEY` | Google Gemini provider. |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage market-data fallback. |

Everything else (models, RAG, market-data, memory) is in `config.yaml`. See
[docs/architecture.md](docs/architecture.md) and
[docs/deployment.md](docs/deployment.md).

---

## ⚠️ Limitations & future work

- The knowledge base ships with a focused set of beginner documents; add more
  markdown files under `src/data/knowledge/` to broaden coverage.
- Projections assume a constant return; real markets vary.
- Roadmap: News Synthesizer + Tax Education agents, a Model Context Protocol
  (MCP) server, conversation persistence, and richer evaluations.

## 📜 License & disclaimer

Released under the [MIT License](LICENSE).

Financier provides **general financial education only**. It is **not** financial,
investment, tax, or legal advice. Always consult a licensed professional before
making financial decisions.
