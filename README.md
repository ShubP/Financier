# 💸 Financier — AI Finance Assistant

> Democratizing financial literacy through an agentic, multi-agent AI.

Financier is a **multi-agent, retrieval-augmented conversational finance
educator**. A router classifies each message and hands it to a specialist
agent — Q&A, Portfolio, Market, or Goal — orchestrated with **LangGraph** and
powered by **Claude on Amazon Bedrock** (with direct-Anthropic, Gemini, and an
offline-mock fallback). It ships with a **Streamlit** multi-tab UI, a curated
finance knowledge base (RAG), live market data, and deterministic
portfolio/goal math.

It is built to **run out of the box with no credentials** (an offline demo
mode) and to scale up to real Claude models on your own AWS account.

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
| LLM | **Claude on AWS Bedrock** | A provider abstraction also supports the direct Anthropic API, Gemini, and an offline mock. |
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
credentials (below) for real Claude answers.

One-off CLI (no Streamlit):

```powershell
python run.py "What is dollar-cost averaging?"
```

---

## ☁️ Run on AWS Bedrock (primary backend)

Financier is configured to run Claude through **Amazon Bedrock** on your own AWS
account (`llm.provider: bedrock` in `config.yaml`).

**1. Install the AWS CLI** (v2.13.23+) and configure credentials:

```bash
aws configure                 # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
aws sts get-caller-identity   # verify it works
```

Credentials from `aws configure` (in `~/.aws/credentials`) are picked up
automatically — you don't need to put keys in `.env`.

**2. Enable Claude model access** in the AWS Console →
**Bedrock → Model access** → request access to the Anthropic Claude models in
your region. Availability varies by region (`us-east-1` and `us-west-2` are
common). See `aws bedrock list-foundation-models --by-provider anthropic
--query "modelSummaries[*].modelId"`.

**3. (Optional) Pick models / region** in `config.yaml`:

```yaml
llm:
  provider: bedrock
  bedrock:
    region: us-east-1
    models:
      router:  "global.anthropic.claude-haiku-4-5-20251001-v1:0"
      default: "global.anthropic.claude-sonnet-4-6"
```

The defaults use **global** inference-profile IDs (cross-region routing, no
pricing premium). The router uses **Haiku 4.5** (cheap classification) and the
specialists use **Sonnet 4.6** (good quality/cost balance). Swap the default to
`global.anthropic.claude-opus-4-6-v1` for top quality, or a `us.`-prefixed ID
for region-pinned routing.

**4. IAM** — the principal needs `bedrock:InvokeModel` (and
`bedrock:InvokeModelWithResponseStream`) on the Claude model resources, plus
`bedrock:ListFoundationModels`. The AWS managed policy `AmazonBedrockFullAccess`
covers this for getting started.

**5. Run it.** The sidebar will show `LLM provider: bedrock`.

> **Cost:** global Sonnet 4.6 is ~$3 / $15 per 1M input/output tokens and Haiku
> 4.5 ~$1 / $5; an educational chat session costs cents. Enable
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
