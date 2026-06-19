# Architecture

Financier is a multi-agent system orchestrated with LangGraph. This document
describes the components and how a request flows through them.

## Components

| Layer | Module | Responsibility |
|-------|--------|----------------|
| Config | `src/core/config.py` | Load `config.yaml` + `.env`; dotted access. |
| State | `src/core/state.py` | The `AgentState` TypedDict threaded through the graph. |
| LLM | `src/llm/` | Provider abstraction: Anthropic, Bedrock, Gemini, Mock. |
| Market data | `src/data/market_data.py` | Quotes/history with caching + fallback chain. |
| RAG | `src/rag/` | Chunking, retriever backends, knowledge base. |
| Finance math | `src/finance/` | Portfolio analytics + goal projections (pure). |
| Agents | `src/agents/` | Router + Q&A, Market, Portfolio, Goal. |
| Workflow | `src/workflow/graph.py` | The compiled LangGraph `StateGraph`. |
| Memory | `src/memory/` | User-profile persistence (JSON / DynamoDB). |
| UI | `src/web_app/streamlit_app.py` | Multi-tab Streamlit interface. |

## Request flow

1. **Router** classifies the message into one intent (`qa`, `market`,
   `portfolio`, `goal`, `smalltalk`) using structured output. On the mock
   provider this is keyword-based, so routing works offline.
2. A **conditional edge** sends state to exactly one specialist node.
3. The **specialist** does its work and writes `response`, `sources`,
   `agent_trace`, and any working data (`context`, `market`, `analysis`) back
   into shared state.
4. **Finalize** guarantees a non-empty response and appends the educational
   disclaimer.

```
router ──(intent)──► {qa | market | portfolio | goal | smalltalk} ──► finalize ──► END
```

## The LLM provider abstraction

`LLMProvider` exposes one abstract method, `complete()`, plus convenience
wrappers `reason()` (default model, adaptive thinking) and `classify()` (router
model). A provider-agnostic `structured()` coerces free text into a validated
Pydantic model, so structured output works the same across every backend.

`build_provider()` auto-detects an available backend in order
**Anthropic → Bedrock → Gemini → Mock**. The mock backend requires no
dependencies or keys and is the guaranteed fallback.

## RAG

Markdown documents under `src/data/knowledge/` are loaded, split into
overlapping paragraph-aware chunks, and indexed. The retriever is chosen at
build time:

- **SemanticBackend** — sentence-transformers embeddings + a FAISS
  inner-product index (requires `requirements-extras.txt`).
- **TfidfBackend** — scikit-learn TF-IDF + cosine similarity (always available).

Each retrieved chunk carries its source document title for attribution.

## Reliability

- **LLM:** provider auto-detection + mock fallback; per-call try/except wraps
  every model call in a uniform `LLMError`.
- **Market data:** yfinance → Alpha Vantage → deterministic mock; quotes and
  history are cached with a configurable TTL.
- **Workflow:** `ask()` wraps graph execution and returns a graceful fallback
  message instead of crashing the UI.
- **Memory:** DynamoDB backend degrades to JSON if boto3/AWS is unavailable.
