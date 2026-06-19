"""Financier — Streamlit multi-tab UI.

Run with:  streamlit run src/web_app/streamlit_app.py

Tabs: Chat (multi-agent conversation), Portfolio (allocation/risk),
Market (live quotes/trends), Goals (savings projections), About.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make ``src`` importable when launched via ``streamlit run``.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from src.core.config import load_config  # noqa: E402
from src.data.market_data import get_market_data  # noqa: E402
from src.finance.portfolio import Holding, analyze_portfolio  # noqa: E402
from src.finance.projections import GoalInput, plan_goal  # noqa: E402
from src.memory.profile_store import get_profile_store  # noqa: E402
from src.workflow.graph import get_assistant  # noqa: E402

USER_ID = "local-user"

_SAMPLE_HOLDINGS = [
    {"symbol": "VTI", "shares": 30, "asset_class": "equity"},
    {"symbol": "VXUS", "shares": 25, "asset_class": "equity"},
    {"symbol": "BND", "shares": 40, "asset_class": "bond"},
    {"symbol": "VNQ", "shares": 10, "asset_class": "reit"},
]


@st.cache_resource(show_spinner="Loading Financier…")
def _assistant() -> Any:
    return get_assistant()


@st.cache_resource
def _market() -> Any:
    return get_market_data()


@st.cache_resource
def _store() -> Any:
    return get_profile_store()


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def _df_to_holdings(df: pd.DataFrame) -> list[dict[str, Any]]:
    holdings: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol") or "").strip()
        if not symbol:
            continue
        asset_class = (
            str(row.get("asset_class") or "equity").strip().lower()
            or "equity"
        )
        holding: dict[str, Any] = {
            "symbol": symbol.upper(),
            "asset_class": asset_class,
        }
        shares = _to_float(row.get("shares"))
        value = _to_float(row.get("value"))
        cost = _to_float(row.get("cost_basis"))
        if shares is not None:
            holding["shares"] = shares
        if value is not None:
            holding["value"] = value
        if cost is not None:
            holding["cost_basis"] = cost
        holdings.append(holding)
    return holdings


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "profile" not in st.session_state:
        st.session_state.profile = _store().get(USER_ID)


def _answer_meta(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace": result.get("agent_trace", []),
        "sources": result.get("sources", []),
    }


def _render_meta(meta: dict[str, Any]) -> None:
    with st.expander("How Financier answered"):
        trace = meta.get("trace") or []
        if trace:
            st.caption("Route: " + " → ".join(trace))
        if meta.get("sources"):
            st.caption("Sources: " + ", ".join(meta["sources"]))


# --- tabs ---
def render_chat() -> None:
    st.subheader("💬 Chat with Financier")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                _render_meta(msg["meta"])

    prompt = st.chat_input(
        "Ask about investing, your portfolio, a stock, or a goal…"
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = _assistant().ask(
                prompt,
                history=history,
                profile=st.session_state.profile,
            )
        answer = result.get("response", "")
        st.markdown(answer)
        meta = _answer_meta(result)
        _render_meta(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )


def render_portfolio() -> None:
    st.subheader("📊 Portfolio")
    st.caption(
        "Enter holdings (shares + symbol, or a direct value). Prices "
        "for share-based rows are fetched live."
    )
    profile = st.session_state.profile
    rows = profile.get("holdings") or _SAMPLE_HOLDINGS
    df = pd.DataFrame(rows)
    for col in ["symbol", "shares", "value", "asset_class", "cost_basis"]:
        if col not in df.columns:
            df[col] = None
    df = df[["symbol", "shares", "value", "asset_class", "cost_basis"]]

    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="holdings_editor",
    )
    holdings = _df_to_holdings(edited)

    save_col, explain_col = st.columns(2)
    if save_col.button("💾 Save holdings"):
        profile["holdings"] = holdings
        _store().save(USER_ID, profile)
        st.success("Holdings saved to your profile.")

    if not holdings:
        st.info("Add at least one holding to see the analysis.")
        return

    analysis = analyze_portfolio(
        [Holding.model_validate(h) for h in holdings], market=_market()
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total value", f"${analysis.total_value:,.0f}")
    c2.metric(
        "Diversification", f"{analysis.diversification_score:.0f}/100"
    )
    c3.metric("Risk", analysis.risk_level)
    c4.metric("Top position", f"{analysis.concentration_top_pct:.0f}%")

    if analysis.allocation_by_class:
        pie = px.pie(
            names=list(analysis.allocation_by_class.keys()),
            values=list(analysis.allocation_by_class.values()),
            title="Allocation by asset class",
        )
        st.plotly_chart(pie, use_container_width=True)

    hold_df = pd.DataFrame([h.model_dump() for h in analysis.holdings])
    if not hold_df.empty:
        bar = px.bar(
            hold_df,
            x="symbol",
            y="weight_pct",
            title="Weight by holding (%)",
        )
        st.plotly_chart(bar, use_container_width=True)

    if analysis.notes:
        st.markdown("**Notes**")
        for note in analysis.notes:
            st.markdown(f"- {note}")

    if explain_col.button("🧠 Ask Financier to explain"):
        with st.spinner("Analyzing…"):
            result = _assistant().ask(
                "Analyze my portfolio.",
                profile={"holdings": holdings},
            )
        st.markdown(result.get("response", ""))


def render_market() -> None:
    st.subheader("📈 Market")
    raw = st.text_input(
        "Symbols (comma-separated)", value="AAPL, MSFT, GOOGL"
    )
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not symbols:
        st.info("Enter at least one ticker symbol.")
        return

    quotes = [_market().get_quote(s) for s in symbols]
    qdf = pd.DataFrame([q.model_dump() for q in quotes])
    st.dataframe(
        qdf[
            [
                "symbol",
                "price",
                "change",
                "change_pct",
                "currency",
                "source",
                "as_of",
            ]
        ],
        use_container_width=True,
    )
    src = ", ".join(sorted({q.source for q in quotes}))
    st.caption(f"Data source: {src}")

    pick = st.selectbox("Chart 6-month history for", symbols)
    hist = _market().get_history(pick)
    if hist:
        hdf = pd.DataFrame(hist)
        line = px.line(
            hdf, x="date", y="close", title=f"{pick} — 6-month trend"
        )
        st.plotly_chart(line, use_container_width=True)


def render_goals() -> None:
    st.subheader("🎯 Goal planner")
    profile = st.session_state.profile
    goal_cfg = profile.get("goal") or {}
    risk_opts = ["conservative", "moderate", "aggressive"]

    c1, c2, c3 = st.columns(3)
    target = c1.number_input(
        "Target amount ($)",
        min_value=0.0,
        value=float(goal_cfg.get("target_amount") or 100000.0),
        step=1000.0,
    )
    current = c2.number_input(
        "Current savings ($)",
        min_value=0.0,
        value=float(goal_cfg.get("current_savings") or 5000.0),
        step=500.0,
    )
    monthly = c3.number_input(
        "Monthly contribution ($)",
        min_value=0.0,
        value=float(goal_cfg.get("monthly_contribution") or 500.0),
        step=50.0,
    )
    c4, c5 = st.columns(2)
    years = c4.number_input(
        "Horizon in years (0 = solve for time)",
        min_value=0.0,
        value=float(goal_cfg.get("years") or 0.0),
        step=1.0,
    )
    risk = c5.selectbox(
        "Risk profile",
        risk_opts,
        index=risk_opts.index(goal_cfg.get("risk_profile", "moderate")),
    )

    projection = plan_goal(
        GoalInput(
            target_amount=target,
            current_savings=current,
            monthly_contribution=monthly,
            years=(years or None),
            risk_profile=risk,
        )
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("On track", "Yes" if projection.on_track else "No")
    if projection.projected_value is not None:
        m2.metric(
            "Projected value", f"${projection.projected_value:,.0f}"
        )
    elif projection.years_to_goal is not None:
        m2.metric("Years to reach", f"{projection.years_to_goal:g}")
    if projection.required_monthly is not None:
        m3.metric(
            "Monthly needed", f"${projection.required_monthly:,.0f}"
        )

    if projection.yearly_path:
        pdf = pd.DataFrame(projection.yearly_path)
        fig = px.line(
            pdf, x="year", y="value", title="Projected savings path"
        )
        fig.add_hline(
            y=target, line_dash="dash", annotation_text="Target"
        )
        st.plotly_chart(fig, use_container_width=True)

    for note in projection.notes:
        st.markdown(f"- {note}")

    if st.button("💾 Save goal"):
        profile["goal"] = {
            "target_amount": target,
            "current_savings": current,
            "monthly_contribution": monthly,
            "years": (years or None),
            "risk_profile": risk,
        }
        _store().save(USER_ID, profile)
        st.success("Goal saved to your profile.")


def render_about(cfg: Any) -> None:
    st.subheader("ℹ️ About Financier")
    st.markdown(
        "Financier is a multi-agent finance educator. A router classifies "
        "each message and hands it to a specialist agent — Q&A "
        "(retrieval-augmented), Portfolio, Market, or Goal — built on "
        "LangGraph and Claude."
    )
    st.markdown(f"**Active LLM provider:** `{_assistant().provider_name}`")
    st.markdown(
        f"**Knowledge base:** "
        f"{getattr(_assistant().kb, 'num_chunks', 0)} chunks"
    )
    st.info(str(cfg.get("app.disclaimer", "")))


def main() -> None:
    st.set_page_config(
        page_title="Financier — AI Finance Assistant",
        page_icon="💸",
        layout="wide",
    )
    cfg = load_config()
    _init_state()

    st.sidebar.title("💸 Financier")
    st.sidebar.caption(str(cfg.get("app.tagline", "")))
    provider = _assistant().provider_name
    st.sidebar.info(f"LLM provider: **{provider}**")
    if provider == "mock":
        st.sidebar.warning(
            "Demo mode (no API key). Add ANTHROPIC_API_KEY to .env "
            "for real Claude answers."
        )

    name = st.sidebar.text_input(
        "Your name", value=st.session_state.profile.get("name", "")
    )
    if name != st.session_state.profile.get("name", ""):
        st.session_state.profile["name"] = name
        _store().save(USER_ID, st.session_state.profile)
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []

    tabs = st.tabs(["Chat", "Portfolio", "Market", "Goals", "About"])
    with tabs[0]:
        render_chat()
    with tabs[1]:
        render_portfolio()
    with tabs[2]:
        render_market()
    with tabs[3]:
        render_goals()
    with tabs[4]:
        render_about(cfg)


if __name__ == "__main__":
    main()
