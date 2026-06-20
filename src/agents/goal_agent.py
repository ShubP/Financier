"""Goal planning agent: savings/retirement projections."""

from __future__ import annotations

import re
from typing import Any

from ..finance.projections import GoalInput, GoalProjection, plan_goal
from ..llm.base import LLMProvider
from .base_agent import history_messages

_UNIT = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "million": 1_000_000,
}
_MONEY = r"\$?\s?(\d+(?:\.\d+)?)\s?(k|m|thousand|million)?"


def _amount(num: str, unit: str | None) -> float:
    value = float(num)
    if unit:
        value *= _UNIT.get(unit.lower(), 1)
    return value


def parse_goal(query: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Merge any stored goal with numbers parsed from the query."""
    base: dict[str, Any] = dict(profile.get("goal") or {})
    text = query.lower().replace(",", "")

    m = re.search(_MONEY + r"\s*(?:/|per|a)\s*month", text)
    if m:
        base["monthly_contribution"] = _amount(m.group(1), m.group(2))

    y = re.search(r"in\s+(\d+)\s+years", text) or re.search(
        r"(\d+)\s*[- ]?year", text
    )
    if y:
        base["years"] = float(y.group(1))

    c = re.search(r"(?:have|saved|currently)\D{0,12}" + _MONEY, text)
    if c:
        base["current_savings"] = _amount(c.group(1), c.group(2))

    t = re.search(
        r"(?:reach|target|goal of|save up|need|retire with)\D{0,12}"
        + _MONEY,
        text,
    )
    if t:
        base["target_amount"] = _amount(t.group(1), t.group(2))

    # Fallback: the largest standalone amount is likely the target.
    if not base.get("target_amount"):
        amounts = [
            _amount(m.group(1), m.group(2))
            for m in re.finditer(_MONEY, text)
        ]
        used = {
            base.get("monthly_contribution"),
            base.get("current_savings"),
        }
        candidates = [a for a in amounts if a and a not in used]
        if candidates:
            base["target_amount"] = max(candidates)
    return base


_SYSTEM = (
    "You are Financier. Using the pre-computed projection, tell the user "
    "in plain language whether their savings goal is on track, the "
    "key numbers (target, projected value or time to reach, and the "
    "monthly amount needed), and one practical suggestion. State that "
    "the return is an assumption, not a guarantee. Educational only."
)


def _summary(p: GoalProjection) -> str:
    lines = [
        f"Target: ${p.target_amount:,.0f}",
        f"Current savings: ${p.current_savings:,.0f}",
        f"Monthly contribution: ${p.monthly_contribution:,.0f}",
        f"Assumed return: {p.annual_return_pct:g}%/yr",
        f"On track: {'yes' if p.on_track else 'no'}",
    ]
    if p.projected_value is not None:
        lines.append(f"Projected value: ${p.projected_value:,.0f}")
    if p.required_monthly is not None:
        lines.append(
            f"Monthly needed to hit target: ${p.required_monthly:,.0f}"
        )
    if p.years_to_goal is not None:
        lines.append(f"Years to reach: {p.years_to_goal:g}")
    if p.notes:
        lines.append("Notes: " + " ".join(p.notes))
    return "\n".join(lines)


class GoalAgent:
    """Projects savings goals using deterministic math."""

    name = "goal"

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        trace = state.get("agent_trace", []) + ["goal"]
        query = state.get("query", "")
        params = parse_goal(query, state.get("profile", {}))
        if not params.get("target_amount"):
            return {
                "response": (
                    "What's your savings goal? Give me a target "
                    "amount, and optionally your current savings, "
                    "monthly contribution, time horizon, and risk "
                    "level — or set them in the Goals tab."
                ),
                "agent_trace": trace,
                "sources": [],
            }
        goal = GoalInput.model_validate(params)
        projection = plan_goal(goal)
        system = f"{_SYSTEM}\n\nProjection:\n{_summary(projection)}"
        answer = self.llm.reason(system, history_messages(state, query))
        return {
            "response": answer,
            "analysis": projection.model_dump(),
            "sources": ["Goal projection (computed locally)"],
            "agent_trace": trace,
        }
