"""Goal planning: compound-growth projections.

Monthly-compounded future value, required contributions, and time-to-
goal — plus a year-by-year path for charting. The Goal Planning agent
narrates the result with appropriate caveats.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

# Default expected annual return by risk appetite (educational estimates).
_RISK_RETURNS: dict[str, float] = {
    "conservative": 4.0,
    "moderate": 7.0,
    "aggressive": 9.0,
}

_MAX_YEARS = 80


class GoalInput(BaseModel):
    target_amount: float
    current_savings: float = 0.0
    monthly_contribution: float = 0.0
    annual_return_pct: float | None = None  # else derived from risk
    years: float | None = None  # horizon, if the user has one
    risk_profile: str = "moderate"


class GoalProjection(BaseModel):
    target_amount: float
    current_savings: float
    monthly_contribution: float
    annual_return_pct: float
    years_to_goal: float | None = None
    projected_value: float | None = None  # at the given horizon
    required_monthly: float | None = None  # to hit target by horizon
    on_track: bool = False
    surplus_or_shortfall: float | None = None
    yearly_path: list[dict[str, float]] = []
    notes: list[str] = []


def future_value(
    present: float, monthly: float, annual_pct: float, years: float
) -> float:
    """Future value with monthly contributions, compounded monthly."""
    rate = annual_pct / 100.0 / 12.0
    months = int(round(years * 12))
    value = present
    for _ in range(months):
        value = value * (1 + rate) + monthly
    return value


def required_monthly(
    target: float, present: float, annual_pct: float, years: float
) -> float:
    """Monthly contribution needed to reach ``target`` by ``years``."""
    rate = annual_pct / 100.0 / 12.0
    months = int(round(years * 12))
    if months <= 0:
        return max(0.0, target - present)
    if rate == 0:
        return max(0.0, (target - present) / months)
    growth = (1 + rate) ** months
    annuity = (growth - 1) / rate
    needed = (target - present * growth) / annuity
    return max(0.0, needed)


def years_to_reach(
    target: float, present: float, monthly: float, annual_pct: float
) -> float | None:
    """Years until savings reach ``target``; None if unreachable."""
    if present >= target:
        return 0.0
    rate = annual_pct / 100.0 / 12.0
    value = present
    for month in range(1, _MAX_YEARS * 12 + 1):
        value = value * (1 + rate) + monthly
        if value >= target:
            return round(month / 12.0, 1)
    return None


def _yearly_path(
    present: float, monthly: float, annual_pct: float, years: float
) -> list[dict[str, float]]:
    rate = annual_pct / 100.0 / 12.0
    path = [{"year": 0.0, "value": round(present, 2)}]
    value = present
    for year in range(1, int(math.ceil(max(years, 0))) + 1):
        for _ in range(12):
            value = value * (1 + rate) + monthly
        path.append({"year": float(year), "value": round(value, 2)})
    return path


def plan_goal(goal: GoalInput) -> GoalProjection:
    """Project a savings goal and report whether it is on track."""
    rate = goal.annual_return_pct
    if rate is None:
        rate = _RISK_RETURNS.get(goal.risk_profile.lower(), 7.0)

    proj = GoalProjection(
        target_amount=goal.target_amount,
        current_savings=goal.current_savings,
        monthly_contribution=goal.monthly_contribution,
        annual_return_pct=rate,
    )
    notes: list[str] = []

    if goal.years and goal.years > 0:
        projected = future_value(
            goal.current_savings,
            goal.monthly_contribution,
            rate,
            goal.years,
        )
        proj.projected_value = round(projected, 2)
        proj.surplus_or_shortfall = round(
            projected - goal.target_amount, 2
        )
        proj.on_track = projected >= goal.target_amount
        proj.required_monthly = round(
            required_monthly(
                goal.target_amount,
                goal.current_savings,
                rate,
                goal.years,
            ),
            2,
        )
        proj.yearly_path = _yearly_path(
            goal.current_savings,
            goal.monthly_contribution,
            rate,
            goal.years,
        )
        if proj.on_track:
            notes.append(
                "On track: projected savings meet the target at the "
                "assumed return."
            )
        else:
            notes.append(
                "Off track for this horizon. To close the gap, raise "
                f"the monthly contribution to about "
                f"${proj.required_monthly:,.0f}."
            )
    else:
        years = years_to_reach(
            goal.target_amount,
            goal.current_savings,
            goal.monthly_contribution,
            rate,
        )
        proj.years_to_goal = years
        proj.on_track = years is not None
        horizon = years if years else 30.0
        proj.yearly_path = _yearly_path(
            goal.current_savings,
            goal.monthly_contribution,
            rate,
            horizon,
        )
        if years is None:
            notes.append(
                "At the current contribution and return, the target "
                "is not reached within a lifetime — increase savings."
            )
        else:
            notes.append(
                f"Reaching the target takes about {years:g} years at "
                f"an assumed {rate:g}% annual return."
            )

    notes.append(
        f"Assumes a constant {rate:g}% annual return, compounded "
        "monthly. Real markets vary; treat this as an estimate."
    )
    proj.notes = notes
    return proj
