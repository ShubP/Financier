"""Deterministic financial calculations (no LLM).

Portfolio analytics and goal projections live here so the numbers are
exact and unit-testable; the agents wrap them in plain-English advice.
"""

from .portfolio import Holding, PortfolioAnalysis, analyze_portfolio
from .projections import GoalInput, GoalProjection, plan_goal

__all__ = [
    "Holding",
    "PortfolioAnalysis",
    "analyze_portfolio",
    "GoalInput",
    "GoalProjection",
    "plan_goal",
]
