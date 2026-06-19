"""Goal-projection math tests."""

from src.finance.projections import (
    GoalInput,
    future_value,
    plan_goal,
    required_monthly,
)


def test_future_value_growth() -> None:
    fv = future_value(0, 100, 12, 1)  # 12%/yr monthly, 1 year
    assert 1200 < fv < 1300


def test_required_monthly_zero_rate() -> None:
    need = required_monthly(12000, 0, 0, 1)
    assert abs(need - 1000) < 1e-6


def test_plan_goal_on_track_with_horizon() -> None:
    goal = GoalInput(
        target_amount=10000,
        current_savings=0,
        monthly_contribution=1000,
        years=1,
        annual_return_pct=0,
    )
    proj = plan_goal(goal)
    assert proj.projected_value == 12000
    assert proj.on_track is True
    assert proj.yearly_path


def test_plan_goal_solves_time() -> None:
    goal = GoalInput(
        target_amount=12000,
        current_savings=0,
        monthly_contribution=1000,
        annual_return_pct=0,
    )
    proj = plan_goal(goal)
    assert proj.years_to_goal == 1.0
