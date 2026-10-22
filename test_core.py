"""
Sanity tests for the financial model and decision logic.
Run with:  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.decision_engine import assign_grade, choose_threshold, threshold_sweep
from src.financial_model import Assumptions, break_even_pd, loan_economics


def test_break_even_pd_gives_zero_profit():
    a = Assumptions()
    pd_star = break_even_pd(a)
    profit = loan_economics([pd_star], [100_000], a)["expected_profit"].iloc[0]
    assert abs(profit) < 1e-6


def test_profit_falls_as_pd_rises():
    a = Assumptions()
    profits = loan_economics([0.01, 0.05, 0.10, 0.30], [100_000] * 4, a)["expected_profit"]
    assert profits.is_monotonic_decreasing


def test_ecl_is_pd_times_lgd_times_ead():
    a = Assumptions(lgd=0.5)
    ecl = loan_economics([0.2], [1_000], a)["expected_credit_loss"].iloc[0]
    assert np.isclose(ecl, 0.2 * 0.5 * 1_000)


def test_chosen_threshold_respects_risk_appetite():
    rng = np.random.default_rng(0)
    pd_ = rng.beta(1, 11, 5_000)                       # mean PD ~8%
    scored = pd.DataFrame({"PD": pd_, "AMT_CREDIT": rng.uniform(5e4, 1e6, 5_000),
                           "TARGET": (rng.random(5_000) < pd_).astype(int)})
    sweep = threshold_sweep(scored, Assumptions())
    t, status = choose_threshold(sweep, max_bad_rate=0.05, min_approval_rate=0.5)
    row = sweep[sweep["threshold"] == t].iloc[0]
    assert row["expected_bad_rate"] <= 0.05
    if status.startswith("optimal"):
        assert row["approval_rate"] >= 0.5


def test_risk_grades_are_ordered():
    grades = assign_grade(pd.Series([0.01, 0.04, 0.08, 0.15, 0.5]))
    assert list(grades) == ["A", "B", "C", "D", "E"]
