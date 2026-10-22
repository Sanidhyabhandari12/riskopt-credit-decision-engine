"""
Step 7: turn PDs into lending decisions.

Decision rule: approve an applicant if PD < threshold.

How the threshold is chosen:
  1. Sweep thresholds on the CALIBRATION set and compute, for each one,
     approval rate, expected bad rate of the approved book and expected profit.
  2. Keep only thresholds inside the bank's risk appetite
     (bad rate <= max, approval rate >= min).
  3. Pick the one with the highest expected profit.
  4. Apply that threshold to the untouched TEST set and back-test it
     using the actual outcomes.

Strategies compared on the test set:
  * Approve everyone             - what happens with no risk model at all
  * Rule-of-thumb cut-off        - "approve if PD < 10%"
  * Break-even (profit only)     - approve every loan with positive expected profit
  * Risk-appetite optimised      - max profit subject to the constraints above
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import FIXED_PD_CUTOFF, OUTPUT_DIR, RISK_APPETITE, RISK_GRADES, TARGET_COL
from src.financial_model import Assumptions, break_even_pd, loan_economics, realized_profit

THRESHOLDS = np.round(np.arange(0.01, 0.501, 0.0025), 4)


def assign_grade(pd_):
    bounds = [0] + list(RISK_GRADES.values())
    return pd.cut(pd_, bins=bounds, labels=list(RISK_GRADES.keys()), include_lowest=True).astype(str)


def portfolio_metrics(approved, scored, econ, realized=None):
    """KPIs for the set of approved loans (approved is a boolean mask)."""
    n = len(scored)
    n_app = int(approved.sum())
    exposure = scored.loc[approved, "AMT_CREDIT"].sum()
    profit = econ.loc[approved, "expected_profit"].sum()
    capital = econ.loc[approved, "capital"].sum()
    out = {
        "approval_rate": n_app / n,
        "approved_loans": n_app,
        "exposure": exposure,
        "expected_bad_rate": scored.loc[approved, "PD"].mean() if n_app else 0.0,
        "expected_credit_loss": econ.loc[approved, "expected_credit_loss"].sum(),
        "expected_profit": profit,
        "raroc": profit / capital if capital else 0.0,
        "return_on_exposure": profit / exposure if exposure else 0.0,
    }
    if realized is not None:
        out["realized_bad_rate"] = scored.loc[approved, TARGET_COL].mean() if n_app else 0.0
        out["realized_profit"] = realized[approved].sum()
    return out


def economics(scored, a):
    """Expected P&L per loan, plus realised profit from the actual outcomes."""
    econ = loan_economics(scored["PD"], scored["AMT_CREDIT"], a).set_index(scored.index)
    realized = pd.Series(realized_profit(scored[TARGET_COL], scored["AMT_CREDIT"], a),
                         index=scored.index)
    return econ, realized


def threshold_sweep(scored, a, with_actuals=True):
    econ, realized = economics(scored, a)
    if not with_actuals:
        realized = None
    rows = []
    for t in THRESHOLDS:
        approved = scored["PD"] < t
        rows.append({"threshold": t, **portfolio_metrics(approved, scored, econ, realized)})
    return pd.DataFrame(rows)


def choose_threshold(sweep, max_bad_rate, min_approval_rate):
    feasible = sweep[(sweep["expected_bad_rate"] <= max_bad_rate)
                     & (sweep["approval_rate"] >= min_approval_rate)]
    if len(feasible):
        best = feasible.loc[feasible["expected_profit"].idxmax()]
        status = "optimal within risk appetite"
    else:
        # constraints conflict: risk limit wins over the growth target
        within_risk = sweep[sweep["expected_bad_rate"] <= max_bad_rate]
        best = within_risk.loc[within_risk["expected_profit"].idxmax()]
        status = "approval target not reachable inside bad-rate limit - risk limit applied"
    return float(best["threshold"]), status


def compare_strategies(scored, a, optimal_threshold):
    econ, realized = economics(scored, a)
    be = break_even_pd(a)
    strategies = {
        "Approve everyone": np.ones(len(scored), dtype=bool),
        f"Rule of thumb (PD < {FIXED_PD_CUTOFF:.0%})": (scored["PD"] < FIXED_PD_CUTOFF).to_numpy(),
        f"Break-even (PD < {be:.1%})": (scored["PD"] < be).to_numpy(),
        f"Risk-appetite optimised (PD < {optimal_threshold:.1%})":
            (scored["PD"] < optimal_threshold).to_numpy(),
    }
    rows = []
    for name, mask in strategies.items():
        approved = pd.Series(mask, index=scored.index)
        rows.append({"strategy": name, **portfolio_metrics(approved, scored, econ, realized)})
    return pd.DataFrame(rows)


def grade_summary(scored, a):
    econ = loan_economics(scored["PD"], scored["AMT_CREDIT"], a)
    df = scored.assign(expected_profit=econ["expected_profit"].to_numpy(),
                       GRADE=assign_grade(scored["PD"]))
    return (df.groupby("GRADE")
              .agg(applications=("PD", "size"), avg_pd=("PD", "mean"),
                   actual_bad_rate=(TARGET_COL, "mean"), exposure=("AMT_CREDIT", "sum"),
                   expected_profit=("expected_profit", "sum"))
              .reset_index())


def sensitivity_analysis(scored_calib, scored_test):
    """How do the optimal threshold and profit move if our assumptions are wrong?"""
    rows = []
    for lgd in [0.45, 0.60, 0.75]:
        for rate in [0.12, 0.14, 0.16]:
            a = Assumptions(interest_rate=rate, lgd=lgd)
            sweep = threshold_sweep(scored_calib, a, with_actuals=False)
            t, _ = choose_threshold(sweep, **RISK_APPETITE)
            econ, realized = economics(scored_test, a)
            res = portfolio_metrics(scored_test["PD"] < t, scored_test, econ, realized)
            rows.append({"lgd": lgd, "interest_rate": rate, "break_even_pd": break_even_pd(a),
                         "optimal_threshold": t, **res})
    return pd.DataFrame(rows)


def plot_profit_curve(sweep, threshold, be, out_dir):
    """Two panels sharing the x-axis (profit on top, approval rate below)
    instead of one chart with two different y-scales."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(sweep["threshold"], sweep["expected_profit"] / 1e6, color="#2a78d6", lw=2,
             label="Expected profit")
    ax1.plot(sweep["threshold"], sweep["realized_profit"] / 1e6, color="#eb6834", lw=2,
             label="Realised profit (back-test)")
    ax1.set_ylabel("Profit (millions)")
    ax1.set_title("Profit and approval rate vs PD threshold (calibration set)")
    ax2.plot(sweep["threshold"], sweep["approval_rate"] * 100, color="#2a78d6", lw=2)
    ax2.set_ylabel("Approval rate (%)")
    ax2.set_xlabel("PD approval threshold")
    for ax in (ax1, ax2):
        ax.axvline(threshold, color="#0b0b0b", ls=":", label=f"Chosen threshold {threshold:.1%}")
        ax.axvline(be, color="#8a8984", ls="--", label=f"Break-even PD {be:.1%}")
        ax.grid(alpha=0.3)
    ax1.axhline(0, color="#8a8984", lw=0.8)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "profit_curve.png", dpi=120)
    plt.close(fig)


def run(scored_calib, scored_test):
    out_dir = OUTPUT_DIR / "decision"
    out_dir.mkdir(parents=True, exist_ok=True)
    a = Assumptions()
    be = break_even_pd(a)

    print("\n[9] Decision optimisation")
    print(f"  assumptions: {a.to_dict()}")
    print(f"  break-even PD = ({a.interest_rate} - {a.cost_of_funds} - {a.operating_cost}) "
          f"/ ({a.interest_rate} + {a.lgd}) = {be:.2%}")

    sweep = threshold_sweep(scored_calib, a)
    threshold, status = choose_threshold(sweep, **RISK_APPETITE)
    sweep.to_csv(out_dir / "threshold_sweep_calib.csv", index=False)
    plot_profit_curve(sweep, threshold, be, out_dir)
    print(f"  risk appetite: {RISK_APPETITE}")
    print(f"  chosen threshold: PD < {threshold:.2%}  ({status})")

    print("\n[10] Back-test on the test set")
    strategies = compare_strategies(scored_test, a, threshold)
    strategies.to_csv(out_dir / "strategy_comparison.csv", index=False)
    show = strategies[["strategy", "approval_rate", "expected_bad_rate", "realized_bad_rate",
                       "expected_profit", "realized_profit", "raroc"]].copy()
    for c in ["approval_rate", "expected_bad_rate", "realized_bad_rate", "raroc"]:
        show[c] = (show[c] * 100).round(2).astype(str) + "%"
    for c in ["expected_profit", "realized_profit"]:
        show[c] = (show[c] / 1e6).round(1).astype(str) + "M"
    print(show.to_string(index=False))

    # confusion matrix at the chosen threshold: "positive" = predicted bad = declined
    declined = (scored_test["PD"] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(scored_test[TARGET_COL], declined).ravel()
    cm = {"approved_good": int(tn), "declined_good": int(fp),
          "approved_bad": int(fn), "declined_bad": int(tp),
          "bads_caught_pct": round(tp / (tp + fn), 4),
          "goods_lost_pct": round(fp / (fp + tn), 4)}
    print(f"\n  at PD < {threshold:.1%}: declined {cm['bads_caught_pct']:.1%} of bad loans "
          f"while turning away {cm['goods_lost_pct']:.1%} of good customers")

    grades = grade_summary(scored_test, a)
    grades.to_csv(out_dir / "risk_grades.csv", index=False)

    sens = sensitivity_analysis(scored_calib, scored_test)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    print(f"  sensitivity: optimal threshold ranges {sens['optimal_threshold'].min():.1%} - "
          f"{sens['optimal_threshold'].max():.1%} across LGD/rate scenarios")

    decision = {"threshold": threshold, "status": status, "break_even_pd": be,
                "assumptions": a.to_dict(), "risk_appetite": RISK_APPETITE, "confusion": cm}
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    scored_test = scored_test.assign(
        GRADE=assign_grade(scored_test["PD"]),
        DECISION=np.where(scored_test["PD"] < threshold, "APPROVE", "DECLINE"),
    )
    return decision, strategies, scored_test
