"""
Step 8: write a short markdown summary of the run (outputs/RESULTS.md),
so the numbers I quote always come from the latest pipeline run.
"""
import pandas as pd

from config import OUTPUT_DIR


def pct(x):
    return f"{x * 100:.2f}%"


def money(x):
    return f"{x / 1e6:,.1f}M"


def run(metrics, decision, strategies, is_sample):
    comparison = pd.read_csv(OUTPUT_DIR / "model" / "model_comparison.csv")
    importance_path = OUTPUT_DIR / "model" / "feature_importance.csv"
    top_features = (pd.read_csv(importance_path)["feature"].head(8).tolist()
                    if importance_path.exists() else [])
    m = metrics["test_calibrated"]

    lines = ["# RiskOpt - results of the latest run", ""]
    if is_sample:
        lines += ["> **Run on SYNTHETIC sample data.** These numbers only prove the pipeline "
                  "works. Re-run with the real Kaggle files before quoting anything.", ""]

    lines += [
        "## Model performance (test set)", "",
        "| Model | Calib AUC | Test AUC | Gini | KS | PR-AUC | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in comparison.iterrows():
        lines.append(f"| {r['model']} | {r['calib_auc']:.4f} | {r['test_roc_auc']:.4f} | "
                     f"{r['test_gini']:.4f} | {r['test_ks']:.4f} | {r['test_pr_auc']:.4f} | "
                     f"{r['test_brier']:.5f} |")
    lines += [
        "",
        f"Champion: **{metrics['champion']}** (picked on calibration-set AUC). After Platt "
        f"calibration the average predicted PD is {pct(m['avg_predicted_pd'])} vs an actual "
        f"bad rate of {pct(m['actual_bad_rate'])}.",
        "",
        "Top risk drivers: " + ", ".join(top_features),
        "",
        "## Decision engine", "",
        f"* Break-even PD: **{pct(decision['break_even_pd'])}**",
        f"* Chosen threshold: **approve if PD < {pct(decision['threshold'])}** ({decision['status']})",
        f"* Risk appetite: bad rate <= {pct(decision['risk_appetite']['max_bad_rate'])}, "
        f"approval rate >= {pct(decision['risk_appetite']['min_approval_rate'])}",
        f"* At this threshold the model declines {pct(decision['confusion']['bads_caught_pct'])} "
        f"of bad loans while turning away {pct(decision['confusion']['goods_lost_pct'])} of good customers",
        "",
        "## Strategy back-test (test set, actual outcomes)", "",
        "| Strategy | Approval | Exp. bad rate | Actual bad rate | Exp. profit | Realised profit | RAROC |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in strategies.iterrows():
        lines.append(f"| {r['strategy']} | {pct(r['approval_rate'])} | {pct(r['expected_bad_rate'])} | "
                     f"{pct(r['realized_bad_rate'])} | {money(r['expected_profit'])} | "
                     f"{money(r['realized_profit'])} | {pct(r['raroc'])} |")

    a = decision["assumptions"]
    lines += [
        "", "## Assumptions used", "",
        f"interest rate {pct(a['interest_rate'])}, cost of funds {pct(a['cost_of_funds'])}, "
        f"operating cost {pct(a['operating_cost'])}, LGD {pct(a['lgd'])}, "
        f"capital {pct(a['capital_ratio'])} x risk weight {pct(a['risk_weight'])}. "
        "EAD = AMT_CREDIT. One-year horizon. TARGET is a payment-difficulty proxy for default.",
        "",
    ]
    (OUTPUT_DIR / "RESULTS.md").write_text("\n".join(lines))
    print("\n[11] Summary written to outputs/RESULTS.md")
