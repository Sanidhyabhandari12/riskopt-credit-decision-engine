"""
RiskOpt - interactive credit decision dashboard.

    streamlit run app/dashboard.py

Reads the files written by run_pipeline.py and recomputes the profit model
live, so the sliders on the left answer "what if" questions:
what if funding costs rise? what if losses on default are higher than we think?
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import FINANCE, OUTPUT_DIR, RISK_APPETITE, RISK_GRADES   # noqa: E402
from src.decision_engine import (assign_grade, choose_threshold,     # noqa: E402
                                 compare_strategies, economics, portfolio_metrics,
                                 threshold_sweep)
from src.financial_model import Assumptions, break_even_pd           # noqa: E402

# one colour per job: blue = main series, orange = comparison series,
# light -> dark blue = increasing risk grade
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
GRADE_COLORS = dict(zip(RISK_GRADES.keys(), ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]))

# plain-English names for the reason codes shown to credit officers
FEATURE_LABELS = {
    "EXT_SOURCE_MEAN": "Weak external credit scores (average)",
    "EXT_SOURCE_MIN": "Weak external credit score (lowest)",
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "AMT_CREDIT": "Loan amount",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "AMT_ANNUITY": "Instalment amount",
    "CREDIT_INCOME_RATIO": "Loan is large relative to income",
    "ANNUITY_INCOME_RATIO": "High EMI-to-income burden",
    "ANNUITY_CREDIT_RATIO": "Loan term / repayment speed",
    "CREDIT_GOODS_RATIO": "Loan exceeds goods price (LTV)",
    "AGE_YEARS": "Applicant age",
    "DAYS_BIRTH": "Applicant age",
    "EMPLOYED_YEARS": "Short employment history",
    "DAYS_EMPLOYED": "Short employment history",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_INCOME_TYPE": "Income type",
    "ORGANIZATION_TYPE": "Employer type",
    "OCCUPATION_TYPE": "Occupation",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults in social circle",
    "BUR_HAS_OVERDUE": "Currently overdue at another lender",
    "BUR_MAX_DAYS_OVERDUE": "Days overdue at other lenders",
    "BUR_DEBT_TO_CREDIT": "High utilisation of existing credit",
    "BUR_LOANS_LAST_YEAR": "Many new loans in the last year",
    "PREV_REFUSED_COUNT": "Previously refused applications",
    "PREV_REFUSAL_RATE": "High past refusal rate",
    "REGION_RATING_CLIENT_W_CITY": "Region risk rating",
    "DAYS_LAST_PHONE_CHANGE": "Recent phone number change",
    "DAYS_ID_PUBLISH": "Recently changed ID document",
}

st.set_page_config(page_title="RiskOpt - Credit Decision Engine", layout="wide")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_outputs():
    if not (OUTPUT_DIR / "scored_test.parquet").exists():
        return None
    model_dir = OUTPUT_DIR / "model"
    return {
        "calib": pd.read_parquet(OUTPUT_DIR / "scored_calib.parquet"),
        "test": pd.read_parquet(OUTPUT_DIR / "scored_test.parquet"),
        "metrics": json.loads((model_dir / "metrics.json").read_text()),
        "comparison": pd.read_csv(model_dir / "model_comparison.csv"),
        "deciles": pd.read_csv(model_dir / "decile_table.csv"),
        "importance": pd.read_csv(model_dir / "feature_importance.csv"),
        "source": (OUTPUT_DIR / "data_source.txt").read_text().strip(),
        "eda": {p.stem: pd.read_csv(p) for p in sorted((OUTPUT_DIR / "eda").glob("*.csv"))},
    }


@st.cache_data
def run_scenario(assumption_values, max_bad_rate, min_approval_rate):
    """Everything that depends on the sidebar inputs, cached per combination."""
    data = load_outputs()
    a = Assumptions(**assumption_values)
    sweep = threshold_sweep(data["calib"], a)            # tune on calibration set
    threshold, status = choose_threshold(sweep, max_bad_rate, min_approval_rate)
    strategies = compare_strategies(data["test"], a, threshold)   # back-test on test set
    return sweep, threshold, status, strategies


data = load_outputs()
if data is None:
    st.error("No pipeline outputs found. Run `python run_pipeline.py` (or `--sample`) first.")
    st.stop()

calib, test = data["calib"], data["test"]

# ---------------------------------------------------------------------------
# Sidebar - scenario inputs
# ---------------------------------------------------------------------------
st.sidebar.title("Scenario inputs")
st.sidebar.caption("Financial assumptions (annual)")
rate = st.sidebar.slider("Interest rate on loan", 0.08, 0.24, FINANCE["interest_rate"], 0.005, format="%.3f")
cof = st.sidebar.slider("Cost of funds", 0.03, 0.10, FINANCE["cost_of_funds"], 0.0025, format="%.4f")
opex = st.sidebar.slider("Operating cost", 0.0, 0.05, FINANCE["operating_cost"], 0.0025, format="%.4f")
lgd = st.sidebar.slider("Loss given default (LGD)", 0.20, 1.00, FINANCE["lgd"], 0.05)

st.sidebar.caption("Risk appetite")
max_bad = st.sidebar.slider("Max expected bad rate of approved book", 0.02, 0.10,
                            RISK_APPETITE["max_bad_rate"], 0.0025, format="%.4f")
min_appr = st.sidebar.slider("Min approval rate", 0.0, 0.95, RISK_APPETITE["min_approval_rate"], 0.05)

values = dict(FINANCE, interest_rate=rate, cost_of_funds=cof, operating_cost=opex, lgd=lgd)
a = Assumptions(**values)
sweep, threshold, status, strategies = run_scenario(values, max_bad, min_appr)
be = break_even_pd(a)

test = test.assign(GRADE=assign_grade(test["PD"]),
                   DECISION=np.where(test["PD"] < threshold, "APPROVE", "DECLINE"))
econ, realized = economics(test, a)
approved = test["DECISION"] == "APPROVE"
book = portfolio_metrics(approved, test, econ, realized)
base = portfolio_metrics(pd.Series(True, index=test.index), test, econ, realized)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("RiskOpt: Credit Risk & Loan Profitability Optimisation")
st.caption("Predict risk -> estimate profit -> optimise the approval decision. "
           "Scenario analysis on the Home Credit dataset; not actual bank financials.")
if data["source"] == "sample":
    st.warning("Running on SYNTHETIC sample data - numbers only demonstrate the pipeline.")

tab_port, tab_model, tab_profit, tab_app, tab_eda = st.tabs(
    ["Portfolio", "Credit model", "Profit optimiser", "Applicant decision", "Risk insights (SQL)"])


def fmt_m(x):
    return f"{x / 1e6:,.1f}M"


# ---------------------------------------------------------------------------
# Tab 1 - Portfolio
# ---------------------------------------------------------------------------
with tab_port:
    st.subheader(f"Recommended policy: approve if PD < {threshold:.2%}")
    st.caption(f"{status.capitalize()}. Break-even PD under current assumptions: {be:.2%}. "
               f"Numbers below are on the held-out test set ({len(test):,} applications).")

    c = st.columns(5)
    c[0].metric("Approval rate", f"{book['approval_rate']:.1%}")
    c[1].metric("Expected bad rate", f"{book['expected_bad_rate']:.2%}",
                f"{(book['expected_bad_rate'] - base['expected_bad_rate']) * 100:+.2f} pp vs approve-all",
                delta_color="inverse")
    c[2].metric("Expected profit", fmt_m(book["expected_profit"]),
                f"{fmt_m(book['expected_profit'] - base['expected_profit'])} vs approve-all")
    c[3].metric("Realised profit (back-test)", fmt_m(book["realized_profit"]))
    c[4].metric("RAROC", f"{book['raroc']:.1%}")

    left, right = st.columns(2)
    grades = (test.assign(expected_profit=econ["expected_profit"])
                  .groupby("GRADE")
                  .agg(applications=("PD", "size"), avg_pd=("PD", "mean"),
                       actual_bad_rate=("TARGET", "mean"), exposure=("AMT_CREDIT", "sum"),
                       expected_profit=("expected_profit", "sum"))
                  .reindex(list(RISK_GRADES.keys())).dropna(how="all").reset_index())
    with left:
        fig = px.bar(grades, x="GRADE", y="actual_bad_rate", color="GRADE",
                     color_discrete_map=GRADE_COLORS, title="Actual bad rate by risk grade",
                     labels={"actual_bad_rate": "Bad rate", "GRADE": "Risk grade"})
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(grades, x="GRADE", y="expected_profit", color="GRADE",
                     color_discrete_map=GRADE_COLORS, title="Expected profit by risk grade",
                     labels={"expected_profit": "Expected profit", "GRADE": "Risk grade"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    grade_bounds = ", ".join(f"{g} < {b:.0%}" for g, b in RISK_GRADES.items() if b < 1)
    st.caption(f"Grades by PD: {grade_bounds}, E = rest.")

    st.markdown("**Strategy comparison (test set)**")
    show = strategies[["strategy", "approval_rate", "expected_bad_rate", "realized_bad_rate",
                       "expected_profit", "realized_profit", "raroc"]]
    st.dataframe(show.style.format({
        "approval_rate": "{:.1%}", "expected_bad_rate": "{:.2%}", "realized_bad_rate": "{:.2%}",
        "expected_profit": fmt_m, "realized_profit": fmt_m, "raroc": "{:.1%}"}),
        hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 - Credit model
# ---------------------------------------------------------------------------
with tab_model:
    m = data["metrics"]
    st.subheader(f"Champion model: {m['champion'].replace('_', ' ')}")
    c = st.columns(4)
    mc = m["test_calibrated"]
    c[0].metric("Test ROC-AUC", f"{mc['roc_auc']:.3f}")
    c[1].metric("Gini", f"{mc['gini']:.3f}")
    c[2].metric("KS", f"{mc['ks']:.3f}")
    c[3].metric("Avg PD vs actual", f"{mc['avg_predicted_pd']:.2%} / {mc['actual_bad_rate']:.2%}")

    st.dataframe(data["comparison"], hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fpr, tpr, _ = roc_curve(test["TARGET"], test["PD"])
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=BLUE, width=2),
                        name=f"Champion (AUC {roc_auc_score(test['TARGET'], test['PD']):.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GREY, dash="dash"),
                        name="Random")
        fig.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        bins = test.assign(bin=pd.qcut(test["PD"].rank(method="first"), 10, labels=False))
        cal = bins.groupby("bin").agg(predicted=("PD", "mean"), observed=("TARGET", "mean"))
        top = float(cal.max().max()) * 1.05
        fig = go.Figure()
        fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers",
                        line=dict(color=BLUE, width=2), marker=dict(size=9), name="Champion")
        fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=GREY, dash="dash"),
                        name="Perfect calibration")
        fig.update_layout(title="Calibration: predicted PD vs observed bad rate",
                          xaxis_title="Predicted PD", yaxis_title="Observed bad rate",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        imp = data["importance"].head(15).iloc[::-1]
        fig = px.bar(imp, x="mean_abs_contribution", y="feature", orientation="h",
                     title="Top 15 risk drivers", color_discrete_sequence=[BLUE],
                     labels={"mean_abs_contribution": "Mean |contribution to log-odds|", "feature": ""})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Decile table** (riskiest 10% first)")
        st.dataframe(data["deciles"][["decile", "applications", "bads", "avg_pd", "bad_rate",
                                      "cum_bads_captured", "lift"]]
                     .style.format({"avg_pd": "{:.2%}", "bad_rate": "{:.2%}",
                                    "cum_bads_captured": "{:.1%}", "lift": "{:.2f}x"}),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3 - Profit optimiser
# ---------------------------------------------------------------------------
with tab_profit:
    st.subheader("How the threshold is chosen")
    st.latex(r"\text{Expected profit} = (1-PD)\,r\,EAD - c_{funds}EAD - c_{ops}EAD - PD \times LGD \times EAD")
    st.latex(rf"PD^* = \frac{{r - c_{{funds}} - c_{{ops}}}}{{r + LGD}} = "
             rf"\frac{{{rate:.3f} - {cof:.4f} - {opex:.4f}}}{{{rate:.3f} + {lgd:.2f}}} = {be * 100:.2f}\%")

    view = sweep[sweep["threshold"] <= 0.35]
    fig = go.Figure()
    fig.add_scatter(x=view["threshold"], y=view["expected_profit"], mode="lines",
                    line=dict(color=BLUE, width=2), name="Expected profit")
    fig.add_scatter(x=view["threshold"], y=view["realized_profit"], mode="lines",
                    line=dict(color=ORANGE, width=2), name="Realised profit (actual outcomes)")
    fig.add_vline(x=threshold, line_dash="dot", annotation_text=f"chosen {threshold:.1%}")
    fig.add_vline(x=be, line_dash="dash", line_color=GREY, annotation_text=f"break-even {be:.1%}",
                  annotation_position="bottom right")
    fig.update_layout(title="Portfolio profit vs PD threshold (calibration set)",
                      xaxis_title="Approve if PD below", yaxis_title="Profit",
                      xaxis_tickformat=".0%", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=view["threshold"], y=view["approval_rate"], mode="lines",
                        line=dict(color=BLUE, width=2), name="Approval rate")
        fig.add_hline(y=min_appr, line_dash="dash", line_color=GREY,
                      annotation_text=f"min approval {min_appr:.0%}")
        fig.add_vline(x=threshold, line_dash="dot")
        fig.update_layout(title="Approval rate", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="Approve if PD below", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_scatter(x=view["threshold"], y=view["expected_bad_rate"], mode="lines",
                        line=dict(color=BLUE, width=2), name="Expected bad rate")
        fig.add_hline(y=max_bad, line_dash="dash", line_color=GREY,
                      annotation_text=f"max bad rate {max_bad:.1%}")
        fig.add_vline(x=threshold, line_dash="dot")
        fig.update_layout(title="Expected bad rate of approved book", xaxis_tickformat=".0%",
                          yaxis_tickformat=".1%", xaxis_title="Approve if PD below",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Where the profit of the approved book comes from (test set)**")
    parts = econ[approved].sum()
    fig = go.Figure(go.Waterfall(
        x=["Interest income", "Funding cost", "Operating cost", "Expected credit loss", "Expected profit"],
        y=[parts["interest_income"], -parts["funding_cost"], -parts["operating_cost"],
           -parts["expected_credit_loss"], 0],
        measure=["relative", "relative", "relative", "relative", "total"],
        increasing=dict(marker_color=BLUE), decreasing=dict(marker_color=ORANGE),
        totals=dict(marker_color="#1c5cab")))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 - Single applicant
# ---------------------------------------------------------------------------
with tab_app:
    st.subheader("Score an applicant from the test set")
    left, right = st.columns([1, 2])
    show_only = left.radio("Show", ["All", "Approved", "Declined"], horizontal=True)
    pool = test if show_only == "All" else test[test["DECISION"] == show_only.upper()]
    ids = pool["SK_ID_CURR"].head(1000).tolist()     # first 1,000 keeps the dropdown fast
    chosen = right.selectbox("Applicant ID (SK_ID_CURR)", ids, index=0)
    row = test[test["SK_ID_CURR"] == chosen].iloc[0]
    loan = econ.loc[row.name]

    c = st.columns(4)
    c[0].metric("Probability of default", f"{row['PD']:.2%}")
    c[1].metric("Risk grade", row["GRADE"])
    c[2].metric("Decision", row["DECISION"])
    c[3].metric("Expected profit on this loan", f"{loan['expected_profit']:,.0f}")

    # risk-based pricing: the rate at which this loan would just break even
    min_rate = (cof + opex + row["PD"] * lgd) / (1 - row["PD"])
    if row["DECISION"] == "DECLINE":
        st.info(f"Declined at the current {rate:.1%} rate. This loan would break even at an "
                f"interest rate of **{min_rate:.1%}** - an option for risk-based pricing "
                f"instead of a flat decline (subject to affordability and pricing policy).")
    else:
        st.success(f"Approved. Break-even rate for this applicant is {min_rate:.1%}, "
                   f"leaving a margin of {rate - min_rate:.1%} over it.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Main factors increasing this applicant's risk**")
        reasons = [r for r in str(row.get("RISK_REASONS", "")).split("; ") if r]
        if reasons:
            for r in reasons:
                st.write(f"- {FEATURE_LABELS.get(r, r)}  (`{r}`)")
        else:
            st.write("No factor pushes this applicant's risk above average.")
    with right:
        st.markdown("**Applicant profile**")
        profile = {
            "Loan amount (EAD)": f"{row['AMT_CREDIT']:,.0f}",
            "Annual income": f"{row['AMT_INCOME_TOTAL']:,.0f}",
            "Loan / income": f"{row['CREDIT_INCOME_RATIO']:.2f}x",
            "EMI / income": f"{row['ANNUITY_INCOME_RATIO']:.1%}",
            "External score (avg)": f"{row['EXT_SOURCE_MEAN']:.3f}",
            "Age": f"{row['AGE_YEARS']:.0f}",
            "Income type": row["NAME_INCOME_TYPE"],
            "Overdue at other lender": "Yes" if row["BUR_HAS_OVERDUE"] == 1 else "No",
            "Previous refusals": int(row["PREV_REFUSED_COUNT"]),
            "Actual outcome (back-test)": "Payment difficulty" if row["TARGET"] == 1 else "Repaid",
        }
        st.table(pd.Series(profile, name="value").astype(str))

    st.markdown("**Expected P&L of this loan**")
    st.dataframe(loan[["interest_income", "funding_cost", "operating_cost", "expected_credit_loss",
                       "expected_profit", "capital"]].rename("amount").to_frame().style.format("{:,.0f}"),
                 use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5 - SQL insights
# ---------------------------------------------------------------------------
with tab_eda:
    st.subheader("Portfolio risk insights (from sql/04_portfolio_analysis.sql)")
    names = [n for n in data["eda"] if n != "portfolio_summary"]
    pick = st.selectbox("Analysis", names, format_func=lambda n: n.replace("_", " ").capitalize())
    df = data["eda"][pick]
    label = df.columns[0]
    fig = px.bar(df, x=df[label].astype(str), y="bad_rate_pct", color_discrete_sequence=[BLUE],
                 hover_data=["applications"],
                 labels={"x": label.replace("_", " "), "bad_rate_pct": "Bad rate (%)"})
    portfolio_rate = data["eda"]["portfolio_summary"]["bad_rate_pct"].iloc[0]
    fig.add_hline(y=portfolio_rate, line_dash="dash", line_color=GREY,
                  annotation_text=f"portfolio {portfolio_rate}%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True)
