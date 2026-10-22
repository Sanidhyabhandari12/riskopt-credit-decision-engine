# RiskOpt: AI-Powered Credit Risk & Loan Profitability Optimisation

**Business question:** how can a bank make better lending decisions by balancing the
expected profit of a loan against its credit default risk?

Most credit-risk projects stop at "predict who defaults". RiskOpt goes three steps further:

1. **Predict risk.** It estimates a calibrated Probability of Default (PD) for every applicant.
2. **Estimate profit.** It turns each PD into rupee-style P&L: interest income, funding cost,
   operating cost and expected credit loss (ECL = PD x LGD x EAD).
3. **Optimise decisions.** It picks the approval cut-off that maximises portfolio profit while
   staying inside the bank's risk appetite, then back-tests it on held-out data.

Everything ends in a Streamlit dashboard where a credit or BI team can change the assumptions
(funding cost, LGD, risk limits) and see the recommended policy update live.

> Built on the Home Credit Default Risk dataset. The financial figures come from scenario
> analysis under stated assumptions. They are not the actual performance of any bank.

---

## Architecture

```
 application_train.csv ─┐
 bureau.csv ────────────┼─► DuckDB SQL ─► model_table (1 row / applicant)
 previous_application ──┘   (sql/01-03)        │
                                               ├─► SQL portfolio analytics (sql/04) ─► EDA charts
                                               ▼
                         Logistic Regression | Random Forest | XGBoost
                                               │  champion picked on calibration set
                                               ▼
                         Platt calibration ─► PD per applicant + reason codes (SHAP)
                                               ▼
                         Financial model: interest - funding - opex - PD x LGD x EAD
                                               ▼
                         Decision engine: threshold sweep under risk-appetite constraints
                                               ▼
                         Back-test on test set ─► Streamlit dashboard
```

## Project structure

```
riskopt/
├── config.py                  # ALL assumptions: rates, LGD, risk appetite, grades
├── run_pipeline.py            # runs everything end to end
├── sql/
│   ├── 01_bureau_features.sql             # credit history at other lenders
│   ├── 02_previous_application_features.sql  # past applications with Home Credit
│   ├── 03_build_model_table.sql           # joins + business ratios
│   └── 04_portfolio_analysis.sql          # BI queries (deciles, bands, window functions)
├── src/
│   ├── data_pipeline.py       # load CSVs, data-quality audit, run SQL, validate joins
│   ├── eda.py                 # run the BI queries, save tables + charts
│   ├── modeling.py            # split, train 3 models, metrics, calibration, SHAP
│   ├── financial_model.py     # P&L per loan, ECL, RAROC, break-even PD
│   ├── decision_engine.py     # threshold optimisation, strategy back-test, sensitivity
│   └── report.py              # writes outputs/RESULTS.md from the latest run
├── app/dashboard.py           # Streamlit dashboard (5 tabs)
├── scripts/make_sample_data.py  # synthetic data with the same schema, for testing
├── tests/test_core.py         # unit tests for the financial / decision logic
└── docs/interview_guide.md    # 90-second pitch + how to defend each choice
```

## How to run

```bash
pip install -r requirements.txt

# 1. Download from https://www.kaggle.com/competitions/home-credit-default-risk/data
#    and put these three files in data/raw/:
#    application_train.csv, bureau.csv, previous_application.csv
python run_pipeline.py

# 2. Open the dashboard
streamlit run app/dashboard.py
```

No data yet? Test the whole pipeline on a synthetic sample with the same columns:

```bash
python scripts/make_sample_data.py --rows 60000
python run_pipeline.py --sample
```

Run the tests with `python -m pytest -q`.

The full run on the real data takes a few minutes, mostly for Random Forest and XGBoost.
Headline numbers are written to **`outputs/RESULTS.md`** after each run.

---

## Methodology

### 1. Data engineering (SQL)
* `bureau` (about 1.7M rows) and `previous_application` (about 1.7M rows) are aggregated to **one row
  per applicant**. The aggregates include loan counts, active debt, utilisation, overdue flags, refusal rate and
  granted-vs-applied ratio.
* `03_build_model_table.sql` LEFT JOINs them onto the application table and adds the ratios
  a credit analyst uses: credit/income, EMI/income, credit/goods price (LTV-like), age and employment years.
* **Join validation:** the pipeline asserts that row count = distinct applicants = original rows.
  This proves the joins didn't duplicate anyone.
* **Leakage check:** every bureau and previous-application record is dated before the current
  application (DAYS_* <= 0), so no feature uses information from after the decision.
* **Data quality:** `DAYS_EMPLOYED = 365243` is a placeholder (about 18% of rows, mostly pensioners).
  It is set to NULL and gets its own flag. Missing categories become a `"Missing"` level.

### 2. Modelling
| Choice | Why |
|---|---|
| 60/20/20 train / calibration / test split, stratified | The calibration set is used to pick the champion, calibrate it and tune the threshold. The test set is touched once. |
| Logistic Regression, Random Forest, XGBoost | Interpretable baseline, then non-linear benchmark, then strongest tabular model. |
| No class weights or SMOTE | Rebalancing makes probabilities wrong. The profit model needs a PD of 5% to really mean 5%. |
| Preprocessing fitted on train only | Imputation medians and one-hot levels can't leak from calib or test. |
| Champion must be explainable (LR or XGBoost) | A declined customer must get reasons. RF stays as a challenger. |
| Platt calibration | One-variable logistic regression on the log-odds. It fixes the level of the PDs without changing the ranking, so AUC is unchanged. |
| Metrics: ROC-AUC, Gini, KS, PR-AUC, Brier, decile lift | Gini and KS are the standard metrics in bank scorecards. PR-AUC suits the 8% bad rate. Brier checks calibration. |
| Reason codes from SHAP (XGBoost) or coef x value (LR) | These are the top 3 features pushing each applicant toward default. |
| `CODE_GENDER` excluded | Protected attributes shouldn't drive credit decisions. |

### 3. Financial model (one year, per loan)

```
interest income   = (1 - PD) x r x EAD
funding cost      = cost_of_funds x EAD
operating cost    = opex x EAD
expected loss     = PD x LGD x EAD
expected profit   = interest - funding - opex - expected loss
RAROC             = expected profit / (EAD x risk weight x capital ratio)
```

Since every term is proportional to EAD, a loan is profitable only when its PD is below the break-even PD:

```
PD* = (r - cost_of_funds - opex) / (r + LGD) = (0.14 - 0.065 - 0.02) / (0.14 + 0.60) = 7.43%
```

### 4. Decision optimisation
* The pipeline sweeps PD thresholds from 1% to 50% on the calibration set.
* Constraints come from the bank's risk appetite: expected bad rate of the approved book <= 5%, and approval
  rate >= 60%. Both are set in `config.py`.
* It picks the feasible threshold with the highest expected profit. If the two constraints conflict,
  the **risk limit wins** over the growth target.
* It back-tests four strategies on the test set using the actual outcomes: approve everyone, a
  rule-of-thumb cut-off (PD < 10%), break-even, and risk-appetite optimised.
* A sensitivity grid (LGD 45/60/75% x rate 12/14/16%) shows how the policy moves if the assumptions are wrong.

### Key assumptions and limitations
* `TARGET` means "had payment difficulties". It is a **proxy** for default, not the regulatory
  90+ DPD definition.
* EAD = `AMT_CREDIT`, with no amortisation within the year and a one-period horizon.
* LGD is a flat 60%, typical for unsecured retail. There is no collateral or recovery data.
* Interest rate, cost of funds and opex are scenario inputs and can be changed in the dashboard.
* The split is random rather than out-of-time, because the dataset has no application dates.
  In production I would validate on a later time window and monitor PSI.
* Declined applicants have no outcomes (reject inference). The model only learns from loans that were granted.

## Dashboard

| Tab | What it shows |
|---|---|
| Portfolio | Recommended cut-off, approval rate, bad rate, expected vs realised profit, RAROC, risk grades, strategy comparison |
| Credit model | AUC / Gini / KS, model comparison, ROC, calibration, top risk drivers, decile table |
| Profit optimiser | Break-even formula, profit vs threshold, constraint charts, P&L waterfall |
| Applicant decision | PD, grade, decision, reason codes, loan P&L, the **break-even interest rate** (for risk-based pricing) |
| Risk insights (SQL) | Bad rate by income decile, EMI burden, bureau history, past refusals, age band |

The sidebar holds all financial assumptions and risk limits, so every number updates live.

## Possible extensions
* Add `installments_payments` and `credit_card_balance` for behavioural features.
* Risk-based pricing: optimise the rate per grade instead of approve or decline.
* A multi-period, amortising cash-flow model with discounting (NPV per loan).
* Out-of-time validation and a PSI / drift monitor.
* Fairness checks of approval rates across groups.
