"""
Step 6: risk-adjusted profitability of each loan.

One-period (1 year) view of a loan with exposure EAD:

    interest income   = (1 - PD) * rate * EAD     (a defaulter pays no interest)
    funding cost      = cost_of_funds * EAD       (paid whether or not the loan defaults)
    operating cost    = opex * EAD
    expected loss     = PD * LGD * EAD            (ECL = PD x LGD x EAD)

    expected profit   = interest - funding - operating - expected loss

    RAROC             = expected profit / capital held
    capital held      = EAD * risk_weight * capital_ratio

Assumptions to state clearly in any discussion:
  * TARGET in Home Credit = "had payment difficulties", used here as a
    proxy for default - it is not the regulatory 90-days-past-due definition.
  * EAD = AMT_CREDIT (full amount drawn, no amortisation within the year).
  * LGD is a flat assumption (no collateral data for unsecured loans).
  * All rates come from config.FINANCE and are scenario inputs, not bank data.
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from config import FINANCE


@dataclass
class Assumptions:
    interest_rate: float = FINANCE["interest_rate"]
    cost_of_funds: float = FINANCE["cost_of_funds"]
    operating_cost: float = FINANCE["operating_cost"]
    lgd: float = FINANCE["lgd"]
    capital_ratio: float = FINANCE["capital_ratio"]
    risk_weight: float = FINANCE["risk_weight"]

    def to_dict(self):
        return asdict(self)


def loan_economics(pd_, ead, a: Assumptions):
    """Expected P&L components for each loan. pd_ and ead are arrays."""
    pd_ = np.asarray(pd_, dtype=float)
    ead = np.asarray(ead, dtype=float)

    interest = (1 - pd_) * a.interest_rate * ead
    funding = a.cost_of_funds * ead
    opex = a.operating_cost * ead
    ecl = pd_ * a.lgd * ead
    profit = interest - funding - opex - ecl
    capital = ead * a.risk_weight * a.capital_ratio

    return pd.DataFrame({
        "interest_income": interest,
        "funding_cost": funding,
        "operating_cost": opex,
        "expected_credit_loss": ecl,
        "expected_profit": profit,
        "capital": capital,
    })


def realized_profit(target, ead, a: Assumptions):
    """Back-test: what each loan actually earned given its observed outcome.
    Same formula with PD replaced by the actual 0/1 outcome."""
    return loan_economics(target, ead, a)["expected_profit"].to_numpy()


def break_even_pd(a: Assumptions):
    """PD at which expected profit = 0. Because every term above is proportional
    to EAD, this cut-off is the same for every loan size:

        (1-PD)*r - cof - opex - PD*LGD = 0   ->   PD* = (r - cof - opex) / (r + LGD)
    """
    return (a.interest_rate - a.cost_of_funds - a.operating_cost) / (a.interest_rate + a.lgd)
