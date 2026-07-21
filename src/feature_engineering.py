"""Feature engineering for European Bank churn data.

Adds five derived features designed to capture customer engagement and
product utilization signals beyond raw demographics:

  - BalanceSalaryRatio    : wealth concentration relative to income
  - ProductDensity        : products acquired per year of tenure
  - EngagementScore       : active membership weighted by product breadth
  - AgeTenureRatio        : life-stage relative to bank relationship
  - ZeroBalance           : flag for the large segment of dormant balances
"""

from __future__ import annotations

import pandas as pd

DROP_COLS = ["Year", "CustomerId", "Surname"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["ProductDensity"] = out["NumOfProducts"] / (out["Tenure"] + 1.0)
    out["EngagementScore"] = out["IsActiveMember"] * out["NumOfProducts"]
    out["AgeTenureRatio"] = out["Age"] / (out["Tenure"] + 1.0)
    out["ZeroBalance"] = (out["Balance"] == 0).astype(int)
    return out


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_derived_features(drop_identifiers(df))
