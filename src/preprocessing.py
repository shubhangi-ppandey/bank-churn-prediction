"""Preprocessing pipeline factory.

Builds an sklearn ColumnTransformer that one-hot encodes Geography/Gender
and standard-scales numerical columns. Designed to slot directly into a
Pipeline so the same transforms are saved with the trained model and reused
in the Streamlit app.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["Geography", "Gender"]
NUMERICAL = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementScore",
    "AgeTenureRatio",
    "ZeroBalance",
]
TARGET = "Exited"


def build_preprocessor(scale_numerical: bool = True) -> ColumnTransformer:
    num_steps = [("scaler", StandardScaler())] if scale_numerical else []
    transformers = [
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            CATEGORICAL,
        ),
        (
            "num",
            Pipeline(num_steps) if num_steps else "passthrough",
            NUMERICAL,
        ),
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return human-readable feature names after fitting."""
    cat_pipe = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_pipe.get_feature_names_out(CATEGORICAL))
    return cat_names + list(NUMERICAL)
