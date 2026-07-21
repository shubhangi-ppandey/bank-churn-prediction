"""Exploratory Data Analysis for European Bank churn data.

Generates a data-quality report and a set of figures saved under
reports/figures/. Run from the project root:

    python src/eda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_engineering import prepare

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "European_Bank.csv"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def quality_report(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "missing_per_col": df.isna().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "target_distribution": df["Exited"].value_counts().to_dict(),
        "churn_rate": float(df["Exited"].mean()),
        "geography_counts": df["Geography"].value_counts().to_dict(),
        "gender_counts": df["Gender"].value_counts().to_dict(),
        "numerical_summary": df.describe().round(3).to_dict(),
    }
    return report


def plot_target_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Exited"].value_counts().sort_index()
    bars = ax.bar(["Retained (0)", "Churned (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 50, f"{v:,}\n({v/len(df):.1%})", ha="center")
    ax.set_title("Target Balance — Exited")
    ax.set_ylabel("Customers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_target_balance.png", dpi=120)
    plt.close(fig)


def plot_churn_by_category(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["Geography", "Gender", "IsActiveMember"]):
        rates = df.groupby(col)["Exited"].mean().sort_values(ascending=False)
        sns.barplot(x=rates.index.astype(str), y=rates.values, ax=ax, color="#C44E52")
        ax.set_title(f"Churn rate by {col}")
        ax.set_ylabel("Churn rate")
        ax.set_ylim(0, max(0.5, rates.max() * 1.2))
        for i, v in enumerate(rates.values):
            ax.text(i, v + 0.005, f"{v:.1%}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_churn_by_category.png", dpi=120)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame) -> None:
    cols = ["Age", "CreditScore", "Balance", "EstimatedSalary", "Tenure", "NumOfProducts"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), cols):
        for label, color in [(0, "#4C72B0"), (1, "#C44E52")]:
            sns.kdeplot(
                df[df["Exited"] == label][col],
                ax=ax,
                fill=True,
                alpha=0.4,
                color=color,
                label=f"Exited={label}",
                warn_singular=False,
            )
        ax.set_title(col)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_distributions_by_target.png", dpi=120)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame) -> None:
    feat = prepare(df)
    feat = pd.get_dummies(feat, columns=["Geography", "Gender"], drop_first=False)
    corr = feat.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation matrix (engineered features)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=120)
    plt.close(fig)


def plot_age_tenure_segments(df: pd.DataFrame) -> None:
    df2 = df.copy()
    df2["AgeBand"] = pd.cut(
        df2["Age"], bins=[17, 30, 40, 50, 60, 100], labels=["18-30", "31-40", "41-50", "51-60", "60+"]
    )
    pivot = df2.groupby(["AgeBand", "NumOfProducts"], observed=True)["Exited"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds", ax=ax)
    ax.set_title("Churn rate: Age band × NumOfProducts")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_age_products_heatmap.png", dpi=120)
    plt.close(fig)


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    report = quality_report(df)
    (REPORT_DIR / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Churn rate: {report['churn_rate']:.2%}")
    print(f"Missing values per column: {sum(report['missing_per_col'].values())}")
    print(f"Duplicates: {report['duplicates']}")

    plot_target_balance(df)
    plot_churn_by_category(df)
    plot_distributions(df)
    plot_correlation(df)
    plot_age_tenure_segments(df)
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
