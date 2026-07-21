"""Generate explainability artifacts for the saved best model.

Outputs:
  - reports/figures/09_feature_importance.png  (gain-based)
  - reports/figures/10_shap_summary.png        (SHAP beeswarm)
  - reports/figures/11_shap_bar.png            (mean |SHAP|)
  - reports/figures/12_partial_dependence.png  (PDP for top-5 drivers)
  - reports/shap_values.npz                    (cached SHAP for the app)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import PartialDependenceDisplay

from feature_engineering import prepare
from preprocessing import TARGET, get_feature_names

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "European_Bank.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"


def main() -> None:
    pipeline = joblib.load(MODEL_DIR / "best_model.pkl")
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    feature_names = get_feature_names(pre)

    df = prepare(pd.read_csv(DATA_PATH))
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_trans = pre.transform(X)
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)

    # 1. Gain-based feature importance
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.tail(15).plot(kind="barh", ax=ax, color="#4C72B0")
        ax.set_title("Feature importance (gain) — top 15")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "09_feature_importance.png", dpi=120)
        plt.close(fig)
        imp.sort_values(ascending=False).to_csv(REPORT_DIR / "feature_importance.csv")
        print("Saved gain-based importance")

    # 2. SHAP — sample 1000 for speed
    sample_idx = np.random.RandomState(42).choice(len(X_trans_df), size=min(1000, len(X_trans_df)), replace=False)
    Xs = X_trans_df.iloc[sample_idx]
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(Xs)
    if isinstance(shap_vals, list):  # binary classifier returning list
        shap_vals = shap_vals[1]

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, Xs, feature_names=feature_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_shap_summary.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, Xs, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_shap_bar.png", dpi=120)
    plt.close()
    print("Saved SHAP plots")

    # cache for streamlit
    np.savez(
        REPORT_DIR / "shap_values.npz",
        shap_values=shap_vals,
        X_sample=Xs.values,
        feature_names=np.array(feature_names),
        expected_value=np.array(explainer.expected_value).reshape(-1),
    )

    # 3. Partial dependence on top-5 raw features
    raw_top = ["Age", "NumOfProducts", "Balance", "IsActiveMember", "CreditScore"]
    X_pdp = X.copy()
    for c in X_pdp.select_dtypes(include="integer").columns:
        X_pdp[c] = X_pdp[c].astype(float)
    fig, ax = plt.subplots(figsize=(14, 8))
    PartialDependenceDisplay.from_estimator(
        pipeline,
        X_pdp,
        features=raw_top,
        ax=ax,
        kind="average",
        grid_resolution=30,
    )
    fig.suptitle("Partial dependence — top drivers")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_partial_dependence.png", dpi=120)
    plt.close(fig)
    print("Saved PDP")


if __name__ == "__main__":
    main()
