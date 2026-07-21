"""Train and evaluate churn models, save the best pipeline.

Five candidates are trained on a stratified 80/20 split with 5-fold CV:
  - Logistic Regression (interpretability baseline)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost

The winner is chosen on test ROC-AUC. Top two candidates are
hyperparameter-tuned via RandomizedSearchCV. Outputs:
  - models/best_model.pkl    : full Pipeline (preprocessor + classifier)
  - models/metadata.json     : feature lists, train shape, churn rate
  - reports/model_metrics.csv
  - reports/figures/06_roc_curves.png
  - reports/figures/07_confusion_matrix_best.png
  - reports/figures/08_pr_curves.png
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from feature_engineering import prepare
from preprocessing import TARGET, build_preprocessor, get_feature_names

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "European_Bank.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="muted")


def make_models() -> dict[str, Pipeline]:
    base = lambda clf: Pipeline([("pre", build_preprocessor()), ("clf", clf)])
    return {
        "LogisticRegression": base(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "DecisionTree": base(
            DecisionTreeClassifier(
                max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
            )
        ),
        "RandomForest": base(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "GradientBoosting": base(
            GradientBoostingClassifier(random_state=RANDOM_STATE)
        ),
        "XGBoost": base(_make_xgb()),
    }


def _make_xgb():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }


def cross_val_auc(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in skf.split(X, y):
        m = joblib.load(joblib.dump(model, MODEL_DIR / "_tmp.pkl")[0])  # clone via pickle
        m.fit(X.iloc[tr], y.iloc[tr])
        proba = m.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y.iloc[va], proba))
    (MODEL_DIR / "_tmp.pkl").unlink(missing_ok=True)
    return float(np.mean(scores))


def tune_top_two(results: pd.DataFrame, models: dict, X_train, y_train) -> dict:
    top2 = results.sort_values("roc_auc", ascending=False).head(2)["model"].tolist()
    print(f"Tuning top-2: {top2}")
    tuned = {}
    param_grids = {
        "RandomForest": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 8, 12, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "clf__n_estimators": [100, 200, 300],
            "clf__learning_rate": [0.05, 0.1, 0.2],
            "clf__max_depth": [3, 4, 5],
            "clf__subsample": [0.8, 1.0],
        },
        "XGBoost": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
        },
        "LogisticRegression": {
            "clf__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        },
        "DecisionTree": {
            "clf__max_depth": [4, 6, 8, 12, None],
            "clf__min_samples_split": [2, 5, 10],
        },
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name in top2:
        if name not in param_grids:
            continue
        rs = RandomizedSearchCV(
            models[name],
            param_grids[name],
            n_iter=15,
            scoring="roc_auc",
            cv=skf,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        rs.fit(X_train, y_train)
        print(f"  {name}: best CV ROC-AUC = {rs.best_score_:.4f} | {rs.best_params_}")
        tuned[name] = rs.best_estimator_
    return tuned


def plot_roc_pr(models: dict, X_test, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set(title="ROC curves", xlabel="FPR", ylabel="TPR")
    axes[0].legend(loc="lower right")
    axes[1].set(title="Precision–Recall curves", xlabel="Recall", ylabel="Precision")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_roc_curves.png", dpi=120)
    plt.close(fig)


def plot_confusion(model: Pipeline, X_test, y_test, name: str) -> None:
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_confusion_matrix_best.png", dpi=120)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = prepare(df)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, churn={y.mean():.2%}")

    models = make_models()
    rows = []
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}…")
        model.fit(X_train, y_train)
        fitted[name] = model
        m = evaluate(model, X_test, y_test)
        m["model"] = name
        rows.append(m)
        print(
            f"  acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"f1={m['f1']:.3f} auc={m['roc_auc']:.3f}"
        )

    results = pd.DataFrame(rows).set_index("model")[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ].round(4)
    print("\nBaseline results:\n", results)

    # Tune top 2
    tuned = tune_top_two(results.reset_index(), models, X_train, y_train)
    for name, m in tuned.items():
        fitted[f"{name}_tuned"] = m
        eval_m = evaluate(m, X_test, y_test)
        eval_m["model"] = f"{name}_tuned"
        rows.append(eval_m)

    final = pd.DataFrame(rows).set_index("model")[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ].round(4)
    final = final.sort_values("roc_auc", ascending=False)
    print("\nFinal results (sorted by ROC-AUC):\n", final)
    final.to_csv(REPORT_DIR / "model_metrics.csv")

    best_name = final.index[0]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    plot_roc_pr(fitted, X_test, y_test)
    plot_confusion(best_model, X_test, y_test, best_name)

    # Save model + metadata
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    feature_names = get_feature_names(best_model.named_steps["pre"])
    metadata = {
        "best_model": best_name,
        "metrics": final.loc[best_name].to_dict(),
        "feature_names_after_preprocessing": feature_names,
        "raw_input_columns": list(X.columns),
        "categorical": ["Geography", "Gender"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "random_state": RANDOM_STATE,
        "feature_value_ranges": {
            c: [float(X[c].min()), float(X[c].max())] for c in X.select_dtypes("number").columns
        },
        "feature_defaults": {
            c: float(X[c].median()) if X[c].dtype.kind in "iuf" else X[c].mode()[0]
            for c in X.columns
        },
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Saved best model -> {MODEL_DIR/'best_model.pkl'}")


if __name__ == "__main__":
    main()
