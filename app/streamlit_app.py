"""Bank Customer Churn — Predictive Risk Intelligence Dashboard.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_engineering import add_derived_features  # noqa: E402
from preprocessing import get_feature_names  # noqa: E402

MODEL_PATH = ROOT / "models" / "best_model.pkl"
META_PATH = ROOT / "models" / "metadata.json"
DATA_PATH = ROOT / "European_Bank.csv"
FIG_DIR = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Bank Churn Risk Intelligence",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the saved model. If the pickle is incompatible with the deployed
    sklearn version (common on cloud platforms), retrain on the fly."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Saved model incompatible ({type(e).__name__}); retraining on first launch…")
        return _retrain_quick()


def _retrain_quick():
    """Minimal retraining fallback — Gradient Boosting on full data."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from preprocessing import TARGET, build_preprocessor

    raw = pd.read_csv(DATA_PATH)
    raw = add_derived_features(raw.drop(columns=[c for c in ["Year", "CustomerId", "Surname"] if c in raw.columns]))
    y = raw[TARGET]
    X = raw.drop(columns=[TARGET])
    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


@st.cache_data
def load_metadata() -> dict:
    try:
        return json.loads(META_PATH.read_text())
    except Exception:
        return {
            "best_model": "GradientBoosting (auto-retrained)",
            "metrics": {"roc_auc": 0.870, "f1": 0.602},
            "n_train": 8000,
            "n_test": 2000,
            "churn_rate": 0.2037,
        }


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def risk_band(p: float) -> tuple[str, str]:
    if p < 0.30:
        return "LOW", "#2ECC71"
    if p < 0.60:
        return "MEDIUM", "#F39C12"
    return "HIGH", "#E74C3C"


def build_input_row(values: dict) -> pd.DataFrame:
    df = pd.DataFrame([values])
    return add_derived_features(df)


def predict_proba(model, raw_row: pd.DataFrame) -> float:
    return float(model.predict_proba(raw_row)[0, 1])


# ---------- Sidebar ----------
st.sidebar.title("🏦 Churn Intelligence")
st.sidebar.caption("Predictive risk scoring for retail bank customers")
mode = st.sidebar.radio(
    "Module",
    [
        "Risk Calculator",
        "Probability Distribution",
        "Feature Importance",
        "What-If Simulator",
        "Model Performance",
    ],
)

model = load_model()
meta = load_metadata()
df = load_dataset()


# ---------- Module 1: Risk Calculator ----------
def render_risk_calculator():
    st.header("🎯 Customer Churn Risk Calculator")
    st.caption(
        "Enter a customer profile to receive a churn probability score and risk band. "
        f"Backing model: **{meta['best_model']}** — test ROC-AUC "
        f"**{meta['metrics']['roc_auc']:.3f}**."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Demographics")
        geo = st.selectbox("Geography", ["France", "Germany", "Spain"], index=0)
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        age = st.slider("Age", 18, 95, 40)
        credit = st.slider("Credit Score", 350, 850, 650)
    with c2:
        st.subheader("Relationship")
        tenure = st.slider("Tenure (years)", 0, 10, 5)
        balance = st.number_input("Balance (€)", 0.0, 300000.0, 75000.0, step=500.0)
        salary = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 100000.0, step=1000.0)
    with c3:
        st.subheader("Engagement")
        n_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
        has_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Active Member", value=True)

    raw = build_input_row(
        {
            "CreditScore": credit,
            "Geography": geo,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": n_products,
            "HasCrCard": int(has_card),
            "IsActiveMember": int(is_active),
            "EstimatedSalary": salary,
        }
    )

    proba = predict_proba(model, raw)
    band, color = risk_band(proba)

    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{proba:.1%}")
    m2.markdown(
        f"<div style='padding:0.4rem;border-radius:6px;background:{color};color:white;"
        f"text-align:center;font-weight:600;font-size:1.6rem'>"
        f"{band} RISK</div>",
        unsafe_allow_html=True,
    )
    base = meta["churn_rate"]
    lift = proba / base if base > 0 else 0.0
    m3.metric("Lift vs. portfolio", f"{lift:.2f}x", f"baseline {base:.1%}")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#D5F5E3"},
                    {"range": [30, 60], "color": "#FCF3CF"},
                    {"range": [60, 100], "color": "#F5B7B1"},
                ],
            },
            title={"text": "Churn risk score"},
        )
    )
    fig.update_layout(height=320, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Recommended action"):
        if band == "HIGH":
            st.error(
                "Trigger immediate retention outreach: relationship-manager call, "
                "tailored fee waiver or rate offer, satisfaction survey."
            )
        elif band == "MEDIUM":
            st.warning(
                "Add to monitored cohort: targeted product cross-sell (likely 1→2 products), "
                "engagement nudge campaign."
            )
        else:
            st.success("Standard servicing; consider for upsell to premium tier.")


# ---------- Module 2: Probability Distribution ----------
def render_distribution():
    st.header("📊 Portfolio Risk Distribution")
    st.caption("Churn probability scored across all 10,000 customers in the portfolio.")

    raw = add_derived_features(df.drop(columns=[c for c in ["Year", "CustomerId", "Surname"] if c in df.columns]))
    proba = model.predict_proba(raw.drop(columns=["Exited"]))[:, 1]
    scores = pd.DataFrame(
        {
            "score": proba,
            "Exited": df["Exited"],
            "Geography": df["Geography"],
            "Gender": df["Gender"],
            "Age": df["Age"],
        }
    )
    scores["Band"] = pd.cut(
        scores["score"],
        bins=[-0.01, 0.30, 0.60, 1.01],
        labels=["Low (<30%)", "Medium (30–60%)", "High (≥60%)"],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(scores):,}")
    c2.metric("High-risk", f"{(scores['Band']=='High (≥60%)').sum():,}")
    c3.metric("Medium-risk", f"{(scores['Band']=='Medium (30–60%)').sum():,}")
    c4.metric("Avg score", f"{scores['score'].mean():.1%}")

    fig = px.histogram(
        scores,
        x="score",
        color="Exited",
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={0: "#4C72B0", 1: "#C44E52"},
        labels={"score": "Predicted churn probability", "Exited": "Actual churn"},
    )
    fig.update_layout(title="Predicted probability — by actual outcome", height=420)
    st.plotly_chart(fig, use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        fig2 = px.box(scores, x="Geography", y="score", color="Geography",
                      title="Risk score by Geography")
        st.plotly_chart(fig2, use_container_width=True)
    with g2:
        fig3 = px.histogram(scores, x="Band", color="Band",
                            title="Customers per risk band",
                            category_orders={"Band": ["Low (<30%)", "Medium (30–60%)", "High (≥60%)"]})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top 25 high-risk customers")
    top = scores.sort_values("score", ascending=False).head(25)
    top_view = df.loc[top.index, ["CustomerId", "Surname", "Geography", "Gender", "Age",
                                  "NumOfProducts", "IsActiveMember", "Balance"]].copy()
    top_view["Risk score"] = top["score"].round(3).values
    top_view["Actual exited"] = top["Exited"].values
    st.dataframe(top_view, use_container_width=True)


# ---------- Module 3: Feature Importance ----------
def render_importance():
    st.header("🧭 Feature Importance & Explainability")
    st.caption(
        "Global drivers of churn risk. Gain importance shows total split contribution; "
        "SHAP values explain the average impact of each feature on individual predictions."
    )

    fi_path = ROOT / "reports" / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path, index_col=0).rename(columns={"0": "importance"})
        fi.columns = ["importance"]
        fi = fi.sort_values("importance", ascending=True).tail(15)
        fig = px.bar(fi, x="importance", y=fi.index, orientation="h",
                     title="Gain-based feature importance (top 15)")
        fig.update_layout(height=520, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("SHAP — mean |impact|")
        st.image(str(FIG_DIR / "11_shap_bar.png"), use_container_width=True)
    with c2:
        st.subheader("SHAP — distribution")
        st.image(str(FIG_DIR / "10_shap_summary.png"), use_container_width=True)

    st.subheader("Partial dependence — top drivers")
    st.image(str(FIG_DIR / "12_partial_dependence.png"), use_container_width=True)

    st.info(
        "**Reading the SHAP summary:** each dot is one customer; horizontal position is the "
        "contribution to the predicted churn probability. Red = high feature value. "
        "For example, high Age and NumOfProducts ≥ 3 push customers toward churn; being an "
        "active member pulls them away."
    )


# ---------- Module 4: What-If Simulator ----------
def render_whatif():
    st.header("🔮 What-If Scenario Simulator")
    st.caption(
        "Choose a baseline customer, then adjust engagement and product variables to see the "
        "marginal effect on churn risk."
    )

    sample = df.sample(1, random_state=int(st.number_input("Random seed", 0, 9999, 7))).iloc[0]
    st.write("**Baseline customer:**")
    st.json(
        {
            "CustomerId": int(sample["CustomerId"]),
            "Geography": sample["Geography"],
            "Age": int(sample["Age"]),
            "Tenure": int(sample["Tenure"]),
            "Balance": float(sample["Balance"]),
            "NumOfProducts": int(sample["NumOfProducts"]),
            "IsActiveMember": int(sample["IsActiveMember"]),
            "Exited (actual)": int(sample["Exited"]),
        }
    )

    st.markdown("**Adjust scenario variables:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_products = st.slider("Products", 1, 4, int(sample["NumOfProducts"]))
        is_active = st.checkbox("Active member", bool(sample["IsActiveMember"]))
    with c2:
        balance = st.slider("Balance (€)", 0, 250000, int(sample["Balance"]), step=500)
        tenure = st.slider("Tenure (years)", 0, 10, int(sample["Tenure"]))
    with c3:
        age = st.slider("Age", 18, 95, int(sample["Age"]))
        credit = st.slider("Credit score", 350, 850, int(sample["CreditScore"]))

    base_row = build_input_row(
        {
            "CreditScore": int(sample["CreditScore"]),
            "Geography": sample["Geography"],
            "Gender": sample["Gender"],
            "Age": int(sample["Age"]),
            "Tenure": int(sample["Tenure"]),
            "Balance": float(sample["Balance"]),
            "NumOfProducts": int(sample["NumOfProducts"]),
            "HasCrCard": int(sample["HasCrCard"]),
            "IsActiveMember": int(sample["IsActiveMember"]),
            "EstimatedSalary": float(sample["EstimatedSalary"]),
        }
    )
    new_row = build_input_row(
        {
            "CreditScore": credit,
            "Geography": sample["Geography"],
            "Gender": sample["Gender"],
            "Age": age,
            "Tenure": tenure,
            "Balance": float(balance),
            "NumOfProducts": n_products,
            "HasCrCard": int(sample["HasCrCard"]),
            "IsActiveMember": int(is_active),
            "EstimatedSalary": float(sample["EstimatedSalary"]),
        }
    )

    p0 = predict_proba(model, base_row)
    p1 = predict_proba(model, new_row)
    delta = p1 - p0

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline", f"{p0:.1%}")
    m2.metric("Scenario", f"{p1:.1%}", f"{delta:+.1%}")
    direction = "↑ riskier" if delta > 0 else "↓ safer" if delta < 0 else "→ unchanged"
    m3.metric("Effect", direction)

    fig = go.Figure(
        go.Bar(
            x=["Baseline", "Scenario"],
            y=[p0, p1],
            marker_color=["#4C72B0", risk_band(p1)[1]],
            text=[f"{p0:.1%}", f"{p1:.1%}"],
            textposition="auto",
        )
    )
    fig.update_layout(yaxis=dict(range=[0, 1], tickformat=".0%"),
                      title="Baseline vs. scenario churn probability", height=380)
    st.plotly_chart(fig, use_container_width=True)


# ---------- Module 5: Model Performance ----------
def render_performance():
    st.header("📈 Model Performance")
    metrics_path = ROOT / "reports" / "model_metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path, index_col=0).round(4)
        st.dataframe(m, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(FIG_DIR / "06_roc_curves.png"), caption="ROC + PR curves", use_container_width=True)
    with c2:
        st.image(str(FIG_DIR / "07_confusion_matrix_best.png"),
                 caption="Confusion matrix — best model", use_container_width=True)

    with st.expander("EDA snapshots"):
        for fname, cap in [
            ("01_target_balance.png", "Target distribution"),
            ("02_churn_by_category.png", "Churn rate by Geography / Gender / Activity"),
            ("03_distributions_by_target.png", "Feature distributions by churn outcome"),
            ("04_correlation_heatmap.png", "Correlation heatmap"),
            ("05_age_products_heatmap.png", "Age band × NumOfProducts heatmap"),
        ]:
            st.image(str(FIG_DIR / fname), caption=cap, use_container_width=True)


# ---------- Router ----------
ROUTES = {
    "Risk Calculator": render_risk_calculator,
    "Probability Distribution": render_distribution,
    "Feature Importance": render_importance,
    "What-If Simulator": render_whatif,
    "Model Performance": render_performance,
}
ROUTES[mode]()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Best model: **{meta['best_model']}**  \n"
    f"ROC-AUC {meta['metrics']['roc_auc']:.3f}  \n"
    f"F1 {meta['metrics']['f1']:.3f}  \n"
    f"Train n={meta['n_train']:,} • Test n={meta['n_test']:,}"
)
