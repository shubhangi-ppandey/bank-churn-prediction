# Predictive Modeling and Risk Scoring for Bank Customer Churn

End-to-end churn-prediction system for a European retail-banking portfolio of 10,000 customers.

## Quick start

```bash
pip install -r requirements.txt

# 1. EDA + figures
python src/eda.py

# 2. Train all models, save the winner
python src/train.py

# 3. Generate explainability artefacts (SHAP, PDP)
python src/explain.py

# 4. Launch the dashboard
streamlit run app/streamlit_app.py
```

## Layout

```
.
├── European_Bank.csv            input data (10,000 × 14)
├── requirements.txt
├── src/
│   ├── feature_engineering.py   derived behavioural features
│   ├── preprocessing.py         ColumnTransformer factory
│   ├── eda.py                   data quality + distribution figures
│   ├── train.py                 5 models + tuning + evaluation
│   └── explain.py               SHAP + partial-dependence plots
├── app/
│   └── streamlit_app.py         5-module interactive dashboard
├── models/
│   ├── best_model.pkl           full Pipeline (preprocess + classifier)
│   └── metadata.json            metrics, feature names, defaults
└── reports/
    ├── research_paper.md
    ├── executive_summary.md
    ├── data_quality.json
    ├── model_metrics.csv
    ├── feature_importance.csv
    ├── shap_values.npz
    └── figures/                 12 PNG plots
```

## Headline result

| Metric | Value |
|---|---|
| Best model | Gradient Boosting (tuned) |
| Test ROC-AUC | 0.870 |
| Test PR-AUC | 0.720 |
| Test F1 | 0.602 |
| Precision | 0.794 |
| Recall (default 0.5 threshold) | 0.484 |

## Top churn drivers
1. Age (especially 50–60)
2. NumOfProducts (3+ → near-certain churn)
3. Engagement (active membership × products)
4. Balance
5. Germany residency

See `reports/research_paper.md` for the full analysis and `reports/executive_summary.md` for the board-level summary.
