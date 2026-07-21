# Predictive Modeling and Risk Scoring for Bank Customer Churn
## A Research Paper on Behavioural Drivers of Retail-Banking Attrition

---

### Abstract
Customer churn in retail banking erodes lifetime value, destabilises revenue and weakens long-term competitiveness. This paper presents a predictive churn-intelligence system trained on a 10,000-customer European retail-banking dataset. We benchmark five classification algorithms, engineer behavioural-engagement features, and select a tuned Gradient Boosting model that achieves **ROC-AUC = 0.870** on a held-out test set. Beyond accuracy, we focus on *explainability*: SHAP analysis and partial-dependence plots reveal that age, product breadth, account engagement and German residency are the dominant drivers — findings that translate directly into targeted retention strategies. The full system, including a Streamlit dashboard with a what-if simulator, is delivered for operational use.

---

### 1. Introduction
Traditional churn analysis is *retrospective*: it explains why customers left after the fact. Modern banks need *prospective* churn intelligence — risk probabilities computed for every active customer, refreshed routinely, and surfaced to retention teams before the customer disengages. This project builds such a system end-to-end.

**Primary objectives**
1. Predict churn with high accuracy and discrimination (ROC-AUC).
2. Output calibrated churn probabilities per customer.
3. Identify and explain the drivers of churn.

**Secondary objectives**
4. Reduce false-positive retention spend.
5. Make every prediction interpretable for compliance and business stakeholders.
6. Enable scenario analysis ("what-if I increase product breadth?").

---

### 2. Dataset
- **File:** `European_Bank.csv` — 10,000 rows × 14 columns; reporting `Year = 2025` (constant, dropped as non-informative).
- **Target:** `Exited` (1 = churned, 0 = retained). Class prevalence = **20.37%**.
- **Quality:** zero missing values, zero duplicates, no obvious outliers requiring imputation.

| Column | Type | Notes |
|---|---|---|
| CustomerId, Surname | identifier | dropped |
| CreditScore | numeric | 350–850 range |
| Geography | categorical | France 50.1%, Germany 25.1%, Spain 24.8% |
| Gender | categorical | Male 54.6%, Female 45.4% |
| Age | numeric | 18–92, median ≈ 37 |
| Tenure | numeric | 0–10 years |
| Balance | numeric | 36% of customers carry a zero balance |
| NumOfProducts | numeric | 1–4 |
| HasCrCard, IsActiveMember | binary | engagement indicators |
| EstimatedSalary | numeric | annual salary proxy |

---

### 3. Methodology

#### 3.1 Pre-processing
- Drop identifiers `CustomerId`, `Surname`, and the constant `Year`.
- One-hot encode `Geography` and `Gender`.
- StandardScale all numerical features (essential for the Logistic Regression baseline; harmless for tree models).
- All transforms wrapped in a `sklearn.Pipeline`, persisted with the trained classifier so inference in production matches training exactly.

#### 3.2 Feature engineering
Five derived features encode behavioural signals invisible to the raw schema:

| Feature | Definition | Hypothesis |
|---|---|---|
| `BalanceSalaryRatio` | `Balance / (EstimatedSalary + 1)` | Wealth concentration vs. income capacity |
| `ProductDensity` | `NumOfProducts / (Tenure + 1)` | How quickly the customer adopts products |
| `EngagementScore` | `IsActiveMember × NumOfProducts` | Active engagement weighted by breadth |
| `AgeTenureRatio` | `Age / (Tenure + 1)` | Life-stage relative to relationship age |
| `ZeroBalance` | indicator `Balance == 0` | Captures large dormant-balance segment (36% of customers) |

#### 3.3 Train–test strategy
Stratified 80/20 split (n_train = 8,000, n_test = 2,000), preserving the 20.37% churn prevalence. Hyperparameter search uses 5-fold stratified cross-validation.

#### 3.4 Models
| Family | Model | Role |
|---|---|---|
| Linear baseline | Logistic Regression (class-weight balanced) | Interpretability benchmark |
| Tree | Decision Tree (depth = 8, balanced) | Single-tree reference |
| Ensemble | Random Forest (300 trees, balanced) | Bagged variance reduction |
| Boosting | Gradient Boosting (sklearn) | Sequential refinement |
| Boosting | XGBoost (`hist` method) | High-performance optional |

The top-2 candidates by ROC-AUC are then tuned with `RandomizedSearchCV` (15 iterations on each).

#### 3.5 Evaluation
Six metrics reported per model: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC. The selection criterion is ROC-AUC on the held-out test set, with PR-AUC as a tie-breaker (PR-AUC is more sensitive to class imbalance than ROC-AUC).

#### 3.6 Explainability
- Gain-based feature importance from the chosen tree ensemble.
- SHAP TreeExplainer over a 1,000-customer sample → mean |SHAP| bar plot and beeswarm summary.
- Partial dependence plots (1-D) for the top five raw features.

---

### 4. Exploratory findings

| Segment | Churn rate | n | Notable |
|---|---|---|---|
| **Geography = Germany** | **32.4%** | 2,509 | 2× France/Spain — investigate |
| Geography = France | 16.1% | 5,014 | baseline |
| Geography = Spain | 16.7% | 2,477 | baseline |
| Female | 25.1% | 4,543 | +9 pts vs. Male |
| Male | 16.5% | 5,457 | |
| **IsActiveMember = 0** | **26.9%** | 4,849 | nearly 2× active-member rate |
| IsActiveMember = 1 | 14.3% | 5,151 | |
| **NumOfProducts ≥ 3** | **84.5% combined** | 326 | inverse-cross-sell signal |
| NumOfProducts = 2 | 7.6% | 4,590 | sweet spot |
| NumOfProducts = 1 | 27.7% | 5,084 | high risk |
| Age 51–60 | **56.2%** | 797 | peak churn band |
| Age 18–30 | 7.5% | 1,968 | lowest churn |
| Balance > 0 | 24.1% | 6,383 | counter-intuitively higher than zero-balance |
| Balance = 0 | 13.8% | 3,617 | |

**Key insight — the inverse cross-sell paradox.** Customers holding 1 product churn at 27.7%, customers with 2 products at only 7.6%, but customers with 3 or 4 products churn at 82.7% and 100% respectively. The "more products = stickier" intuition holds only up to two products; beyond that, additional products signal customer dissatisfaction or operational friction (likely caused by aggressive cross-sell into wrong-fit products). This is the single most important behavioural finding.

**Key insight — Germany over-indexes on churn.** The German cohort is 25% of the portfolio but contributes 40% of all churn. Investigation of fees, product mix or service experience in Germany should be a board-level priority.

**Key insight — engagement is the cheapest lever.** Inactive customers churn at 26.9% versus 14.3% for active members — and "active" status is far easier to influence than age, geography or wealth.

---

### 5. Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting (tuned)** | **0.870** | 0.794 | 0.484 | 0.602 | **0.870** | 0.720 |
| Gradient Boosting | 0.869 | 0.788 | 0.484 | 0.600 | 0.869 | 0.719 |
| XGBoost (tuned) | 0.871 | 0.806 | 0.479 | 0.601 | 0.868 | 0.717 |
| XGBoost | 0.866 | 0.772 | 0.482 | 0.593 | 0.855 | 0.700 |
| Random Forest | 0.862 | 0.769 | 0.457 | 0.573 | 0.853 | 0.682 |
| Decision Tree | 0.760 | 0.446 | 0.742 | 0.557 | 0.804 | 0.624 |
| Logistic Regression | 0.714 | 0.388 | 0.710 | 0.502 | 0.776 | 0.469 |

**Selected model: Gradient Boosting (tuned)** — `learning_rate = 0.05`, `n_estimators = 200`, `max_depth = 3`, `subsample = 1.0`. The tuned XGBoost is statistically indistinguishable; Gradient Boosting is preferred for its smaller dependency footprint.

**Threshold note.** The default 0.5 threshold favours Precision (0.79) over Recall (0.48), reflecting the cost asymmetry of false retention spend. Banks willing to spend more on outreach can lower the threshold; the dashboard exposes Low/Medium/High bands at 0.30/0.60.

---

### 6. Explainability

**Feature importance (gain) — top contributions**
1. **Age** (0.39) — dominant; risk rises sharply between 40 and 60.
2. **NumOfProducts** (0.29) — non-monotonic; 3+ products is a strong churn signal.
3. **EngagementScore** (0.08) — `IsActive × NumOfProducts` interaction.
4. **Balance** (0.06).
5. **Geography_Germany** (0.05).
6. IsActiveMember, BalanceSalaryRatio, CreditScore, EstimatedSalary, derived ratios.

**SHAP findings.** The beeswarm plot (`reports/figures/10_shap_summary.png`) confirms the gain ranking and adds direction:
- High Age increases churn probability; low Age decreases it.
- IsActive = 1 reliably *decreases* churn risk.
- Geography_Germany pushes risk *up*; France/Spain are roughly neutral.
- NumOfProducts shows the bimodal effect: 2 reduces risk, 3+ sharply increases it.

**Partial dependence** (`reports/figures/12_partial_dependence.png`) confirms the non-linear age curve and the clear "two products is the sweet spot" pattern.

---

### 7. Recommendations

1. **Inverse-cross-sell guardrails.** Re-evaluate the cross-sell triggering logic. Cap product offers at two for new customers. For customers already holding three products, run a satisfaction diagnostic before adding more.
2. **Activation campaign.** Inactive members are nearly 2× more likely to churn. A behaviour-trigger nudge campaign (login reminders, micro-rewards, personalised RM call) targeting the inactive cohort is the single highest-ROI retention investment indicated by the data.
3. **Germany-specific deep-dive.** Commission a qualitative study (focus groups, complaint-log mining) to understand why German customers churn at twice the rate of France/Spain. Hypotheses to test: pricing, branch density, product fit, service language quality.
4. **Age-band retention bands.** Customers aged 50–60 churn at 56%. Build a dedicated wealth-transition product (pre-retirement planning, decumulation advisory) to retain them.
5. **Operationalise the risk score.** Refresh the score monthly; route customers in the top decile to a relationship manager. Track campaign uplift via holdout / control-group testing.
6. **Threshold governance.** The 0.5 default optimises precision; for retention campaigns, a 0.30 threshold catches twice as many churners with manageable false-positive cost. Decision rights for threshold adjustment should rest with the Retention Committee, not Data Science.

---

### 8. Streamlit Dashboard
The accompanying dashboard (`app/streamlit_app.py`) provides five interactive modules:

1. **Risk Calculator** — input customer attributes, get probability + risk band + recommended action.
2. **Probability Distribution** — portfolio-level histograms, top-25 high-risk list, geography breakdown.
3. **Feature Importance** — gain plot + SHAP bar + beeswarm + partial dependence.
4. **What-If Simulator** — sample a real customer, perturb engagement/products/age, observe marginal probability change.
5. **Model Performance** — full metrics table, ROC/PR curves, confusion matrix, EDA snapshots.

---

### 9. Limitations and Future Work
- The dataset is a single snapshot; we cannot detect time-varying drivers (rate environment, macro shocks). A panel dataset would enable survival analysis (when, not just whether).
- Recall at the 0.5 threshold is 48%; a cost-sensitive optimisation aligned with the actual unit economics of retention vs. acquisition is required before rolling out to operations.
- Model fairness has not been audited. Geography_Germany is a strong driver; before deployment, the team should test for adverse outcomes on protected characteristics and apply re-weighting if needed (the European Central Bank — see https://www.ecb.europa.eu/home/html/index.en.html — supervises retail-banking compliance frameworks that bear on this).
- We did not address calibration. For pricing-aware retention budgets, downstream consumers will need a Platt or isotonic calibration on top of the raw probabilities.

---

### 10. Conclusion
Churn in this retail-banking portfolio is not random. Four behavioural levers — engagement, product breadth, age-stage and geography — explain most of it. The Gradient Boosting model achieves a 0.87 ROC-AUC and produces explainable, actionable scores. The strongest finding is operational rather than algorithmic: aggressive cross-sell beyond two products correlates with near-certain churn, and the Germany cohort needs a targeted intervention. With the dashboard and scoring service in place, the bank can move from reactive reporting to proactive retention.

---

*Generated artefacts: `models/best_model.pkl`, `models/metadata.json`, `reports/model_metrics.csv`, `reports/feature_importance.csv`, `reports/data_quality.json`, `reports/figures/01–12.png`, `reports/shap_values.npz`.*
