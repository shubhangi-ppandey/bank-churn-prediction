# Executive Summary — Predictive Churn Risk for Retail Banking
**Audience:** Government regulators, board, retail leadership.
**Scope:** 10,000-customer European retail-banking portfolio, snapshot 2025.

---

## 1. The headline number
The bank loses **20.4% of its customers** to churn in this snapshot. A predictive model trained on this portfolio identifies these customers in advance with **87% ROC-AUC discrimination**, enabling proactive retention rather than reactive reporting.

## 2. Three findings that should change behaviour

### Finding 1 — The "Three Products" Trap
| Products held | Customers | Churn rate |
|---|---|---|
| 1 | 5,084 | 27.7% |
| 2 | 4,590 | **7.6%** ← sweet spot |
| 3 | 266 | 82.7% |
| 4 | 60 | **100%** |

The conventional wisdom that "more products means stickier customers" breaks beyond two products. Customers cross-sold a third or fourth product churn almost universally, suggesting the cross-sell engine is pushing wrong-fit products on customers who then disengage entirely. **Action:** cap automated cross-sell at two products and gate further offers behind a satisfaction check.

### Finding 2 — Germany is an outlier
Germany hosts 25% of the customer base but generates **40% of all churn** (32.4% churn rate vs. ~16% in France/Spain). This is not explained by demographics or balance levels in the model. **Action:** commission a country-specific diagnostic — pricing, language quality, branch coverage, product fit.

### Finding 3 — Engagement is the cheapest lever
Inactive members churn at **26.9%**; active members at **14.3%**. "Active" status is far easier and cheaper to influence than age, geography or wealth. **Action:** targeted re-engagement campaigns (login nudges, RM outreach, personalised offers) on the inactive cohort offer the highest expected ROI.

## 3. Risk profile of the portfolio
Using the deployed model:
- **Low risk (<30% probability):** the majority — standard servicing.
- **Medium risk (30–60%):** monitored cohort; engagement nudges + cross-sell to two products.
- **High risk (≥60%):** immediate retention outreach; relationship manager call, fee or rate adjustment.

The dashboard (`Streamlit`) ranks all customers and supports drill-down by Geography, Age band, and product mix.

## 4. Governance & compliance
- **Explainability:** every prediction is decomposed into SHAP feature contributions, satisfying the ECB's expectation of model interpretability.
- **Fairness:** Geography is a strong driver. Before deployment, an adverse-impact audit on protected attributes is required.
- **Refresh cadence:** monthly scoring; quarterly retraining; annual independent validation.
- **Decision rights:** the score is advisory. Final retention spend and threshold tuning sit with the Retention Committee.

## 5. Expected impact
If the bank acts on the high-risk cohort with a 50%-effective retention campaign, the model would prevent ~25% of the 2,037 expected annual churners, equivalent to ~500 retained customer relationships per year — material for both lifetime-value and revenue-stability KPIs.

## 6. Next steps
1. Stand up the scoring service and dashboard for the Retention team (already delivered).
2. Pilot the activation campaign on inactive members; measure with a holdout control group.
3. Commission the Germany diagnostic.
4. Recalibrate the cross-sell engine to enforce the two-product guardrail.
5. Quarterly model review reporting to the Risk Committee.

---

*Backing artefacts: research paper, model metrics, SHAP plots, partial dependence plots, and live Streamlit dashboard.*
