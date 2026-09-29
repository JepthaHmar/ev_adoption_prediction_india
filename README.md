# Predicting Regions with High EV Adoption in India

**Predictive analytics for charging-infrastructure planning** — DSML Group Project, MBA (Analytics) 2025–27, IIM Kashipur

## Problem
Charging infrastructure in India is usually deployed *after* demand shows up. This project asks whether historical vehicle-registration data can flag the states and union territories most likely to reach high EV adoption, so infrastructure can be planned ahead of demand.

## Data
- **Source:** Vahan vehicle-registration data, Ministry of Road Transport & Highways (MoRTH)
- **Scale:** 418,372 monthly records, Jan 2019 – May 2024, 34 states/UTs, 1,414 registration offices, 22 fuel types
- Aggregated into a state-level modelling dataset (34 regions × 11 engineered features)
- EV share of national registrations grew from **0.68% (2019) to ~6.85% (Jan–May 2024)**

## Approach
| Task | Models |
|---|---|
| Classification (top-25% adoption in 2024) | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| Regression (2024 adoption rate) | Linear Regression, Random Forest, Gradient Boosting |
| Segmentation | K-Means (k chosen by silhouette score) + PCA for visualisation |

- Features: adoption rates 2021–23, 3-year average share, share change 2022→23, EV and total registration growth, EV CAGR, office counts
- Scaling inside scikit-learn `Pipeline`s to avoid data leakage
- 5-fold stratified cross-validation, ROC-AUC as primary metric

## Key Results
- Classifiers reached **ROC-AUC of 1.00** under cross-validation (Random Forest, Logistic Regression, Gradient Boosting), reflecting a cleanly separable problem on a small sample
- Random Forest regression explained **~67% of variance** in the 2024 adoption rate (RMSE ≈ 2.8 pp)
- **Three adoption-level features carried ~95% of feature importance**; the 2023 rate alone ~54%
- A consensus rule (both Logistic Regression and Random Forest ≥ 0.50) flagged **8 priority regions**: Goa, Karnataka, Chandigarh, Tripura, Assam, Delhi, Uttar Pradesh, Kerala. Maharashtra was a borderline disagreement case
- K-Means identified 5 profiles, separating **established leaders** from **fast climbers**, a distinction the classifiers can't make

## Limitations
Small modelling sample (n = 34), correlated features, and registration history acting as a proxy for unobserved drivers (incentives, charging availability, income). See the full report for a detailed discussion.

## Repository Structure
```
├── notebooks/ev_adoption_prediction.ipynb   # full analysis
├── data/vahan-registrations.zip             # raw Vahan CSV (zipped)
├── reports/project_report.docx              # detailed project report
├── reports/presentation.pdf                 # presentation deck
└── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
unzip data/vahan-registrations.zip -d data/
jupyter notebook notebooks/ev_adoption_prediction.ipynb
```

## Tech Stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter

## Team
Abhijay Kumar Jayant · Anjali · Bhavika Garua · Jeptha H Lalawmpuia · Nandini Talwar · Vaidehee Ghorpade · Shruthi G Bhosle
