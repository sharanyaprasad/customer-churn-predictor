# Customer Churn Predictor

A machine learning application that predicts customer churn for a telecom company and translates model outputs into actionable business recommendations.

---

## Overview

Customer churn - when subscribers cancel their service.
It is one of the most expensive problems in any recurring revenue business. Acquiring a new customer costs 5 to 7 times more than retaining an existing one.

This project builds an end-to-end churn prediction pipeline: from raw data exploration to a deployed interactive dashboard that a non-technical analyst can use without writing a single line of code.

---

## What the App Does

- Predicts the probability of churn for any customer profile
- Classifies customers into High, Medium, or Low risk tiers
- Recommends a specific retention strategy based on risk level
- Identifies the key drivers pushing a customer toward churn
- Calculates the monthly revenue impact of a retention programme via an ROI calculator

---

## Dataset

Source: Telco Customer Churn — Kaggle

7,032 customers. 21 features covering demographics, services subscribed, contract type, and billing information.

---

## Key Findings

- Churn rate: 26.6% — roughly 1 in 4 customers left the company
- Customers on month-to-month contracts churn at significantly higher rates than those on annual contracts
- Churn is heavily concentrated in the first 12 months — early tenure is the highest risk window
- Customers paying above-average monthly charges are more likely to churn despite being higher-value

---

## Models

Two models were built and compared:

| Model | Accuracy | ROC-AUC | Recall (Churners) |
|---|---|---|---|
| Logistic Regression | 78.7% | 0.832 | 0.51 |
| Random Forest | 77.8% | 0.815 | 0.45 |

Logistic Regression was selected as the final model. It outperformed Random Forest on both ROC-AUC and recall, and offers greater interpretability — important in a consulting context where predictions must be communicated to non-technical stakeholders.

---

## Strategic Recommendations

Based on feature importance analysis, the top three drivers of churn are tenure, contract type, and monthly charges. Three recommendations follow:

**Early Intervention** — Target month-to-month customers in their first 12 months with a proactive contract upgrade offer before loyalty erodes.

**Pricing Review** — Customers paying above average monthly charges show higher churn. Review value perception for high-charge segments and introduce loyalty discounts at the 6-month mark.

**Service Bundling** — Customers without security and support add-ons churn more. Bundling these into base plans increases switching costs and perceived value at low incremental cost.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Pandas / NumPy | Data cleaning and manipulation |
| Scikit-learn | Model building and evaluation |
| Matplotlib / Seaborn | Exploratory data analysis and visualisation |
| Streamlit | Interactive dashboard and deployment |
| Pickle | Model serialisation |

---

## Project Structure
```
customer-churn-predictor/
├── app.py                          # Streamlit dashboard
├── Customer Churn.ipynb            # Full analysis notebook
├── customer_churn_model.pkl        # Trained Logistic Regression model
├── feature_columns.pkl             # Feature column order for inference
└── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
```

---

## How to Run Locally
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
streamlit run app.py
```

---

*Built as part of a consulting internship preparation project. March 2026.*
