# AI-Driven CRM Optimization
## Predictive Lead Scoring and Intelligent Agent Automation Using the Salesforce Ecosystem

**M.Tech. Dissertation — Data Science & Engineering**  
**BITS Pilani, Work Integrated Learning Programmes (WILP)**  
**Student:** Thakker Ridhi Vishal  
**Supervisor:** Avinash Nawani  
**Organisation:** Tata Consultancy Services, Ahmedabad

---

## Project Overview

This dissertation presents an end-to-end data science pipeline that replaces static, rule-based CRM lead prioritisation with a machine learning-driven approach. The system predicts lead conversion probability and automates routing decisions using an agent framework grounded in Salesforce Agentforce architecture.

**Stage 1 — Predictive Lead Scoring**  
A Logistic Regression model trained on 9,240 historical CRM leads produces a conversion probability score (0–1) for each lead, achieving AUC-ROC of 0.9511 and F1 Score of 0.8573.

**Stage 2 — Intelligent Agent Automation**  
A Python-based agent simulation implements Agent Script-inspired conditional logic, routing leads to one of four actions based on their score. Evaluated using Agentforce Observability-inspired metrics including Action Hallucination Rate, Deflection Rate, and Threshold Adherence Score.

---

## Key Results

| Metric | Baseline (Rules) | ML Pipeline | Improvement |
|---|---|---|---|
| Accuracy | 58.9% | 89.1% | +30.2% |
| F1 Score | 36.9% | 85.7% | +48.8% |
| Precision | 45.2% | 86.2% | +41.0% |
| Recall | 31.2% | 85.3% | +54.1% |
| AUC-ROC | N/A | 0.9511 | — |

### Observability Metrics (Phase 6)

| Metric | Value |
|---|---|
| Deflection Rate | 60.3% |
| Threshold Adherence Score | 100.00% |
| Action Hallucination Rate | 0.00% |
| False Positive Rate (High Priority) | 8.9% |
| Precision at 0.75 Threshold | 91.1% |
| HITL Band Conversion Rate | 53.5% |
| Avg Lead Response Time | 102.7 hours |

---

## Agent Routing Logic

| Score Range | Action |
|---|---|
| > 0.75 | Route to Sales Representative |
| 0.55 – 0.75 | Trigger Automated Email Sequence |
| 0.45 – 0.55 | HITL Escalation — Human Review |
| < 0.45 | Nurture Queue + Deflect from Sales |

---

## Project Structure

```
crm-lead-scoring/
├── config.py                          # Portable path configuration
├── requirements.txt                   # Python dependencies
├── .env.example                       # Template for Salesforce credentials
├── data/
│   └── Leads.csv                      # X Education dataset (not tracked in git)
├── models/
│   ├── lead_scoring_model.pkl         # Trained Logistic Regression
│   ├── scaler.pkl                     # Fitted StandardScaler
│   └── model_columns.pkl             # Feature column schema
├── notebooks/
│   ├── 01_EDA.ipynb                  # Phase 2 & 3: EDA, cleaning, feature engineering
│   ├── 02_Models.ipynb               # Phase 4: Model training, SHAP, threshold analysis
│   └── 03_Agent_Simulation.ipynb    # Phase 5 & 6: Baseline comparison, agent simulation
├── outputs/
│   ├── 01_overview.png              # EDA overview charts
│   ├── 02_conversion_analysis.png   # Conversion by source and engagement
│   ├── 03_correlation.png           # Feature correlation heatmap
│   ├── 04_model_comparison.png      # ROC curves and model comparison
│   ├── 05_precision_recall.png      # Precision-Recall and threshold analysis
│   ├── 06_shap_importance.png       # SHAP feature importance
│   ├── 07_confusion_matrices.png    # Confusion matrices for all models
│   ├── phase6_agent_simulation.csv  # Full agent simulation results
│   └── salesforce_demo_leads.csv    # Sample scored leads for the Salesforce demo
└── streamlit-app/
    ├── app.py                        # Interactive demo application
```

---

## Setup and Installation

### Prerequisites
- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ridhi-thakker/crm-lead-scoring.git
cd crm-lead-scoring

# Install dependencies
pip install -r requirements.txt

# Download the dataset
# Get Leads.csv from: https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset
# Place it in the data/ folder

# Trained model files are already in models/. To retrain, run the notebooks in order:
# 1. notebooks/01_EDA.ipynb
# 2. notebooks/02_Models.ipynb
# 3. notebooks/03_Agent_Simulation.ipynb

# Launch the Streamlit app
cd streamlit-app
python3.11 -m streamlit run app.py
```

---

## Notebooks

### 01_EDA.ipynb — Exploratory Data Analysis
- Dataset loading and validation (9,240 leads, 37 features)
- Missing value analysis and imputation strategy
- Conversion pattern analysis by lead source, origin, and engagement
- Correlation analysis and feature importance identification
- Five engineered features: Engagement Level, High Interaction Flag, Lead Source Group, Is Webinar Lead, Engagement Score
- SMOTE class balancing (applied post-split on training set only)

### 02_Models.ipynb — Model Building and Evaluation
- Four classifiers trained: Logistic Regression, Random Forest, XGBoost, SVM
- XGBoost hyperparameter tuning via GridSearchCV
- ROC curves, Precision-Recall curves, confusion matrices
- SHAP explainability analysis — top predictors: Tags, SMS Sent, Olark Chat
- Threshold analysis derived from Precision-Recall curve
- Model validation cell proving Streamlit app produces identical scores

### 03_Agent_Simulation.ipynb — Phase 5 and Phase 6
- Baseline rule-based system comparison
- Batch agent simulation across all 1,848 test leads
- Full observability metrics: Deflection Rate, Threshold Adherence, Action Hallucination Rate
- HITL band validation (53.5% conversion rate confirms genuine uncertainty)

---

## Streamlit Application

The interactive demo combines Stage 1 scoring and Stage 2 agent routing in a single interface.

**Features:**
- Dataset Overview tab — probability distribution, tier breakdown (Hot/Warm/Cold)
- All Predictions tab — scored and filterable view of all leads with CSV export
- Single Lead tab — score any lead by entering CRM attributes
- Salesforce tab — push scored leads directly to Salesforce via OAuth 2.0

**Salesforce credentials (optional):** copy `.env.example` to `.env` and fill in your Connected App Consumer Key, Secret and My Domain URL, or type them into the app sidebar. Credentials are never stored in the code.

**Running the app:**
```bash
cd streamlit-app
python3.11 -m streamlit run app.py
```

---

## Salesforce Integration

The app integrates with Salesforce via OAuth 2.0 Client Credentials Flow:

1. Model scores the lead and determines agent action
2. Lead record pushed to Salesforce with `Lead_Score__c` populated
3. Record-Triggered Flow fires automatically on `Lead_Score__c` change
4. Flow sets `Agent_Action__c` based on dissertation threshold logic
5. Downstream actions: Task creation, Email Alerts, Lead assignment

**Custom Salesforce Fields:**
- `Lead_Score__c` — Number (3,4) — stores model probability
- `Agent_Action__c` — Picklist — stores routing decision

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| ML Models | scikit-learn 1.9.0 |
| Gradient Boosting | XGBoost 3.2.0 |
| Class Balancing | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| Web App | Streamlit |
| CRM Integration | simple-salesforce (OAuth 2.0) |
| Visualisation | matplotlib, seaborn |
| Data | pandas, numpy |

---

## Dataset

**X Education Lead Scoring Dataset**  
Source: [Kaggle](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)  
Size: 9,240 leads, 37 features  
Target: Binary conversion outcome (Converted: 0/1)

The dataset is not included in this repository due to size. Download from Kaggle and place in `data/Leads.csv`.

---

## License

This project is submitted as an academic dissertation. All rights reserved.
