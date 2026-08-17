# DropoutGuard — AI-Powered Academic Dropout Prediction & Intervention System

> **Smart India Hackathon 2026 (PSID 7-L)**  
> **UN Sustainable Development Goal 4 (SDG 4: Quality Education)**  
> Production-grade predictive intelligence, explainable SHAP attributions, prescriptive institutional interventions, and algorithmic fairness for higher education.

---

## Quick Links & Documentation

- [Detailed ML Architecture, Feature Selection & Backend Handover Spec](file:///Users/dhrupal/Documents/SIH%202026/dropout-risk-ai/docs/ml_architecture_and_pipeline.md)
- [Ethics, Responsible AI & Algorithmic Fairness Audit Report](file:///Users/dhrupal/Documents/SIH%202026/dropout-risk-ai/docs/ethics_and_fairness.md)
- [Implementation Plan & Phase Roadmap](file:///Users/dhrupal/Documents/SIH%202026/dropout-risk-ai/dropout-prediction-implementation-plan.md)

---

## System Overview

DropoutGuard continuously ingests multi-source student data across **4 Core Pillars**:
1. **Attendance & Discipline**: Overall 3-month attendance %, recent-month trajectory, consecutive absence streaks, and the mandatory 75% AICTE/UGC debarment rule.
2. **Academic Performance**: Current & previous semester CGPA, semester grade velocity, uncleared active backlogs, continuous assessment marks, and core course failure status.
3. **Digital Learning Behavior (LMS)**: Weekly login frequency, assignment submission delays/lags, inactivity recency, and resource access volumes.
4. **Socio-Economic & Financial Resilience**: Tuition fee payment overdue days, family income brackets, scholarship buffers, and first-generation learner status.

```
                  ┌────────────────────────────────────────────────────────┐
                  │             DropoutGuard ML Core Pipeline             │
                  └────────────────────────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [ Predictive Intelligence ]                       [ Prescriptive Recourse ]
  • Calibrated Probabilities (Platt / Sigmoid)       • 12 Codified Institutional Actions
  • Configurable Risk Tiers (Low/Med/High)           • Counterfactual Recourse Simulation
  • TreeSHAP Signed Local Attributions               • Prioritized Mentor Triage Queue
  • Counselor Plain-Language Statements              • Intervention Lifecycle Tracking
```

---

## Key Performance Highlights (Held-Out Test Set $N=300$)

- **At-Risk Recall (Sensitivity)**: `83.96%` (Minimizes missed vulnerable students)
- **At-Risk Precision**: `89.90%` (Prevents mentor alert fatigue)
- **Minority Class F1 Score**: `0.8683`
- **Macro-Averaged F1 Score**: `0.9000`
- **ROC-AUC**: `0.9752`
- **Overall Accuracy**: `91.00%`
- **Brier Calibration Score**: `0.0689`
- **Demographic Disparity**: Gender FNR gap $4.70\text{ pp}$, Economic proxy gap $3.11\text{ pp}$, First-Gen gap $2.35\text{ pp}$

---

## Quickstart & Local Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-org/dropout-risk-ai.git
cd dropout-risk-ai

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# (or pip install numpy pandas scikit-learn xgboost shap imbalanced-learn joblib pytest ucimlrepo requests)
```

### 2. Run Test Suites
```bash
# Run all 27 automated unit and integration tests
pytest ml/tests/ -v
```

### 3. Run End-to-End Clinical Validation Report
```bash
# Run complete validation, decile calibration check, fairness audit, and 6 student report cards
python -m ml.validate_pipeline

# To force a clean re-generation and model retraining:
python -m ml.validate_pipeline --regenerate
```

---

## Repository Structure

```
dropout-risk-ai/
├── data/
│   ├── raw/                       # Raw benchmark datasets (UCI, OULAD)
│   └── processed/                 # Standardized features.csv & feature_metadata.json
├── docs/
│   ├── ml_architecture_and_pipeline.md  # Comprehensive ML & Backend handover guide
│   └── ethics_and_fairness.md           # Quantitative fairness audit report
├── ml/
│   ├── artifacts/                 # Serialized model, explainer, and intervention JSONs
│   │   ├── calibrated_model.joblib
│   │   ├── base_xgboost_model.joblib
│   │   ├── shap_explainer.joblib
│   │   ├── feature_names.json
│   │   └── interventions.json
│   ├── config.py                  # Central configuration & configurable risk thresholds
│   ├── data_pipeline/             # Ingestion & feature engineering scripts
│   ├── models/                    # Training, calibration, SHAP & fairness audit
│   ├── intervention/              # Prescriptive recommendation & counterfactual engine
│   ├── tests/                     # 27 comprehensive automated test cases
│   └── validate_pipeline.py       # End-to-end report generator and sanity assertions
├── pytest.ini
├── pyrightconfig.json
├── pyproject.toml
└── README.md
```

---

## License & Compliance
Built under the **MIT License** for **Smart India Hackathon 2026**. Designed in compliance with **UN SDG 4** and responsible AI fairness principles.