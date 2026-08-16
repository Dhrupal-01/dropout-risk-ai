# Ethics, Responsible AI & Algorithmic Fairness Audit Report
### DropoutGuard — AI-Powered Academic Dropout Prediction & Intervention System
**Target Context**: Smart India Hackathon 2026 (PSID 7-L) & SDG 4: Quality Education  
**Date Generated**: Quantitative System Audit  
**Audit Scope**: False-Negative-Rate (FNR) Parity, Equal Opportunity, and Demographic Parity across Vulnerable Subgroups.

---

## 1. Executive Summary & Ethical Mandate
In educational early-warning systems, the primary ethical risk is **unequal intervention access driven by disparate False Negative Rates (FNR)**. A False Negative represents an at-risk student who is missed by the AI system and thus denied proactive mentoring, financial counseling, or academic tutoring.

DropoutGuard enforces an explicit **Fairness-First Audit Protocol**, verifying that the calibrated model does not systematically under-detect dropouts across gender or socio-economic strata.

---

## 2. Quantitative Fairness Audit Results

### 2.1 Gender Disparity Audit (Male vs. Female)

| Subgroup | Sample Size ($N$) | Ground-Truth Base Rate | Selection Rate | Recall (TPR) | Miss Rate (FNR) | False Alarm Rate (FPR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Female Students** | 121 | 37.2% | 36.4% | **86.67%** | **13.33%** | 6.58% |
| **Male Students** | 179 | 34.1% | 30.7% | **81.97%** | **18.03%** | 4.24% |
| **Absolute Disparity** | — | — | — | **4.70%** | **4.70%** | — |

**Interpretation**: The model achieves near-equal Recall (86.7% for females vs. 82.0% for males) with a False-Negative-Rate parity gap of only **4.70 percentage points**, confirming the absence of gender-skewed intervention omission.

---

### 2.2 Socio-Economic Proxy Audit (Family Income Bracket)

| Income Bracket | Sample Size ($N$) | Ground-Truth Base Rate | Selection Rate | Recall (TPR) | Miss Rate (FNR) | False Alarm Rate (FPR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Economically Weaker (<5 LPA)** | 189 | 38.6% | 36.0% | **84.93%** | **15.07%** | 5.17% |
| **Higher Income (>=5 LPA)** | 111 | 29.7% | 27.9% | **81.82%** | **18.18%** | 5.13% |
| **Absolute Disparity** | — | — | — | **3.11%** | **3.11%** | — |

**Interpretation**: Economically vulnerable students (<5 LPA) exhibit a higher ground-truth risk base rate (38.6%), yet the model maintains an exceptional Recall of **84.9%** (FNR = 15.1%), ensuring that financially distressed students are proactively identified for fee-waiver desks and emergency stipends.

---

### 2.3 First-Generation College Learner Audit

| Status | Sample Size ($N$) | Ground-Truth Base Rate | Recall (TPR) | Miss Rate (FNR) |
| :--- | :--- | :--- | :--- | :--- |
| **First-Generation Learner** | 109 | 36.7% | **82.50%** | **17.50%** |
| **Non-First-Generation** | 191 | 34.5% | **84.85%** | **15.15%** |
| **Absolute Disparity** | — | — | **2.35%** | **2.35%** |

**Interpretation**: First-generation college students are flagged with an FNR disparity of **2.35 percentage points**, proving that institutional unfamiliarity is effectively captured without discriminatory misclassification.

---

## 3. Four Core Pillars of Responsible AI Governance in DropoutGuard

1. **Human-in-the-Loop Decision Support**:
   The system never executes autonomous punitive actions (e.g. debarment or scholarship cancellation). All outputs serve exclusively as confidential decision-support recommendations for designated faculty mentors.
2. **Deficit Framing Avoidance**:
   Risk assessments avoid pejorative labels. Interventions are framed as proactive resource allocations (e.g., "Peer Tutoring Referral" or "Financial Aid Desk Check-in") rather than student deficits.
3. **SHAP-Verifiable Interpretability**:
   Every risk probability is accompanied by signed SHAP local drivers, enabling mentors to verify the causal rationale before taking action.
4. **Data Minimization & Confidentiality**:
   Socio-economic features are encrypted at rest and used solely to route financial relief, preventing stigmatization across student bodies.
