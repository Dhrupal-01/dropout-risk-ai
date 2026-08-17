# Ethics, Responsible AI & Algorithmic Fairness Audit Report
### DropoutGuard — AI-Powered Academic Dropout Prediction & Intervention System
**Target Context**: Smart India Hackathon 2026 (PSID 7-L) & SDG 4: Quality Education  
**Date Generated**: Quantitative System Audit & Cross-Validated Verification  
**Audit Scope**: False-Negative-Rate (FNR) Parity, Equal Opportunity, and Demographic Parity across Vulnerable Subgroups.

---

## 1. Executive Summary & Ethical Mandate
In educational early-warning systems, the primary ethical risk is **unequal intervention access driven by disparate False Negative Rates (FNR)**. A False Negative represents an at-risk student who is missed by the AI system and thus denied proactive mentoring, financial counseling, or academic tutoring.

DropoutGuard enforces an explicit **Fairness-First Audit Protocol**, verifying that the calibrated model does not systematically under-detect dropouts across gender or socio-economic strata.

> [!IMPORTANT]
> **Sample Size & Statistical Significance Note**:
> In a single held-out test split of $N = 300$ students, the total number of False Negatives is small ($N = 17$ total FNs across the entire test set). Consequently, single-split disparity numbers are subject to small-sample variance and should be interpreted as **suggestive rather than conclusive**.
> To provide greater statistical stability, we execute **5-Fold Stratified Cross-Validation ($N = 2,000$ out-of-fold predictions, $106$ total False Negatives across the full cohort)** alongside the single test split.

---

## 2. Quantitative Fairness Audit Results

### 2.1 Comparative Disparity Table: Single Split ($N=300$, $17$ FNs) vs. 5-Fold Cross-Validation ($N=2,000$, $106$ FNs)

| Protected Attribute / Subgroup Breakdown | Single Test Split ($N=300$, $17$ Total FNs) | 5-Fold Cross-Validation ($N=2,000$, $106$ Total FNs) | Empirical Trend |
| :--- | :--- | :--- | :--- |
| **Gender Disparity Gap** (Female vs. Male FNR) | **4.70 percentage points** (FNR 13.3% vs 18.0%) | **1.18 percentage points** (FNR 15.6% vs 14.4%) | **Shrinks by 3.52 pp** (Effective sample: 106 FNs) |
| **Economic Proxy Gap** (<5 LPA vs. $\ge$5 LPA) | **3.11 percentage points** (FNR 15.1% vs 18.2%) | **0.89 percentage points** (FNR 14.6% vs 15.5%) | **Shrinks by 2.22 pp** (Effective sample: 106 FNs) |
| **First-Generation Gap** (First-Gen vs. Non-First-Gen) | **2.35 percentage points** (FNR 17.5% vs 15.2%) | **1.36 percentage points** (FNR 15.7% vs 14.4%) | **Shrinks by 0.99 pp** (Effective sample: 106 FNs) |

---

### 2.2 5-Fold Stratified Cross-Validation Full Metrics ($N = 2,000$ Students, $106$ Total FNs)

#### A. Gender Parity Audit (Male vs. Female)

| Subgroup | Sample Size ($N$) | Total FNs | Ground-Truth Base Rate | Selection Rate | Recall (TPR) | Miss Rate (FNR) | False Alarm Rate (FPR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Female Students** | 845 | 47 | 35.6% | 34.7% | **84.39%** | **15.61%** | 7.17% |
| **Male Students** | 1155 | 59 | 35.4% | 35.0% | **85.57%** | **14.43%** | 7.24% |
| **Cross-Validated Disparity** | — | — | — | — | **1.18%** | **1.18%** (106 total FNs) | — |

**Interpretation**: Across the full cross-validated cohort ($106$ total False Negatives), the cross-validated FNR disparity gap is **1.18 percentage points** (Recall: 84.4% female vs. 85.6% male). This narrowing from the single-split gap (4.70 pp) is consistent with the single-split gaps being largely sampling noise, though not a formal statistical confirmation.

---

#### B. Socio-Economic Proxy Audit (Family Income Bracket)

| Income Bracket | Sample Size ($N$) | Total FNs | Ground-Truth Base Rate | Selection Rate | Recall (TPR) | Miss Rate (FNR) | False Alarm Rate (FPR) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Economically Weaker (<5 LPA)** | 1206 | 68 | 38.6% | 37.9% | **85.38%** | **14.62%** | 8.10% |
| **Higher Income (>=5 LPA)** | 794 | 38 | 30.9% | 30.2% | **84.49%** | **15.51%** | 6.01% |
| **Cross-Validated Disparity** | — | — | — | — | **0.89%** | **0.89%** (106 total FNs) | — |

**Interpretation**: Economically weaker students (<5 LPA) exhibit a higher ground-truth risk base rate (38.6%), yet the cross-validated model achieves a Recall of **85.38%**, with an FNR disparity gap of only **0.89 percentage points** across the 106-FN effective sample. This observation is consistent with stable detection across income brackets without strong subgroup disparity.

---

#### C. First-Generation College Learner Audit

| Status | Sample Size ($N$) | Total FNs | Ground-Truth Base Rate | Recall (TPR) | Miss Rate (FNR) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **First-Generation Learner** | 761 | 47 | 39.3% | **84.28%** | **15.72%** |
| **Non-First-Generation** | 1239 | 59 | 33.2% | **85.64%** | **14.36%** |
| **Cross-Validated Disparity** | — | — | — | **1.36%** | **1.36%** (106 total FNs) |

**Interpretation**: The cross-validated FNR disparity gap for first-generation learners is **1.36 percentage points** across the 106-FN effective sample, indicating that the larger single-split gap (2.35 pp) was likely influenced by small subgroup sample sizes.

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
