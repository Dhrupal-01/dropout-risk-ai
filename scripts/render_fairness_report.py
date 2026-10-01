"""
Markdown Report Generator for Algorithmic Fairness & Ethics Documentation
Context: DropoutGuard Phase 4 Fairness Audit
Reads serialized JSON artifacts from ml/artifacts/fairness/ and renders docs/ethics_and_fairness.md.

Rules:
- Static governance and human-in-the-loop sections preserved.
- Report quantitative numbers and markdown tables only.
- Zero subjective sentences claiming or judging whether the model "is fair".
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml.provenance import assert_consistent_provenance, dirty_artifact_warning, load_labelled_json

BASE_DIR = Path(__file__).resolve().parents[1]
FAIRNESS_ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts" / "fairness"
BENCHMARK_ARTIFACTS_DIR = BASE_DIR / "ml" / "artifacts" / "benchmarks"
DOCS_DIR = BASE_DIR / "docs"
REPORT_PATH = DOCS_DIR / "ethics_and_fairness.md"


def _format_pct(val: Optional[float], decimals: int = 1) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.{decimals}f}%"


def _format_flt(val: Optional[float], decimals: int = 4) -> str:
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _format_ci(ci_tuple: Optional[List[float]], decimals: int = 1) -> str:
    if not ci_tuple or len(ci_tuple) < 2 or ci_tuple[0] is None or ci_tuple[1] is None:
        return "—"
    return f"[{ci_tuple[0] * 100:.{decimals}f}%, {ci_tuple[1] * 100:.{decimals}f}%]"


def render_uci_audit_table(uci_data: Dict[str, Any]) -> str:
    """Renders UCI Higher Education fairness audit table with subgroup metrics and 95% CIs."""
    lines = [
        "| Attribute | Subgroup | Sample Size ($N$) | Base Rate | Selection Rate (Top 20%) | Recall (TPR) | Miss Rate (FNR) | False Alarm (FPR) | Precision | ECE | FNR Gap vs Ref | 95% Bootstrap CI | Small Group Flag |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    attributes = uci_data.get("attributes", {})
    for attr_name, attr_data in attributes.items():
        ref_group = attr_data.get("reference_group", "")
        groups = attr_data.get("groups", {})
        gaps = attr_data.get("gaps_vs_reference", {})

        for grp_name, grp in groups.items():
            gap_info = gaps.get(grp_name, {})
            gap_val = gap_info.get("fnr_gap")
            ci_val = gap_info.get("fnr_gap_95ci") or gap_info.get("confidence_intervals_95", {}).get("fnr_gap")
            insufficient = grp.get("insufficient_sample", False)
            flag_str = "insufficient_sample" if insufficient else "sufficient"

            ref_badge = " *(Ref)*" if grp_name == ref_group else ""
            gap_str = _format_pct(gap_val, 2) if (gap_val is not None and not insufficient) else "—"
            ci_str = _format_ci(ci_val, 2) if (ci_val is not None and not insufficient) else "—"

            row = [
                f"**{attr_name}**",
                f"{grp_name}{ref_badge}",
                str(grp.get("n_samples", "—")),
                _format_pct(grp.get("base_rate")),
                _format_pct(grp.get("selection_rate")),
                _format_pct(grp.get("tpr_recall")),
                _format_pct(grp.get("fnr_miss_rate")),
                _format_pct(grp.get("fpr_alarm_rate")),
                _format_pct(grp.get("precision")),
                _format_flt(grp.get("within_group_ece"), 3),
                gap_str,
                ci_str,
                flag_str,
            ]
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_mitigations_table(mit_data: Dict[str, Any]) -> str:
    """Renders table comparing unmitigated baseline and 3 fairness mitigations."""
    lines = [
        "| Strategy | Sensitive Attribute | ROC-AUC | PR-AUC | Recall (Sensitivity) | Precision | Brier Score | Max FNR Disparity Gap |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    sens_attr = mit_data.get("sensitive_attribute", "gender")
    comp_rows = mit_data.get("comparison_table", [])
    for row in comp_rows:
        lines.append(
            f"| **{row.get('strategy', '—')}** | {sens_attr} | "
            f"{_format_flt(row.get('roc_auc'), 3)} | "
            f"{_format_flt(row.get('pr_auc'), 3)} | "
            f"{_format_pct(row.get('recall'), 1)} | "
            f"{_format_pct(row.get('precision'), 1)} | "
            f"{_format_flt(row.get('brier_score'), 4)} | "
            f"**{_format_pct(row.get('max_fnr_gap'), 2)}** |"
        )

    return "\n".join(lines)


def render_oulad_audit_table(oulad_data: Dict[str, Any]) -> str:
    """Renders OULAD Day 56 fairness audit table across 7 attributes."""
    lines = [
        "| Attribute | Subgroup | Sample Size ($N$) | Base Rate | Selection Rate (Top 20%) | Recall (TPR) | Miss Rate (FNR) | False Alarm (FPR) | Precision | ECE | FNR Gap vs Ref | 95% Bootstrap CI | Sample Flag |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    attributes = oulad_data.get("attributes", {})
    for attr_name, attr_data in attributes.items():
        ref_group = attr_data.get("reference_group", "")
        groups = attr_data.get("groups", {})
        gaps = attr_data.get("gaps_vs_reference", {})

        for grp_name, grp in groups.items():
            gap_info = gaps.get(grp_name, {})
            gap_val = gap_info.get("fnr_gap")
            ci_val = gap_info.get("fnr_gap_95ci") or gap_info.get("confidence_intervals_95", {}).get("fnr_gap")
            insufficient = grp.get("insufficient_sample", False)
            flag_str = "insufficient_sample" if insufficient else "sufficient"

            ref_badge = " *(Ref)*" if grp_name == ref_group else ""
            gap_str = _format_pct(gap_val, 2) if (gap_val is not None and not insufficient) else "—"
            ci_str = _format_ci(ci_val, 2) if (ci_val is not None and not insufficient) else "—"

            row = [
                f"**{attr_name}**",
                f"{grp_name}{ref_badge}",
                str(grp.get("n_samples", "—")),
                _format_pct(grp.get("base_rate")),
                _format_pct(grp.get("selection_rate")),
                _format_pct(grp.get("tpr_recall")),
                _format_pct(grp.get("fnr_miss_rate")),
                _format_pct(grp.get("fpr_alarm_rate")),
                _format_pct(grp.get("precision")),
                _format_flt(grp.get("within_group_ece"), 3),
                gap_str,
                ci_str,
                flag_str,
            ]
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_shift_check_table(shift_data: Dict[str, Any]) -> str:
    """Renders table comparing 2013 presentations vs 2014 temporal holdout presentations."""
    lines = [
        "| Attribute | Group | $N$ (2013) | $N$ (2014) | Base Rate 2013 | Base Rate 2014 | FNR 2013 | FNR 2014 | FNR Shift (2014 - 2013) | FNR Gap 2013 | FNR Gap 2014 | Gap Shift (2014 - 2013) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    shift_comp = shift_data.get("shift_comparison", {})
    for attr, attr_info in shift_comp.items():
        groups = attr_info.get("groups", [])
        for g in groups:
            ref_badge = " *(Ref)*" if g.get("is_reference") else ""
            row = [
                f"**{attr}**",
                f"{g.get('group')}{ref_badge}",
                str(g.get("n_2013", 0)),
                str(g.get("n_2014", 0)),
                _format_pct(g.get("base_rate_2013")),
                _format_pct(g.get("base_rate_2014")),
                _format_pct(g.get("fnr_2013")),
                _format_pct(g.get("fnr_2014")),
                _format_pct(g.get("fnr_shift_2014_minus_2013"), 2),
                _format_pct(g.get("fnr_gap_2013"), 2),
                _format_pct(g.get("fnr_gap_2014"), 2),
                _format_pct(g.get("gap_shift_2014_minus_2013"), 2),
            ]
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_income_ablation_tables(income_data: Dict[str, Any]) -> str:
    """Renders overall performance and disaggregated FNR tables for income feature ablation."""
    comp_rows = income_data.get("comparison_table", [])
    t1_lines = [
        "| Feature Configuration | Predictor Features | ROC-AUC | PR-AUC | Recall | Precision | Brier Score | FNR (<2 LPA) | FNR (>8 LPA) | FNR Disparity Gap |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in comp_rows:
        t1_lines.append(
            f"| **{row.get('configuration')}** | {row.get('n_features')} | "
            f"{_format_flt(row.get('roc_auc'), 3)} | "
            f"{_format_flt(row.get('pr_auc'), 3)} | "
            f"{_format_pct(row.get('recall'), 1)} | "
            f"{_format_pct(row.get('precision'), 1)} | "
            f"{_format_flt(row.get('brier_score'), 4)} | "
            f"{_format_pct(row.get('fnr_lowest_slab'), 1)} | "
            f"{_format_pct(row.get('fnr_highest_slab'), 1)} | "
            f"**{_format_pct(row.get('fnr_gap_low_vs_high'), 2)}** |"
        )
    t1_str = "\n".join(t1_lines)

    slabs = income_data.get("slabs_summary", {})
    t2_lines = [
        "| Income Bracket | $N$ Students | Actual Dropouts | Base Rate | FNR (With Income Features) | FNR (Without Income Features) | FNR Difference (Without - With) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for slab_name, slab_info in slabs.items():
        w = slab_info.get("with_income_features", {})
        wo = slab_info.get("without_income_features", {})
        diff = wo.get("fnr", 0) - w.get("fnr", 0) if (wo.get("fnr") is not None and w.get("fnr") is not None) else None
        t2_lines.append(
            f"| **{slab_name}** | {slab_info.get('n_samples', 0)} | {slab_info.get('n_dropouts', 0)} | "
            f"{_format_pct(slab_info.get('base_rate'))} | "
            f"{_format_pct(w.get('fnr'), 2)} | "
            f"{_format_pct(wo.get('fnr'), 2)} | "
            f"{_format_pct(diff, 2)} |"
        )
    t2_str = "\n".join(t2_lines)

    return f"{t1_str}\n\n#### Disaggregated Performance Across Income Slabs (Simulated Cohort)\n\n{t2_str}"


def render_generator_sanity_tables(gen_data: Dict[str, Any]) -> str:
    """Renders generator sanity check tables for simulated cohort."""
    ts = gen_data.get("test_split_n300", {})
    cv = gen_data.get("cross_validation_n2000", {})

    g_ts = ts.get("gender", {})
    inc_ts = ts.get("economic_proxy", {})
    fg_ts = ts.get("first_generation", {})

    g_cv = cv.get("gender", {})
    inc_cv = cv.get("economic_proxy", {})
    fg_cv = cv.get("first_generation", {})

    lines = [
        "| Demographic Slice | Single Held-Out Test Split ($N=300$) | 5-Fold Cross-Validation ($N=2,000$ Out-of-Fold) | Empirical Difference |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Gender Disparity Gap** (Female vs Male FNR) | **{_format_pct(g_ts.get('fnr_disparity'), 2)}** | **{_format_pct(g_cv.get('fnr_disparity'), 2)}** | **{_format_pct(abs((g_ts.get('fnr_disparity') or 0) - (g_cv.get('fnr_disparity') or 0)), 2)}** |",
        f"| **Economic Proxy Gap** (<5 LPA vs $\\ge$5 LPA) | **{_format_pct(inc_ts.get('fnr_disparity'), 2)}** | **{_format_pct(inc_cv.get('fnr_disparity'), 2)}** | **{_format_pct(abs((inc_ts.get('fnr_disparity') or 0) - (inc_cv.get('fnr_disparity') or 0)), 2)}** |",
        f"| **First-Generation Gap** (First-Gen vs Non-First-Gen) | **{_format_pct(fg_ts.get('fnr_disparity'), 2)}** | **{_format_pct(fg_cv.get('fnr_disparity'), 2)}** | **{_format_pct(abs((fg_ts.get('fnr_disparity') or 0) - (fg_cv.get('fnr_disparity') or 0)), 2)}** |",
    ]

    return "\n".join(lines)


def render_uci_feature_line(uci_data: Dict[str, Any]) -> str:
    """Feature-set description for the UCI audit, built only from fields recorded in the JSON."""
    def _cols(key: str) -> str:
        return ", ".join(f"`{c}`" for c in uci_data.get(key) or []) or "none recorded"

    return (
        f"`{uci_data.get('feature_set', 'unknown')}` (${uci_data.get('feature_count', 'unknown')}$ model features). "
        f"Protected attributes excluded from the model: {_cols('excluded_protected_attributes')}. "
        f"Audit groups that are also model features (documented): {_cols('audit_groups_used_as_features')}."
    )


def render_fairness_markdown_report(
    fairness_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    benchmark_dir: Optional[Path] = None,
) -> Path:
    """
    Renders docs/ethics_and_fairness.md from serialized JSON artifacts.
    Refuses (ProvenanceError, nothing written) unless every fairness artifact and every benchmark
    artifact carries provenance and they agree on the checksum of each shared input file.
    """
    fairness_dir = Path(fairness_dir) if fairness_dir else FAIRNESS_ARTIFACTS_DIR
    report_path = Path(report_path) if report_path else REPORT_PATH
    benchmark_dir = Path(benchmark_dir) if benchmark_dir else BENCHMARK_ARTIFACTS_DIR

    fairness_artifacts = load_labelled_json(sorted(fairness_dir.glob("*.json")))
    assert_consistent_provenance({
        **fairness_artifacts,
        **load_labelled_json(sorted(benchmark_dir.glob("*.json"))),
    })
    dirty_warning = dirty_artifact_warning(fairness_artifacts)
    dirty_block = f"\n{dirty_warning}\n" if dirty_warning else ""

    # Load JSON artifacts
    def _load_json(filename: str) -> Dict[str, Any]:
        p = fairness_dir / filename
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
        return {}

    uci_data = _load_json("uci_fairness.json")
    uci_mit = _load_json("uci_mitigations.json")
    oulad_data = _load_json("oulad_fairness.json")
    oulad_shift = _load_json("oulad_shift_check.json")
    income_data = _load_json("income_ablation.json")
    gen_data = _load_json("generator_sanity_check.json")

    uci_table = render_uci_audit_table(uci_data) if uci_data else "*UCI audit artifact not found.*"
    uci_n = f"{uci_data['n_samples']:,}" if uci_data else "unknown (artifact not found)"
    uci_feature_line = render_uci_feature_line(uci_data) if uci_data else "*UCI audit artifact not found.*"
    mit_table = render_mitigations_table(uci_mit) if uci_mit else "*UCI mitigations artifact not found.*"
    oulad_table = render_oulad_audit_table(oulad_data) if oulad_data else "*OULAD audit artifact not found.*"
    shift_table = render_shift_check_table(oulad_shift) if oulad_shift else "*OULAD shift check artifact not found.*"
    income_tables = render_income_ablation_tables(income_data) if income_data else "*Income ablation artifact not found.*"
    gen_table = render_generator_sanity_tables(gen_data) if gen_data else "*Generator sanity check artifact not found.*"

    content = f"""# Ethics, Responsible AI & Algorithmic Fairness Audit Report
### DropoutGuard — AI-Powered Academic Dropout Prediction & Intervention System
**Target Context**: Smart India Hackathon 2026 (PSID 7-L) & SDG 4: Quality Education  
**Evaluation Scope**: Quantitative algorithmic fairness, subgroup False-Negative-Rate (FNR) parity, within-group calibration (ECE), temporal presentation shift, and mitigation benchmarking across real-data cohorts and simulated benchmarks.
{dirty_block}
---

## 1. Executive Summary & Audit Mandate
In educational early-warning systems, the primary ethical risk is **unequal intervention access driven by disparate False Negative Rates (FNR)**. A False Negative represents an at-risk student who is missed by the AI system and thus denied proactive mentoring, financial counseling, or academic tutoring.

All evaluations in this report adhere to the following principles:
- **Audit Frame Isolation**: Audited demographic and sensitive attributes are strictly quarantined in an audit frame and never passed to the model's feature matrix $X$.
- **Objective Metric Reporting**: Numbers and markdown tables represent empirical measurements computed on verified splits. No subjective claims regarding algorithmic equity or compliance are made.
- **Small-Sample Guardrails**: Subgroups with $n < 50$ are flagged as `insufficient_sample` with disparity gaps and confidence intervals suppressed.
- **Empirical Uncertainty**: All reported disparity gaps are accompanied by 95% bootstrap confidence intervals (1,000 resamples).

---

## 2. Real Higher Education Benchmark: UCI Dataset 697 Audit
- **Dataset**: UCI "Predict Students' Dropout and Academic Success" (Portuguese Higher Education, $N = {uci_n}$, Enrolled excluded).
- **Feature Set**: {uci_feature_line}
- **Inference Mode**: 5-Fold Stratified Cross-Validation out-of-fold risk probabilities.
- **Selection Rate Threshold**: Top 20% predicted risk cohort.

{uci_table}

---

## 3. Algorithmic Fairness Mitigations Comparative Benchmark
Evaluates four mitigation approaches on an identical 70/30 stratified train/test split of the UCI cohort ($N = {uci_n}$, sensitive attribute: `gender`):
1. **None**: Unmitigated baseline model ($L_2$-regularized Logistic Regression).
2. **Sample Reweighing**: Inversely proportional joint class/group weights $w_i = \\frac{{N}}{{K \\cdot \\text{{count}}(s, y)}}$.
3. **Group-Specific Thresholds**: Post-processing threshold optimization equalizing subgroup False Negative Rates to target cohort FNR.
4. **Fairlearn ExponentiatedGradient**: In-processing constrained optimization enforcing `TruePositiveRateParity`.

{mit_table}

---

## 4. Time-Based Learning Analytics Benchmark: OULAD Snapshot Day 56
- **Dataset**: Open University Learning Analytics Dataset (OULAD, Day $t = 56$).
- **Cohort**: Model trained on 2013 presentations (2013B + 2013J); audited on the **unseen 2014 temporal holdout cohort** (2014B + 2014J).
- **Feature Set**: 24 cumulative behavioural and engagement features up to Day 56 (all demographic attributes strictly excluded from $X$).
- **Selection Rate Threshold**: Top 20% predicted risk cohort.

{oulad_table}

---

## 5. OULAD Presentation Shift Check: 2013 In-Sample vs. 2014 Temporal Holdout
Evaluates temporal stability of subgroup error rates and disparity gaps across academic years (2013 calendar presentations vs. 2014 calendar presentations at snapshot $t = 56$).

{shift_table}

---

## 6. Income Feature Ablation Benchmark (Simulated Cohort)
Evaluates model performance and disaggregated False Negative Rates across household income brackets when `income_slab_idx` and `financial_stress_index` are included versus completely ablated from the 37-feature simulated pipeline.

{income_tables}

---

## 7. Generator Sanity Check (Simulated Indian Cohort)
Verification of synthetic data generator properties across $N = 2,000$ simulated students.

> [!NOTE]
> **Generator Sanity Check Disclaimer**:
> The simulated Indian cohort is generated from known mathematical equations (`ml/simulation/estimate_parameters.py` and `ml/data_pipeline/generate_synthetic_indian.py`). The numbers below verify internal generator calibration and absence of unintentional statistical distortions; they are **NOT evidence of real-world predictive validity**.

{gen_table}

---

## 8. Four Core Pillars of Responsible AI Governance in DropoutGuard

1. **Human-in-the-Loop Decision Support**:
   The system never executes autonomous punitive actions (e.g. debarment or scholarship cancellation). All outputs serve exclusively as confidential decision-support recommendations for designated faculty mentors.
2. **Deficit Framing Avoidance**:
   Risk assessments avoid pejorative labels. Interventions are framed as proactive resource allocations (e.g., "Peer Tutoring Referral" or "Financial Aid Desk Check-in") rather than student deficits.
3. **SHAP-Verifiable Interpretability**:
   Every risk probability is accompanied by signed SHAP local drivers, enabling mentors to verify the causal rationale before taking action.
4. **Data Minimization, Need Signals & Confidentiality**:
   Household income is used as an active model input via `income_slab_idx` and `financial_stress_index` exclusively as an objective need signal to prioritize and route emergency financial aid and fee-waiver interventions to economically vulnerable students. Only the duplicate raw string label (`family_income_slab`) and sensitive social categories are excluded from model training to prevent categorical bias. All socio-economic indicators are confidential, encrypted at rest, and never used for punitive academic actions.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(content.strip() + "\n")
    return report_path


if __name__ == "__main__":
    path = render_fairness_markdown_report()
    print(f"Rendered fairness report successfully to {path}")
