"""
Render Benchmark Report Script
Reads benchmark JSON artifacts from ml/artifacts/benchmarks/
Generates:
1. docs/benchmarks.md (tables with point estimates and 95% bootstrap CIs only, no narrative interpretation)
2. docs/figures/uci_reliability_curves.png (UCI reliability diagram)
3. docs/figures/earliness_curve.png (OULAD earliness curve: PR-AUC vs t)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from ml.fairness.attributes import PROTECTED
from ml.provenance import assert_consistent_provenance

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = BASE_DIR / "ml" / "artifacts" / "benchmarks"
DOCS_DIR = BASE_DIR / "docs"
REPORT_MD_PATH = DOCS_DIR / "benchmarks.md"
FIGURES_DIR = DOCS_DIR / "figures"
RELIABILITY_IMG_PATH = FIGURES_DIR / "uci_reliability_curves.png"
EARLINESS_IMG_PATH = FIGURES_DIR / "earliness_curve.png"

FEATURE_SET_LABELS: Dict[str, str] = {
    "enrolment_time": "ENROLMENT_TIME (Admission / Baseline)",
    "end_of_sem1": "END_OF_SEM1 (First Semester Completed)",
    "full": "FULL (not early warning)",
    "snapshot_t14": "Snapshot t=14 (2 weeks)",
    "snapshot_t28": "Snapshot t=28 (4 weeks)",
    "snapshot_t56": "Snapshot t=56 (8 weeks)",
    "snapshot_t84": "Snapshot t=84 (12 weeks)",
}

MODEL_NAMES: Dict[str, str] = {
    "majority_class": "Majority Class (Prior)",
    "logistic_regression": "Logistic Regression",
    "xgboost": "XGBoost",
    "pytorch_gru": "PyTorch GRU",
}

STRATEGY_NAMES: Dict[str, str] = {
    "predefined_split": "Temporal Holdout (Train 2013, Test 2014)",
    "repeated_stratified_cv": "Repeated Stratified 5-Fold CV (3 repeats)",
    "leave_one_group_out": "Leave-One-Course/Module-Out (LOGO)",
}


def format_stat(stat: Optional[Dict[str, float]]) -> str:
    """Format point estimate with 95% bootstrap confidence interval."""
    if not stat:
        return "N/A"
    pt = stat.get("point", 0.0)
    lo = stat.get("ci_lower", 0.0)
    hi = stat.get("ci_upper", 0.0)
    return f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"


FOLD_METRICS: List[Tuple[str, str]] = [
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("precision_top_10", "Top 10% Prec"),
    ("recall_top_10", "Top 10% Recall"),
    ("precision_top_20", "Top 20% Prec"),
    ("recall_top_20", "Top 20% Recall"),
    ("brier_score", "Brier Score"),
    ("expected_calibration_error", "ECE"),
]


def format_fold_summary(summary: Optional[Dict[str, Any]]) -> str:
    """Per-fold mean ± SD with the number of folds where the metric is defined (k = defined/total)."""
    if not summary or summary.get("mean") is None:
        return "N/A"
    sd = summary.get("sd")
    sd_str = f"{sd:.4f}" if sd is not None else "n/a"
    return f"{summary['mean']:.4f} ± {sd_str} (k={summary['n_folds_defined']}/{summary['n_folds_total']})"


def _logo_per_fold(m_data: Dict[str, Any], artifact_label: str) -> Dict[str, Any]:
    """The per-fold LOGO summary; refuses artifacts produced before per-fold metrics existed."""
    if "per_fold" not in m_data:
        raise ValueError(
            f"LOGO result in {artifact_label} has no per-fold metrics; rerun the benchmark "
            "(pooled metrics are never shown as the primary LOGO result)."
        )
    return m_data["per_fold"]["summary"]


def _logo_tables(rows: List[Tuple[str, str, Dict[str, Any], str]], unit: str) -> List[str]:
    """
    Primary per-fold mean ± SD table and secondary pooled table for leave-one-group-out.
    rows: (row label, model label, model result dict, artifact label).
    """
    head = " | ".join(name for _, name in FOLD_METRICS)
    sep = " | ".join(":---" for _ in FOLD_METRICS)
    out = [
        f"#### Leave-One-{unit}-Out: per-fold mean ± SD (primary)",
        "",
        f"Each held-out {unit.lower()} is scored on its own; values are mean ± SD across folds. "
        "k = folds where the metric is defined (single-class folds have no ROC-AUC/PR-AUC).",
        "",
        f"| Setting | Model | {head} |",
        f"| :--- | :--- | {sep} |",
    ]
    for row_label, m_label, m_data, art_label in rows:
        summary = _logo_per_fold(m_data, art_label)
        cells = " | ".join(format_fold_summary(summary.get(key)) for key, _ in FOLD_METRICS)
        out.append(f"| {row_label} | {m_label} | {cells} |")
    out.extend([
        "",
        f"#### Leave-One-{unit}-Out: pooled out-of-fold (secondary)",
        "",
        f"Out-of-fold predictions from all held-out {unit.lower()}s pooled before scoring, which compares "
        f"scores across {unit.lower()}s. Shown for reference only; not the primary LOGO estimate.",
        "",
        f"| Setting | Model | " + " | ".join(f"{name} (95% CI)" for _, name in FOLD_METRICS) + " |",
        f"| :--- | :--- | {sep} |",
    ])
    for row_label, m_label, m_data, _ in rows:
        metrics = m_data.get("metrics", {})
        cells = " | ".join(format_stat(metrics.get(key)) for key, _ in FOLD_METRICS)
        out.append(f"| {row_label} | {m_label} | {cells} |")
    out.append("")
    return out


def generate_reliability_figure(artifacts: List[Dict[str, Any]], target_path: Path):
    """
    Renders clean, publication-grade reliability diagrams using Pillow.
    Plots mean predicted probability vs observed fraction of positives across decile bins.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    img_width, img_height = 900, 600
    margin_left, margin_right = 100, 220
    margin_top, margin_bottom = 60, 80

    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    def to_pixel(x: float, y: float) -> Tuple[int, int]:
        px = int(margin_left + x * plot_w)
        py = int(margin_top + (1.0 - y) * plot_h)
        return px, py

    # Grid
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        p1 = to_pixel(tick, 0.0)
        p2 = to_pixel(tick, 1.0)
        draw.line([p1, p2], fill=(235, 235, 235), width=1)
        p3 = to_pixel(0.0, tick)
        p4 = to_pixel(1.0, tick)
        draw.line([p3, p4], fill=(235, 235, 235), width=1)
        draw.text((p1[0] - 10, p1[1] + 8), f"{tick:.1f}", fill=(80, 80, 80))
        draw.text((margin_left - 35, p3[1] - 6), f"{tick:.1f}", fill=(80, 80, 80))

    # Axes
    tl = to_pixel(0.0, 1.0)
    br = to_pixel(1.0, 0.0)
    draw.rectangle([tl, br], outline=(150, 150, 150), width=2)

    # Diagonal
    p_diag_0 = to_pixel(0.0, 0.0)
    p_diag_1 = to_pixel(1.0, 1.0)
    dash_len = 10
    total_len = ((p_diag_1[0] - p_diag_0[0]) ** 2 + (p_diag_0[1] - p_diag_1[1]) ** 2) ** 0.5
    steps = int(total_len / (dash_len * 2))
    for s in range(steps):
        t0 = (s * 2 * dash_len) / total_len
        t1 = ((s * 2 + 1) * dash_len) / total_len
        x0, y0 = p_diag_0[0] + t0 * (p_diag_1[0] - p_diag_0[0]), p_diag_0[1] + t0 * (p_diag_1[1] - p_diag_0[1])
        x1, y1 = p_diag_0[0] + t1 * (p_diag_1[0] - p_diag_0[0]), p_diag_0[1] + t1 * (p_diag_1[1] - p_diag_0[1])
        draw.line([(int(x0), int(y0)), (int(x1), int(y1))], fill=(160, 160, 160), width=2)

    colors = {
        "xgboost": (33, 115, 204),
        "logistic_regression": (218, 68, 83),
        "majority_class": (120, 120, 120),
    }

    legend_x = img_width - margin_right + 25
    legend_y = margin_top + 10

    draw.line([(legend_x, legend_y + 8), (legend_x + 30, legend_y + 8)], fill=(160, 160, 160), width=2)
    draw.text((legend_x + 38, legend_y), "Perfect Calibration", fill=(40, 40, 40))
    legend_y += 30

    primary_artifacts = [a for a in artifacts if a.get("label_variant") == "primary"]
    rep_art = next((a for a in primary_artifacts if a.get("feature_set") == "end_of_sem1"), None)
    if not rep_art and primary_artifacts:
        rep_art = primary_artifacts[0]

    if rep_art:
        strat_results = rep_art.get("results", {}).get("repeated_stratified_cv", {})
        for model_key in ["xgboost", "logistic_regression", "majority_class"]:
            model_info = strat_results.get(model_key, {})
            cal_bins = model_info.get("calibration_bins", [])
            color = colors.get(model_key, (0, 0, 0))

            valid_points = [
                to_pixel(b["mean_pred"], b["true_rate"])
                for b in cal_bins
                if b["count"] > 0
            ]

            if len(valid_points) >= 2:
                draw.line(valid_points, fill=color, width=3)
            for pt in valid_points:
                draw.ellipse([pt[0] - 4, pt[1] - 4, pt[0] + 4, pt[1] + 4], fill=color, outline=(255, 255, 255))

            disp_name = MODEL_NAMES.get(model_key, model_key)
            draw.line([(legend_x, legend_y + 8), (legend_x + 30, legend_y + 8)], fill=color, width=3)
            draw.ellipse([legend_x + 15 - 3, legend_y + 8 - 3, legend_x + 15 + 3, legend_y + 8 + 3], fill=color)
            draw.text((legend_x + 38, legend_y), disp_name, fill=(40, 40, 40))
            legend_y += 30

    draw.text((margin_left, 20), "UCI Benchmark Reliability Diagram (END_OF_SEM1, Primary Cohort)", fill=(20, 20, 20))
    draw.text((margin_left + plot_w // 2 - 80, img_height - 35), "Mean Predicted Probability", fill=(50, 50, 50))
    img.save(target_path, format="PNG")
    logger.info("Saved reliability curve figure to %s", target_path)


def generate_earliness_figure(artifacts: List[Dict[str, Any]], target_path: Path):
    """
    Renders clean, publication-grade earliness curves (PR-AUC with 95% bootstrap CI vs t).
    Plots trajectories for Majority Class, Logistic Regression, XGBoost, and PyTorch GRU.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    img_width, img_height = 900, 600
    margin_left, margin_right = 100, 220
    margin_top, margin_bottom = 60, 80

    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom

    img = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Days mapping: x from 0 to 90
    min_x, max_x = 0.0, 90.0
    min_y, max_y = 0.0, 1.0

    def to_pixel(x: float, y: float) -> Tuple[int, int]:
        px = int(margin_left + ((x - min_x) / (max_x - min_x)) * plot_w)
        py = int(margin_top + (1.0 - ((y - min_y) / (max_y - min_y))) * plot_h)
        return px, py

    # Draw grid lines & ticks
    for tick_x in [14, 28, 56, 84]:
        p1 = to_pixel(tick_x, min_y)
        p2 = to_pixel(tick_x, max_y)
        draw.line([p1, p2], fill=(235, 235, 235), width=1)
        draw.text((p1[0] - 12, p1[1] + 8), f"t={tick_x}", fill=(80, 80, 80))

    for tick_y in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        p3 = to_pixel(min_x, tick_y)
        p4 = to_pixel(max_x, tick_y)
        draw.line([p3, p4], fill=(235, 235, 235), width=1)
        draw.text((margin_left - 35, p3[1] - 6), f"{tick_y:.1f}", fill=(80, 80, 80))

    # Border
    tl = to_pixel(min_x, max_y)
    br = to_pixel(max_x, min_y)
    draw.rectangle([tl, br], outline=(150, 150, 150), width=2)

    colors = {
        "xgboost": (33, 115, 204),           # Blue
        "logistic_regression": (218, 68, 83),  # Red
        "pytorch_gru": (46, 139, 87),         # Sea Green
        "majority_class": (120, 120, 120),    # Gray
    }

    # Filter OULAD artifacts
    oulad_arts = [a for a in artifacts if "oulad" in a.get("dataset", "").lower()]
    # Sort by snapshot t
    def get_t(art: Dict[str, Any]) -> int:
        fs = art.get("feature_set", "")
        if "t" in fs:
            try:
                return int(fs.split("t")[-1])
            except Exception:
                return 0
        return 0

    sorted_arts = sorted(oulad_arts, key=get_t)
    t_values = [get_t(a) for a in sorted_arts if get_t(a) > 0]

    legend_x = img_width - margin_right + 25
    legend_y = margin_top + 10

    for model_key in ["xgboost", "pytorch_gru", "logistic_regression", "majority_class"]:
        pts: List[Tuple[int, int]] = []
        color = colors.get(model_key, (0, 0, 0))
        disp_name = MODEL_NAMES.get(model_key, model_key)

        for art in sorted_arts:
            t_val = get_t(art)
            if t_val <= 0:
                continue
            strat_res = art.get("results", {}).get("predefined_split", {})
            m_res = strat_res.get(model_key, {})
            metrics = m_res.get("metrics", {})
            pr_stat = metrics.get("pr_auc")
            if not pr_stat:
                continue

            pt_val = pr_stat.get("point", 0.0)
            ci_lo = pr_stat.get("ci_lower", pt_val)
            ci_hi = pr_stat.get("ci_upper", pt_val)

            px, py = to_pixel(t_val, pt_val)
            pts.append((px, py))

            # Draw CI vertical bar
            _, py_lo = to_pixel(t_val, ci_lo)
            _, py_hi = to_pixel(t_val, ci_hi)
            draw.line([(px, py_lo), (px, py_hi)], fill=color, width=1)
            # CI caps
            draw.line([(px - 4, py_lo), (px + 4, py_lo)], fill=color, width=1)
            draw.line([(px - 4, py_hi), (px + 4, py_hi)], fill=color, width=1)
            # Point marker
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=color, outline=(255, 255, 255))

        if len(pts) >= 2:
            draw.line(pts, fill=color, width=3)

        # Legend entry
        draw.line([(legend_x, legend_y + 8), (legend_x + 30, legend_y + 8)], fill=color, width=3)
        draw.ellipse([legend_x + 15 - 3, legend_y + 8 - 3, legend_x + 15 + 3, legend_y + 8 + 3], fill=color)
        draw.text((legend_x + 38, legend_y), disp_name, fill=(40, 40, 40))
        legend_y += 30

    draw.text((margin_left, 20), "OULAD Earliness Curves (PR-AUC vs Days from Course Start)", fill=(20, 20, 20))
    draw.text((margin_left + plot_w // 2 - 80, img_height - 35), "Snapshot Day t (Days)", fill=(50, 50, 50))
    img.save(target_path, format="PNG")
    logger.info("Saved earliness curve figure to %s", target_path)


def _single_value(artifacts: List[Dict[str, Any]], key: str, **filters: Any) -> Any:
    """The one value of `key` shared by all artifacts matching `filters`; raises if missing or inconsistent."""
    matching = [a for a in artifacts if all(a.get(k) == v for k, v in filters.items())]
    values = {a.get(key) for a in matching}
    if len(values) != 1 or None in values:
        raise ValueError(f"Expected one recorded '{key}' for artifacts {filters}, found {values}")
    return values.pop()


def _cohort_line(art: Dict[str, Any], pos_label: str, neg_label: str) -> str:
    """Class counts and prevalence for one UCI label variant, from its JSON."""
    n, pos = art["n_samples"], art["total_positives"]
    return f"{pos:,} {pos_label}, {n - pos:,} {neg_label}; Prevalence = {art['prevalence'] * 100:.2f}%"


def _uci_feature_count(artifacts: List[Dict[str, Any]], feature_set: str) -> str:
    """Feature count recorded in the benchmark JSON for a UCI feature set."""
    counts = {a.get("n_features") for a in artifacts if a.get("feature_set") == feature_set}
    if len(counts) != 1:
        raise ValueError(f"Expected one n_features for UCI feature set '{feature_set}', found {counts}")
    return str(counts.pop())


def _uci_protected_line(artifacts: List[Dict[str, Any]]) -> str:
    """States whether any PROTECTED attribute appears in the feature_names recorded in the JSON."""
    recorded = {c for a in artifacts for c in a.get("feature_names", [])}
    leaked = sorted(recorded & set(PROTECTED["uci"]))
    protected = ", ".join(f"`{c}`" for c in PROTECTED["uci"])
    if leaked:
        return (f"- **WARNING**: these artifacts were computed with PROTECTED attributes as features "
                f"({', '.join(f'`{c}`' for c in leaked)}). Rerun `python -m ml.evaluation.run --source uci`.")
    return f"- **Protected attributes** ({protected}) are not model features in any set (see `ml/fairness/attributes.py`)."


def render_benchmark_report():
    """
    Scans ml/artifacts/benchmarks/ and builds docs/benchmarks.md.
    """
    if not BENCHMARK_DIR.exists():
        raise FileNotFoundError(f"Benchmark directory not found at {BENCHMARK_DIR}")

    json_files = sorted(BENCHMARK_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No benchmark JSON files found in {BENCHMARK_DIR}")

    labelled: Dict[str, Dict[str, Any]] = {}
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            labelled[jf.name] = json.load(f)

    # Refuse before writing any figure or doc: every artifact must carry provenance and agree on inputs
    assert_consistent_provenance(labelled)
    artifacts = list(labelled.values())

    uci_artifacts = [a for a in artifacts if "uci" in a.get("dataset", "").lower()]
    oulad_artifacts = [a for a in artifacts if "oulad" in a.get("dataset", "").lower()]

    if uci_artifacts:
        generate_reliability_figure(uci_artifacts, RELIABILITY_IMG_PATH)
    if oulad_artifacts:
        generate_earliness_figure(oulad_artifacts, EARLINESS_IMG_PATH)

    n_boot = _single_value([a for a in artifacts if "dataset" in a], "n_bootstraps")
    lines: List[str] = [
        "# Real-Data Benchmark Evaluation Report",
        "",
        "> Rigorous, leak-free empirical evaluation on official educational benchmarks: UCI ID 697 and OULAD (UCI ID 349).",
        "> Conducted via the standardized evaluation harness across temporal holdouts, Leave-One-Course/Module-Out, and repeated cross-validation.",
        f"> Point estimates and 95% bootstrap confidence intervals ({n_boot:,} resamples of out-of-fold predictions).",
        "",
        "---",
        "",
    ]

    # SECTION 1: UCI BENCHMARK
    if uci_artifacts:
        uci_n_all = _single_value(uci_artifacts, "n_samples", label_variant="sensitivity")
        uci_n_primary = _single_value(uci_artifacts, "n_samples", label_variant="primary")
        uci_n_courses = _single_value(uci_artifacts, "n_groups")
        uci_primary = next(a for a in uci_artifacts if a.get("label_variant") == "primary")
        uci_sensitivity = next(a for a in uci_artifacts if a.get("label_variant") == "sensitivity")
        lines.extend([
            "## 1. Portuguese Higher Education Benchmark (UCI ID 697)",
            "",
            f"- **Dataset**: UCI ID 697 ($N = {uci_n_all:,}$, {uci_n_courses} Degree Programs / Courses)",
            "- **Feature Sets**:",
            f"  - `ENROLMENT_TIME`: Baseline features available at matriculation (admission credentials, family background, socio-economic signals, macroeconomic indicators; {_uci_feature_count(uci_artifacts, 'enrolment_time')} features).",
            f"  - `END_OF_SEM1`: `ENROLMENT_TIME` + 1st-semester curricular unit evaluations and approved units ({_uci_feature_count(uci_artifacts, 'end_of_sem1')} features). Zero 2nd-semester features.",
            f"  - `FULL`: Everything, including 2nd-semester curricular units ({_uci_feature_count(uci_artifacts, 'full')} features). Explicitly labeled as **not early warning**.",
            _uci_protected_line(uci_artifacts),
            "- **Label Variants**:",
            f"  - `Primary`: Binary outcome ($N = {uci_n_primary:,}$). `Dropout = 1` vs `Graduate = 0`. Students with `Target == 'Enrolled'` are excluded.",
            f"  - `Sensitivity`: Binary outcome ($N = {uci_n_all:,}$). `Dropout = 1` vs `Graduate / Enrolled = 0`. Students with `Target == 'Enrolled'` are coded as $0$.",
            "",
            "### Reliability & Probability Calibration",
            "",
            "![UCI Benchmark Reliability Diagram](figures/uci_reliability_curves.png)",
            "",
            "---",
            "",
        ])

        for label_var, title, desc in [
            ("primary", f"UCI Primary Cohort Evaluation (Dropout vs Graduate, N = {uci_n_primary:,})", f"Excludes active Enrolled students ({_cohort_line(uci_primary, 'Dropouts', 'Graduates')})."),
            ("sensitivity", f"UCI Sensitivity Cohort Evaluation (Enrolled as Negative, N = {uci_n_all:,})", f"Treats Enrolled students as non-dropouts ({_cohort_line(uci_sensitivity, 'Dropouts', 'Non-Dropouts')})."),
        ]:
            lines.append(f"### {title}")
            lines.append("")
            lines.append(desc)
            lines.append("")

            matching = [a for a in uci_artifacts if a.get("label_variant") == label_var]
            if not matching:
                continue

            for strat_key in ["repeated_stratified_cv"]:
                strat_name = STRATEGY_NAMES.get(strat_key, strat_key)
                lines.append(f"#### {strat_name}")
                lines.append("")
                lines.append("| Feature Set | Model | ROC-AUC (95% CI) | PR-AUC (95% CI) | Top 10% Prec (95% CI) | Top 10% Recall (95% CI) | Top 20% Prec (95% CI) | Top 20% Recall (95% CI) | Brier Score (95% CI) | ECE (95% CI) |")
                lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

                for fs_key in ["enrolment_time", "end_of_sem1", "full"]:
                    art = next((a for a in matching if a.get("feature_set") == fs_key), None)
                    if not art:
                        continue
                    fs_label = FEATURE_SET_LABELS.get(fs_key, fs_key)
                    strat_res = art.get("results", {}).get(strat_key, {})

                    for m_key in ["majority_class", "logistic_regression", "xgboost"]:
                        m_data = strat_res.get(m_key, {})
                        metrics = m_data.get("metrics", {})
                        m_label = MODEL_NAMES.get(m_key, m_key)
                        roc = format_stat(metrics.get("roc_auc"))
                        pr = format_stat(metrics.get("pr_auc"))
                        p10 = format_stat(metrics.get("precision_top_10"))
                        r10 = format_stat(metrics.get("recall_top_10"))
                        p20 = format_stat(metrics.get("precision_top_20"))
                        r20 = format_stat(metrics.get("recall_top_20"))
                        brier = format_stat(metrics.get("brier_score"))
                        ece = format_stat(metrics.get("expected_calibration_error"))
                        lines.append(f"| {fs_label} | {m_label} | {roc} | {pr} | {p10} | {r10} | {p20} | {r20} | {brier} | {ece} |")

                lines.append("")

            logo_rows = []
            for fs_key in ["enrolment_time", "end_of_sem1", "full"]:
                art = next((a for a in matching if a.get("feature_set") == fs_key), None)
                if not art:
                    continue
                logo_res = art.get("results", {}).get("leave_one_group_out", {})
                for m_key in ["majority_class", "logistic_regression", "xgboost"]:
                    if m_key in logo_res:
                        logo_rows.append((FEATURE_SET_LABELS.get(fs_key, fs_key), MODEL_NAMES.get(m_key, m_key),
                                          logo_res[m_key], f"uci {fs_key}/{label_var}"))
            if logo_rows:
                lines.extend(_logo_tables(logo_rows, "Course"))
            lines.append("---")
            lines.append("")

    # SECTION 2: OULAD TIME-BASED EARLY WARNING BENCHMARK
    if oulad_artifacts:
        oulad_n_registrations = _single_value(oulad_artifacts, "n_registrations_total")
        oulad_n_modules = _single_value(oulad_artifacts, "n_groups")
        oulad_horizons = ", ".join(str(a["snapshot_t"]) for a in sorted(oulad_artifacts, key=lambda a: a["snapshot_t"]))
        lines.extend([
            "## 2. Open University Learning Analytics Benchmark (OULAD / UCI ID 349)",
            "",
            f"- **Dataset**: OULAD ($N = {oulad_n_registrations:,}$ student registrations across {oulad_n_modules} modules).",
            f"- **Evaluation Horizons**: Time-bounded snapshots $t \\in \\{{{oulad_horizons}\\}}$ days from course start.",
            "- **Population Filtering**: Only registrations where `date_unregistration` is null or $> t$. Students withdrawing on or before $t$ are excluded.",
            "- **Temporal Cutoff**: Clickstream activity, weekly interaction sequences, and assessment submissions restricted strictly to `date <= t`.",
            f"- **Demographic Isolation**: PROTECTED attributes ({', '.join(f'`{c}`' for c in PROTECTED['oulad'])}) are not in $X$ and are held in a separate audit frame. `highest_education` is an audited group that is also a model feature.",
            "- **Evaluation Protocol**:",
            "  - Primary: Temporal split — train on `2013B + 2013J`, test on `2014B + 2014J`.",
            f"  - Secondary: Leave-One-Module-Out (LOGO across {oulad_n_modules} modules).",
            "",
            "### Early-Warning Earliness Curves",
            "",
            "![OULAD Earliness Curves](figures/earliness_curve.png)",
            "",
            "### Temporal Holdout Evaluation (Train 2013B/J, Test 2014B/J)",
            "",
            "| Snapshot Horizon | Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Top 10% Prec (95% CI) | Top 10% Recall (95% CI) | Brier Score (95% CI) | ECE (95% CI) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        # Sort OULAD artifacts by t
        def get_t(art: Dict[str, Any]) -> int:
            fs = art.get("feature_set", "")
            if "t" in fs:
                try:
                    return int(fs.split("t")[-1])
                except Exception:
                    return 0
            return 0

        sorted_oulad = sorted(oulad_artifacts, key=get_t)

        for art in sorted_oulad:
            fs = art.get("feature_set", "")
            fs_label = FEATURE_SET_LABELS.get(fs, fs)
            strat_res = art.get("results", {}).get("predefined_split", {})

            for m_key in ["majority_class", "logistic_regression", "xgboost", "pytorch_gru"]:
                if m_key not in strat_res:
                    continue
                m_data = strat_res.get(m_key, {})
                metrics = m_data.get("metrics", {})
                m_label = MODEL_NAMES.get(m_key, m_key)
                pr = format_stat(metrics.get("pr_auc"))
                roc = format_stat(metrics.get("roc_auc"))
                p10 = format_stat(metrics.get("precision_top_10"))
                r10 = format_stat(metrics.get("recall_top_10"))
                brier = format_stat(metrics.get("brier_score"))
                ece = format_stat(metrics.get("expected_calibration_error"))
                lines.append(f"| {fs_label} | {m_label} | {pr} | {roc} | {p10} | {r10} | {brier} | {ece} |")

        lines.append("")
        oulad_logo_rows = []
        for art in sorted_oulad:
            fs = art.get("feature_set", "")
            logo_res = art.get("results", {}).get("leave_one_group_out", {})
            for m_key in ["majority_class", "logistic_regression", "xgboost", "pytorch_gru"]:
                if m_key in logo_res:
                    oulad_logo_rows.append((FEATURE_SET_LABELS.get(fs, fs), MODEL_NAMES.get(m_key, m_key),
                                            logo_res[m_key], f"oulad {fs}"))
        if oulad_logo_rows:
            lines.extend(_logo_tables(oulad_logo_rows, "Module"))

        lines.extend([
            "",
            "---",
            "",
        ])

    # SECTION 3: SIM-TO-REAL CROSS-DOMAIN TRANSFER BENCHMARK
    sim_to_real_path = BENCHMARK_DIR / "sim_to_real.json"
    if sim_to_real_path.exists():
        with open(sim_to_real_path, "r", encoding="utf-8") as f:
            s2r_data = json.load(f)

        lines.extend([
            "## 3. Sim-to-Real Cross-Domain Transfer Benchmark",
            "",
            "- **Evaluation Scope**: Cross-domain generalization between empirical benchmark (UCI ID 697) and simulated Indian cohort.",
            f"- **Shared Proxies ({len(s2r_data['shared_proxies'])})**: {', '.join(f'`{c}`' for c in s2r_data['shared_proxies'])}.",
            "- **Normalization**: Standardized within domain to reflect relative cohort position.",
            f"- **Confidence Intervals**: 95% bootstrap confidence intervals ({s2r_data['n_bootstraps']:,} resamples of holdout predictions).",
            "",
            "| Evaluation Mode | Training Domain | Test Domain | N (Train / Test) | ROC-AUC (95% CI) | PR-AUC (95% CI) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        evals = s2r_data.get("transfer_evaluations", {})
        for key in ["real_on_real", "sim_on_sim", "sim_to_real", "real_to_sim"]:
            if key not in evals:
                continue
            item = evals[key]
            desc = item.get("description", key)
            tr_dom = item.get("train_domain", "")
            te_dom = item.get("test_domain", "")
            n_tr = item.get("n_train", 0)
            n_te = item.get("n_test", 0)
            roc = format_stat(item.get("metrics", {}).get("roc_auc"))
            pr = format_stat(item.get("metrics", {}).get("pr_auc"))
            lines.append(f"| {desc} | {tr_dom} | {te_dom} | {n_tr} / {n_te} | {roc} | {pr} |")

        lines.extend([
            "",
            "---",
            "",
        ])

    REPORT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD_PATH.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Successfully generated benchmark report at %s", REPORT_MD_PATH)
    print(f"Successfully generated benchmark report at {REPORT_MD_PATH}")

    # Keep the README real-data benchmarks block in sync with the same artifacts
    from scripts.render_readme_benchmarks import update_readme as update_readme_benchmarks
    update_readme_benchmarks()


if __name__ == "__main__":
    render_benchmark_report()
