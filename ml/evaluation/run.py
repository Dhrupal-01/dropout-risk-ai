"""
Benchmark Runner CLI Entrypoint
Usage:
    python -m ml.evaluation.run --source uci
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from ml.evaluation.harness import run_benchmark_for_dataset
from ml.sources.uci import FEATURE_SETS, get_uci_benchmark_dataset
from scripts.render_benchmark_report import render_benchmark_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_uci_benchmark_suite(seed: int = 42, output_dir: Path = None):
    """
    Executes full evaluation across all feature sets and label variants for UCI ID 697.
    - Feature sets: enrolment_time, end_of_sem1, full
    - Label variants: primary (N=3630), sensitivity (N=4424)
    """
    print("=" * 80)
    print(" DROPOUTGUARD — BENCHMARK EVALUATION HARNESS ")
    print(" Source: UCI ID 697 (Portuguese Higher-Ed Dropout Benchmark)")
    print("=" * 80)

    start_time = time.time()
    feature_sets = list(FEATURE_SETS.keys())
    label_variants = ["primary", "sensitivity"]

    total_runs = len(feature_sets) * len(label_variants)
    run_idx = 1

    for fs in feature_sets:
        for lv in label_variants:
            print(f"\n[{run_idx}/{total_runs}] Running benchmark: feature_set='{fs}', label_variant='{lv}'...")
            t0 = time.time()

            X, y, groups, cols = get_uci_benchmark_dataset(feature_set=fs, label_variant=lv)
            logger.info("Dataset loaded: %d samples, %d features, %d positive targets (%.2f%%)",
                        len(y), X.shape[1], int(y.sum()), float(y.mean() * 100))

            artifact = run_benchmark_for_dataset(
                X=X,
                y=y,
                groups=groups,
                dataset_name="uci_697",
                feature_set=fs,
                label_variant=lv,
                seed=seed,
                output_dir=output_dir,
            )

            dt = time.time() - t0
            print(f" [✓] Completed in {dt:.1f}s: ROC-AUC={artifact['results']['repeated_stratified_cv']['xgboost']['metrics']['roc_auc']['point']}")
            run_idx += 1

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" ALL {total_runs} BENCHMARK RUNS COMPLETED IN {total_time:.1f}s ")
    print("=" * 80)

    # Automatically generate markdown report and reliability diagram
    print("\n[i] Generating docs/benchmarks.md and reliability curves...")
    render_benchmark_report()
    print(" [✓] Benchmarks report and figures successfully generated.")


def run_oulad_benchmark_suite(seed: int = 42, output_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
    """
    Executes OULAD time-based early-warning benchmark across snapshot horizons t in {14, 28, 56, 84}.
    - Predefined temporal split: train on 2013B + 2013J, test on 2014B + 2014J
    - Secondary: Leave-one-module-out across 7 modules
    - Evaluates Majority Class, Logistic Regression, XGBoost, and PyTorch GRU
    """
    from ml.sources.oulad import SNAPSHOT_DAYS, build_snapshot_dataset, load_raw_tables

    print("=" * 80)
    print(" DROPOUTGUARD — BENCHMARK EVALUATION HARNESS ")
    print(" Source: OULAD / UCI ID 349 (Time-Based Early-Warning Dropout Benchmark)")
    print("=" * 80)

    start_time = time.time()
    logger.info("Loading OULAD tables...")
    tables = load_raw_tables(data_dir=data_dir)

    total_runs = len(SNAPSHOT_DAYS)
    for idx, t in enumerate(SNAPSHOT_DAYS, 1):
        print(f"\n[{idx}/{total_runs}] Extracting snapshot t={t} days and running benchmark...")
        t0 = time.time()

        X, y, predefined_splits, groups, audit_df, feature_names = build_snapshot_dataset(
            t=t, tables=tables, data_dir=data_dir
        )
        logger.info(
            "Snapshot t=%d: %d samples, %d features, %d positives (%.2f%%)",
            t, len(y), X.shape[1], int(y.sum()), float(y.mean() * 100)
        )

        artifact = run_benchmark_for_dataset(
            X=X,
            y=y,
            groups=groups,
            dataset_name="oulad_349",
            feature_set=f"snapshot_t{t}",
            label_variant="withdrawn",
            predefined_splits=predefined_splits,
            include_repeated_cv=False,
            include_gru=True,
            seed=seed,
            output_dir=output_dir,
        )

        dt = time.time() - t0
        xgb_prauc = artifact["results"]["predefined_split"]["xgboost"]["metrics"]["pr_auc"]["point"]
        print(f" [✓] Completed snapshot t={t} in {dt:.1f}s: XGBoost PR-AUC={xgb_prauc}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" ALL {total_runs} OULAD SNAPSHOT BENCHMARKS COMPLETED IN {total_time:.1f}s ")
    print("=" * 80)

    print("\n[i] Updating docs/benchmarks.md and figures...")
    render_benchmark_report()
    print(" [✓] Benchmark report and figures successfully updated.")


def main():
    parser = argparse.ArgumentParser(description="DropoutGuard Benchmark Evaluation Runner")
    parser.add_argument(
        "--source",
        type=str,
        default="uci",
        choices=["uci", "oulad"],
        help="Source dataset to benchmark (choices: uci, oulad; default: uci)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom path to raw data directory (e.g. data/raw/oulad/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility and bootstrap sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for benchmark JSON artifacts",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    data_dir = Path(args.data_dir) if args.data_dir else None

    if args.source == "uci":
        run_uci_benchmark_suite(seed=args.seed, output_dir=out_dir)
    elif args.source == "oulad":
        run_oulad_benchmark_suite(seed=args.seed, output_dir=out_dir, data_dir=data_dir)
    else:
        logger.error("Unsupported benchmark source '%s'", args.source)
        sys.exit(1)


if __name__ == "__main__":
    main()
