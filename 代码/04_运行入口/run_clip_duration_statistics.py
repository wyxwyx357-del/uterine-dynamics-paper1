#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze patient-specific proportional clip-duration robustness.

Input is the long table produced by run_clip_duration_robustness.py.
For every frozen F01-F20 measurement, each shortened duration condition is
compared with the same patient's 100% reference using absolute-agreement ICC,
patient-cluster bootstrap 95% CI, Bland-Altman summaries, absolute/relative
errors, and Spearman rank correlation. No feature selection or exclusion
threshold is created here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline.clip_duration_statistics import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPETITIONS,
    DEFAULT_BOOTSTRAP_SEED,
    pairwise_duration_statistics,
)
from peristalsis_pipeline.formal_feature_extraction import (  # noqa: E402
    BIOLOGICAL_FEATURE_COLUMNS,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "输出" / "Paper1_时长比例鲁棒性"
DEFAULT_LONG_CSV = DEFAULT_OUTPUT_DIR / "clip_duration_proportional_f01_f20_long.csv"
DEFAULT_PAIRWISE_SUMMARY = DEFAULT_OUTPUT_DIR / "clip_duration_pairwise_summary.csv"
DEFAULT_CASE_ERRORS = DEFAULT_OUTPUT_DIR / "clip_duration_case_errors.csv"
DEFAULT_QC_SUMMARY = DEFAULT_OUTPUT_DIR / "clip_duration_qc_summary.csv"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "clip_duration_statistics_manifest.json"

FEATURE_CODE_MAP = tuple(
    (f"F{index:02d}", name)
    for index, name in enumerate(BIOLOGICAL_FEATURE_COLUMNS, start=1)
)
REQUIRED_METADATA = {
    "case_id",
    "target_percent",
    "actual_fraction",
    "full_pair_duration_s",
    "window_pair_duration_s",
    "rsr_valid_ratio",
    "cavity_valid_ratio",
    "longitudinal_valid_ratio",
    "curvature_valid_ratio",
    "f01_f20_all_evaluable",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def validate_long_table(table: pd.DataFrame) -> tuple[int, ...]:
    required = REQUIRED_METADATA | set(BIOLOGICAL_FEATURE_COLUMNS)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "clip-duration long table is missing required columns: "
            + ", ".join(missing)
        )
    if table.empty:
        raise ValueError("clip-duration long table is empty")
    if table.duplicated(["case_id", "target_percent"]).any():
        raise ValueError("case_id + target_percent must be unique")

    proportions = tuple(
        sorted(
            {int(x) for x in table["target_percent"].dropna().tolist()},
            reverse=True,
        )
    )
    if 100 not in proportions:
        raise ValueError("clip-duration table must contain the 100% reference")

    expected = set(proportions)
    for case_id, group in table.groupby("case_id", sort=False):
        actual = {int(x) for x in group["target_percent"].tolist()}
        if actual != expected:
            raise ValueError(
                f"{case_id}: proportional rows are incomplete; "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
    return proportions


def build_case_errors(table: pd.DataFrame, proportions: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    reference = table.loc[table["target_percent"] == 100].set_index("case_id")

    for percent in proportions:
        if percent == 100:
            continue
        shortened = table.loc[table["target_percent"] == percent].set_index("case_id")
        common = sorted(set(reference.index) & set(shortened.index))
        for feature_code, feature_name in FEATURE_CODE_MAP:
            ref_values = pd.to_numeric(
                reference.loc[common, feature_name], errors="coerce"
            )
            short_values = pd.to_numeric(
                shortened.loc[common, feature_name], errors="coerce"
            )
            for case_id, ref_value, short_value in zip(
                common, ref_values.to_numpy(), short_values.to_numpy()
            ):
                finite_pair = bool(
                    np.isfinite(ref_value) and np.isfinite(short_value)
                )
                difference = (
                    float(short_value - ref_value) if finite_pair else math.nan
                )
                absolute_difference = (
                    abs(difference) if finite_pair else math.nan
                )
                relative_error_pct = math.nan
                if finite_pair and float(ref_value) != 0.0:
                    relative_error_pct = (
                        abs(difference) / abs(float(ref_value)) * 100.0
                    )
                rows.append(
                    {
                        "case_id": str(case_id),
                        "target_percent": percent,
                        "feature_code": feature_code,
                        "feature_name": feature_name,
                        "reference_value_100pct": (
                            float(ref_value) if np.isfinite(ref_value) else math.nan
                        ),
                        "shortened_value": (
                            float(short_value)
                            if np.isfinite(short_value)
                            else math.nan
                        ),
                        "finite_pair": finite_pair,
                        "difference_short_minus_reference": difference,
                        "absolute_difference": absolute_difference,
                        "absolute_relative_error_pct": relative_error_pct,
                    }
                )
    return pd.DataFrame(rows)


def build_pairwise_summary(
    table: pd.DataFrame,
    proportions: tuple[int, ...],
    *,
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    reference = table.loc[table["target_percent"] == 100].set_index("case_id")
    rows: list[dict[str, object]] = []

    for proportion_index, percent in enumerate(proportions):
        if percent == 100:
            continue
        shortened = table.loc[table["target_percent"] == percent].set_index("case_id")
        common = sorted(set(reference.index) & set(shortened.index))
        for feature_index, (feature_code, feature_name) in enumerate(FEATURE_CODE_MAP):
            stats = pairwise_duration_statistics(
                pd.to_numeric(
                    reference.loc[common, feature_name], errors="coerce"
                ).to_numpy(),
                pd.to_numeric(
                    shortened.loc[common, feature_name], errors="coerce"
                ).to_numpy(),
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed + proportion_index * 1000 + feature_index,
            )
            rows.append(
                {
                    "feature_code": feature_code,
                    "feature_name": feature_name,
                    "target_percent": percent,
                    "reference_percent": 100,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def _quartiles(values: pd.Series) -> tuple[float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan, math.nan, math.nan
    q1, median, q3 = np.percentile(numeric.to_numpy(dtype=float), [25, 50, 75])
    return float(q1), float(median), float(q3)


def build_qc_summary(table: pd.DataFrame, proportions: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for percent in proportions:
        group = table.loc[table["target_percent"] == percent]
        duration_q1, duration_median, duration_q3 = _quartiles(
            group["window_pair_duration_s"]
        )
        row: dict[str, object] = {
            "target_percent": percent,
            "n_cases": int(len(group)),
            "window_pair_duration_s_min": float(
                pd.to_numeric(group["window_pair_duration_s"], errors="coerce").min()
            ),
            "window_pair_duration_s_q1": duration_q1,
            "window_pair_duration_s_median": duration_median,
            "window_pair_duration_s_q3": duration_q3,
            "window_pair_duration_s_max": float(
                pd.to_numeric(group["window_pair_duration_s"], errors="coerce").max()
            ),
            "all_f01_f20_evaluable_n": int(
                group["f01_f20_all_evaluable"].astype(bool).sum()
            ),
        }
        for column in (
            "rsr_valid_ratio",
            "cavity_valid_ratio",
            "longitudinal_valid_ratio",
            "curvature_valid_ratio",
        ):
            q1, median, q3 = _quartiles(group[column])
            row[f"{column}_q1"] = q1
            row[f"{column}_median"] = median
            row[f"{column}_q3"] = q3
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-long-csv", type=Path, default=DEFAULT_LONG_CSV)
    parser.add_argument(
        "--output-pairwise-summary",
        type=Path,
        default=DEFAULT_PAIRWISE_SUMMARY,
    )
    parser.add_argument("--output-case-errors", type=Path, default=DEFAULT_CASE_ERRORS)
    parser.add_argument("--output-qc-summary", type=Path, default=DEFAULT_QC_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_long_csv.resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"clip-duration long CSV not found: {input_csv}")

    outputs = [
        args.output_pairwise_summary.resolve(),
        args.output_case_errors.resolve(),
        args.output_qc_summary.resolve(),
        args.manifest.resolve(),
    ]
    for path in outputs:
        if path.exists() and not args.allow_overwrite:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    table = pd.read_csv(input_csv)
    proportions = validate_long_table(table)

    pairwise = build_pairwise_summary(
        table,
        proportions,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    case_errors = build_case_errors(table, proportions)
    qc_summary = build_qc_summary(table, proportions)

    pairwise_path, errors_path, qc_path, manifest_path = outputs
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig", na_rep="NA")
    case_errors.to_csv(errors_path, index=False, encoding="utf-8-sig", na_rep="NA")
    qc_summary.to_csv(qc_path, index=False, encoding="utf-8-sig", na_rep="NA")

    manifest = {
        "status": "paper1_proportional_clip_duration_statistics_created",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "input_long_csv": str(input_csv),
        "input_long_csv_sha256": sha256(input_csv),
        "case_count": int(table["case_id"].nunique()),
        "proportions_pct": list(proportions),
        "feature_count": len(BIOLOGICAL_FEATURE_COLUMNS),
        "primary_comparisons": [
            f"{percent}% vs 100%" for percent in proportions if percent != 100
        ],
        "icc": {
            "form": "ICC(A,1)",
            "model": (
                "two-way absolute-agreement single-measure ICC with patient rows "
                "and fixed duration-condition columns"
            ),
            "ci": (
                f"patient-cluster percentile bootstrap, "
                f"{args.bootstrap_repetitions} repetitions"
            ),
            "seed": args.seed,
        },
        "bland_altman_difference": "shortened minus 100% reference",
        "relative_error": (
            "absolute(shortened-reference)/absolute(reference)*100; "
            "undefined when reference is exactly zero"
        ),
        "spearman": "secondary descriptive rank agreement",
        "feature_selection_performed": False,
        "new_exclusion_threshold_created": False,
        "outcome_information_used": False,
        "outputs": {
            "pairwise_summary": str(pairwise_path),
            "case_errors": str(errors_path),
            "qc_summary": str(qc_path),
        },
        "output_sha256": {
            "pairwise_summary": sha256(pairwise_path),
            "case_errors": sha256(errors_path),
            "qc_summary": sha256(qc_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
