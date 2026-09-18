#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the audited 28-case formal patient-level F01-F20 table.

The runner reads only frozen formal RSR, anatomical deformation, and DICOM
curvature results.  It reuses the current five-case reader/lineage checks,
which call ``extract_formal_candidate_case_features``.  Before writing the
28-case output it must reproduce the frozen five-case feature table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline.formal_feature_extraction import (  # noqa: E402
    BIOLOGICAL_FEATURE_COLUMNS,
    FORMAL_CANDIDATE_FEATURE_COLUMNS,
    QC_METADATA_COLUMNS,
)


FIVE_CASE_RUNNER_PATH = Path(__file__).with_name(
    "run_feature_v1_formal_candidate_5case.py"
)
FIVE_CASE_SPEC = importlib.util.spec_from_file_location(
    "formal_candidate_5case_runner", FIVE_CASE_RUNNER_PATH
)
if FIVE_CASE_SPEC is None or FIVE_CASE_SPEC.loader is None:
    raise ImportError(f"cannot load formal five-case runner: {FIVE_CASE_RUNNER_PATH}")
FIVE_CASE_RUNNER = importlib.util.module_from_spec(FIVE_CASE_SPEC)
sys.modules[FIVE_CASE_SPEC.name] = FIVE_CASE_RUNNER
FIVE_CASE_SPEC.loader.exec_module(FIVE_CASE_RUNNER)


DEFAULT_NEW_PIPELINE_DIR = PROJECT_ROOT / "输出" / "新患者全流程_20260903"
DEFAULT_RSR_DIR = DEFAULT_NEW_PIPELINE_DIR / "10_rsr_fourway"
DEFAULT_ANATOMICAL_DIR = DEFAULT_NEW_PIPELINE_DIR / "11_anatomical_deformation"
DEFAULT_DICOM_DIR = PROJECT_ROOT / "输出" / "新患者DICOM物理曲率_20260904"
DEFAULT_DICOM_STATUS_CSV = DEFAULT_DICOM_DIR / "new_patient_dicom_curvature_status.csv"

DEFAULT_FIVE_RSR_DIR = FIVE_CASE_RUNNER.DEFAULT_RSR_DIR
DEFAULT_FIVE_ANATOMICAL_DIR = FIVE_CASE_RUNNER.DEFAULT_ANATOMICAL_DIR
DEFAULT_FIVE_DICOM_DIR = FIVE_CASE_RUNNER.DEFAULT_DICOM_DIR
DEFAULT_FIVE_REFERENCE_CSV = FIVE_CASE_RUNNER.DEFAULT_OUTPUT_CSV

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "输出" / "14_28例正式患者级特征_20260905"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "feature_v1_formal_candidate_28case.csv"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "feature_v1_formal_candidate_28case_manifest.json"

EXPECTED_CASE_COUNT = 28
REGRESSION_ATOL = 1e-12
REGRESSION_RTOL = 1e-12
MISSING_AUDIT_COLUMNS = (
    "f01_f20_all_evaluable",
    "f01_f20_missing_codes",
    "f01_f20_missing_reason",
)
FEATURE_CODE_MAP = tuple(
    (f"F{index:02d}", name)
    for index, name in enumerate(BIOLOGICAL_FEATURE_COLUMNS, start=1)
)
CURVATURE_CODES = {
    code for code, name in FEATURE_CODE_MAP if "curvature_change_rate" in name
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


def build_formal_table(
    *,
    rsr_dir: Path,
    anatomical_dir: Path,
    dicom_dir: Path,
    expected_case_count: int,
) -> pd.DataFrame:
    for label, directory in (
        ("formal RSR", rsr_dir),
        ("formal anatomical", anatomical_dir),
        ("formal DICOM", dicom_dir),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {directory}")

    rsr_paths = sorted(rsr_dir.glob(f"*{FIVE_CASE_RUNNER.RSR_SUFFIX}"))
    if len(rsr_paths) != expected_case_count:
        raise ValueError(
            f"expected exactly {expected_case_count} formal RSR cases, "
            f"found {len(rsr_paths)} in {rsr_dir}"
        )
    rows = [
        FIVE_CASE_RUNNER.extract_case(path, anatomical_dir, dicom_dir)
        for path in rsr_paths
    ]
    case_ids = [str(row["case_id"]) for row in rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_id values must be unique")
    expected_columns = set(FORMAL_CANDIDATE_FEATURE_COLUMNS)
    for case_id, row in zip(case_ids, rows):
        if set(row) != expected_columns:
            raise ValueError(f"{case_id}: formal candidate column contract mismatch")

    table = pd.DataFrame(rows, columns=FORMAL_CANDIDATE_FEATURE_COLUMNS)
    expected_shape = (expected_case_count, len(FORMAL_CANDIDATE_FEATURE_COLUMNS))
    if table.shape != expected_shape:
        raise ValueError(
            f"formal candidate table must have shape {expected_shape}, got {table.shape}"
        )
    return table


def run_five_case_regression(
    *,
    rsr_dir: Path,
    anatomical_dir: Path,
    dicom_dir: Path,
    reference_csv: Path,
) -> dict[str, Any]:
    if not reference_csv.is_file():
        raise FileNotFoundError(f"five-case reference CSV not found: {reference_csv}")
    actual = build_formal_table(
        rsr_dir=rsr_dir,
        anatomical_dir=anatomical_dir,
        dicom_dir=dicom_dir,
        expected_case_count=5,
    ).set_index("case_id")
    expected = pd.read_csv(reference_csv).set_index("case_id")
    if set(actual.index) != set(expected.index):
        raise RuntimeError(
            "five-case regression case_id mismatch: "
            f"actual={sorted(actual.index)}, expected={sorted(expected.index)}"
        )

    actual = actual.loc[sorted(actual.index), list(BIOLOGICAL_FEATURE_COLUMNS)]
    expected = expected.loc[sorted(expected.index), list(BIOLOGICAL_FEATURE_COLUMNS)]
    mismatches: list[dict[str, Any]] = []
    maximum_absolute_difference = 0.0
    for code, feature_name in FEATURE_CODE_MAP:
        actual_values = pd.to_numeric(actual[feature_name], errors="coerce").to_numpy()
        expected_values = pd.to_numeric(expected[feature_name], errors="coerce").to_numpy()
        equal = np.isclose(
            actual_values,
            expected_values,
            rtol=REGRESSION_RTOL,
            atol=REGRESSION_ATOL,
            equal_nan=True,
        )
        finite_pair = np.isfinite(actual_values) & np.isfinite(expected_values)
        if np.any(finite_pair):
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(np.max(np.abs(actual_values[finite_pair] - expected_values[finite_pair]))),
            )
        for row_index in np.flatnonzero(~equal):
            mismatches.append(
                {
                    "case_id": str(actual.index[row_index]),
                    "feature_code": code,
                    "feature_name": feature_name,
                    "actual": (
                        None
                        if math.isnan(float(actual_values[row_index]))
                        else float(actual_values[row_index])
                    ),
                    "expected": (
                        None
                        if math.isnan(float(expected_values[row_index]))
                        else float(expected_values[row_index])
                    ),
                }
            )
    if mismatches:
        raise RuntimeError(
            "five-case F01-F20 regression failed:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
    return {
        "passed": True,
        "case_count": len(actual),
        "feature_count": len(BIOLOGICAL_FEATURE_COLUMNS),
        "comparison_count": len(actual) * len(BIOLOGICAL_FEATURE_COLUMNS),
        "rtol": REGRESSION_RTOL,
        "atol": REGRESSION_ATOL,
        "maximum_absolute_difference": maximum_absolute_difference,
        "reference_csv": str(reference_csv.resolve()),
    }


def load_dicom_failure_reasons(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"case_id", "failure_reason"}
    if not required.issubset(table.columns):
        raise ValueError(f"DICOM status CSV lacks required fields: {path}")
    return {
        str(row.case_id): str(row.failure_reason).strip()
        for row in table.itertuples(index=False)
        if str(row.failure_reason).strip()
    }


def add_missing_audit_columns(
    table: pd.DataFrame, *, dicom_failure_reasons: dict[str, str]
) -> pd.DataFrame:
    audited = table.copy()
    all_evaluable: list[bool] = []
    missing_codes_column: list[str] = []
    missing_reason_column: list[str] = []
    for row in audited.itertuples(index=False):
        missing_codes = [
            code
            for code, feature_name in FEATURE_CODE_MAP
            if pd.isna(getattr(row, feature_name))
        ]
        all_evaluable.append(not missing_codes)
        missing_codes_column.append("|".join(missing_codes))
        if not missing_codes:
            missing_reason_column.append("")
        elif set(missing_codes).issubset(CURVATURE_CODES):
            source_reason = dicom_failure_reasons.get(str(row.case_id), "")
            prefix = "DICOM physical curvature rate unavailable"
            missing_reason_column.append(
                f"{prefix}: {source_reason}" if source_reason else prefix
            )
        else:
            missing_reason_column.append(
                "No evaluable formal samples after upstream QC for "
                + "|".join(missing_codes)
            )
    audited[MISSING_AUDIT_COLUMNS[0]] = all_evaluable
    audited[MISSING_AUDIT_COLUMNS[1]] = missing_codes_column
    audited[MISSING_AUDIT_COLUMNS[2]] = missing_reason_column
    return audited


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rsr-dir", type=Path, default=DEFAULT_RSR_DIR)
    parser.add_argument("--anatomical-dir", type=Path, default=DEFAULT_ANATOMICAL_DIR)
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    parser.add_argument(
        "--dicom-status-csv", type=Path, default=DEFAULT_DICOM_STATUS_CSV
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--five-rsr-dir", type=Path, default=DEFAULT_FIVE_RSR_DIR)
    parser.add_argument(
        "--five-anatomical-dir", type=Path, default=DEFAULT_FIVE_ANATOMICAL_DIR
    )
    parser.add_argument("--five-dicom-dir", type=Path, default=DEFAULT_FIVE_DICOM_DIR)
    parser.add_argument(
        "--five-reference-csv", type=Path, default=DEFAULT_FIVE_REFERENCE_CSV
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regression = run_five_case_regression(
        rsr_dir=args.five_rsr_dir.resolve(),
        anatomical_dir=args.five_anatomical_dir.resolve(),
        dicom_dir=args.five_dicom_dir.resolve(),
        reference_csv=args.five_reference_csv.resolve(),
    )

    output_csv = args.output_csv.resolve()
    manifest_path = args.manifest.resolve()
    for path in (output_csv, manifest_path):
        if path.exists() and not args.allow_overwrite:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    rsr_dir = args.rsr_dir.resolve()
    anatomical_dir = args.anatomical_dir.resolve()
    dicom_dir = args.dicom_dir.resolve()
    dicom_status_csv = args.dicom_status_csv.resolve()
    formal_table = build_formal_table(
        rsr_dir=rsr_dir,
        anatomical_dir=anatomical_dir,
        dicom_dir=dicom_dir,
        expected_case_count=EXPECTED_CASE_COUNT,
    )
    table = add_missing_audit_columns(
        formal_table,
        dicom_failure_reasons=load_dicom_failure_reasons(dicom_status_csv),
    )
    expected_columns = len(FORMAL_CANDIDATE_FEATURE_COLUMNS) + len(
        MISSING_AUDIT_COLUMNS
    )
    if table.shape != (EXPECTED_CASE_COUNT, expected_columns):
        raise RuntimeError(f"unexpected audited output shape: {table.shape}")
    if not table["case_id"].is_unique:
        raise RuntimeError("case_id must be unique before writing the formal CSV")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.parent != output_csv.parent:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False, encoding="utf-8-sig", na_rep="NA")

    feature_missing_counts = {
        code: int(table[feature_name].isna().sum())
        for code, feature_name in FEATURE_CODE_MAP
    }
    curvature_unavailable_cases = []
    for row in table.loc[
        ~table["dicom_physical_curvature_rate_available"].astype(bool)
    ].itertuples(index=False):
        curvature_unavailable_cases.append(
            {
                "case_id": str(row.case_id),
                "missing_feature_codes": str(row.f01_f20_missing_codes),
                "reason": str(row.f01_f20_missing_reason),
            }
        )

    manifest = {
        "status": "formal_patient_level_feature_table_created",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "input_directories": {
            "formal_rsr": str(rsr_dir),
            "formal_anatomical_deformation": str(anatomical_dir),
            "formal_dicom_curvature": str(dicom_dir),
        },
        "input_file_counts": {
            "formal_rsr_npz": len(
                list(rsr_dir.glob(f"*{FIVE_CASE_RUNNER.RSR_SUFFIX}"))
            ),
            "formal_anatomical_npz": len(
                list(anatomical_dir.glob(f"*{FIVE_CASE_RUNNER.ANATOMICAL_SUFFIX}"))
            ),
            "formal_dicom_npz": len(
                list(dicom_dir.glob(f"*{FIVE_CASE_RUNNER.DICOM_SUFFIX}"))
            ),
        },
        "aggregation_function": {
            "qualified_name": (
                "peristalsis_pipeline.formal_feature_extraction."
                "extract_formal_candidate_case_features"
            ),
            "path": str(
                (CODE_ROOT / "peristalsis_pipeline" / "formal_feature_extraction.py").resolve()
            ),
            "sha256": sha256(
                CODE_ROOT / "peristalsis_pipeline" / "formal_feature_extraction.py"
            ),
        },
        "entrypoint": str(Path(__file__).resolve()),
        "five_case_regression": regression,
        "output_csv": str(output_csv),
        "output_patient_count": len(table),
        "output_column_count": len(table.columns),
        "case_id_unique": bool(table["case_id"].is_unique),
        "feature_code_name_order": [
            {"feature_code": code, "feature_name": name}
            for code, name in FEATURE_CODE_MAP
        ],
        "feature_missing_counts": feature_missing_counts,
        "curvature_unavailable_cases": curvature_unavailable_cases,
        "missing_value_representation": "NA in CSV; parsed as missing/NaN",
        "excluded_patient_count": 0,
        "imputation_performed": False,
        "standardization_performed": False,
        "feature_selection_performed": False,
        "forbidden_test_derived_sources_used": [],
        "output_csv_sha256": sha256(output_csv),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
