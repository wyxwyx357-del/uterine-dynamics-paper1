#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the independent five-case, 27-column formal candidate table.

This runner is intentionally outside the formal 16-step CMD and published
view. It reads frozen formal RSR, formal anatomical deformation, and optional
independent DICOM curvature outputs without recomputing any upstream result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline.formal_feature_extraction import (
    FORMAL_CANDIDATE_FEATURE_COLUMNS,
    extract_formal_candidate_case_features,
)
from peristalsis_pipeline.formal_qc_contract import (
    FORMAL_QC_LINEAGE_ID_FIELD,
    FORMAL_QC_VERSION_FIELD,
    validate_case_id,
    validate_formal_qc_lineage,
    validate_matching_formal_qc_lineage,
)
from peristalsis_pipeline.dicom_curvature_calibration import (
    validate_dicom_curvature_rate_lineage,
)


DEFAULT_RSR_DIR = (
    PROJECT_ROOT
    / "输出"
    / "06_影子质控与四路RSR_敏感性"
    / "step1_5f_rsr_fourway_v1_5case"
)
DEFAULT_ANATOMICAL_DIR = (
    PROJECT_ROOT
    / "输出"
    / "07_解剖形变_敏感性"
    / "step1_5f_anatomical_deformation_v1_5case"
)
DEFAULT_DICOM_DIR = (
    PROJECT_ROOT
    / "输出"
    / "11_DICOM物理标定_独立"
    / "step1_5f_dicom_curvature_calibration_v1_5case"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "输出"
    / "12_正式候选特征"
    / "feature_v1_formal_candidate_5case.csv"
)

RSR_SUFFIX = "_rsr_fourway_v1.npz"
ANATOMICAL_SUFFIX = "_anatomical_deformation_v1.npz"
DICOM_SUFFIX = "_dicom_curvature_v1.npz"
EXPECTED_CASE_COUNT = 5

REQUIRED_RSR_FIELDS = {
    "case_id",
    "current_qc_radial_strain_rate_s",
    "current_qc_semantics",
    FORMAL_QC_VERSION_FIELD,
    FORMAL_QC_LINEAGE_ID_FIELD,
    "fps",
}
REQUIRED_ANATOMICAL_FIELDS = {
    "case_id",
    "pair_qc_valid",
    "cavity_qc_valid",
    "longitudinal_qc_valid",
    "qc_cavity_width_strain_rate_s",
    "qc_longitudinal_wall_strain_rate_s",
    "fps",
    "evidence_layer",
    "qc_feature_branch_is_formal",
    "formal_propagation_released",
    "current_qc_semantics",
    FORMAL_QC_VERSION_FIELD,
    FORMAL_QC_LINEAGE_ID_FIELD,
}
REQUIRED_DICOM_FIELDS = {
    "case_id",
    "pair_qc_valid",
    "qc_wall_curvature_change_rate_mm_inv_s",
    "physical_curvature_available",
    "physical_curvature_rate_available",
    "curvature_rate_timebase",
    "dicom_timing_numeric_match",
    "current_qc_semantics",
    FORMAL_QC_VERSION_FIELD,
    FORMAL_QC_LINEAGE_ID_FIELD,
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def require_fields(
    payload: dict[str, np.ndarray], required: set[str], *, path: Path
) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise KeyError(f"{path}: required fields are missing: {', '.join(missing)}")


def scalar_text(value: np.ndarray, *, field: str, path: Path) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: {field} must be scalar")
    return str(array.reshape(-1)[0])


def scalar_bool(value: np.ndarray, *, field: str, path: Path) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: {field} must be scalar")
    return bool(array.reshape(-1)[0])


def extract_case(
    rsr_path: Path,
    anatomical_dir: Path,
    dicom_dir: Path,
) -> dict[str, object]:
    case_id = rsr_path.name[: -len(RSR_SUFFIX)]
    rsr = load_npz(rsr_path)
    require_fields(rsr, REQUIRED_RSR_FIELDS, path=rsr_path)
    validate_formal_qc_lineage(rsr, source=rsr_path)
    validate_case_id(rsr, expected_case_id=case_id, source=rsr_path)

    anatomical_path = anatomical_dir / f"{case_id}{ANATOMICAL_SUFFIX}"
    if not anatomical_path.is_file():
        raise FileNotFoundError(
            f"formal anatomical NPZ not found for {case_id}: {anatomical_path}"
        )
    anatomical = load_npz(anatomical_path)
    require_fields(anatomical, REQUIRED_ANATOMICAL_FIELDS, path=anatomical_path)
    validate_formal_qc_lineage(anatomical, source=anatomical_path)
    validate_case_id(
        anatomical, expected_case_id=case_id, source=anatomical_path
    )
    validate_matching_formal_qc_lineage(
        rsr,
        anatomical,
        left_source=rsr_path,
        right_source=anatomical_path,
    )
    if scalar_text(
        anatomical["evidence_layer"], field="evidence_layer", path=anatomical_path
    ) != "roi_deformation_activity":
        raise ValueError(
            f"{anatomical_path}: evidence_layer is not roi_deformation_activity"
        )
    if not scalar_bool(
        anatomical["qc_feature_branch_is_formal"],
        field="qc_feature_branch_is_formal",
        path=anatomical_path,
    ):
        raise ValueError(f"{anatomical_path}: anatomical branch is not formal QC")
    if scalar_bool(
        anatomical["formal_propagation_released"],
        field="formal_propagation_released",
        path=anatomical_path,
    ):
        raise ValueError(
            f"{anatomical_path}: formal propagation must remain unreleased"
        )

    dicom_path = dicom_dir / f"{case_id}{DICOM_SUFFIX}"
    if dicom_path.is_file():
        dicom = load_npz(dicom_path)
        require_fields(dicom, REQUIRED_DICOM_FIELDS, path=dicom_path)
        validate_formal_qc_lineage(dicom, source=dicom_path)
        validate_case_id(dicom, expected_case_id=case_id, source=dicom_path)
        validate_matching_formal_qc_lineage(
            anatomical,
            dicom,
            left_source=anatomical_path,
            right_source=dicom_path,
        )
        validate_dicom_curvature_rate_lineage(dicom, source=dicom_path)
        curvature_rate = dicom["qc_wall_curvature_change_rate_mm_inv_s"]
        dicom_pair_valid = dicom["pair_qc_valid"]
        curvature_available = dicom["physical_curvature_available"]
        curvature_rate_available = dicom[
            "physical_curvature_rate_available"
        ]
    else:
        curvature_rate = None
        dicom_pair_valid = None
        curvature_available = False
        curvature_rate_available = False

    return extract_formal_candidate_case_features(
        case_id=case_id,
        current_qc_radial_strain_rate_s=rsr[
            "current_qc_radial_strain_rate_s"
        ],
        rsr_fps=rsr["fps"],
        anatomical_pair_qc_valid=anatomical["pair_qc_valid"],
        anatomical_fps=anatomical["fps"],
        qc_cavity_width_strain_rate_s=anatomical[
            "qc_cavity_width_strain_rate_s"
        ],
        cavity_qc_valid=anatomical["cavity_qc_valid"],
        qc_longitudinal_wall_strain_rate_s=anatomical[
            "qc_longitudinal_wall_strain_rate_s"
        ],
        longitudinal_qc_valid=anatomical["longitudinal_qc_valid"],
        qc_wall_curvature_change_rate_mm_inv_s=curvature_rate,
        dicom_pair_qc_valid=dicom_pair_valid,
        physical_curvature_available=curvature_available,
        physical_curvature_rate_available=curvature_rate_available,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rsr-dir", type=Path, default=DEFAULT_RSR_DIR)
    parser.add_argument(
        "--anatomical-dir", type=Path, default=DEFAULT_ANATOMICAL_DIR
    )
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.rsr_dir.is_dir():
        raise FileNotFoundError(f"formal RSR directory not found: {args.rsr_dir}")
    if not args.anatomical_dir.is_dir():
        raise FileNotFoundError(
            f"formal anatomical directory not found: {args.anatomical_dir}"
        )
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_csv}")

    rsr_paths = sorted(args.rsr_dir.glob(f"*{RSR_SUFFIX}"))
    if not rsr_paths:
        raise FileNotFoundError(f"no formal RSR NPZ found in: {args.rsr_dir}")

    rows = [
        extract_case(path, args.anatomical_dir, args.dicom_dir)
        for path in rsr_paths
    ]
    case_ids = [str(row["case_id"]) for row in rows]
    if len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"expected exactly {EXPECTED_CASE_COUNT} cases, found {len(rows)}"
        )
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_id values must be unique before writing CSV")
    expected_columns = set(FORMAL_CANDIDATE_FEATURE_COLUMNS)
    for case_id, row in zip(case_ids, rows):
        if set(row) != expected_columns:
            raise ValueError(f"{case_id}: formal candidate column contract mismatch")

    table = pd.DataFrame(rows, columns=FORMAL_CANDIDATE_FEATURE_COLUMNS)
    expected_shape = (EXPECTED_CASE_COUNT, len(FORMAL_CANDIDATE_FEATURE_COLUMNS))
    if table.shape != expected_shape:
        raise ValueError(
            f"formal candidate table must have shape {expected_shape}, got {table.shape}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"wrote {table.shape[0]} case rows and {table.shape[1]} columns: {args.output_csv}")


if __name__ == "__main__":
    main()
