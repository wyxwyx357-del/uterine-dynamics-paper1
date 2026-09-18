#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the independent five-case minimal feature table.

This runner is intentionally not called by the formal project CMD. It only
reads frozen formal-QC RSR and independent DICOM curvature outputs. An explicit
output path is required, and an existing file is never overwritten unless
``--overwrite`` is supplied.
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
    MINIMAL_FEATURE_COLUMNS,
    extract_minimal_case_features,
)
from peristalsis_pipeline.formal_qc_contract import (
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
DEFAULT_DICOM_DIR = (
    PROJECT_ROOT
    / "输出"
    / "11_DICOM物理标定_独立"
    / "step1_5f_dicom_curvature_calibration_v1_5case"
)
RSR_SUFFIX = "_rsr_fourway_v1.npz"
DICOM_SUFFIX = "_dicom_curvature_v1.npz"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def extract_case(rsr_path: Path, dicom_dir: Path) -> dict[str, object]:
    case_id = rsr_path.name[: -len(RSR_SUFFIX)]
    rsr = load_npz(rsr_path)
    if "current_qc_radial_strain_rate_s" not in rsr:
        raise KeyError(f"{rsr_path}: current_qc_radial_strain_rate_s is missing")
    validate_formal_qc_lineage(rsr, source=rsr_path)
    validate_case_id(rsr, expected_case_id=case_id, source=rsr_path)

    dicom_path = dicom_dir / f"{case_id}{DICOM_SUFFIX}"
    if dicom_path.exists():
        dicom = load_npz(dicom_path)
        validate_formal_qc_lineage(dicom, source=dicom_path)
        validate_case_id(dicom, expected_case_id=case_id, source=dicom_path)
        validate_matching_formal_qc_lineage(
            rsr,
            dicom,
            left_source=rsr_path,
            right_source=dicom_path,
        )
        validate_dicom_curvature_rate_lineage(dicom, source=dicom_path)
        curvature_rate = dicom.get("qc_wall_curvature_change_rate_mm_inv_s")
        curvature_available = dicom.get("physical_curvature_available", False)
        curvature_rate_available = dicom.get(
            "physical_curvature_rate_available", False
        )
    else:
        curvature_rate = None
        curvature_available = False
        curvature_rate_available = False

    return extract_minimal_case_features(
        case_id=case_id,
        current_qc_radial_strain_rate_s=rsr[
            "current_qc_radial_strain_rate_s"
        ],
        qc_wall_curvature_change_rate_mm_inv_s=curvature_rate,
        physical_curvature_available=curvature_available,
        physical_curvature_rate_available=curvature_rate_available,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rsr-dir", type=Path, default=DEFAULT_RSR_DIR)
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.rsr_dir.is_dir():
        raise FileNotFoundError(f"formal RSR directory not found: {args.rsr_dir}")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_csv}")
    if not args.output_csv.parent.is_dir():
        raise FileNotFoundError(
            f"output parent directory does not exist: {args.output_csv.parent}"
        )

    rsr_paths = sorted(args.rsr_dir.glob(f"*{RSR_SUFFIX}"))
    if not rsr_paths:
        raise FileNotFoundError(f"no formal RSR NPZ found in: {args.rsr_dir}")

    rows = [extract_case(path, args.dicom_dir) for path in rsr_paths]
    table = pd.DataFrame(rows, columns=MINIMAL_FEATURE_COLUMNS)
    table.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"wrote {len(table)} case rows: {args.output_csv}")


if __name__ == "__main__":
    main()
