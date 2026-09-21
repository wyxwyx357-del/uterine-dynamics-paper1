#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate patient-specific proportional clip-duration F01-F20 tables.

This Paper 1 experiment reads only frozen formal RSR, anatomical deformation,
and optional DICOM curvature outputs. It does not rerun tracking and it does
not redefine F01-F20. For each case, the complete formal pair-frame sequence
is the 100% reference; deterministic centered 75%, 50%, and 25% windows are
sliced from that same sequence and passed to the frozen formal feature
extractor.

No outcome information, motion-strength selection, interpolation, frame
repetition, or stitching is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline.clip_duration_robustness import (  # noqa: E402
    DEFAULT_PROPORTIONS_PCT,
    extract_proportional_duration_features,
)
from peristalsis_pipeline.dicom_curvature_calibration import (  # noqa: E402
    validate_dicom_curvature_rate_lineage,
)
from peristalsis_pipeline.formal_feature_extraction import (  # noqa: E402
    BIOLOGICAL_FEATURE_COLUMNS,
)
from peristalsis_pipeline.formal_qc_contract import (  # noqa: E402
    validate_case_id,
    validate_formal_qc_lineage,
    validate_matching_formal_qc_lineage,
)


FORMAL_READER_PATH = (
    PROJECT_ROOT
    / "代码"
    / "02_已并入主流程"
    / "05_输出审查与发布"
    / "run_feature_v1_formal_candidate_5case.py"
)
READER_SPEC = importlib.util.spec_from_file_location(
    "paper1_formal_candidate_reader", FORMAL_READER_PATH
)
if READER_SPEC is None or READER_SPEC.loader is None:
    raise ImportError(f"cannot load formal candidate reader: {FORMAL_READER_PATH}")
FORMAL_READER = importlib.util.module_from_spec(READER_SPEC)
sys.modules[READER_SPEC.name] = FORMAL_READER
READER_SPEC.loader.exec_module(FORMAL_READER)


DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "输出" / "新患者全流程_20260903"
DEFAULT_RSR_DIR = DEFAULT_PIPELINE_DIR / "10_rsr_fourway"
DEFAULT_ANATOMICAL_DIR = DEFAULT_PIPELINE_DIR / "11_anatomical_deformation"
DEFAULT_DICOM_DIR = PROJECT_ROOT / "输出" / "新患者DICOM物理曲率_20260904"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "输出" / "Paper1_时长比例鲁棒性"
DEFAULT_LONG_CSV = DEFAULT_OUTPUT_DIR / "clip_duration_proportional_f01_f20_long.csv"
DEFAULT_INVENTORY_CSV = DEFAULT_OUTPUT_DIR / "clip_duration_case_inventory.csv"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "clip_duration_manifest.json"

WINDOW_METADATA_COLUMNS = (
    "target_percent",
    "target_fraction",
    "actual_fraction",
    "full_pair_frames",
    "window_pair_frames",
    "full_pair_duration_s",
    "window_pair_duration_s",
    "window_start_pair_frame",
    "window_stop_pair_frame_exclusive",
    "window_start_s",
    "window_stop_s",
)
QC_COLUMNS = (
    "rsr_valid_ratio",
    "cavity_valid_ratio",
    "longitudinal_valid_ratio",
    "curvature_valid_ratio",
    "dicom_physical_curvature_available",
    "dicom_physical_curvature_rate_available",
)


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


def _load_case_inputs(
    *,
    rsr_path: Path,
    anatomical_dir: Path,
    dicom_dir: Path,
) -> dict[str, Any]:
    case_id = rsr_path.name[: -len(FORMAL_READER.RSR_SUFFIX)]

    rsr = FORMAL_READER.load_npz(rsr_path)
    FORMAL_READER.require_fields(
        rsr, FORMAL_READER.REQUIRED_RSR_FIELDS, path=rsr_path
    )
    validate_formal_qc_lineage(rsr, source=rsr_path)
    validate_case_id(rsr, expected_case_id=case_id, source=rsr_path)

    anatomical_path = anatomical_dir / (
        f"{case_id}{FORMAL_READER.ANATOMICAL_SUFFIX}"
    )
    if not anatomical_path.is_file():
        raise FileNotFoundError(
            f"formal anatomical NPZ not found for {case_id}: {anatomical_path}"
        )
    anatomical = FORMAL_READER.load_npz(anatomical_path)
    FORMAL_READER.require_fields(
        anatomical,
        FORMAL_READER.REQUIRED_ANATOMICAL_FIELDS,
        path=anatomical_path,
    )
    validate_formal_qc_lineage(anatomical, source=anatomical_path)
    validate_case_id(anatomical, expected_case_id=case_id, source=anatomical_path)
    validate_matching_formal_qc_lineage(
        rsr,
        anatomical,
        left_source=rsr_path,
        right_source=anatomical_path,
    )

    if FORMAL_READER.scalar_text(
        anatomical["evidence_layer"],
        field="evidence_layer",
        path=anatomical_path,
    ) != "roi_deformation_activity":
        raise ValueError(
            f"{anatomical_path}: evidence_layer is not roi_deformation_activity"
        )
    if not FORMAL_READER.scalar_bool(
        anatomical["qc_feature_branch_is_formal"],
        field="qc_feature_branch_is_formal",
        path=anatomical_path,
    ):
        raise ValueError(f"{anatomical_path}: anatomical branch is not formal QC")
    if FORMAL_READER.scalar_bool(
        anatomical["formal_propagation_released"],
        field="formal_propagation_released",
        path=anatomical_path,
    ):
        raise ValueError(
            f"{anatomical_path}: formal propagation must remain unreleased"
        )

    dicom_path = dicom_dir / f"{case_id}{FORMAL_READER.DICOM_SUFFIX}"
    if dicom_path.is_file():
        dicom = FORMAL_READER.load_npz(dicom_path)
        FORMAL_READER.require_fields(
            dicom, FORMAL_READER.REQUIRED_DICOM_FIELDS, path=dicom_path
        )
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
        curvature_rate_available = dicom["physical_curvature_rate_available"]
    else:
        dicom_path = None
        curvature_rate = None
        dicom_pair_valid = None
        curvature_available = False
        curvature_rate_available = False

    return {
        "case_id": case_id,
        "current_qc_radial_strain_rate_s": rsr["current_qc_radial_strain_rate_s"],
        "rsr_fps": rsr["fps"],
        "anatomical_pair_qc_valid": anatomical["pair_qc_valid"],
        "anatomical_fps": anatomical["fps"],
        "qc_cavity_width_strain_rate_s": anatomical[
            "qc_cavity_width_strain_rate_s"
        ],
        "cavity_qc_valid": anatomical["cavity_qc_valid"],
        "qc_longitudinal_wall_strain_rate_s": anatomical[
            "qc_longitudinal_wall_strain_rate_s"
        ],
        "longitudinal_qc_valid": anatomical["longitudinal_qc_valid"],
        "qc_wall_curvature_change_rate_mm_inv_s": curvature_rate,
        "dicom_pair_qc_valid": dicom_pair_valid,
        "physical_curvature_available": curvature_available,
        "physical_curvature_rate_available": curvature_rate_available,
        "_source_rsr": str(rsr_path.resolve()),
        "_source_anatomical": str(anatomical_path.resolve()),
        "_source_dicom": "" if dicom_path is None else str(dicom_path.resolve()),
    }


def _case_rows(
    *,
    rsr_path: Path,
    anatomical_dir: Path,
    dicom_dir: Path,
    proportions_pct: tuple[int, ...],
) -> list[dict[str, object]]:
    payload = _load_case_inputs(
        rsr_path=rsr_path,
        anatomical_dir=anatomical_dir,
        dicom_dir=dicom_dir,
    )
    payload.pop("_source_rsr")
    payload.pop("_source_anatomical")
    payload.pop("_source_dicom")
    return extract_proportional_duration_features(
        **payload,
        proportions_pct=proportions_pct,
    )


def _missing_feature_codes(row: pd.Series) -> str:
    codes = [
        f"F{index:02d}"
        for index, feature in enumerate(BIOLOGICAL_FEATURE_COLUMNS, start=1)
        if pd.isna(row[feature])
    ]
    return "|".join(codes)


def _build_inventory(long_table: pd.DataFrame) -> pd.DataFrame:
    full = long_table.loc[long_table["target_percent"] == 100].copy()
    if len(full) != long_table["case_id"].nunique():
        raise RuntimeError("each case must have exactly one 100% reference row")

    rows: list[dict[str, object]] = []
    for case_id, group in long_table.groupby("case_id", sort=True):
        group = group.sort_values("target_percent", ascending=False)
        reference = group.loc[group["target_percent"] == 100].iloc[0]
        row: dict[str, object] = {
            "case_id": case_id,
            "full_pair_frames": int(reference["full_pair_frames"]),
            "full_pair_duration_s": float(reference["full_pair_duration_s"]),
            "dicom_physical_curvature_available": bool(
                reference["dicom_physical_curvature_available"]
            ),
            "dicom_physical_curvature_rate_available": bool(
                reference["dicom_physical_curvature_rate_available"]
            ),
        }
        for item in group.itertuples(index=False):
            percent = int(item.target_percent)
            prefix = f"p{percent:03d}"
            row[f"{prefix}_window_pair_duration_s"] = float(item.window_pair_duration_s)
            row[f"{prefix}_actual_fraction"] = float(item.actual_fraction)
            row[f"{prefix}_rsr_valid_ratio"] = float(item.rsr_valid_ratio)
            row[f"{prefix}_cavity_valid_ratio"] = float(item.cavity_valid_ratio)
            row[f"{prefix}_longitudinal_valid_ratio"] = float(
                item.longitudinal_valid_ratio
            )
            row[f"{prefix}_curvature_valid_ratio"] = float(
                item.curvature_valid_ratio
            )
            row[f"{prefix}_missing_feature_codes"] = str(
                item.f01_f20_missing_codes
            )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rsr-dir", type=Path, default=DEFAULT_RSR_DIR)
    parser.add_argument("--anatomical-dir", type=Path, default=DEFAULT_ANATOMICAL_DIR)
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    parser.add_argument("--output-long-csv", type=Path, default=DEFAULT_LONG_CSV)
    parser.add_argument(
        "--output-inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--proportions",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROPORTIONS_PCT),
        help="Patient-specific duration percentages; must include 100.",
    )
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=None,
        help="Optional frozen Paper 1 proportional-truncation cohort size.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rsr_dir = args.rsr_dir.resolve()
    anatomical_dir = args.anatomical_dir.resolve()
    dicom_dir = args.dicom_dir.resolve()
    for label, directory in (
        ("formal RSR", rsr_dir),
        ("formal anatomical", anatomical_dir),
        ("formal DICOM", dicom_dir),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {directory}")

    proportions = tuple(int(x) for x in args.proportions)
    rsr_paths = sorted(rsr_dir.glob(f"*{FORMAL_READER.RSR_SUFFIX}"))
    if not rsr_paths:
        raise FileNotFoundError(f"no formal RSR NPZ found in: {rsr_dir}")
    if args.expected_case_count is not None and len(rsr_paths) != args.expected_case_count:
        raise ValueError(
            f"expected exactly {args.expected_case_count} formal RSR cases, "
            f"found {len(rsr_paths)}"
        )

    output_long = args.output_long_csv.resolve()
    output_inventory = args.output_inventory_csv.resolve()
    manifest_path = args.manifest.resolve()
    for path in (output_long, output_inventory, manifest_path):
        if path.exists() and not args.allow_overwrite:
            raise FileExistsError(f"refusing to overwrite existing output: {path}")

    rows: list[dict[str, object]] = []
    for rsr_path in rsr_paths:
        rows.extend(
            _case_rows(
                rsr_path=rsr_path,
                anatomical_dir=anatomical_dir,
                dicom_dir=dicom_dir,
                proportions_pct=proportions,
            )
        )

    long_table = pd.DataFrame(rows)
    expected_rows = len(rsr_paths) * len(proportions)
    if len(long_table) != expected_rows:
        raise RuntimeError(
            f"unexpected proportional table row count: {len(long_table)} "
            f"!= {expected_rows}"
        )
    if long_table.duplicated(["case_id", "target_percent"]).any():
        raise RuntimeError("case_id + target_percent must be unique")

    long_table["f01_f20_missing_codes"] = long_table.apply(
        _missing_feature_codes, axis=1
    )
    long_table["f01_f20_all_evaluable"] = (
        long_table["f01_f20_missing_codes"] == ""
    )

    inventory = _build_inventory(long_table)

    output_long.parent.mkdir(parents=True, exist_ok=True)
    output_inventory.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    long_table.to_csv(output_long, index=False, encoding="utf-8-sig", na_rep="NA")
    inventory.to_csv(
        output_inventory, index=False, encoding="utf-8-sig", na_rep="NA"
    )

    durations = pd.to_numeric(inventory["full_pair_duration_s"], errors="coerce")
    manifest = {
        "status": "paper1_proportional_clip_duration_table_created",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain"),
        "case_count": len(inventory),
        "proportions_pct": list(sorted(proportions, reverse=True)),
        "window_rule": (
            "100% = complete formal pair-frame sequence; shorter windows are "
            "deterministic centered nested slices; floor frame count; no "
            "selection by motion/QC/feature/outcome; no repetition/interpolation/stitching"
        ),
        "feature_extractor": (
            "peristalsis_pipeline.formal_feature_extraction."
            "extract_formal_candidate_case_features"
        ),
        "clip_module_sha256": sha256(
            CODE_ROOT
            / "peristalsis_pipeline"
            / "clip_duration_robustness.py"
        ),
        "formal_feature_extraction_sha256": sha256(
            CODE_ROOT
            / "peristalsis_pipeline"
            / "formal_feature_extraction.py"
        ),
        "input_directories": {
            "formal_rsr": str(rsr_dir),
            "formal_anatomical_deformation": str(anatomical_dir),
            "formal_dicom_curvature": str(dicom_dir),
        },
        "full_pair_duration_s_summary": {
            "min": float(durations.min()),
            "median": float(durations.median()),
            "max": float(durations.max()),
        },
        "output_long_csv": str(output_long),
        "output_inventory_csv": str(output_inventory),
        "output_long_csv_sha256": sha256(output_long),
        "output_inventory_csv_sha256": sha256(output_inventory),
        "outcome_information_used": False,
        "tracking_rerun": False,
        "feature_definition_changed": False,
        "new_qc_threshold_created": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
