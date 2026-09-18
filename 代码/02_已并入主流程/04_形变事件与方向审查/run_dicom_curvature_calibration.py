#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate an independent DICOM-calibrated curvature NPZ for one case.

This runner is intentionally not called by the formal project CMD.  It requires
an externally verified DICOM/MP4 match and analysis-to-DICOM affine transform.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline.dicom_curvature_calibration import (
    DICOM_CURVATURE_RATE_TIMEBASE,
    physical_curvature_from_wall_centers,
    read_dicom_cine_metadata,
)
from peristalsis_pipeline.formal_qc_contract import (
    FORMAL_QC_LINEAGE_ID_FIELD,
    FORMAL_QC_VERSION_FIELD,
    validate_case_id,
    validate_formal_qc_lineage,
)


MATCHED = "matched"
MAPPING_EXAMPLE = """verified mapping JSON example:
{
  "case_id": "CASE_ID",
  "dicom_mp4_match_status": "matched",
  "frame_mapping_status": "matched",
  "coordinate_mapping_status": "matched",
  "timing_match_status": "matched",
  "selected_us_region_index": 0,
  "analysis_to_dicom_transform": [
    [2.0, 0.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 1.0]
  ]
}
"""


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_verified_mapping(path: Path, case_id: str) -> dict[str, object]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("case_id") != case_id:
        raise ValueError("calibration mapping case_id does not match the command")
    required_statuses = (
        "dicom_mp4_match_status",
        "frame_mapping_status",
        "coordinate_mapping_status",
        "timing_match_status",
    )
    missing = [name for name in required_statuses if name not in mapping]
    if missing:
        raise ValueError(f"calibration mapping is missing statuses: {missing}")
    for name in required_statuses[:3]:
        if mapping[name] != MATCHED:
            raise ValueError(f"{name} must be matched before physical curvature")
    if "selected_us_region_index" not in mapping:
        raise ValueError("calibration mapping is missing selected_us_region_index")
    if "analysis_to_dicom_transform" not in mapping:
        raise ValueError("calibration mapping is missing analysis_to_dicom_transform")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=MAPPING_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dicom-file", action="append", type=Path, required=True)
    parser.add_argument("--anatomical-npz", type=Path, required=True)
    parser.add_argument("--verified-mapping-json", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_npz.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_npz}")

    anatomical = load_npz(args.anatomical_npz)
    validate_formal_qc_lineage(anatomical, source=args.anatomical_npz)
    validate_case_id(
        anatomical,
        expected_case_id=args.case_id,
        source=args.anatomical_npz,
    )
    if str(anatomical.get("evidence_layer", "")) != "roi_deformation_activity":
        raise ValueError("anatomical input is not ROI deformation activity")
    if bool(anatomical.get("formal_propagation_released", True)):
        raise ValueError("physical curvature input must not release propagation")

    mapping = load_verified_mapping(args.verified_mapping_json, args.case_id)
    metadata = read_dicom_cine_metadata(args.dicom_file)
    for field, actual in (
        ("study_instance_uid", metadata.study_instance_uid),
        ("series_instance_uid", metadata.series_instance_uid),
    ):
        expected = mapping.get(field)
        if expected is not None and str(expected) != actual:
            raise ValueError(f"verified mapping {field} does not match DICOM")
    frame_count = len(anatomical["wall_center_source"])
    if metadata.frame_count != frame_count:
        raise ValueError("verified DICOM/MP4 frame mapping has inconsistent frame count")

    region_index = int(mapping["selected_us_region_index"])
    regions = {region.index: region for region in metadata.regions}
    if region_index not in regions:
        raise ValueError("selected ultrasound region is not spatially calibrated")
    region = regions[region_index]
    timing_matched = mapping["timing_match_status"] == MATCHED
    calibrated = physical_curvature_from_wall_centers(
        wall_center_source=anatomical["wall_center_source"],
        wall_center_measured=anatomical["wall_center_measured"],
        pair_qc_valid=anatomical["pair_qc_valid"],
        analysis_to_dicom_transform=np.asarray(
            mapping["analysis_to_dicom_transform"], dtype=np.float64
        ),
        region=region,
        fps=float(anatomical["fps"]),
        timing_matched=timing_matched,
        pair_dt_s=metadata.pair_dt_s,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        **calibrated,
        case_id=np.asarray(args.case_id),
        current_qc_semantics=anatomical["current_qc_semantics"],
        **{
            FORMAL_QC_VERSION_FIELD: anatomical[FORMAL_QC_VERSION_FIELD],
            FORMAL_QC_LINEAGE_ID_FIELD: anatomical[FORMAL_QC_LINEAGE_ID_FIELD],
        },
        pair_qc_valid=anatomical["pair_qc_valid"],
        normalized_cervix_to_fundus_position=anatomical[
            "normalized_cervix_to_fundus_position"
        ],
        analysis_to_dicom_transform=np.asarray(
            mapping["analysis_to_dicom_transform"], dtype=np.float64
        ),
        dicom_pair_dt_s=metadata.pair_dt_s,
        study_instance_uid=np.asarray(metadata.study_instance_uid),
        series_instance_uid=np.asarray(metadata.series_instance_uid),
        dicom_frame_count=np.asarray(metadata.frame_count, dtype=np.int32),
        dicom_rows=np.asarray(metadata.rows, dtype=np.int32),
        dicom_columns=np.asarray(metadata.columns, dtype=np.int32),
        selected_us_region_index=np.asarray(region.index, dtype=np.int32),
        region_min_x=np.asarray(region.min_x, dtype=np.int32),
        region_min_y=np.asarray(region.min_y, dtype=np.int32),
        region_max_x=np.asarray(region.max_x, dtype=np.int32),
        region_max_y=np.asarray(region.max_y, dtype=np.int32),
        physical_units_x_raw=np.asarray(region.physical_units_x, dtype=np.int32),
        physical_units_y_raw=np.asarray(region.physical_units_y, dtype=np.int32),
        physical_delta_x_raw=np.asarray(region.physical_delta_x, dtype=np.float64),
        physical_delta_y_raw=np.asarray(region.physical_delta_y, dtype=np.float64),
        dicom_delta_x_mm_per_pixel=np.asarray(
            region.delta_x_mm_per_pixel, dtype=np.float32
        ),
        dicom_delta_y_mm_per_pixel=np.asarray(
            region.delta_y_mm_per_pixel, dtype=np.float32
        ),
        dicom_mp4_match_status=np.asarray(mapping["dicom_mp4_match_status"]),
        frame_mapping_status=np.asarray(mapping["frame_mapping_status"]),
        coordinate_mapping_status=np.asarray(mapping["coordinate_mapping_status"]),
        timing_match_status=np.asarray(mapping["timing_match_status"]),
        calibration_region_status=np.asarray(MATCHED),
        physical_curvature_available=np.asarray(True),
        physical_curvature_rate_available=np.asarray(timing_matched),
        curvature_rate_timebase=np.asarray(DICOM_CURVATURE_RATE_TIMEBASE),
        dicom_timing_numeric_match=np.asarray(timing_matched),
        evidence_layer=np.asarray("roi_deformation_activity"),
        formal_propagation_released=np.asarray(False),
        calibration_changes_formal_qc=np.asarray(False),
    )
    print(f"wrote independent DICOM curvature calibration: {args.output_npz}")


if __name__ == "__main__":
    main()
