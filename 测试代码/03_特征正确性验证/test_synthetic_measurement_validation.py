from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "代码" / "01_底层算法"))
sys.path.insert(0, str(PROJECT_ROOT / "代码" / "03_实验与历史代码"))

from paper1_feature_validity.synthetic_measurement_validation import (
    CURVATURE_REGION,
    SCENARIOS,
    _tracking_payload,
    make_synthetic_geometry,
    run_all_synthetic_cases,
)
from peristalsis_pipeline.dicom_curvature_calibration import (
    curvature_change_rate_from_pair_dt,
)
from peristalsis_pipeline.formal_feature_extraction import (
    BIOLOGICAL_FEATURE_COLUMNS,
    extract_formal_candidate_case_features,
)


def test_all_requested_scenarios_cover_the_frozen_contract() -> None:
    payload = run_all_synthetic_cases()
    assert payload["summary_status"] == "PASS"
    assert payload["scenario_count"] == len(SCENARIOS)
    assert payload["feature_count"] == 20
    assert all(row["formal_contract_20_features"] for row in payload["results"])


def test_masked_positions_are_not_reintroduced_by_feature_aggregation() -> None:
    geometry = make_synthetic_geometry("local_radial_change")
    geometry.pair_valid[1, 0, 2] = False
    tracking = _tracking_payload(geometry)
    from peristalsis_pipeline.anatomical_deformation_features import (
        anatomical_deformation_from_tracking,
    )
    from peristalsis_pipeline.tracking_huang_radial_pairs import (
        radial_deformation_from_tracking,
    )

    radial = radial_deformation_from_tracking(tracking, 7, geometry.fps)
    deformation = anatomical_deformation_from_tracking(
        tracking, section_count=7, fps=geometry.fps
    )
    assert np.isnan(radial["radial_strain_rate_s"][1, 0, 2])
    assert np.isnan(deformation["cavity_width_strain_rate_s"][1, 2])
    assert np.isnan(deformation["longitudinal_wall_strain_rate_s"][1, 0, 1])


def test_missing_physical_calibration_is_reported_as_missing_not_pixel_units() -> None:
    geometry = make_synthetic_geometry("curvature_change")
    tracking = _tracking_payload(geometry)
    from peristalsis_pipeline.anatomical_deformation_features import (
        anatomical_deformation_from_tracking,
    )
    from peristalsis_pipeline.tracking_huang_radial_pairs import (
        radial_deformation_from_tracking,
    )

    radial = radial_deformation_from_tracking(tracking, 7, geometry.fps)[
        "radial_strain_rate_s"
    ]
    deformation = anatomical_deformation_from_tracking(
        tracking, section_count=7, fps=geometry.fps
    )
    row = extract_formal_candidate_case_features(
        case_id="synthetic_missing_calibration",
        current_qc_radial_strain_rate_s=radial,
        rsr_fps=geometry.fps,
        anatomical_pair_qc_valid=np.isfinite(radial),
        anatomical_fps=geometry.fps,
        qc_cavity_width_strain_rate_s=deformation["cavity_width_strain_rate_s"],
        cavity_qc_valid=deformation["cavity_feature_valid"],
        qc_longitudinal_wall_strain_rate_s=deformation[
            "longitudinal_wall_strain_rate_s"
        ],
        longitudinal_qc_valid=deformation["longitudinal_feature_valid"],
        physical_curvature_available=False,
        physical_curvature_rate_available=False,
    )
    assert all(np.isnan(row[name]) for name in BIOLOGICAL_FEATURE_COLUMNS[14:])


def test_pair_specific_time_unit_conversion_is_independent_of_frozen_fps_check() -> None:
    change = np.asarray([np.nan, 0.02, 0.02], dtype=np.float32).reshape(3, 1, 1)
    rate = curvature_change_rate_from_pair_dt(
        change,
        np.asarray([np.nan, 0.05, 0.10], dtype=np.float64),
        np.asarray([False, True, True]),
    )
    assert np.allclose(rate[:, 0, 0], [np.nan, 0.4, 0.2], equal_nan=True)
    # The full physical path separately requires timing to match 1/fps when
    # timing_matched=True; this test documents the lower-level pair-time unit.
