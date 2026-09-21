from __future__ import annotations

import numpy as np
import pytest

from peristalsis_pipeline.clip_duration_robustness import (
    centered_proportional_windows,
    extract_proportional_duration_features,
)


def _synthetic_payload(frame_count: int = 100, section_count: int = 4):
    fps = np.asarray(2.0)
    rsr = np.arange(frame_count * 2 * section_count, dtype=float).reshape(
        frame_count, 2, section_count
    )
    rsr[5, 0, 0] = np.nan
    pair_valid = np.isfinite(rsr)

    cavity = np.arange(frame_count * section_count, dtype=float).reshape(
        frame_count, section_count
    )
    cavity_valid = np.ones_like(cavity, dtype=bool)
    cavity_valid[7, 0] = False

    longitudinal = np.arange(
        frame_count * 2 * (section_count - 1), dtype=float
    ).reshape(frame_count, 2, section_count - 1)
    longitudinal_valid = np.ones_like(longitudinal, dtype=bool)
    longitudinal_valid[8, 0, 0] = False

    curvature = (rsr * 0.01).copy()
    dicom_pair_valid = pair_valid.copy()

    return {
        "case_id": "CASE_SYNTH",
        "current_qc_radial_strain_rate_s": rsr,
        "rsr_fps": fps,
        "anatomical_pair_qc_valid": pair_valid,
        "anatomical_fps": fps,
        "qc_cavity_width_strain_rate_s": cavity,
        "cavity_qc_valid": cavity_valid,
        "qc_longitudinal_wall_strain_rate_s": longitudinal,
        "longitudinal_qc_valid": longitudinal_valid,
        "qc_wall_curvature_change_rate_mm_inv_s": curvature,
        "dicom_pair_qc_valid": dicom_pair_valid,
        "physical_curvature_available": np.asarray(True),
        "physical_curvature_rate_available": np.asarray(True),
    }


def test_centered_proportional_windows_are_deterministic_and_nested():
    windows = centered_proportional_windows(
        pair_frame_count=100,
        fps=2.0,
        proportions_pct=(100, 75, 50, 25),
    )
    by_percent = {w.target_percent: w for w in windows}

    assert (by_percent[100].start, by_percent[100].stop) == (0, 100)
    assert (by_percent[75].start, by_percent[75].stop) == (12, 87)
    assert (by_percent[50].start, by_percent[50].stop) == (25, 75)
    assert (by_percent[25].start, by_percent[25].stop) == (37, 62)

    assert by_percent[100].actual_pair_duration_s == 50.0
    assert by_percent[75].actual_pair_duration_s == 37.5
    assert by_percent[50].actual_pair_duration_s == 25.0
    assert by_percent[25].actual_pair_duration_s == 12.5

    ref = by_percent[100]
    for percent in (75, 50, 25):
        current = by_percent[percent]
        assert ref.start <= current.start < current.stop <= ref.stop


def test_proportional_windows_use_floor_for_non_divisible_lengths():
    windows = centered_proportional_windows(
        pair_frame_count=101,
        fps=1.0,
        proportions_pct=(100, 75, 50, 25),
    )
    counts = {w.target_percent: w.sample_count for w in windows}

    assert counts == {100: 101, 75: 75, 50: 50, 25: 25}
    for window in windows:
        assert window.actual_fraction <= window.target_percent / 100.0 + 1e-12


def test_proportional_windows_require_100_percent_reference():
    with pytest.raises(ValueError, match="must include the 100% reference"):
        centered_proportional_windows(
            pair_frame_count=100,
            fps=1.0,
            proportions_pct=(75, 50, 25),
        )


def test_duration_extraction_reuses_frozen_feature_function():
    payload = _synthetic_payload()
    rows = extract_proportional_duration_features(
        **payload,
        proportions_pct=(100, 75, 50, 25),
    )

    assert [row["target_percent"] for row in rows] == [100, 75, 50, 25]
    assert [row["window_pair_frames"] for row in rows] == [100, 75, 50, 25]
    assert [row["window_pair_duration_s"] for row in rows] == [50.0, 37.5, 25.0, 12.5]

    for row in rows:
        assert row["case_id"] == "CASE_SYNTH"
        assert np.isfinite(row["rsr_abs_median"])
        assert np.isfinite(row["cavity_width_strain_rate_abs_median"])
        assert np.isfinite(row["longitudinal_wall_strain_rate_abs_median"])
        assert np.isfinite(row["wall_curvature_change_rate_mm_inv_s_abs_median"])


def test_duration_extraction_preserves_missing_curvature_semantics():
    payload = _synthetic_payload()
    payload["qc_wall_curvature_change_rate_mm_inv_s"] = None
    payload["dicom_pair_qc_valid"] = None
    payload["physical_curvature_available"] = np.asarray(False)
    payload["physical_curvature_rate_available"] = np.asarray(False)

    rows = extract_proportional_duration_features(
        **payload,
        proportions_pct=(100, 50),
    )

    for row in rows:
        assert np.isnan(row["wall_curvature_change_rate_mm_inv_s_abs_median"])
        assert np.isnan(row["wall_curvature_change_rate_mm_inv_s_abs_p95"])
        assert row["dicom_physical_curvature_available"] is False
        assert row["dicom_physical_curvature_rate_available"] is False


def test_duration_extraction_rejects_mismatched_time_axes():
    payload = _synthetic_payload()
    payload["cavity_qc_valid"] = payload["cavity_qc_valid"][:-1]

    with pytest.raises(ValueError, match="must share the full pair-frame axis"):
        extract_proportional_duration_features(**payload)
