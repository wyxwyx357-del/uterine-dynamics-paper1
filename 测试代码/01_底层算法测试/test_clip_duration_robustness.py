from __future__ import annotations

import numpy as np
import pytest

from peristalsis_pipeline.clip_duration_robustness import (
    centered_nested_windows,
    extract_duration_features,
)


def _synthetic_payload(frame_count: int = 100, section_count: int = 4):
    fps = np.asarray(1.0)
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


def test_centered_nested_windows_are_deterministic_and_nested():
    windows = centered_nested_windows(
        pair_frame_count=100,
        fps=1.0,
        durations_s=(60, 30, 20, 10),
    )
    by_duration = {w.duration_s: w for w in windows}

    assert (by_duration[60].start, by_duration[60].stop) == (20, 80)
    assert (by_duration[30].start, by_duration[30].stop) == (35, 65)
    assert (by_duration[20].start, by_duration[20].stop) == (40, 60)
    assert (by_duration[10].start, by_duration[10].stop) == (45, 55)

    ref = by_duration[60]
    for duration in (30, 20, 10):
        current = by_duration[duration]
        assert ref.start <= current.start < current.stop <= ref.stop


def test_centered_nested_windows_rejects_short_reference():
    with pytest.raises(ValueError, match="insufficient pair-frame duration"):
        centered_nested_windows(
            pair_frame_count=59,
            fps=1.0,
            durations_s=(60, 30, 20, 10),
        )


def test_duration_extraction_reuses_frozen_feature_function():
    payload = _synthetic_payload()
    rows = extract_duration_features(
        **payload,
        durations_s=(60, 30, 20, 10),
    )

    assert [row["duration_s"] for row in rows] == [60, 30, 20, 10]
    assert [row["window_pair_frames"] for row in rows] == [60, 30, 20, 10]

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

    rows = extract_duration_features(
        **payload,
        durations_s=(60, 30),
    )

    for row in rows:
        assert np.isnan(row["wall_curvature_change_rate_mm_inv_s_abs_median"])
        assert np.isnan(row["wall_curvature_change_rate_mm_inv_s_abs_p95"])
        assert row["dicom_physical_curvature_available"] is False
        assert row["dicom_physical_curvature_rate_available"] is False
