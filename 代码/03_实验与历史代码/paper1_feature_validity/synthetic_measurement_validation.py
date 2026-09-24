from __future__ import annotations

"""Synthetic validation for the frozen Paper 1 F01-F20 measurement contract.

The scenario generator and the oracle functions in this module intentionally do
not call the frozen measurement helpers for their expected values.  The frozen
implementation is called only for the measured side of each comparison.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from peristalsis_pipeline.anatomical_deformation_features import (
    anatomical_deformation_from_tracking,
)
from peristalsis_pipeline.dicom_curvature_calibration import (
    UltrasoundRegionCalibration,
    physical_curvature_from_wall_centers,
)
from peristalsis_pipeline.formal_feature_extraction import (
    BIOLOGICAL_FEATURE_COLUMNS,
    extract_formal_candidate_case_features,
)
from peristalsis_pipeline.tracking_huang_radial_pairs import radial_deformation_from_tracking


FPS = 10.0
CURVATURE_REGION = UltrasoundRegionCalibration(
    index=0,
    min_x=0,
    min_y=0,
    max_x=120,
    max_y=120,
    physical_units_x=3,
    physical_units_y=3,
    physical_delta_x=0.1,
    physical_delta_y=0.1,
)
ANALYSIS_TO_DICOM = np.eye(3, dtype=np.float64)
SCENARIOS = (
    "static",
    "translation",
    "rotation",
    "uniform_scaling",
    "local_radial_change",
    "cavity_opening",
    "cavity_closing",
    "along_wall_stretch",
    "along_wall_compression",
    "curvature_change",
)


@dataclass(frozen=True)
class SyntheticGeometry:
    scenario: str
    source_rails: np.ndarray
    measured_rails: np.ndarray
    source_centers: np.ndarray
    measured_centers: np.ndarray
    pair_valid: np.ndarray
    fps: float


def _curve_curvature_oracle(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Independent three-point curvature oracle in the input coordinate unit."""

    coordinates = np.asarray(points, dtype=np.float64)
    point_valid = np.asarray(valid, dtype=bool)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("points must have shape (frame, side, section, 2)")
    result = np.full(coordinates.shape[:-1], np.nan, dtype=np.float64)
    for frame in range(coordinates.shape[0]):
        for side in range(coordinates.shape[1]):
            for section in range(1, coordinates.shape[2] - 1):
                if not np.all(point_valid[frame, side, section - 1 : section + 2]):
                    continue
                a, b, c = coordinates[frame, side, section - 1 : section + 2]
                ab = b - a
                bc = c - b
                ac = c - a
                denominator = (
                    np.linalg.norm(ab)
                    * np.linalg.norm(bc)
                    * np.linalg.norm(ac)
                )
                if denominator <= 1e-12:
                    continue
                cross = abs(ab[0] * ac[1] - ab[1] * ac[0])
                result[frame, side, section] = 2.0 * cross / denominator
    return result


def _finite_abs_stats(values: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(array)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    selected = np.abs(array[mask])
    if selected.size == 0:
        return {"median": float("nan"), "p95": float("nan")}
    return {
        "median": float(np.median(selected)),
        "p95": float(np.percentile(selected, 95)),
    }


def _radial_rate_oracle(
    source_inner: np.ndarray,
    source_outer: np.ndarray,
    measured_inner: np.ndarray,
    measured_outer: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Independent radial-rate oracle; do not call the frozen geometry helper."""

    source_length = np.linalg.norm(
        np.asarray(source_outer, dtype=np.float64)
        - np.asarray(source_inner, dtype=np.float64),
        axis=-1,
    )
    measured_length = np.linalg.norm(
        np.asarray(measured_outer, dtype=np.float64)
        - np.asarray(measured_inner, dtype=np.float64),
        axis=-1,
    )
    result = np.full(source_length.shape, np.nan, dtype=np.float64)
    valid = (source_length > 1e-12) & np.isfinite(source_length) & np.isfinite(measured_length)
    result[valid] = (measured_length[valid] - source_length[valid]) / source_length[valid] * float(fps)
    return result


def _wall_centers(section_count: int = 7) -> np.ndarray:
    x = np.linspace(20.0, 100.0, section_count)
    curvature = 0.004 * (x - 60.0) ** 2
    anterior = np.column_stack((x, 40.0 + curvature))
    posterior = np.column_stack((x, 80.0 - 0.003 * (x - 60.0) ** 2))
    return np.stack((anterior, posterior), axis=0).astype(np.float32)


def _rails_from_centers(centers: np.ndarray, radial_scale: float = 1.0) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float32)
    section_count = centers.shape[1]
    outward = np.asarray(((0.0, -1.0), (0.0, 1.0)), dtype=np.float32)
    offset = 4.0 * float(radial_scale)
    rails = []
    for side in range(2):
        half = outward[side] * (offset / 2.0)
        rails.extend((centers[side] - half, centers[side] + half))
    return np.concatenate(rails, axis=0).reshape(4, section_count, 2)


def _rotate(points: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))),
        dtype=np.float64,
    )
    return ((np.asarray(points, dtype=np.float64) - center) @ rotation.T + center).astype(
        np.float32
    )


def make_synthetic_geometry(scenario: str, *, fps: float = FPS) -> SyntheticGeometry:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    source_centers = _wall_centers()
    measured_centers = source_centers.copy()
    source_scale = 1.0
    measured_scale = 1.0
    if scenario == "translation":
        measured_centers = measured_centers + np.asarray((3.0, -2.0), dtype=np.float32)
    elif scenario == "rotation":
        measured_centers = _rotate(measured_centers, 0.08, np.asarray((60.0, 60.0)))
    elif scenario == "uniform_scaling":
        measured_centers = ((measured_centers - 60.0) * 1.08 + 60.0).astype(np.float32)
        measured_scale = 1.08
    elif scenario == "local_radial_change":
        measured_scale = 1.18
    elif scenario == "cavity_opening":
        measured_centers = measured_centers.copy()
        midline = 60.0
        measured_centers[:, :, 1] = midline + (measured_centers[:, :, 1] - midline) * 1.12
    elif scenario == "cavity_closing":
        measured_centers = measured_centers.copy()
        midline = 60.0
        measured_centers[:, :, 1] = midline + (measured_centers[:, :, 1] - midline) * 0.88
    elif scenario == "along_wall_stretch":
        measured_centers = measured_centers.copy()
        measured_centers[:, :, 0] = 60.0 + (measured_centers[:, :, 0] - 60.0) * 1.12
    elif scenario == "along_wall_compression":
        measured_centers = measured_centers.copy()
        measured_centers[:, :, 0] = 60.0 + (measured_centers[:, :, 0] - 60.0) * 0.88
    elif scenario == "curvature_change":
        measured_centers = measured_centers.copy()
        x = measured_centers[:, :, 0]
        measured_centers[0, :, 1] = 40.0 + 0.006 * (x[0] - 60.0) ** 2
        measured_centers[1, :, 1] = 80.0 - 0.005 * (x[1] - 60.0) ** 2
    if scenario == "local_radial_change":
        measured_rails = _rails_from_centers(measured_centers, measured_scale)
    elif scenario == "uniform_scaling":
        measured_rails = ((
            _rails_from_centers(source_centers) - 60.0
        ) * measured_scale + 60.0).astype(np.float32)
    else:
        measured_rails = _rails_from_centers(measured_centers, measured_scale)
    source_rails = _rails_from_centers(source_centers, source_scale)

    source = source_rails.reshape(4 * source_rails.shape[1], 2)
    measured = measured_rails.reshape(4 * measured_rails.shape[1], 2)
    source_frames = np.stack((source, source), axis=0)
    measured_frames = np.stack((source, measured), axis=0)
    pair_valid = np.ones((2, 2, source_rails.shape[1]), dtype=bool)
    pair_valid[0] = False
    return SyntheticGeometry(
        scenario=scenario,
        source_rails=source_frames,
        measured_rails=measured_frames,
        source_centers=np.stack((source_centers, source_centers), axis=0),
        measured_centers=np.stack((source_centers, measured_centers), axis=0),
        pair_valid=pair_valid,
        fps=float(fps),
    )


def _tracking_payload(geometry: SyntheticGeometry) -> dict[str, np.ndarray]:
    return {
        "radial_pair_source": geometry.source_rails,
        "radial_lk_measured": geometry.measured_rails,
        "radial_global_prediction": geometry.measured_rails,
        "radial_pair_valid": geometry.pair_valid,
        "radial_pcc": np.ones(geometry.source_rails.shape[:2], dtype=np.float32),
        "radial_fb_error_px": np.zeros(geometry.source_rails.shape[:2], dtype=np.float32),
    }


def _oracle_outputs(geometry: SyntheticGeometry) -> dict[str, np.ndarray]:
    source = geometry.source_rails
    measured = geometry.measured_rails
    centers_source = geometry.source_centers
    centers_measured = geometry.measured_centers
    valid = geometry.pair_valid.copy()
    valid[0] = False
    source_inner = source[:, :7]
    source_outer = source[:, 7:14]
    measured_inner = measured[:, :7]
    measured_outer = measured[:, 7:14]
    rsr = _radial_rate_oracle(
        source_inner, source_outer, measured_inner, measured_outer, geometry.fps
    )
    rsr = np.stack((rsr, _radial_rate_oracle(
        source[:, 14:21], source[:, 21:28], measured[:, 14:21], measured[:, 21:28], geometry.fps
    )), axis=1)
    rsr[~valid] = np.nan
    cavity_source = centers_source[:, 1] - centers_source[:, 0]
    cavity_measured = centers_measured[:, 1] - centers_measured[:, 0]
    cavity = np.linalg.norm(cavity_measured, axis=-1) - np.linalg.norm(cavity_source, axis=-1)
    cavity = cavity * geometry.fps / np.linalg.norm(cavity_source, axis=-1)
    cavity[~(valid[:, 0] & valid[:, 1])] = np.nan
    longitudinal = np.full((2, 2, 6), np.nan, dtype=np.float64)
    for side in range(2):
        source_seg = np.diff(centers_source[:, side], axis=1)
        measured_seg = np.diff(centers_measured[:, side], axis=1)
        source_len = np.linalg.norm(source_seg, axis=-1)
        measured_len = np.linalg.norm(measured_seg, axis=-1)
        longitudinal[:, side] = (measured_len - source_len) * geometry.fps / source_len
    longitudinal[~(valid[:, :, :-1] & valid[:, :, 1:])] = np.nan
    source_curve = _curve_curvature_oracle(centers_source, valid)
    measured_curve = _curve_curvature_oracle(centers_measured, valid)
    curvature_px = (measured_curve - source_curve) * geometry.fps
    curvature_px[~valid] = np.nan
    return {"rsr": rsr, "cavity": cavity, "longitudinal": longitudinal, "curvature_px": curvature_px}


def run_synthetic_case(scenario: str, *, fps: float = FPS) -> dict[str, Any]:
    geometry = make_synthetic_geometry(scenario, fps=fps)
    tracking = _tracking_payload(geometry)
    measured_deformation = anatomical_deformation_from_tracking(
        tracking, section_count=7, fps=fps
    )
    measured_rsr = radial_deformation_from_tracking(tracking, anterior_count=7, fps=fps)[
        "radial_strain_rate_s"
    ]
    measured_physical = physical_curvature_from_wall_centers(
        wall_center_source=geometry.source_centers,
        wall_center_measured=geometry.measured_centers,
        pair_qc_valid=geometry.pair_valid,
        analysis_to_dicom_transform=ANALYSIS_TO_DICOM,
        region=CURVATURE_REGION,
        fps=fps,
        timing_matched=True,
        pair_dt_s=np.asarray((np.nan, 1.0 / fps), dtype=np.float64),
    )
    oracle = _oracle_outputs(geometry)
    # The region is isotropic in this validation, so one scalar conversion is enough.
    oracle_physical = (
        _curve_curvature_oracle(
            geometry.source_centers * CURVATURE_REGION.delta_x_mm_per_pixel,
            geometry.pair_valid,
        ),
        _curve_curvature_oracle(
            geometry.measured_centers * CURVATURE_REGION.delta_x_mm_per_pixel,
            geometry.pair_valid,
        ),
    )
    expected_physical = (oracle_physical[1] - oracle_physical[0]) * fps
    expected_physical[~geometry.pair_valid] = np.nan
    checks = {
        "rsr": np.allclose(measured_rsr, oracle["rsr"], equal_nan=True, rtol=2e-4, atol=2e-5),
        "cavity": np.allclose(
            measured_deformation["cavity_width_strain_rate_s"],
            oracle["cavity"], equal_nan=True, rtol=2e-4, atol=2e-5,
        ),
        "longitudinal": np.allclose(
            measured_deformation["longitudinal_wall_strain_rate_s"],
            oracle["longitudinal"], equal_nan=True, rtol=2e-4, atol=2e-5,
        ),
        "pixel_curvature": np.allclose(
            measured_deformation["wall_curvature_change_rate_px_inv_s"],
            oracle["curvature_px"], equal_nan=True, rtol=3e-4, atol=3e-5,
        ),
        "physical_curvature": np.allclose(
            measured_physical["qc_wall_curvature_change_rate_mm_inv_s"],
            expected_physical, equal_nan=True, rtol=3e-4, atol=3e-5,
        ),
    }
    formal = extract_formal_candidate_case_features(
        case_id=f"synthetic_{scenario}",
        current_qc_radial_strain_rate_s=measured_rsr,
        rsr_fps=fps,
        anatomical_pair_qc_valid=np.isfinite(measured_rsr),
        anatomical_fps=fps,
        qc_cavity_width_strain_rate_s=measured_deformation["cavity_width_strain_rate_s"],
        cavity_qc_valid=measured_deformation["cavity_feature_valid"],
        qc_longitudinal_wall_strain_rate_s=measured_deformation["longitudinal_wall_strain_rate_s"],
        longitudinal_qc_valid=measured_deformation["longitudinal_feature_valid"],
        qc_wall_curvature_change_rate_mm_inv_s=measured_physical[
            "qc_wall_curvature_change_rate_mm_inv_s"
        ],
        dicom_pair_qc_valid=np.isfinite(measured_rsr),
        physical_curvature_available=True,
        physical_curvature_rate_available=True,
    )
    formal_keys_present = set(BIOLOGICAL_FEATURE_COLUMNS).issubset(formal)
    return {
        "scenario": scenario,
        "status": "PASS" if all(checks.values()) and formal_keys_present else "FAIL",
        "checks": checks,
        "formal_contract_20_features": formal_keys_present,
        "limitations": [
            "This test validates geometry and aggregation from supplied point tracks; it does not validate LK recovery from image texture.",
            "The radial-pair semantics do not prove long-term tissue identity across frames.",
            "Curvature truth is the independent three-point curvature definition, not a continuous-curve curvature claim.",
        ],
    }


def run_all_synthetic_cases() -> dict[str, Any]:
    results = [run_synthetic_case(scenario) for scenario in SCENARIOS]
    return {
        "tool": "paper1_feature_validity.synthetic_measurement_validation",
        "oracle_independent_of_frozen_measurement_functions": True,
        "scenario_count": len(results),
        "feature_count": len(BIOLOGICAL_FEATURE_COLUMNS),
        "results": results,
        "summary_status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
    }


def write_json_report(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
