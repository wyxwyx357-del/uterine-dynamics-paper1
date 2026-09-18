from __future__ import annotations

import math
from typing import Any

import numpy as np


MINIMAL_FEATURE_COLUMNS = (
    "case_id",
    "usable_frame_ratio",
    "formal_nan_ratio",
    "dicom_physical_curvature_available",
    "dicom_physical_curvature_rate_available",
    "rsr_abs_median",
    "rsr_abs_p95",
    "curvature_change_abs_median",
    "curvature_change_abs_p95",
    "anterior_curvature_change_abs_p95",
    "posterior_curvature_change_abs_p95",
)

IDENTIFIER_COLUMNS = (
    "case_id",
)

QC_METADATA_COLUMNS = (
    "rsr_valid_ratio",
    "cavity_valid_ratio",
    "longitudinal_valid_ratio",
    "curvature_valid_ratio",
    "dicom_physical_curvature_available",
    "dicom_physical_curvature_rate_available",
)

BIOLOGICAL_FEATURE_COLUMNS = (
    "rsr_abs_median",
    "rsr_abs_p95",
    "anterior_rsr_abs_median",
    "anterior_rsr_abs_p95",
    "posterior_rsr_abs_median",
    "posterior_rsr_abs_p95",
    "cavity_width_strain_rate_abs_median",
    "cavity_width_strain_rate_abs_p95",
    "longitudinal_wall_strain_rate_abs_median",
    "longitudinal_wall_strain_rate_abs_p95",
    "anterior_longitudinal_wall_strain_rate_abs_median",
    "anterior_longitudinal_wall_strain_rate_abs_p95",
    "posterior_longitudinal_wall_strain_rate_abs_median",
    "posterior_longitudinal_wall_strain_rate_abs_p95",
    "wall_curvature_change_rate_mm_inv_s_abs_median",
    "wall_curvature_change_rate_mm_inv_s_abs_p95",
    "anterior_wall_curvature_change_rate_mm_inv_s_abs_median",
    "anterior_wall_curvature_change_rate_mm_inv_s_abs_p95",
    "posterior_wall_curvature_change_rate_mm_inv_s_abs_median",
    "posterior_wall_curvature_change_rate_mm_inv_s_abs_p95",
)

MODEL_INPUT_FEATURE_COLUMNS = BIOLOGICAL_FEATURE_COLUMNS

# Preserve the historical 27-column CSV order while keeping model inputs explicit.
FORMAL_CANDIDATE_FEATURE_COLUMNS = (
    IDENTIFIER_COLUMNS + QC_METADATA_COLUMNS + BIOLOGICAL_FEATURE_COLUMNS
)

if not set(IDENTIFIER_COLUMNS).isdisjoint(QC_METADATA_COLUMNS):
    raise RuntimeError("identifier and QC metadata columns must be disjoint")
if not set(IDENTIFIER_COLUMNS).isdisjoint(BIOLOGICAL_FEATURE_COLUMNS):
    raise RuntimeError("identifier and biological feature columns must be disjoint")
if not set(QC_METADATA_COLUMNS).isdisjoint(BIOLOGICAL_FEATURE_COLUMNS):
    raise RuntimeError("QC metadata and biological feature columns must be disjoint")
if len(BIOLOGICAL_FEATURE_COLUMNS) != 20:
    raise RuntimeError("the frozen biological feature contract must contain 20 columns")
if len(FORMAL_CANDIDATE_FEATURE_COLUMNS) != 27:
    raise RuntimeError("the formal candidate output contract must contain 27 columns")


def finite_values(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Return finite values, optionally restricted by an existing valid mask."""

    array = np.asarray(values)
    selected = np.isfinite(array)
    if valid is not None:
        valid_array = np.asarray(valid, dtype=bool)
        if valid_array.shape != array.shape:
            raise ValueError("valid mask must have the same shape as values")
        selected &= valid_array
    return np.asarray(array[selected], dtype=np.float64)


def finite_abs_statistics(
    values: np.ndarray,
    valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize absolute finite values without interpolation or imputation."""

    finite = np.abs(finite_values(values, valid))
    if finite.size == 0:
        return {"median": math.nan, "p95": math.nan, "rms": math.nan}
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
    }


def finite_ratio(
    values: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    """Return the proportion of array elements usable for a statistic."""

    array = np.asarray(values)
    if array.size == 0:
        return math.nan
    selected = np.isfinite(array)
    if valid is not None:
        valid_array = np.asarray(valid, dtype=bool)
        if valid_array.shape != array.shape:
            raise ValueError("valid mask must have the same shape as values")
        selected &= valid_array
    return float(np.mean(selected))


def _scalar_bool(value: Any, *, field: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{field} must be a scalar boolean")
    return bool(array.reshape(-1)[0])


def _scalar_float(value: Any, *, field: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{field} must be a scalar number")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _two_side_abs_statistics(
    values: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    array = np.asarray(values)
    valid_array = None if valid is None else np.asarray(valid, dtype=bool)
    return (
        finite_abs_statistics(array, valid_array),
        finite_abs_statistics(
            array[:, 0], None if valid_array is None else valid_array[:, 0]
        ),
        finite_abs_statistics(
            array[:, 1], None if valid_array is None else valid_array[:, 1]
        ),
    )


def extract_minimal_case_features(
    *,
    case_id: str,
    current_qc_radial_strain_rate_s: np.ndarray,
    qc_wall_curvature_change_rate_mm_inv_s: np.ndarray | None = None,
    physical_curvature_available: Any = False,
    physical_curvature_rate_available: Any = False,
) -> dict[str, object]:
    """Extract the frozen five-case minimal ROI activity/deformation summary.

    RSR is expected as ``(pair_frame, side, section)`` in 1/s. DICOM curvature
    change rate is expected with the same axis meaning in mm^-1/s. Existing NaN
    placement is the formal validity contract; this function creates no new QC,
    threshold, event, exclusion, direction, or propagation rule.
    """

    rsr = np.asarray(current_qc_radial_strain_rate_s)
    if rsr.ndim != 3 or rsr.shape[1] != 2:
        raise ValueError("formal RSR must have shape (pair_frame, side=2, section)")

    finite_rsr = np.isfinite(rsr)
    usable_frame_ratio = (
        float(np.mean(np.any(finite_rsr, axis=(1, 2)))) if rsr.shape[0] else math.nan
    )
    formal_nan_ratio = float(np.mean(~finite_rsr)) if rsr.size else math.nan
    rsr_stats = finite_abs_statistics(rsr)

    curvature_available = _scalar_bool(
        physical_curvature_available, field="physical_curvature_available"
    )
    curvature_rate_available = _scalar_bool(
        physical_curvature_rate_available,
        field="physical_curvature_rate_available",
    )

    curvature_stats = {"median": math.nan, "p95": math.nan, "rms": math.nan}
    anterior_p95 = math.nan
    posterior_p95 = math.nan
    if curvature_rate_available:
        if qc_wall_curvature_change_rate_mm_inv_s is None:
            raise ValueError(
                "physical curvature rate is marked available but its array is missing"
            )
        curvature_rate = np.asarray(qc_wall_curvature_change_rate_mm_inv_s)
        if curvature_rate.ndim != 3 or curvature_rate.shape[1] != 2:
            raise ValueError(
                "DICOM curvature change rate must have shape "
                "(pair_frame, side=2, section)"
            )
        if curvature_rate.shape != rsr.shape:
            raise ValueError("DICOM curvature rate and formal RSR shapes must match")
        curvature_stats = finite_abs_statistics(curvature_rate)
        anterior_p95 = finite_abs_statistics(curvature_rate[:, 0])["p95"]
        posterior_p95 = finite_abs_statistics(curvature_rate[:, 1])["p95"]

    return {
        "case_id": str(case_id),
        "usable_frame_ratio": usable_frame_ratio,
        "formal_nan_ratio": formal_nan_ratio,
        "dicom_physical_curvature_available": curvature_available,
        "dicom_physical_curvature_rate_available": curvature_rate_available,
        "rsr_abs_median": rsr_stats["median"],
        "rsr_abs_p95": rsr_stats["p95"],
        "curvature_change_abs_median": curvature_stats["median"],
        "curvature_change_abs_p95": curvature_stats["p95"],
        "anterior_curvature_change_abs_p95": anterior_p95,
        "posterior_curvature_change_abs_p95": posterior_p95,
    }


def extract_formal_candidate_case_features(
    *,
    case_id: str,
    current_qc_radial_strain_rate_s: np.ndarray,
    rsr_fps: Any,
    anatomical_pair_qc_valid: np.ndarray,
    anatomical_fps: Any,
    qc_cavity_width_strain_rate_s: np.ndarray,
    cavity_qc_valid: np.ndarray,
    qc_longitudinal_wall_strain_rate_s: np.ndarray,
    longitudinal_qc_valid: np.ndarray,
    qc_wall_curvature_change_rate_mm_inv_s: np.ndarray | None = None,
    dicom_pair_qc_valid: np.ndarray | None = None,
    physical_curvature_available: Any = False,
    physical_curvature_rate_available: Any = False,
) -> dict[str, object]:
    """Extract the frozen 27-column five-case formal candidate summary.

    Each feature family uses its existing formal validity domain. Cross-input
    shape, time-base, and mask-lineage checks only prevent mixing different
    formal result versions; they do not create new QC or exclusion rules.
    """

    rsr = np.asarray(current_qc_radial_strain_rate_s)
    if rsr.ndim != 3 or rsr.shape[1] != 2:
        raise ValueError("formal RSR must have shape (pair_frame, side=2, section)")
    frame_count, _, section_count = rsr.shape

    pair_valid = np.asarray(anatomical_pair_qc_valid, dtype=bool)
    if pair_valid.shape != rsr.shape:
        raise ValueError("anatomical pair_qc_valid must match formal RSR shape")
    if not np.array_equal(pair_valid, np.isfinite(rsr)):
        raise ValueError(
            "anatomical pair_qc_valid must equal finite positions of formal RSR"
        )

    rsr_fps_value = _scalar_float(rsr_fps, field="rsr_fps")
    anatomical_fps_value = _scalar_float(anatomical_fps, field="anatomical_fps")
    if not math.isclose(
        rsr_fps_value, anatomical_fps_value, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("formal RSR and anatomical fps must match")

    cavity = np.asarray(qc_cavity_width_strain_rate_s)
    cavity_valid = np.asarray(cavity_qc_valid, dtype=bool)
    expected_cavity_shape = (frame_count, section_count)
    if cavity.shape != expected_cavity_shape:
        raise ValueError("formal cavity strain rate must have shape (T, S)")
    if cavity_valid.shape != cavity.shape:
        raise ValueError("cavity_qc_valid must match formal cavity strain rate")

    longitudinal = np.asarray(qc_longitudinal_wall_strain_rate_s)
    longitudinal_valid = np.asarray(longitudinal_qc_valid, dtype=bool)
    expected_longitudinal_shape = (frame_count, 2, section_count - 1)
    if longitudinal.shape != expected_longitudinal_shape:
        raise ValueError(
            "formal longitudinal strain rate must have shape (T, 2, S-1)"
        )
    if longitudinal_valid.shape != longitudinal.shape:
        raise ValueError(
            "longitudinal_qc_valid must match formal longitudinal strain rate"
        )

    curvature_available = _scalar_bool(
        physical_curvature_available, field="physical_curvature_available"
    )
    curvature_rate_available = _scalar_bool(
        physical_curvature_rate_available,
        field="physical_curvature_rate_available",
    )
    if curvature_rate_available and not curvature_available:
        raise ValueError(
            "physical curvature rate cannot be available when curvature is unavailable"
        )

    curvature_rate = None
    if qc_wall_curvature_change_rate_mm_inv_s is not None:
        curvature_rate = np.asarray(qc_wall_curvature_change_rate_mm_inv_s)
        if curvature_rate.shape != rsr.shape:
            raise ValueError("DICOM curvature rate must have shape (T, 2, S)")
    if dicom_pair_qc_valid is not None:
        dicom_pair_valid = np.asarray(dicom_pair_qc_valid, dtype=bool)
        if dicom_pair_valid.shape != pair_valid.shape:
            raise ValueError("DICOM pair_qc_valid must match anatomical shape")
        if not np.array_equal(dicom_pair_valid, pair_valid):
            raise ValueError(
                "DICOM pair_qc_valid must match current anatomical pair_qc_valid"
            )
    elif curvature_rate is not None or curvature_available or curvature_rate_available:
        raise ValueError("DICOM pair_qc_valid is required when DICOM results are present")
    if curvature_rate_available and curvature_rate is None:
        raise ValueError(
            "physical curvature rate is marked available but its array is missing"
        )

    rsr_stats, anterior_rsr, posterior_rsr = _two_side_abs_statistics(rsr)
    cavity_stats = finite_abs_statistics(cavity, cavity_valid)
    longitudinal_stats, anterior_longitudinal, posterior_longitudinal = (
        _two_side_abs_statistics(longitudinal, longitudinal_valid)
    )

    curvature_valid_ratio = math.nan
    curvature_stats = {"median": math.nan, "p95": math.nan, "rms": math.nan}
    anterior_curvature = curvature_stats.copy()
    posterior_curvature = curvature_stats.copy()
    if curvature_rate_available:
        curvature_valid_ratio = finite_ratio(curvature_rate)
        curvature_stats, anterior_curvature, posterior_curvature = (
            _two_side_abs_statistics(curvature_rate)
        )

    return {
        "case_id": str(case_id),
        "rsr_valid_ratio": finite_ratio(rsr),
        "cavity_valid_ratio": finite_ratio(cavity, cavity_valid),
        "longitudinal_valid_ratio": finite_ratio(
            longitudinal, longitudinal_valid
        ),
        "curvature_valid_ratio": curvature_valid_ratio,
        "dicom_physical_curvature_available": curvature_available,
        "dicom_physical_curvature_rate_available": curvature_rate_available,
        "rsr_abs_median": rsr_stats["median"],
        "rsr_abs_p95": rsr_stats["p95"],
        "anterior_rsr_abs_median": anterior_rsr["median"],
        "anterior_rsr_abs_p95": anterior_rsr["p95"],
        "posterior_rsr_abs_median": posterior_rsr["median"],
        "posterior_rsr_abs_p95": posterior_rsr["p95"],
        "cavity_width_strain_rate_abs_median": cavity_stats["median"],
        "cavity_width_strain_rate_abs_p95": cavity_stats["p95"],
        "longitudinal_wall_strain_rate_abs_median": longitudinal_stats["median"],
        "longitudinal_wall_strain_rate_abs_p95": longitudinal_stats["p95"],
        "anterior_longitudinal_wall_strain_rate_abs_median": (
            anterior_longitudinal["median"]
        ),
        "anterior_longitudinal_wall_strain_rate_abs_p95": (
            anterior_longitudinal["p95"]
        ),
        "posterior_longitudinal_wall_strain_rate_abs_median": (
            posterior_longitudinal["median"]
        ),
        "posterior_longitudinal_wall_strain_rate_abs_p95": (
            posterior_longitudinal["p95"]
        ),
        "wall_curvature_change_rate_mm_inv_s_abs_median": curvature_stats[
            "median"
        ],
        "wall_curvature_change_rate_mm_inv_s_abs_p95": curvature_stats["p95"],
        "anterior_wall_curvature_change_rate_mm_inv_s_abs_median": (
            anterior_curvature["median"]
        ),
        "anterior_wall_curvature_change_rate_mm_inv_s_abs_p95": (
            anterior_curvature["p95"]
        ),
        "posterior_wall_curvature_change_rate_mm_inv_s_abs_median": (
            posterior_curvature["median"]
        ),
        "posterior_wall_curvature_change_rate_mm_inv_s_abs_p95": (
            posterior_curvature["p95"]
        ),
    }
