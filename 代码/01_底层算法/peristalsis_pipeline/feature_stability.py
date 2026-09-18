from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .formal_feature_extraction import (
    BIOLOGICAL_FEATURE_COLUMNS,
    extract_formal_candidate_case_features,
)


SENSITIVITY_POSITIONS = 5
TOP_FRACTION = 0.01
TOP_FRACTION_MINIMUM_N = 100


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    statistic_type: str
    domain: str


@dataclass(frozen=True)
class StabilityCaseData:
    case_id: str
    rsr: np.ndarray
    rsr_fps: Any
    pair_valid: np.ndarray
    anatomical_fps: Any
    cavity: np.ndarray
    cavity_valid: np.ndarray
    longitudinal: np.ndarray
    longitudinal_valid: np.ndarray
    curvature_rate: np.ndarray | None
    physical_curvature_available: Any
    physical_curvature_rate_available: Any


FEATURE_SPECS = (
    FeatureSpec("rsr_abs_median", "rsr", "median", "global"),
    FeatureSpec("rsr_abs_p95", "rsr", "p95", "global"),
    FeatureSpec("anterior_rsr_abs_median", "rsr", "median", "anterior"),
    FeatureSpec("anterior_rsr_abs_p95", "rsr", "p95", "anterior"),
    FeatureSpec("posterior_rsr_abs_median", "rsr", "median", "posterior"),
    FeatureSpec("posterior_rsr_abs_p95", "rsr", "p95", "posterior"),
    FeatureSpec(
        "cavity_width_strain_rate_abs_median", "cavity", "median", "global"
    ),
    FeatureSpec(
        "cavity_width_strain_rate_abs_p95", "cavity", "p95", "global"
    ),
    FeatureSpec(
        "longitudinal_wall_strain_rate_abs_median",
        "longitudinal",
        "median",
        "global",
    ),
    FeatureSpec(
        "longitudinal_wall_strain_rate_abs_p95",
        "longitudinal",
        "p95",
        "global",
    ),
    FeatureSpec(
        "anterior_longitudinal_wall_strain_rate_abs_median",
        "longitudinal",
        "median",
        "anterior",
    ),
    FeatureSpec(
        "anterior_longitudinal_wall_strain_rate_abs_p95",
        "longitudinal",
        "p95",
        "anterior",
    ),
    FeatureSpec(
        "posterior_longitudinal_wall_strain_rate_abs_median",
        "longitudinal",
        "median",
        "posterior",
    ),
    FeatureSpec(
        "posterior_longitudinal_wall_strain_rate_abs_p95",
        "longitudinal",
        "p95",
        "posterior",
    ),
    FeatureSpec(
        "wall_curvature_change_rate_mm_inv_s_abs_median",
        "curvature",
        "median",
        "global",
    ),
    FeatureSpec(
        "wall_curvature_change_rate_mm_inv_s_abs_p95",
        "curvature",
        "p95",
        "global",
    ),
    FeatureSpec(
        "anterior_wall_curvature_change_rate_mm_inv_s_abs_median",
        "curvature",
        "median",
        "anterior",
    ),
    FeatureSpec(
        "anterior_wall_curvature_change_rate_mm_inv_s_abs_p95",
        "curvature",
        "p95",
        "anterior",
    ),
    FeatureSpec(
        "posterior_wall_curvature_change_rate_mm_inv_s_abs_median",
        "curvature",
        "median",
        "posterior",
    ),
    FeatureSpec(
        "posterior_wall_curvature_change_rate_mm_inv_s_abs_p95",
        "curvature",
        "p95",
        "posterior",
    ),
)

if tuple(spec.name for spec in FEATURE_SPECS) != BIOLOGICAL_FEATURE_COLUMNS:
    raise RuntimeError("stability feature specifications must match the frozen 20 columns")


def compute_feature_row(data: StabilityCaseData) -> dict[str, object]:
    return extract_formal_candidate_case_features(
        case_id=data.case_id,
        current_qc_radial_strain_rate_s=data.rsr,
        rsr_fps=data.rsr_fps,
        anatomical_pair_qc_valid=data.pair_valid,
        anatomical_fps=data.anatomical_fps,
        qc_cavity_width_strain_rate_s=data.cavity,
        cavity_qc_valid=data.cavity_valid,
        qc_longitudinal_wall_strain_rate_s=data.longitudinal,
        longitudinal_qc_valid=data.longitudinal_valid,
        qc_wall_curvature_change_rate_mm_inv_s=data.curvature_rate,
        dicom_pair_qc_valid=(
            data.pair_valid if data.curvature_rate is not None else None
        ),
        physical_curvature_available=data.physical_curvature_available,
        physical_curvature_rate_available=data.physical_curvature_rate_available,
    )


def propagate_pair_exclusion(
    pair_exclusion: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map a pair-level sensitivity exclusion through frozen feature geometry."""

    excluded = np.asarray(pair_exclusion, dtype=bool)
    if excluded.ndim != 3 or excluded.shape[1] != 2:
        raise ValueError("pair exclusion must have shape (T, side=2, S)")
    cavity = excluded[:, 0] | excluded[:, 1]
    longitudinal = excluded[:, :, :-1] | excluded[:, :, 1:]
    curvature = np.zeros_like(excluded)
    if excluded.shape[2] >= 3:
        curvature[:, :, 1:-1] = (
            excluded[:, :, :-2]
            | excluded[:, :, 1:-1]
            | excluded[:, :, 2:]
        )
    return {
        "pair": excluded,
        "cavity": cavity,
        "longitudinal": longitudinal,
        "curvature": curvature,
    }


def _assert_values_only_removed(before: np.ndarray, after: np.ndarray) -> None:
    original = np.asarray(before)
    perturbed = np.asarray(after)
    if original.shape != perturbed.shape:
        raise RuntimeError("sensitivity perturbation changed an array shape")
    if np.any(~np.isfinite(original) & np.isfinite(perturbed)):
        raise RuntimeError("sensitivity perturbation created NaN-to-finite values")
    retained = np.isfinite(perturbed)
    if not np.array_equal(original[retained], perturbed[retained]):
        raise RuntimeError("unmasked values changed during sensitivity perturbation")


def apply_pair_exclusion(
    data: StabilityCaseData, pair_exclusion: np.ndarray
) -> StabilityCaseData:
    """Return a mask-only sensitivity copy without recomputing upstream arrays."""

    requested = np.asarray(pair_exclusion, dtype=bool)
    if requested.shape != data.pair_valid.shape:
        raise ValueError("pair exclusion must match the frozen pair validity shape")
    effective = requested & np.asarray(data.pair_valid, dtype=bool)
    propagated = propagate_pair_exclusion(effective)

    rsr = np.asarray(data.rsr).copy()
    rsr[propagated["pair"]] = np.nan
    pair_valid = np.asarray(data.pair_valid, dtype=bool) & ~propagated["pair"]
    cavity_valid = (
        np.asarray(data.cavity_valid, dtype=bool) & ~propagated["cavity"]
    )
    longitudinal_valid = (
        np.asarray(data.longitudinal_valid, dtype=bool)
        & ~propagated["longitudinal"]
    )
    curvature_rate = None
    if data.curvature_rate is not None:
        curvature_rate = np.asarray(data.curvature_rate).copy()
        curvature_rate[propagated["curvature"]] = np.nan

    result = replace(
        data,
        rsr=rsr,
        pair_valid=pair_valid,
        cavity_valid=cavity_valid,
        longitudinal_valid=longitudinal_valid,
        curvature_rate=curvature_rate,
    )
    assert_mask_only_invariants(data, result)
    return result


def realized_mask_identity(mask: np.ndarray) -> str:
    """Return a deterministic identity for one actually applied boolean mask."""

    array = np.asarray(mask, dtype=bool)
    shape = ",".join(str(value) for value in array.shape).encode("ascii")
    packed = np.packbits(array.reshape(-1)).tobytes()
    return hashlib.sha256(shape + b":" + packed).hexdigest()


def assert_mask_only_invariants(
    before: StabilityCaseData, after: StabilityCaseData
) -> None:
    """Fail closed if a sensitivity copy restores or changes formal data."""

    for name in ("pair_valid", "cavity_valid", "longitudinal_valid"):
        old = np.asarray(getattr(before, name), dtype=bool)
        new = np.asarray(getattr(after, name), dtype=bool)
        if old.shape != new.shape or np.any(new & ~old):
            raise RuntimeError(f"{name} must only lose valid positions")
        if np.count_nonzero(new) > np.count_nonzero(old):
            raise RuntimeError(f"{name} valid count increased")

    _assert_values_only_removed(before.rsr, after.rsr)
    if not np.array_equal(before.cavity, after.cavity, equal_nan=True):
        raise RuntimeError("cavity values changed instead of mask-only exclusion")
    if not np.array_equal(before.longitudinal, after.longitudinal, equal_nan=True):
        raise RuntimeError("longitudinal values changed instead of mask-only exclusion")
    if before.curvature_rate is None:
        if after.curvature_rate is not None:
            raise RuntimeError("sensitivity perturbation created curvature data")
    elif after.curvature_rate is None:
        raise RuntimeError("sensitivity perturbation removed the curvature array")
    else:
        _assert_values_only_removed(before.curvature_rate, after.curvature_rate)


def domain_values_and_valid(
    data: StabilityCaseData, spec: FeatureSpec
) -> tuple[np.ndarray, np.ndarray]:
    if spec.family == "rsr":
        values = np.asarray(data.rsr)
        valid = np.isfinite(values)
    elif spec.family == "cavity":
        values = np.asarray(data.cavity)
        valid = np.asarray(data.cavity_valid, dtype=bool) & np.isfinite(values)
    elif spec.family == "longitudinal":
        values = np.asarray(data.longitudinal)
        valid = np.asarray(data.longitudinal_valid, dtype=bool) & np.isfinite(values)
    elif spec.family == "curvature":
        if data.curvature_rate is None:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=bool)
        values = np.asarray(data.curvature_rate)
        valid = np.isfinite(values)
    else:
        raise ValueError(f"unknown feature family: {spec.family}")

    if spec.domain == "anterior":
        values = values[:, 0]
        valid = valid[:, 0]
    elif spec.domain == "posterior":
        values = values[:, 1]
        valid = valid[:, 1]
    elif spec.domain != "global":
        raise ValueError(f"unknown feature domain: {spec.domain}")
    return values, valid


def valid_count_and_ratio(
    data: StabilityCaseData, spec: FeatureSpec
) -> tuple[int, float]:
    values, valid = domain_values_and_valid(data, spec)
    if values.size == 0:
        return 0, math.nan
    count = int(np.count_nonzero(valid))
    return count, float(count / values.size)


def symmetric_relative_difference_pct(baseline: float, perturbed: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(perturbed):
        return math.nan
    denominator = abs(baseline) + abs(perturbed)
    if denominator == 0.0:
        return 0.0
    return 200.0 * abs(perturbed - baseline) / denominator


def sensitivity_warning(
    statistic_type: str, baseline: float, perturbed: float
) -> str:
    """Return an engineering sensitivity warning, never a deletion decision."""

    if not math.isfinite(baseline) or not math.isfinite(perturbed):
        return "证据不足"
    srd = symmetric_relative_difference_pct(baseline, perturbed)
    if statistic_type == "median":
        low, high = 10.0, 20.0
    elif statistic_type == "p95":
        low, high = 15.0, 30.0
    else:
        raise ValueError(f"unknown statistic type: {statistic_type}")
    if srd <= low:
        return "低敏感"
    if srd <= high:
        return "中等敏感"
    return "高敏感"


def _block_start(length: int, width: int, center_fraction: float) -> int:
    center = center_fraction * (length - 1)
    start = int(round(center - (width - 1) / 2.0))
    return min(max(start, 0), length - width)


def fixed_contiguous_pair_exclusion(
    pair_valid: np.ndarray,
    *,
    axis: str,
    target_removed_fraction: float,
    position_index: int,
    position_count: int = SENSITIVITY_POSITIONS,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Choose one deterministic contiguous block closest to a valid-count target."""

    valid = np.asarray(pair_valid, dtype=bool)
    if valid.ndim != 3 or valid.shape[1] != 2:
        raise ValueError("pair_valid must have shape (T, side=2, S)")
    if not 0.0 < target_removed_fraction < 1.0:
        raise ValueError("target removed fraction must be between zero and one")
    if not 0 <= position_index < position_count:
        raise ValueError("position index is outside the fixed position count")
    if axis == "time":
        axis_counts = np.count_nonzero(valid, axis=(1, 2))
        axis_length = valid.shape[0]
    elif axis == "space":
        axis_counts = np.count_nonzero(valid, axis=(0, 1))
        axis_length = valid.shape[2]
    else:
        raise ValueError("axis must be time or space")

    total_valid = int(np.count_nonzero(valid))
    if total_valid == 0:
        raise ValueError("cannot perturb a case with no valid pairs")
    target_count = target_removed_fraction * total_valid
    center_fraction = (position_index + 0.5) / position_count
    prefix = np.concatenate(([0], np.cumsum(axis_counts, dtype=np.int64)))
    candidates: list[tuple[float, int, int, int]] = []
    for width in range(1, axis_length + 1):
        start = _block_start(axis_length, width, center_fraction)
        removed = int(prefix[start + width] - prefix[start])
        candidates.append((abs(removed - target_count), width, start, removed))
    _, width, start, removed = min(candidates)
    stop = start + width

    selected = np.zeros(axis_length, dtype=bool)
    selected[start:stop] = True
    if axis == "time":
        exclusion = valid & selected[:, None, None]
    else:
        exclusion = valid & selected[None, None, :]
    return exclusion, {
        "realized_axis": axis,
        "realized_removed_indices_count": width,
        "realized_axis_length": axis_length,
        "realized_upstream_removed_fraction": width / axis_length,
        "block_start_index": start,
        "block_stop_index_exclusive": stop,
        "pair_valid_count_before": total_valid,
        "pair_valid_count_removed": removed,
        "pair_actual_removed_valid_fraction": removed / total_valid,
        "fixed_position_index": position_index + 1,
        "fixed_position_count": position_count,
    }


def random_pair_exclusion(
    pair_valid: np.ndarray,
    *,
    target_removed_fraction: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Remove an exact seeded fraction of currently valid pair positions."""

    valid = np.asarray(pair_valid, dtype=bool)
    if valid.ndim != 3 or valid.shape[1] != 2:
        raise ValueError("pair_valid must have shape (T, side=2, S)")
    if not 0.0 < target_removed_fraction < 1.0:
        raise ValueError("target removed fraction must be between zero and one")
    valid_indices = np.flatnonzero(valid.reshape(-1))
    before = int(valid_indices.size)
    target_removed_n = int(math.floor(target_removed_fraction * before))
    exclusion = np.zeros_like(valid)
    if target_removed_n:
        rng = np.random.default_rng(seed)
        selected = rng.choice(valid_indices, size=target_removed_n, replace=False)
        exclusion.reshape(-1)[selected] = True
    return exclusion, {
        "status": "ok" if target_removed_n else "insufficient_n",
        "seed": int(seed),
        "realized_axis": "valid_pair_position",
        "realized_removed_indices_count": target_removed_n,
        "realized_axis_length": before,
        "realized_upstream_removed_fraction": (
            target_removed_n / before if before else math.nan
        ),
        "pair_valid_count_before": before,
        "target_removed_n": target_removed_n,
        "pair_valid_count_removed": target_removed_n,
        "pair_actual_removed_valid_fraction": (
            target_removed_n / before if before else math.nan
        ),
    }


def temporal_quintile_pair_exclusion(
    pair_valid: np.ndarray, quintile_index: int
) -> tuple[np.ndarray, dict[str, float | int]]:
    valid = np.asarray(pair_valid, dtype=bool)
    if not 0 <= quintile_index < 5:
        raise ValueError("temporal quintile index must be between zero and four")
    indices = np.array_split(np.arange(valid.shape[0]), 5)[quintile_index]
    selected = np.zeros(valid.shape[0], dtype=bool)
    selected[indices] = True
    exclusion = valid & selected[:, None, None]
    before = int(np.count_nonzero(valid))
    removed = int(np.count_nonzero(exclusion))
    return exclusion, {
        "realized_axis": "time_frame",
        "realized_removed_indices_count": int(len(indices)),
        "realized_axis_length": int(valid.shape[0]),
        "realized_upstream_removed_fraction": float(len(indices) / valid.shape[0]),
        "block_start_index": int(indices[0]),
        "block_stop_index_exclusive": int(indices[-1] + 1),
        "pair_valid_count_before": before,
        "pair_valid_count_removed": removed,
        "pair_actual_removed_valid_fraction": removed / before,
        "fixed_position_index": quintile_index + 1,
        "fixed_position_count": 5,
    }


def spatial_quintile_pair_exclusion(
    pair_valid: np.ndarray,
    normalized_position: np.ndarray,
    quintile_index: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    valid = np.asarray(pair_valid, dtype=bool)
    position = np.asarray(normalized_position, dtype=np.float64)
    if position.shape != (valid.shape[2],):
        raise ValueError("normalized position must match the pair section axis")
    if not 0 <= quintile_index < 5:
        raise ValueError("spatial quintile index must be between zero and four")
    lower = quintile_index / 5.0
    upper = (quintile_index + 1) / 5.0
    if quintile_index == 4:
        selected = (position >= lower) & (position <= upper)
    else:
        selected = (position >= lower) & (position < upper)
    exclusion = valid & selected[None, None, :]
    before = int(np.count_nonzero(valid))
    removed = int(np.count_nonzero(exclusion))
    selected_indices = np.flatnonzero(selected)
    return exclusion, {
        "realized_axis": "spatial_section",
        "realized_removed_indices_count": int(len(selected_indices)),
        "realized_axis_length": int(valid.shape[2]),
        "realized_upstream_removed_fraction": float(
            len(selected_indices) / valid.shape[2]
        ),
        "block_start_index": int(selected_indices[0]),
        "block_stop_index_exclusive": int(selected_indices[-1] + 1),
        "pair_valid_count_before": before,
        "pair_valid_count_removed": removed,
        "pair_actual_removed_valid_fraction": removed / before,
        "fixed_position_index": quintile_index + 1,
        "fixed_position_count": 5,
    }


def remove_top_fraction_mask(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    fraction: float = TOP_FRACTION,
    minimum_n: int = TOP_FRACTION_MINIMUM_N,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Remove an exact top fraction in one feature domain, or report insufficient N."""

    array = np.asarray(values)
    keep = np.asarray(valid, dtype=bool).copy()
    if keep.shape != array.shape:
        raise ValueError("top-fraction valid mask must match values")
    keep &= np.isfinite(array)
    valid_indices = np.flatnonzero(keep.reshape(-1))
    count = int(valid_indices.size)
    if count < minimum_n:
        return keep, {
            "status": "insufficient_n",
            "valid_count_before": count,
            "valid_count_removed": 0,
        }
    remove_count = int(math.ceil(fraction * count))
    absolute = np.abs(array.reshape(-1)[valid_indices])
    order = np.argsort(absolute, kind="stable")
    removed_indices = valid_indices[order[-remove_count:]]
    keep.reshape(-1)[removed_indices] = False
    return keep, {
        "status": "ok",
        "valid_count_before": count,
        "valid_count_removed": remove_count,
    }


def top_fraction_feature_value(
    data: StabilityCaseData, spec: FeatureSpec
) -> tuple[float, int, int, str]:
    values, valid = domain_values_and_valid(data, spec)
    if values.size == 0:
        return math.nan, 0, 0, "DICOM_rate_unavailable"
    keep, metadata = remove_top_fraction_mask(values, valid)
    before = int(metadata["valid_count_before"])
    removed = int(metadata["valid_count_removed"])
    status = str(metadata["status"])
    if status != "ok":
        return math.nan, before, before, status
    removed_mask = valid & ~keep
    perturbed = feature_domain_exclusion_copy(data, spec, removed_mask)
    value = float(compute_feature_row(perturbed)[spec.name])
    return value, before, before - removed, status


def feature_domain_exclusion_copy(
    data: StabilityCaseData, spec: FeatureSpec, domain_exclusion: np.ndarray
) -> StabilityCaseData:
    """Create a feature-specific mask-only copy for top-fraction evaluation."""

    values, valid = domain_values_and_valid(data, spec)
    exclusion = np.asarray(domain_exclusion, dtype=bool)
    if exclusion.shape != values.shape or np.any(exclusion & ~valid):
        raise ValueError("feature-domain exclusion must select only valid values")

    def expand_domain(mask: np.ndarray, full_shape: tuple[int, ...]) -> np.ndarray:
        if spec.domain == "global":
            return mask
        expanded = np.zeros(full_shape, dtype=bool)
        side = 0 if spec.domain == "anterior" else 1
        expanded[:, side] = mask
        return expanded

    if spec.family == "rsr":
        expanded = expand_domain(exclusion, np.asarray(data.rsr).shape)
        array = np.asarray(data.rsr).copy()
        array[expanded] = np.nan
        pair_valid = np.asarray(data.pair_valid, dtype=bool).copy()
        pair_valid[expanded] = False
        return replace(data, rsr=array, pair_valid=pair_valid)
    if spec.family == "cavity":
        valid_mask = np.asarray(data.cavity_valid, dtype=bool).copy()
        valid_mask[exclusion] = False
        return replace(data, cavity_valid=valid_mask)
    if spec.family == "longitudinal":
        valid_mask = np.asarray(data.longitudinal_valid, dtype=bool).copy()
        valid_mask[expand_domain(exclusion, valid_mask.shape)] = False
        return replace(data, longitudinal_valid=valid_mask)
    if spec.family == "curvature":
        if data.curvature_rate is None:
            raise ValueError("curvature exclusion requires curvature data")
        array = np.asarray(data.curvature_rate).copy()
        array[expand_domain(exclusion, array.shape)] = np.nan
        return replace(data, curvature_rate=array)
    raise ValueError(f"unknown feature family: {spec.family}")
