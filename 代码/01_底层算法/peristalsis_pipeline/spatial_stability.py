"""Normalized cervix-to-fundus spatial-profile utilities for Paper 1.

This module summarizes frozen local dynamic measurements inside fixed
normalized anatomical-position bins. It does not rerun tracking, redefine
features, infer waves, or prove same-tissue identity across frames.
"""
from __future__ import annotations

import math

import numpy as np

from .feature_stability import (
    FeatureSpec,
    StabilityCaseData,
    domain_values_and_valid,
    symmetric_relative_difference_pct,
)
from .temporal_stability import paired_spearman


DEFAULT_SPATIAL_BINS = 5
_POSITION_TOLERANCE = 1e-6


def normalized_spatial_bins(
    positions: np.ndarray, bins: int = DEFAULT_SPATIAL_BINS
) -> np.ndarray:
    """Assign normalized 0-1 anatomical positions to equal-width spatial bins."""

    if bins < 1:
        raise ValueError("bins must be positive")
    coordinate = np.asarray(positions, dtype=np.float64)
    if coordinate.ndim != 1:
        raise ValueError("normalized spatial positions must be one-dimensional")
    if coordinate.size == 0:
        return np.asarray([], dtype=np.int32)
    if not np.isfinite(coordinate).all():
        raise ValueError("normalized spatial positions must be finite")
    if (
        np.min(coordinate) < -_POSITION_TOLERANCE
        or np.max(coordinate) > 1.0 + _POSITION_TOLERANCE
    ):
        raise ValueError("normalized spatial positions must lie within 0-1")
    clipped = np.clip(coordinate, 0.0, 1.0)
    result = np.floor(clipped * bins).astype(np.int64)
    return np.minimum(result, bins - 1).astype(np.int32)


def feature_spatial_positions(
    normalized_position: np.ndarray,
    spec: FeatureSpec,
    *,
    position_count: int,
) -> np.ndarray:
    """Return the anatomical coordinate represented by each feature position.

    RSR, cavity-width strain rate, and curvature-rate positions use the frozen
    section coordinates. Longitudinal strain rate is defined between adjacent
    sections, so its coordinate is the midpoint of the two frozen section
    coordinates.
    """

    base = np.asarray(normalized_position, dtype=np.float64)
    if base.ndim != 1:
        raise ValueError("normalized position must be one-dimensional")
    if spec.family == "longitudinal":
        positions = (base[:-1] + base[1:]) / 2.0
    else:
        positions = base
    if len(positions) != position_count:
        raise ValueError(
            f"{spec.name}: spatial coordinate count {len(positions)} "
            f"does not match feature position count {position_count}"
        )
    # Validate range/finite values using the same binning contract.
    normalized_spatial_bins(positions, DEFAULT_SPATIAL_BINS)
    return positions


def _finite_stat(values: np.ndarray, statistic_type: str) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan
    if statistic_type == "median":
        return float(np.median(finite))
    if statistic_type == "p95":
        return float(np.percentile(finite, 95))
    raise ValueError(f"unknown statistic type: {statistic_type}")


def _scalar_bool(value, *, field: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{field} must be a scalar boolean")
    return bool(array.reshape(-1)[0])


def spatial_profile(
    data: StabilityCaseData,
    spec: FeatureSpec,
    normalized_position: np.ndarray,
    *,
    bins: int = DEFAULT_SPATIAL_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a cervix-to-fundus spatial profile and valid count per bin.

    Each bin uses the same absolute-value statistic and anatomical side domain
    as the corresponding frozen F01-F20 feature. Values are aggregated across
    the original frame axis inside each anatomical-position bin. No spatial
    interpolation, resampling, gap filling, or cross-bin stitching is used.
    """

    profile = np.full(bins, np.nan, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    if spec.family == "curvature" and not _scalar_bool(
        data.physical_curvature_rate_available,
        field="physical_curvature_rate_available",
    ):
        return profile, counts

    values, valid = domain_values_and_valid(data, spec)
    if values.size == 0:
        return profile, counts
    if values.ndim < 2:
        raise ValueError("feature-domain values must include frame and space axes")

    position_count = int(values.shape[-1])
    positions = feature_spatial_positions(
        normalized_position, spec, position_count=position_count
    )
    spatial_bins = normalized_spatial_bins(positions, bins)
    finite_valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    absolute = np.abs(np.asarray(values, dtype=np.float64))

    for index in range(bins):
        selected_positions = spatial_bins == index
        if not np.any(selected_positions):
            continue
        selected_values = absolute[..., selected_positions]
        selected_valid = finite_valid[..., selected_positions]
        counts[index] = int(np.count_nonzero(selected_valid))
        if counts[index]:
            profile[index] = _finite_stat(
                selected_values[selected_valid], spec.statistic_type
            )
    return profile, counts


def summarize_spatial_profile_pair(
    original: np.ndarray,
    shadow: np.ndarray,
    original_counts: np.ndarray,
    shadow_counts: np.ndarray,
) -> dict[str, float | int]:
    """Summarize spatial-profile preservation while exposing availability loss."""

    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(shadow, dtype=np.float64)
    ca = np.asarray(original_counts, dtype=np.int64)
    cb = np.asarray(shadow_counts, dtype=np.int64)
    if a.shape != b.shape or a.shape != ca.shape or a.shape != cb.shape:
        raise ValueError("profile values and counts must share the same shape")
    if np.any((~np.isfinite(a)) & np.isfinite(b)):
        raise RuntimeError("mask-only spatial sensitivity created NaN-to-finite bin")
    if np.any(cb > ca):
        raise RuntimeError("mask-only spatial sensitivity increased valid positions")

    original_finite = np.isfinite(a)
    shadow_finite = np.isfinite(b)
    paired = original_finite & shadow_finite
    srd = np.full(a.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(paired):
        srd[index] = symmetric_relative_difference_pct(a[index], b[index])

    n_original = int(np.count_nonzero(original_finite))
    n_shadow = int(np.count_nonzero(shadow_finite))
    n_paired = int(np.count_nonzero(paired))
    finite_to_nan = int(np.count_nonzero(original_finite & ~shadow_finite))
    original_positions = int(ca.sum())
    shadow_positions = int(cb.sum())
    paired_srd = srd[np.isfinite(srd)]
    paired_abs_change = np.abs(b - a)[paired]

    return {
        "n_bins": int(len(a)),
        "n_original_finite_bins": n_original,
        "n_shadow_finite_bins": n_shadow,
        "n_paired_finite_bins": n_paired,
        "finite_to_nan_bins": finite_to_nan,
        "bin_availability_retention_fraction": (
            n_shadow / n_original if n_original else math.nan
        ),
        "original_valid_positions": original_positions,
        "shadow_valid_positions": shadow_positions,
        "valid_position_retention_fraction": (
            shadow_positions / original_positions if original_positions else math.nan
        ),
        "spatial_spearman": paired_spearman(a, b),
        "median_bin_srd_pct": (
            float(np.median(paired_srd)) if paired_srd.size else math.nan
        ),
        "p95_bin_srd_pct": (
            float(np.percentile(paired_srd, 95)) if paired_srd.size else math.nan
        ),
        "median_bin_absolute_change": (
            float(np.median(paired_abs_change)) if paired_abs_change.size else math.nan
        ),
        "p95_bin_absolute_change": (
            float(np.percentile(paired_abs_change, 95))
            if paired_abs_change.size
            else math.nan
        ),
    }


def spatial_bin_srd_values(original: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    """Return one SRD value per normalized anatomical-position bin."""

    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(shadow, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired profiles must have matching shapes")
    result = np.full(a.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(np.isfinite(a) & np.isfinite(b)):
        result[index] = symmetric_relative_difference_pct(a[index], b[index])
    return result
