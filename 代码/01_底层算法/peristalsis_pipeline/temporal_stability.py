"""Normalized-time profile utilities for Paper 1 temporal-structure robustness.

This module does not infer waves, frequency, direction, speed, or propagation.
It summarizes the already frozen measurement arrays inside fixed normalized
time bins and compares a baseline profile with a mask-only sensitivity copy.
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


DEFAULT_TEMPORAL_BINS = 5


def normalized_time_bins(frame_count: int, bins: int = DEFAULT_TEMPORAL_BINS) -> np.ndarray:
    """Assign every frame to exactly one equal-width normalized-time bin."""

    if frame_count < 0:
        raise ValueError("frame_count must be nonnegative")
    if bins < 1:
        raise ValueError("bins must be positive")
    if frame_count == 0:
        return np.asarray([], dtype=np.int32)
    result = (np.arange(frame_count, dtype=np.int64) * bins) // frame_count
    return np.minimum(result, bins - 1).astype(np.int32)


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


def temporal_profile(
    data: StabilityCaseData,
    spec: FeatureSpec,
    *,
    bins: int = DEFAULT_TEMPORAL_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one normalized-time profile and valid-position count per bin.

    Each bin uses the same absolute-value statistic and anatomical domain as the
    corresponding frozen F01-F20 feature. No interpolation, temporal resampling,
    gap filling, or cross-bin stitching is performed.
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
    if values.ndim < 1:
        raise ValueError("feature-domain values must have a frame axis")

    frame_bins = normalized_time_bins(values.shape[0], bins)
    finite_valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    absolute = np.abs(np.asarray(values, dtype=np.float64))

    for index in range(bins):
        selected_frames = frame_bins == index
        if not np.any(selected_frames):
            continue
        selected_values = absolute[selected_frames]
        selected_valid = finite_valid[selected_frames]
        counts[index] = int(np.count_nonzero(selected_valid))
        if counts[index]:
            profile[index] = _finite_stat(
                selected_values[selected_valid], spec.statistic_type
            )
    return profile, counts


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling, 1-based."""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def paired_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation for paired finite values; undefined cases are NaN."""

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired profiles must have matching shapes")
    paired = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(paired)) < 2:
        return math.nan
    ra = _average_ranks(a[paired])
    rb = _average_ranks(b[paired])
    if np.ptp(ra) == 0.0 or np.ptp(rb) == 0.0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_profile_pair(
    original: np.ndarray,
    shadow: np.ndarray,
    original_counts: np.ndarray,
    shadow_counts: np.ndarray,
) -> dict[str, float | int]:
    """Summarize temporal profile preservation without hiding availability loss."""

    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(shadow, dtype=np.float64)
    ca = np.asarray(original_counts, dtype=np.int64)
    cb = np.asarray(shadow_counts, dtype=np.int64)
    if a.shape != b.shape or a.shape != ca.shape or a.shape != cb.shape:
        raise ValueError("profile values and counts must share the same shape")
    if np.any((~np.isfinite(a)) & np.isfinite(b)):
        raise RuntimeError("mask-only temporal sensitivity created NaN-to-finite bin")
    if np.any(cb > ca):
        raise RuntimeError("mask-only temporal sensitivity increased valid positions")

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
    abs_change = np.abs(b - a)
    paired_abs_change = abs_change[paired]

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
        "temporal_spearman": paired_spearman(a, b),
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


def bin_srd_values(original: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    """Return one SRD value per normalized-time bin."""

    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(shadow, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired profiles must have matching shapes")
    result = np.full(a.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(np.isfinite(a) & np.isfinite(b)):
        result[index] = symmetric_relative_difference_pct(a[index], b[index])
    return result
