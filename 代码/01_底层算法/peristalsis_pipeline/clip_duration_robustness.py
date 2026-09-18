from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .formal_feature_extraction import extract_formal_candidate_case_features


DEFAULT_PROPORTIONS_PCT = (100, 75, 50, 25)


@dataclass(frozen=True)
class ClipWindow:
    target_percent: int
    start: int
    stop: int
    fps: float
    full_pair_frame_count: int

    @property
    def sample_count(self) -> int:
        return self.stop - self.start

    @property
    def actual_fraction(self) -> float:
        return self.sample_count / self.full_pair_frame_count

    @property
    def actual_duration_s(self) -> float:
        return self.sample_count / self.fps

    @property
    def full_duration_s(self) -> float:
        return self.full_pair_frame_count / self.fps

    @property
    def start_s(self) -> float:
        return self.start / self.fps

    @property
    def stop_s(self) -> float:
        return self.stop / self.fps


def _scalar_fps(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("fps must be scalar")
    fps = float(array.reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    return fps


def centered_proportional_windows(
    *,
    pair_frame_count: int,
    fps: Any,
    proportions_pct: Sequence[int] = DEFAULT_PROPORTIONS_PCT,
) -> tuple[ClipWindow, ...]:
    """Create deterministic nested center windows from each case's own time axis.

    The complete formal pair-frame sequence is the 100% reference. Shorter
    windows are centered within that same sequence. Window length uses integer
    floor division, so the realized fraction never exceeds the requested
    percentage. No frames are repeated, interpolated, stitched, or selected
    according to motion strength, QC outcome, feature value, or clinical
    outcome.
    """

    if pair_frame_count < 1:
        raise ValueError("pair_frame_count must be positive")

    fps_value = _scalar_fps(fps)
    proportions = tuple(int(x) for x in proportions_pct)
    if not proportions:
        raise ValueError("proportions_pct must not be empty")
    if any(x <= 0 or x > 100 for x in proportions):
        raise ValueError("proportions_pct values must be in 1..100")
    if len(set(proportions)) != len(proportions):
        raise ValueError("proportions_pct must be unique")
    if 100 not in proportions:
        raise ValueError("proportions_pct must include the 100% reference")

    ordered = tuple(sorted(proportions, reverse=True))
    windows: list[ClipWindow] = []
    for percent in ordered:
        count = (pair_frame_count * percent) // 100
        count = max(count, 1)
        start = (pair_frame_count - count) // 2
        stop = start + count
        windows.append(
            ClipWindow(
                target_percent=percent,
                start=start,
                stop=stop,
                fps=fps_value,
                full_pair_frame_count=pair_frame_count,
            )
        )

    ref = windows[0]
    if ref.target_percent != 100 or ref.start != 0 or ref.stop != pair_frame_count:
        raise RuntimeError("100% reference must cover the full pair-frame sequence")

    for window in windows[1:]:
        if not (ref.start <= window.start < window.stop <= ref.stop):
            raise RuntimeError("nested proportional window construction failed")

    return tuple(windows)


def _slice_first_axis(array: np.ndarray, window: ClipWindow, *, field: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim < 1:
        raise ValueError(f"{field} must have a pair-frame axis")
    if value.shape[0] != window.full_pair_frame_count:
        raise ValueError(
            f"{field} must share the full pair-frame axis: "
            f"expected {window.full_pair_frame_count}, got {value.shape[0]}"
        )
    return value[window.start : window.stop]


def extract_proportional_duration_features(
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
    proportions_pct: Sequence[int] = DEFAULT_PROPORTIONS_PCT,
) -> list[dict[str, object]]:
    """Extract the frozen F01-F20 on patient-specific proportional windows."""

    rsr = np.asarray(current_qc_radial_strain_rate_s)
    if rsr.ndim < 1:
        raise ValueError("current_qc_radial_strain_rate_s must have a pair-frame axis")

    windows = centered_proportional_windows(
        pair_frame_count=rsr.shape[0],
        fps=rsr_fps,
        proportions_pct=proportions_pct,
    )

    rows: list[dict[str, object]] = []
    for window in windows:
        curvature = (
            None
            if qc_wall_curvature_change_rate_mm_inv_s is None
            else _slice_first_axis(
                qc_wall_curvature_change_rate_mm_inv_s,
                window,
                field="qc_wall_curvature_change_rate_mm_inv_s",
            )
        )
        dicom_valid = (
            None
            if dicom_pair_qc_valid is None
            else _slice_first_axis(
                dicom_pair_qc_valid,
                window,
                field="dicom_pair_qc_valid",
            )
        )

        feature_row = extract_formal_candidate_case_features(
            case_id=case_id,
            current_qc_radial_strain_rate_s=_slice_first_axis(
                rsr, window, field="current_qc_radial_strain_rate_s"
            ),
            rsr_fps=rsr_fps,
            anatomical_pair_qc_valid=_slice_first_axis(
                anatomical_pair_qc_valid,
                window,
                field="anatomical_pair_qc_valid",
            ),
            anatomical_fps=anatomical_fps,
            qc_cavity_width_strain_rate_s=_slice_first_axis(
                qc_cavity_width_strain_rate_s,
                window,
                field="qc_cavity_width_strain_rate_s",
            ),
            cavity_qc_valid=_slice_first_axis(
                cavity_qc_valid,
                window,
                field="cavity_qc_valid",
            ),
            qc_longitudinal_wall_strain_rate_s=_slice_first_axis(
                qc_longitudinal_wall_strain_rate_s,
                window,
                field="qc_longitudinal_wall_strain_rate_s",
            ),
            longitudinal_qc_valid=_slice_first_axis(
                longitudinal_qc_valid,
                window,
                field="longitudinal_qc_valid",
            ),
            qc_wall_curvature_change_rate_mm_inv_s=curvature,
            dicom_pair_qc_valid=dicom_valid,
            physical_curvature_available=physical_curvature_available,
            physical_curvature_rate_available=physical_curvature_rate_available,
        )
        rows.append(
            {
                "target_percent": window.target_percent,
                "target_fraction": window.target_percent / 100.0,
                "actual_fraction": window.actual_fraction,
                "full_pair_frames": window.full_pair_frame_count,
                "window_pair_frames": window.sample_count,
                "full_duration_s": window.full_duration_s,
                "window_duration_s": window.actual_duration_s,
                "window_start_pair_frame": window.start,
                "window_stop_pair_frame_exclusive": window.stop,
                "window_start_s": window.start_s,
                "window_stop_s": window.stop_s,
                **feature_row,
            }
        )
    return rows
