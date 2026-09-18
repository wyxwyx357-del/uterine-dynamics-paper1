from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .formal_feature_extraction import extract_formal_candidate_case_features


DEFAULT_DURATIONS_S = (60, 30, 20, 10)


@dataclass(frozen=True)
class ClipWindow:
    duration_s: int
    start: int
    stop: int
    fps: float

    @property
    def sample_count(self) -> int:
        return self.stop - self.start

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


def centered_nested_windows(
    *,
    pair_frame_count: int,
    fps: Any,
    durations_s: Sequence[int] = DEFAULT_DURATIONS_S,
) -> tuple[ClipWindow, ...]:
    """Create deterministic nested center windows on the pair-frame time axis.

    The longest requested duration is centered within the available pair-frame
    sequence. Every shorter duration is centered inside that same longest
    reference window. No frames are repeated, interpolated, or stitched.
    """

    if pair_frame_count < 1:
        raise ValueError("pair_frame_count must be positive")

    fps_value = _scalar_fps(fps)
    durations = tuple(int(x) for x in durations_s)
    if not durations or any(x <= 0 for x in durations):
        raise ValueError("durations_s must contain positive integers")
    if len(set(durations)) != len(durations):
        raise ValueError("durations_s must be unique")

    ordered = tuple(sorted(durations, reverse=True))
    counts = {d: int(round(d * fps_value)) for d in ordered}
    if any(n <= 0 for n in counts.values()):
        raise ValueError("duration produced an empty window")

    ref_duration = ordered[0]
    ref_count = counts[ref_duration]
    if pair_frame_count < ref_count:
        raise ValueError(
            f"insufficient pair-frame duration for {ref_duration}s reference: "
            f"need {ref_count}, have {pair_frame_count}"
        )

    ref_start = (pair_frame_count - ref_count) // 2
    ref_stop = ref_start + ref_count

    windows: list[ClipWindow] = []
    for duration in ordered:
        count = counts[duration]
        start = ref_start + (ref_count - count) // 2
        stop = start + count
        if start < ref_start or stop > ref_stop:
            raise RuntimeError("nested window construction failed")
        windows.append(
            ClipWindow(
                duration_s=duration,
                start=start,
                stop=stop,
                fps=fps_value,
            )
        )
    return tuple(windows)


def _slice_first_axis(array: np.ndarray, window: ClipWindow, *, field: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim < 1:
        raise ValueError(f"{field} must have a pair-frame axis")
    if value.shape[0] < window.stop:
        raise ValueError(
            f"{field} has insufficient pair frames: need stop={window.stop}, "
            f"shape={value.shape}"
        )
    return value[window.start : window.stop]


def extract_duration_features(
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
    durations_s: Sequence[int] = DEFAULT_DURATIONS_S,
) -> list[dict[str, object]]:
    """Extract frozen F01-F20 on deterministic nested duration windows."""

    rsr = np.asarray(current_qc_radial_strain_rate_s)
    if rsr.ndim < 1:
        raise ValueError("current_qc_radial_strain_rate_s must have a pair-frame axis")

    windows = centered_nested_windows(
        pair_frame_count=rsr.shape[0],
        fps=rsr_fps,
        durations_s=durations_s,
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
                "duration_s": window.duration_s,
                "window_start_pair_frame": window.start,
                "window_stop_pair_frame_exclusive": window.stop,
                "window_start_s": window.start_s,
                "window_stop_s": window.stop_s,
                "window_pair_frames": window.sample_count,
                **feature_row,
            }
        )
    return rows
