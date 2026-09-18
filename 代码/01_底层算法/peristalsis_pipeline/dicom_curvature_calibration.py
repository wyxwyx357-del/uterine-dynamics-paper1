"""Independent DICOM physical calibration for existing wall curvature.

This module does not track images or change any existing QC decision.  It maps
the already frozen wall-center coordinates into a calibrated ultrasound region
and reuses the existing three-point curvature definition in millimetres.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .anatomical_deformation_features import _three_point_curvature


SPATIAL_DISTANCE_UNIT_CODE_CM = 3
# These tolerances audit DICOM timing against the analysis time base.  They are
# not part of the curvature-rate formula, which uses each DICOM pair interval.
DICOM_TIMING_MATCH_RTOL = 1e-3
DICOM_TIMING_MATCH_ATOL_S = 5e-4
DICOM_CURVATURE_RATE_TIMEBASE = "dicom_pair_dt_s"


def validate_dicom_curvature_rate_lineage(
    payload: Mapping[str, Any], *, source: object
) -> None:
    """Reject DICOM-rate files not produced from verified pair timing."""

    expected_fields = (
        "physical_curvature_rate_available",
        "curvature_rate_timebase",
        "dicom_timing_numeric_match",
    )
    missing = [field for field in expected_fields if field not in payload]
    if missing:
        raise KeyError(
            f"{source}: required DICOM-rate lineage fields are missing: "
            f"{', '.join(missing)}"
        )

    timebase_array = np.asarray(payload["curvature_rate_timebase"])
    numeric_match_array = np.asarray(payload["dicom_timing_numeric_match"])
    rate_available_array = np.asarray(payload["physical_curvature_rate_available"])
    if timebase_array.size != 1:
        raise ValueError(f"{source}: curvature_rate_timebase must be scalar")
    if numeric_match_array.size != 1:
        raise ValueError(f"{source}: dicom_timing_numeric_match must be scalar")
    if rate_available_array.size != 1:
        raise ValueError(
            f"{source}: physical_curvature_rate_available must be scalar"
        )
    actual_timebase = str(timebase_array.reshape(-1)[0])
    if actual_timebase != DICOM_CURVATURE_RATE_TIMEBASE:
        raise ValueError(
            f"{source}: curvature_rate_timebase mismatch; "
            f"expected={DICOM_CURVATURE_RATE_TIMEBASE!r}, "
            f"actual={actual_timebase!r}"
        )
    if bool(rate_available_array.reshape(-1)[0]) and not bool(
        numeric_match_array.reshape(-1)[0]
    ):
        raise ValueError(
            f"{source}: physical curvature rate requires a successful "
            "numeric DICOM timing match"
        )


@dataclass(frozen=True)
class UltrasoundRegionCalibration:
    """One item from DICOM Sequence of Ultrasound Regions."""

    index: int
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    physical_units_x: int
    physical_units_y: int
    physical_delta_x: float
    physical_delta_y: float
    region_spatial_format: int | None = None
    region_data_type: int | None = None

    def __post_init__(self) -> None:
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError("ultrasound region bounds are reversed")
        if self.physical_units_x != SPATIAL_DISTANCE_UNIT_CODE_CM:
            raise ValueError("ultrasound region X unit is not spatial distance in cm")
        if self.physical_units_y != SPATIAL_DISTANCE_UNIT_CODE_CM:
            raise ValueError("ultrasound region Y unit is not spatial distance in cm")
        for value, axis in (
            (self.physical_delta_x, "X"),
            (self.physical_delta_y, "Y"),
        ):
            if not np.isfinite(value) or value == 0.0:
                raise ValueError(f"ultrasound region Physical Delta {axis} is invalid")

    @property
    def delta_x_mm_per_pixel(self) -> float:
        return float(self.physical_delta_x) * 10.0

    @property
    def delta_y_mm_per_pixel(self) -> float:
        return float(self.physical_delta_y) * 10.0

    def contains(self, points: np.ndarray, valid: np.ndarray) -> bool:
        coordinates = np.asarray(points, dtype=np.float64)
        point_valid = np.asarray(valid, dtype=bool)
        if coordinates.shape[:-1] != point_valid.shape or coordinates.shape[-1] != 2:
            raise ValueError("point validity must match coordinates")
        usable = point_valid & np.all(np.isfinite(coordinates), axis=-1)
        if not np.any(usable):
            return False
        selected = coordinates[usable]
        return bool(
            np.all(selected[:, 0] >= self.min_x)
            and np.all(selected[:, 0] <= self.max_x)
            and np.all(selected[:, 1] >= self.min_y)
            and np.all(selected[:, 1] <= self.max_y)
        )


@dataclass(frozen=True)
class DicomCineMetadata:
    study_instance_uid: str
    series_instance_uid: str
    frame_count: int
    rows: int
    columns: int
    pair_dt_s: np.ndarray
    regions: tuple[UltrasoundRegionCalibration, ...]


def _required_value(item: Any, name: str) -> Any:
    if not hasattr(item, name):
        raise ValueError(f"DICOM ultrasound region is missing {name}")
    return getattr(item, name)


def ultrasound_regions_from_dataset(
    dataset: Any,
) -> tuple[UltrasoundRegionCalibration, ...]:
    """Read spatial-distance ultrasound regions without choosing the first one."""

    sequence = getattr(dataset, "SequenceOfUltrasoundRegions", ())
    regions: list[UltrasoundRegionCalibration] = []
    for index, item in enumerate(sequence):
        try:
            regions.append(
                UltrasoundRegionCalibration(
                    index=index,
                    min_x=int(_required_value(item, "RegionLocationMinX0")),
                    min_y=int(_required_value(item, "RegionLocationMinY0")),
                    max_x=int(_required_value(item, "RegionLocationMaxX1")),
                    max_y=int(_required_value(item, "RegionLocationMaxY1")),
                    physical_units_x=int(
                        _required_value(item, "PhysicalUnitsXDirection")
                    ),
                    physical_units_y=int(
                        _required_value(item, "PhysicalUnitsYDirection")
                    ),
                    physical_delta_x=float(
                        _required_value(item, "PhysicalDeltaX")
                    ),
                    physical_delta_y=float(
                        _required_value(item, "PhysicalDeltaY")
                    ),
                    region_spatial_format=(
                        int(item.RegionSpatialFormat)
                        if hasattr(item, "RegionSpatialFormat")
                        else None
                    ),
                    region_data_type=(
                        int(item.RegionDataType)
                        if hasattr(item, "RegionDataType")
                        else None
                    ),
                )
            )
        except ValueError:
            # Non-distance or incomplete regions remain unavailable for physical
            # curvature; they are not silently selected as a fallback.
            continue
    return tuple(regions)


def _pair_dt_from_dataset(dataset: Any, frame_count: int) -> np.ndarray:
    pair_dt_s = np.full(frame_count, np.nan, dtype=np.float32)
    vector = getattr(dataset, "FrameTimeVector", None)
    if vector is not None:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if len(values) != frame_count:
            raise ValueError("DICOM Frame Time Vector length does not match frames")
        pair_dt_s[:] = values / 1000.0
        pair_dt_s[0] = np.nan
        return pair_dt_s
    frame_time_ms = getattr(dataset, "FrameTime", None)
    if frame_time_ms is not None:
        value = float(frame_time_ms) / 1000.0
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("DICOM Frame Time is invalid")
        pair_dt_s[1:] = value
    return pair_dt_s


def read_dicom_cine_metadata(paths: Sequence[Path]) -> DicomCineMetadata:
    """Read one multi-frame DICOM or one single-frame DICOM series.

    Pixel data are deliberately not decoded here.  Visual DICOM/MP4 matching is
    a separate required step whose result is supplied to the calibration runner.
    """

    if not paths:
        raise ValueError("at least one DICOM file is required")
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("DICOM calibration requires the pydicom package") from exc

    datasets = [pydicom.dcmread(str(Path(path)), stop_before_pixels=True) for path in paths]
    study_uids = {str(getattr(dataset, "StudyInstanceUID", "")) for dataset in datasets}
    series_uids = {str(getattr(dataset, "SeriesInstanceUID", "")) for dataset in datasets}
    if len(study_uids) != 1 or "" in study_uids:
        raise ValueError("DICOM files do not have one Study Instance UID")
    if len(series_uids) != 1 or "" in series_uids:
        raise ValueError("DICOM files do not have one Series Instance UID")

    rows = {int(getattr(dataset, "Rows", 0)) for dataset in datasets}
    columns = {int(getattr(dataset, "Columns", 0)) for dataset in datasets}
    if len(rows) != 1 or 0 in rows or len(columns) != 1 or 0 in columns:
        raise ValueError("DICOM files do not share one valid image size")

    per_file_frames = [int(getattr(dataset, "NumberOfFrames", 1)) for dataset in datasets]
    frame_count = int(sum(per_file_frames))
    if len(datasets) == 1:
        pair_dt_s = _pair_dt_from_dataset(datasets[0], frame_count)
    else:
        pair_dt_s = np.full(frame_count, np.nan, dtype=np.float32)
        frame_time_values = {
            float(dataset.FrameTime)
            for dataset in datasets
            if hasattr(dataset, "FrameTime")
        }
        if len(frame_time_values) == 1 and all(
            hasattr(dataset, "FrameTime") for dataset in datasets
        ):
            frame_time_s = next(iter(frame_time_values)) / 1000.0
            if np.isfinite(frame_time_s) and frame_time_s > 0.0:
                pair_dt_s[1:] = frame_time_s

    regions = ultrasound_regions_from_dataset(datasets[0])
    for dataset in datasets[1:]:
        if ultrasound_regions_from_dataset(dataset) != regions:
            raise ValueError("DICOM series does not share one ultrasound calibration")
    return DicomCineMetadata(
        study_instance_uid=next(iter(study_uids)),
        series_instance_uid=next(iter(series_uids)),
        frame_count=frame_count,
        rows=next(iter(rows)),
        columns=next(iter(columns)),
        pair_dt_s=pair_dt_s,
        regions=regions,
    )


def apply_analysis_to_dicom_transform(
    points: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Apply one verified affine analysis-to-DICOM pixel transform."""

    coordinates = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if coordinates.ndim < 2 or coordinates.shape[-1] != 2:
        raise ValueError("points must end with an x,y coordinate axis")
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("analysis-to-DICOM transform must be a finite 3x3 matrix")
    if not np.allclose(matrix[2], np.asarray([0.0, 0.0, 1.0])):
        raise ValueError("analysis-to-DICOM transform must be affine")

    result = np.full(coordinates.shape, np.nan, dtype=np.float32)
    finite = np.all(np.isfinite(coordinates), axis=-1)
    if np.any(finite):
        selected = coordinates[finite]
        homogeneous = np.column_stack((selected, np.ones(len(selected))))
        mapped = homogeneous @ matrix.T
        result[finite] = mapped[:, :2].astype(np.float32)
    return result


def dicom_pixels_to_mm(
    points: np.ndarray, region: UltrasoundRegionCalibration
) -> np.ndarray:
    coordinates = np.asarray(points, dtype=np.float32)
    result = np.full_like(coordinates, np.nan, dtype=np.float32)
    result[..., 0] = (
        coordinates[..., 0] - float(region.min_x)
    ) * region.delta_x_mm_per_pixel
    result[..., 1] = (
        coordinates[..., 1] - float(region.min_y)
    ) * region.delta_y_mm_per_pixel
    return result


def curvature_change_rate_from_pair_dt(
    curvature_change_mm_inv: np.ndarray,
    pair_dt_s: np.ndarray,
    valid_pair_frames: np.ndarray,
) -> np.ndarray:
    """Convert curvature changes using explicit per-pair DICOM intervals."""

    change = np.asarray(curvature_change_mm_inv, dtype=np.float32)
    if change.ndim < 1:
        raise ValueError("curvature change must have a frame axis")
    dt = np.asarray(pair_dt_s, dtype=np.float64)
    if dt.shape != (change.shape[0],):
        raise ValueError("DICOM pair_dt_s must have shape (frame_count,)")
    valid_frames = np.asarray(valid_pair_frames, dtype=bool)
    if valid_frames.shape != dt.shape:
        raise ValueError("valid_pair_frames must match DICOM pair_dt_s")
    if np.any(~np.isfinite(dt[valid_frames])) or np.any(dt[valid_frames] <= 0.0):
        raise ValueError(
            "DICOM pair_dt_s must be finite and positive for every valid pair"
        )

    result = np.full_like(change, np.nan, dtype=np.float32)
    if np.any(valid_frames):
        denominator_shape = (np.count_nonzero(valid_frames),) + (1,) * (
            change.ndim - 1
        )
        selected_change = change[valid_frames]
        computed = selected_change / dt[valid_frames].reshape(denominator_shape)
        finite_input = np.isfinite(selected_change)
        if np.any(finite_input & ~np.isfinite(computed)):
            raise ValueError("DICOM pair timing produced a non-finite curvature rate")
        result[valid_frames] = computed.astype(np.float32)
    return result


def validate_dicom_timing_against_analysis_fps(
    *,
    pair_dt_s: np.ndarray,
    valid_pair_frames: np.ndarray,
    analysis_fps: float,
) -> None:
    """Fail closed when DICOM pair timing disagrees with the analysis lineage."""

    if not np.isfinite(analysis_fps) or analysis_fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    dt = np.asarray(pair_dt_s, dtype=np.float64)
    valid_frames = np.asarray(valid_pair_frames, dtype=bool)
    if dt.shape != valid_frames.shape:
        raise ValueError("valid_pair_frames must match DICOM pair_dt_s")
    selected = dt[valid_frames]
    if np.any(~np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError(
            "DICOM pair_dt_s must be finite and positive for every valid pair"
        )
    expected_dt_s = 1.0 / float(analysis_fps)
    if not np.allclose(
        selected,
        expected_dt_s,
        rtol=DICOM_TIMING_MATCH_RTOL,
        atol=DICOM_TIMING_MATCH_ATOL_S,
    ):
        actual_min = float(np.min(selected)) if selected.size else float("nan")
        actual_max = float(np.max(selected)) if selected.size else float("nan")
        raise ValueError(
            "DICOM timing mismatch with analysis fps; "
            f"expected_pair_dt_s={expected_dt_s!r}, "
            f"actual_pair_dt_s_range=({actual_min!r}, {actual_max!r})"
        )


def physical_curvature_from_wall_centers(
    *,
    wall_center_source: np.ndarray,
    wall_center_measured: np.ndarray,
    pair_qc_valid: np.ndarray,
    analysis_to_dicom_transform: np.ndarray,
    region: UltrasoundRegionCalibration,
    fps: float,
    timing_matched: bool,
    pair_dt_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return calibrated curvature without changing the existing formal mask."""

    source = np.asarray(wall_center_source, dtype=np.float32)
    measured = np.asarray(wall_center_measured, dtype=np.float32)
    valid = np.asarray(pair_qc_valid, dtype=bool)
    if source.shape != measured.shape or source.ndim != 4 or source.shape[-1] != 2:
        raise ValueError("wall centers must share shape (frame, side, section, 2)")
    if valid.shape != source.shape[:-1]:
        raise ValueError("pair_qc_valid must match wall-center frame, side, section")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")

    source_dicom = apply_analysis_to_dicom_transform(
        source, analysis_to_dicom_transform
    )
    measured_dicom = apply_analysis_to_dicom_transform(
        measured, analysis_to_dicom_transform
    )
    if not region.contains(source_dicom, valid):
        raise ValueError("selected ultrasound region does not cover valid source points")
    if not region.contains(measured_dicom, valid):
        raise ValueError("selected ultrasound region does not cover valid measured points")

    source_mm = dicom_pixels_to_mm(source_dicom, region)
    measured_mm = dicom_pixels_to_mm(measured_dicom, region)
    source_curvature = _three_point_curvature(source_mm, valid)
    measured_curvature = _three_point_curvature(measured_mm, valid)
    curvature_change_rate = np.full_like(measured_curvature, np.nan)
    if timing_matched:
        if pair_dt_s is None:
            raise ValueError(
                "DICOM pair_dt_s is required when timing is marked matched"
            )
        valid_pair_frames = np.any(valid, axis=(1, 2))
        validate_dicom_timing_against_analysis_fps(
            pair_dt_s=pair_dt_s,
            valid_pair_frames=valid_pair_frames,
            analysis_fps=float(fps),
        )
        curvature_change_rate = curvature_change_rate_from_pair_dt(
            measured_curvature - source_curvature,
            pair_dt_s,
            valid_pair_frames,
        )

    return {
        "wall_center_source_mm": source_mm,
        "wall_center_measured_mm": measured_mm,
        "qc_wall_curvature_mm_inv": measured_curvature,
        "qc_wall_curvature_change_rate_mm_inv_s": curvature_change_rate,
    }
