"""Adjacent-frame anatomical deformation features from radial LK rails."""

from __future__ import annotations

import numpy as np


def _safe_unit(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = np.linalg.norm(vectors, axis=-1)
    unit = np.divide(
        vectors,
        length[..., None],
        out=np.full_like(vectors, np.nan, dtype=np.float32),
        where=length[..., None] > 1e-6,
    )
    return unit.astype(np.float32), length.astype(np.float32)


def _three_point_curvature(
    points: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Return unsigned three-point curvature along each ordered wall curve."""
    coordinates = np.asarray(points, dtype=np.float32)
    point_valid = np.asarray(valid, dtype=bool)
    if coordinates.ndim != 4 or coordinates.shape[-1] != 2:
        raise ValueError("wall points must have shape (frame, side, section, 2)")
    if point_valid.shape != coordinates.shape[:-1]:
        raise ValueError("wall point validity must match frame, side, and section")

    curvature = np.full(coordinates.shape[:-1], np.nan, dtype=np.float32)
    if coordinates.shape[2] < 3:
        return curvature

    a_point = coordinates[:, :, :-2]
    b_point = coordinates[:, :, 1:-1]
    c_point = coordinates[:, :, 2:]
    ab = b_point - a_point
    bc = c_point - b_point
    ac = c_point - a_point
    a_length = np.linalg.norm(ab, axis=-1)
    b_length = np.linalg.norm(bc, axis=-1)
    c_length = np.linalg.norm(ac, axis=-1)
    cross = np.abs(ab[..., 0] * ac[..., 1] - ab[..., 1] * ac[..., 0])
    tri_valid = (
        point_valid[:, :, :-2]
        & point_valid[:, :, 1:-1]
        & point_valid[:, :, 2:]
        & (a_length > 1e-6)
        & (b_length > 1e-6)
        & (c_length > 1e-6)
    )
    denominator = a_length * b_length * c_length
    interior = np.divide(
        2.0 * cross,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float32),
        where=tri_valid,
    )
    curvature[:, :, 1:-1] = interior
    return curvature


def anatomical_deformation_from_tracking(
    tracking: dict[str, np.ndarray],
    *,
    section_count: int,
    fps: float,
) -> dict[str, np.ndarray]:
    """Measure cavity, normal, and along-wall deformation without P3 deltas."""
    if section_count < 2 or fps <= 0.0:
        raise ValueError("at least two sections and positive fps are required")
    source = np.asarray(tracking["radial_pair_source"], dtype=np.float32)
    measured_key = (
        "radial_lk_measured"
        if "radial_lk_measured" in tracking
        else "radial_lk_measured_p3_initial"
    )
    measured = np.asarray(tracking[measured_key], dtype=np.float32)
    global_prediction = np.asarray(
        tracking["radial_global_prediction"], dtype=np.float32
    )
    pair_valid = np.asarray(tracking["radial_pair_valid"], dtype=bool)
    expected_points = 4 * section_count
    if (
        source.shape != measured.shape
        or source.shape != global_prediction.shape
        or source.shape[1:] != (expected_points, 2)
        or pair_valid.shape != (len(source), 2, section_count)
    ):
        raise ValueError("tracking arrays do not match four radial rails")

    source_centers = np.full((len(source), 2, section_count, 2), np.nan, np.float32)
    measured_centers = np.full_like(source_centers, np.nan)
    global_centers = np.full_like(source_centers, np.nan)
    radial_units = np.full_like(source_centers, np.nan)
    for side, (inner_rail, outer_rail) in enumerate(((0, 1), (2, 3))):
        inner = slice(inner_rail * section_count, (inner_rail + 1) * section_count)
        outer = slice(outer_rail * section_count, (outer_rail + 1) * section_count)
        source_centers[:, side] = 0.5 * (source[:, inner] + source[:, outer])
        measured_centers[:, side] = 0.5 * (
            measured[:, inner] + measured[:, outer]
        )
        global_centers[:, side] = 0.5 * (
            global_prediction[:, inner] + global_prediction[:, outer]
        )
        radial_units[:, side], _ = _safe_unit(source[:, outer] - source[:, inner])

    local_displacement = measured_centers - global_centers
    wall_normal_velocity = (
        np.sum(local_displacement * radial_units, axis=-1) * float(fps)
    ).astype(np.float32)

    cavity_source_vector = source_centers[:, 1] - source_centers[:, 0]
    cavity_measured_vector = measured_centers[:, 1] - measured_centers[:, 0]
    cavity_unit, cavity_source_width = _safe_unit(cavity_source_vector)
    cavity_measured_width = np.linalg.norm(cavity_measured_vector, axis=-1).astype(
        np.float32
    )
    cavity_width_strain_rate = np.divide(
        (cavity_measured_width - cavity_source_width) * float(fps),
        cavity_source_width,
        out=np.full_like(cavity_source_width, np.nan),
        where=cavity_source_width > 1e-6,
    ).astype(np.float32)
    cavity_closing_rate = (
        (cavity_source_width - cavity_measured_width) * float(fps)
    ).astype(np.float32)

    anterior_inward = (
        np.sum(local_displacement[:, 0] * cavity_unit, axis=-1) * float(fps)
    )
    posterior_inward = (
        -np.sum(local_displacement[:, 1] * cavity_unit, axis=-1) * float(fps)
    )
    wall_inward_velocity = np.stack((anterior_inward, posterior_inward), axis=1).astype(
        np.float32
    )
    coordinated_inward_velocity = np.mean(wall_inward_velocity, axis=1).astype(
        np.float32
    )
    inward_velocity_asymmetry = np.abs(
        wall_inward_velocity[:, 0] - wall_inward_velocity[:, 1]
    ).astype(np.float32)
    inward_same_sign = (
        wall_inward_velocity[:, 0] * wall_inward_velocity[:, 1] > 0.0
    )

    longitudinal_strain_rate = np.full(
        (len(source), 2, section_count - 1), np.nan, dtype=np.float32
    )
    longitudinal_valid = np.zeros(longitudinal_strain_rate.shape, dtype=bool)
    for side in range(2):
        source_segment = np.diff(source_centers[:, side], axis=1)
        measured_segment = np.diff(measured_centers[:, side], axis=1)
        source_length = np.linalg.norm(source_segment, axis=-1)
        measured_length = np.linalg.norm(measured_segment, axis=-1)
        longitudinal_strain_rate[:, side] = np.divide(
            (measured_length - source_length) * float(fps),
            source_length,
            out=np.full_like(source_length, np.nan),
            where=source_length > 1e-6,
        )
        longitudinal_valid[:, side] = (
            pair_valid[:, side, :-1] & pair_valid[:, side, 1:]
        )

    cavity_valid = pair_valid[:, 0] & pair_valid[:, 1]
    pair_valid = pair_valid.copy()
    pair_valid[0] = False
    cavity_valid[0] = False
    longitudinal_valid[0] = False
    source_curvature = _three_point_curvature(source_centers, pair_valid)
    measured_curvature = _three_point_curvature(measured_centers, pair_valid)
    # Curvature is unsigned: the rate sign means more or less curved, not direction.
    curvature_change_rate = (
        (measured_curvature - source_curvature) * float(fps)
    ).astype(np.float32)
    wall_normal_velocity[~pair_valid] = np.nan
    wall_inward_velocity[~pair_valid] = np.nan
    cavity_width_strain_rate[~cavity_valid] = np.nan
    cavity_closing_rate[~cavity_valid] = np.nan
    coordinated_inward_velocity[~cavity_valid] = np.nan
    inward_velocity_asymmetry[~cavity_valid] = np.nan
    inward_same_sign[~cavity_valid] = False
    longitudinal_strain_rate[~longitudinal_valid] = np.nan

    return {
        "wall_center_source": source_centers,
        "wall_center_measured": measured_centers,
        "wall_center_local_displacement_px": local_displacement,
        "wall_normal_velocity_px_s": wall_normal_velocity,
        "wall_inward_velocity_px_s": wall_inward_velocity,
        "cavity_source_width_px": cavity_source_width,
        "cavity_measured_width_px": cavity_measured_width,
        "cavity_width_strain_rate_s": cavity_width_strain_rate,
        "cavity_closing_rate_px_s": cavity_closing_rate,
        "coordinated_inward_velocity_px_s": coordinated_inward_velocity,
        "inward_velocity_asymmetry_px_s": inward_velocity_asymmetry,
        "inward_same_sign": inward_same_sign,
        "longitudinal_wall_strain_rate_s": longitudinal_strain_rate,
        "wall_curvature_px_inv": measured_curvature,
        "wall_curvature_change_rate_px_inv_s": curvature_change_rate,
        "wall_feature_valid": pair_valid,
        "cavity_feature_valid": cavity_valid,
        "longitudinal_feature_valid": longitudinal_valid,
        "p3_coordinate_delta_used_as_deformation": np.asarray(False),
    }
