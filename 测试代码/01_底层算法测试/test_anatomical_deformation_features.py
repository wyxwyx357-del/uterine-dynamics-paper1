from __future__ import annotations

import unittest

import numpy as np

from peristalsis_pipeline.anatomical_deformation_features import (
    anatomical_deformation_from_tracking,
)


def tracking_case(
    anterior: np.ndarray,
    posterior: np.ndarray,
    measured_anterior: np.ndarray,
    measured_posterior: np.ndarray,
    global_translation: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray]:
    frames, sections = anterior.shape[:2]

    def rails(ant: np.ndarray, post: np.ndarray) -> np.ndarray:
        offset = np.asarray([0.0, 0.5], dtype=np.float32)
        return np.concatenate((ant - offset, ant + offset, post - offset, post + offset), axis=1)

    source = rails(anterior, posterior)
    measured = rails(measured_anterior, measured_posterior)
    global_prediction = source + np.asarray(global_translation, dtype=np.float32)
    return {
        "radial_pair_source": source,
        "radial_lk_measured_p3_initial": measured,
        "radial_global_prediction": global_prediction,
        "radial_pair_valid": np.ones((frames, 2, sections), dtype=bool),
    }


class AnatomicalDeformationFeatureTest(unittest.TestCase):
    @staticmethod
    def circle_points(radius: float, frames: int = 2) -> np.ndarray:
        angles = np.linspace(-0.4, 0.4, 5, dtype=np.float32)
        points = np.stack(
            (radius * np.cos(angles), radius * np.sin(angles)), axis=-1
        ).astype(np.float32)
        return np.repeat(points[None], frames, axis=0)

    def test_rigid_translation_is_zero_after_global_prediction(self):
        ant = np.asarray([[[0, 0], [10, 0], [20, 0]]] * 2, dtype=np.float32)
        post = ant + [0, 10]
        shift = np.asarray([2, 3], dtype=np.float32)
        tracking = tracking_case(ant, post, ant + shift, post + shift, (2, 3))
        result = anatomical_deformation_from_tracking(
            tracking, section_count=3, fps=1.0
        )
        self.assertLess(float(np.nanmax(np.abs(result["cavity_width_strain_rate_s"]))), 1e-6)
        self.assertLess(float(np.nanmax(np.abs(result["longitudinal_wall_strain_rate_s"]))), 1e-6)
        self.assertLess(float(np.nanmax(np.abs(result["wall_inward_velocity_px_s"]))), 1e-6)

    def test_symmetric_closure_has_negative_width_strain_and_positive_inward_motion(self):
        ant = np.asarray([[[0, 0], [10, 0], [20, 0]]] * 2, dtype=np.float32)
        post = ant + [0, 10]
        measured_ant = ant.copy()
        measured_post = post.copy()
        measured_ant[1, :, 1] += 1
        measured_post[1, :, 1] -= 1
        result = anatomical_deformation_from_tracking(
            tracking_case(ant, post, measured_ant, measured_post),
            section_count=3,
            fps=1.0,
        )
        self.assertTrue(np.allclose(result["cavity_width_strain_rate_s"][1], -0.2))
        self.assertTrue(np.allclose(result["wall_inward_velocity_px_s"][1], 1.0))
        self.assertTrue(np.all(result["inward_same_sign"][1]))

    def test_along_wall_stretch_is_detected(self):
        ant = np.asarray([[[0, 0], [10, 0], [20, 0]]] * 2, dtype=np.float32)
        post = ant + [0, 10]
        measured_ant = ant.copy()
        measured_post = post.copy()
        measured_ant[1, :, 0] = [0, 11, 22]
        measured_post[1, :, 0] = [0, 11, 22]
        result = anatomical_deformation_from_tracking(
            tracking_case(ant, post, measured_ant, measured_post),
            section_count=3,
            fps=1.0,
        )
        self.assertTrue(
            np.allclose(result["longitudinal_wall_strain_rate_s"][1], 0.1)
        )

    def test_curvature_known_geometry_sides_endpoints_and_validity(self):
        ant = self.circle_points(10.0)
        x = np.linspace(-10.0, 10.0, 5, dtype=np.float32)
        post = np.repeat(np.stack((x, np.full_like(x, 20.0)), axis=-1)[None], 2, axis=0)
        tracking = tracking_case(ant, post, ant, post)
        result = anatomical_deformation_from_tracking(
            tracking, section_count=5, fps=10.0
        )
        curvature = result["wall_curvature_px_inv"]
        self.assertEqual(curvature.shape, (2, 2, 5))
        self.assertTrue(np.all(np.isnan(curvature[0])))
        self.assertTrue(np.all(np.isnan(curvature[1, :, [0, -1]])))
        self.assertTrue(np.allclose(curvature[1, 0, 1:-1], 0.1, atol=1e-5))
        self.assertTrue(np.allclose(curvature[1, 1, 1:-1], 0.0, atol=1e-6))

        tracking["radial_pair_valid"][1, 0, 2] = False
        invalid_result = anatomical_deformation_from_tracking(
            tracking, section_count=5, fps=10.0
        )
        self.assertTrue(
            np.all(np.isnan(invalid_result["wall_curvature_px_inv"][1, 0, 1:-1]))
        )
        self.assertTrue(
            np.allclose(
                invalid_result["wall_curvature_px_inv"][1, 1, 1:-1],
                0.0,
                atol=1e-6,
            )
        )

    def test_curvature_change_is_rigid_transform_invariant(self):
        ant = self.circle_points(10.0)
        post = ant + np.asarray([0.0, 20.0], dtype=np.float32)
        angle = np.float32(0.35)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        transforms = (
            (ant + np.asarray([3.0, -4.0], dtype=np.float32), post + np.asarray([3.0, -4.0], dtype=np.float32)),
            (ant @ rotation.T, post @ rotation.T),
        )
        for measured_ant, measured_post in transforms:
            with self.subTest():
                result = anatomical_deformation_from_tracking(
                    tracking_case(ant, post, measured_ant, measured_post),
                    section_count=5,
                    fps=10.0,
                )
                rate = result["wall_curvature_change_rate_px_inv_s"][1, :, 1:-1]
                self.assertTrue(np.allclose(rate, 0.0, atol=1e-5))

    def test_curvature_change_rate_scales_with_fps(self):
        source_ant = self.circle_points(10.0)
        measured_ant = self.circle_points(5.0)
        source_post = source_ant + np.asarray([0.0, 20.0], dtype=np.float32)
        measured_post = measured_ant + np.asarray([0.0, 20.0], dtype=np.float32)
        tracking = tracking_case(
            source_ant, source_post, measured_ant, measured_post
        )
        slow = anatomical_deformation_from_tracking(
            tracking, section_count=5, fps=10.0
        )["wall_curvature_change_rate_px_inv_s"]
        fast = anatomical_deformation_from_tracking(
            tracking, section_count=5, fps=20.0
        )["wall_curvature_change_rate_px_inv_s"]
        self.assertTrue(np.allclose(fast[1, :, 1:-1], 2.0 * slow[1, :, 1:-1]))


if __name__ == "__main__":
    unittest.main()
