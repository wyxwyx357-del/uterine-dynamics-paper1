from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from peristalsis_pipeline.dicom_curvature_calibration import (
    UltrasoundRegionCalibration,
    _pair_dt_from_dataset,
    apply_analysis_to_dicom_transform,
    curvature_change_rate_from_pair_dt,
    physical_curvature_from_wall_centers,
    ultrasound_regions_from_dataset,
)


class DicomCurvatureCalibrationTest(unittest.TestCase):
    @staticmethod
    def circle_in_anisotropic_pixels(radius_mm: float) -> np.ndarray:
        angles = np.linspace(-0.4, 0.4, 5, dtype=np.float32)
        x_mm = radius_mm * np.cos(angles)
        y_mm = radius_mm * np.sin(angles)
        # Region calibration below is 2 mm/px in X and 1 mm/px in Y.
        points = np.stack((x_mm / 2.0 + 50.0, y_mm + 50.0), axis=-1)
        return points.astype(np.float32)

    @staticmethod
    def region() -> UltrasoundRegionCalibration:
        return UltrasoundRegionCalibration(
            index=2,
            min_x=0,
            min_y=0,
            max_x=100,
            max_y=100,
            physical_units_x=3,
            physical_units_y=3,
            physical_delta_x=0.2,
            physical_delta_y=0.1,
        )

    def test_anisotropic_pixels_are_converted_before_curvature(self):
        source_curve = self.circle_in_anisotropic_pixels(10.0)
        measured_curve = self.circle_in_anisotropic_pixels(5.0)
        source = np.repeat(source_curve[None, None], 4, axis=0)
        source = np.repeat(source, 2, axis=1)
        measured = np.repeat(measured_curve[None, None], 4, axis=0)
        measured = np.repeat(measured, 2, axis=1)
        valid = np.ones((4, 2, 5), dtype=bool)
        valid[0] = False

        result = physical_curvature_from_wall_centers(
            wall_center_source=source,
            wall_center_measured=measured,
            pair_qc_valid=valid,
            analysis_to_dicom_transform=np.eye(3),
            region=self.region(),
            fps=2.0,
            timing_matched=True,
            pair_dt_s=np.asarray([np.nan, 0.5, 0.5, 0.5]),
        )
        curvature = result["qc_wall_curvature_mm_inv"]
        rate = result["qc_wall_curvature_change_rate_mm_inv_s"]
        self.assertTrue(np.allclose(curvature[1:, :, 1:-1], 0.2, atol=1e-5))
        self.assertTrue(np.allclose(rate[1:, :, 1:-1], 0.2, atol=2e-5))
        self.assertTrue(np.all(np.isnan(curvature[0])))
        self.assertTrue(np.all(np.isnan(curvature[:, :, [0, -1]])))

    def test_timing_mismatch_keeps_static_curvature_but_not_rate(self):
        curve = self.circle_in_anisotropic_pixels(10.0)
        centers = np.repeat(curve[None, None], 2, axis=0)
        centers = np.repeat(centers, 2, axis=1)
        valid = np.ones((2, 2, 5), dtype=bool)
        valid[0] = False
        result = physical_curvature_from_wall_centers(
            wall_center_source=centers,
            wall_center_measured=centers,
            pair_qc_valid=valid,
            analysis_to_dicom_transform=np.eye(3),
            region=self.region(),
            fps=30.0,
            timing_matched=False,
        )
        self.assertTrue(
            np.all(np.isfinite(result["qc_wall_curvature_mm_inv"][1, :, 1:-1]))
        )
        self.assertTrue(
            np.all(np.isnan(result["qc_wall_curvature_change_rate_mm_inv_s"]))
        )

    def test_region_must_cover_valid_points(self):
        centers = np.full((2, 2, 3, 2), 150.0, dtype=np.float32)
        valid = np.ones((2, 2, 3), dtype=bool)
        valid[0] = False
        with self.assertRaisesRegex(ValueError, "does not cover"):
            physical_curvature_from_wall_centers(
                wall_center_source=centers,
                wall_center_measured=centers,
                pair_qc_valid=valid,
                analysis_to_dicom_transform=np.eye(3),
                region=self.region(),
                fps=30.0,
                timing_matched=True,
                pair_dt_s=np.asarray([np.nan, 1.0 / 30.0]),
            )

    def test_frame_time_is_used_for_rate(self):
        pair_dt_s = _pair_dt_from_dataset(
            SimpleNamespace(FrameTime=50.0), frame_count=3
        )
        change = np.asarray([np.nan, 0.02, 0.02], dtype=np.float32).reshape(3, 1, 1)
        rate = curvature_change_rate_from_pair_dt(
            change,
            pair_dt_s,
            np.asarray([False, True, True]),
        )
        self.assertTrue(np.isnan(pair_dt_s[0]))
        self.assertTrue(np.allclose(pair_dt_s[1:], 0.05))
        self.assertTrue(np.allclose(rate[1:, 0, 0], 0.4, atol=1e-7))

    def test_nonuniform_frame_time_vector_produces_pair_specific_rates(self):
        pair_dt_s = _pair_dt_from_dataset(
            SimpleNamespace(FrameTimeVector=[0.0, 40.0, 60.0, 50.0]),
            frame_count=4,
        )
        change = np.asarray([np.nan, 0.02, 0.02, 0.02], dtype=np.float32).reshape(
            4, 1, 1
        )
        rate = curvature_change_rate_from_pair_dt(
            change,
            pair_dt_s,
            np.asarray([False, True, True, True]),
        )
        self.assertTrue(
            np.allclose(rate[1:, 0, 0], [0.5, 1.0 / 3.0, 0.4], atol=1e-6)
        )

    def test_full_physical_function_uses_pair_specific_nonuniform_dt(self):
        source_curve = self.circle_in_anisotropic_pixels(10.0)
        measured_curve = self.circle_in_anisotropic_pixels(5.0)
        source = np.repeat(source_curve[None, None], 4, axis=0)
        source = np.repeat(source, 2, axis=1)
        measured = np.repeat(measured_curve[None, None], 4, axis=0)
        measured = np.repeat(measured, 2, axis=1)
        valid = np.ones((4, 2, 5), dtype=bool)
        valid[0] = False
        pair_dt_s = np.asarray([np.nan, 0.0496, 0.05, 0.0504])

        result = physical_curvature_from_wall_centers(
            wall_center_source=source,
            wall_center_measured=measured,
            pair_qc_valid=valid,
            analysis_to_dicom_transform=np.eye(3),
            region=self.region(),
            fps=20.0,
            timing_matched=True,
            pair_dt_s=pair_dt_s,
        )

        rate = result["qc_wall_curvature_change_rate_mm_inv_s"]
        expected = np.asarray([2.0162683, 2.0001380, 1.9842639])
        self.assertTrue(np.allclose(rate[1:, 0, 2], expected, atol=2e-5))
        self.assertEqual(len(np.unique(np.round(rate[1:, 0, 2], 6))), 3)

    def test_numeric_timing_mismatch_is_rejected(self):
        curve = self.circle_in_anisotropic_pixels(10.0)
        centers = np.repeat(curve[None, None], 2, axis=0)
        centers = np.repeat(centers, 2, axis=1)
        valid = np.ones((2, 2, 5), dtype=bool)
        valid[0] = False
        with self.assertRaisesRegex(ValueError, "timing mismatch"):
            physical_curvature_from_wall_centers(
                wall_center_source=centers,
                wall_center_measured=centers,
                pair_qc_valid=valid,
                analysis_to_dicom_transform=np.eye(3),
                region=self.region(),
                fps=25.0,
                timing_matched=True,
                pair_dt_s=np.asarray([np.nan, 0.05]),
            )

    def test_missing_or_illegal_pair_timing_is_rejected(self):
        change = np.zeros((2, 1, 1), dtype=np.float32)
        valid_frames = np.asarray([False, True])
        for dt in (np.nan, 0.0, -0.05):
            with self.subTest(dt=dt):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    curvature_change_rate_from_pair_dt(
                        change,
                        np.asarray([np.nan, dt]),
                        valid_frames,
                    )

    def test_matched_timing_requires_pair_dt(self):
        curve = self.circle_in_anisotropic_pixels(10.0)
        centers = np.repeat(curve[None, None], 2, axis=0)
        centers = np.repeat(centers, 2, axis=1)
        valid = np.ones((2, 2, 5), dtype=bool)
        valid[0] = False
        with self.assertRaisesRegex(ValueError, "pair_dt_s is required"):
            physical_curvature_from_wall_centers(
                wall_center_source=centers,
                wall_center_measured=centers,
                pair_qc_valid=valid,
                analysis_to_dicom_transform=np.eye(3),
                region=self.region(),
                fps=20.0,
                timing_matched=True,
            )

    def test_transform_is_affine_and_preserves_nan(self):
        points = np.asarray([[1.0, 2.0], [np.nan, np.nan]], dtype=np.float32)
        transform = np.asarray([[2.0, 0.0, 3.0], [0.0, 4.0, 5.0], [0.0, 0.0, 1.0]])
        mapped = apply_analysis_to_dicom_transform(points, transform)
        self.assertTrue(np.allclose(mapped[0], [5.0, 13.0]))
        self.assertTrue(np.all(np.isnan(mapped[1])))

    def test_only_distance_calibrated_regions_are_returned(self):
        spatial = SimpleNamespace(
            RegionLocationMinX0=0,
            RegionLocationMinY0=0,
            RegionLocationMaxX1=100,
            RegionLocationMaxY1=100,
            PhysicalUnitsXDirection=3,
            PhysicalUnitsYDirection=3,
            PhysicalDeltaX=0.1,
            PhysicalDeltaY=0.1,
        )
        non_spatial = SimpleNamespace(
            RegionLocationMinX0=0,
            RegionLocationMinY0=0,
            RegionLocationMaxX1=100,
            RegionLocationMaxY1=100,
            PhysicalUnitsXDirection=4,
            PhysicalUnitsYDirection=3,
            PhysicalDeltaX=0.1,
            PhysicalDeltaY=0.1,
        )
        dataset = SimpleNamespace(SequenceOfUltrasoundRegions=[non_spatial, spatial])
        regions = ultrasound_regions_from_dataset(dataset)
        self.assertEqual([region.index for region in regions], [1])


if __name__ == "__main__":
    unittest.main()
