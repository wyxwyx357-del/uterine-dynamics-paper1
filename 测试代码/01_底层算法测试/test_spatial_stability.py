"""Data-free tests for normalized-position spatial stability."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "代码/01_底层算法"))

from peristalsis_pipeline.feature_stability import FEATURE_SPECS, StabilityCaseData
from peristalsis_pipeline.spatial_stability import (
    feature_spatial_positions,
    normalized_spatial_bins,
    spatial_bin_srd_values,
    spatial_profile,
    summarize_spatial_profile_pair,
)


class SpatialStabilityTests(unittest.TestCase):
    def data(self):
        # Two frames, two sides, five frozen radial/cavity/curvature sections.
        section_values = np.arange(1, 6, dtype=np.float64)
        rsr = np.broadcast_to(section_values[None, None, :], (2, 2, 5)).copy()
        cavity = np.broadcast_to(section_values[None, :], (2, 5)).copy()
        longitudinal_values = np.arange(1, 5, dtype=np.float64)
        longitudinal = np.broadcast_to(
            longitudinal_values[None, None, :], (2, 2, 4)
        ).copy()
        curvature = rsr / 10.0
        return StabilityCaseData(
            "test",
            rsr,
            30.0,
            np.ones_like(rsr, dtype=bool),
            30.0,
            cavity,
            np.ones_like(cavity, dtype=bool),
            longitudinal,
            np.ones_like(longitudinal, dtype=bool),
            curvature,
            True,
            True,
        )

    def test_normalized_bins_use_coordinate_not_section_rank(self):
        positions = np.array([0.01, 0.05, 0.45, 0.65, 0.95])
        bins = normalized_spatial_bins(positions, 5)
        np.testing.assert_array_equal(bins, np.array([0, 0, 2, 3, 4]))

    def test_last_coordinate_one_belongs_to_last_bin(self):
        bins = normalized_spatial_bins(np.array([0.0, 0.2, 0.4, 0.6, 1.0]), 5)
        np.testing.assert_array_equal(bins, np.array([0, 1, 2, 3, 4]))

    def test_out_of_range_coordinate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "within 0-1"):
            normalized_spatial_bins(np.array([0.0, 1.01]), 5)

    def test_longitudinal_positions_are_adjacent_section_midpoints(self):
        positions = np.array([0.0, 0.1, 0.5, 0.8, 1.0])
        result = feature_spatial_positions(
            positions, FEATURE_SPECS[8], position_count=4
        )
        np.testing.assert_allclose(result, [0.05, 0.3, 0.65, 0.9])

    def test_spatial_profile_uses_frozen_anatomical_coordinate(self):
        positions = np.array([0.01, 0.05, 0.45, 0.65, 0.95])
        profile, counts = spatial_profile(
            self.data(), FEATURE_SPECS[0], positions, bins=5
        )
        # First spatial fifth contains sections with values 1 and 2.
        # The second fifth has no frozen section and remains unavailable.
        np.testing.assert_allclose(
            profile[[0, 2, 3, 4]], [1.5, 3.0, 4.0, 5.0]
        )
        self.assertTrue(math.isnan(profile[1]))
        np.testing.assert_array_equal(counts, [8, 0, 4, 4, 4])

    def test_anterior_domain_does_not_mix_posterior_values(self):
        data = self.data()
        rsr = data.rsr.copy()
        rsr[:, 0, :] = 2.0
        rsr[:, 1, :] = 100.0
        sided = StabilityCaseData(
            data.case_id,
            rsr,
            data.rsr_fps,
            data.pair_valid,
            data.anatomical_fps,
            data.cavity,
            data.cavity_valid,
            data.longitudinal,
            data.longitudinal_valid,
            data.curvature_rate,
            data.physical_curvature_available,
            data.physical_curvature_rate_available,
        )
        positions = np.array([0.01, 0.21, 0.41, 0.61, 0.81])
        profile, counts = spatial_profile(
            sided, FEATURE_SPECS[2], positions, bins=5
        )
        np.testing.assert_allclose(profile, np.full(5, 2.0))
        np.testing.assert_array_equal(counts, np.full(5, 2))

    def test_curvature_profile_stays_unavailable_when_formal_rate_is_unavailable(self):
        data = self.data()
        unavailable = StabilityCaseData(
            data.case_id,
            data.rsr,
            data.rsr_fps,
            data.pair_valid,
            data.anatomical_fps,
            data.cavity,
            data.cavity_valid,
            data.longitudinal,
            data.longitudinal_valid,
            data.curvature_rate,
            True,
            False,
        )
        positions = np.array([0.01, 0.21, 0.41, 0.61, 0.81])
        profile, counts = spatial_profile(
            unavailable, FEATURE_SPECS[14], positions, bins=5
        )
        self.assertTrue(np.isnan(profile).all())
        np.testing.assert_array_equal(counts, np.zeros(5, dtype=np.int64))

    def test_identity_profile_has_perfect_spearman_and_zero_srd(self):
        profile = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        counts = np.array([10, 10, 10, 10, 10])
        summary = summarize_spatial_profile_pair(
            profile, profile.copy(), counts, counts.copy()
        )
        self.assertAlmostEqual(summary["spatial_spearman"], 1.0)
        self.assertEqual(summary["median_bin_srd_pct"], 0.0)
        self.assertEqual(summary["p95_bin_srd_pct"], 0.0)
        self.assertEqual(summary["bin_availability_retention_fraction"], 1.0)
        self.assertEqual(summary["valid_position_retention_fraction"], 1.0)

    def test_availability_loss_is_reported_not_imputed(self):
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        shadow = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        original_counts = np.array([5, 5, 5, 5, 5])
        shadow_counts = np.array([5, 0, 5, 4, 5])
        summary = summarize_spatial_profile_pair(
            original, shadow, original_counts, shadow_counts
        )
        self.assertEqual(summary["finite_to_nan_bins"], 1)
        self.assertEqual(summary["n_paired_finite_bins"], 4)
        self.assertAlmostEqual(summary["bin_availability_retention_fraction"], 0.8)
        self.assertAlmostEqual(summary["valid_position_retention_fraction"], 19 / 25)

    def test_mask_only_summary_rejects_nan_to_finite(self):
        with self.assertRaisesRegex(RuntimeError, "NaN-to-finite"):
            summarize_spatial_profile_pair(
                np.array([1.0, np.nan]),
                np.array([1.0, 2.0]),
                np.array([2, 0]),
                np.array([2, 1]),
            )

    def test_mask_only_summary_rejects_increased_valid_positions(self):
        with self.assertRaisesRegex(RuntimeError, "increased valid positions"):
            summarize_spatial_profile_pair(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([2, 2]),
                np.array([2, 3]),
            )

    def test_spatial_bin_srd_preserves_missing_bins(self):
        result = spatial_bin_srd_values(
            np.array([1.0, 2.0, np.nan]),
            np.array([1.0, 1.0, np.nan]),
        )
        self.assertEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 200.0 / 3.0)
        self.assertTrue(math.isnan(result[2]))


if __name__ == "__main__":
    unittest.main()
