"""Data-free tests for normalized-time temporal stability."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "代码/01_底层算法"))

from peristalsis_pipeline.feature_stability import FEATURE_SPECS, StabilityCaseData
from peristalsis_pipeline.temporal_stability import (
    bin_srd_values,
    normalized_time_bins,
    paired_spearman,
    summarize_profile_pair,
    temporal_profile,
)


class TemporalStabilityTests(unittest.TestCase):
    def data(self):
        frame_values = np.arange(1, 11, dtype=np.float64)
        rsr = np.broadcast_to(frame_values[:, None, None], (10, 2, 2)).copy()
        cavity = np.broadcast_to(frame_values[:, None], (10, 2)).copy()
        longitudinal = np.broadcast_to(
            frame_values[:, None, None], (10, 2, 1)
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

    def test_normalized_bins_cover_each_frame_once(self):
        bins = normalized_time_bins(10, 5)
        np.testing.assert_array_equal(
            bins, np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        )

    def test_temporal_profile_uses_feature_statistic_in_relative_bins(self):
        profile, counts = temporal_profile(self.data(), FEATURE_SPECS[0], bins=5)
        np.testing.assert_allclose(profile, [1.5, 3.5, 5.5, 7.5, 9.5])
        np.testing.assert_array_equal(counts, [8, 8, 8, 8, 8])

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
        profile, counts = temporal_profile(unavailable, FEATURE_SPECS[14], bins=5)
        self.assertTrue(np.isnan(profile).all())
        np.testing.assert_array_equal(counts, np.zeros(5, dtype=np.int64))

    def test_identity_profile_has_perfect_spearman_and_zero_srd(self):
        profile = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        counts = np.array([10, 10, 10, 10, 10])
        summary = summarize_profile_pair(profile, profile.copy(), counts, counts.copy())
        self.assertAlmostEqual(summary["temporal_spearman"], 1.0)
        self.assertEqual(summary["median_bin_srd_pct"], 0.0)
        self.assertEqual(summary["p95_bin_srd_pct"], 0.0)
        self.assertEqual(summary["bin_availability_retention_fraction"], 1.0)
        self.assertEqual(summary["valid_position_retention_fraction"], 1.0)

    def test_availability_loss_is_reported_not_imputed(self):
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        shadow = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        original_counts = np.array([5, 5, 5, 5, 5])
        shadow_counts = np.array([5, 0, 5, 4, 5])
        summary = summarize_profile_pair(
            original, shadow, original_counts, shadow_counts
        )
        self.assertEqual(summary["finite_to_nan_bins"], 1)
        self.assertEqual(summary["n_paired_finite_bins"], 4)
        self.assertAlmostEqual(summary["bin_availability_retention_fraction"], 0.8)
        self.assertAlmostEqual(summary["valid_position_retention_fraction"], 19 / 25)

    def test_mask_only_summary_rejects_nan_to_finite(self):
        with self.assertRaisesRegex(RuntimeError, "NaN-to-finite"):
            summarize_profile_pair(
                np.array([1.0, np.nan]),
                np.array([1.0, 2.0]),
                np.array([2, 0]),
                np.array([2, 1]),
            )

    def test_mask_only_summary_rejects_increased_valid_positions(self):
        with self.assertRaisesRegex(RuntimeError, "increased valid positions"):
            summarize_profile_pair(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([2, 2]),
                np.array([2, 3]),
            )

    def test_bin_srd_values_preserve_missing_bins(self):
        result = bin_srd_values(
            np.array([1.0, 2.0, np.nan]),
            np.array([1.0, 1.0, np.nan]),
        )
        self.assertEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 200.0 / 3.0)
        self.assertTrue(math.isnan(result[2]))

    def test_paired_spearman_constant_profile_is_undefined(self):
        self.assertTrue(
            math.isnan(
                paired_spearman(
                    np.array([1.0, 1.0, 1.0]),
                    np.array([1.0, 2.0, 3.0]),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
