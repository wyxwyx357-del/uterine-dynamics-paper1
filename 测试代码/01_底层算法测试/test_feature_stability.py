from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from peristalsis_pipeline.feature_stability import (
    FEATURE_SPECS,
    StabilityCaseData,
    apply_pair_exclusion,
    assert_mask_only_invariants,
    compute_feature_row,
    fixed_contiguous_pair_exclusion,
    propagate_pair_exclusion,
    random_pair_exclusion,
    realized_mask_identity,
    remove_top_fraction_mask,
    sensitivity_warning,
    spatial_quintile_pair_exclusion,
    symmetric_relative_difference_pct,
    top_fraction_feature_value,
    valid_count_and_ratio,
)


class FeatureStabilityTest(unittest.TestCase):
    @staticmethod
    def case_data() -> StabilityCaseData:
        rsr = np.arange(20, dtype=np.float64).reshape(2, 2, 5) + 1.0
        pair_valid = np.ones_like(rsr, dtype=bool)
        cavity = np.arange(10, dtype=np.float64).reshape(2, 5) + 1.0
        longitudinal = np.arange(16, dtype=np.float64).reshape(2, 2, 4) + 1.0
        curvature = np.arange(20, dtype=np.float64).reshape(2, 2, 5) + 1.0
        curvature[:, :, [0, -1]] = np.nan
        return StabilityCaseData(
            case_id="TEST_CASE",
            rsr=rsr,
            rsr_fps=20.0,
            pair_valid=pair_valid,
            anatomical_fps=20.0,
            cavity=cavity,
            cavity_valid=np.ones_like(cavity, dtype=bool),
            longitudinal=longitudinal,
            longitudinal_valid=np.ones_like(longitudinal, dtype=bool),
            curvature_rate=curvature,
            physical_curvature_available=True,
            physical_curvature_rate_available=True,
        )

    @staticmethod
    def ten_by_ten_case_data(rsr: np.ndarray) -> StabilityCaseData:
        curvature = np.ones_like(rsr, dtype=np.float64)
        curvature[:, :, [0, -1]] = np.nan
        return StabilityCaseData(
            case_id="TEST_CASE",
            rsr=rsr,
            rsr_fps=20.0,
            pair_valid=np.isfinite(rsr),
            anatomical_fps=20.0,
            cavity=np.ones((10, 10), dtype=np.float64),
            cavity_valid=np.ones((10, 10), dtype=bool),
            longitudinal=np.ones((10, 2, 9), dtype=np.float64),
            longitudinal_valid=np.ones((10, 2, 9), dtype=bool),
            curvature_rate=curvature,
            physical_curvature_available=True,
            physical_curvature_rate_available=True,
        )

    def test_pair_exclusion_propagates_by_frozen_geometry(self):
        exclusion = np.zeros((2, 2, 5), dtype=bool)
        exclusion[0, 0, 2] = True
        propagated = propagate_pair_exclusion(exclusion)

        self.assertTrue(propagated["cavity"][0, 2])
        self.assertEqual(np.flatnonzero(propagated["longitudinal"][0, 0]).tolist(), [1, 2])
        self.assertEqual(np.flatnonzero(propagated["curvature"][0, 0]).tolist(), [1, 2, 3])
        self.assertFalse(np.any(propagated["longitudinal"][0, 1]))
        self.assertFalse(np.any(propagated["curvature"][0, 1]))

    def test_apply_pair_exclusion_is_mask_only_and_does_not_modify_baseline(self):
        baseline = self.case_data()
        original_rsr = baseline.rsr.copy()
        original_cavity = baseline.cavity.copy()
        original_longitudinal = baseline.longitudinal.copy()
        original_curvature = baseline.curvature_rate.copy()
        exclusion = np.zeros_like(baseline.pair_valid)
        exclusion[0, 0, 2] = True

        perturbed = apply_pair_exclusion(baseline, exclusion)

        self.assertTrue(np.isnan(perturbed.rsr[0, 0, 2]))
        self.assertFalse(perturbed.cavity_valid[0, 2])
        self.assertFalse(perturbed.longitudinal_valid[0, 0, 1])
        self.assertFalse(perturbed.longitudinal_valid[0, 0, 2])
        self.assertTrue(np.all(np.isnan(perturbed.curvature_rate[0, 0, 1:4])))
        self.assertTrue(np.array_equal(baseline.rsr, original_rsr, equal_nan=True))
        self.assertTrue(np.array_equal(baseline.cavity, original_cavity, equal_nan=True))
        self.assertTrue(
            np.array_equal(baseline.longitudinal, original_longitudinal, equal_nan=True)
        )
        self.assertTrue(
            np.array_equal(baseline.curvature_rate, original_curvature, equal_nan=True)
        )
        self.assertTrue(np.array_equal(perturbed.cavity, baseline.cavity, equal_nan=True))
        self.assertTrue(
            np.array_equal(perturbed.longitudinal, baseline.longitudinal, equal_nan=True)
        )

    def test_mask_propagation_reaches_the_frozen_final_extractor(self):
        baseline = self.case_data()
        exclusion = np.zeros_like(baseline.pair_valid)
        exclusion[0, 0, 2] = True

        baseline_row = compute_feature_row(baseline)
        perturbed_row = compute_feature_row(apply_pair_exclusion(baseline, exclusion))

        self.assertNotEqual(
            perturbed_row["rsr_abs_median"], baseline_row["rsr_abs_median"]
        )
        self.assertNotEqual(
            perturbed_row["cavity_width_strain_rate_abs_median"],
            baseline_row["cavity_width_strain_rate_abs_median"],
        )

    def test_mask_only_invariants_reject_nan_restoration_and_value_changes(self):
        baseline = self.case_data()
        rsr = baseline.rsr.copy()
        pair_valid = baseline.pair_valid.copy()
        rsr[0, 0, 0] = np.nan
        pair_valid[0, 0, 0] = False
        baseline = replace(baseline, rsr=rsr, pair_valid=pair_valid)

        restored = replace(baseline, rsr=np.nan_to_num(baseline.rsr, nan=123.0))
        with self.assertRaisesRegex(RuntimeError, "NaN-to-finite"):
            assert_mask_only_invariants(baseline, restored)

        changed_cavity = baseline.cavity.copy()
        changed_cavity[0, 0] += 1.0
        changed = replace(baseline, cavity=changed_cavity)
        with self.assertRaisesRegex(RuntimeError, "mask-only"):
            assert_mask_only_invariants(baseline, changed)

    def test_fixed_blocks_are_deterministic_and_report_actual_removal(self):
        valid = np.ones((100, 2, 10), dtype=bool)
        first, first_metadata = fixed_contiguous_pair_exclusion(
            valid,
            axis="time",
            target_removed_fraction=0.05,
            position_index=2,
        )
        second, second_metadata = fixed_contiguous_pair_exclusion(
            valid,
            axis="time",
            target_removed_fraction=0.05,
            position_index=2,
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["pair_actual_removed_valid_fraction"], 0.05)
        self.assertEqual(first_metadata["pair_valid_count_removed"], 100)

    def test_random_exclusion_is_reproducible_and_seed_sensitive(self):
        valid = np.ones((100, 2, 10), dtype=bool)
        first, first_metadata = random_pair_exclusion(
            valid, target_removed_fraction=0.05, seed=1729
        )
        repeated, repeated_metadata = random_pair_exclusion(
            valid, target_removed_fraction=0.05, seed=1729
        )
        different, _ = random_pair_exclusion(
            valid, target_removed_fraction=0.05, seed=2718
        )

        self.assertTrue(np.array_equal(first, repeated))
        self.assertEqual(first_metadata, repeated_metadata)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(np.count_nonzero(first), 100)
        self.assertEqual(first_metadata["target_removed_n"], 100)
        self.assertEqual(first_metadata["pair_actual_removed_valid_fraction"], 0.05)

    def test_random_exclusion_never_selects_formally_invalid_positions(self):
        valid = np.ones((20, 2, 5), dtype=bool)
        valid[::2, 0, 0] = False
        exclusion, _ = random_pair_exclusion(
            valid, target_removed_fraction=0.10, seed=31415
        )
        self.assertFalse(np.any(exclusion & ~valid))
        self.assertEqual(
            np.count_nonzero(exclusion),
            math.floor(0.10 * np.count_nonzero(valid)),
        )

    def test_realized_identity_uses_mask_not_perturbed_feature_value(self):
        first = np.zeros((2, 2, 5), dtype=bool)
        second = np.zeros_like(first)
        first[:, :, 1] = True
        second[:, :, 2] = True

        self.assertEqual(realized_mask_identity(first), realized_mask_identity(first.copy()))
        self.assertNotEqual(realized_mask_identity(first), realized_mask_identity(second))

    def test_curvature_neighborhood_loss_can_exceed_upstream_spatial_loss(self):
        data = self.case_data()
        exclusion = np.zeros_like(data.pair_valid)
        exclusion[:, :, 2] = True
        perturbed = apply_pair_exclusion(data, exclusion)
        curvature = next(
            spec for spec in FEATURE_SPECS if spec.name.startswith("wall_curvature")
        )
        before, _ = valid_count_and_ratio(data, curvature)
        after, _ = valid_count_and_ratio(perturbed, curvature)

        self.assertEqual(np.count_nonzero(np.any(exclusion, axis=(0, 1))), 1)
        self.assertEqual(data.pair_valid.shape[2], 5)
        self.assertGreater((before - after) / before, 1 / 5)

    def test_spatial_quintiles_use_half_open_normalized_boundaries(self):
        valid = np.ones((1, 2, 6), dtype=bool)
        position = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        first, _ = spatial_quintile_pair_exclusion(valid, position, 0)
        second, _ = spatial_quintile_pair_exclusion(valid, position, 1)

        self.assertEqual(np.flatnonzero(first[0, 0]).tolist(), [0])
        self.assertEqual(np.flatnonzero(second[0, 0]).tolist(), [1])

    def test_top_one_percent_requires_at_least_100_values(self):
        small = np.arange(37, dtype=np.float64)
        keep, metadata = remove_top_fraction_mask(small, np.ones(37, dtype=bool))
        self.assertEqual(metadata["status"], "insufficient_n")
        self.assertEqual(metadata["valid_count_removed"], 0)
        self.assertTrue(np.all(keep))

        enough = np.arange(100, dtype=np.float64)
        keep, metadata = remove_top_fraction_mask(enough, np.ones(100, dtype=bool))
        self.assertEqual(metadata["status"], "ok")
        self.assertEqual(metadata["valid_count_removed"], 1)
        self.assertFalse(keep[-1])
        self.assertEqual(np.count_nonzero(keep), 99)

    def test_top_fraction_uses_each_feature_domain_separately(self):
        data = self.case_data()
        rsr = np.zeros((10, 2, 10), dtype=np.float64)
        rsr[:, 0] = np.arange(100).reshape(10, 10)
        rsr[:, 1] = np.arange(100, 200).reshape(10, 10)
        data = self.ten_by_ten_case_data(rsr)
        anterior = next(
            spec for spec in FEATURE_SPECS if spec.name == "anterior_rsr_abs_p95"
        )
        posterior = next(
            spec for spec in FEATURE_SPECS if spec.name == "posterior_rsr_abs_p95"
        )

        anterior_value, anterior_before, anterior_after, anterior_status = (
            top_fraction_feature_value(data, anterior)
        )
        posterior_value, posterior_before, posterior_after, posterior_status = (
            top_fraction_feature_value(data, posterior)
        )

        self.assertEqual((anterior_before, anterior_after, anterior_status), (100, 99, "ok"))
        self.assertEqual((posterior_before, posterior_after, posterior_status), (100, 99, "ok"))
        self.assertLess(anterior_value, posterior_value)

    def test_top_fraction_reuses_the_frozen_formal_extractor(self):
        rsr = np.arange(200, dtype=np.float64).reshape(10, 2, 10)
        data = self.ten_by_ten_case_data(rsr)
        spec = next(item for item in FEATURE_SPECS if item.name == "rsr_abs_p95")
        from peristalsis_pipeline import feature_stability as module

        with patch.object(
            module,
            "extract_formal_candidate_case_features",
            wraps=module.extract_formal_candidate_case_features,
        ) as extractor:
            value, before, after, status = top_fraction_feature_value(data, spec)

        self.assertEqual(status, "ok")
        self.assertTrue(math.isfinite(value))
        self.assertEqual((before, after), (200, 198))
        extractor.assert_called_once()

    def test_srd_is_saved_as_warning_not_feature_failure(self):
        self.assertEqual(symmetric_relative_difference_pct(0.0, 0.0), 0.0)
        self.assertAlmostEqual(symmetric_relative_difference_pct(1.0, 1.1), 200 / 21)
        self.assertTrue(math.isnan(symmetric_relative_difference_pct(math.nan, 1.0)))
        self.assertEqual(sensitivity_warning("median", 1.0, 1.1), "低敏感")
        self.assertEqual(sensitivity_warning("median", 1.0, 2.0), "高敏感")
        self.assertEqual(sensitivity_warning("p95", math.nan, 1.0), "证据不足")


if __name__ == "__main__":
    unittest.main()
