"""Independent checks of frozen-mask interventions and frame-block controls."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "代码/04_运行入口"))
import run_tracking_quality_sensitivity as subject


class SensitivityTests(unittest.TestCase):
    def data(self):
        rsr=np.arange(1,49,dtype=np.float32).reshape(8,2,3)
        cavity=np.arange(1,25,dtype=np.float32).reshape(8,3)
        longitudinal=np.arange(1,33,dtype=np.float32).reshape(8,2,2)
        curvature=np.full_like(rsr,np.nan)
        curvature[:,:,1]=rsr[:,:,1]/10
        return subject.StabilityCaseData("test",rsr,30.,np.ones_like(rsr,dtype=bool),30.,
            cavity,np.ones_like(cavity,dtype=bool),longitudinal,np.ones_like(longitudinal,dtype=bool),
            curvature,True,True)

    def test_whole_frame_exclusion_matches_direct_statistics(self):
        original=self.data()
        changed=subject.exclude_frames(original,[2,5])
        a=subject.feature_vector(changed)
        keep=np.array([0,1,3,4,6,7])
        self.assertEqual(a[0],np.median(original.rsr[keep]))
        self.assertAlmostEqual(a[1],np.percentile(original.rsr[keep].astype(float),95))
        self.assertEqual(a[6],np.median(original.cavity[keep]))
        self.assertAlmostEqual(a[9],np.percentile(original.longitudinal[keep].astype(float),95))
        self.assertAlmostEqual(a[15],np.nanpercentile(original.curvature_rate[keep].astype(float),95),places=7)
        self.assertTrue(np.isfinite(original.rsr).all())
        self.assertEqual(changed.rsr.shape,original.rsr.shape)
        np.testing.assert_array_equal(changed.rsr[keep],original.rsr[keep])

    def test_random_pool_includes_original_candidate_frames(self):
        pool=np.arange(1,8)
        rng=np.random.default_rng(12)
        sampled=np.array([rng.choice(pool,size=2,replace=False) for _ in range(500)])
        self.assertFalse(np.any(sampled==0))
        self.assertTrue(np.any(sampled==2))
        self.assertTrue(all(len(np.unique(x))==2 for x in sampled))

    def test_block_preserves_run_count_lengths_and_excludes_frame_zero(self):
        rng=np.random.default_rng(42)
        for _ in range(50):
            selected=subject.block_sample(np.arange(1,50),[2,4,1],rng)
            self.assertEqual(sorted(map(len,subject.runs(selected))),[1,2,4])
            self.assertEqual(len(np.unique(selected)),7)
            self.assertGreater(selected.min(),0)

    def test_formal_summary_keeps_finite_to_nan_as_availability_loss(self):
        rows = pd.DataFrame(
            {
                "original": [1.0, 2.0, np.nan],
                "grade3_shadow": [1.1, np.nan, np.nan],
                "absolute_change": [0.1, np.nan, np.nan],
                "srd_pct": [subject.symmetric_relative_difference_pct(1.0, 1.1), np.nan, np.nan],
                "relative_change_pct": [10.0, np.nan, np.nan],
            }
        )
        summary = subject.summarize_qc_feature_group(rows)
        self.assertEqual(summary["n_total_cases"], 3)
        self.assertEqual(summary["n_original_finite"], 2)
        self.assertEqual(summary["n_paired_finite"], 1)
        self.assertEqual(summary["n_shadow_finite"], 1)
        self.assertEqual(summary["finite_to_nan_n"], 1)
        self.assertAlmostEqual(summary["availability_retention_fraction"], 0.5)
        self.assertAlmostEqual(
            summary["median_srd_pct"],
            subject.symmetric_relative_difference_pct(1.0, 1.1),
        )

    def test_formal_summary_rejects_nan_to_finite(self):
        rows = pd.DataFrame(
            {
                "original": [np.nan],
                "grade3_shadow": [1.0],
                "absolute_change": [np.nan],
                "srd_pct": [np.nan],
                "relative_change_pct": [np.nan],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "NaN-to-finite"):
            subject.summarize_qc_feature_group(rows)

    def test_empty_intervention_is_identity(self):
        original=self.data()
        np.testing.assert_array_equal(subject.feature_vector(original),
            subject.feature_vector(subject.exclude_frames(original,[])))


if __name__=="__main__":unittest.main()
