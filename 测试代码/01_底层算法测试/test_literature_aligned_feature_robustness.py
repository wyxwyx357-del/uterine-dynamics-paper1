from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    PROJECT_ROOT
    / "代码"
    / "03_实验与历史代码"
    / "run_literature_aligned_feature_robustness.py"
)
SPEC = importlib.util.spec_from_file_location("literature_robustness_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class LiteratureAlignedRobustnessTest(unittest.TestCase):
    def test_icc_1_1_matches_hand_calculated_balanced_example_and_ci(self):
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = RUNNER.icc_1_1(matrix)

        # Hand ANOVA: MS_between=8, MS_within=0.5, k=2.
        self.assertAlmostEqual(result.estimate, 7.5 / 8.5, places=12)
        self.assertAlmostEqual(result.ms_between, 8.0, places=12)
        self.assertAlmostEqual(result.ms_within, 0.5, places=12)
        # Fixed reference values from central-F inversion of F(2,3)=16.
        self.assertAlmostEqual(result.ci_lower, -0.0013764287, places=8)
        self.assertAlmostEqual(result.ci_upper, 0.9968135001, places=8)

    def test_identical_repeats_have_icc_one(self):
        matrix = np.repeat(np.arange(1.0, 8.0)[:, None], 5, axis=1)
        result = RUNNER.icc_1_1(matrix)
        self.assertEqual(result.estimate, 1.0)
        self.assertEqual((result.ci_lower, result.ci_upper), (1.0, 1.0))

    def test_within_subject_noise_reduces_icc(self):
        subjects = np.arange(20.0)[:, None]
        low_noise = subjects + np.asarray([[-0.01, 0.0, 0.01]])
        high_noise = subjects + np.asarray([[-5.0, 0.0, 5.0]])
        self.assertGreater(
            RUNNER.icc_1_1(low_noise).estimate,
            RUNNER.icc_1_1(high_noise).estimate,
        )

    def test_larger_between_patient_spread_increases_icc(self):
        repeats = np.asarray([[-1.0, 0.0, 1.0]])
        narrow = np.arange(10.0)[:, None] * 0.1 + repeats
        wide = np.arange(10.0)[:, None] * 10.0 + repeats
        self.assertGreater(
            RUNNER.icc_1_1(wide).estimate,
            RUNNER.icc_1_1(narrow).estimate,
        )

    def test_icc_rejects_incomplete_or_wrong_shape_matrix(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            RUNNER.icc_1_1(np.ones(4))
        with self.assertRaisesRegex(ValueError, "complete"):
            RUNNER.icc_1_1(np.asarray([[1.0, math.nan], [2.0, 2.0]]))

    def test_cv_uses_sample_sd_and_fails_closed_near_zero(self):
        cv, status = RUNNER.within_case_cv_pct([9.0, 10.0, 11.0])
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(cv, 10.0)
        cv, status = RUNNER.within_case_cv_pct([-1.0, 1.0])
        self.assertTrue(math.isnan(cv))
        self.assertEqual(status, "mean_near_zero")

    def test_runner_has_no_upstream_tracking_p3_qc_or_extractor_import(self):
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        forbidden = (
            "tracking",
            "p3_",
            "formal_qc",
            "formal_feature_extraction",
            "anatomical_deformation",
        )
        self.assertFalse(any(term in module for term in forbidden for module in imports))

    def test_frozen_data_integration_matrix_dedup_missingness_and_determinism(self):
        input_dir = RUNNER.DEFAULT_INPUT_DIR
        frozen_paths = [
            input_dir / RUNNER.LONG_NAME,
            input_dir / RUNNER.SUMMARY_NAME,
            input_dir / RUNNER.SOURCE_MANIFEST_NAME,
        ]
        before = {path.name: file_hash(path) for path in frozen_paths}
        with tempfile.TemporaryDirectory(prefix="literature_robustness_") as first_dir:
            first = Path(first_dir)
            RUNNER.run_analysis(
                input_dir, first, bootstrap_repetitions=120
            )
            summary = pd.read_csv(
                first / "literature_aligned_feature_robustness_summary.csv"
            )
            detail = pd.read_csv(
                first / "literature_aligned_feature_robustness_long.csv",
                low_memory=False,
            )
            self.assertEqual(len(summary), 20)
            curvature = summary[summary["feature_family"] == "curvature"]
            self.assertEqual(set(curvature["n_evaluable_cases"]), {27})
            curvature_names = set(curvature["feature_name"])
            curvature_baselines = detail[
                (detail["record_type"] == "baseline")
                & detail["feature"].isin(curvature_names)
            ]
            self.assertEqual(len(curvature_baselines), 28 * 6)
            self.assertTrue(
                (
                    curvature_baselines.groupby("feature")["analysis_value"]
                    .apply(lambda values: int(values.isna().sum()))
                    == 1
                ).all()
            )
            non_curvature = summary[summary["feature_family"] != "curvature"]
            self.assertEqual(set(non_curvature["n_evaluable_cases"]), {28})
            first_feature = str(summary.iloc[0]["feature_name"])
            selected = detail[
                (detail["feature"] == first_feature)
                & detail["included_primary_icc"]
                & (
                    detail["actual_mask_dedup_status"]
                    != "excluded_alias_of_space_block_target_5pct"
                )
            ]
            self.assertEqual(
                set(selected.groupby("case_id").size()),
                {26},
            )
            self.assertFalse(
                selected.duplicated(["case_id", "realized_perturbation_id"]).any()
            )

            with tempfile.TemporaryDirectory(
                prefix="literature_robustness_repeat_"
            ) as second_dir:
                second = Path(second_dir)
                RUNNER.run_analysis(
                    input_dir, second, bootstrap_repetitions=120
                )
                for name in (
                    "literature_aligned_feature_robustness_summary.csv",
                    "literature_aligned_feature_robustness_long.csv",
                    "perturbation_family_icc_summary.csv",
                    "within_case_cv_summary.csv",
                    "jackknife_sensitivity_comparison.csv",
                    "正式候选特征文献标准鲁棒性分析报告.md",
                    "新增代码与测试完整内容.md",
                ):
                    self.assertEqual(file_hash(first / name), file_hash(second / name))
        after = {path.name: file_hash(path) for path in frozen_paths}
        self.assertEqual(before, after)

    def test_same_realized_hash_in_different_cases_remains_two_subjects(self):
        rows = pd.DataFrame(
            [
                {
                    "case_id": case,
                    "realized_perturbation_id": "same-hash",
                    "analysis_value": value,
                }
                for case, value in (("A", 1.0), ("B", 2.0))
            ]
        )
        self.assertEqual(
            rows[["case_id", "realized_perturbation_id"]].drop_duplicates().shape[0],
            2,
        )


if __name__ == "__main__":
    unittest.main()
