from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    PROJECT_ROOT
    / "代码"
    / "03_实验与历史代码"
    / "run_new_patient_feature_stability_validation.py"
)
SPEC = importlib.util.spec_from_file_location("new_patient_stability_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class NewPatientFeatureStabilityValidationTest(unittest.TestCase):
    def test_load_case_preserves_values_and_reads_upstream_eligibility(self):
        data = self.case_data()
        lineage = {
            "case_id": "TEST_CASE",
            "current_qc_semantics": "formal_human_artifact_and_manual_p3_position_exclusion",
            "artifact_qc_version": RUNNER.FORMAL_QC_VERSION,
            "formal_qc_lineage_id": "synthetic-lineage",
        }
        rsr = {**lineage, "current_qc_radial_strain_rate_s": data.rsr, "fps": data.rsr_fps}
        anatomical = {
            **lineage, "pair_qc_valid": data.pair_valid,
            "qc_cavity_width_strain_rate_s": data.cavity, "cavity_qc_valid": data.cavity_valid,
            "qc_longitudinal_wall_strain_rate_s": data.longitudinal,
            "longitudinal_qc_valid": data.longitudinal_valid, "fps": data.anatomical_fps,
            "normalized_cervix_to_fundus_position": np.linspace(0, 1, 10),
            "formal_feature_eligible": False, "has_tracking_topology_risk": True,
        }
        shadow = {"pair_quality_red_candidate": np.zeros_like(data.pair_valid)}
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(RUNNER, "load_npz", side_effect=[rsr, anatomical, shadow]), \
             patch.object(RUNNER, "load_artifact_metadata", return_value={}):
            loaded, _, _, metadata, _ = RUNNER.load_case(
                "TEST_CASE", rsr_dir=Path("rsr"), anatomical_dir=Path("anatomical"),
                shadow_dir=Path("shadow"), dicom_dir=None, artifact_dir=Path("artifact")
            )
        self.assertFalse(metadata["formal_analysis_eligible"])
        self.assertEqual(metadata["formal_analysis_status"], "ineligible")
        np.testing.assert_array_equal(loaded.rsr, data.rsr)
        self.assertEqual(RUNNER.compute_feature_row(loaded)["rsr_abs_median"], 1.0)

    def test_risk_unknown_and_conflicting_cases_stay_diagnostic_only(self):
        feature = RUNNER.BIOLOGICAL_FEATURE_COLUMNS[0]
        raw = pd.DataFrame([
            {"case_id": case, "feature": feature, "baseline_value": value,
             "perturbed_value": value, "relative_or_SRD_change": 0.0,
             "absolute_change": 0.0, "perturbation": "random_delete_5pct",
             "status": "ok", "assessment": "低敏感"}
            for case, value in [("GOOD", 1.0), ("RISK", 100.0), ("UNKNOWN", 200.0), ("CONFLICT", 300.0), ("NO_METADATA", 400.0)]
        ])
        metadata = pd.DataFrame([
            {"case_id": case, **RUNNER.formal_analysis_eligibility(payload)}
            for case, payload in [
                ("GOOD", {"formal_feature_eligible": True, "has_tracking_topology_risk": False}),
                ("RISK", {"formal_feature_eligible": False, "has_tracking_topology_risk": True}),
                ("UNKNOWN", {}),
                ("CONFLICT", {"formal_feature_eligible": True, "has_tracking_topology_risk": True}),
            ]
        ])
        before = raw.copy(deep=True)
        diagnostic, formal = RUNNER.partition_stability_results(raw, metadata)
        self.assertEqual(formal.case_id.tolist(), ["GOOD"])
        self.assertEqual(len(diagnostic), 5)
        pd.testing.assert_frame_equal(diagnostic[raw.columns], before)
        pd.testing.assert_frame_equal(raw, before)
        self.assertEqual(diagnostic.set_index("case_id").loc["NO_METADATA", "formal_analysis_status"], "unknown")
        self.assertEqual(RUNNER.summarize_features(formal).set_index("feature_name").loc[feature, "n_evaluable_cases"], 1)
        self.assertEqual(RUNNER.summarize_features(diagnostic).set_index("feature_name").loc[feature, "n_evaluable_cases"], 5)

    def test_duplicate_eligibility_keys_cannot_multiply_stability_rows(self):
        raw = pd.DataFrame([{"case_id": "A", "baseline_value": 1.0}])
        with self.assertRaisesRegex(ValueError, "unique case_id"):
            RUNNER.partition_stability_results(raw, pd.DataFrame([{"case_id": "A"}, {"case_id": "A"}]))

    def test_absent_eligibility_columns_do_not_default_to_eligible(self):
        raw = pd.DataFrame([{"case_id": "A", "baseline_value": 1.0}])
        diagnostic, formal = RUNNER.partition_stability_results(raw, pd.DataFrame([{"case_id": "A"}]))
        self.assertTrue(formal.empty)
        self.assertEqual(diagnostic.iloc[0].formal_analysis_status, "unknown")

    @staticmethod
    def case_data() -> object:
        shape = (20, 2, 10)
        curvature = np.ones(shape, dtype=np.float64)
        curvature[:, :, [0, -1]] = np.nan
        return RUNNER.StabilityCaseData(
            case_id="TEST_CASE",
            rsr=np.ones(shape),
            rsr_fps=20.0,
            pair_valid=np.ones(shape, dtype=bool),
            anatomical_fps=20.0,
            cavity=np.ones((20, 10)),
            cavity_valid=np.ones((20, 10), dtype=bool),
            longitudinal=np.ones((20, 2, 9)),
            longitudinal_valid=np.ones((20, 2, 9), dtype=bool),
            curvature_rate=curvature,
            physical_curvature_available=True,
            physical_curvature_rate_available=True,
        )

    def test_frozen_contract_is_exactly_twenty_biological_features(self):
        self.assertEqual(
            tuple(RUNNER.MODEL_INPUT_FEATURE_COLUMNS),
            tuple(RUNNER.BIOLOGICAL_FEATURE_COLUMNS),
        )
        self.assertEqual(len(RUNNER.BIOLOGICAL_FEATURE_COLUMNS), 20)
        self.assertTrue(
            set(RUNNER.QC_METADATA_COLUMNS).isdisjoint(
                RUNNER.BIOLOGICAL_FEATURE_COLUMNS
            )
        )

    def test_common_perturbation_contract_is_frozen_and_seeded(self):
        data = self.case_data()
        perturbations = RUNNER.common_perturbations(
            data,
            np.linspace(0.0, 1.0, 10),
            np.zeros_like(data.pair_valid),
        )

        self.assertEqual(len(perturbations), 41)
        random_rows = [
            row for row in perturbations if row["perturbation"].startswith("random_")
        ]
        self.assertEqual(len(random_rows), 10)
        self.assertEqual(
            sorted({row["seed"] for row in random_rows}),
            sorted(RUNNER.RANDOM_SEEDS),
        )

    def test_collapsed_nominal_space_blocks_share_realized_identity_and_group(self):
        data = self.case_data()
        perturbations = RUNNER.common_perturbations(
            data, np.linspace(0.0, 1.0, 10), np.zeros_like(data.pair_valid)
        )
        five = next(
            row for row in perturbations if row["perturbation_id"] == "space_5pct_position_3"
        )
        ten = next(
            row for row in perturbations if row["perturbation_id"] == "space_10pct_position_3"
        )

        self.assertTrue(np.array_equal(five["exclusion"], ten["exclusion"]))
        self.assertEqual(
            RUNNER.realized_mask_identity(five["exclusion"]),
            RUNNER.realized_mask_identity(ten["exclusion"]),
        )
        self.assertEqual(
            RUNNER.realized_classification_group(five),
            RUNNER.realized_classification_group(ten),
        )

    def test_different_masks_are_not_merged_because_feature_values_match(self):
        data = self.case_data()
        first = np.zeros_like(data.pair_valid)
        second = np.zeros_like(data.pair_valid)
        first[:, :, 1] = True
        second[:, :, 2] = True
        first_value = RUNNER.compute_feature_row(
            RUNNER.apply_pair_exclusion(data, first)
        )["rsr_abs_median"]
        second_value = RUNNER.compute_feature_row(
            RUNNER.apply_pair_exclusion(data, second)
        )["rsr_abs_median"]

        self.assertEqual(first_value, second_value)
        self.assertNotEqual(
            RUNNER.realized_mask_identity(first),
            RUNNER.realized_mask_identity(second),
        )

    def test_nominal_upstream_and_downstream_losses_are_separate_fields(self):
        data = self.case_data()
        baseline = RUNNER.compute_feature_row(data)
        rows = RUNNER.analyze_case(
            data,
            np.linspace(0.0, 1.0, 10),
            np.zeros_like(data.pair_valid),
            baseline,
        )
        row = next(
            item
            for item in rows
            if item["perturbation_id"] == "space_5pct_position_3"
            and item["feature"]
            == "wall_curvature_change_rate_mm_inv_s_abs_median"
        )

        self.assertEqual(row["nominal_target_fraction"], 0.05)
        self.assertEqual(row["realized_upstream_removed_fraction"], 0.1)
        self.assertGreater(
            row["downstream_lost_fraction"],
            row["realized_upstream_removed_fraction"],
        )
        self.assertEqual(row["actual_removed_n"], row["downstream_lost_n"])

    def test_duplicate_realized_group_is_counted_once_for_classification(self):
        rows = pd.DataFrame(
            [
                {
                    "case_id": case,
                    "status": "ok",
                    "assessment": "高敏感",
                    "perturbation": nominal,
                    "realized_classification_group": "space_block_realized_width_1_of_8",
                }
                for case in ("A", "B", "C")
                for nominal in ("space_block_target_5pct", "space_block_target_10pct")
            ]
        )
        classification, _ = RUNNER.frozen_feature_classification(rows, 3)
        self.assertEqual(classification, "需谨慎")

        extra = rows.iloc[[0]].copy()
        extra["case_id"] = "A"
        extra["perturbation"] = "spatial_jackknife"
        extra["realized_classification_group"] = "spatial_jackknife"
        classification, _ = RUNNER.frozen_feature_classification(
            pd.concat([rows, extra], ignore_index=True), 3
        )
        self.assertEqual(classification, "不稳定")

    def test_cross_family_same_case_same_actual_mask_counts_once(self):
        rows = pd.DataFrame(
            [
                {
                    "case_id": "A",
                    "status": "ok",
                    "assessment": "高敏感",
                    "perturbation": nominal,
                    "realized_classification_group": group,
                    "realized_perturbation_id": "same-mask",
                }
                for nominal, group in (
                    ("space_block_target_5pct", "space_block"),
                    ("spatial_jackknife", "spatial_jackknife"),
                )
            ]
        )
        self.assertEqual(RUNNER.independent_realized_group_count(rows), 1)
        classification, _ = RUNNER.frozen_feature_classification(rows, 3)
        self.assertNotEqual(classification, "不稳定")

    def test_cross_family_different_actual_masks_remain_independent(self):
        rows = pd.DataFrame(
            [
                {
                    "case_id": "A",
                    "status": "ok",
                    "assessment": "高敏感",
                    "perturbation": "space_block_target_5pct",
                    "realized_classification_group": "space_block",
                    "realized_perturbation_id": "mask-a",
                },
                {
                    "case_id": "A",
                    "status": "ok",
                    "assessment": "高敏感",
                    "perturbation": "spatial_jackknife",
                    "realized_classification_group": "spatial_jackknife",
                    "realized_perturbation_id": "mask-b",
                },
            ]
        )
        self.assertEqual(RUNNER.independent_realized_group_count(rows), 2)

    def test_identical_hash_in_different_cases_preserves_case_evidence(self):
        rows = pd.DataFrame(
            [
                {
                    "case_id": "A",
                    "status": "ok",
                    "assessment": "高敏感",
                    "perturbation": "space_block_target_5pct",
                    "realized_classification_group": "space_block",
                    "realized_perturbation_id": "same-structural-hash",
                },
                {
                    "case_id": "B",
                    "status": "ok",
                    "assessment": "高敏感",
                    "perturbation": "spatial_jackknife",
                    "realized_classification_group": "spatial_jackknife",
                    "realized_perturbation_id": "same-structural-hash",
                },
            ]
        )
        self.assertEqual(rows["case_id"].nunique(), 2)
        self.assertEqual(
            rows[["case_id", "realized_perturbation_id"]]
            .drop_duplicates()
            .shape[0],
            2,
        )
        self.assertEqual(RUNNER.independent_realized_group_count(rows), 2)

    def test_classification_dedup_does_not_change_measurement_columns(self):
        columns = [
            "baseline_value",
            "perturbed_value",
            "absolute_change",
            "relative_or_SRD_change",
            "status",
            "assessment",
        ]
        rows = pd.DataFrame(
            [
                {
                    "case_id": "A",
                    "perturbation": nominal,
                    "realized_classification_group": group,
                    "realized_perturbation_id": "same-mask",
                    "baseline_value": 1.0,
                    "perturbed_value": 1.25,
                    "absolute_change": 0.25,
                    "relative_or_SRD_change": 22.22,
                    "status": "ok",
                    "assessment": "高敏感",
                }
                for nominal, group in (
                    ("space_block_target_5pct", "space_block"),
                    ("spatial_jackknife", "spatial_jackknife"),
                )
            ]
        )
        before = rows[columns].copy(deep=True)
        RUNNER.frozen_feature_classification(rows, 3)
        pd.testing.assert_frame_equal(rows[columns], before)

    def test_runner_does_not_import_or_call_upstream_tracking_or_p3(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
        for forbidden in (
            "tracking_huang",
            "p3_image_boundary_correction",
            "run_new_patient_full_pipeline",
        ):
            self.assertFalse(
                any(forbidden in module for module in imported_modules),
                f"unexpected upstream import: {forbidden}",
            )

    def test_each_feature_has_an_independent_evaluable_case_count(self):
        feature = RUNNER.BIOLOGICAL_FEATURE_COLUMNS[0]
        rows = []
        for case_id in ("A", "B", "C"):
            rows.append(
                {
                    "case_id": case_id,
                    "feature": feature,
                    "perturbation": "random_delete_5pct",
                    "baseline_value": 1.0,
                    "perturbed_value": 1.0,
                    "absolute_change": 0.0,
                    "relative_or_SRD_change": 0.0,
                    "status": "ok",
                    "assessment": "低敏感",
                }
            )
        summary = RUNNER.summarize_features(pd.DataFrame(rows)).set_index(
            "feature_name"
        )

        self.assertEqual(summary.loc[feature, "n_evaluable_cases"], 3)
        curvature = next(
            name
            for name in RUNNER.BIOLOGICAL_FEATURE_COLUMNS
            if "curvature_change_rate" in name
        )
        self.assertEqual(summary.loc[curvature, "n_evaluable_cases"], 0)
        self.assertEqual(summary.loc[curvature, "final_classification"], "证据不足")

    def test_frozen_classification_does_not_reclassify_one_outlier_as_systemic(self):
        rows = pd.DataFrame(
            [
                {
                    "case_id": "A",
                    "status": "ok",
                    "assessment": "高敏感",
                    "perturbation": "random_delete_5pct",
                },
                {
                    "case_id": "B",
                    "status": "ok",
                    "assessment": "低敏感",
                    "perturbation": "random_delete_5pct",
                },
                {
                    "case_id": "C",
                    "status": "ok",
                    "assessment": "低敏感",
                    "perturbation": "random_delete_10pct",
                },
            ]
        )
        classification, _ = RUNNER.frozen_feature_classification(rows, 3)
        self.assertEqual(classification, "需谨慎")


if __name__ == "__main__":
    unittest.main()
