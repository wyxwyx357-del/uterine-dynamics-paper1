from __future__ import annotations

import math
import unittest

import numpy as np

from peristalsis_pipeline.formal_feature_extraction import (
    BIOLOGICAL_FEATURE_COLUMNS,
    FORMAL_CANDIDATE_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_INPUT_FEATURE_COLUMNS,
    QC_METADATA_COLUMNS,
    extract_formal_candidate_case_features,
    finite_abs_statistics,
)
from peristalsis_pipeline.formal_qc_contract import (
    FORMAL_QC_LINEAGE_ID_FIELD,
    FORMAL_QC_SEMANTICS,
    FORMAL_QC_VERSION,
    validate_formal_qc_lineage,
    validate_formal_qc_version,
)


class FormalFeatureExtractionTest(unittest.TestCase):
    @staticmethod
    def inputs() -> dict[str, object]:
        rsr = np.ones((2, 2, 3), dtype=np.float32)
        pair_valid = np.ones_like(rsr, dtype=bool)
        cavity = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        longitudinal = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
        return {
            "case_id": "TEST_CASE",
            "current_qc_radial_strain_rate_s": rsr,
            "rsr_fps": 20.0,
            "anatomical_pair_qc_valid": pair_valid,
            "anatomical_fps": 20.0,
            "qc_cavity_width_strain_rate_s": cavity,
            "cavity_qc_valid": np.ones_like(cavity, dtype=bool),
            "qc_longitudinal_wall_strain_rate_s": longitudinal,
            "longitudinal_qc_valid": np.ones_like(longitudinal, dtype=bool),
            "physical_curvature_available": False,
            "physical_curvature_rate_available": False,
        }

    def test_known_median_and_p95_are_numeric(self):
        values = np.arange(-20, 0, dtype=np.float64)
        statistics = finite_abs_statistics(values)
        self.assertEqual(statistics["median"], 10.5)
        self.assertAlmostEqual(statistics["p95"], 19.05, places=12)

    def test_final_extractor_keeps_distinct_known_median_and_p95(self):
        inputs = self.inputs()
        rsr = np.arange(-20, 0, dtype=np.float64).reshape(2, 2, 5)
        inputs["current_qc_radial_strain_rate_s"] = rsr
        inputs["anatomical_pair_qc_valid"] = np.ones_like(rsr, dtype=bool)
        inputs["qc_cavity_width_strain_rate_s"] = np.ones((2, 5))
        inputs["cavity_qc_valid"] = np.ones((2, 5), dtype=bool)
        inputs["qc_longitudinal_wall_strain_rate_s"] = np.ones((2, 2, 4))
        inputs["longitudinal_qc_valid"] = np.ones((2, 2, 4), dtype=bool)

        row = extract_formal_candidate_case_features(**inputs)

        self.assertEqual(row["rsr_abs_median"], 10.5)
        self.assertAlmostEqual(row["rsr_abs_p95"], 19.05, places=12)
        self.assertNotEqual(row["rsr_abs_median"], row["rsr_abs_p95"])

    def test_anterior_and_posterior_are_not_swapped(self):
        inputs = self.inputs()
        rsr = np.empty((2, 2, 3), dtype=np.float32)
        rsr[:, 0] = 1.0
        rsr[:, 1] = 10.0
        inputs["current_qc_radial_strain_rate_s"] = rsr
        inputs["anatomical_pair_qc_valid"] = np.isfinite(rsr)
        row = extract_formal_candidate_case_features(**inputs)
        self.assertEqual(row["anterior_rsr_abs_median"], 1.0)
        self.assertEqual(row["anterior_rsr_abs_p95"], 1.0)
        self.assertEqual(row["posterior_rsr_abs_median"], 10.0)
        self.assertEqual(row["posterior_rsr_abs_p95"], 10.0)

    def test_valid_masks_intersect_finite_values(self):
        inputs = self.inputs()
        cavity = np.asarray([[1.0, np.nan, 100.0], [3.0, 5.0, 7.0]])
        cavity_valid = np.asarray([[True, True, False], [True, False, False]])
        inputs["qc_cavity_width_strain_rate_s"] = cavity
        inputs["cavity_qc_valid"] = cavity_valid
        row = extract_formal_candidate_case_features(**inputs)
        self.assertEqual(row["cavity_width_strain_rate_abs_median"], 2.0)
        self.assertEqual(row["cavity_valid_ratio"], 2.0 / 6.0)

    def test_rsr_mask_must_equal_finite_positions(self):
        inputs = self.inputs()
        inputs["anatomical_pair_qc_valid"][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "must equal finite positions"):
            extract_formal_candidate_case_features(**inputs)

    def test_zero_effective_values_return_nan_not_zero(self):
        inputs = self.inputs()
        inputs["cavity_qc_valid"][:] = False
        row = extract_formal_candidate_case_features(**inputs)
        self.assertTrue(math.isnan(row["cavity_width_strain_rate_abs_median"]))
        self.assertTrue(math.isnan(row["cavity_width_strain_rate_abs_p95"]))

    def test_rsr_shape_mismatch_raises(self):
        inputs = self.inputs()
        inputs["anatomical_pair_qc_valid"] = np.ones((2, 2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "pair_qc_valid must match"):
            extract_formal_candidate_case_features(**inputs)

    def test_anatomical_shapes_must_match_contract(self):
        for field, bad_value, message in (
            ("qc_cavity_width_strain_rate_s", np.ones((2, 2)), "cavity strain"),
            ("cavity_qc_valid", np.ones((2, 2), dtype=bool), "cavity_qc_valid"),
            (
                "qc_longitudinal_wall_strain_rate_s",
                np.ones((2, 2, 3)),
                "longitudinal strain",
            ),
            (
                "longitudinal_qc_valid",
                np.ones((2, 2, 3), dtype=bool),
                "longitudinal_qc_valid",
            ),
        ):
            with self.subTest(field=field):
                inputs = self.inputs()
                inputs[field] = bad_value
                with self.assertRaisesRegex(ValueError, message):
                    extract_formal_candidate_case_features(**inputs)

    def test_fps_mismatch_raises(self):
        inputs = self.inputs()
        inputs["anatomical_fps"] = 25.0
        with self.assertRaisesRegex(ValueError, "fps must match"):
            extract_formal_candidate_case_features(**inputs)

    def test_dicom_mask_shape_mismatch_raises(self):
        inputs = self.inputs()
        inputs["dicom_pair_qc_valid"] = np.ones((2, 2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "DICOM pair_qc_valid"):
            extract_formal_candidate_case_features(**inputs)

    def test_missing_dicom_preserves_case_and_non_dicom_features(self):
        row = extract_formal_candidate_case_features(**self.inputs())
        self.assertEqual(row["case_id"], "TEST_CASE")
        self.assertFalse(row["dicom_physical_curvature_available"])
        self.assertFalse(row["dicom_physical_curvature_rate_available"])
        self.assertEqual(row["rsr_abs_median"], 1.0)
        rate_features = [
            name for name in BIOLOGICAL_FEATURE_COLUMNS if "curvature_change_rate" in name
        ]
        self.assertEqual(len(rate_features), 6)
        self.assertTrue(all(math.isnan(row[name]) for name in rate_features))

    def test_column_contract_separates_model_inputs(self):
        excluded = {
            "rsr_valid_ratio",
            "cavity_valid_ratio",
            "longitudinal_valid_ratio",
            "curvature_valid_ratio",
            "dicom_physical_curvature_available",
            "dicom_physical_curvature_rate_available",
        }
        self.assertEqual(len(IDENTIFIER_COLUMNS), 1)
        self.assertEqual(len(QC_METADATA_COLUMNS), 6)
        self.assertEqual(len(BIOLOGICAL_FEATURE_COLUMNS), 20)
        self.assertEqual(len(FORMAL_CANDIDATE_FEATURE_COLUMNS), 27)
        self.assertEqual(MODEL_INPUT_FEATURE_COLUMNS, BIOLOGICAL_FEATURE_COLUMNS)
        self.assertTrue(excluded.isdisjoint(BIOLOGICAL_FEATURE_COLUMNS))
        self.assertTrue(excluded.isdisjoint(MODEL_INPUT_FEATURE_COLUMNS))

    def test_formal_qc_lineage_mismatch_fails_closed(self):
        valid = {
            "current_qc_semantics": np.asarray(FORMAL_QC_SEMANTICS),
            "artifact_qc_version": np.asarray(FORMAL_QC_VERSION),
            FORMAL_QC_LINEAGE_ID_FIELD: np.asarray("LINEAGE_A"),
        }
        validate_formal_qc_lineage(valid, source="TEST_CASE")
        for field, actual in (
            ("current_qc_semantics", "old_semantics"),
            ("artifact_qc_version", "2.0"),
        ):
            with self.subTest(field=field):
                payload = dict(valid)
                payload[field] = np.asarray(actual)
                with self.assertRaisesRegex(
                    ValueError, rf"{field} mismatch; expected=.*actual="
                ):
                    validate_formal_qc_lineage(payload, source="TEST_CASE")

    def test_missing_formal_qc_lineage_field_fails_closed(self):
        with self.assertRaisesRegex(KeyError, "artifact_qc_version"):
            validate_formal_qc_lineage(
                {"current_qc_semantics": np.asarray(FORMAL_QC_SEMANTICS)},
                source="TEST_CASE",
            )

    def test_upstream_artifact_version_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "expected='2.1'.*actual='2.0'"):
            validate_formal_qc_version(
                {"artifact_qc_version": np.asarray("2.0")},
                source="TEST_CASE",
            )


if __name__ == "__main__":
    unittest.main()
