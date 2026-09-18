from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from peristalsis_pipeline.formal_qc_contract import (
    FORMAL_QC_LINEAGE_ID_FIELD,
    FORMAL_QC_SEMANTICS,
    FORMAL_QC_VERSION,
)
from peristalsis_pipeline.dicom_curvature_calibration import (
    DICOM_CURVATURE_RATE_TIMEBASE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    PROJECT_ROOT
    / "代码"
    / "02_已并入主流程"
    / "05_输出审查与发布"
    / "run_feature_v1_formal_candidate_5case.py"
)
SPEC = importlib.util.spec_from_file_location("formal_candidate_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
MINIMAL_RUNNER_PATH = RUNNER_PATH.with_name("run_feature_v1_minimal_5case.py")
MINIMAL_SPEC = importlib.util.spec_from_file_location(
    "minimal_feature_runner", MINIMAL_RUNNER_PATH
)
assert MINIMAL_SPEC is not None and MINIMAL_SPEC.loader is not None
MINIMAL_RUNNER = importlib.util.module_from_spec(MINIMAL_SPEC)
MINIMAL_SPEC.loader.exec_module(MINIMAL_RUNNER)


class FormalCandidateEntrypointTest(unittest.TestCase):
    @staticmethod
    def write_rsr(
        path: Path, *, semantics: str, version: str, lineage_id: str = "LINEAGE_A"
    ) -> None:
        np.savez_compressed(
            path,
            current_qc_radial_strain_rate_s=np.ones((2, 2, 3), dtype=np.float32),
            current_qc_semantics=np.asarray(semantics),
            artifact_qc_version=np.asarray(version),
            case_id=np.asarray("TEST_CASE"),
            **{FORMAL_QC_LINEAGE_ID_FIELD: np.asarray(lineage_id)},
            fps=np.asarray(20.0),
        )

    @staticmethod
    def write_anatomical(
        path: Path, *, semantics: str, version: str, lineage_id: str = "LINEAGE_A"
    ) -> None:
        np.savez_compressed(
            path,
            pair_qc_valid=np.ones((2, 2, 3), dtype=bool),
            cavity_qc_valid=np.ones((2, 3), dtype=bool),
            longitudinal_qc_valid=np.ones((2, 2, 2), dtype=bool),
            qc_cavity_width_strain_rate_s=np.ones((2, 3), dtype=np.float32),
            qc_longitudinal_wall_strain_rate_s=np.ones(
                (2, 2, 2), dtype=np.float32
            ),
            fps=np.asarray(20.0),
            evidence_layer=np.asarray("roi_deformation_activity"),
            qc_feature_branch_is_formal=np.asarray(True),
            formal_propagation_released=np.asarray(False),
            current_qc_semantics=np.asarray(semantics),
            artifact_qc_version=np.asarray(version),
            case_id=np.asarray("TEST_CASE"),
            **{FORMAL_QC_LINEAGE_ID_FIELD: np.asarray(lineage_id)},
        )

    @staticmethod
    def write_dicom(
        path: Path,
        *,
        timebase: str,
        numeric_match: bool,
        lineage_id: str = "LINEAGE_A",
    ) -> None:
        np.savez_compressed(
            path,
            case_id=np.asarray("TEST_CASE"),
            pair_qc_valid=np.ones((2, 2, 3), dtype=bool),
            qc_wall_curvature_change_rate_mm_inv_s=np.ones(
                (2, 2, 3), dtype=np.float32
            ),
            physical_curvature_available=np.asarray(True),
            physical_curvature_rate_available=np.asarray(True),
            curvature_rate_timebase=np.asarray(timebase),
            dicom_timing_numeric_match=np.asarray(numeric_match),
            current_qc_semantics=np.asarray(FORMAL_QC_SEMANTICS),
            artifact_qc_version=np.asarray(FORMAL_QC_VERSION),
            **{FORMAL_QC_LINEAGE_ID_FIELD: np.asarray(lineage_id)},
        )

    def test_entrypoint_accepts_only_matching_formal_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rsr_path = root / "TEST_CASE_rsr_fourway_v1.npz"
            anatomical_dir = root / "anatomical"
            dicom_dir = root / "dicom"
            anatomical_dir.mkdir()
            dicom_dir.mkdir()
            anatomical_path = anatomical_dir / "TEST_CASE_anatomical_deformation_v1.npz"
            self.write_rsr(
                rsr_path,
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
            )
            self.write_anatomical(
                anatomical_path,
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
            )
            row = RUNNER.extract_case(rsr_path, anatomical_dir, dicom_dir)
            self.assertEqual(row["case_id"], "TEST_CASE")
            self.assertEqual(len(row), 27)

            self.write_rsr(
                rsr_path,
                semantics="old_semantics",
                version=FORMAL_QC_VERSION,
            )
            with self.assertRaisesRegex(ValueError, "current_qc_semantics mismatch"):
                RUNNER.extract_case(rsr_path, anatomical_dir, dicom_dir)

            self.write_rsr(
                rsr_path,
                semantics=FORMAL_QC_SEMANTICS,
                version="2.0",
            )
            with self.assertRaisesRegex(ValueError, "artifact_qc_version mismatch"):
                RUNNER.extract_case(rsr_path, anatomical_dir, dicom_dir)

    def test_entrypoint_rejects_anatomical_lineage_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rsr_path = root / "TEST_CASE_rsr_fourway_v1.npz"
            anatomical_dir = root / "anatomical"
            dicom_dir = root / "dicom"
            anatomical_dir.mkdir()
            dicom_dir.mkdir()
            self.write_rsr(
                rsr_path,
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
            )
            self.write_anatomical(
                anatomical_dir / "TEST_CASE_anatomical_deformation_v1.npz",
                semantics=FORMAL_QC_SEMANTICS,
                version="2.0",
            )
            with self.assertRaisesRegex(ValueError, "artifact_qc_version mismatch"):
                RUNNER.extract_case(rsr_path, anatomical_dir, dicom_dir)

    def test_entrypoint_rejects_old_or_unverified_dicom_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rsr_path = root / "TEST_CASE_rsr_fourway_v1.npz"
            anatomical_dir = root / "anatomical"
            dicom_dir = root / "dicom"
            anatomical_dir.mkdir()
            dicom_dir.mkdir()
            self.write_rsr(
                rsr_path,
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
            )
            self.write_anatomical(
                anatomical_dir / "TEST_CASE_anatomical_deformation_v1.npz",
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
            )
            dicom_path = dicom_dir / "TEST_CASE_dicom_curvature_v1.npz"
            self.write_dicom(
                dicom_path,
                timebase="analysis_fps",
                numeric_match=True,
            )
            with self.assertRaisesRegex(ValueError, "timebase mismatch"):
                RUNNER.extract_case(rsr_path, anatomical_dir, dicom_dir)

            self.write_dicom(
                dicom_path,
                timebase=DICOM_CURVATURE_RATE_TIMEBASE,
                numeric_match=False,
            )
            with self.assertRaisesRegex(ValueError, "numeric DICOM timing match"):
                RUNNER.extract_case(rsr_path, anatomical_dir, dicom_dir)

    def test_entrypoint_rejects_different_run_with_same_semantics_and_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rsr_path = root / "TEST_CASE_rsr_fourway_v1.npz"
            anatomical_dir = root / "anatomical"
            dicom_dir = root / "dicom"
            anatomical_dir.mkdir()
            dicom_dir.mkdir()
            self.write_rsr(
                rsr_path,
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
                lineage_id="RUN_A",
            )
            self.write_anatomical(
                anatomical_dir / "TEST_CASE_anatomical_deformation_v1.npz",
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
                lineage_id="RUN_B",
            )

            with self.assertRaisesRegex(ValueError, "formal_qc_lineage_id mismatch"):
                RUNNER.extract_case(rsr_path, anatomical_dir, dicom_dir)

    def test_minimal_rejects_missing_dicom_case_id_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rsr_path = root / "TEST_CASE_rsr_fourway_v1.npz"
            dicom_dir = root / "dicom"
            dicom_dir.mkdir()
            self.write_rsr(
                rsr_path,
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
            )
            dicom_path = dicom_dir / "TEST_CASE_dicom_curvature_v1.npz"
            np.savez_compressed(
                dicom_path,
                pair_qc_valid=np.ones((2, 2, 3), dtype=bool),
                qc_wall_curvature_change_rate_mm_inv_s=np.ones(
                    (2, 2, 3), dtype=np.float32
                ),
                physical_curvature_available=np.asarray(True),
                physical_curvature_rate_available=np.asarray(True),
                curvature_rate_timebase=np.asarray(DICOM_CURVATURE_RATE_TIMEBASE),
                dicom_timing_numeric_match=np.asarray(True),
                current_qc_semantics=np.asarray(FORMAL_QC_SEMANTICS),
                artifact_qc_version=np.asarray(FORMAL_QC_VERSION),
                **{FORMAL_QC_LINEAGE_ID_FIELD: np.asarray("LINEAGE_A")},
            )

            with self.assertRaisesRegex(KeyError, "case_id"):
                MINIMAL_RUNNER.extract_case(rsr_path, dicom_dir)

    def test_minimal_rejects_different_formal_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rsr_path = root / "TEST_CASE_rsr_fourway_v1.npz"
            dicom_dir = root / "dicom"
            dicom_dir.mkdir()
            self.write_rsr(
                rsr_path,
                semantics=FORMAL_QC_SEMANTICS,
                version=FORMAL_QC_VERSION,
                lineage_id="RUN_A",
            )
            self.write_dicom(
                dicom_dir / "TEST_CASE_dicom_curvature_v1.npz",
                timebase=DICOM_CURVATURE_RATE_TIMEBASE,
                numeric_match=True,
                lineage_id="RUN_B",
            )

            with self.assertRaisesRegex(ValueError, "formal_qc_lineage_id mismatch"):
                MINIMAL_RUNNER.extract_case(rsr_path, dicom_dir)


if __name__ == "__main__":
    unittest.main()
