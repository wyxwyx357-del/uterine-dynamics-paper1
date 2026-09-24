from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "代码" / "03_实验与历史代码"))

from paper1_feature_validity.real_reference_measurements import (  # noqa: E402
    compare_algorithm_to_reference,
    compute_all_reference_measurements,
    summarize_patient_level_algorithm_reference,
    summarize_observer_agreement,
    validate_annotations,
    validate_tasks,
)


def _task(measure: str, side: str, *, calibration: str = "NOT_REQUIRED") -> dict[str, str]:
    return {
        "task_id": f"task_{measure}_{side}",
        "case_id": "case_anon_01",
        "video_token": "video_token_01",
        "measure": measure,
        "frame_i": "0",
        "frame_j": "1",
        "side": side,
        "section_i": "1",
        "section_j": "2" if measure in {"F09", "F15"} else "",
        "section_k": "3" if measure == "F15" else "",
        "dt_s": "1",
        "dicom_calibration_status": calibration,
        "scale_x_mm_per_px": "0.1" if calibration == "VALID" else "",
        "scale_y_mm_per_px": "0.1" if calibration == "VALID" else "",
        "image_ref": "",
    }


def _annotation_rows(task: dict[str, str], annotator: str, points: dict[str, tuple[float, float]]) -> list[dict[str, str]]:
    rows = []
    for role, (x_px, y_px) in points.items():
        rows.append(
            {
                "task_id": task["task_id"],
                "annotator_id": annotator,
                "point_role": role,
                "frame_index": task["frame_i"] if role.startswith("source_") else task["frame_j"],
                "x_px": str(x_px),
                "y_px": str(y_px),
                "visible": "1",
                "measurability": "MEASURABLE",
                "note": "",
            }
        )
    return rows


def test_independent_reference_formulas_compute_four_requested_measures() -> None:
    tasks = [
        _task("F01", "anterior"),
        _task("F07", "both"),
        _task("F09", "anterior"),
        _task("F15", "anterior", calibration="VALID"),
    ]
    tasks = validate_tasks(tasks)
    annotations = []
    annotations += _annotation_rows(tasks[0], "A", {
        "source_inner": (0, 0), "source_outer": (0, 2), "target_inner": (0, 0), "target_outer": (0, 3)
    })
    annotations += _annotation_rows(tasks[1], "A", {
        "source_anterior": (0, 0), "source_posterior": (0, 10), "target_anterior": (0, 0), "target_posterior": (0, 12)
    })
    annotations += _annotation_rows(tasks[2], "A", {
        "source_a": (0, 0), "source_b": (2, 0), "target_a": (0, 0), "target_b": (3, 0)
    })
    annotations += _annotation_rows(tasks[3], "A", {
        "source_a": (0, 0), "source_b": (1, 0), "source_c": (2, 1),
        "target_a": (0, 0), "target_b": (1, 0), "target_c": (2, 2),
    })
    results = compute_all_reference_measurements(tasks, annotations)
    by_measure = {row["measure"]: row for row in results}
    assert by_measure["F01"]["status"] == "VALID"
    assert np.isclose(by_measure["F01"]["value"], 0.5)
    assert np.isclose(by_measure["F07"]["value"], 0.2)
    assert np.isclose(by_measure["F09"]["value"], 0.5)
    assert by_measure["F15"]["status"] == "VALID"
    assert by_measure["F15"]["unit"] == "1/(mm*s)"


def test_f15_missing_calibration_is_not_evaluable_and_not_rescaled() -> None:
    task = validate_tasks([_task("F15", "posterior", calibration="MISSING")])[0]
    annotations = _annotation_rows(task, "A", {
        "source_a": (0, 0), "source_b": (1, 0), "source_c": (2, 1),
        "target_a": (0, 0), "target_b": (1, 0), "target_c": (2, 2),
    })
    row = compute_all_reference_measurements([task], annotations)[0]
    assert row["status"] == "NOT_EVALUABLE"
    assert row["reason"] == "physical_calibration_unavailable"


def test_two_observer_and_algorithm_agreement_are_task_level() -> None:
    task = validate_tasks([_task("F01", "anterior")])[0]
    points = {"source_inner": (0, 0), "source_outer": (0, 2), "target_inner": (0, 0), "target_outer": (0, 3)}
    annotations = _annotation_rows(task, "A", points) + _annotation_rows(task, "B", {
        **points, "target_outer": (0, 3.2)
    })
    measurements = compute_all_reference_measurements([task], annotations)
    agreement = summarize_observer_agreement([task], measurements)[0]
    assert agreement["evaluable_two_observer_tasks"] == 1
    assert agreement["coverage"] == 1.0
    algorithm = [{"task_id": task["task_id"], "measure": "F01", "side": "anterior", "algorithm_value": "0.5", "algorithm_status": "VALID"}]
    comparison = compare_algorithm_to_reference([task], measurements, algorithm)[0]
    assert comparison["evaluable_algorithm_reference_tasks"] == 1


def test_algorithm_primary_summary_aggregates_repeated_tasks_within_case() -> None:
    first = validate_tasks([_task("F01", "anterior")])[0]
    second = dict(first)
    second["task_id"] = "task_F01_anterior_second_frame"
    tasks = validate_tasks([first, second])
    points = {"source_inner": (0, 0), "source_outer": (0, 2), "target_inner": (0, 0), "target_outer": (0, 3)}
    annotations = _annotation_rows(first, "A", points) + _annotation_rows(second, "A", points)
    measurements = compute_all_reference_measurements(tasks, annotations)
    algorithm = [
        {"task_id": first["task_id"], "measure": "F01", "side": "anterior", "algorithm_value": "0.5", "algorithm_status": "VALID"},
        {"task_id": second["task_id"], "measure": "F01", "side": "anterior", "algorithm_value": "0.7", "algorithm_status": "VALID"},
    ]
    summary = summarize_patient_level_algorithm_reference(tasks, measurements, algorithm)[0]
    assert summary["expected_patients"] == 1
    assert summary["evaluable_patients"] == 1
    assert np.isclose(summary["bias"], 0.1)


def test_annotations_require_complete_independent_roles() -> None:
    task = validate_tasks([_task("F07", "both")])[0]
    incomplete = _annotation_rows(task, "A", {
        "source_anterior": (0, 0), "source_posterior": (0, 10), "target_anterior": (0, 0)
    })
    validated = validate_annotations(incomplete, [task])
    result = compute_all_reference_measurements([task], validated)[0]
    assert result["status"] == "NOT_EVALUABLE"
    assert result["reason"] == "missing_or_extra_point_roles"
