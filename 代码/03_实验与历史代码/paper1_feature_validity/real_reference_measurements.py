from __future__ import annotations

"""Independent reference measurements for four Paper 1 feature families.

This module consumes anonymized task metadata and points marked directly on the
raw image.  It never consumes the frozen tracking coordinates.  The resulting
values are reference measurements of the requested geometry; F01 is explicitly
not interpreted as proof of tissue identity or true material strain.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


REFERENCE_MEASURES = ("F01", "F07", "F09", "F15")
VALID_CALIBRATION_STATUSES = {"VALID", "MISSING", "INVALID", "NOT_CHECKED", "NOT_REQUIRED"}
MEASURABILITY_VALUES = {"MEASURABLE", "NOT_MEASURABLE", "UNCERTAIN"}

TASK_COLUMNS = (
    "task_id",
    "case_id",
    "video_token",
    "measure",
    "frame_i",
    "frame_j",
    "side",
    "section_i",
    "section_j",
    "section_k",
    "dt_s",
    "dicom_calibration_status",
    "scale_x_mm_per_px",
    "scale_y_mm_per_px",
    "image_ref",
)

ANNOTATION_COLUMNS = (
    "task_id",
    "annotator_id",
    "point_role",
    "frame_index",
    "x_px",
    "y_px",
    "visible",
    "measurability",
    "note",
)

ALGORITHM_RESULT_COLUMNS = (
    "task_id",
    "measure",
    "side",
    "algorithm_value",
    "algorithm_status",
)

_FORBIDDEN_TOKENS = (
    "pregnan",
    "outcome",
    "auc",
    "patient_name",
    "patientid",
    "patient_id",
    "fullname",
)

_ROLES = {
    "F01": ("source_inner", "source_outer", "target_inner", "target_outer"),
    "F07": ("source_anterior", "source_posterior", "target_anterior", "target_posterior"),
    "F09": ("source_a", "source_b", "target_a", "target_b"),
    "F15": (
        "source_a",
        "source_b",
        "source_c",
        "target_a",
        "target_b",
        "target_c",
    ),
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if any(name is None or not str(name).strip() for name in reader.fieldnames):
            raise ValueError(f"CSV has an empty header: {path}")
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Iterable[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fields})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _check_required(rows: list[dict[str, str]], required: Iterable[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    missing = set(required) - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _reject_forbidden_headers(rows: list[dict[str, str]], label: str) -> None:
    if not rows:
        return
    bad = [
        name
        for name in rows[0]
        if any(token in name.lower().replace("-", "_") for token in _FORBIDDEN_TOKENS)
    ]
    if bad:
        raise ValueError(f"{label} contains forbidden outcome/identity columns: {bad}")


def _required_roles(measure: str) -> tuple[str, ...]:
    try:
        return _ROLES[measure]
    except KeyError as exc:
        raise ValueError(f"unsupported reference measure: {measure}") from exc


def _float_or_none(value: object, field: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite when present")
    return result


def _int_field(value: object, field: str) -> int:
    result = int(str(value).strip())
    return result


def _parse_binary(value: object, field: str) -> bool:
    normalized = str(value).strip().upper()
    if normalized in {"1", "TRUE", "YES"}:
        return True
    if normalized in {"0", "FALSE", "NO"}:
        return False
    raise ValueError(f"{field} must be 0/1")


def validate_tasks(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    _check_required(rows, TASK_COLUMNS, "task list")
    _reject_forbidden_headers(rows, "task list")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = {key: str(value or "").strip() for key, value in row.items()}
        task_id = item["task_id"]
        if not task_id or task_id in seen:
            raise ValueError(f"task_id must be non-empty and unique: {task_id!r}")
        seen.add(task_id)
        if not item["case_id"] or not item["video_token"]:
            raise ValueError(f"task {task_id} needs case_id and anonymous video_token")
        if item["measure"] not in REFERENCE_MEASURES:
            raise ValueError(f"task {task_id} has unsupported measure {item['measure']!r}")
        expected_sides = {
            "F01": {"anterior", "posterior"},
            "F07": {"both"},
            "F09": {"anterior", "posterior"},
            "F15": {"anterior", "posterior"},
        }[item["measure"]]
        if item["side"] not in expected_sides:
            raise ValueError(f"task {task_id} has invalid side {item['side']!r}")
        frame_i = _int_field(item["frame_i"], f"{task_id}.frame_i")
        frame_j = _int_field(item["frame_j"], f"{task_id}.frame_j")
        if frame_i < 0 or frame_j != frame_i + 1:
            raise ValueError(f"task {task_id} must contain adjacent non-negative frames")
        section_i = _int_field(item["section_i"], f"{task_id}.section_i")
        if section_i < 0:
            raise ValueError(f"task {task_id}.section_i must be non-negative")
        if item["measure"] in {"F09", "F15"}:
            section_j = _int_field(item["section_j"], f"{task_id}.section_j")
            if section_j != section_i + 1:
                raise ValueError(f"task {task_id} requires adjacent section_i/section_j")
        if item["measure"] == "F15":
            section_k = _int_field(item["section_k"], f"{task_id}.section_k")
            if section_k != section_i + 2:
                raise ValueError(f"task {task_id} requires three consecutive F15 sections")
        dt_s = _float_or_none(item["dt_s"], f"{task_id}.dt_s")
        if dt_s is not None and dt_s <= 0:
            raise ValueError(f"task {task_id}.dt_s must be positive")
        status = item["dicom_calibration_status"].upper()
        if status not in VALID_CALIBRATION_STATUSES:
            raise ValueError(f"task {task_id} has invalid DICOM calibration status")
        for name in ("scale_x_mm_per_px", "scale_y_mm_per_px"):
            scale = _float_or_none(item[name], f"{task_id}.{name}")
            if scale is not None and scale <= 0:
                raise ValueError(f"task {task_id}.{name} must be positive")
        if item["measure"] == "F15" and status == "VALID":
            if any(_float_or_none(item[name], f"{task_id}.{name}") is None for name in ("scale_x_mm_per_px", "scale_y_mm_per_px", "dt_s")):
                raise ValueError(f"task {task_id} marked VALID for F15 but calibration/timing is incomplete")
        normalized.append(item)
    return normalized


def validate_annotations(
    rows: list[dict[str, str]], tasks: list[dict[str, str]]
) -> list[dict[str, str]]:
    _check_required(rows, ANNOTATION_COLUMNS, "annotation table")
    _reject_forbidden_headers(rows, "annotation table")
    task_by_id = {row["task_id"]: row for row in tasks}
    seen: set[tuple[str, str, str]] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = {key: str(value or "").strip() for key, value in row.items()}
        task_id = item["task_id"]
        if task_id not in task_by_id:
            raise ValueError(f"annotation refers to unknown task {task_id!r}")
        if not item["annotator_id"]:
            raise ValueError(f"annotation {task_id} has no annotator_id")
        task = task_by_id[task_id]
        if item["point_role"] not in _required_roles(task["measure"]):
            raise ValueError(f"unknown point role for {task['measure']}: {item['point_role']!r}")
        key = (task_id, item["annotator_id"], item["point_role"])
        if key in seen:
            raise ValueError(f"duplicate annotation row: {key}")
        seen.add(key)
        frame_index = _int_field(item["frame_index"], f"{task_id}.frame_index")
        expected_frame = int(task["frame_i"] if item["point_role"].startswith("source_") else task["frame_j"])
        if frame_index != expected_frame:
            raise ValueError(f"{task_id}.{item['point_role']} is assigned to the wrong frame")
        visible = _parse_binary(item["visible"], f"{task_id}.visible")
        measurability = item["measurability"].upper()
        if measurability not in MEASURABILITY_VALUES:
            raise ValueError(f"{task_id}.measurability is invalid")
        x_px = _float_or_none(item["x_px"], f"{task_id}.x_px")
        y_px = _float_or_none(item["y_px"], f"{task_id}.y_px")
        if visible and (x_px is None or y_px is None):
            raise ValueError(f"visible annotation {task_id}.{item['point_role']} needs x_px and y_px")
        item["visible"] = "1" if visible else "0"
        item["measurability"] = measurability
        normalized.append(item)
    return normalized


def _points_for_group(
    task: Mapping[str, str], group: list[dict[str, str]]
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    role_map = {row["point_role"]: row for row in group}
    required = _required_roles(task["measure"])
    if set(role_map) != set(required):
        return None, "missing_or_extra_point_roles"
    points: dict[str, np.ndarray] = {}
    for role in required:
        row = role_map[role]
        if row["visible"] != "1" or row["measurability"] != "MEASURABLE":
            return None, "point_not_measurable"
        x_px = _float_or_none(row["x_px"], f"{task['task_id']}.{role}.x_px")
        y_px = _float_or_none(row["y_px"], f"{task['task_id']}.{role}.y_px")
        if x_px is None or y_px is None:
            return None, "point_coordinate_missing"
        points[role] = np.asarray((x_px, y_px), dtype=np.float64)
    return points, None


def _distance(first: np.ndarray, second: np.ndarray) -> float:
    value = float(np.linalg.norm(second - first))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("reference segment length must be positive")
    return value


def _curvature(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    ab = second - first
    bc = third - second
    ac = third - first
    denominator = float(np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ac))
    if denominator <= 1e-12:
        raise ValueError("reference curvature has a degenerate three-point geometry")
    cross = abs(float(ab[0] * ac[1] - ab[1] * ac[0]))
    return 2.0 * cross / denominator


def compute_reference_measurement(
    task: Mapping[str, str], group: list[dict[str, str]]
) -> dict[str, object]:
    points, reason = _points_for_group(task, group)
    base = {
        "task_id": task["task_id"],
        "case_id": task["case_id"],
        "measure": task["measure"],
        "side": task["side"],
        "annotator_id": group[0]["annotator_id"] if group else "",
        "status": "NOT_EVALUABLE",
        "value": math.nan,
        "unit": "1/s" if task["measure"] != "F15" else "1/(mm*s)",
        "reason": reason or "",
        "semantic_note": "",
    }
    if points is None:
        return base
    if task["measure"] == "F15" and task["dicom_calibration_status"].upper() != "VALID":
        base["reason"] = "physical_calibration_unavailable"
        return base
    dt_s = _float_or_none(task["dt_s"], f"{task['task_id']}.dt_s")
    if dt_s is None or dt_s <= 0:
        base["reason"] = "pair_time_unavailable"
        return base
    try:
        if task["measure"] == "F01":
            source = _distance(points["source_inner"], points["source_outer"])
            target = _distance(points["target_inner"], points["target_outer"])
            value = (target - source) / source / dt_s
            base["semantic_note"] = "paired_geometric_reference_not_tissue_identity_or_material_strain"
        elif task["measure"] == "F07":
            source = _distance(points["source_anterior"], points["source_posterior"])
            target = _distance(points["target_anterior"], points["target_posterior"])
            value = (target - source) / source / dt_s
        elif task["measure"] == "F09":
            source = _distance(points["source_a"], points["source_b"])
            target = _distance(points["target_a"], points["target_b"])
            value = (target - source) / source / dt_s
        else:
            scale_x = _float_or_none(task["scale_x_mm_per_px"], f"{task['task_id']}.scale_x_mm_per_px")
            scale_y = _float_or_none(task["scale_y_mm_per_px"], f"{task['task_id']}.scale_y_mm_per_px")
            if scale_x is None or scale_y is None or scale_x <= 0 or scale_y <= 0:
                base["reason"] = "physical_calibration_scale_missing"
                return base
            scale = np.asarray((scale_x, scale_y), dtype=np.float64)
            source_points = [points[name] * scale for name in ("source_a", "source_b", "source_c")]
            target_points = [points[name] * scale for name in ("target_a", "target_b", "target_c")]
            value = (_curvature(*target_points) - _curvature(*source_points)) / dt_s
        if not math.isfinite(float(value)):
            raise ValueError("reference result is not finite")
    except ValueError as exc:
        base["reason"] = str(exc)
        return base
    base.update({"status": "VALID", "value": float(value), "reason": ""})
    return base


def compute_all_reference_measurements(
    tasks: list[dict[str, str]], annotations: list[dict[str, str]]
) -> list[dict[str, object]]:
    validated_tasks = validate_tasks(tasks)
    validated_annotations = validate_annotations(annotations, validated_tasks)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in validated_annotations:
        groups[(row["task_id"], row["annotator_id"])].append(row)
    results: list[dict[str, object]] = []
    for task in validated_tasks:
        task_groups = [group for (task_id, _), group in groups.items() if task_id == task["task_id"]]
        for group in sorted(task_groups, key=lambda value: value[0]["annotator_id"]):
            results.append(compute_reference_measurement(task, group))
    return results


def _agreement_stats(differences: np.ndarray) -> dict[str, float | int]:
    differences = np.asarray(differences, dtype=np.float64)
    if differences.size == 0:
        return {"n": 0, "bias": math.nan, "mae": math.nan, "rmse": math.nan, "loa_lower": math.nan, "loa_upper": math.nan}
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if differences.size >= 2 else math.nan
    return {
        "n": int(differences.size),
        "bias": bias,
        "mae": float(np.mean(np.abs(differences))),
        "rmse": float(np.sqrt(np.mean(differences * differences))),
        "loa_lower": bias - 1.96 * sd if math.isfinite(sd) else math.nan,
        "loa_upper": bias + 1.96 * sd if math.isfinite(sd) else math.nan,
    }


def summarize_observer_agreement(
    tasks: list[dict[str, str]], measurements: list[dict[str, object]]
) -> list[dict[str, object]]:
    validated_tasks = validate_tasks(tasks)
    expected_by_measure = defaultdict(int)
    for task in validated_tasks:
        expected_by_measure[task["measure"]] += 1
    by_task: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in measurements:
        if row.get("status") == "VALID":
            by_task[(str(row["task_id"]), str(row["measure"]))].append(row)
    output: list[dict[str, object]] = []
    for measure in REFERENCE_MEASURES:
        differences = []
        pair_count = 0
        for (task_id, task_measure), rows in by_task.items():
            if task_measure != measure or len(rows) < 2:
                continue
            ordered = sorted(rows, key=lambda row: str(row["annotator_id"]))
            differences.append(float(ordered[1]["value"]) - float(ordered[0]["value"]))
            pair_count += 1
        stats = _agreement_stats(np.asarray(differences, dtype=np.float64))
        output.append(
            {
                "measure": measure,
                "expected_tasks": expected_by_measure[measure],
                "evaluable_two_observer_tasks": pair_count,
                "coverage": pair_count / expected_by_measure[measure] if expected_by_measure[measure] else math.nan,
                **stats,
            }
        )
    return output


def compare_algorithm_to_reference(
    tasks: list[dict[str, str]],
    measurements: list[dict[str, object]],
    algorithm_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    _check_required(algorithm_rows, ALGORITHM_RESULT_COLUMNS, "algorithm result table")
    _reject_forbidden_headers(algorithm_rows, "algorithm result table")
    task_by_id = {row["task_id"]: row for row in validate_tasks(tasks)}
    reference_by_task: dict[str, list[float]] = defaultdict(list)
    for row in measurements:
        if row.get("status") == "VALID":
            reference_by_task[str(row["task_id"])].append(float(row["value"]))
    by_measure: dict[str, list[float]] = defaultdict(list)
    expected = defaultdict(int)
    seen_algorithm_tasks: set[tuple[str, str, str]] = set()
    for task in task_by_id.values():
        expected[task["measure"]] += 1
    for row in algorithm_rows:
        task_id = str(row["task_id"]).strip()
        if task_id not in task_by_id:
            raise ValueError(f"algorithm result refers to unknown task {task_id!r}")
        task = task_by_id[task_id]
        if row["measure"].strip() != task["measure"] or row["side"].strip() != task["side"]:
            raise ValueError(f"algorithm result metadata does not match task {task_id}")
        algorithm_key = (task_id, row["measure"].strip(), row["side"].strip())
        if algorithm_key in seen_algorithm_tasks:
            raise ValueError(f"duplicate algorithm result for task {task_id}")
        seen_algorithm_tasks.add(algorithm_key)
        status = row["algorithm_status"].strip().upper()
        if status != "VALID" or not reference_by_task.get(task_id):
            continue
        algorithm_value = _float_or_none(row["algorithm_value"], f"{task_id}.algorithm_value")
        if algorithm_value is None:
            continue
        reference_value = float(np.mean(reference_by_task[task_id]))
        by_measure[task["measure"]].append(algorithm_value - reference_value)
    output = []
    for measure in REFERENCE_MEASURES:
        stats = _agreement_stats(np.asarray(by_measure[measure], dtype=np.float64))
        output.append(
            {
                "measure": measure,
                "expected_tasks": expected[measure],
                "evaluable_algorithm_reference_tasks": stats["n"],
                "coverage": stats["n"] / expected[measure] if expected[measure] else math.nan,
                **{key: value for key, value in stats.items() if key != "n"},
            }
        )
    return output


def summarize_patient_level_algorithm_reference(
    tasks: list[dict[str, str]],
    measurements: list[dict[str, object]],
    algorithm_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Compare case-level means, keeping repeated frames within a case.

    Only tasks with both a valid algorithm value and at least one valid manual
    reference contribute to a case-level mean.  The case, not an individual
    frame, is the unit in the returned agreement statistics.
    """

    validated_tasks = validate_tasks(tasks)
    task_by_id = {row["task_id"]: row for row in validated_tasks}
    expected_cases: dict[str, set[str]] = defaultdict(set)
    for task in validated_tasks:
        expected_cases[task["measure"]].add(task["case_id"])
    reference_by_task: dict[str, list[float]] = defaultdict(list)
    for row in measurements:
        if row.get("status") == "VALID":
            reference_by_task[str(row["task_id"])].append(float(row["value"]))

    _check_required(algorithm_rows, ALGORITHM_RESULT_COLUMNS, "algorithm result table")
    _reject_forbidden_headers(algorithm_rows, "algorithm result table")
    seen_algorithm_tasks: set[tuple[str, str, str]] = set()
    matched_by_case_measure: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"reference": [], "algorithm": []}
    )
    for row in algorithm_rows:
        task_id = str(row["task_id"]).strip()
        if task_id not in task_by_id:
            raise ValueError(f"algorithm result refers to unknown task {task_id!r}")
        task = task_by_id[task_id]
        measure = row["measure"].strip()
        side = row["side"].strip()
        key = (task_id, measure, side)
        if measure != task["measure"] or side != task["side"]:
            raise ValueError(f"algorithm result metadata does not match task {task_id}")
        if key in seen_algorithm_tasks:
            raise ValueError(f"duplicate algorithm result for task {task_id}")
        seen_algorithm_tasks.add(key)
        if row["algorithm_status"].strip().upper() != "VALID":
            continue
        algorithm_value = _float_or_none(row["algorithm_value"], f"{task_id}.algorithm_value")
        if algorithm_value is None or not reference_by_task.get(task_id):
            continue
        case_measure = (task["case_id"], task["measure"])
        matched_by_case_measure[case_measure]["reference"].append(float(np.mean(reference_by_task[task_id])))
        matched_by_case_measure[case_measure]["algorithm"].append(algorithm_value)

    output: list[dict[str, object]] = []
    for measure in REFERENCE_MEASURES:
        differences = []
        for (case_id, case_measure), values in matched_by_case_measure.items():
            if case_measure != measure:
                continue
            differences.append(float(np.mean(values["algorithm"])) - float(np.mean(values["reference"])))
        stats = _agreement_stats(np.asarray(differences, dtype=np.float64))
        if stats["n"] >= 2:
            bias_se = (float(stats["loa_upper"]) - float(stats["loa_lower"])) / (2.0 * 1.96 * math.sqrt(int(stats["n"])))
            bias_ci_lower = float(stats["bias"]) - 1.96 * bias_se
            bias_ci_upper = float(stats["bias"]) + 1.96 * bias_se
        else:
            bias_ci_lower = math.nan
            bias_ci_upper = math.nan
        output.append(
            {
                "measure": measure,
                "expected_patients": len(expected_cases[measure]),
                "evaluable_patients": int(stats["n"]),
                "coverage": int(stats["n"]) / len(expected_cases[measure]) if expected_cases[measure] else math.nan,
                "n": stats["n"],
                "bias": stats["bias"],
                "bias_ci_lower": bias_ci_lower,
                "bias_ci_upper": bias_ci_upper,
                "mae": stats["mae"],
                "rmse": stats["rmse"],
                "loa_lower": stats["loa_lower"],
                "loa_upper": stats["loa_upper"],
            }
        )
    return output
