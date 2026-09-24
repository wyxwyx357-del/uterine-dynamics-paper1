from __future__ import annotations

"""Create an anonymized, blinded two-observer reference-measurement task list."""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR.parent))
from paper1_feature_validity.real_reference_measurements import (
    TASK_COLUMNS,
    read_csv_rows,
    validate_tasks,
    write_csv_rows,
)


CASE_COLUMNS = (
    "case_id",
    "video_token",
    "frame_count",
    "fps",
    "section_count",
    "quality_tier",
    "motion_coverage_tier",
    "tracking_risk",
    "dicom_calibration_status",
    "dicom_pair_dt_s",
    "scale_x_mm_per_px",
    "scale_y_mm_per_px",
    "source_video_ref",
)


def _required_case_columns(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("case manifest must contain at least one row")
    missing = set(CASE_COLUMNS[:9]) - set(rows[0])
    if missing:
        raise ValueError(f"case manifest is missing columns: {sorted(missing)}")
    forbidden = ("pregnan", "outcome", "auc", "patient_name", "patientid", "patient_id", "fullname")
    bad = [name for name in rows[0] if any(token in name.lower() for token in forbidden)]
    if bad:
        raise ValueError(f"case manifest contains forbidden outcome/identity columns: {bad}")


def _number(value: str, field: str, *, positive: bool = False) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def validate_case_manifest(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    _required_case_columns(rows)
    seen: set[str] = set()
    output = []
    for raw in rows:
        row = {key: str(value or "").strip() for key, value in raw.items()}
        case_id = row["case_id"]
        if not case_id or case_id in seen:
            raise ValueError(f"case_id must be non-empty and unique: {case_id!r}")
        seen.add(case_id)
        if not row["video_token"]:
            raise ValueError(f"{case_id} needs an anonymous video_token")
        frame_count = int(row["frame_count"])
        section_count = int(row["section_count"])
        fps = _number(row["fps"], f"{case_id}.fps", positive=True)
        if frame_count < 2 or section_count < 3 or fps is None:
            raise ValueError(f"{case_id} needs frame_count>=2, section_count>=3 and positive fps")
        status = row["dicom_calibration_status"].upper()
        if status not in {"VALID", "MISSING", "INVALID", "NOT_CHECKED"}:
            raise ValueError(f"{case_id} has invalid dicom_calibration_status")
        pair_dt = _number(row.get("dicom_pair_dt_s", ""), f"{case_id}.dicom_pair_dt_s", positive=True)
        scale_x = _number(row.get("scale_x_mm_per_px", ""), f"{case_id}.scale_x_mm_per_px", positive=True)
        scale_y = _number(row.get("scale_y_mm_per_px", ""), f"{case_id}.scale_y_mm_per_px", positive=True)
        if status == "VALID" and (pair_dt is None or scale_x is None or scale_y is None):
            raise ValueError(f"{case_id} marked VALID but DICOM timing/scale is incomplete")
        output.append(row)
    return output


def select_feasibility_cases(rows: list[dict[str, str]], n_cases: int) -> list[dict[str, str]]:
    if n_cases <= 0:
        raise ValueError("n_cases must be positive")
    if len(rows) < n_cases:
        raise ValueError(f"case manifest has {len(rows)} rows but {n_cases} are required")
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[(row["quality_tier"], row["motion_coverage_tier"])].append(row)
    for values in strata.values():
        values.sort(key=lambda row: row["case_id"])
    selected: list[dict[str, str]] = []
    keys = sorted(strata)
    while len(selected) < n_cases:
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop(0))
                progressed = True
                if len(selected) == n_cases:
                    break
        if not progressed:
            break
    return selected


def _frame_pair(frame_count: int, fraction: float) -> tuple[int, int]:
    if not 0 < fraction < 1:
        raise ValueError("frame fractions must be strictly between 0 and 1")
    frame_i = min(frame_count - 2, max(0, round(fraction * (frame_count - 2))))
    return frame_i, frame_i + 1


def build_tasks(
    cases: list[dict[str, str]], frame_fractions: tuple[float, ...]
) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for case in cases:
        section_count = int(case["section_count"])
        mid = section_count // 2
        frame_dt = 1.0 / float(case["fps"])
        for fraction in frame_fractions:
            frame_i, frame_j = _frame_pair(int(case["frame_count"]), fraction)
            prefix = f"{case['case_id']}_f{fraction:.3f}_t{frame_i:05d}"

            def add(
                measure: str,
                side: str,
                suffix: str,
                *,
                section_i: int,
                section_j: int = -1,
                section_k: int = -1,
            ) -> None:
                is_physical = measure == "F15"
                tasks.append(
                    {
                        "task_id": f"{prefix}_{measure}_{side}_{suffix}",
                        "case_id": case["case_id"],
                        "video_token": case["video_token"],
                        "measure": measure,
                        "frame_i": str(frame_i),
                        "frame_j": str(frame_j),
                        "side": side,
                        "section_i": str(section_i),
                        "section_j": "" if section_j < 0 else str(section_j),
                        "section_k": "" if section_k < 0 else str(section_k),
                        "dt_s": str(case.get("dicom_pair_dt_s", "")) if is_physical else f"{frame_dt:.12g}",
                        "dicom_calibration_status": case["dicom_calibration_status"] if is_physical else "NOT_REQUIRED",
                        "scale_x_mm_per_px": case.get("scale_x_mm_per_px", "") if is_physical else "",
                        "scale_y_mm_per_px": case.get("scale_y_mm_per_px", "") if is_physical else "",
                        "image_ref": "",
                    }
                )

            add("F01", "anterior", "a", section_i=mid)
            add("F01", "posterior", "p", section_i=mid)
            add("F07", "both", "cavity", section_i=mid)
            add("F09", "anterior", "a", section_i=mid - 1, section_j=mid)
            add("F09", "posterior", "p", section_i=mid - 1, section_j=mid)
            add("F15", "anterior", "a", section_i=mid - 1, section_j=mid, section_k=mid + 1)
            add("F15", "posterior", "p", section_i=mid - 1, section_j=mid, section_k=mid + 1)
    return validate_tasks(tasks)


def write_annotation_template(tasks: list[dict[str, str]], output: Path) -> None:
    rows = []
    roles = {
        "F01": ("source_inner", "source_outer", "target_inner", "target_outer"),
        "F07": ("source_anterior", "source_posterior", "target_anterior", "target_posterior"),
        "F09": ("source_a", "source_b", "target_a", "target_b"),
        "F15": ("source_a", "source_b", "source_c", "target_a", "target_b", "target_c"),
    }
    for task in tasks:
        for annotator in ("annotator_A", "annotator_B"):
            for role in roles[task["measure"]]:
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "annotator_id": annotator,
                        "point_role": role,
                        "frame_index": task["frame_i"] if role.startswith("source_") else task["frame_j"],
                        "x_px": "",
                        "y_px": "",
                        "visible": "",
                        "measurability": "",
                        "note": "",
                    }
                )
    write_csv_rows(
        output,
        rows,
        ("task_id", "annotator_id", "point_role", "frame_index", "x_px", "y_px", "visible", "measurability", "note"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blinded Paper 1 reference tasks")
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--frame-fractions", default="0.25,0.5,0.75")
    args = parser.parse_args()
    fractions = tuple(float(value.strip()) for value in args.frame_fractions.split(",") if value.strip())
    cases = validate_case_manifest(read_csv_rows(args.case_manifest))
    selected = select_feasibility_cases(cases, args.n_cases)
    tasks = build_tasks(selected, fractions)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.output / "selected_cases.csv", selected, CASE_COLUMNS)
    write_csv_rows(args.output / "annotation_tasks.csv", tasks, TASK_COLUMNS)
    write_annotation_template(tasks, args.output / "annotation_template.csv")
    (args.output / "README.md").write_text(
        "# Paper 1 independent reference-task bundle\n\n"
        f"Selected cases: {len(selected)}; tasks: {len(tasks)}.\n\n"
        "This is a feasibility task list only. Annotators must open the raw image/video, remain blinded to algorithm output, pregnancy/outcome information, and the other annotator, and fill only annotation_template.csv. F15 remains NOT_EVALUABLE unless the DICOM physical calibration fields are independently verified.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
