from __future__ import annotations

"""Compute independent manual references and blinded agreement summaries."""

import argparse
from pathlib import Path
import sys

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR.parent))
from paper1_feature_validity.real_reference_measurements import (
    compare_algorithm_to_reference,
    compute_all_reference_measurements,
    read_csv_rows,
    summarize_observer_agreement,
    validate_annotations,
    validate_tasks,
    write_csv_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Paper 1 independent reference measurements")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm-results", type=Path)
    args = parser.parse_args()

    tasks = validate_tasks(read_csv_rows(args.tasks))
    annotations = validate_annotations(read_csv_rows(args.annotations), tasks)
    measurements = compute_all_reference_measurements(tasks, annotations)
    agreement = summarize_observer_agreement(tasks, measurements)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        args.output / "reference_measurements.csv",
        measurements,
        ("task_id", "case_id", "measure", "side", "annotator_id", "status", "value", "unit", "reason", "semantic_note"),
    )
    write_csv_rows(
        args.output / "observer_agreement.csv",
        agreement,
        ("measure", "expected_tasks", "evaluable_two_observer_tasks", "coverage", "n", "bias", "mae", "rmse", "loa_lower", "loa_upper"),
    )

    algorithm_agreement = None
    if args.algorithm_results is not None:
        algorithm_rows = read_csv_rows(args.algorithm_results)
        algorithm_agreement = compare_algorithm_to_reference(tasks, measurements, algorithm_rows)
        write_csv_rows(
            args.output / "algorithm_reference_agreement.csv",
            algorithm_agreement,
            ("measure", "expected_tasks", "evaluable_algorithm_reference_tasks", "coverage", "bias", "mae", "rmse", "loa_lower", "loa_upper"),
        )

    lines = [
        "# Independent Paper 1 reference-measurement report",
        "",
        "This report is a measurement-agreement report, not a clinical association or pregnancy-outcome analysis.",
        "",
        "## Observer agreement",
        "",
        "| Measure | Expected tasks | Two-observer tasks | Coverage | Bias | MAE | RMSE | LoA |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in agreement:
        loa = f"{row['loa_lower']:.6g} to {row['loa_upper']:.6g}" if row["n"] >= 2 else "not estimable (n<2)"
        lines.append(
            f"| {row['measure']} | {row['expected_tasks']} | {row['evaluable_two_observer_tasks']} | {row['coverage']:.3f} | {row['bias']:.6g} | {row['mae']:.6g} | {row['rmse']:.6g} | {loa} |"
        )
    if algorithm_agreement is None:
        lines.extend(["", "Algorithm-vs-reference comparison: not run; provide an independently exported algorithm-result table when ready."])
    else:
        lines.extend(["", "## Algorithm versus reference", "", "The algorithm comparison uses the mean of evaluable observer values per task as the reference and reports only agreement metrics.", ""])
        lines.extend(["| Measure | Tasks | Coverage | Bias | MAE | RMSE | LoA |", "|---|---:|---:|---:|---:|---:|---|"])
        for row in algorithm_agreement:
            loa = f"{row['loa_lower']:.6g} to {row['loa_upper']:.6g}" if row["evaluable_algorithm_reference_tasks"] >= 2 else "not estimable (n<2)"
            lines.append(
                f"| {row['measure']} | {row['evaluable_algorithm_reference_tasks']} | {row['coverage']:.3f} | {row['bias']:.6g} | {row['mae']:.6g} | {row['rmse']:.6g} | {loa} |"
            )
    lines.extend(
        [
            "",
            "F01 values are paired geometric references and do not prove tissue identity or material strain. F15 is not evaluated when independently verified DICOM physical calibration is missing or invalid; no pixel-to-millimetre ratio is imputed.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
