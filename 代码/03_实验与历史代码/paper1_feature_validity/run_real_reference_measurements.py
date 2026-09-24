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
    summarize_patient_level_algorithm_reference,
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
    parser.add_argument(
        "--observer-only",
        action="store_true",
        help="feasibility-only observer review; formal validation requires algorithm results",
    )
    args = parser.parse_args()

    if args.algorithm_results is None and not args.observer_only:
        raise SystemExit(
            "--algorithm-results is required for formal validation; use --observer-only only for feasibility review"
        )

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
    patient_algorithm_agreement = None
    if args.algorithm_results is not None:
        algorithm_rows = read_csv_rows(args.algorithm_results)
        algorithm_agreement = compare_algorithm_to_reference(tasks, measurements, algorithm_rows)
        patient_algorithm_agreement = summarize_patient_level_algorithm_reference(
            tasks, measurements, algorithm_rows
        )
        write_csv_rows(
            args.output / "algorithm_reference_agreement.csv",
            algorithm_agreement,
            ("measure", "expected_tasks", "evaluable_algorithm_reference_tasks", "coverage", "bias", "mae", "rmse", "loa_lower", "loa_upper"),
        )
        write_csv_rows(
            args.output / "patient_level_algorithm_reference_agreement.csv",
            patient_algorithm_agreement,
            (
                "measure",
                "expected_patients",
                "evaluable_patients",
                "coverage",
                "n",
                "bias",
                "bias_ci_lower",
                "bias_ci_upper",
                "mae",
                "rmse",
                "loa_lower",
                "loa_upper",
            ),
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
        lines.extend(["", "Observer-only feasibility report. Formal validation was not run because no algorithm-result table was supplied."])
    else:
        lines.extend(["", "## Algorithm versus reference", "", "The formal algorithm comparison is required and uses the mean of evaluable observer values per task as the reference. Task-level and patient-level agreement are reported; repeated frames remain within the same patient/case.", ""])
        lines.extend(["| Measure | Tasks | Coverage | Bias | MAE | RMSE | LoA |", "|---|---:|---:|---:|---:|---:|---|"])
        for row in algorithm_agreement:
            loa = f"{row['loa_lower']:.6g} to {row['loa_upper']:.6g}" if row["evaluable_algorithm_reference_tasks"] >= 2 else "not estimable (n<2)"
            lines.append(
                f"| {row['measure']} | {row['evaluable_algorithm_reference_tasks']} | {row['coverage']:.3f} | {row['bias']:.6g} | {row['mae']:.6g} | {row['rmse']:.6g} | {loa} |"
            )
        lines.extend(["", "### Patient-level primary analysis", "", "| Measure | Expected patients | Evaluable patients | Coverage | Bias | 95% CI for bias | MAE | RMSE |", "|---|---:|---:|---:|---:|---|---:|---:|"])
        for row in patient_algorithm_agreement or []:
            ci = f"{row['bias_ci_lower']:.6g} to {row['bias_ci_upper']:.6g}" if row["evaluable_patients"] >= 2 else "not estimable (n<2)"
            lines.append(
                f"| {row['measure']} | {row['expected_patients']} | {row['evaluable_patients']} | {row['coverage']:.3f} | {row['bias']:.6g} | {ci} | {row['mae']:.6g} | {row['rmse']:.6g} |"
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
