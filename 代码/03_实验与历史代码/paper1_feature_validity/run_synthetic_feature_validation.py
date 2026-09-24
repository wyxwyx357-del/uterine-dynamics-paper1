from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR.parent))
sys.path.insert(0, str(PACKAGE_DIR.parent.parent / "01_底层算法"))

from paper1_feature_validity.synthetic_measurement_validation import (
    run_all_synthetic_cases,
    write_json_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run data-free independent synthetic F01-F20 validation."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output directory must be empty or absent")
    args.output.mkdir(parents=True, exist_ok=True)
    payload = run_all_synthetic_cases()
    write_json_report(payload, args.output / "synthetic_validation.json")
    with (args.output / "scenario_status.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["scenario", "status", "failed_checks"]
        )
        writer.writeheader()
        for row in payload["results"]:
            failed = [name for name, passed in row["checks"].items() if not passed]
            writer.writerow(
                {
                    "scenario": row["scenario"],
                    "status": row["status"],
                    "failed_checks": "|".join(failed),
                }
            )
    report = [
        "# Synthetic F01–F20 validation report",
        "",
        f"Overall status: **{payload['summary_status']}**",
        "",
        f"Scenarios: {payload['scenario_count']}; formal features checked: {payload['feature_count']}.",
        "",
        "The expected values came from independent geometry/oracle functions. The frozen implementation was used only for measured outputs.",
        "",
        "This is not an ultrasound-texture/LK validation, a clinical validity study, or proof of tissue identity across frames.",
        "",
        "## Scenario results",
        "",
        "| Scenario | Status | Failed checks |",
        "|---|---|---|",
    ]
    for row in payload["results"]:
        failed = [name for name, passed in row["checks"].items() if not passed]
        report.append(
            f"| {row['scenario']} | {row['status']} | {', '.join(failed) or 'none'} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
