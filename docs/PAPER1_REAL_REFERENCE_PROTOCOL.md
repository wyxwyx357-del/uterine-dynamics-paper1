# Paper 1 independent reference-measurement protocol

## Scope

This protocol evaluates measurement agreement for F01, F07, F09 and F15. It does not use pregnancy labels, clinical outcomes, AUC, or correlation as a substitute for measurement correctness. The first 10 cases are a feasibility exercise for task clarity, observer agreement and data handling; they are not a formal accuracy sample size.

## Case selection

`run_real_reference_tasks.py` selects anonymized cases by deterministic round-robin coverage of `quality_tier × motion_coverage_tier`. The input manifest must contain frame count, FPS, section count and DICOM calibration status, but must not contain patient identity or outcome columns. Multiple frames from one video remain repeated measurements within one patient/video, never independent patient samples.

## Blinding and annotation

Two annotators independently open the raw ultrasound image/video. They must not see frozen algorithm trajectories, algorithm-derived coordinates, pregnancy/outcome information, or the other annotator's marks. Every point is recorded in pixel coordinates with frame number, visibility and measurability. An uncertain or non-measurable point is retained as a reason for exclusion, not silently replaced.

The task generator creates:

- F01: anterior and posterior radial inner/outer point pairs at the source and adjacent target frames;
- F07: anterior and posterior wall points defining cavity width at source and target frames;
- F09: adjacent along-wall point pairs on each wall at source and target frames;
- F15: three consecutive points on each wall at source and target frames.

The exact section indices and frame pair are written in `annotation_tasks.csv`; annotators do not choose them after seeing algorithm output.

## Independent calculations

The reference module calculates signed pair rates from the marked points. F01 uses `(target_length - source_length) / source_length / dt` and is labelled a paired geometric reference, not proof of material strain or cross-frame tissue identity. F07 and F09 use the analogous width/segment-length rate. F15 uses the unsigned three-point circumcircle curvature after independently verified anisotropic DICOM pixel-to-mm conversion, then `(target_curvature - source_curvature) / dt` in mm^-1 s^-1.

F15 is `NOT_EVALUABLE` if calibration status is missing, invalid, unchecked, or lacks positive x/y scale and pair timing. No pixel-to-mm ratio is guessed. The pixel-space curvature path is not reported as a physical F15 substitute.

## Agreement report

Observer agreement reports evaluable two-observer task coverage, bias, MAE, RMSE and Bland–Altman limits of agreement when at least two paired tasks exist. For formal validation, an independently exported algorithm-result table is mandatory: the primary analysis aggregates matched task measurements within each case and reports patient-level coverage, bias, 95% bias confidence intervals, MAE, RMSE and Bland–Altman limits. A task-level diagnostic table is also retained. The `--observer-only` flag is restricted to feasibility review and must not be reported as algorithm accuracy. Correlation, if later added, is auxiliary only and must not replace agreement statistics.

## Commands

Create the feasibility bundle from a completed anonymized manifest:

```text
python 代码/03_实验与历史代码/paper1_feature_validity/run_real_reference_tasks.py --case-manifest <case_manifest.csv> --output <tasks_dir> --n-cases 10
```

After two annotators independently fill the generated annotation template:

```text
python 代码/03_实验与历史代码/paper1_feature_validity/run_real_reference_measurements.py --tasks <tasks_dir>/annotation_tasks.csv --annotations <filled_annotations.csv> --algorithm-results <algorithm_results.csv> --output <reference_report_dir>
```

The observer-only feasibility path is explicit:

```text
python 代码/03_实验与历史代码/paper1_feature_validity/run_real_reference_measurements.py --tasks <tasks_dir>/annotation_tasks.csv --annotations <filled_annotations.csv> --observer-only --output <feasibility_report_dir>
```

The algorithm table must contain only task-level algorithm values and status; it must not contain clinical labels. Repeated frames from one case are aggregated within that case and are never counted as independent patients.
