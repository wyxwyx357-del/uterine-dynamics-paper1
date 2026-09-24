# Paper 1 feature-validity status

## Evidence levels

1. Code-level regression: passed for the frozen feature, deformation, DICOM-calibration, geometry, stability and formal-entrypoint tests.
2. Data-free synthetic validation: passed for all 10 requested geometry scenarios and the 20-feature contract. The independent oracle checks RSR, cavity-width rate, along-wall segment rate and calibrated physical curvature rate, including static/rigid-motion/scale/local-change/masking cases.
3. Real-video independent reference validation: not started. No raw video, DICOM, annotation or trajectory files are present in this checkout, so no patient task list or patient result has been created. The formal path now requires an independent algorithm-result table and produces patient-level primary agreement summaries; observer-only output is explicitly feasibility-only.

## Current test result

```text
68 passed, 11 subtests passed
```

The synthetic report was generated outside the repository with:

```text
python 代码/03_实验与历史代码/paper1_feature_validity/run_synthetic_feature_validation.py --output <data-free-output-dir>
```

The report showed all 10 scenarios as `PASS` and 20 formal feature columns present. This is geometry/aggregation evidence only; it does not validate LK recovery from ultrasound texture, clinical validity, or tissue identity across frames.

## Minimum data needed for the next stage

Provide an anonymized case manifest with the schema in `case_manifest_template.csv`, plus access-controlled raw cine frames/videos and the original DICOM metadata needed to verify FPS and physical pixel calibration. The manifest must include frame count, FPS, section count, quality tier and motion-coverage tier. Do not provide pregnancy labels, outcomes, patient names or direct identifiers.

After the 10-case feasibility task is complete, provide two separately blinded annotation tables using the generated template. For F15, independently verified DICOM x/y scale and pair timing are mandatory; otherwise the result remains `NOT_EVALUABLE`.

Formal accuracy sample size and patient-level confidence intervals remain to be specified after the feasibility error distribution is observed. Multiple frames from one video will not be treated as independent patients.
