# Independent reference templates

`case_manifest_template.csv` contains only schema. Fill it from an anonymized formal-cohort inventory without patient names, identifiers, pregnancy labels or outcomes. Use `dicom_calibration_status=VALID` only after independently checking the original DICOM physical region and pair timing; then fill the three DICOM fields.

The task generator creates the annotation template with two blinded annotator IDs and the exact frame/point roles. `algorithm_results_template.csv` is required for formal algorithm-versus-reference validation and must be populated from an independent task-level export, without clinical labels or tracking-coordinate columns. The explicit `--observer-only` command is reserved for feasibility review.
