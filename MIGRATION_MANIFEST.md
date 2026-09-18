# Paper 1 migration manifest

## Frozen sources

### Tracking / P3 / QC

- Repository: `wyxwyx357-del/uterine-tracking`
- Commit: `3b74fdab2650c62e272b7e1bedf701cf48efc3ac`
- Rationale: independent tracking-only repository whose merge commit records single-case migration equivalence and 45/45 migrated tests passing in that source repository.

### Measurement / F01–F20 / analytical robustness

- Repository: `wyxwyx357-del/end`
- Commit: `a9cf11b95293e395e0a90a1e3340393d782a2d5d`

## Migration rule

Tracking/P3/QC code is copied from the frozen `uterine-tracking` source. Measurement extraction, anatomical deformation, DICOM curvature calibration, F01–F20 stability/robustness code, and their directly relevant tests are copied from the frozen `end` source.

No pregnancy-outcome modeling, clinical+AUC analysis, deep-learning encoder code, Doppler fusion, or propagation-direction analysis is included.

## Dependency note

The source repositories did not pin `pandas` or `scipy` in their tracked requirements even though the migrated Paper 1 robustness scripts import them. They are declared in the Paper 1 requirements without fabricated version pins. Their exact versions must be recorded from the validated runtime and pinned before the Paper 1 baseline is tagged.

## Current validation state

- Repository structure and required-module presence: checked.
- Forbidden Paper 2/prediction filenames in the migrated tree: none detected.
- Source tracking repository validation: recorded in source commit `3b74fdab...`.
- Target-repository pytest/regression execution: pending. Do not label this Paper 1 baseline as regression-validated until the migrated tests are executed in the target environment.
