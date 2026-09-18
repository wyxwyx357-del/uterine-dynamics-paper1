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

## Frozen validated runtime

The single-case migration regression and final data-free CI were validated with:

- Python 3.14.0
- numpy 2.4.6
- pandas 3.0.3
- scipy 1.17.1
- opencv-python 4.13.0.92
- matplotlib 3.10.9
- pytest 9.1.1

These versions are pinned in the Paper 1 requirements where applicable.

## Validation state

### Static migration checks

- Repository structure and required-module presence: PASS.
- Source-file blob SHA comparison for migrated code/tests: PASS.
- Forbidden Paper 2/prediction content check: PASS.

### CASE_001 migration regression

Using identical inputs, parameters, and runtime environment:

- Tracking stages 01–07: PASS.
  - 335/335 common arrays exact.
  - 335/335 common arrays allclose.
  - Maximum absolute difference: 0.
  - Boolean/QC masks, integer/index arrays, shapes, and NaN patterns identical.
- Anatomical deformation: PASS.
  - Maximum absolute difference: 0.
  - Masks identical.
- DICOM curvature: PASS.
  - 40/40 common arrays exact and allclose.
  - Maximum absolute difference: 0.
  - Flags/masks and NaN patterns identical.
- Formal F01–F20: PASS.
  - 20/20 measurements identical.
  - 27/27 formal function return fields allclose.
  - Maximum absolute difference: 0.
  - QC metadata and NaN patterns identical.

Overall single-case verdict:

`MIGRATION_REGRESSION_PASS`

### Data-free CI

The final CI run on the frozen Python/runtime specification passed. The historical frozen-data integration test that requires local project output CSVs remains outside GitHub CI and is not treated as a code failure.

This migration baseline is eligible to be merged to `main` and tagged as `paper1-baseline-v1.0`.
