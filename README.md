# uterine-dynamics-paper1

Codebase for Paper 1 on analytical robustness of local uterine dynamic measurements from short-cine transvaginal ultrasound.

## Scope

Included:

- frozen P3 / wall / radial tracking and QC;
- anatomical deformation and DICOM physical curvature calibration;
- frozen F01-F20 patient-level measurement extraction;
- all-patient analytical perturbation robustness;
- Grade-3 mask sensitivity;
- normalized-time temporal profile preservation;
- normalized cervix-to-fundus spatial profile preservation.

Excluded:

- pregnancy-outcome prediction / AUC modeling;
- deep encoders;
- Doppler or multimodal fusion;
- peristaltic wave count, propagation direction, speed, or coordination.

## Start here

- `PAPER1_CODE_MAP.md` — current formal entry points, dependencies, historical-code status, tests, and reproducibility chain.
- `PAPER1_SCOPE.md` — scientific scope.
- `MIGRATION_MANIFEST.md` — frozen source repositories and migration validation.
- `代码/04_运行入口/README.md` — executable Paper 1 entry-point map.

## Current reproducibility status

The repository currently contains formal entry points for tracking/P3/QC, 319-case analytical perturbation robustness, Grade-3 quality sensitivity, temporal structure analysis, and spatial structure analysis.

The completed 310-case proportional observation-window truncation analysis is part of the current Paper 1 evidence, but its exact generating/statistics scripts are not yet present on this repository's `main` tree. The original scripts should be migrated and output-equivalence checked; they should not be reconstructed from manuscript text or result tables.

The cleanup documented here changes repository navigation only. It does not change algorithms, thresholds, masks, feature definitions, statistics, or existing results.
