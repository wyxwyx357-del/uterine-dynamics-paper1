# uterine-dynamics-paper1

Codebase for Paper 1 on analytical robustness of local uterine dynamic measurements from short-cine transvaginal ultrasound.

## Scope

Included:

- frozen P3 / wall / radial tracking and QC;
- anatomical deformation and DICOM physical curvature calibration;
- frozen F01-F20 patient-level measurement extraction;
- all-patient analytical perturbation robustness;
- 310-case proportional observation-window truncation;
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
- `CLIP_DURATION_PROTOCOL.md` — frozen proportional observation-window truncation protocol.
- `代码/04_运行入口/README.md` — executable Paper 1 entry-point map.

## Current reproducibility status

The repository contains formal entry points for tracking/P3/QC, 319-case analytical perturbation robustness, the completed 310-case proportional observation-window truncation analysis, Grade-3 quality sensitivity, temporal structure analysis, and spatial structure analysis.

The proportional-truncation implementation was recovered from the repository's frozen `exp/clip-duration` branch at `c279ca95f29da64da183cc1f54b22f1c9785e803`. The initial migration was verified against the frozen branch by Git blob SHA. After that migration, `main` adds only a default frozen-cohort guard (`--expected-case-count 310`) to `run_clip_duration_robustness.py`; the proportional-window construction, frozen F01-F20 extraction, and duration-statistics implementations are unchanged.

The 319-case perturbation entry point likewise defaults to a frozen `--expected-cases 319` guard unless an explicit `--cases` subset is supplied. These guards prevent silent cohort drift and do not change algorithms, thresholds, masks, feature definitions, statistics, or existing results.
