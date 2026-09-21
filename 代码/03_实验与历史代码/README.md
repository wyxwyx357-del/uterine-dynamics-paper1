# Historical and reused implementation code

This directory contains code that originated as experimental or historical analysis code.

It must **not** be treated as a disposable archive.

## Active dependencies of the current Paper 1 perturbation analysis

`代码/04_运行入口/run_all_patient_perturbation_stability.py` directly imports:

- `run_new_patient_feature_stability_validation.py`
- `run_literature_aligned_feature_robustness.py`

These two files therefore remain part of the current reproducibility chain. Do not move, rename, or delete them unless the importing entry point and its regression tests are updated and equivalence is demonstrated.

## Historical-only top-level script

- `run_final_sparse_anchor_wall_tracking.py` is retained for provenance/history and is not a current Paper 1 manuscript analysis entry point.

The 5-case/28-case development history must not be counted as an independent validation cohort in the current Paper 1 manuscript.
