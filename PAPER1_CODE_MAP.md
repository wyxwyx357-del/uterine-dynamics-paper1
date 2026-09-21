# Paper 1 code map

Updated: 2026-09-21

This file is the repository-level map for the code actually relevant to Paper 1. It is an organizational record only. It does **not** change tracking, QC, feature definitions, thresholds, statistical formulas, or any existing result.

## 1. Current formal Paper 1 analysis entry points

The current top-level Paper 1 analysis entry points are under `代码/04_运行入口/`.

| Paper 1 analysis | Current entry point | Current role |
|---|---|---|
| Frozen tracking / P3 / QC production path | `代码/04_运行入口/run_tracking_only_pipeline.py` | Produces the frozen tracking/QC upstream outputs used by downstream measurements. Not rerun by the later sensitivity scripts. |
| 319-case analytical perturbation robustness | `代码/04_运行入口/run_all_patient_perturbation_stability.py` | Runs the historical 42-condition perturbation implementation and summarizes the canonical primary/family ICC/CV results; defaults to all currently discovered cases, with `--expected-cases N` available after a final cohort is frozen. |
| 310-case proportional observation-window truncation | `代码/04_运行入口/run_clip_duration_robustness.py` | Generates frozen F01-F20 on deterministic centered 100/75/50/25% nested windows without rerunning tracking; defaults to all currently available cases, with `--expected-case-count N` available after a final cohort is frozen. |
| Proportional-truncation statistics | `代码/04_运行入口/run_clip_duration_statistics.py` | Computes the predefined pairwise ICC(A,1), 2000-repeat patient bootstrap CI, Bland-Altman, absolute/relative error, Spearman and QC-duration summaries. |
| Grade-3 mask sensitivity | `代码/04_运行入口/run_tracking_quality_sensitivity.py` | Applies the predefined pending Grade-3 whole-frame mask-only sensitivity analysis to frozen outputs. Does not rerun tracking or change QC decisions. |
| Grade-3 sensitivity report/resummary | `代码/04_运行入口/report_quality_sensitivity.py` | Reporting/resummarization layer for the quality-sensitivity outputs, including legacy 319-case outputs. |
| Normalized-time profile preservation | `代码/04_运行入口/run_temporal_stability.py` | Uses the same pending Grade-3 mask-only intervention and fixed normalized-time bins. |
| Normalized cervix-to-fundus spatial profile preservation | `代码/04_运行入口/run_spatial_stability.py` | Uses the same pending Grade-3 mask-only intervention and frozen normalized section coordinate. |

### Proportional-truncation provenance

The clip-duration implementation was recovered from the repository's frozen branch:

- branch: `exp/clip-duration`
- frozen branch head: `c279ca95f29da64da183cc1f54b22f1c9785e803`
- protocol: `CLIP_DURATION_PROTOCOL.md`

The following branch files were initially copied byte-for-byte to `main`:

- `代码/01_底层算法/peristalsis_pipeline/clip_duration_robustness.py`
- `代码/01_底层算法/peristalsis_pipeline/clip_duration_statistics.py`
- `代码/04_运行入口/run_clip_duration_robustness.py`
- `代码/04_运行入口/run_clip_duration_statistics.py`
- the four corresponding unit/entry-point tests;
- `CLIP_DURATION_PROTOCOL.md`.

The initial copy was verified by Git blob SHA. `run_clip_duration_robustness.py` retains an optional `--expected-case-count N` assertion for use after final cohort freezing; it does not impose a default cohort size. The underlying proportional-window module, statistics module, statistical entry point, tests, and frozen protocol remain the recovered source versions. No implementation was reconstructed from result tables or manuscript text.

## 2. Frozen measurement and algorithm dependencies

The principal implementation modules are under `代码/01_底层算法/peristalsis_pipeline/`.

### Measurement definitions and robustness

- `formal_feature_extraction.py`: frozen formal F01-F20 patient-level extraction.
- `feature_stability.py`: shared feature definitions, masking helpers, SRD and related robustness utilities.
- `anatomical_deformation_features.py`: anatomical deformation quantities used by the formal features.
- `dicom_curvature_calibration.py`: physical curvature calibration used by F15-F20.
- `clip_duration_robustness.py`: deterministic proportional-window construction and frozen F01-F20 extraction on each window.
- `clip_duration_statistics.py`: proportional-truncation ICC(A,1), patient bootstrap CI and paired error statistics.
- `temporal_stability.py`: normalized-time profile utilities.
- `spatial_stability.py`: normalized-position spatial profile utilities.
- `formal_qc_contract.py`: formal QC contract used by the frozen measurement pipeline.

### Tracking / P3 / artifact-QC dependencies

- `tracking_lk.py`
- `tracking_mesh.py`
- `tracking_huang_fusion.py`
- `tracking_huang_radial_pairs.py`
- `tracking_huang_wall_fusion.py`
- `radial_pair_geometry.py`
- `p3_image_boundary_correction.py`
- `p3_anatomical_position_qc.py`
- `tracking_artifact_qc_v2.py`

These modules are upstream dependencies. This cleanup does not alter them.

## 3. Migrated formal-flow utilities

`代码/02_已并入主流程/` contains migrated scripts used to establish or verify the frozen pipeline, including P3/radial tracking, artifact QC, anatomical deformation, DICOM curvature calibration, and formal F01-F20 extraction.

These files are **not** the preferred top-level entry points for the current Paper 1 cohort analyses. They remain part of the provenance and regression chain and should not be deleted merely because the current manuscript analyses are launched from `代码/04_运行入口/`.

In particular:

- `代码/02_已并入主流程/05_输出审查与发布/run_feature_v1_formal_candidate_5case.py` is still loaded by the quality-sensitivity code as the official extractor for consistency checks.
- the 28-case/5-case scripts document method development and migration history; they are not independent validation cohorts for the current manuscript.

## 4. Historical implementation dependencies: do not delete

`代码/03_实验与历史代码/` is historical in origin, but it is **not wholly disposable**.

The current 319-case formal perturbation entry point directly imports:

- `run_new_patient_feature_stability_validation.py` for the historical 42-perturbation implementation and F01-F20 perturbation calculation;
- `run_literature_aligned_feature_robustness.py` for ICC/CV/jackknife robustness functions.

Therefore these two files are active dependencies of the current Paper 1 perturbation analysis even though they remain in the historical directory. Moving or deleting them would change/break the current reproducibility chain.

`run_final_sparse_anchor_wall_tracking.py` remains a historical/legacy script and is not a current Paper 1 top-level analysis entry point.

## 5. Tests

Tests are under `测试代码/`.

Current Paper 1-relevant coverage includes:

- formal feature extraction;
- feature stability/robustness;
- proportional clip-duration window construction;
- proportional clip-duration statistics;
- proportional clip-duration generation/statistics entry points;
- anatomical deformation;
- DICOM curvature calibration;
- P3 position/boundary correction;
- artifact QC and tracking;
- Grade-3 quality sensitivity;
- temporal stability;
- spatial stability;
- formal candidate entry-point regression.

Tests associated with historical 28-case robustness are retained because current all-patient perturbation code still reuses historical implementation functions.

## 6. Reproducibility chain for the current manuscript

The intended provenance chain is:

```text
video / labels / DICOM
        |
        v
frozen tracking + P3 + QC
run_tracking_only_pipeline.py
        |
        v
frozen per-patient arrays
RSR + anatomical deformation + physical curvature
        |
        v
formal F01-F20
formal_feature_extraction.py
        |
        +--> 319-case perturbation
        |    run_all_patient_perturbation_stability.py
        |
        +--> 310-case proportional truncation
        |    run_clip_duration_robustness.py
        |          |
        |          +--> pairwise statistics
        |               run_clip_duration_statistics.py
        |
        +--> Grade-3 mask sensitivity
             run_tracking_quality_sensitivity.py
                   |
                   +--> temporal profiles
                   |    run_temporal_stability.py
                   |
                   +--> spatial profiles
                        run_spatial_stability.py
```

## 7. Repository cleanup rule

1. do not delete or move existing Python files solely for naming/visual cleanliness;
2. do not alter formulas, thresholds, masks, feature definitions or statistics during cleanup;
3. do not treat 5-case/28-case historical scripts as independent manuscript validation cohorts;
4. do not add pregnancy-outcome prediction, AUC modeling, deep encoders, Doppler fusion, or propagation analysis to the Paper 1 formal entry-point set;
5. preserve the `exp/clip-duration` branch and its frozen head as provenance for the proportional-truncation implementation.
