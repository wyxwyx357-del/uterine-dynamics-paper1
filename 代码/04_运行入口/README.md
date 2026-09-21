# Paper 1 formal run entry points

This directory contains the current top-level execution scripts used for Paper 1 analyses. File placement here indicates a **run entry point**, not that every script reruns the full upstream pipeline.

## Current entry points

- `run_tracking_only_pipeline.py`  
  Frozen tracking/P3/QC production path.

- `run_all_patient_perturbation_stability.py`  
  All-patient analytical perturbation robustness. It intentionally reuses the historical 42-perturbation and robustness implementations from `代码/03_实验与历史代码/`, defaults to all currently discovered cases, and supports `--expected-cases N` after final cohort freezing.

- `run_clip_duration_robustness.py`  
  Completed 310-case proportional observation-window analysis using deterministic centered nested 100/75/50/25% windows and the frozen F01-F20 extractor; defaults to all currently available cases and supports `--expected-case-count N` after final cohort freezing.

- `run_clip_duration_statistics.py`  
  Statistical summary for the proportional-window analysis: ICC(A,1), patient-cluster percentile bootstrap CI, Bland-Altman, absolute/relative error, secondary Spearman, and duration/QC summaries.

- `run_tracking_quality_sensitivity.py`  
  Pending Grade-3 mask-only quality-sensitivity analysis on frozen outputs.

- `report_quality_sensitivity.py`  
  Reporting/resummarization of quality-sensitivity outputs.

- `run_temporal_stability.py`  
  Normalized-time 5-bin structure analysis under the same Grade-3 mask-only intervention.

- `run_spatial_stability.py`  
  Normalized cervix-to-fundus 5-bin spatial structure analysis under the same Grade-3 mask-only intervention.

## Clip-duration source provenance

The proportional-window code and its tests were recovered from the frozen `exp/clip-duration` branch at commit `c279ca95f29da64da183cc1f54b22f1c9785e803` and the initial copy was verified by Git blob SHA. The entry point accepts an optional `--expected-case-count N` guard after final cohort freezing; the underlying proportional-window and statistics implementations are unchanged.

The perturbation entry point likewise accepts an optional `--expected-cases N` guard after final cohort freezing. Analysis rules are frozen, but cohort size is not yet frozen; both entry points default to the current input cohort.

See `CLIP_DURATION_PROTOCOL.md` for the frozen protocol and `PAPER1_CODE_MAP.md` for the full dependency map.
