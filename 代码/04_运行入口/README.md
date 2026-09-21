# Paper 1 formal run entry points

This directory contains the current top-level execution scripts used for Paper 1 analyses. File placement here indicates a **run entry point**, not that every script reruns the full upstream pipeline.

## Current entry points

- `run_tracking_only_pipeline.py`  
  Frozen tracking/P3/QC production path.

- `run_all_patient_perturbation_stability.py`  
  All-patient analytical perturbation robustness. It intentionally reuses the historical 42-perturbation and robustness implementations from `代码/03_实验与历史代码/`.

- `run_tracking_quality_sensitivity.py`  
  Pending Grade-3 mask-only quality-sensitivity analysis on frozen outputs.

- `report_quality_sensitivity.py`  
  Reporting/resummarization of quality-sensitivity outputs.

- `run_temporal_stability.py`  
  Normalized-time 5-bin structure analysis under the same Grade-3 mask-only intervention.

- `run_spatial_stability.py`  
  Normalized cervix-to-fundus 5-bin spatial structure analysis under the same Grade-3 mask-only intervention.

## Missing formal entry point

The completed 310-case proportional observation-window truncation experiment (100/75/50/25%, centered nested windows) is part of the current manuscript evidence but its exact generating/statistics scripts have not yet been migrated into this repository.

Do not implement a replacement from the manuscript description alone. Migrate the original scripts, preserve their recorded hashes/manifests, and verify output equivalence first.

See `PAPER1_CODE_MAP.md` at repository root for the full dependency map.
