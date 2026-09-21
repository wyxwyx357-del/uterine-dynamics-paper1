# Paper 1 clip-duration robustness protocol

## Purpose

Assess whether the frozen patient-level image-derived dynamic measurements
F01-F20 remain stable when the available formal cine-analysis time axis is
progressively shortened within the same patient.

This experiment does not use pregnancy outcome information and does not alter
tracking, P3 correction, QC, DICOM calibration, or F01-F20 definitions.

## Reference time axis

The 100% reference is the complete formal pair-frame sequence available to the
frozen F01-F20 extractor for each case.

This is the formal analysis time axis, not a claim that it is exactly identical
to the raw MP4 container duration. The analysis table therefore reports
pair-frame duration explicitly.

## Duration conditions

Primary conditions:

- 100%
- 75%
- 50%
- 25%

For every case:

1. 100% uses the complete formal pair-frame sequence.
2. 75%, 50%, and 25% use deterministic center-aligned nested windows.
3. Integer window length is calculated by floor division, so the realized
   fraction never exceeds the requested percentage.
4. Actual pair-frame counts, realized fractions, start/stop indices, and
   duration in seconds are recorded.

No segment is selected because it looks more stable, has stronger motion,
produces better QC, has a favorable feature value, or has a favorable clinical
outcome.

No frame repetition, interpolation, temporal stitching, or replacement of a
predefined window with another window is allowed.

## Motion artifact and jitter

Existing frozen QC/validity masks remain active inside every proportional
window.

Transient probe motion is therefore represented through the existing validity
domains and through the window-specific:

- RSR valid ratio
- cavity valid ratio
- longitudinal valid ratio
- curvature valid ratio

No new motion-artifact exclusion threshold is introduced by this duration
experiment.

A proportional window that leaves a formal measurement unevaluable is retained
as missing for that measurement. It is not replaced by a different time
segment.

This separation is deliberate: duration shortening and the existing frozen QC
must not be confounded by post-hoc selection of visually favorable clips.

## Frozen feature extraction

Every proportional window is passed to the existing frozen function:

peristalsis_pipeline.formal_feature_extraction.extract_formal_candidate_case_features

The F01-F20 formulas and aggregation rules are unchanged.

## Primary statistical comparisons

Each shortened condition is compared with the same patient's 100% reference:

- 75% vs 100%
- 50% vs 100%
- 25% vs 100%

For every F01-F20 measurement, report:

- evaluable paired patient count
- ICC(A,1), two-way absolute-agreement single-measure
- patient-cluster bootstrap 95% confidence interval, 2,000 repetitions
- Bland-Altman mean difference and 95% limits of agreement
- median and 95th percentile absolute difference
- median and 95th percentile absolute relative error where the 100% reference
  is non-zero
- Spearman rho as a secondary rank-agreement description

No ICC threshold, error threshold, or feature-selection rule is introduced by
this experiment.

## Required descriptive reporting

For each proportional condition report:

- number of cases
- actual pair-frame duration distribution
- number with all F01-F20 evaluable
- RSR valid-ratio distribution
- cavity valid-ratio distribution
- longitudinal valid-ratio distribution
- curvature valid-ratio distribution

These fields are retained so that apparent duration sensitivity can be
interpreted alongside variation in usable/QC-valid information.

## Outputs

The batch entrypoint produces:

- clip_duration_proportional_f01_f20_long.csv
- clip_duration_case_inventory.csv
- clip_duration_manifest.json

The statistical entrypoint produces:

- clip_duration_pairwise_summary.csv
- clip_duration_case_errors.csv
- clip_duration_qc_summary.csv
- clip_duration_statistics_manifest.json

## Scope boundary

This experiment is duration-dependent measurement robustness. It is not:

- pregnancy-outcome prediction
- proof of endometrial peristaltic wave frequency/direction
- acquisition test-retest reproducibility
- inter-operator acquisition reproducibility
- external validation
