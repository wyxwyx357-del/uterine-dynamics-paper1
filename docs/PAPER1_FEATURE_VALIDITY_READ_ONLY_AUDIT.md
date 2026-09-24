# Paper 1 F01–F20 read-only measurement audit

## Audit boundary

The initial read-only audit was performed from the local Paper 1 snapshot at
commit `96e4f21bbc5ade611b9cd6e82d50a7544101d3ea` on
`migration/paper1-baseline`, before the GitHub connection recovered. After
reconnection, `exp/paper1-feature-validity` was fast-forwarded to
`ec9091dd014b2150f640b5f494788c111a5e4e9d`; remote `main` and the remote
experiment branch currently resolve to the same commit, so their verified
common start is `ec9091dd014b2150f640b5f494788c111a5e4e9d`. Frozen source
files were not modified by this audit.

The audit read the frozen feature extractor, anatomical deformation module,
DICOM calibration module, radial-pair geometry/tracking modules, formal
entrypoint, and their tests. It did not read pregnancy labels or patient-level
outcome files.

## Frozen feature contract

The 20 biological outputs are absolute-value summaries of four measurement
families:

| Family | Features | Frozen output unit | Domain |
|---|---|---|---|
| RSR | F01–F06 | `s^-1` | global, anterior, posterior; median/P95 |
| cavity width strain rate | F07–F08 | `s^-1` | global; median/P95 |
| longitudinal wall strain rate | F09–F14 | `s^-1` | global, anterior, posterior; median/P95 |
| physical wall curvature change rate | F15–F20 | `mm^-1 s^-1` | global, anterior, posterior; median/P95 |

The final aggregation uses finite absolute values and computes the median and
95th percentile. It does not impute missing values or create a new threshold.
The formal extractor requires the RSR validity mask to equal the finite RSR
positions, requires RSR and anatomical FPS to match, and requires DICOM
curvature masks to match the anatomical pair mask when DICOM data are present.

## Actual formulas

### RSR

For each radial pair, with source inner/outer points `a_s, b_s` and measured
inner/outer points `a_t, b_t`:

```text
L_s = ||b_s - a_s||
L_t = ||b_t - a_t||
RSR = (L_t - L_s) / L_s * fps
```

The radial pair order is anterior inner/outer followed by posterior inner/outer.
Candidate outer points are offset from each wall along an oriented outward
normal. Pair validity is quality/topology gated, but the tracking module
explicitly does not prove long-term tissue identity.

### Cavity width strain rate (F07–F08)

The two same-section wall-center points are the midpoint of each wall's radial
pair. With cavity width `W = ||posterior_center - anterior_center||`:

```text
cavity_width_strain_rate = (W_t - W_s) / W_s * fps
```

The formal feature is the absolute finite value; the signed closing rate is a
separate diagnostic output and is not one of F07–F08.

### Along-wall longitudinal strain rate (F09–F14)

For adjacent longitudinal wall-center sections `i, i+1`:

```text
S_s = ||center_s[i+1] - center_s[i]||
S_t = ||center_t[i+1] - center_t[i]||
longitudinal_strain_rate = (S_t - S_s) / S_s * fps
```

The required validity is the conjunction of both neighboring radial-pair
validity positions on the same wall.

### Curvature change rate (F15–F20)

The frozen local curvature is an unsigned three-point circumcircle curvature:

```text
kappa = 2 * |cross(b-a, c-a)| / (||b-a|| * ||c-b|| * ||c-a||)
```

The anatomical module can produce this in pixel units. F15–F20 use the DICOM
calibration module, which maps analysis pixels into a spatial-distance
ultrasound region, computes the same three-point curvature in millimetres, and
divides the change by the DICOM pair interval:

```text
physical_curvature_change_rate = (kappa_t_mm - kappa_s_mm) / pair_dt_s
```

The current calibrated path also checks that valid DICOM pair intervals match
`1 / analysis_fps` within its declared timing tolerance when timing is marked
matched. This is a lineage/safety check, not a replacement for pair-specific
timing. Missing or non-distance ultrasound regions, missing pair timing, or a
failed timing match leave the physical rate unavailable.

## Input and QC lineage

The formal entrypoint consumes frozen RSR, anatomical deformation, and optional
DICOM curvature NPZ outputs. The core array contracts are:

- RSR: `(pair_frame, side=2, section)` in `s^-1`;
- cavity width: `(pair_frame, section)` in `s^-1`;
- longitudinal wall strain: `(pair_frame, side=2, section-1)` in `s^-1`;
- DICOM curvature rate: `(pair_frame, side=2, section)` in `mm^-1 s^-1`;
- pair validity: the matching boolean shape, with NaN placement preserved.

The radial tracking implementation uses LK status, curve support, and pair
orientation/topology. The pair-validity semantics are deliberately weaker than
a proof of material-point identity. The feature extractor preserves this
boundary and does not create a propagation, direction, frequency, or tissue
identity rule.

## Audit findings and differences

1. The planned Paper 1 F01–F20 definitions are present in the frozen extractor
   and match the four families above.
2. The formal extractor aggregates already-computed arrays; it does not itself
   recompute tracking, RSR, cavity width, longitudinal strain, or curvature.
3. Pixel curvature from the anatomical module and physical curvature used by
   F15–F20 are different measurement units and must not be silently compared.
4. The formal extractor uses family-specific existing validity masks and does
   not define a minimum valid fraction or a biological accuracy threshold.
5. The existing tests check shapes, mask lineage, timing lineage, and selected
   mathematical cases, but did not provide one independent synthetic report
   covering all F01–F20 and the scenario families requested here.
6. The new validation directory therefore treats synthetic results as
   geometry/aggregation evidence only. It does not claim validation of LK
   tracking from ultrasound texture, clinical validity, or tissue identity.
