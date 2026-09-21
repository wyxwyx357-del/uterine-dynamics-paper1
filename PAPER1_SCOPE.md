# Paper 1 scope

## Core question

How robust are frozen patient-level local uterine dynamic measurements from short cine transvaginal ultrasound to predefined data perturbation, proportional observation-window truncation, and quality-mask sensitivity?

## Included

- P3 / wall / radial tracking used by the frozen measurement pipeline
- P3 correction and anatomical position QC
- artifact and topology-related QC used by the frozen measurement pipeline
- anatomical deformation measurements
- DICOM physical curvature calibration used by F15–F20
- frozen F01–F20 patient-level measurement extraction
- 319-case analytical perturbation robustness
- 310-case proportional observation-window truncation (100/75/50/25%)
- pending Grade-3 mask-only sensitivity analysis
- normalized-time temporal-structure preservation under the same predefined mask-only sensitivity analysis
- normalized cervix-to-fundus spatial-structure preservation under the same predefined mask-only sensitivity analysis

## Excluded

- pregnancy-outcome prediction
- AUC / Elastic Net clinical outcome modeling
- frozen deep encoders, ResNet, R3D
- Doppler / multimodal fusion
- peristaltic wave count, propagation direction, wave speed, coordination
- independent acquisition test-retest reproducibility
- inter-operator acquisition reproducibility
- new initialization-reproducibility experiments

Changes that alter the frozen measurement definition must be versioned explicitly and the affected validation results rerun.
