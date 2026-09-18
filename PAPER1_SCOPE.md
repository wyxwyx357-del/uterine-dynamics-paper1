# Paper 1 scope

## Core question

Can patient-level image-derived uterine dynamic measurements be reproducibly obtained from short cine transvaginal ultrasound using a predefined quality-control-aware analysis framework?

## Included

- P3 / wall / radial tracking
- P3 correction and anatomical position QC
- artifact and topology-related QC used by the frozen measurement pipeline
- anatomical deformation measurements
- DICOM physical curvature calibration used by F15–F20
- frozen F01–F20 patient-level measurement extraction
- analytical perturbation robustness
- future Paper 1 validation code: clip-duration robustness, QC ablation, initialization reproducibility, cohort-level technical feasibility

## Excluded

- pregnancy-outcome prediction
- AUC / Elastic Net clinical outcome modeling
- frozen deep encoders, ResNet, R3D
- Doppler / multimodal fusion
- peristaltic wave count, propagation direction, wave speed, coordination

Changes that alter the frozen measurement definition must be versioned explicitly and the affected validation results rerun.
