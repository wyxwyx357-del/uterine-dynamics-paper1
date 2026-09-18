#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate method-level mask-perturbation stability of 20 frozen features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline.feature_stability import (  # noqa: E402
    FEATURE_SPECS,
    StabilityCaseData,
    apply_pair_exclusion,
    compute_feature_row,
    domain_values_and_valid,
    fixed_contiguous_pair_exclusion,
    random_pair_exclusion,
    realized_mask_identity,
    remove_top_fraction_mask,
    sensitivity_warning,
    spatial_quintile_pair_exclusion,
    symmetric_relative_difference_pct,
    temporal_quintile_pair_exclusion,
    top_fraction_feature_value,
    valid_count_and_ratio,
)
from peristalsis_pipeline.formal_feature_extraction import (  # noqa: E402
    BIOLOGICAL_FEATURE_COLUMNS,
    FORMAL_CANDIDATE_FEATURE_COLUMNS,
    MODEL_INPUT_FEATURE_COLUMNS,
    QC_METADATA_COLUMNS,
)
from peristalsis_pipeline.dicom_curvature_calibration import (  # noqa: E402
    validate_dicom_curvature_rate_lineage,
)
from peristalsis_pipeline.formal_qc_contract import (  # noqa: E402
    FORMAL_QC_VERSION,
    formal_analysis_eligibility,
    validate_case_id,
    validate_formal_qc_lineage,
    validate_matching_formal_qc_lineage,
)


FEATURE_CONTRACT_VERSION = "formal_candidate_feature_v1_20_biological"
STABILITY_TEST_VERSION = "method_level_mask_perturbation_v1_3_analysis_eligibility"
RANDOM_SEEDS = (1729, 2718, 31415, 57721, 65537)
LENGTH_STRATA = (
    ("short_lt_30s", 0.0, 30.0),
    ("medium_30_to_lt_60s", 30.0, 60.0),
    ("long_ge_60s", 60.0, math.inf),
)
OUTPUT_NAMES = (
    "正式候选特征方法级稳定性验证报告.md",
    "正式候选特征稳定性_feature_summary.csv",
    "正式候选特征稳定性_long.csv",
    "stability_validation_manifest.json",
    "nominal_to_realized_spatial_mapping.csv",
    "v1_duplicate_group_rescore.csv",
    "v1_1_cross_family_realized_mask_rescore.csv",
    "diagnostic_stability_long.csv",
    "diagnostic_stability_feature_summary.csv",
    "analysis_eligibility_audit.csv",
)
PERTURBATION_GROUPS = (
    "random_delete_5pct",
    "random_delete_10pct",
    "time_block_target_5pct",
    "space_block_target_5pct",
    "time_block_target_10pct",
    "space_block_target_10pct",
    "temporal_jackknife",
    "spatial_jackknife",
    "top1_removal",
    "shadow_sensitivity",
)
CLASSIFICATION_RULE = {
    "median_low_srd_pct": 10.0,
    "median_high_srd_pct": 20.0,
    "p95_low_srd_pct": 15.0,
    "p95_high_srd_pct": 30.0,
    "minimum_evaluable_cases": 3,
    "unstable_if_finite_to_nan_cases_at_least": 2,
    "unstable_if_high_sensitivity_cases_at_least": 2,
    "unstable_if_high_sensitivity_perturbation_groups_at_least": 2,
    "method_pass_requires_unstable_features": 0,
    "method_pass_requires_insufficient_features": 0,
    "method_pass_minimum_stable_feature_fraction": 0.50,
    "semantics": "项目预设工程稳定性规则，不是子宫内膜临床文献公认阈值",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def scalar_bool(value: Any) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected one scalar boolean")
    return bool(array.reshape(-1)[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def git_provenance() -> dict[str, object]:
    return {
        "branch": git_value("branch", "--show-current"),
        "commit": git_value("rev-parse", "HEAD"),
        "status_porcelain": git_value("status", "--porcelain"),
    }


def case_ids_for_suffix(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.name[: -len(suffix)]
        for path in directory.glob(f"*{suffix}")
        if path.name.endswith(suffix)
    }


def load_artifact_metadata(path: Path, frame_count: int) -> dict[str, object]:
    if not path.is_file():
        return {
            "artifact_metadata_available": False,
            "artifact_grade2_percent": math.nan,
            "automatic_grade3_candidate_frames": math.nan,
            "automatic_grade3_candidate_fraction": math.nan,
            "artifact_risk_stratum": "artifact_metadata_unavailable",
        }
    table = pd.read_csv(path)
    if len(table) != 1:
        raise ValueError(f"{path}: artifact summary must contain one row")
    row = table.iloc[0]
    grade2 = float(row["2级百分比"])
    grade3_frames = int(row["自动3级候选帧数"])
    fraction = grade3_frames / frame_count if frame_count else math.nan
    if grade3_frames == 0:
        stratum = "no_pending_auto_grade3"
    elif fraction <= 0.05:
        stratum = "pending_auto_grade3_le_5pct"
    else:
        stratum = "pending_auto_grade3_gt_5pct"
    return {
        "artifact_metadata_available": True,
        "artifact_grade2_percent": grade2,
        "automatic_grade3_candidate_frames": grade3_frames,
        "automatic_grade3_candidate_fraction": fraction,
        "artifact_risk_stratum": stratum,
    }


def length_stratum(duration_s: float) -> str:
    for name, lower, upper in LENGTH_STRATA:
        if lower <= duration_s < upper:
            return name
    raise RuntimeError("video duration did not match a frozen length stratum")


def load_case(
    case_id: str,
    *,
    rsr_dir: Path,
    anatomical_dir: Path,
    shadow_dir: Path,
    dicom_dir: Path | None,
    artifact_dir: Path,
) -> tuple[StabilityCaseData, np.ndarray, np.ndarray, dict[str, object], list[Path]]:
    rsr_path = rsr_dir / f"{case_id}_rsr_fourway_v1.npz"
    anatomical_path = anatomical_dir / f"{case_id}_anatomical_deformation_v1.npz"
    shadow_path = shadow_dir / f"{case_id}_pair_quality_shadow_qc_v1.npz"
    for path in (rsr_path, anatomical_path, shadow_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is missing: {path}")

    rsr = load_npz(rsr_path)
    anatomical = load_npz(anatomical_path)
    shadow = load_npz(shadow_path)
    validate_formal_qc_lineage(rsr, source=rsr_path)
    validate_formal_qc_lineage(anatomical, source=anatomical_path)
    validate_case_id(rsr, expected_case_id=case_id, source=rsr_path)
    validate_case_id(anatomical, expected_case_id=case_id, source=anatomical_path)
    validate_matching_formal_qc_lineage(
        rsr,
        anatomical,
        left_source=rsr_path,
        right_source=anatomical_path,
    )

    pair_valid = np.asarray(anatomical["pair_qc_valid"], dtype=bool)
    curvature_rate: np.ndarray | None = None
    physical_curvature_available = False
    physical_curvature_rate_available = False
    input_paths = [rsr_path, anatomical_path, shadow_path]
    dicom_path = (
        None if dicom_dir is None else dicom_dir / f"{case_id}_dicom_curvature_v1.npz"
    )
    if dicom_path is not None and dicom_path.is_file():
        dicom = load_npz(dicom_path)
        validate_dicom_curvature_rate_lineage(dicom, source=dicom_path)
        validate_formal_qc_lineage(dicom, source=dicom_path)
        validate_case_id(dicom, expected_case_id=case_id, source=dicom_path)
        validate_matching_formal_qc_lineage(
            anatomical,
            dicom,
            left_source=anatomical_path,
            right_source=dicom_path,
        )
        physical_curvature_available = scalar_bool(
            dicom["physical_curvature_available"]
        )
        physical_curvature_rate_available = scalar_bool(
            dicom["physical_curvature_rate_available"]
        )
        if physical_curvature_rate_available:
            curvature_rate = np.asarray(
                dicom["qc_wall_curvature_change_rate_mm_inv_s"]
            )
        input_paths.append(dicom_path)

    data = StabilityCaseData(
        case_id=case_id,
        rsr=np.asarray(rsr["current_qc_radial_strain_rate_s"]),
        rsr_fps=rsr["fps"],
        pair_valid=pair_valid,
        anatomical_fps=anatomical["fps"],
        cavity=np.asarray(anatomical["qc_cavity_width_strain_rate_s"]),
        cavity_valid=np.asarray(anatomical["cavity_qc_valid"], dtype=bool),
        longitudinal=np.asarray(anatomical["qc_longitudinal_wall_strain_rate_s"]),
        longitudinal_valid=np.asarray(anatomical["longitudinal_qc_valid"], dtype=bool),
        curvature_rate=curvature_rate,
        physical_curvature_available=physical_curvature_available,
        physical_curvature_rate_available=physical_curvature_rate_available,
    )
    compute_feature_row(data)
    normalized_position = np.asarray(
        anatomical["normalized_cervix_to_fundus_position"], dtype=np.float64
    )
    pair_shadow = np.asarray(shadow["pair_quality_red_candidate"], dtype=bool)
    if pair_shadow.shape != pair_valid.shape:
        raise ValueError(f"{shadow_path}: shadow mask does not match pair validity")
    fps = float(np.asarray(data.rsr_fps).reshape(-1)[0])
    duration_s = len(data.rsr) / fps
    metadata: dict[str, object] = {
        "case_id": case_id,
        "frame_count": len(data.rsr),
        "fps": fps,
        "video_duration_s": duration_s,
        "video_length_stratum": length_stratum(duration_s),
        "topology_risk": scalar_bool(
            anatomical.get("has_tracking_topology_risk", np.asarray(False))
        ),
        "dicom_physical_curvature_rate_available": physical_curvature_rate_available,
    }
    metadata.update(
        load_artifact_metadata(
            artifact_dir / f"{case_id}_artifact_qc_summary.csv", len(data.rsr)
        )
    )
    metadata.update(formal_analysis_eligibility(anatomical))
    return data, normalized_position, pair_shadow, metadata, input_paths


def partition_stability_results(
    long_table: pd.DataFrame, case_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve diagnostic rows; only explicit upstream eligibility enters main results."""
    fields = [
        "case_id", "formal_analysis_eligible", "formal_analysis_status", "formal_analysis_reason"
    ]
    audit = case_metadata.reindex(columns=fields)
    if audit["case_id"].isna().any() or not audit["case_id"].is_unique:
        raise ValueError("eligibility metadata must contain unique case_id values")
    diagnostic = long_table.merge(audit, on="case_id", how="left", validate="many_to_one")
    # Metadata comes from the shared validator; a missing join remains ineligible.
    accepted = diagnostic["formal_analysis_eligible"].eq(True) & diagnostic["formal_analysis_status"].eq("eligible")
    diagnostic["formal_analysis_eligible"] = accepted
    diagnostic["formal_analysis_status"] = diagnostic["formal_analysis_status"].fillna("unknown")
    diagnostic["formal_analysis_reason"] = diagnostic["formal_analysis_reason"].fillna("missing_eligibility_metadata")
    diagnostic["analysis_scope"] = "diagnostic_all_cases"
    formal = diagnostic.loc[accepted].copy()
    formal["analysis_scope"] = "upstream_eligible_imaging_only"
    return diagnostic, formal


def common_perturbations(
    data: StabilityCaseData,
    normalized_position: np.ndarray,
    shadow: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fraction, label in ((0.05, "5pct"), (0.10, "10pct")):
        for seed in RANDOM_SEEDS:
            exclusion, metadata = random_pair_exclusion(
                data.pair_valid,
                target_removed_fraction=fraction,
                seed=seed,
            )
            rows.append(
                {
                    "perturbation": f"random_delete_{label}",
                    "perturbation_id": f"random_{label}_seed_{seed}",
                    "target_removed_fraction": fraction,
                    "seed": seed,
                    "exclusion": exclusion,
                    **metadata,
                }
            )
        for axis in ("time", "space"):
            for position_index in range(5):
                exclusion, metadata = fixed_contiguous_pair_exclusion(
                    data.pair_valid,
                    axis=axis,
                    target_removed_fraction=fraction,
                    position_index=position_index,
                )
                rows.append(
                    {
                        "perturbation": f"{axis}_block_target_{label}",
                        "perturbation_id": f"{axis}_{label}_position_{position_index + 1}",
                        "target_removed_fraction": fraction,
                        "seed": math.nan,
                        "exclusion": exclusion,
                        **metadata,
                    }
                )
    for index in range(5):
        exclusion, metadata = temporal_quintile_pair_exclusion(data.pair_valid, index)
        rows.append(
            {
                "perturbation": "temporal_jackknife",
                "perturbation_id": f"T{index + 1}",
                "target_removed_fraction": 0.20,
                "seed": math.nan,
                "exclusion": exclusion,
                **metadata,
            }
        )
        exclusion, metadata = spatial_quintile_pair_exclusion(
            data.pair_valid, normalized_position, index
        )
        rows.append(
            {
                "perturbation": "spatial_jackknife",
                "perturbation_id": f"S{index + 1}",
                "target_removed_fraction": 0.20,
                "seed": math.nan,
                "exclusion": exclusion,
                **metadata,
            }
        )
    pair_shadow = np.asarray(shadow, dtype=bool) & data.pair_valid
    before = int(np.count_nonzero(data.pair_valid))
    removed = int(np.count_nonzero(pair_shadow))
    rows.append(
        {
            "perturbation": "shadow_sensitivity",
            "perturbation_id": "formal_plus_red_shadow",
            "target_removed_fraction": math.nan,
            "seed": math.nan,
            "exclusion": pair_shadow,
            "pair_valid_count_before": before,
            "pair_valid_count_removed": removed,
            "pair_actual_removed_valid_fraction": removed / before if before else math.nan,
            "realized_axis": "valid_pair_position",
            "realized_removed_indices_count": removed,
            "realized_axis_length": before,
            "realized_upstream_removed_fraction": removed / before if before else math.nan,
        }
    )
    return rows


def status_for(
    spec_family: str,
    baseline: float,
    perturbed: float,
    curvature_available: bool,
) -> str:
    if not math.isfinite(baseline):
        if spec_family == "curvature" and not curvature_available:
            return "DICOM_rate_unavailable"
        return "baseline_unavailable"
    if not math.isfinite(perturbed):
        return "finite_to_nan"
    return "ok"


def change_values(baseline: float, perturbed: float) -> tuple[float, float, float]:
    if not math.isfinite(baseline) or not math.isfinite(perturbed):
        return math.nan, math.nan, math.nan
    signed = perturbed - baseline
    return signed, abs(signed), symmetric_relative_difference_pct(baseline, perturbed)


def realized_classification_group(perturbation: dict[str, object]) -> str:
    """Preserve perturbation-family grouping while collapsing nominal aliases."""

    nominal_name = str(perturbation["perturbation"])
    if nominal_name.startswith("space_block_target_"):
        return (
            "space_block_realized_contiguous_"
            f"{perturbation['realized_removed_indices_count']}_section"
        )
    return nominal_name


def analyze_case(
    data: StabilityCaseData,
    normalized_position: np.ndarray,
    shadow: np.ndarray,
    baseline_row: dict[str, object],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    input_snapshot = {
        name: None if getattr(data, name) is None else np.asarray(getattr(data, name)).copy()
        for name in (
            "rsr",
            "pair_valid",
            "cavity",
            "cavity_valid",
            "longitudinal",
            "longitudinal_valid",
            "curvature_rate",
        )
    }
    curvature_available = scalar_bool(data.physical_curvature_rate_available)
    fps = float(np.asarray(data.rsr_fps).reshape(-1)[0])
    for perturbation in common_perturbations(data, normalized_position, shadow):
        exclusion = np.asarray(perturbation["exclusion"], dtype=bool)
        realized_id = realized_mask_identity(exclusion)
        nominal_name = str(perturbation["perturbation"])
        realized_group = realized_classification_group(perturbation)
        perturbed_data = apply_pair_exclusion(data, exclusion)
        perturbed_row = compute_feature_row(perturbed_data)
        selected_frame_count = int(np.count_nonzero(np.any(exclusion, axis=(1, 2))))
        for spec in FEATURE_SPECS:
            baseline = float(baseline_row[spec.name])
            perturbed = float(perturbed_row[spec.name])
            baseline_n, _ = valid_count_and_ratio(data, spec)
            perturbed_n, _ = valid_count_and_ratio(perturbed_data, spec)
            status = status_for(spec.family, baseline, perturbed, curvature_available)
            signed, absolute, srd = change_values(baseline, perturbed)
            assessment = (
                "高敏感"
                if status == "finite_to_nan"
                else sensitivity_warning(spec.statistic_type, baseline, perturbed)
            )
            output.append(
                {
                    "case_id": data.case_id,
                    "feature": spec.name,
                    "feature_family": spec.family,
                    "statistic_type": spec.statistic_type,
                    "feature_domain": spec.domain,
                    "perturbation": perturbation["perturbation"],
                    "perturbation_id": perturbation["perturbation_id"],
                    "nominal_perturbation_name": nominal_name,
                    "nominal_target_fraction": perturbation[
                        "target_removed_fraction"
                    ],
                    "realized_perturbation_id": realized_id,
                    "realized_classification_group": realized_group,
                    "realized_axis": perturbation.get("realized_axis", "unknown"),
                    "realized_removed_indices_count": perturbation.get(
                        "realized_removed_indices_count", math.nan
                    ),
                    "realized_axis_length": perturbation.get(
                        "realized_axis_length", math.nan
                    ),
                    "realized_upstream_removed_fraction": perturbation.get(
                        "realized_upstream_removed_fraction", math.nan
                    ),
                    "realized_index_start": perturbation.get(
                        "block_start_index", math.nan
                    ),
                    "realized_index_stop_exclusive": perturbation.get(
                        "block_stop_index_exclusive", math.nan
                    ),
                    "seed": perturbation.get("seed", math.nan),
                    "baseline_value": baseline,
                    "perturbed_value": perturbed,
                    "signed_change": signed,
                    "absolute_change": absolute,
                    "relative_or_SRD_change": srd,
                    "target_removed_fraction": perturbation["target_removed_fraction"],
                    "actual_removed_fraction": (
                        (baseline_n - perturbed_n) / baseline_n
                        if baseline_n
                        else math.nan
                    ),
                    "baseline_valid_n": baseline_n,
                    "perturbed_valid_n": perturbed_n,
                    "target_removed_n": perturbation.get("target_removed_n", math.nan),
                    "actual_removed_n": baseline_n - perturbed_n,
                    "downstream_valid_n_before": baseline_n,
                    "downstream_valid_n_after": perturbed_n,
                    "downstream_lost_n": baseline_n - perturbed_n,
                    "downstream_lost_fraction": (
                        (baseline_n - perturbed_n) / baseline_n
                        if baseline_n
                        else math.nan
                    ),
                    "selected_frame_count": selected_frame_count,
                    "selected_duration_s": selected_frame_count / fps,
                    "insufficient_n": False,
                    "status": status,
                    "assessment": assessment,
                }
            )

    for spec in FEATURE_SPECS:
        baseline = float(baseline_row[spec.name])
        values, valid = domain_values_and_valid(data, spec)
        keep, top_metadata = remove_top_fraction_mask(values, valid)
        removed_mask = valid & ~keep
        perturbed, baseline_n, perturbed_n, status = top_fraction_feature_value(
            data, spec
        )
        signed, absolute, srd = change_values(baseline, perturbed)
        output.append(
            {
                "case_id": data.case_id,
                "feature": spec.name,
                "feature_family": spec.family,
                "statistic_type": spec.statistic_type,
                "feature_domain": spec.domain,
                "perturbation": "top1_removal",
                "perturbation_id": f"top1_{spec.domain}",
                "nominal_perturbation_name": "top1_removal",
                "nominal_target_fraction": 0.01,
                "realized_perturbation_id": realized_mask_identity(removed_mask),
                "realized_classification_group": "top1_removal",
                "realized_axis": "feature_domain_sample",
                "realized_removed_indices_count": int(
                    top_metadata["valid_count_removed"]
                ),
                "realized_axis_length": int(top_metadata["valid_count_before"]),
                "realized_upstream_removed_fraction": (
                    int(top_metadata["valid_count_removed"])
                    / int(top_metadata["valid_count_before"])
                    if int(top_metadata["valid_count_before"])
                    else math.nan
                ),
                "realized_index_start": math.nan,
                "realized_index_stop_exclusive": math.nan,
                "seed": math.nan,
                "baseline_value": baseline,
                "perturbed_value": perturbed,
                "signed_change": signed,
                "absolute_change": absolute,
                "relative_or_SRD_change": srd,
                "target_removed_fraction": 0.01,
                "actual_removed_fraction": (
                    (baseline_n - perturbed_n) / baseline_n if baseline_n else math.nan
                ),
                "baseline_valid_n": baseline_n,
                "perturbed_valid_n": perturbed_n,
                "target_removed_n": math.ceil(0.01 * baseline_n) if baseline_n >= 100 else 0,
                "actual_removed_n": baseline_n - perturbed_n,
                "downstream_valid_n_before": baseline_n,
                "downstream_valid_n_after": perturbed_n,
                "downstream_lost_n": baseline_n - perturbed_n,
                "downstream_lost_fraction": (
                    (baseline_n - perturbed_n) / baseline_n if baseline_n else math.nan
                ),
                "selected_frame_count": math.nan,
                "selected_duration_s": math.nan,
                "insufficient_n": status == "insufficient_n",
                "status": status,
                "assessment": (
                    sensitivity_warning(spec.statistic_type, baseline, perturbed)
                    if status == "ok"
                    else "证据不足"
                ),
            }
        )

    for name, original in input_snapshot.items():
        current = getattr(data, name)
        if original is None:
            if current is not None:
                raise RuntimeError(f"baseline field changed in place: {name}")
        elif current is None or not np.array_equal(original, current, equal_nan=True):
            raise RuntimeError(f"baseline field changed in place: {name}")
    return output


def independent_realized_group_count(high: pd.DataFrame) -> int:
    """Count group evidence after same-case identical-mask aliases are collapsed."""

    if len(high) == 0:
        return 0
    group_column = (
        "realized_classification_group"
        if "realized_classification_group" in high.columns
        else "perturbation"
    )
    if "realized_perturbation_id" not in high.columns:
        return int(high[group_column].nunique())
    support_signatures = []
    for _, group_rows in high.groupby(group_column, sort=False):
        support_signatures.append(
            frozenset(
                zip(
                    group_rows["case_id"].astype(str),
                    group_rows["realized_perturbation_id"].astype(str),
                )
            )
        )
    return len(set(support_signatures))


def frozen_feature_classification(rows: pd.DataFrame, n_evaluable: int) -> tuple[str, str]:
    evaluated = rows[rows["status"].isin(("ok", "finite_to_nan"))]
    if n_evaluable < CLASSIFICATION_RULE["minimum_evaluable_cases"] or len(evaluated) == 0:
        return "证据不足", "少于3例有可计算baseline或无可评价扰动"
    finite_to_nan_cases = evaluated.loc[
        evaluated["status"] == "finite_to_nan", "case_id"
    ].nunique()
    high = evaluated[evaluated["assessment"] == "高敏感"]
    high_case_count = high["case_id"].nunique()
    high_group_count = independent_realized_group_count(high)
    if finite_to_nan_cases >= 2:
        return "不稳定", "至少2例出现finite_to_nan"
    if high_case_count >= 2 and high_group_count >= 2:
        return "不稳定", "至少2例且至少2类由不同actual mask支持的扰动达到冻结高敏感警示线"
    moderate_or_high_cases = evaluated.loc[
        evaluated["assessment"].isin(("中等敏感", "高敏感")), "case_id"
    ].nunique()
    if high_case_count == 0 and moderate_or_high_cases <= 1:
        return "稳定", "未出现高敏感且中高敏感仅见于至多1例"
    return "需谨慎", "存在局部敏感性但未达到系统性不稳定规则"


def group_metrics(rows: pd.DataFrame, statistic_type: str) -> dict[str, object]:
    evaluated = rows[rows["status"].isin(("ok", "finite_to_nan"))]
    finite = evaluated[np.isfinite(evaluated["relative_or_SRD_change"])]
    absolute = evaluated[np.isfinite(evaluated["absolute_change"])]
    low_limit = 10.0 if statistic_type == "median" else 15.0
    scored = evaluated.assign(
        _risk_srd=np.where(
            evaluated["status"] == "finite_to_nan",
            200.0,
            evaluated["relative_or_SRD_change"],
        )
    )
    case_worst = scored.groupby("case_id")["_risk_srd"].max()
    return {
        "n_evaluable_cases": int(evaluated["case_id"].nunique()),
        "median_srd_pct": float(finite["relative_or_SRD_change"].median()) if len(finite) else math.nan,
        "p90_srd_pct": float(finite["relative_or_SRD_change"].quantile(0.90)) if len(finite) else math.nan,
        "p95_srd_pct": float(finite["relative_or_SRD_change"].quantile(0.95)) if len(finite) else math.nan,
        "maximum_srd_pct": float(finite["relative_or_SRD_change"].max()) if len(finite) else math.nan,
        "median_absolute_change": float(absolute["absolute_change"].median()) if len(absolute) else math.nan,
        "p95_absolute_change": float(absolute["absolute_change"].quantile(0.95)) if len(absolute) else math.nan,
        "maximum_absolute_change": float(absolute["absolute_change"].max()) if len(absolute) else math.nan,
        "stable_case_fraction": float(np.mean(case_worst <= low_limit)) if len(case_worst) else math.nan,
    }


def summarize_features(long_table: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for index, spec in enumerate(FEATURE_SPECS, start=1):
        rows = long_table[long_table["feature"] == spec.name]
        n_evaluable = rows.loc[np.isfinite(rows["baseline_value"]), "case_id"].nunique()
        classification, reason = frozen_feature_classification(rows, int(n_evaluable))
        high = rows[
            rows["status"].isin(("ok", "finite_to_nan"))
            & (rows["assessment"] == "高敏感")
        ]
        realized_group_column = (
            "realized_classification_group"
            if "realized_classification_group" in high.columns
            else "perturbation"
        )
        summary: dict[str, object] = {
            "feature_code": f"F{index:02d}",
            "feature_name": spec.name,
            "feature_family": spec.family,
            "statistic_type": spec.statistic_type,
            "feature_domain": spec.domain,
            "n_evaluable_cases": int(n_evaluable),
        }
        group_p95: dict[str, float] = {}
        for group in PERTURBATION_GROUPS:
            metrics = group_metrics(rows[rows["perturbation"] == group], spec.statistic_type)
            for key, value in metrics.items():
                summary[f"{group}_{key}"] = value
            group_p95[group] = float(metrics["p95_srd_pct"])
        finite_groups = {k: v for k, v in group_p95.items() if math.isfinite(v)}
        worst_group = max(finite_groups, key=finite_groups.get) if finite_groups else "not_evaluable"
        overall = group_metrics(rows, spec.statistic_type)
        summary.update(
            {
                "overall_median_srd_pct": overall["median_srd_pct"],
                "overall_p90_srd_pct": overall["p90_srd_pct"],
                "overall_p95_srd_pct": overall["p95_srd_pct"],
                "overall_maximum_srd_pct": overall["maximum_srd_pct"],
                "overall_stable_case_fraction": overall["stable_case_fraction"],
                "overall_worst_perturbation": worst_group,
                "high_sensitivity_case_count": int(high["case_id"].nunique()),
                "high_sensitivity_nominal_group_count": int(
                    high["perturbation"].nunique()
                ),
                "high_sensitivity_realized_group_count": int(
                    high[realized_group_column].nunique()
                ),
                "high_sensitivity_independent_realized_group_count": (
                    independent_realized_group_count(high)
                ),
                "finite_to_nan_count": int((rows["status"] == "finite_to_nan").sum()),
                "top1_insufficient_n_count": int((rows["status"] == "insufficient_n").sum()),
                "final_classification": classification,
                "classification_reason": reason,
                "classification_rule_semantics": CLASSIFICATION_RULE["semantics"],
                "feature_elimination_allowed": False,
            }
        )
        summaries.append(summary)
    return pd.DataFrame(summaries)


def duplicate_only_rescore(
    v1_long: pd.DataFrame, v1_summary: pd.DataFrame
) -> pd.DataFrame:
    original = v1_summary.set_index("feature_name")
    rows: list[dict[str, object]] = []
    for spec in FEATURE_SPECS:
        feature_rows = v1_long[v1_long["feature"] == spec.name].copy()
        feature_rows["realized_classification_group"] = feature_rows[
            "perturbation"
        ].replace(
            {
                "space_block_target_5pct": "space_block_realized_contiguous_1_section",
                "space_block_target_10pct": "space_block_realized_contiguous_1_section",
            }
        )
        high = feature_rows[
            feature_rows["status"].isin(("ok", "finite_to_nan"))
            & (feature_rows["assessment"] == "高敏感")
        ]
        n_evaluable = feature_rows.loc[
            np.isfinite(feature_rows["baseline_value"]), "case_id"
        ].nunique()
        classification, reason = frozen_feature_classification(
            feature_rows, int(n_evaluable)
        )
        original_classification = str(
            original.loc[spec.name, "final_classification"]
        )
        rows.append(
            {
                "feature": spec.name,
                "original_classification": original_classification,
                "original_high_sensitivity_group_count": int(
                    high["perturbation"].nunique()
                ),
                "deduplicated_high_sensitivity_group_count": int(
                    high["realized_classification_group"].nunique()
                ),
                "high_sensitivity_case_count": int(high["case_id"].nunique()),
                "classification_after_dedup_only": classification,
                "changed_yes_no": (
                    "yes" if classification != original_classification else "no"
                ),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def cross_family_realized_mask_rescore(
    v1_1_long: pd.DataFrame, v1_1_summary: pd.DataFrame
) -> pd.DataFrame:
    """Reclassify v1.1 rows after same-case actual-mask alias deduplication."""

    current = v1_1_summary.set_index("feature_name")
    rows: list[dict[str, object]] = []
    for spec in FEATURE_SPECS:
        feature_rows = v1_1_long[v1_1_long["feature"] == spec.name]
        high = feature_rows[
            feature_rows["status"].isin(("ok", "finite_to_nan"))
            & (feature_rows["assessment"] == "高敏感")
        ]
        n_evaluable = feature_rows.loc[
            np.isfinite(feature_rows["baseline_value"]), "case_id"
        ].nunique()
        classification, reason = frozen_feature_classification(
            feature_rows, int(n_evaluable)
        )
        current_classification = str(
            current.loc[spec.name, "final_classification"]
        )
        rows.append(
            {
                "feature": spec.name,
                "current_classification": current_classification,
                "current_high_sensitivity_nominal_group_count": int(
                    high["perturbation"].nunique()
                ),
                "current_high_sensitivity_realized_group_count": int(
                    high["realized_classification_group"].nunique()
                ),
                "corrected_independent_realized_group_count": (
                    independent_realized_group_count(high)
                ),
                "independent_case_mask_evidence_count": int(
                    high[["case_id", "realized_perturbation_id"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "high_sensitivity_case_count": int(high["case_id"].nunique()),
                "classification_after_cross_family_dedup": classification,
                "changed_yes_no": (
                    "yes"
                    if classification != current_classification
                    else "no"
                ),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def worst_case_feature_table(long_table: pd.DataFrame) -> pd.DataFrame:
    evaluated = long_table[
        long_table["status"].isin(("ok", "finite_to_nan"))
    ].copy()
    evaluated["risk_srd_pct"] = np.where(
        evaluated["status"] == "finite_to_nan",
        200.0,
        evaluated["relative_or_SRD_change"],
    )
    evaluated = evaluated[np.isfinite(evaluated["risk_srd_pct"])]
    indices = evaluated.groupby(["case_id", "feature"])["risk_srd_pct"].idxmax()
    return evaluated.loc[indices].reset_index(drop=True)


def risk_strata_rows(
    long_table: pd.DataFrame, case_metadata: pd.DataFrame
) -> list[dict[str, object]]:
    worst = worst_case_feature_table(long_table).merge(case_metadata, on="case_id")
    rows: list[dict[str, object]] = []
    dimensions = {
        "video_length": "video_length_stratum",
        "artifact_risk": "artifact_risk_stratum",
        "topology_risk": "topology_risk",
    }
    for dimension, column in dimensions.items():
        for value, group in worst.groupby(column, dropna=False):
            srd = group["risk_srd_pct"]
            unstable = group["assessment"] == "高敏感"
            rows.append(
                {
                    "risk_dimension": dimension,
                    "risk_group": str(value),
                    "n_cases": int(group["case_id"].nunique()),
                    "n_evaluable_case_features": int(len(group)),
                    "median_worst_srd_pct": float(srd.median()),
                    "p90_worst_srd_pct": float(srd.quantile(0.90)),
                    "p95_worst_srd_pct": float(srd.quantile(0.95)),
                    "maximum_worst_srd_pct": float(srd.max()),
                    "high_sensitivity_case_feature_fraction": float(np.mean(unstable)),
                    "systemic_pattern_flag": bool(
                        group["case_id"].nunique() >= 3 and np.mean(unstable) >= 0.50
                    ),
                }
            )
    return rows


def anomaly_rows(
    long_table: pd.DataFrame, case_metadata: pd.DataFrame
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case_id, case_rows in long_table.groupby("case_id"):
        evaluated = case_rows[
            case_rows["status"].isin(("ok", "finite_to_nan"))
        ].copy()
        evaluated["risk_srd_pct"] = np.where(
            evaluated["status"] == "finite_to_nan",
            200.0,
            evaluated["relative_or_SRD_change"],
        )
        evaluated = evaluated[np.isfinite(evaluated["risk_srd_pct"])]
        high_features = sorted(evaluated.loc[evaluated["assessment"] == "高敏感", "feature"].unique())
        if len(evaluated):
            worst = evaluated.loc[evaluated["risk_srd_pct"].idxmax()]
            most_sensitive = str(worst["perturbation"])
            maximum_srd = float(worst["risk_srd_pct"])
        else:
            most_sensitive = "not_evaluable"
            maximum_srd = math.nan
        metadata = case_metadata.set_index("case_id").loc[case_id].to_dict()
        rows.append(
            {
                "case_id": case_id,
                "has_obviously_unstable_feature": bool(high_features),
                "obviously_unstable_feature_count": len(high_features),
                "obviously_unstable_features": ";".join(high_features),
                "most_sensitive_perturbation": most_sensitive,
                "maximum_srd_pct": maximum_srd,
                "video_duration_s": metadata["video_duration_s"],
                "video_length_stratum": metadata["video_length_stratum"],
                "artifact_grade2_percent": metadata["artifact_grade2_percent"],
                "automatic_grade3_candidate_fraction": metadata[
                    "automatic_grade3_candidate_fraction"
                ],
                "artifact_risk_stratum": metadata["artifact_risk_stratum"],
                "topology_risk": metadata["topology_risk"],
                "dicom_physical_curvature_rate_available": metadata[
                    "dicom_physical_curvature_rate_available"
                ],
            }
        )
    return rows


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "无。"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "NA" if not math.isfinite(value) else f"{value:.4g}"
            values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    provenance: dict[str, object],
    discovered: list[str],
    excluded: list[dict[str, str]],
    tested: list[str],
    summary: pd.DataFrame,
    long_table: pd.DataFrame,
    risk_rows: list[dict[str, object]],
    anomalies: list[dict[str, object]],
    method_pass: bool,
    v1_summary: pd.DataFrame,
    v1_1_summary: pd.DataFrame,
    duplicate_rescore: pd.DataFrame,
    cross_family_rescore: pd.DataFrame,
) -> None:
    counts = summary["final_classification"].value_counts().to_dict()
    dicom_features = summary[summary["feature_family"] == "curvature"]
    dicom_n_text = "; ".join(
        f"{row.feature_name}={int(row.n_evaluable_cases)}"
        for row in dicom_features.itertuples()
    )
    perturbation_runs = len(tested) * (41 + len(FEATURE_SPECS))
    systematic = [row for row in risk_rows if row["systemic_pattern_flag"]]
    abnormal = [row for row in anomalies if row["has_obviously_unstable_feature"]]
    excluded_text = markdown_table(excluded, ["case_id", "reason"])
    risk_text = markdown_table(
        risk_rows,
        [
            "risk_dimension",
            "risk_group",
            "n_cases",
            "n_evaluable_case_features",
            "median_worst_srd_pct",
            "p95_worst_srd_pct",
            "maximum_worst_srd_pct",
            "high_sensitivity_case_feature_fraction",
            "systemic_pattern_flag",
        ],
    )
    anomaly_text = markdown_table(
        abnormal,
        [
            "case_id",
            "obviously_unstable_feature_count",
            "most_sensitive_perturbation",
            "maximum_srd_pct",
            "video_duration_s",
            "artifact_risk_stratum",
            "topology_risk",
            "dicom_physical_curvature_rate_available",
        ],
    )
    method_text = "PASS" if method_pass else "FAIL"
    v1_counts = v1_summary["final_classification"].value_counts().to_dict()
    v1_1_counts = v1_1_summary["final_classification"].value_counts().to_dict()
    cross_counts = cross_family_rescore[
        "classification_after_cross_family_dedup"
    ].value_counts().to_dict()
    comparison_rows = []
    current_by_feature = summary.set_index("feature_name")
    v1_1_by_feature = v1_1_summary.set_index("feature_name")
    cross_by_feature = cross_family_rescore.set_index("feature")
    for row in v1_summary.itertuples():
        comparison_rows.append(
            {
                "feature": row.feature_name,
                "v1": row.final_classification,
                "v1_1": v1_1_by_feature.loc[
                    row.feature_name, "final_classification"
                ],
                "cross_family_dedup_rescore": cross_by_feature.loc[
                    row.feature_name,
                    "classification_after_cross_family_dedup",
                ],
                "current_rerun": current_by_feature.loc[
                    row.feature_name, "final_classification"
                ],
            }
        )
    comparison_text = markdown_table(
        comparison_rows,
        ["feature", "v1", "v1_1", "cross_family_dedup_rescore", "current_rerun"],
    )
    review_columns = [
        "feature_name",
        "n_evaluable_cases",
        "high_sensitivity_case_count",
        "high_sensitivity_nominal_group_count",
        "high_sensitivity_realized_group_count",
        "high_sensitivity_independent_realized_group_count",
        "overall_worst_perturbation",
        "overall_p95_srd_pct",
        "overall_maximum_srd_pct",
        "classification_reason",
    ]
    unstable_text = markdown_table(
        summary.loc[summary["final_classification"] == "不稳定", review_columns]
        .to_dict(orient="records"),
        review_columns,
    )
    caution_text = markdown_table(
        summary.loc[summary["final_classification"] == "需谨慎", review_columns]
        .to_dict(orient="records"),
        review_columns,
    )
    high_rows = long_table[
        long_table["status"].isin(("ok", "finite_to_nan"))
        & (long_table["assessment"] == "高敏感")
    ]
    high_source_rows: list[dict[str, object]] = []
    for (feature, group), group_rows in high_rows.groupby(
        ["feature", "realized_classification_group"], sort=True
    ):
        high_source_rows.append(
            {
                "feature": feature,
                "realized_perturbation_group": group,
                "n_high_cases": int(group_rows["case_id"].nunique()),
                "n_high_rows": int(len(group_rows)),
                "maximum_srd_pct": float(
                    group_rows["relative_or_SRD_change"].max()
                ),
            }
        )
    high_source_text = markdown_table(
        high_source_rows,
        [
            "feature",
            "realized_perturbation_group",
            "n_high_cases",
            "n_high_rows",
            "maximum_srd_pct",
        ],
    )
    space_rows = long_table[
        long_table["nominal_perturbation_name"].str.startswith(
            "space_block_target_"
        )
    ]
    space_axes = space_rows["realized_axis_length"].dropna().astype(int)
    space_fractions = space_rows["realized_upstream_removed_fraction"].dropna()
    path.write_text(
        f"""# 新患者合格队列的正式候选特征扰动稳定性/鲁棒性验证

## 总体结论

- 发现病例数：{len(discovered)}
- 符合稳定性测试条件病例数：{len(tested)}
- 未进入主队列病例数（含输入失败及仅诊断病例）：{len(excluded)}
- 正式生物学候选特征数：{len(summary)}
- 总扰动运行次数：{perturbation_runs}（每例41个共享mask扰动 + 20个feature-specific top1%扰动）
- long-format特征评估行数：{len(long_table)}
- 稳定feature：{counts.get('稳定', 0)}
- 需谨慎feature：{counts.get('需谨慎', 0)}
- 不稳定feature：{counts.get('不稳定', 0)}
- 证据不足feature：{counts.get('证据不足', 0)}
- 方法级稳定性：**{method_text}**

本轮只在冻结正式NPZ的现有有效数据上增加内存扰动mask，并调用同一正式特征提取器重新汇总；没有重新运行或修改Huang tracking、P3、artifact QC、formal QC、人工复核或DICOM曲率计算，也没有覆盖正式NPZ。

当前主表和方法级结论仅使用上游资格明确合格且无拓扑风险的病例。所有可读取病例的特征及风险状态保存在analysis_eligibility_audit.csv；包含风险/未知病例的诊断性扰动和汇总另存diagnostic_stability_*.csv，不能直接作为正式分析队列。影像资格不替代患者/治疗周期匹配。
历史版本对照保留原队列，本版本队列可能不同，分类变化不能只归因于扰动去重。

SRD及稳定/需谨慎/不稳定分界属于项目预设工程稳定性规则，不是子宫内膜临床文献公认阈值。所有判断同时保留absolute change与SRD。

## v1重复计数根因与修复

v1的空间block nominal target是“希望删除的正式有效pair数量比例”，不是空间坐标范围、section数量或下游feature样本比例。本轮病例为{int(space_axes.min())}–{int(space_axes.max())}个section，实际空间删除比例范围为{100 * float(space_fractions.min()):.1f}%–{100 * float(space_fractions.max()):.1f}%。历史28例的5%和10%在28例×5位置共140组比较中均落到同一个单section mask。v1分类按nominal `perturbation`字段计数，因而把同一realized mask的5%和10%标签重复算作两个组。

v1.1没有改变mask生成。它保留两个nominal行用于审计，但增加精确mask identity和归一化realized classification group；相同单section空间扰动在高敏感组计数中只算一次。

`actual_removed_n/fraction`的真实含义是downstream feature有效样本损失，不是上游空间位置删除比例。v1.1新增独立的nominal、realized upstream和downstream字段。曲率依赖同侧连续三个section，因此删除一个上游section可以使多个曲率样本失效，这是邻域依赖的数学后果，不代表程序删除了30%–60%的空间section。

## 历史版本与当前合格队列对照

| 指标 | v1 | v1.1 | cross-family dedup re-score | current rerun |
|---|---:|---:|---:|---:|
| 稳定特征数 | {v1_counts.get('稳定', 0)} | {v1_1_counts.get('稳定', 0)} | {cross_counts.get('稳定', 0)} | {counts.get('稳定', 0)} |
| 需谨慎特征数 | {v1_counts.get('需谨慎', 0)} | {v1_1_counts.get('需谨慎', 0)} | {cross_counts.get('需谨慎', 0)} | {counts.get('需谨慎', 0)} |
| 不稳定特征数 | {v1_counts.get('不稳定', 0)} | {v1_1_counts.get('不稳定', 0)} | {cross_counts.get('不稳定', 0)} | {counts.get('不稳定', 0)} |
| 证据不足数 | {v1_counts.get('证据不足', 0)} | {v1_1_counts.get('证据不足', 0)} | {cross_counts.get('证据不足', 0)} | {counts.get('证据不足', 0)} |
| method PASS/FAIL | FAIL | FAIL | FAIL | {method_text} |

{comparison_text}

## 不稳定特征逐项复核

以下{counts.get('不稳定', 0)}项在同病例跨家族相同actual mask去重后仍不稳定。判定依据同时要求至少2个高敏感病例、至少2个由不同actual mask支持的扰动组。

{unstable_text}

## 需谨慎特征逐项复核

以下{counts.get('需谨慎', 0)}项存在局部敏感性，但未达到冻结的系统性不稳定规则；本轮不自动删除或修改任何feature。

{caution_text}

## 高敏感来源：feature × 实际扰动组

分类先按`case_id + realized_perturbation_id`识别actual mask证据；不同family group若由完全相同的患者+mask集合支持，只计为一个独立扰动组。`spatial_jackknife`仍保留，只有actual mask相同时才去重。

{high_source_text}

## 版本与契约

- Git branch：`{provenance['branch']}`
- Git commit：`{provenance['commit']}`
- 执行时工作树：`{str(provenance['status_porcelain']).replace(chr(10), '; ')}`
- feature contract：`{FEATURE_CONTRACT_VERSION}`
- formal QC version：`{FORMAL_QC_VERSION}`
- stability test version：`{STABILITY_TEST_VERSION}`
- MODEL_INPUT_FEATURE_COLUMNS与BIOLOGICAL_FEATURE_COLUMNS完全一致：True

## 病例范围

实际测试case_id：{', '.join(tested)}

未进入病例：

{excluded_text}

## DICOM曲率

6个DICOM physical curvature-rate特征各自的n_evaluable_cases范围为：{int(dicom_features['n_evaluable_cases'].min()) if len(dicom_features) else 0}–{int(dicom_features['n_evaluable_cases'].max()) if len(dicom_features) else 0}。

{dicom_n_text}

DICOM曲率率不可用时保持NaN，不填0、不插补、不以pixel curvature替代。

## 程序一致性检查

- 原始NPZ运行前后SHA-256一致；扰动数组仅在内存中生成。
- baseline由冻结正式特征提取器生成；每次扰动复用同一提取器，未导入或运行Huang/P3/QC入口。
- 20项MODEL_INPUT_FEATURE_COLUMNS与BIOLOGICAL_FEATURE_COLUMNS严格一致，QC metadata未进入评分。
- v1.1已修复5%/10%同mask重复计组；本轮进一步修复space block与spatial jackknife跨家族同mask重复计组。
- 本轮只改变独立扰动证据计数及其分类；未发现空间mask生成、mask传播或正式feature提取公式错误。这不是对上游算法临床正确性的证明。

## 风险分层

长度分组在查看稳定性结果前冻结为：<30秒、30至<60秒、>=60秒。artifact按无自动3级、自动3级负担<=5%、>5%描述分层；topology使用既有风险字段。分层只描述模式，不修改特征或QC阈值。

{risk_text}

发现系统性风险分层失稳模式：{bool(systematic)}。

## 异常病例标记

这里不是患者合格/不合格判定，只列出至少一个扰动达到冻结高敏感线的病例。

{anomaly_text}

## 方法级冻结判定

预先冻结的PASS条件：20项均可评价；不稳定与证据不足特征均为0；至少50%的特征分类为稳定。否则为FAIL，不在结果产生后调整标准。

若本轮为PASS，后续同类型新增患者不需逐例重复扰动测试，只运行冻结正式主流程、常规QC和正式特征提取。若为FAIL，不能写入`method_level_stability_validated`，应先由负责人处理阻断原因。

## 科学表述边界

本轮是旧五例之外新患者样本上的特征汇总扰动稳定性证据，不是正式test-retest、跨设备、跨操作者reproducibility或临床泛化验证。不进行AUC、机器学习、方向或传播分析。
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-output-dir",
        type=Path,
        default=PROJECT_ROOT / "输出" / "新患者全流程_20260903",
    )
    parser.add_argument("--dicom-dir", type=Path)
    parser.add_argument(
        "--v1-result-dir",
        type=Path,
        default=PROJECT_ROOT / "输出" / "新患者正式候选特征稳定性验证_20260903",
    )
    parser.add_argument(
        "--v1-1-result-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "输出"
            / "新患者正式候选特征稳定性验证_v1_1_realized_group_fix_20260904"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "输出"
            / "新患者正式候选特征稳定性验证_v1_3_analysis_eligibility"
        ),
    )
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if tuple(MODEL_INPUT_FEATURE_COLUMNS) != tuple(BIOLOGICAL_FEATURE_COLUMNS):
        raise RuntimeError("MODEL_INPUT_FEATURE_COLUMNS must equal BIOLOGICAL_FEATURE_COLUMNS")
    if len(BIOLOGICAL_FEATURE_COLUMNS) != 20 or len(FEATURE_SPECS) != 20:
        raise RuntimeError("frozen stability contract must contain exactly 20 features")
    if set(QC_METADATA_COLUMNS) & set(BIOLOGICAL_FEATURE_COLUMNS):
        raise RuntimeError("QC metadata leaked into biological feature columns")

    pipeline = args.pipeline_output_dir.resolve()
    rsr_dir = pipeline / "10_rsr_fourway"
    anatomical_dir = pipeline / "11_anatomical_deformation"
    shadow_dir = pipeline / "09_pair_quality_shadow"
    artifact_dir = pipeline / "07_artifact_qc"
    rsr_suffix = "_rsr_fourway_v1.npz"
    anatomical_suffix = "_anatomical_deformation_v1.npz"
    shadow_suffix = "_pair_quality_shadow_qc_v1.npz"
    discovered_set = (
        case_ids_for_suffix(rsr_dir, rsr_suffix)
        | case_ids_for_suffix(anatomical_dir, anatomical_suffix)
        | case_ids_for_suffix(shadow_dir, shadow_suffix)
    )
    if args.cases:
        requested = {case.casefold(): case for case in args.cases}
        discovered_set = {
            case for case in discovered_set if case.casefold() in requested
        }
        missing_requested = sorted(
            set(requested) - {case.casefold() for case in discovered_set}
        )
        if missing_requested:
            raise ValueError(f"requested cases were not discovered: {missing_requested}")
    discovered = sorted(discovered_set, key=str.casefold)
    if not discovered:
        raise RuntimeError("no new-patient frozen stability inputs were discovered")

    provenance = git_provenance()
    loaded: dict[str, tuple[StabilityCaseData, np.ndarray, np.ndarray]] = {}
    metadata_rows: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    source_paths: list[Path] = []
    for case_id in discovered:
        try:
            data, position, shadow, metadata, paths = load_case(
                case_id,
                rsr_dir=rsr_dir,
                anatomical_dir=anatomical_dir,
                shadow_dir=shadow_dir,
                dicom_dir=None if args.dicom_dir is None else args.dicom_dir.resolve(),
                artifact_dir=artifact_dir,
            )
            loaded[case_id] = (data, position, shadow)
            metadata_rows.append(metadata)
            source_paths.extend(paths)
        except Exception as exc:
            excluded.append(
                {
                    "case_id": case_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    if not loaded:
        raise RuntimeError("no case passed frozen-input validation")

    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    baseline_rows = [compute_feature_row(loaded[case][0]) for case in sorted(loaded)]
    baseline = pd.DataFrame(baseline_rows, columns=FORMAL_CANDIDATE_FEATURE_COLUMNS)
    if baseline.shape != (len(loaded), 27):
        raise RuntimeError("formal baseline must contain one 27-column row per case")
    baseline_by_case = baseline.set_index("case_id").to_dict(orient="index")

    long_rows: list[dict[str, object]] = []
    for case_id in sorted(loaded, key=str.casefold):
        data, position, shadow = loaded[case_id]
        long_rows.extend(
            analyze_case(data, position, shadow, baseline_by_case[case_id])
        )
    long_table = pd.DataFrame(long_rows)
    expected_rows = len(loaded) * len(FEATURE_SPECS) * 42
    if len(long_table) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} long rows, got {len(long_table)}")
    if set(long_table["feature"].unique()) != set(BIOLOGICAL_FEATURE_COLUMNS):
        raise RuntimeError("long table does not contain exactly the frozen 20 features")

    case_metadata = pd.DataFrame(metadata_rows)
    diagnostic_long, long_table = partition_stability_results(long_table, case_metadata)
    if long_table.empty:
        raise RuntimeError("no case has explicit upstream eligibility for the main stability analysis")
    for row in case_metadata.loc[~case_metadata["formal_analysis_eligible"]].itertuples():
        excluded.append({"case_id": row.case_id, "reason": row.formal_analysis_reason})
    eligible_ids = set(long_table["case_id"])
    formal_metadata = case_metadata[case_metadata["case_id"].isin(eligible_ids)]
    summary = summarize_features(long_table)
    summary["analysis_scope"] = "upstream_eligible_imaging_only"
    diagnostic_summary = summarize_features(diagnostic_long)
    diagnostic_summary["analysis_scope"] = "diagnostic_all_cases"
    v1_dir = args.v1_result_dir.resolve()
    v1_long = pd.read_csv(v1_dir / OUTPUT_NAMES[2])
    v1_summary = pd.read_csv(v1_dir / OUTPUT_NAMES[1])
    duplicate_rescore = duplicate_only_rescore(v1_long, v1_summary)
    v1_1_dir = args.v1_1_result_dir.resolve()
    v1_1_long = pd.read_csv(v1_1_dir / OUTPUT_NAMES[2])
    v1_1_summary = pd.read_csv(v1_1_dir / OUTPUT_NAMES[1])
    cross_family_rescore = cross_family_realized_mask_rescore(
        v1_1_long, v1_1_summary
    )
    comparison = duplicate_rescore[
        ["feature", "classification_after_dedup_only"]
    ].rename(columns={"feature": "feature_name"})
    original = v1_summary[
        ["feature_name", "final_classification"]
    ].rename(columns={"final_classification": "v1_final_classification"})
    v1_1_original = v1_1_summary[
        ["feature_name", "final_classification"]
    ].rename(columns={"final_classification": "v1_1_final_classification"})
    cross_comparison = cross_family_rescore[
        ["feature", "classification_after_cross_family_dedup"]
    ].rename(columns={"feature": "feature_name"})
    summary = (
        summary.merge(original, on="feature_name", how="left")
        .merge(v1_1_original, on="feature_name", how="left")
        .merge(comparison, on="feature_name", how="left")
        .merge(cross_comparison, on="feature_name", how="left")
    )
    summary["classification_changed_from_v1"] = (
        summary["final_classification"] != summary["v1_final_classification"]
    )
    summary["classification_changed_from_v1_1"] = (
        summary["final_classification"] != summary["v1_1_final_classification"]
    )
    counts = summary["final_classification"].value_counts().to_dict()
    method_pass = bool(
        len(summary) == 20
        and counts.get("不稳定", 0) == 0
        and counts.get("证据不足", 0) == 0
        and counts.get("稳定", 0) / 20
        >= CLASSIFICATION_RULE["method_pass_minimum_stable_feature_fraction"]
    )
    risk_rows = risk_strata_rows(long_table, formal_metadata)
    anomalies = anomaly_rows(long_table, formal_metadata)

    for path_text, before_hash in source_hashes.items():
        if sha256(Path(path_text)) != before_hash:
            raise RuntimeError(f"frozen source NPZ changed during validation: {path_text}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_long.to_csv(output_dir / OUTPUT_NAMES[7], index=False, encoding="utf-8-sig")
    diagnostic_summary.to_csv(output_dir / OUTPUT_NAMES[8], index=False, encoding="utf-8-sig")
    case_metadata.merge(baseline, on="case_id", validate="one_to_one").to_csv(
        output_dir / OUTPUT_NAMES[9], index=False, encoding="utf-8-sig"
    )
    long_table.to_csv(
        output_dir / OUTPUT_NAMES[2], index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output_dir / OUTPUT_NAMES[1], index=False, encoding="utf-8-sig"
    )
    duplicate_rescore.to_csv(
        output_dir / OUTPUT_NAMES[5], index=False, encoding="utf-8-sig"
    )
    cross_family_rescore.to_csv(
        output_dir / OUTPUT_NAMES[6], index=False, encoding="utf-8-sig"
    )
    spatial_mapping = long_table[
        long_table["nominal_perturbation_name"].str.startswith(
            "space_block_target_"
        )
    ][
        [
            "case_id",
            "perturbation_id",
            "nominal_perturbation_name",
            "nominal_target_fraction",
            "realized_perturbation_id",
            "realized_classification_group",
            "realized_axis",
            "realized_removed_indices_count",
            "realized_axis_length",
            "realized_upstream_removed_fraction",
            "realized_index_start",
            "realized_index_stop_exclusive",
        ]
    ].drop_duplicates()
    spatial_mapping.to_csv(
        output_dir / OUTPUT_NAMES[4], index=False, encoding="utf-8-sig"
    )
    tested = sorted(eligible_ids, key=str.casefold)
    write_report(
        output_dir / OUTPUT_NAMES[0],
        provenance=provenance,
        discovered=discovered,
        excluded=excluded,
        tested=tested,
        summary=summary,
        long_table=long_table,
        risk_rows=risk_rows,
        anomalies=anomalies,
        method_pass=method_pass,
        v1_summary=v1_summary,
        v1_1_summary=v1_1_summary,
        duplicate_rescore=duplicate_rescore,
        cross_family_rescore=cross_family_rescore,
    )
    manifest = {
        "status": (
            "method_level_stability_validated"
            if method_pass
            else "method_level_stability_not_validated"
        ),
        "git_branch": provenance["branch"],
        "git_commit": provenance["commit"],
        "git_status_porcelain": provenance["status_porcelain"],
        "formal_qc_version": FORMAL_QC_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_columns": list(BIOLOGICAL_FEATURE_COLUMNS),
        "feature_count": len(BIOLOGICAL_FEATURE_COLUMNS),
        "model_input_matches_biological": tuple(MODEL_INPUT_FEATURE_COLUMNS)
        == tuple(BIOLOGICAL_FEATURE_COLUMNS),
        "tracking_algorithm_identifier": "tracking_huang_radial_pairs.py@recorded_git_commit",
        "p3_algorithm_identifier": "p3_image_boundary_correction.py@recorded_git_commit",
        "stability_test_version": STABILITY_TEST_VERSION,
        "validation_timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "validation_case_count": len(tested),
        "analysis_scope": "upstream_eligible_imaging_only",
        "diagnostic_case_count": len(loaded),
        "clinical_cycle_matching_checked": False,
        "analysis_eligibility_audit": OUTPUT_NAMES[9],
        "discovered_case_count": len(discovered),
        "excluded_cases": excluded,
        "test_case_ids": tested,
        "perturbation_definitions": {
            "random_pair_deletion": [0.05, 0.10],
            "fixed_random_seeds": list(RANDOM_SEEDS),
            "fixed_time_blocks": {"fractions": [0.05, 0.10], "positions": 5},
            "fixed_space_blocks": {"fractions": [0.05, 0.10], "positions": 5},
            "space_block_realized_classification_semantics": (
                "same realized contiguous section count counts as one group; "
                "nominal rows remain separate"
            ),
            "cross_family_realized_mask_dedup_semantics": (
                "within-case aliases are identified by case_id plus "
                "realized_perturbation_id; group support signatures that are "
                "identical after this deduplication count once"
            ),
            "temporal_jackknife": "five fixed quintiles",
            "spatial_jackknife": "five fixed cervix-to-fundus quintiles",
            "top_fraction_removal": 0.01,
            "top_fraction_minimum_n": 100,
            "shadow_sensitivity": "existing pair_quality_red_candidate mask",
            "arrays_saved": False,
        },
        "random_seeds": list(RANDOM_SEEDS),
        "classification_rule": CLASSIFICATION_RULE,
        "method_level_pass": method_pass,
        "feature_classification_counts": counts,
        "length_strata": [
            {
                "name": name,
                "lower_s_inclusive": lower,
                "upper_s_exclusive": None if math.isinf(upper) else upper,
            }
            for name, lower, upper in LENGTH_STRATA
        ],
        "revalidation_triggers": [
            "substantive Huang tracking change",
            "substantive P3 localization or correction change",
            "formal QC mask definition change",
            "formal artifact exclusion rule change",
            "20-feature formula or aggregation change",
            "median or P95 aggregation definition change",
            "anterior/posterior definition change",
            "DICOM curvature-rate definition or timebase change",
            "materially different input image type, device, or acquisition protocol",
        ],
        "source_npz_sha256": source_hashes,
    }
    (output_dir / OUTPUT_NAMES[3]).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    actual_names = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if actual_names != sorted(OUTPUT_NAMES):
        raise RuntimeError(f"output contract violation: {actual_names}")
    print(
        json.dumps(
            {
                "discovered_cases": len(discovered),
                "tested_cases": len(tested),
                "excluded_cases": len(excluded),
                "long_rows": len(long_table),
                "feature_counts": counts,
                "method_level_stability": "PASS" if method_pass else "FAIL",
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
