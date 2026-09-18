#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Literature-aligned ICC/CV analysis of the frozen v1.2 perturbation table.

This is an independent downstream analysis.  It reads the frozen CSV/JSON
artifacts and never imports or runs tracking, P3, QC, or feature extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.stats import f


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "输出"
    / "新患者正式候选特征稳定性验证_v1_2_cross_family_realized_mask_dedup_20260905"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "输出" / "新患者正式候选特征文献标准鲁棒性分析_20260905"
)
LONG_NAME = "正式候选特征稳定性_long.csv"
SUMMARY_NAME = "正式候选特征稳定性_feature_summary.csv"
SOURCE_MANIFEST_NAME = "stability_validation_manifest.json"
ANALYSIS_VERSION = "literature_aligned_perturbation_robustness_v1"
ICC_REFERENCE_THRESHOLD = 0.90
CV_REFERENCE_THRESHOLDS_PCT = (10.0, 20.0)
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260905

PRIMARY_PERTURBATIONS = (
    "random_delete_5pct",
    "random_delete_10pct",
    "time_block_target_5pct",
    "time_block_target_10pct",
    "space_block_target_5pct",
    "shadow_sensitivity",
)
PRIMARY_MEASUREMENT_IDS = (
    "random_5pct_seed_1729",
    "random_5pct_seed_2718",
    "random_5pct_seed_31415",
    "random_5pct_seed_57721",
    "random_5pct_seed_65537",
    "random_10pct_seed_1729",
    "random_10pct_seed_2718",
    "random_10pct_seed_31415",
    "random_10pct_seed_57721",
    "random_10pct_seed_65537",
    "time_5pct_position_1",
    "time_5pct_position_2",
    "time_5pct_position_3",
    "time_5pct_position_4",
    "time_5pct_position_5",
    "time_10pct_position_1",
    "time_10pct_position_2",
    "time_10pct_position_3",
    "time_10pct_position_4",
    "time_10pct_position_5",
    "space_5pct_position_1",
    "space_5pct_position_2",
    "space_5pct_position_3",
    "space_5pct_position_4",
    "space_5pct_position_5",
    "formal_plus_red_shadow",
)
FAMILY_MEASUREMENT_IDS = {
    "random_deletion": PRIMARY_MEASUREMENT_IDS[:10],
    "time_block": PRIMARY_MEASUREMENT_IDS[10:20],
    "space_block": PRIMARY_MEASUREMENT_IDS[20:25],
    "shadow_sensitivity": PRIMARY_MEASUREMENT_IDS[25:],
    "temporal_jackknife": tuple(f"T{index}" for index in range(1, 6)),
    "spatial_jackknife": tuple(f"S{index}" for index in range(1, 6)),
}
PERTURBATION_TO_FAMILY = {
    "random_delete_5pct": "random_deletion",
    "random_delete_10pct": "random_deletion",
    "time_block_target_5pct": "time_block",
    "time_block_target_10pct": "time_block",
    "space_block_target_5pct": "space_block",
    "space_block_target_10pct": "space_block",
    "shadow_sensitivity": "shadow_sensitivity",
    "temporal_jackknife": "temporal_jackknife",
    "spatial_jackknife": "spatial_jackknife",
    "top1_removal": "top1_removal",
}
REQUIRED_LONG_COLUMNS = {
    "case_id",
    "feature",
    "feature_family",
    "statistic_type",
    "feature_domain",
    "perturbation",
    "perturbation_id",
    "realized_perturbation_id",
    "realized_upstream_removed_fraction",
    "baseline_value",
    "perturbed_value",
    "relative_or_SRD_change",
    "status",
}


@dataclass(frozen=True)
class ICCResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    n_cases: int
    n_measurements_min: int
    n_measurements_max: int
    ci_method: str
    ms_between: float
    ms_within: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def _icc_from_groups(groups: Sequence[np.ndarray]) -> tuple[float, float, float]:
    """One-way random single-measure ICC by ANOVA method of moments.

    For balanced groups this reduces exactly to Shrout-Fleiss ICC(1,1):
    (MS_between - MS_within) / (MS_between + (k - 1) MS_within).
    The unequal-k form uses the standard effective cluster size n0.
    """

    arrays = [np.asarray(group, dtype=np.float64).reshape(-1) for group in groups]
    if len(arrays) < 2 or any(len(group) < 2 for group in arrays):
        return math.nan, math.nan, math.nan
    if any(not np.all(np.isfinite(group)) for group in arrays):
        raise ValueError("ICC groups must contain only finite values")
    counts = np.asarray([len(group) for group in arrays], dtype=np.float64)
    total_n = int(np.sum(counts))
    n_cases = len(arrays)
    means = np.asarray([np.mean(group) for group in arrays])
    grand = float(sum(float(np.sum(group)) for group in arrays) / total_n)
    ss_between = float(np.sum(counts * np.square(means - grand)))
    ss_within = float(
        sum(float(np.sum(np.square(group - mean))) for group, mean in zip(arrays, means))
    )
    ms_between = ss_between / (n_cases - 1)
    ms_within = ss_within / (total_n - n_cases)
    effective_k = (
        total_n - float(np.sum(np.square(counts))) / total_n
    ) / (n_cases - 1)
    denominator = ms_between + (effective_k - 1.0) * ms_within
    if denominator == 0.0:
        estimate = math.nan
    else:
        estimate = (ms_between - ms_within) / denominator
    return float(estimate), float(ms_between), float(ms_within)


def icc_1_1(matrix: np.ndarray, confidence: float = 0.95) -> ICCResult:
    """Balanced ICC(1,1) with the exact central-F confidence interval.

    Formula: Shrout & Fleiss (1979), as used by standard ICC implementations.
    The interval inverts the F ratio MS_between/MS_within.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("ICC matrix must be two-dimensional")
    n_cases, n_measurements = values.shape
    if n_cases < 2 or n_measurements < 2:
        raise ValueError("ICC(1,1) needs at least two cases and two measurements")
    if not np.all(np.isfinite(values)):
        raise ValueError("ICC matrix must be complete and finite")
    estimate, ms_between, ms_within = _icc_from_groups(list(values))
    if not math.isfinite(estimate):
        lower = upper = math.nan
    elif ms_within == 0.0 and ms_between > 0.0:
        lower = upper = 1.0
    else:
        alpha = 1.0 - confidence
        df_between = n_cases - 1
        df_within = n_cases * (n_measurements - 1)
        f_value = ms_between / ms_within if ms_within > 0.0 else math.nan
        lower_f = f_value / f.ppf(1.0 - alpha / 2.0, df_between, df_within)
        upper_f = f_value * f.ppf(1.0 - alpha / 2.0, df_within, df_between)
        lower = (lower_f - 1.0) / (lower_f + n_measurements - 1.0)
        upper = (upper_f - 1.0) / (upper_f + n_measurements - 1.0)
    return ICCResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        n_cases=n_cases,
        n_measurements_min=n_measurements,
        n_measurements_max=n_measurements,
        ci_method="exact_central_F_balanced",
        ms_between=ms_between,
        ms_within=ms_within,
    )


def icc_1_1_unbalanced(
    groups: Sequence[np.ndarray],
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> ICCResult:
    """Unequal-repeat one-way random ICC with a subject-cluster bootstrap CI."""

    arrays = [np.asarray(group, dtype=np.float64).reshape(-1) for group in groups]
    estimate, ms_between, ms_within = _icc_from_groups(arrays)
    if len(arrays) < 2 or not math.isfinite(estimate):
        lower = upper = math.nan
    else:
        rng = np.random.default_rng(seed)
        sampled: list[float] = []
        for _ in range(bootstrap_repetitions):
            indices = rng.integers(0, len(arrays), size=len(arrays))
            value, _, _ = _icc_from_groups([arrays[index] for index in indices])
            if math.isfinite(value):
                sampled.append(value)
        if len(sampled) < max(100, bootstrap_repetitions // 2):
            lower = upper = math.nan
        else:
            lower, upper = np.quantile(sampled, [0.025, 0.975])
    counts = [len(group) for group in arrays]
    return ICCResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        n_cases=len(arrays),
        n_measurements_min=min(counts) if counts else 0,
        n_measurements_max=max(counts) if counts else 0,
        ci_method=f"subject_cluster_percentile_bootstrap_{bootstrap_repetitions}",
        ms_between=ms_between,
        ms_within=ms_within,
    )


def within_case_cv_pct(values: Iterable[float]) -> tuple[float, str]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 2:
        return math.nan, "fewer_than_2_finite_measurements"
    mean = float(np.mean(array))
    scale = max(1.0, float(np.max(np.abs(array))))
    if abs(mean) <= np.finfo(np.float64).eps * 100.0 * scale:
        return math.nan, "mean_near_zero"
    return float(np.std(array, ddof=1) / abs(mean) * 100.0), "ok"


def load_and_audit_inputs(
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, str]]:
    paths = {
        "v1_2_long_csv": input_dir / LONG_NAME,
        "v1_2_feature_summary_csv": input_dir / SUMMARY_NAME,
        "v1_2_manifest_json": input_dir / SOURCE_MANIFEST_NAME,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: sha256(path) for name, path in paths.items()}
    long_table = pd.read_csv(paths["v1_2_long_csv"])
    old_summary = pd.read_csv(paths["v1_2_feature_summary_csv"])
    source_manifest = json.loads(
        paths["v1_2_manifest_json"].read_text(encoding="utf-8")
    )
    missing = REQUIRED_LONG_COLUMNS - set(long_table.columns)
    if missing:
        raise ValueError(f"v1.2 long table is missing columns: {sorted(missing)}")
    if long_table.shape[0] != 23520:
        raise ValueError(f"expected 23520 frozen rows, got {len(long_table)}")
    if long_table["case_id"].nunique() != 28 or long_table["feature"].nunique() != 20:
        raise ValueError("expected 28 cases and 20 features")
    group_sizes = long_table.groupby(["case_id", "feature"]).size()
    if not bool((group_sizes == 42).all()):
        raise ValueError("each case-feature pair must have 42 frozen perturbation rows")
    if len(old_summary) != 20:
        raise ValueError("v1.2 feature summary must contain 20 rows")
    if source_manifest.get("stability_test_version") != (
        "method_level_mask_perturbation_v1_2_cross_family_realized_mask_dedup"
    ):
        raise ValueError("input manifest is not the frozen v1.2 analysis")
    source_hashes = source_manifest.get("source_npz_sha256", {})
    if len(source_hashes) != 111:
        raise ValueError(f"expected 111 source NPZ hashes, got {len(source_hashes)}")
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source NPZ hash mismatch: {path}")
    return long_table, old_summary, source_manifest, hashes


def audit_space_aliases(long_table: pd.DataFrame) -> dict[str, object]:
    first_feature = str(long_table["feature"].iloc[0])
    masks = long_table.loc[
        long_table["feature"] == first_feature,
        ["case_id", "perturbation_id", "realized_perturbation_id"],
    ]
    checked = 0
    for case_id, case_rows in masks.groupby("case_id", sort=True):
        indexed = case_rows.set_index("perturbation_id")
        for position in range(1, 6):
            five = f"space_5pct_position_{position}"
            ten = f"space_10pct_position_{position}"
            if indexed.loc[five, "realized_perturbation_id"] != indexed.loc[
                ten, "realized_perturbation_id"
            ]:
                raise ValueError(
                    f"space 5%/10% masks are not aliases for {case_id}, position {position}"
                )
            checked += 1
    return {
        "alias_pairs_checked": checked,
        "canonical_rule": (
            "space_block_target_10pct is excluded because every case-position actual "
            "mask is identical to space_block_target_5pct; the retained labels are "
            "interpreted by realized removal, not nominal 5%."
        ),
    }


def build_analysis_long(long_table: pd.DataFrame) -> pd.DataFrame:
    result = long_table.copy()
    result.insert(0, "record_type", "perturbation")
    result["analysis_value"] = pd.to_numeric(result["perturbed_value"], errors="coerce")
    result["analysis_family"] = result["perturbation"].map(PERTURBATION_TO_FAMILY)
    result["canonical_measurement_id"] = result["perturbation_id"].astype(str)
    result["actual_mask_dedup_status"] = "unique_retained"
    space_alias = result["perturbation"] == "space_block_target_10pct"
    result.loc[space_alias, "canonical_measurement_id"] = result.loc[
        space_alias, "perturbation_id"
    ].str.replace("space_10pct_", "space_5pct_", regex=False)
    result.loc[space_alias, "actual_mask_dedup_status"] = (
        "excluded_alias_of_space_block_target_5pct"
    )
    result["included_primary_icc"] = result["perturbation"].isin(
        PRIMARY_PERTURBATIONS
    )
    result["included_primary_cv"] = result["included_primary_icc"]
    result["included_temporal_jackknife_icc"] = (
        result["perturbation"] == "temporal_jackknife"
    )
    result["included_spatial_jackknife_icc"] = (
        result["perturbation"] == "spatial_jackknife"
    )
    null_shadow = (result["perturbation"] == "shadow_sensitivity") & (
        pd.to_numeric(
            result["realized_upstream_removed_fraction"], errors="coerce"
        ).fillna(0.0)
        == 0.0
    )
    result.loc[null_shadow, "included_primary_cv"] = False
    result.loc[null_shadow, "actual_mask_dedup_status"] = (
        "excluded_from_cv_as_duplicate_of_baseline"
    )

    baseline_source = (
        long_table.sort_values(["case_id", "feature", "perturbation_id"])
        .drop_duplicates(["case_id", "feature"])
        .copy()
    )
    baseline_columns = (
        "case_id",
        "feature",
        "feature_family",
        "statistic_type",
        "feature_domain",
        "baseline_value",
    )
    baseline = baseline_source.loc[:, baseline_columns].copy()
    baseline["record_type"] = "baseline"
    baseline["perturbation"] = "baseline"
    baseline["perturbation_id"] = "baseline"
    baseline["canonical_measurement_id"] = "baseline"
    baseline["analysis_value"] = baseline_source["baseline_value"]
    baseline["analysis_family"] = "baseline"
    baseline["status"] = np.where(
        np.isfinite(pd.to_numeric(baseline_source["baseline_value"], errors="coerce")),
        "baseline_ok",
        "DICOM_rate_unavailable",
    )
    baseline["actual_mask_dedup_status"] = "baseline_retained_for_cv_only"
    baseline["included_primary_icc"] = False
    baseline["included_primary_cv"] = True
    baseline["included_temporal_jackknife_icc"] = False
    baseline["included_spatial_jackknife_icc"] = False
    combined = pd.concat([baseline, result], ignore_index=True)
    boolean_columns = [
        "included_primary_icc",
        "included_primary_cv",
        "included_temporal_jackknife_icc",
        "included_spatial_jackknife_icc",
    ]
    combined[boolean_columns] = combined[boolean_columns].fillna(False).astype(bool)
    return combined


def evaluable_cases(analysis_long: pd.DataFrame, feature: str) -> list[str]:
    rows = analysis_long[
        (analysis_long["feature"] == feature)
        & (analysis_long["record_type"] == "baseline")
    ]
    rows = rows[np.isfinite(pd.to_numeric(rows["analysis_value"], errors="coerce"))]
    return sorted(rows["case_id"].astype(str).tolist())


def balanced_feature_matrix(
    analysis_long: pd.DataFrame,
    feature: str,
    measurement_ids: Sequence[str],
    *,
    allowed_families: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    cases = evaluable_cases(analysis_long, feature)
    rows = analysis_long[
        (analysis_long["feature"] == feature)
        & (analysis_long["record_type"] == "perturbation")
        & (analysis_long["analysis_family"].isin(allowed_families))
        & (analysis_long["canonical_measurement_id"].isin(measurement_ids))
        & (
            analysis_long["actual_mask_dedup_status"]
            != "excluded_alias_of_space_block_target_5pct"
        )
    ].copy()
    duplicate_count = int(
        rows.duplicated(["case_id", "canonical_measurement_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(f"duplicate case x measurement cells: {duplicate_count}")
    pivot = rows.pivot(
        index="case_id", columns="canonical_measurement_id", values="analysis_value"
    ).reindex(index=cases, columns=list(measurement_ids))
    if pivot.shape != (len(cases), len(measurement_ids)):
        raise ValueError("unexpected ICC matrix shape")
    if not np.all(np.isfinite(pivot.to_numpy(dtype=np.float64))):
        raise ValueError(f"incomplete ICC matrix for {feature}")
    return pivot.to_numpy(dtype=np.float64), cases


def unbalanced_primary_plus_spatial(
    analysis_long: pd.DataFrame, feature: str
) -> tuple[list[np.ndarray], list[str]]:
    cases = evaluable_cases(analysis_long, feature)
    rows = analysis_long[
        (analysis_long["feature"] == feature)
        & (analysis_long["record_type"] == "perturbation")
        & (
            analysis_long["included_primary_icc"]
            | analysis_long["included_spatial_jackknife_icc"]
        )
        & (
            analysis_long["actual_mask_dedup_status"]
            != "excluded_alias_of_space_block_target_5pct"
        )
    ].copy()
    rows = rows.sort_values(["case_id", "canonical_measurement_id"])
    rows = rows.drop_duplicates(["case_id", "realized_perturbation_id"], keep="first")
    groups: list[np.ndarray] = []
    for case_id in cases:
        values = pd.to_numeric(
            rows.loc[rows["case_id"] == case_id, "analysis_value"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid primary+spatial measurements for {feature}/{case_id}")
        groups.append(values)
    return groups, cases


def icc_columns(prefix: str, result: ICCResult) -> dict[str, object]:
    return {
        f"{prefix}_icc": result.estimate,
        f"{prefix}_ci_lower": result.ci_lower,
        f"{prefix}_ci_upper": result.ci_upper,
        f"{prefix}_n_cases": result.n_cases,
        f"{prefix}_n_measurements_min": result.n_measurements_min,
        f"{prefix}_n_measurements_max": result.n_measurements_max,
        f"{prefix}_ci_method": result.ci_method,
    }


def calculate_family_iccs(
    analysis_long: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for feature in features:
        for family, ids in FAMILY_MEASUREMENT_IDS.items():
            row: dict[str, object] = {
                "feature": feature,
                "perturbation_family": family,
                "included_measurement_ids": "|".join(ids),
            }
            if len(ids) < 2:
                row.update(
                    {
                        "estimability": "not estimable",
                        "reason": "family contains only one repeated measurement",
                        "icc_1_1": math.nan,
                        "icc_ci_lower": math.nan,
                        "icc_ci_upper": math.nan,
                        "n_cases": len(evaluable_cases(analysis_long, feature)),
                        "n_measurements": len(ids),
                        "ci_method": "not_applicable",
                    }
                )
            else:
                matrix, cases = balanced_feature_matrix(
                    analysis_long,
                    feature,
                    ids,
                    allowed_families=[family],
                )
                icc = icc_1_1(matrix)
                row.update(
                    {
                        "estimability": "estimable",
                        "reason": "",
                        "icc_1_1": icc.estimate,
                        "icc_ci_lower": icc.ci_lower,
                        "icc_ci_upper": icc.ci_upper,
                        "n_cases": len(cases),
                        "n_measurements": matrix.shape[1],
                        "ci_method": icc.ci_method,
                    }
                )
            output.append(row)
        output.append(
            {
                "feature": feature,
                "perturbation_family": "top1_removal",
                "included_measurement_ids": "feature-domain-specific top1 removal",
                "estimability": "not estimable",
                "reason": "feature-specific influence test contains one measurement",
                "icc_1_1": math.nan,
                "icc_ci_lower": math.nan,
                "icc_ci_upper": math.nan,
                "n_cases": len(evaluable_cases(analysis_long, feature)),
                "n_measurements": 1,
                "ci_method": "not_applicable",
            }
        )
    return pd.DataFrame(output)


def calculate_within_case_cv(
    analysis_long: pd.DataFrame, features: Sequence[str]
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for feature in features:
        for case_id in evaluable_cases(analysis_long, feature):
            rows = analysis_long[
                (analysis_long["feature"] == feature)
                & (analysis_long["case_id"] == case_id)
                & analysis_long["included_primary_cv"]
                & (
                    (analysis_long["record_type"] == "baseline")
                    | (analysis_long["actual_mask_dedup_status"] == "unique_retained")
                )
            ]
            values = pd.to_numeric(rows["analysis_value"], errors="coerce").to_numpy()
            cv, status = within_case_cv_pct(values)
            output.append(
                {
                    "case_id": case_id,
                    "feature": feature,
                    "n_measurements": int(np.isfinite(values).sum()),
                    "measurement_mean": float(np.nanmean(values)),
                    "measurement_sd_ddof1": float(np.nanstd(values, ddof=1)),
                    "within_case_cv_pct": cv,
                    "cv_status": status,
                    "cv_lt_10pct": bool(math.isfinite(cv) and cv < 10.0),
                    "cv_lt_20pct": bool(math.isfinite(cv) and cv < 20.0),
                }
            )
    return pd.DataFrame(output)


def calculate_sensitivity(
    analysis_long: pd.DataFrame,
    features: Sequence[str],
    *,
    bootstrap_repetitions: int,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    primary_families = [
        "random_deletion",
        "time_block",
        "space_block",
        "shadow_sensitivity",
    ]
    for feature_index, feature in enumerate(features):
        matrix_a, _ = balanced_feature_matrix(
            analysis_long,
            feature,
            PRIMARY_MEASUREMENT_IDS,
            allowed_families=primary_families,
        )
        a = icc_1_1(matrix_a)
        ids_b = PRIMARY_MEASUREMENT_IDS + FAMILY_MEASUREMENT_IDS["temporal_jackknife"]
        matrix_b, _ = balanced_feature_matrix(
            analysis_long,
            feature,
            ids_b,
            allowed_families=primary_families + ["temporal_jackknife"],
        )
        b = icc_1_1(matrix_b)
        groups_c, _ = unbalanced_primary_plus_spatial(analysis_long, feature)
        c = icc_1_1_unbalanced(
            groups_c,
            bootstrap_repetitions=bootstrap_repetitions,
            seed=BOOTSTRAP_SEED + feature_index,
        )
        row: dict[str, object] = {"feature": feature}
        row.update(icc_columns("analysis_a_primary", a))
        row.update(icc_columns("analysis_b_plus_temporal_jackknife", b))
        row.update(icc_columns("analysis_c_plus_spatial_jackknife", c))
        row.update(
            {
                "analysis_b_delta_icc_vs_a": b.estimate - a.estimate,
                "analysis_b_delta_ci_lower_vs_a": b.ci_lower - a.ci_lower,
                "analysis_b_delta_ci_upper_vs_a": b.ci_upper - a.ci_upper,
                "analysis_c_delta_icc_vs_a": c.estimate - a.estimate,
                "analysis_c_delta_ci_lower_vs_a": c.ci_lower - a.ci_lower,
                "analysis_c_delta_ci_upper_vs_a": c.ci_upper - a.ci_upper,
                "analysis_c_actual_mask_deduplication": (
                    "within each case, duplicate realized_perturbation_id values across "
                    "primary space-block and spatial jackknife rows are retained once; "
                    "unequal repeat counts are handled by one-way random-effects ANOVA "
                    "method of moments and a subject-cluster bootstrap CI"
                ),
            }
        )
        output.append(row)
    return pd.DataFrame(output)


def aggregate_feature_summary(
    analysis_long: pd.DataFrame,
    old_summary: pd.DataFrame,
    family_iccs: pd.DataFrame,
    cv_rows: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    old_by_feature = old_summary.set_index("feature_name")
    family_lookup = family_iccs.set_index(["feature", "perturbation_family"])
    sensitivity_lookup = sensitivity.set_index("feature")
    for feature in old_summary["feature_name"].astype(str):
        old = old_by_feature.loc[feature]
        primary = sensitivity_lookup.loc[feature]
        cv = cv_rows[cv_rows["feature"] == feature]
        cv_values = cv["within_case_cv_pct"].to_numpy(dtype=np.float64)
        selected = analysis_long[
            (analysis_long["feature"] == feature)
            & analysis_long["included_primary_icc"]
            & (
                analysis_long["actual_mask_dedup_status"]
                != "excluded_alias_of_space_block_target_5pct"
            )
        ]
        srd = pd.to_numeric(selected["relative_or_SRD_change"], errors="coerce")
        temporal = family_lookup.loc[(feature, "temporal_jackknife")]
        spatial = family_lookup.loc[(feature, "spatial_jackknife")]
        random_family = family_lookup.loc[(feature, "random_deletion")]
        time_family = family_lookup.loc[(feature, "time_block")]
        space_family = family_lookup.loc[(feature, "space_block")]
        temporal_srd = pd.to_numeric(
            analysis_long.loc[
                (analysis_long["feature"] == feature)
                & analysis_long["included_temporal_jackknife_icc"],
                "relative_or_SRD_change",
            ],
            errors="coerce",
        )
        spatial_srd = pd.to_numeric(
            analysis_long.loc[
                (analysis_long["feature"] == feature)
                & analysis_long["included_spatial_jackknife_icc"],
                "relative_or_SRD_change",
            ],
            errors="coerce",
        )
        estimate = float(primary["analysis_a_primary_icc"])
        lower = float(primary["analysis_a_primary_ci_lower"])
        output.append(
            {
                "feature_code": old["feature_code"],
                "feature_name": feature,
                "feature_family": old["feature_family"],
                "statistic_type": old["statistic_type"],
                "feature_domain": old["feature_domain"],
                "n_evaluable_cases": int(old["n_evaluable_cases"]),
                "primary_icc_type": (
                    "ICC(1,1): one-way random effects, single measurement, "
                    "absolute-value preservation interpretation"
                ),
                "primary_icc_1_1": estimate,
                "primary_icc_ci_lower": lower,
                "primary_icc_ci_upper": float(primary["analysis_a_primary_ci_upper"]),
                "primary_n_measurements": int(
                    primary["analysis_a_primary_n_measurements_min"]
                ),
                "primary_icc_ci_method": primary["analysis_a_primary_ci_method"],
                "icc_reference_threshold": ICC_REFERENCE_THRESHOLD,
                "primary_icc_point_ge_0_90": "Yes" if estimate >= 0.90 else "No",
                "primary_icc_ci_lower_ge_0_90": "Yes" if lower >= 0.90 else "No",
                "primary_within_case_cv_median_pct": float(np.nanmedian(cv_values)),
                "primary_within_case_cv_p90_pct": float(np.nanquantile(cv_values, 0.90)),
                "primary_within_case_cv_p95_pct": float(np.nanquantile(cv_values, 0.95)),
                "primary_within_case_cv_max_pct": float(np.nanmax(cv_values)),
                "fraction_cases_cv_lt_10pct": float(cv["cv_lt_10pct"].mean()),
                "fraction_cases_cv_lt_20pct": float(cv["cv_lt_20pct"].mean()),
                "primary_srd_median_pct": float(np.nanmedian(srd)),
                "primary_srd_p90_pct": float(np.nanquantile(srd, 0.90)),
                "primary_srd_p95_pct": float(np.nanquantile(srd, 0.95)),
                "primary_srd_max_pct": float(np.nanmax(srd)),
                "random_deletion_icc": float(random_family["icc_1_1"]),
                "random_deletion_ci_lower": float(random_family["icc_ci_lower"]),
                "random_deletion_ci_upper": float(random_family["icc_ci_upper"]),
                "time_block_icc": float(time_family["icc_1_1"]),
                "time_block_ci_lower": float(time_family["icc_ci_lower"]),
                "time_block_ci_upper": float(time_family["icc_ci_upper"]),
                "space_block_icc": float(space_family["icc_1_1"]),
                "space_block_ci_lower": float(space_family["icc_ci_lower"]),
                "space_block_ci_upper": float(space_family["icc_ci_upper"]),
                "space_block_icc_point_ge_0_90": (
                    "Yes" if float(space_family["icc_1_1"]) >= 0.90 else "No"
                ),
                "space_block_icc_ci_lower_ge_0_90": (
                    "Yes" if float(space_family["icc_ci_lower"]) >= 0.90 else "No"
                ),
                "temporal_jackknife_icc": float(temporal["icc_1_1"]),
                "temporal_jackknife_ci_lower": float(temporal["icc_ci_lower"]),
                "temporal_jackknife_ci_upper": float(temporal["icc_ci_upper"]),
                "temporal_jackknife_srd_median_pct": float(np.nanmedian(temporal_srd)),
                "temporal_jackknife_srd_p95_pct": float(np.nanquantile(temporal_srd, 0.95)),
                "spatial_jackknife_icc": float(spatial["icc_1_1"]),
                "spatial_jackknife_ci_lower": float(spatial["icc_ci_lower"]),
                "spatial_jackknife_ci_upper": float(spatial["icc_ci_upper"]),
                "spatial_jackknife_srd_median_pct": float(np.nanmedian(spatial_srd)),
                "spatial_jackknife_srd_p95_pct": float(np.nanquantile(spatial_srd, 0.95)),
                "current_v1_2_engineering_classification": old[
                    "final_classification"
                ],
                "current_v1_2_high_sensitivity_case_count": int(
                    old["high_sensitivity_case_count"]
                ),
            }
        )
    return pd.DataFrame(output)


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not math.isfinite(number) else f"{number:.{digits}f}"


def markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    family_iccs: pd.DataFrame,
    old_summary: pd.DataFrame,
    alias_audit: dict[str, object],
    analysis_long: pd.DataFrame,
) -> None:
    point_count = int((summary["primary_icc_1_1"] >= 0.90).sum())
    lower_count = int((summary["primary_icc_ci_lower"] >= 0.90).sum())
    caution = summary[
        summary["current_v1_2_engineering_classification"] == "需谨慎"
    ]
    caution_point = int((caution["primary_icc_1_1"] >= 0.90).sum())
    caution_lower = int((caution["primary_icc_ci_lower"] >= 0.90).sum())
    caution_zero = caution[
        caution["current_v1_2_high_sensitivity_case_count"] == 0
    ]
    space_point_count = int((summary["space_block_icc"] >= 0.90).sum())
    space_lower_count = int((summary["space_block_ci_lower"] >= 0.90).sum())
    space_below = summary[summary["space_block_icc"] < 0.90]
    space_indeterminate = summary[
        (summary["space_block_icc"] >= 0.90)
        & (summary["space_block_ci_lower"] < 0.90)
    ]
    spatial_point_count = int((summary["spatial_jackknife_icc"] >= 0.90).sum())
    spatial_lower_count = int((summary["spatial_jackknife_ci_lower"] >= 0.90).sum())

    readable_rows: list[list[str]] = []
    for row in summary.itertuples(index=False):
        readable_rows.append(
            [
                f"{row.feature_code} `{row.feature_name}`",
                fmt(row.primary_icc_1_1),
                f"{fmt(row.primary_icc_ci_lower)}–{fmt(row.primary_icc_ci_upper)}",
                fmt(row.primary_within_case_cv_median_pct, 2),
                fmt(row.primary_within_case_cv_p95_pct, 2),
                row.primary_icc_point_ge_0_90,
                row.primary_icc_ci_lower_ge_0_90,
                fmt(100.0 * row.fraction_cases_cv_lt_20pct, 1) + "%",
                fmt(row.spatial_jackknife_icc),
                str(row.current_v1_2_engineering_classification),
            ]
        )

    median_rows = summary[summary["statistic_type"] == "median"]
    p95_rows = summary[summary["statistic_type"] == "p95"]
    domain_rows: list[list[str]] = []
    for domain_label, domain_filter in (
        ("global", summary["feature_domain"] == "global"),
        ("anterior/posterior", summary["feature_domain"] != "global"),
    ):
        part = summary[domain_filter]
        domain_rows.append(
            [
                domain_label,
                str(len(part)),
                fmt(part["primary_icc_1_1"].median()),
                fmt(part["primary_within_case_cv_median_pct"].median(), 2),
                str(int((part["primary_icc_1_1"] >= 0.90).sum())),
            ]
        )
    family_rows: list[list[str]] = []
    for family, part in summary.groupby("feature_family", sort=False):
        family_rows.append(
            [
                str(family),
                str(len(part)),
                fmt(part["primary_icc_1_1"].median()),
                fmt(part["primary_icc_ci_lower"].median()),
                fmt(part["primary_within_case_cv_median_pct"].median(), 2),
                str(int((part["primary_icc_1_1"] >= 0.90).sum())),
                str(int((part["primary_icc_ci_lower"] >= 0.90).sum())),
            ]
        )
    sensitivity_rows: list[list[str]] = []
    for row in sensitivity.itertuples(index=False):
        sensitivity_rows.append(
            [
                str(row.feature),
                fmt(row.analysis_a_primary_icc),
                fmt(row.analysis_b_plus_temporal_jackknife_icc),
                fmt(row.analysis_b_delta_icc_vs_a),
                fmt(row.analysis_c_plus_spatial_jackknife_icc),
                fmt(row.analysis_c_delta_icc_vs_a),
            ]
        )

    space_rows = analysis_long[
        (analysis_long["feature"] == summary.iloc[0]["feature_name"])
        & (analysis_long["perturbation"] == "space_block_target_5pct")
    ]
    space_min = float(space_rows["realized_upstream_removed_fraction"].min() * 100)
    space_max = float(space_rows["realized_upstream_removed_fraction"].max() * 100)
    shadow_rows = analysis_long[
        (analysis_long["feature"] == summary.iloc[0]["feature_name"])
        & (analysis_long["perturbation"] == "shadow_sensitivity")
    ]
    shadow_null = int(
        (shadow_rows["realized_upstream_removed_fraction"].fillna(0.0) == 0.0).sum()
    )
    temporal_rows = analysis_long[
        (analysis_long["feature"] == summary.iloc[0]["feature_name"])
        & (analysis_long["perturbation"] == "temporal_jackknife")
    ]
    spatial_rows = analysis_long[
        (analysis_long["feature"] == summary.iloc[0]["feature_name"])
        & (analysis_long["perturbation"] == "spatial_jackknife")
    ]
    temporal_range = (
        float(temporal_rows["realized_upstream_removed_fraction"].min() * 100),
        float(temporal_rows["realized_upstream_removed_fraction"].max() * 100),
    )
    spatial_range = (
        float(spatial_rows["realized_upstream_removed_fraction"].min() * 100),
        float(spatial_rows["realized_upstream_removed_fraction"].max() * 100),
    )

    space_below_lines = [
        f"`{row.feature_name}` {fmt(row.space_block_icc)} "
        f"({fmt(row.space_block_ci_lower)}–{fmt(row.space_block_ci_upper)})"
        for row in space_below.itertuples(index=False)
    ]
    space_indeterminate_lines = [
        f"`{row.feature_name}` {fmt(row.space_block_icc)} "
        f"({fmt(row.space_block_ci_lower)}–{fmt(row.space_block_ci_upper)})"
        for row in space_indeterminate.itertuples(index=False)
    ]

    caution_zero_lines = [
        f"- `{row.feature_name}`：ICC {fmt(row.primary_icc_1_1)} "
        f"({fmt(row.primary_icc_ci_lower)}–{fmt(row.primary_icc_ci_upper)})，"
        f"median CV {fmt(row.primary_within_case_cv_median_pct, 2)}%，"
        f"CV<20%病例 {fmt(100 * row.fraction_cases_cv_lt_20pct, 1)}%。"
        for row in caution_zero.itertuples(index=False)
    ]
    if not caution_zero_lines:
        caution_zero_lines = ["- 无。"]

    curvature = summary[summary["feature_family"] == "curvature"]
    curvature_primary_median = float(curvature["primary_icc_1_1"].median())
    curvature_spatial_median = float(curvature["spatial_jackknife_icc"].median())
    text = f"""# 正式候选特征文献标准鲁棒性分析报告

## 结论先行

在预先确定的 primary 小扰动集合下，20 项中有 **{point_count} 项** ICC(1,1) 点估计达到 0.90 文献参考线，**{lower_count} 项**的 95%CI 下限达到 0.90。这里不生成新的“稳定/需谨慎/不稳定”分类；v1.2 工程结果仅作为历史对照。

当前 v1.2 的 14 项“需谨慎”特征中，**{caution_point} 项** ICC 点估计达到 0.90，**{caution_lower} 项**的 95%CI 下限达到 0.90。结果说明 SRD 病例计数与患者间可靠性回答的是不同问题，不能互相替代。

**不能把 20/20 的合并 primary ICC 解释成每个扰动族都达到参考线。** random deletion 与 time block 均为 20/20 点估计及 CI 下限≥0.90；space block 只有 **{space_point_count}/20** 点估计≥0.90、**{space_lower_count}/20** CI 下限≥0.90。space-block 点估计低于 0.90 的是 {', '.join(space_below_lines)}；点估计达到但 CI 下限未达到的还有 {', '.join(space_indeterminate_lines)}。合并 primary ICC 中 random/time 重复数较多，不能替代最弱 family 的单独结果。

## 20 项主结果

{markdown_table(readable_rows, ["Feature", "Primary ICC", "95%CI", "Median CV (%)", "P95 CV (%)", "ICC点≥0.90", "CI下限≥0.90", "CV<20%病例比例", "Spatial jackknife ICC", "v1.2旧工程结果"])}

## ICC 统计设计审计

1. **subject（受试对象）**：患者/病例；不是 23,520 行、frame 或 pair。
2. **repeated measurement（重复测量）**：同一特征在不同实际扰动 mask 下重新汇总得到的患者级数值。
3. **病例数**：RSR、cavity、longitudinal 为 28；6 个 DICOM physical curvature-rate 特征为 27，缺失病例保持 NaN，不填 0、不插补。
4. **primary 重复数**：每病例 26 个扰动值；baseline 不进入 primary ICC，只进入 CV。
5. **ICC 形式**：ICC(1,1)，one-way random-effects（一元随机效应）、single measurement（单次测量）。一元模型不单列固定“测量者效应”；本报告按保持绝对数值的 reliability/agreement 目的解释，而不只解释排序一致。
6. **选择原因**：与 Zwanenburg 等的 perturbation ICC 形式一致，并把不同随机删除/区块位置视为可能扰动集合中的单次实现。
7. **一致之处**：均以患者为独立单位、以多次人为扰动为重复测量、用患者间方差相对于患者内扰动方差量化可靠性，并报告 95%CI。
8. **不可等同之处**：Zwanenburg 研究 CT 的噪声、平移、旋转、体积和轮廓随机化，并以 test–retest 作参照；本项目只对冻结的超声患者级特征做 mask/data-loss 扰动，没有真实复扫、独立操作者、独立设备或独立 session。

ICC 点估计使用一元随机效应 ANOVA：`(MS_between - MS_within) / (MS_between + (k-1) MS_within)`。平衡矩阵的 95%CI 使用中心 F 分布反演。阈值判断同时保留点估计是否≥0.90和 CI 下限是否≥0.90两个纯描述字段。

## actual-mask 去重与实际扰动强度

- 审计了 {alias_audit['alias_pairs_checked']} 对病例-空间位置别名。所有病例中，nominal space 5% 与 10% 在相同位置均产生完全相同的 actual mask，因此 primary 与 space-family ICC 全局只保留 5 个 canonical 列，不重复加权。
- 保留列的实际空间 section 删除比例为 **{space_min:.1f}%–{space_max:.1f}%**，所以不能按 nominal 5% 解释。
- primary 固定矩阵为：random deletion 10 列 + time block 10 列 + canonical space block 5 列 + shadow 1 列 = 26 列。各患者、各特征使用相同列；病例内 actual mask 无重复。
- shadow sensitivity 在 {shadow_null}/28 例没有删除任何位置；它在 perturbation ICC 中如实作为一次零改变实现，但 CV 中不与 baseline 重复计权。
- 不同病例即使碰巧拥有相同 mask hash，仍保持为不同 subject。

## CV 定义

本报告的 CV 是 **within-case CV（病例内变异系数）**：每位患者、每个特征在 `baseline + primary canonical perturbations` 中，以样本标准差（ddof=1）除以测量均值绝对值，再乘 100%。病例内 null-shadow 与 baseline 完全相同时只保留 baseline 一次；NaN 仅按预先冻结的 DICOM 不可评价病例处理。若均值数值上接近 0，则 CV fail closed 为 NaN；本批正值特征没有触发该保护。

CV<10% 与 CV<20%只作为文献参考线。特别是 CV<20%来自超声 phantom/scanner/scan-setting 场景，不能称为子宫内膜超声临床阈值。

## 强压力测试敏感性

Analysis A 是 26 列 primary；Analysis B 在其上加入 5 个 temporal jackknife（实际删除 {temporal_range[0]:.1f}%–{temporal_range[1]:.1f}%）；Analysis C 加入 spatial jackknife（实际删除 {spatial_range[0]:.1f}%–{spatial_range[1]:.1f}%）后按 `case_id + realized_perturbation_id` 再去重。由于部分 spatial jackknife 与 primary space mask 在某些病例重合，Analysis C 的每病例 unique measurement 数不同，点估计使用不等重复数的一元随机效应 method-of-moments，95%CI 使用固定种子的患者整群 percentile bootstrap；不能把它的 CI 精度与平衡 F 区间完全等同。

加入 5 个压力测量后的合并 ΔICC 会被原有 26 个 primary 测量稀释，因此还必须看单独 stress-family ICC：temporal jackknife 为 20/20 点估计≥0.90、16/20 CI 下限≥0.90；spatial jackknife 为 **{spatial_point_count}/20** 点估计≥0.90、**{spatial_lower_count}/20** CI 下限≥0.90。

{markdown_table(sensitivity_rows, ["Feature", "A ICC", "B + temporal", "ΔB-A", "C + spatial", "ΔC-A"])}

## 八个项目问题

1. v1.2 只有 2 项为“稳定”；本轮有 **{point_count}/20** 的 primary ICC 点估计≥0.90，**{lower_count}/20** 的 CI 下限≥0.90。
2. 14 项 v1.2“需谨慎”中，**{caution_point} 项**点估计≥0.90。
3. 14 项 v1.2“需谨慎”中，**{caution_lower} 项**CI 下限≥0.90。
4. “需谨慎”且 0 个高敏感病例的特征如下；只描述 ICC/CV，不改旧分类：
{chr(10).join(caution_zero_lines)}
5. median 组的 primary ICC 中位数为 **{fmt(median_rows['primary_icc_1_1'].median())}**，病例内 CV 中位数的特征间中位数为 **{fmt(median_rows['primary_within_case_cv_median_pct'].median(), 2)}%**；P95 组分别为 **{fmt(p95_rows['primary_icc_1_1'].median())}** 与 **{fmt(p95_rows['primary_within_case_cv_median_pct'].median(), 2)}%**。这是 10 对特征的描述性比较，不做显著性筛选。
6. global 与 anterior/posterior 的描述性对照：

{markdown_table(domain_rows, ["domain", "n_features", "median ICC", "median of feature median-CV (%)", "ICC点≥0.90数"])}

7. curvature 的合并 primary ICC 特征间中位数为 **{fmt(curvature_primary_median)}**，但空间子族已经显示前壁曲率 median/P95 的 ICC 仅为 **{fmt(summary.loc[summary['feature_code'] == 'F17', 'space_block_icc'].iloc[0])}** 和 **{fmt(summary.loc[summary['feature_code'] == 'F18', 'space_block_icc'].iloc[0])}**；单独 spatial jackknife 对应为 **{fmt(summary.loc[summary['feature_code'] == 'F17', 'spatial_jackknife_icc'].iloc[0])}** 和 **{fmt(summary.loc[summary['feature_code'] == 'F18', 'spatial_jackknife_icc'].iloc[0])}**。因此空间敏感性在 primary 的实际 11.1%–14.3% section 删除中已经出现，不能说完全由额外 jackknife 才产生；更准确的解释是 **curvature 尤其前壁 curvature 对空间区域删除敏感，而对 random/time 扰动仍很高**。不据此创建标签。
8. 四个 family 的描述性汇总：

{markdown_table(family_rows, ["family", "n", "median ICC", "median CI lower", "median feature-CV (%)", "ICC点≥0.90数", "CI下限≥0.90数"])}

## 文献依据与阈值边界

- [Zwanenburg et al., Scientific Reports 2019](https://www.nature.com/articles/s41598-018-36938-4)：人为图像扰动、ICC(1,1)、95%CI 与 0.90 参考线。原文将 CI 整体位于 0.90 以上视为 robust，整体低于阈值视为 non-robust，跨线为 indeterminate；本报告只输出阈值是否达到，不移植其三分类名称。
- [Koo & Li 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/)：要求报告 model、type、definition 和 95%CI；>0.90 常被解释为 excellent，但该解释不是唯一国际强制标准。
- [Soleymani et al. 2022](https://pubmed.ncbi.nlm.nih.gov/36266173/)：超声 phantom 在不同扫描设置/设备下使用 ICC>0.90 与 CV<20%作为 reproducibility 参考。其对象和扰动机制与本项目不同。
- [Traverso et al. systematic review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6690209/)：radiomics 重复性/再现性研究的指标和阈值并不统一，支持把 0.90 写成预选的严格文献参考线，而不是“唯一国际标准”。

## 科学边界

本分析只能称为 **perturbation-based feature robustness（基于扰动的特征鲁棒性）**。它不是 true test–retest repeatability、inter-operator reproducibility、inter-device reproducibility、inter-session reproducibility、external validation 或 clinical predictive validity。ICC≥0.90 与 CV<20%均为文献参考阈值，不是经验证的子宫内膜临床 cutoff。SRD继续用于描述实际相对数值变化，不再生成新的特征分类。
"""
    path.write_text(text, encoding="utf-8")


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    *,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Path]:
    long_table, old_summary, source_manifest, frozen_hashes_before = (
        load_and_audit_inputs(input_dir)
    )
    formal_feature_code = (
        PROJECT_ROOT
        / "代码"
        / "01_底层算法"
        / "peristalsis_pipeline"
        / "formal_feature_extraction.py"
    )
    feature_code_hash_before = sha256(formal_feature_code)
    alias_audit = audit_space_aliases(long_table)
    analysis_long = build_analysis_long(long_table)
    features = old_summary["feature_name"].astype(str).tolist()

    primary = analysis_long[
        analysis_long["included_primary_icc"]
        & (
            analysis_long["actual_mask_dedup_status"]
            != "excluded_alias_of_space_block_target_5pct"
        )
    ]
    first_feature = features[0]
    primary_masks = primary[primary["feature"] == first_feature]
    duplicate_masks = primary_masks.duplicated(
        ["case_id", "realized_perturbation_id"]
    )
    if bool(duplicate_masks.any()):
        raise ValueError("primary ICC contains a repeated actual mask within a case")
    counts = primary_masks.groupby("case_id").size()
    if not bool((counts == len(PRIMARY_MEASUREMENT_IDS)).all()):
        raise ValueError("primary ICC does not have a fixed 26-measurement matrix")

    family_iccs = calculate_family_iccs(analysis_long, features)
    cv_rows = calculate_within_case_cv(analysis_long, features)
    sensitivity = calculate_sensitivity(
        analysis_long,
        features,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    summary = aggregate_feature_summary(
        analysis_long,
        old_summary,
        family_iccs,
        cv_rows,
        sensitivity,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": output_dir / "literature_aligned_feature_robustness_summary.csv",
        "long": output_dir / "literature_aligned_feature_robustness_long.csv",
        "family_icc": output_dir / "perturbation_family_icc_summary.csv",
        "within_case_cv": output_dir / "within_case_cv_summary.csv",
        "jackknife_sensitivity": output_dir / "jackknife_sensitivity_comparison.csv",
        "manifest": output_dir / "literature_aligned_robustness_manifest.json",
        "report": output_dir / "正式候选特征文献标准鲁棒性分析报告.md",
        "source_snapshot": output_dir / "新增代码与测试完整内容.md",
    }
    summary.to_csv(outputs["summary"], index=False, encoding="utf-8-sig")
    analysis_long.to_csv(outputs["long"], index=False, encoding="utf-8-sig")
    family_iccs.to_csv(outputs["family_icc"], index=False, encoding="utf-8-sig")
    cv_rows.to_csv(outputs["within_case_cv"], index=False, encoding="utf-8-sig")
    sensitivity.to_csv(
        outputs["jackknife_sensitivity"], index=False, encoding="utf-8-sig"
    )
    write_report(
        outputs["report"],
        summary,
        sensitivity,
        family_iccs,
        old_summary,
        alias_audit,
        analysis_long,
    )
    test_path = (
        PROJECT_ROOT
        / "测试代码"
        / "01_底层算法测试"
        / "test_literature_aligned_feature_robustness.py"
    )
    outputs["source_snapshot"].write_text(
        "# 新增代码与测试完整内容\n\n"
        "普通 `git diff` 为空，因为新增文件尚未加入 Git 索引。以下保存两个本轮新增文件的完整内容。\n\n"
        f"## `{Path(__file__).relative_to(PROJECT_ROOT)}`\n\n"
        "```python\n"
        + Path(__file__).read_text(encoding="utf-8")
        + "\n```\n\n"
        f"## `{test_path.relative_to(PROJECT_ROOT)}`\n\n"
        "```python\n"
        + test_path.read_text(encoding="utf-8")
        + "\n```\n",
        encoding="utf-8",
    )

    frozen_paths = {
        "v1_2_long_csv": input_dir / LONG_NAME,
        "v1_2_feature_summary_csv": input_dir / SUMMARY_NAME,
        "v1_2_manifest_json": input_dir / SOURCE_MANIFEST_NAME,
    }
    frozen_hashes_after = {name: sha256(path) for name, path in frozen_paths.items()}
    source_hashes = source_manifest["source_npz_sha256"]
    source_npz_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == expected
        for path, expected in source_hashes.items()
    )
    feature_code_hash_after = sha256(formal_feature_code)
    output_hashes = {
        name: sha256(path) for name, path in outputs.items() if name != "manifest"
    }
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_date": "2026-09-05",
        "analysis_scope": "perturbation-based feature robustness",
        "not_claimed": [
            "true test-retest repeatability",
            "inter-operator reproducibility",
            "inter-device reproducibility",
            "inter-session reproducibility",
            "external validation",
            "clinical predictive validity",
        ],
        "input_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "frozen_input_shape": list(long_table.shape),
        "n_cases": int(long_table["case_id"].nunique()),
        "n_features": int(long_table["feature"].nunique()),
        "icc": {
            "form": "ICC(1,1)",
            "model": "one-way random effects",
            "type": "single measurement",
            "definition": (
                "reliability under the no-consistent-bias one-way model, interpreted "
                "for absolute-value preservation rather than rank-only consistency"
            ),
            "subject": "patient/case",
            "repeated_measurement": (
                "same feature value under a distinct realized perturbation mask"
            ),
            "balanced_ci": "exact central-F 95% interval",
            "unbalanced_sensitivity_ci": (
                f"subject-cluster percentile bootstrap, {bootstrap_repetitions} repeats"
            ),
            "reference_threshold": ICC_REFERENCE_THRESHOLD,
            "reference_threshold_semantics": (
                "preselected literature reference threshold; not an endometrial "
                "ultrasound validated clinical cutoff"
            ),
        },
        "primary_measurement_matrix": {
            "baseline_included_in_icc": False,
            "n_measurements": len(PRIMARY_MEASUREMENT_IDS),
            "measurement_ids": list(PRIMARY_MEASUREMENT_IDS),
            "n_cases_non_curvature": 28,
            "n_cases_curvature": 27,
            "complete_and_finite_within_each_feature": True,
            "actual_mask_duplicate_count_within_case": 0,
        },
        "actual_mask_deduplication": alias_audit,
        "cv": {
            "form": "within-case sample SD / absolute mean * 100%",
            "includes_baseline": True,
            "includes_primary_canonical_perturbations": True,
            "null_shadow_duplicate_of_baseline_removed": True,
            "nan_handling": "feature-specific non-evaluable case excluded; no imputation",
            "near_zero_mean_handling": "return NaN and cv_status=mean_near_zero",
            "reference_lines_pct": list(CV_REFERENCE_THRESHOLDS_PCT),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "third_party_icc_package": None,
            "implementation": "self-contained ANOVA formula with scipy.stats.f CI",
        },
        "integrity": {
            "source_npz_count": len(source_hashes),
            "source_npz_sha256_unchanged": source_npz_unchanged,
            "frozen_input_sha256_before": frozen_hashes_before,
            "frozen_input_sha256_after": frozen_hashes_after,
            "frozen_inputs_unchanged": frozen_hashes_before == frozen_hashes_after,
            "formal_feature_extraction_sha256_before": feature_code_hash_before,
            "formal_feature_extraction_sha256_after": feature_code_hash_after,
            "formal_feature_extraction_unchanged": (
                feature_code_hash_before == feature_code_hash_after
            ),
            "tracking_p3_qc_executed": False,
        },
        "output_sha256_excluding_manifest": output_hashes,
    }
    outputs["manifest"].write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_analysis(
        args.input_dir,
        args.output_dir,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print("literature-aligned robustness analysis complete")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
