"""Paper 1 normalized-time temporal-structure robustness analysis.

The analysis reuses the frozen patient-level source arrays and the same pending
Grade-3 mask-only intervention as run_tracking_quality_sensitivity.py. It does
not rerun tracking, alter QC, interpolate time, or infer peristaltic waves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ENTRY = Path(__file__).resolve().parent
sys.path.insert(0, str(ENTRY))
sys.path.insert(0, str(PROJECT / "代码/01_底层算法"))

import run_tracking_quality_sensitivity as quality
from peristalsis_pipeline.feature_stability import (
    FEATURE_SPECS,
    compute_feature_row,
    symmetric_relative_difference_pct,
)
from peristalsis_pipeline.temporal_stability import (
    DEFAULT_TEMPORAL_BINS,
    bin_srd_values,
    summarize_profile_pair,
    temporal_profile,
)


ROOT = quality.ROOT
DEFAULT_OUT = ROOT / "全患者汇总/时间结构稳定性_20260920"


def write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def analyze_case(folder: Path, out: Path, bins: int) -> dict[str, object]:
    case_out = out / "patients" / folder.name
    complete = case_out / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    case_out.mkdir(parents=True, exist_ok=True)

    data, tracking, artifact, _ = quality.load_case(folder)
    pool = np.flatnonzero(np.any(np.isfinite(data.rsr), axis=(1, 2)))
    pool = pool[pool > 0]
    pending = np.flatnonzero(artifact["automatic_grade3_pending_manual_review"])
    removed = np.intersect1d(pool, pending)
    shadow = quality.exclude_frames(data, removed)

    original_feature_row = compute_feature_row(data)
    shadow_feature_row = compute_feature_row(shadow)

    summary_rows = []
    profile_rows = []
    for feature_index, spec in enumerate(FEATURE_SPECS, 1):
        original_profile, original_counts = temporal_profile(data, spec, bins=bins)
        shadow_profile, shadow_counts = temporal_profile(shadow, spec, bins=bins)
        temporal = summarize_profile_pair(
            original_profile, shadow_profile, original_counts, shadow_counts
        )
        per_bin_srd = bin_srd_values(original_profile, shadow_profile)

        aggregate_original = float(original_feature_row[spec.name])
        aggregate_shadow = float(shadow_feature_row[spec.name])
        aggregate_srd = symmetric_relative_difference_pct(
            aggregate_original, aggregate_shadow
        )
        summary_rows.append(
            dict(
                case_id=folder.name,
                feature_id=f"F{feature_index:02}",
                feature=spec.name,
                family=spec.family,
                domain=spec.domain,
                statistic=spec.statistic_type,
                aggregate_original=aggregate_original,
                aggregate_shadow=aggregate_shadow,
                aggregate_srd_pct=aggregate_srd,
                **temporal,
                temporal_minus_aggregate_median_srd_pct=(
                    temporal["median_bin_srd_pct"] - aggregate_srd
                    if np.isfinite(temporal["median_bin_srd_pct"])
                    and np.isfinite(aggregate_srd)
                    else np.nan
                ),
                pending_grade3_frames=int(len(pending)),
                removed_frames=int(len(removed)),
                eligible_frames=int(len(pool)),
                grade3_frame_fraction=(
                    len(removed) / len(pool) if len(pool) else np.nan
                ),
                has_pending_grade3=bool(len(pending)),
                grade3_effective=bool(len(removed)),
                topology_risk=bool(tracking["has_tracking_topology_risk"]),
                physical_calibration=bool(
                    data.physical_curvature_rate_available
                ),
            )
        )

        for bin_index in range(bins):
            original_finite = bool(np.isfinite(original_profile[bin_index]))
            shadow_finite = bool(np.isfinite(shadow_profile[bin_index]))
            if shadow_finite and not original_finite:
                raise RuntimeError(
                    f"{folder.name} {spec.name} bin {bin_index + 1}: "
                    "mask-only analysis created NaN-to-finite"
                )
            availability = (
                "paired_finite"
                if original_finite and shadow_finite
                else "finite_to_nan"
                if original_finite
                else "both_nan"
            )
            profile_rows.append(
                dict(
                    case_id=folder.name,
                    feature_id=f"F{feature_index:02}",
                    feature=spec.name,
                    family=spec.family,
                    domain=spec.domain,
                    statistic=spec.statistic_type,
                    bin_index=bin_index + 1,
                    normalized_time_start=bin_index / bins,
                    normalized_time_end=(bin_index + 1) / bins,
                    original=original_profile[bin_index],
                    grade3_shadow=shadow_profile[bin_index],
                    srd_pct=per_bin_srd[bin_index],
                    original_valid_positions=int(original_counts[bin_index]),
                    shadow_valid_positions=int(shadow_counts[bin_index]),
                    valid_position_loss_n=int(
                        original_counts[bin_index] - shadow_counts[bin_index]
                    ),
                    availability_status=availability,
                    finite_to_nan=availability == "finite_to_nan",
                )
            )

    write_csv(case_out / "temporal_summary.csv", summary_rows)
    write_csv(case_out / "temporal_profiles.csv", profile_rows)
    meta = dict(
        case_id=folder.name,
        bins=int(bins),
        frame_count=int(len(data.rsr)),
        fps=float(data.rsr_fps),
        pending_grade3_frames=int(len(pending)),
        removed_frames=int(len(removed)),
        eligible_frames=int(len(pool)),
        topology_risk=bool(tracking["has_tracking_topology_risk"]),
        analysis=(
            "fixed normalized-time bins; same feature domain/statistic as F01-F20; "
            "no interpolation or temporal resampling"
        ),
    )
    quality.save_json(complete, meta)
    return meta


def summarize(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strata = {
        "all": np.ones(len(long), dtype=bool),
        "pending_grade3": long.has_pending_grade3.to_numpy(bool),
        "effective_grade3": long.grade3_effective.to_numpy(bool),
        "topology_yes": long.topology_risk.to_numpy(bool),
        "topology_no": ~long.topology_risk.to_numpy(bool),
    }
    for stratum, mask in strata.items():
        for feature, group in long.loc[mask].groupby("feature", sort=False):
            rho = pd.to_numeric(
                group.temporal_spearman, errors="coerce"
            ).dropna()
            median_srd = pd.to_numeric(
                group.median_bin_srd_pct, errors="coerce"
            ).dropna()
            p95_srd = pd.to_numeric(
                group.p95_bin_srd_pct, errors="coerce"
            ).dropna()
            retention = pd.to_numeric(
                group.valid_position_retention_fraction, errors="coerce"
            ).dropna()
            aggregate_srd = pd.to_numeric(
                group.aggregate_srd_pct, errors="coerce"
            ).dropna()
            gap = pd.to_numeric(
                group.temporal_minus_aggregate_median_srd_pct,
                errors="coerce",
            ).dropna()
            rows.append(
                dict(
                    stratum=stratum,
                    feature_id=group.feature_id.iloc[0],
                    feature=feature,
                    family=group.family.iloc[0],
                    domain=group.domain.iloc[0],
                    statistic=group.statistic.iloc[0],
                    n_cases=int(len(group)),
                    n_temporal_spearman_available=int(len(rho)),
                    median_temporal_spearman=(
                        rho.median() if len(rho) else np.nan
                    ),
                    p05_temporal_spearman=(
                        rho.quantile(0.05) if len(rho) else np.nan
                    ),
                    median_valid_position_retention_fraction=(
                        retention.median() if len(retention) else np.nan
                    ),
                    finite_to_nan_bins_total=int(group.finite_to_nan_bins.sum()),
                    cases_with_finite_to_nan_bins=int(
                        (group.finite_to_nan_bins > 0).sum()
                    ),
                    median_case_median_bin_srd_pct=(
                        median_srd.median() if len(median_srd) else np.nan
                    ),
                    p95_case_median_bin_srd_pct=(
                        median_srd.quantile(0.95)
                        if len(median_srd)
                        else np.nan
                    ),
                    median_case_p95_bin_srd_pct=(
                        p95_srd.median() if len(p95_srd) else np.nan
                    ),
                    median_aggregate_srd_pct=(
                        aggregate_srd.median()
                        if len(aggregate_srd)
                        else np.nan
                    ),
                    median_temporal_minus_aggregate_srd_pct=(
                        gap.median() if len(gap) else np.nan
                    ),
                    interpretation=(
                        "描述性时间结构鲁棒性；不设稳定/不稳定阈值"
                    ),
                )
            )
    return pd.DataFrame(rows)


def aggregate(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_files = sorted((out / "patients").glob("*/temporal_summary.csv"))
    if not case_files:
        raise RuntimeError("no completed temporal stability cases")
    long = pd.concat([pd.read_csv(path) for path in case_files], ignore_index=True)
    profiles = pd.concat(
        [
            pd.read_csv(path.parent / "temporal_profiles.csv")
            for path in case_files
        ],
        ignore_index=True,
    )
    write_csv(out / "患者级时间结构比较_长表.csv", long)
    write_csv(out / "归一化时间分箱曲线_长表.csv", profiles)
    summary = summarize(long)
    write_csv(out / "Paper1_时间结构稳定性正式汇总.csv", summary)
    return long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=DEFAULT_TEMPORAL_BINS)
    parser.add_argument("--expected-cases", type=int, default=319)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.bins < 2:
        raise ValueError("temporal bins must be at least 2")
    if args.output.resolve() == args.root.resolve():
        raise ValueError("output must be separate from the source root")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("nonempty output; use explicit --resume")
    args.output.mkdir(parents=True, exist_ok=True)

    cases = sorted(
        p
        for p in args.root.iterdir()
        if p.is_dir() and (p / "04_患者特征/患者特征.csv").is_file()
    )
    if args.limit:
        cases = cases[: args.limit]
    elif args.expected_cases is not None and len(cases) != args.expected_cases:
        raise ValueError(
            f"Paper 1 temporal stability expected {args.expected_cases} cases, "
            f"found {len(cases)}; do not silently change the frozen cohort"
        )

    extractor = quality.official_extractor()
    checks = []
    source_hashes = {}
    for index, folder in enumerate(cases, 1):
        rows, hashes = quality.original_check(folder, extractor)
        checks.extend(rows)
        source_hashes.update(hashes)
        if index % 25 == 0 or index == len(cases):
            print(f"Original check {index}/{len(cases)}", flush=True)
    write_csv(args.output / "Original复现检查.csv", checks)
    if not all(row["passed"] for row in checks):
        raise RuntimeError("Original reproduction failed; temporal analysis stopped")

    contract = dict(
        cases=[folder.name for folder in cases],
        expected_cases=args.expected_cases,
        bins=int(args.bins),
        normalized_time_definition=(
            "frame_index/frame_count partitioned into equal-width bins"
        ),
        profile_definition=(
            "same absolute-value statistic and anatomical domain as F01-F20 "
            "within each normalized-time bin"
        ),
        interpolation=False,
        temporal_resampling=False,
        clinical_labels_used=False,
        intervention=(
            "same effective pending Grade-3 whole-frame mask-only shadow as "
            "run_tracking_quality_sensitivity.py; frame 0 excluded from removal pool"
        ),
        source_sha256=source_hashes,
        code_sha256={
            str(Path(__file__).resolve()): quality.sha(__file__),
            str(
                PROJECT
                / "代码/01_底层算法/peristalsis_pipeline/temporal_stability.py"
            ): quality.sha(
                PROJECT
                / "代码/01_底层算法/peristalsis_pipeline/temporal_stability.py"
            ),
            str(Path(quality.__file__).resolve()): quality.sha(quality.__file__),
        },
    )
    manifest = args.output / "source_manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("resume source/code/config identity changed")
    quality.save_json(manifest, contract)

    for index, folder in enumerate(cases, 1):
        print(f"Temporal stability {index}/{len(cases)}: {folder.name}", flush=True)
        analyze_case(folder, args.output, args.bins)

    long, summary = aggregate(args.output)

    changed = [
        path for path, digest in source_hashes.items() if quality.sha(path) != digest
    ]
    quality.save_json(
        args.output / "source_integrity.json",
        dict(files=len(source_hashes), unchanged=not changed, changed=changed),
    )
    if changed:
        raise RuntimeError("source integrity mismatch")

    pending = long[long.has_pending_grade3]
    report = f"""# Paper 1 时间结构稳定性分析

本轮使用 {long.case_id.nunique()} 例冻结结果，在不重新追踪、不改变正式 QC、
不读取临床结局的前提下，比较 Original 与同一 pending Grade-3 mask-only shadow。

时间轴按每例原始帧顺序归一化到 0-1，并固定分为 {args.bins} 个等比例时间段。
每个时间段内使用与对应 F01-F20 完全相同的解剖域、绝对值和 median/P95 统计。
没有按固定秒数截取，没有插值、补帧、跨缺失段拼接或时间重采样。

输出同时报告：时间段可计算性、有效位置保留比例、逐时间段 SRD、病例内时间曲线
Spearman，以及患者级聚合 SRD。后两者并列保存，用于识别“聚合值变化小，但局部
时间段变化相对更明显”的情况；本程序不设置稳定/不稳定阈值，也不自动下结论。

有 pending Grade-3 的病例数：{pending.case_id.nunique()}。
完整逐病例结果见“患者级时间结构比较_长表.csv”，逐时间段曲线见
“归一化时间分箱曲线_长表.csv”，跨病例描述见“Paper1_时间结构稳定性正式汇总.csv”。

本分析仅评价现有局部动态测量在预定义 QC 屏蔽下的时间结构保持程度。
它不能证明蠕动波、传播方向、波速、频率或生理真实性。
"""
    (args.output / "分析报告.md").write_text(report, encoding="utf-8")
    print(summary[summary.stratum.eq("pending_grade3")].to_string(index=False))
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
