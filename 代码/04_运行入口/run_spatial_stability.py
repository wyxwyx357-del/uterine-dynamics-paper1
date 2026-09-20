"""Paper 1 normalized-position spatial-structure robustness analysis.

The analysis reuses frozen per-patient arrays and the same pending Grade-3
whole-frame mask-only intervention as the temporal analysis. It does not rerun
tracking, alter QC, interpolate space, or infer peristaltic propagation.
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
from peristalsis_pipeline.spatial_stability import (
    DEFAULT_SPATIAL_BINS,
    spatial_bin_srd_values,
    spatial_profile,
    summarize_spatial_profile_pair,
)


ROOT = quality.ROOT
DEFAULT_OUT = ROOT / "全患者汇总/空间结构稳定性_20260920"


def write_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def load_normalized_position(paths: dict[str, Path]) -> np.ndarray:
    anatomical = quality.npz(paths["anatomical"])
    if "normalized_cervix_to_fundus_position" not in anatomical:
        raise KeyError(
            f"{paths['anatomical']}: missing normalized_cervix_to_fundus_position"
        )
    position = np.asarray(
        anatomical["normalized_cervix_to_fundus_position"], dtype=np.float64
    )
    if position.ndim != 1:
        raise ValueError("normalized_cervix_to_fundus_position must be 1D")
    return position


def analyze_case(folder: Path, out: Path, bins: int) -> dict[str, object]:
    case_out = out / "patients" / folder.name
    complete = case_out / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    case_out.mkdir(parents=True, exist_ok=True)

    data, tracking, artifact, paths = quality.load_case(folder)
    normalized_position = load_normalized_position(paths)
    if len(normalized_position) != data.rsr.shape[2]:
        raise ValueError(
            f"{folder.name}: normalized position length does not match RSR section axis"
        )

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
        original_profile, original_counts = spatial_profile(
            data, spec, normalized_position, bins=bins
        )
        shadow_profile, shadow_counts = spatial_profile(
            shadow, spec, normalized_position, bins=bins
        )
        spatial = summarize_spatial_profile_pair(
            original_profile, shadow_profile, original_counts, shadow_counts
        )
        per_bin_srd = spatial_bin_srd_values(original_profile, shadow_profile)

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
                **spatial,
                spatial_minus_aggregate_median_srd_pct=(
                    spatial["median_bin_srd_pct"] - aggregate_srd
                    if np.isfinite(spatial["median_bin_srd_pct"])
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
                physical_calibration=bool(data.physical_curvature_rate_available),
            )
        )

        coordinate_definition = (
            "adjacent-section midpoint"
            if spec.family == "longitudinal"
            else "frozen section coordinate"
        )
        for bin_index in range(bins):
            original_finite = bool(np.isfinite(original_profile[bin_index]))
            shadow_finite = bool(np.isfinite(shadow_profile[bin_index]))
            if shadow_finite and not original_finite:
                raise RuntimeError(
                    f"{folder.name} {spec.name} spatial bin {bin_index + 1}: "
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
                    normalized_position_start=bin_index / bins,
                    normalized_position_end=(bin_index + 1) / bins,
                    coordinate_definition=coordinate_definition,
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

    write_csv(case_out / "spatial_summary.csv", summary_rows)
    write_csv(case_out / "spatial_profiles.csv", profile_rows)
    meta = dict(
        case_id=folder.name,
        bins=int(bins),
        frame_count=int(len(data.rsr)),
        fps=float(data.rsr_fps),
        section_count=int(data.rsr.shape[2]),
        pending_grade3_frames=int(len(pending)),
        removed_frames=int(len(removed)),
        eligible_frames=int(len(pool)),
        topology_risk=bool(tracking["has_tracking_topology_risk"]),
        analysis=(
            "fixed normalized cervix-to-fundus bins; same feature domain/statistic "
            "as F01-F20; longitudinal positions use adjacent-section midpoints; "
            "no spatial interpolation or resampling"
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
            rho = pd.to_numeric(group.spatial_spearman, errors="coerce").dropna()
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
                group.spatial_minus_aggregate_median_srd_pct,
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
                    n_spatial_spearman_available=int(len(rho)),
                    median_spatial_spearman=(rho.median() if len(rho) else np.nan),
                    p05_spatial_spearman=(
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
                        median_srd.quantile(0.95) if len(median_srd) else np.nan
                    ),
                    median_case_p95_bin_srd_pct=(
                        p95_srd.median() if len(p95_srd) else np.nan
                    ),
                    median_aggregate_srd_pct=(
                        aggregate_srd.median() if len(aggregate_srd) else np.nan
                    ),
                    median_spatial_minus_aggregate_srd_pct=(
                        gap.median() if len(gap) else np.nan
                    ),
                    interpretation="描述性空间结构鲁棒性；不设稳定/不稳定阈值",
                )
            )
    return pd.DataFrame(rows)


def aggregate(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_files = sorted((out / "patients").glob("*/spatial_summary.csv"))
    if not case_files:
        raise RuntimeError("no completed spatial stability cases")
    long = pd.concat([pd.read_csv(path) for path in case_files], ignore_index=True)
    profiles = pd.concat(
        [
            pd.read_csv(path.parent / "spatial_profiles.csv")
            for path in case_files
        ],
        ignore_index=True,
    )
    write_csv(out / "患者级空间结构比较_长表.csv", long)
    write_csv(out / "归一化空间分箱曲线_长表.csv", profiles)
    summary = summarize(long)
    write_csv(out / "Paper1_空间结构稳定性正式汇总.csv", summary)
    return long, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=DEFAULT_SPATIAL_BINS)
    parser.add_argument("--expected-cases", type=int, default=319)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.bins < 2:
        raise ValueError("spatial bins must be at least 2")
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
            f"Paper 1 spatial stability expected {args.expected_cases} cases, "
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
        raise RuntimeError("Original reproduction failed; spatial analysis stopped")

    contract = dict(
        cases=[folder.name for folder in cases],
        expected_cases=args.expected_cases,
        bins=int(args.bins),
        normalized_position_definition=(
            "frozen normalized_cervix_to_fundus_position partitioned into "
            "equal-width 0-1 bins"
        ),
        longitudinal_position_definition=(
            "midpoint of the two adjacent frozen section coordinates"
        ),
        profile_definition=(
            "same absolute-value statistic and anatomical side domain as F01-F20 "
            "within each normalized-position bin, aggregated across original frames"
        ),
        spatial_interpolation=False,
        spatial_resampling=False,
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
                / "代码/01_底层算法/peristalsis_pipeline/spatial_stability.py"
            ): quality.sha(
                PROJECT
                / "代码/01_底层算法/peristalsis_pipeline/spatial_stability.py"
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
        print(f"Spatial stability {index}/{len(cases)}: {folder.name}", flush=True)
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
    report = f"""# Paper 1 空间结构稳定性分析

本轮使用 {long.case_id.nunique()} 例冻结结果，在不重新追踪、不改变正式 QC、
不读取临床结局的前提下，比较 Original 与同一 pending Grade-3 mask-only shadow。

空间轴使用冻结的 normalized_cervix_to_fundus_position（宫颈→宫底 0-1），
固定分为 {args.bins} 个等比例解剖位置段。RSR、腔宽形变率和曲率变化率使用
原 section 坐标；纵向壁形变率定义在相邻 section 之间，因此使用两端坐标中点。

每个空间段内使用与对应 F01-F20 相同的解剖侧域、绝对值和 median/P95 统计，
并跨原始时间帧汇总。没有空间插值、补点、等距重采样或跨空缺段拼接。

输出同时报告：空间段可计算性、有效位置保留比例、逐空间段 SRD、病例内空间曲线
Spearman，以及患者级聚合 SRD。两类结果并列保存，用于识别“聚合值变化小，但
局部解剖位置变化相对更明显”的情况；本程序不设置稳定/不稳定阈值，也不自动下结论。

有 pending Grade-3 的病例数：{pending.case_id.nunique()}。
完整逐病例结果见“患者级空间结构比较_长表.csv”，逐空间段曲线见
“归一化空间分箱曲线_长表.csv”，跨病例描述见“Paper1_空间结构稳定性正式汇总.csv”。

本分析仅评价现有局部动态测量在预定义 QC 屏蔽下的空间结构保持程度。
它不能证明跨帧同一组织身份、蠕动波、传播方向、波速、频率或生理真实性。
"""
    (args.output / "分析报告.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
