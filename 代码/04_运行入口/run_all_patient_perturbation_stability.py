"""Run the historical 42-perturbation method on per-patient frozen outputs."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "代码/01_底层算法"))


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "代码/03_实验与历史代码" / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


old = module("historical_perturbation", "run_new_patient_feature_stability_validation.py")
stats = module("historical_robustness", "run_literature_aligned_feature_robustness.py")


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def run(args):
    out = args.output
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    out.mkdir(parents=True)
    cases = sorted(p for p in args.root.iterdir() if (p / "04_患者特征/患者特征.csv").is_file())
    if args.cases:
        requested = set(args.cases)
        cases = [p for p in cases if p.name in requested]
        if set(p.name for p in cases) != requested:
            raise ValueError("Requested cases not found")
    elif args.expected_cases is not None and len(cases) != args.expected_cases:
        raise ValueError(
            f"Paper 1 perturbation robustness expected {args.expected_cases} cases, "
            f"found {len(cases)}; do not silently change the frozen cohort"
        )
    hashes, records, all_rows = {}, [], []
    for i, folder in enumerate(cases, 1):
        pipeline = folder / "02_追踪质控与形变"
        data, position, shadow, meta, paths = old.load_case(folder.name,
            rsr_dir=pipeline / "10_rsr_fourway", anatomical_dir=pipeline / "11_anatomical_deformation",
            shadow_dir=pipeline / "09_pair_quality_shadow", artifact_dir=pipeline / "07_artifact_qc",
            dicom_dir=folder / "03_DICOM物理标定")
        paths.append(folder / "04_患者特征/患者特征.csv")
        for path in paths:
            hashes[str(path)] = old.sha256(path)
        baseline = old.compute_feature_row(data)
        saved = pd.read_csv(paths[-1]).iloc[0]
        np.testing.assert_allclose([baseline[s.name] for s in old.FEATURE_SPECS],
                                   [saved[s.name] for s in old.FEATURE_SPECS],
                                   rtol=1e-7, atol=1e-10, equal_nan=True)
        rows = pd.DataFrame(old.analyze_case(data, position, shadow, baseline))
        if len(rows) != 840:
            raise ValueError("Expected 20 features x 42 perturbations per case")
        case_out = out / "patients" / folder.name
        case_out.mkdir(parents=True)
        write_csv(case_out / "perturbations.csv", rows)
        all_rows.append(rows)
        records.append(dict(**meta, baseline_matches_saved=True, perturbation_rows=len(rows)))
        print(f"[{i}/{len(cases)}] {folder.name}: 840 rows", flush=True)
    long = pd.concat(all_rows, ignore_index=True)
    write_csv(out / "case_audit.csv", records)
    write_csv(out / "perturbations_long.csv", long)
    # The historical canonical 26-column design is only valid if these masks alias.
    aliases = stats.audit_space_aliases(long)
    analysis = stats.build_analysis_long(long)
    features = [s.name for s in old.FEATURE_SPECS]
    print("Computing primary and family ICC, CV, jackknife intervals", flush=True)
    # Feature-specific incomplete subjects cannot enter balanced ICC. Retain them in
    # raw output and explicit exclusions instead of filling missing measurements.
    incomplete = []
    for feature in features:
        rows = analysis[(analysis.feature == feature) & (analysis.record_type == "perturbation")
                        & analysis.analysis_family.ne("top1_removal")]
        for case, group in rows.groupby("case_id"):
            if not np.isfinite(group.analysis_value.to_numpy(float)).all():
                incomplete.append(dict(case_id=case, feature=feature, reason="nonfinite_repeated_measurement"))
    for row in incomplete:
        analysis = analysis[~(analysis.case_id.eq(row["case_id"]) & analysis.feature.eq(row["feature"]))]
    write_csv(out / "icc_excluded_case_features.csv", pd.DataFrame(incomplete, columns=["case_id", "feature", "reason"]))
    families = stats.calculate_family_iccs(analysis, features)
    write_csv(out / "perturbation_family_icc.csv", families)
    # Same CV formula and baseline/null-shadow handling, grouped for larger cohorts.
    cv_rows = []
    cv_data = analysis[analysis.included_primary_cv &
        (analysis.record_type.eq("baseline") | analysis.actual_mask_dedup_status.eq("unique_retained"))]
    for (case, feature), group in cv_data.groupby(["case_id", "feature"]):
        value, status = stats.within_case_cv_pct(group.analysis_value.to_numpy(float))
        cv_rows.append(dict(case_id=case, feature=feature, within_case_cv_pct=value, cv_status=status,
                            measurements=len(group)))
    cv = pd.DataFrame(cv_rows)
    write_csv(out / "within_case_cv.csv", cv)
    sensitivity = stats.calculate_sensitivity(analysis, features, bootstrap_repetitions=args.bootstrap_repetitions)
    write_csv(out / "jackknife_sensitivity.csv", sensitivity)
    summary = []
    for code, feature in enumerate(features, 1):
        r = sensitivity.set_index("feature").loc[feature]
        c = cv[cv.feature.eq(feature)].within_case_cv_pct
        summary.append(dict(feature_code=f"F{code:02}", feature=feature,
            n_cases=r.analysis_a_primary_n_cases, primary_icc=r.analysis_a_primary_icc,
            ci_lower=r.analysis_a_primary_ci_lower, ci_upper=r.analysis_a_primary_ci_upper,
            median_cv_pct=c.median(), p95_cv_pct=c.quantile(.95),
            cv_lt20_fraction=(c<20).mean()))
    summary = pd.DataFrame(summary)
    write_csv(out / "feature_summary.csv", summary)
    srd = long.groupby(["feature", "perturbation"]).agg(
        median_srd_pct=("relative_or_SRD_change", "median"),
        p95_srd_pct=("relative_or_SRD_change", lambda x: x.quantile(.95)),
        finite_to_nan=("status", lambda x: int(x.eq("finite_to_nan").sum())),
        observations=("case_id", "size")).reset_index()
    write_csv(out / "srd_by_perturbation.csv", srd)
    for path, digest in hashes.items():
        if old.sha256(Path(path)) != digest:
            raise RuntimeError(f"Source changed: {path}")
    manifest = dict(status="complete", cases=len(cases), long_rows=len(long),
        scope="all available cases, exploratory robustness; formal eligibility retained in case_audit",
        method="historical 42 perturbations; canonical 26 primary; ICC(1,1), within-case CV",
        aliases=aliases, bootstrap_repetitions=args.bootstrap_repetitions,
        source_npz_and_csv_sha256=hashes, source_unchanged=True,
        code_sha256={str(p): old.sha256(p) for p in [Path(__file__), Path(old.__file__), Path(stats.__file__)]},
        differences_from_historical="per-patient directories; all cases including flagged cases; no historical classification rescore; incomplete case-feature exclusions explicit",
        clinical_labels_used=False, tracking_rerun=False)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 全患者多扰动特征鲁棒性", "", f"完成{len(cases)}例、{len(long)}条扰动记录。",
        "复用旧28例42次扰动及ICC/CV数值函数。primary保留26列；空间5%/10%实际掩码相同经逐例核验后去重。",
        "全病例为探索性队列，既有影像准入/拓扑风险记录在case_audit.csv；并不自动升级为正式合格病例。",
        "使用已有正式特征作为基线；LK红色shadow沿用旧方法。Grade-3对照保留在之前独立结果中，不替代shadow。",
        "未使用临床结局或日期，不重新追踪。曲率缺失及扰动后缺失见icc_excluded_case_features.csv，未插补。",
        "这里是数据删除扰动鲁棒性，不是复扫/操作者/设备重复性。0.90和CV20%为参考线，不是临床合格标准。", "",
        f"Primary ICC点估计≥0.90：{int((summary.primary_icc>=.9).sum())}/20；95%CI下限≥0.90：{int((summary.ci_lower>=.9).sum())}/20。",
        "必须结合单独空间、时间扰动结果，不能用合并ICC掩盖空间敏感性。", "",
        "|特征|例数|ICC|95%CI|CV中位数%|CV P95%|", "|---|---:|---:|---|---:|---:|"]
    for r in summary.itertuples():
        lines.append(f"|{r.feature_code}|{int(r.n_cases)}|{r.primary_icc:.4f}|{r.ci_lower:.4f}–{r.ci_upper:.4f}|{r.median_cv_pct:.2f}|{r.p95_cv_pct:.2f}|")
    (out / "结果报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"COMPLETE: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "输出")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=None,
        help="Optional frozen Paper 1 perturbation cohort size; ignored when --cases is explicitly supplied.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    run(parser.parse_args())
