"""Read-only-source tracking quality sensitivity; writes a separate audit bundle.

No tracking, QC decisions or clinical outcomes are computed or changed here.
Numerical screening is not independent validation of automatic artifact labels.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "代码/01_底层算法"))
from peristalsis_pipeline.feature_stability import (
    StabilityCaseData, FEATURE_SPECS, apply_pair_exclusion, compute_feature_row,
    symmetric_relative_difference_pct, valid_count_and_ratio,
)
from peristalsis_pipeline.formal_feature_extraction import BIOLOGICAL_FEATURE_COLUMNS

FEATURES = list(BIOLOGICAL_FEATURE_COLUMNS)
ROOT = PROJECT / "输出"
DEFAULT_OUT = ROOT / "全患者汇总/追踪质量敏感性验证_20260914"
SEED = 20260914


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def table(path, value):
    pd.DataFrame(value).to_csv(path, index=False, encoding="utf-8-sig")


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def source_paths(folder):
    case = folder.name
    p = folder / "02_追踪质控与形变"
    result = {
        "rsr": p / f"10_rsr_fourway/{case}_rsr_fourway_v1.npz",
        "anatomical": p / f"11_anatomical_deformation/{case}_anatomical_deformation_v1.npz",
        "tracking": p / f"05_radial_tracking_corrected/{case}_p3_image_boundary_evidence_v2.npz",
        "artifact": p / f"07_artifact_qc/{case}_artifact_qc_v2_1.npz",
        "csv": folder / "04_患者特征/患者特征.csv",
    }
    d = folder / f"03_DICOM物理标定/{case}_dicom_curvature_v1.npz"
    if d.is_file():
        result["dicom"] = d
    return result


def load_case(folder):
    paths = source_paths(folder)
    r, a, t, q = [npz(paths[k]) for k in ("rsr", "anatomical", "tracking", "artifact")]
    d = npz(paths["dicom"]) if "dicom" in paths else {}
    data = StabilityCaseData(
        folder.name, r["current_qc_radial_strain_rate_s"], r["fps"], a["pair_qc_valid"], a["fps"],
        a["qc_cavity_width_strain_rate_s"], a["cavity_qc_valid"],
        a["qc_longitudinal_wall_strain_rate_s"], a["longitudinal_qc_valid"],
        d.get("qc_wall_curvature_change_rate_mm_inv_s"),
        d.get("physical_curvature_available", False), d.get("physical_curvature_rate_available", False),
    )
    shape = data.rsr.shape
    for field in ("radial_pair_pcc", "radial_pair_fb_error_px", "radial_pair_valid", "radial_strain_rate_s"):
        if t[field].shape != shape:
            raise ValueError(f"{folder.name}: misaligned {field}")
    if len(q["automatic_grade3_pending_manual_review"]) != shape[0]:
        raise ValueError("artifact frame mismatch")
    if not np.isclose(float(t["fps"]), float(data.rsr_fps), rtol=0, atol=1e-6):
        raise ValueError("tracking timebase mismatch")
    if not np.allclose(q["qc_radial_strain_rate_grade3_blank_s"], data.rsr, rtol=0, atol=0, equal_nan=True):
        raise ValueError("artifact/final RSR version mismatch")
    return data, t, q, paths


def official_extractor():
    path = PROJECT / "代码/02_已并入主流程/05_输出审查与发布/run_feature_v1_formal_candidate_5case.py"
    spec = importlib.util.spec_from_file_location("quality_official_extractor", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def original_check(folder, extractor):
    data, _, _, paths = load_case(folder)
    validated = extractor.extract_case(paths["rsr"], paths["anatomical"].parent,
                                      folder / "03_DICOM物理标定")
    reconstructed = compute_feature_row(data)
    saved = pd.read_csv(paths["csv"]).iloc[0]
    rows = []
    for i, field in enumerate(FEATURES, 1):
        x, y, z = float(saved[field]), float(validated[field]), float(reconstructed[field])
        good = bool(np.isclose(x, y, rtol=1e-7, atol=1e-10, equal_nan=True)
                    and np.isclose(y, z, rtol=0, atol=0, equal_nan=True))
        rows.append(dict(case_id=folder.name, feature_id=f"F{i:02}", feature=field,
                         saved=x, reconstructed=y, difference=y-x, passed=good))
    return rows, {str(p): sha(p) for p in paths.values()}


def frame_stats(values, statistic):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        flat = np.asarray(values).reshape(len(values), -1)
        if statistic == "mean":
            return np.nanmean(flat, axis=1)
        return np.nanpercentile(flat, statistic, axis=1)


def aligned_tables(data, t, q):
    T, sides, sections = data.rsr.shape
    formal = np.isfinite(data.rsr)
    pcc, fb = t["radial_pair_pcc"], t["radial_pair_fb_error_px"]
    f = pd.DataFrame(dict(
        case_id=data.case_id, frame=np.arange(T), time_s=np.arange(T)/float(data.rsr_fps),
        RSR_p95=frame_stats(abs(data.rsr), 95), RSR_median=frame_stats(abs(data.rsr), 50),
        pair_PCC_mean=frame_stats(np.where(formal, pcc, np.nan), "mean"),
        pair_FB_p95=frame_stats(np.where(formal, fb, np.nan), 95),
        all_observed_pair_PCC_mean=frame_stats(pcc, "mean"),
        all_observed_pair_FB_p95=frame_stats(fb, 95),
        pair_valid_fraction=t["radial_pair_valid"].mean(axis=(1, 2)),
        formal_valid_fraction=formal.mean(axis=(1, 2)),
        automatic_grade3=q["automatic_artifact_grade"] >= 3,
        pending_grade3=q["automatic_grade3_pending_manual_review"],
        automatic_grade=q["automatic_artifact_grade"], action_grade=q["artifact_grade"],
        global_translation_px=q["global_translation_step_px"],
        global_rotation_deg=q["global_rotation_step_deg"],
        simultaneous_deformation_fraction=q["simultaneous_deformation_fraction"],
        synchronized_motion_score=q["synchronized_motion_anomaly_score"],
        global_motion_score=q["global_motion_anomaly_score"],
    ))
    f = f.iloc[1:].copy()
    f["high_RSR_top5pct"] = False
    good = f.index[np.isfinite(f.RSR_p95)]
    # Exactly ceil(5%); ties resolved by frame index, documented sampling only.
    high = f.loc[good].sort_values(["RSR_p95", "frame"], ascending=[False, True]).head(math.ceil(.05*len(good))).index
    f.loc[high, "high_RSR_top5pct"] = True
    axes = np.indices((T-1, sides, sections))
    pairs = pd.DataFrame(dict(
        case_id=data.case_id, frame=(axes[0]+1).ravel(), side=axes[1].ravel(), section=axes[2].ravel(),
        raw_RSR=t["radial_strain_rate_s"][1:].ravel(), formal_RSR=data.rsr[1:].ravel(),
        pair_PCC=pcc[1:].ravel(), pair_FB_px=fb[1:].ravel(),
        radial_pair_valid=t["radial_pair_valid"][1:].ravel(), formal_valid=formal[1:].ravel(),
    ))
    pairs["abs_formal_RSR"] = abs(pairs.formal_RSR)
    pairs["rank_within_frame"] = pairs.groupby("frame").abs_formal_RSR.rank(ascending=False, method="min")
    pairs["high_RSR_frame"] = pairs.frame.isin(high)
    comparison = dict(case_id=data.case_id, high_frames=len(high), ordinary_frames=len(good)-len(high))
    for metric in ("pair_PCC_mean", "pair_FB_p95", "pair_valid_fraction", "formal_valid_fraction",
                   "global_translation_px", "global_rotation_deg", "simultaneous_deformation_fraction", "pending_grade3"):
        x = f.loc[high, metric].mean()
        y = f.loc[good.difference(high), metric].mean()
        comparison.update({metric+"_high_mean": x, metric+"_ordinary_mean": y, "delta_"+metric: x-y})
    return f, pairs, comparison


def exclude_frames(data, frames):
    mask = np.zeros(data.pair_valid.shape, dtype=bool)
    mask[np.asarray(frames, dtype=int)] = True
    result = apply_pair_exclusion(data, mask)
    # Whole-frame intervention must remove every formally finite feature value.
    for spec in FEATURE_SPECS:
        from peristalsis_pipeline.feature_stability import domain_values_and_valid
        _, valid = domain_values_and_valid(result, spec)
        if valid.size and np.any(valid[np.asarray(frames, dtype=int)]):
            raise ValueError(f"whole-frame mask not propagated to {spec.name}")
    return result


def feature_vector(data):
    row = compute_feature_row(data)
    return np.array([row[k] for k in FEATURES], dtype=float)


def runs(indices):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)


def block_sample(pool, lengths, rng):
    # Uniform eligible start per block, sequential non-overlap, bounded retries.
    available_base = np.zeros(int(pool.max())+1, dtype=bool)
    available_base[pool] = True
    for attempt in range(100):
        available = available_base.copy()
        chosen = []
        for width in sorted(lengths, reverse=True):
            counts = np.convolve(available.astype(int), np.ones(width, dtype=int), mode="valid")
            starts = np.flatnonzero(counts == width)
            # Separate placed runs by >=1 frame, preserving run count.
            if not len(starts):
                break
            start = int(rng.choice(starts))
            chosen.extend(range(start, start+width))
            available[max(0, start-1):min(len(available), start+width+1)] = False
        else:
            return np.sort(chosen)
    raise ValueError("cannot place matching nonoverlapping blocks in valid frame pool")


def analyze_case(folder, out, repetitions):
    case_out = out / "patients" / folder.name
    if (case_out / "complete.json").exists():
        return json.loads((case_out / "complete.json").read_text(encoding="utf-8"))
    case_out.mkdir(parents=True, exist_ok=True)
    data, t, q, paths = load_case(folder)
    f, pairs, comparison = aligned_tables(data, t, q)
    f.to_csv(case_out / "frames.csv.gz", index=False, compression="gzip")
    pairs.to_csv(case_out / "pairs.csv.gz", index=False, compression="gzip")
    table(case_out / "within_patient_comparison.csv", [comparison])
    pool = np.flatnonzero(np.any(np.isfinite(data.rsr), axis=(1, 2)))
    pool = pool[pool > 0]
    pending = np.flatnonzero(q["automatic_grade3_pending_manual_review"])
    removed = np.intersect1d(pool, pending)
    A = feature_vector(data)
    shadow_data = exclude_frames(data, removed)
    B = feature_vector(shadow_data)
    seed = SEED + int(hashlib.sha256(folder.name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    C = np.empty((repetitions, 20))
    indices = np.empty((repetitions, len(removed)), dtype=np.int32)
    for i in range(repetitions):
        selected = np.sort(rng.choice(pool, size=len(removed), replace=False))
        indices[i] = selected
        C[i] = feature_vector(exclude_frames(data, selected)) if len(removed) else A
    np.savez_compressed(case_out / "random_control.npz", features=C, removed_frame_indices=indices,
                        grade3_removed_frames=removed, eligible_frames=pool, seed=seed, feature_names=FEATURES)
    difference = B-A
    random_delta = C-A
    numerical_screen = np.zeros(20, dtype=bool)
    rows = []
    for j, spec in enumerate(FEATURE_SPECS):
        finite = random_delta[np.isfinite(random_delta[:, j]), j]
        exact_tail = ((1+np.sum(abs(finite) >= abs(difference[j])))/(1+len(finite))) if len(finite) and np.isfinite(difference[j]) else np.nan
        # Exploratory trigger only, not a feature pass/fail or formal QC threshold.
        trigger = bool(len(removed) and np.isfinite(exact_tail) and exact_tail <= .05)
        numerical_screen[j] = trigger
        stats = np.percentile(finite, [2.5, 50, 97.5]) if len(finite) else [np.nan]*3
        original_finite = bool(np.isfinite(A[j]))
        shadow_finite = bool(np.isfinite(B[j]))
        if (not original_finite) and shadow_finite:
            raise RuntimeError(
                f"{folder.name} {spec.name}: mask-only QC sensitivity created NaN-to-finite"
            )
        availability_status = (
            "paired_finite"
            if original_finite and shadow_finite
            else "finite_to_nan"
            if original_finite
            else "both_nan"
        )
        valid_before = valid_count_and_ratio(data, spec)[0]
        valid_after = valid_count_and_ratio(shadow_data, spec)[0]
        row = dict(case_id=folder.name, feature_id=f"F{j+1:02}", feature=spec.name,
                   statistic=spec.statistic_type, original=A[j], grade3_shadow=B[j],
                   signed_change=difference[j], absolute_change=abs(difference[j]),
                   srd_pct=symmetric_relative_difference_pct(A[j], B[j]),
                   relative_change_pct=100*difference[j]/abs(A[j]) if original_finite and A[j] != 0 else np.nan,
                   availability_status=availability_status,
                   finite_to_nan=availability_status == "finite_to_nan",
                   random_delta_p025=stats[0], random_delta_median=stats[1], random_delta_p975=stats[2],
                   random_abs_tail_fraction=exact_tail, numerical_screen=trigger,
                   valid_before=valid_before, valid_after=valid_after,
                   valid_loss_n=valid_before-valid_after,
                   valid_loss_fraction=(valid_before-valid_after)/valid_before if valid_before else np.nan,
                   grade3_pending_frames=len(pending), grade3_removed_frames=len(removed), eligible_frames=len(pool),
                   grade3_frame_fraction=len(removed)/len(pool) if len(pool) else np.nan,
                   topology_risk=bool(t["has_tracking_topology_risk"]),
                   has_pending_grade3=bool(len(pending)), grade3_effective=bool(len(removed)),
                   physical_calibration=bool(data.physical_curvature_rate_available),
                   final_interpretation="证据不足_待盲法复核")
        rows.append(row)
    lengths = [len(r) for r in runs(removed)]
    block_status = "not_triggered"
    if np.any(numerical_screen) and any(n > 1 for n in lengths):
        block_rng = np.random.default_rng(seed+1)
        BC, BI = [], []
        try:
            for i in range(repetitions):
                sampled = block_sample(pool, lengths, block_rng)
                BI.append(sampled)
                BC.append(feature_vector(exclude_frames(data, sampled)))
            BC = np.asarray(BC)
            np.savez_compressed(case_out / "block_random_control.npz", features=BC,
                                removed_frame_indices=np.asarray(BI), original_run_lengths=lengths, seed=seed+1)
            for j, row in enumerate(rows):
                finite = (BC-A)[:, j]
                finite = finite[np.isfinite(finite)]
                row["block_random_abs_tail_fraction"] = (1+sum(abs(finite) >= abs(difference[j])))/(len(finite)+1) if len(finite) else np.nan
            block_status = "completed"
        except ValueError as exc:
            block_status = str(exc)
    table(case_out / "features.csv", rows)
    meta = dict(case_id=folder.name, frame_count=len(data.rsr), fps=float(data.rsr_fps),
                eligible_frames=len(pool), pending_grade3_frames=len(pending), removed_frames=len(removed),
                topology_risk=bool(t["has_tracking_topology_risk"]), block_status=block_status,
                block_trigger="any feature absolute effect upper-tail fraction <=0.05 AND any consecutive run; exploratory only",
                numerical_screen_features=[FEATURES[j] for j in np.flatnonzero(numerical_screen)],
                repetitions=repetitions, seed=seed)
    save_json(case_out / "complete.json", meta)
    return meta


def summarize_qc_feature_group(all_g):
    """Summarize one feature/stratum without hiding availability loss."""
    original_finite = np.isfinite(all_g.original)
    shadow_finite = np.isfinite(all_g.grade3_shadow)
    nan_to_finite_n = int((~original_finite & shadow_finite).sum())
    if nan_to_finite_n:
        raise RuntimeError("NaN-to-finite violates mask-only QC sensitivity")
    paired = all_g.loc[original_finite & shadow_finite].copy()
    n_original_finite = int(original_finite.sum())
    n_shadow_finite = int(shadow_finite.sum())
    finite_to_nan_n = int((original_finite & ~shadow_finite).sum())
    rho = (
        paired.original.corr(paired.grade3_shadow, method="spearman")
        if len(paired) > 1
        else np.nan
    )
    ranks_a = paired.original.rank(ascending=False)
    ranks_b = paired.grade3_shadow.rank(ascending=False)
    return dict(
        n_total_cases=int(len(all_g)),
        n_original_finite=n_original_finite,
        n_paired_finite=int(len(paired)),
        n_shadow_finite=n_shadow_finite,
        finite_to_nan_n=finite_to_nan_n,
        finite_to_nan_fraction=(
            finite_to_nan_n / n_original_finite if n_original_finite else np.nan
        ),
        availability_retention_fraction=(
            n_shadow_finite / n_original_finite if n_original_finite else np.nan
        ),
        spearman=rho,
        median_absolute_change=paired.absolute_change.median() if len(paired) else np.nan,
        p95_absolute_change=paired.absolute_change.quantile(.95) if len(paired) else np.nan,
        median_srd_pct=paired.srd_pct.median() if len(paired) else np.nan,
        p95_srd_pct=paired.srd_pct.quantile(.95) if len(paired) else np.nan,
        # Legacy denominator-sensitive metric retained only for audit/backward comparison.
        median_relative_change_pct=(
            paired.relative_change_pct.median() if len(paired) else np.nan
        ),
        median_abs_relative_change_pct=(
            paired.relative_change_pct.abs().median() if len(paired) else np.nan
        ),
        p95_abs_relative_change_pct=(
            paired.relative_change_pct.abs().quantile(.95) if len(paired) else np.nan
        ),
        median_abs_within_stratum_rank_change=(
            (ranks_b-ranks_a).abs().median() if len(paired) else np.nan
        ),
        max_abs_within_stratum_rank_change=(
            (ranks_b-ranks_a).abs().max() if len(paired) else np.nan
        ),
    )


def aggregate(out):
    folders = sorted((out / "patients").glob("*/complete.json"))
    long = pd.concat([pd.read_csv(p.parent/"features.csv") for p in folders], ignore_index=True)
    comparisons = pd.concat([pd.read_csv(p.parent/"within_patient_comparison.csv") for p in folders], ignore_index=True)
    for field in ("original", "grade3_shadow"):
        long["rank_"+field] = long.groupby("feature")[field].rank(ascending=False, method="average")
    long["rank_change"] = long.rank_grade3_shadow-long.rank_original
    table(out/"患者级特征比较_长表.csv", long)
    table(out/"患者内高RSR与普通帧比较.csv", comparisons)
    wide = long.pivot(index="case_id", columns="feature_id", values=[
        "original", "grade3_shadow", "absolute_change", "relative_change_pct",
        "random_delta_p025", "random_delta_median", "random_delta_p975",
        "random_abs_tail_fraction", "rank_original", "rank_grade3_shadow", "rank_change",
    ])
    wide.columns = [b+"_"+a for a,b in wide.columns]
    metadata = long.drop_duplicates("case_id").set_index("case_id")[["grade3_frame_fraction", "topology_risk", "has_pending_grade3", "grade3_effective", "physical_calibration"]]
    table(out/"患者级特征比较_宽表.csv", wide.join(metadata).reset_index())
    rows=[]
    strata={"all": np.ones(len(long),bool), "pending_grade3": long.has_pending_grade3,
            "effective_grade3": long.grade3_effective, "topology_yes": long.topology_risk,
            "topology_no": ~long.topology_risk,
            "pending_topology_yes": long.has_pending_grade3 & long.topology_risk,
            "pending_topology_no": long.has_pending_grade3 & ~long.topology_risk}
    for label, mask in strata.items():
        for feature, all_g in long.loc[mask].groupby("feature", sort=False):
            metrics = summarize_qc_feature_group(all_g)
            rows.append(dict(
                stratum=label,
                feature=feature,
                statistic=all_g.statistic.iloc[0],
                **metrics,
                screen_count=int(all_g.numerical_screen.sum()),
                conclusion="数值QC敏感性_不等同于伪影真值验证",
            ))
    table(out/"总体分层汇总.csv", rows)
    table(out/"Paper1_QC敏感性正式汇总.csv", rows)
    # Patient resampling, never frame-level pseudo-replicated significance tests.
    rng=np.random.default_rng(SEED)
    summaries=[]
    for col in [c for c in comparisons if c.startswith("delta_")]:
        vals=comparisons[col].dropna().to_numpy()
        boot=np.median(rng.choice(vals,size=(2000,len(vals)),replace=True),axis=1)
        summaries.append(dict(metric=col, patients=len(vals), median=np.median(vals),
                              median_ci_low=np.percentile(boot,2.5),median_ci_high=np.percentile(boot,97.5),
                              negative_patients=int(sum(vals<0)),positive_patients=int(sum(vals>0))))
    table(out/"患者内质量差异_患者重采样汇总.csv",summaries)
    table(out/"特征字典.csv",[dict(feature_id=f"F{i+1:02}", feature=s.name, family=s.family,statistic=s.statistic_type) for i,s in enumerate(FEATURE_SPECS)])
    return long


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--repetitions",type=int,default=500)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--expected-cases",type=int,default=319,
                        help="Paper 1当前冻结QC敏感性队列数；默认319，数量不符则停止")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.repetitions<1: raise ValueError("repetitions must be positive")
    if args.output.resolve()==args.root.resolve(): raise ValueError("output must be separate")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("nonempty output; use explicit --resume")
    args.output.mkdir(parents=True,exist_ok=True)
    cases=sorted(p for p in args.root.iterdir() if p.is_dir() and (p/"04_患者特征/患者特征.csv").is_file())
    if args.limit:
        cases=cases[:args.limit]
    elif args.expected_cases is not None and len(cases) != args.expected_cases:
        raise ValueError(
            f"Paper 1 QC sensitivity expected {args.expected_cases} cases, found {len(cases)}; "
            "do not silently change the frozen cohort"
        )
    extractor=official_extractor()
    checks=[]; hashes={}
    for i,folder in enumerate(cases,1):
        rows,h=original_check(folder,extractor); checks.extend(rows); hashes.update(h)
        if i%25==0 or i==len(cases): print(f"Original check {i}/{len(cases)}",flush=True)
    table(args.output/"Original复现检查.csv",checks)
    if not all(r["passed"] for r in checks): raise RuntimeError("Original reproduction failed; no perturbations run")
    manifest=args.output/"source_manifest.json"
    source_code={str(Path(__file__).resolve()):sha(__file__)}
    for name in ("feature_stability.py","formal_feature_extraction.py"):
        p=PROJECT/"代码/01_底层算法/peristalsis_pipeline"/name;source_code[str(p)]=sha(p)
    contract=dict(cases=[p.name for p in cases],expected_cases=args.expected_cases,
                  repetitions=args.repetitions,source_sha256=hashes,code_sha256=source_code,
                  seed=SEED,original_rtol=1e-7,original_atol=1e-10,
                  first_frame_excluded_from_sampling=True,
                  formal_effect_metric="symmetric_relative_difference_pct",
                  availability_event="finite_to_nan")
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8"))!=contract:
        raise ValueError("resume source/code/config identity changed")
    save_json(manifest,contract)
    for i,folder in enumerate(cases,1):
        print(f"Sensitivity {i}/{len(cases)}: {folder.name}",flush=True)
        meta=analyze_case(folder,args.output,args.repetitions)
        print(f"  removed={meta['removed_frames']} block={meta['block_status']}",flush=True)
    aggregate(args.output)
    changed=[p for p,h in hashes.items() if sha(p)!=h]
    save_json(args.output/"source_integrity.json",dict(files=len(hashes),unchanged=not changed,changed=changed))
    if changed: raise RuntimeError("source integrity mismatch")
    print("Numerical sensitivity complete; manual review pending",flush=True)


if __name__=="__main__":
    main()
