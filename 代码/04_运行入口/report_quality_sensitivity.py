"""Audit and descriptive report for saved sensitivity outputs; no source mutation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_tracking_quality_sensitivity import (
    DEFAULT_OUT, ROOT, FEATURES, FEATURE_SPECS, load_case, exclude_frames, feature_vector,
    runs, sha, save_json, summarize_qc_feature_group, table,
)


def add_formal_summary_columns(long):
    """Backfill Paper 1 metrics from saved A/B values without rerunning perturbations."""
    result=long.copy()
    original=pd.to_numeric(result["original"],errors="coerce").to_numpy(float)
    shadow=pd.to_numeric(result["grade3_shadow"],errors="coerce").to_numpy(float)
    paired=np.isfinite(original)&np.isfinite(shadow)
    denominator=np.abs(original)+np.abs(shadow)
    srd=np.full(len(result),np.nan,dtype=float)
    zero=paired&(denominator==0)
    nonzero=paired&(denominator>0)
    srd[zero]=0.0
    srd[nonzero]=200.0*np.abs(shadow[nonzero]-original[nonzero])/denominator[nonzero]
    result["srd_pct"]=srd
    result["finite_to_nan"]=np.isfinite(original)&~np.isfinite(shadow)
    if np.any(~np.isfinite(original)&np.isfinite(shadow)):
        raise RuntimeError("saved QC sensitivity contains NaN-to-finite, violating mask-only semantics")
    result["availability_status"]=np.where(
        paired,"paired_finite",
        np.where(np.isfinite(original),"finite_to_nan","both_nan")
    )
    return result


def formal_summary(long):
    rows=[]
    strata={"all": np.ones(len(long),bool), "pending_grade3": long.has_pending_grade3,
            "effective_grade3": long.grade3_effective, "topology_yes": long.topology_risk,
            "topology_no": ~long.topology_risk,
            "pending_topology_yes": long.has_pending_grade3 & long.topology_risk,
            "pending_topology_no": long.has_pending_grade3 & ~long.topology_risk}
    for label,mask in strata.items():
        for feature,g in long.loc[mask].groupby("feature",sort=False):
            rows.append(dict(
                stratum=label,feature=feature,statistic=g.statistic.iloc[0],
                **summarize_qc_feature_group(g),
                screen_count=int(g.numerical_screen.sum()),
                conclusion="数值QC敏感性_不等同于伪影真值验证",
            ))
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--expected-cases",type=int,default=None,
                        help="可选的Paper 1冻结QC敏感性队列数；显式指定时数量不符则停止")
    args=parser.parse_args();out=args.output
    long=add_formal_summary_columns(pd.read_csv(out/"患者级特征比较_长表.csv"))
    n_cases=int(long.case_id.nunique())
    if args.expected_cases is not None and n_cases != args.expected_cases:
        raise ValueError(
            f"Paper 1 QC sensitivity expected {args.expected_cases} cases, found {n_cases}; "
            "do not silently change the frozen cohort"
        )
    table(out/"患者级特征比较_长表_Paper1正式重汇总.csv",long)
    summary=formal_summary(long)
    table(out/"Paper1_QC敏感性正式汇总.csv",summary)
    checks=[]; block_rows=[]; manifests=[]
    for i,case in enumerate(long.case_id.unique(),1):
        p=out/"patients"/case
        meta=json.loads((p/"complete.json").read_text(encoding="utf-8"));manifests.append(meta)
        data,tracking,q,paths=load_case(args.root/case)
        with np.load(p/"random_control.npz",allow_pickle=False) as z:
            indices=z["removed_frame_indices"];C=z["features"];pool=z["eligible_frames"]
            removed=z["grade3_removed_frames"]
        good=bool(np.all(np.isin(indices,pool)) and not np.any(indices==0)
                  and indices.shape==(meta["repetitions"],meta["removed_frames"])
                  and all(len(np.unique(row))==len(row) for row in indices))
        A=feature_vector(data)
        B=feature_vector(exclude_frames(data,removed))
        recorded=long[long.case_id==case].set_index("feature").loc[FEATURES]
        same=bool(np.allclose(recorded.original,A,rtol=1e-7,atol=1e-10,equal_nan=True)
                  and np.allclose(recorded.grade3_shadow,B,rtol=1e-7,atol=1e-10,equal_nan=True))
        recompute=bool(np.allclose(feature_vector(exclude_frames(data,indices[0])),C[0],rtol=0,atol=0,equal_nan=True))
        block_file=p/"block_random_control.npz"
        if block_file.exists():
            with np.load(block_file,allow_pickle=False) as z:
                bi=z["removed_frame_indices"];bc=z["features"];lengths=sorted(z["original_run_lengths"].tolist())
            good=good and all(sorted(map(len,runs(row)))==lengths for row in bi)
            good=good and bool(np.all(np.isin(bi,pool)))
            recompute=recompute and bool(np.allclose(feature_vector(exclude_frames(data,bi[0])),bc[0],rtol=0,atol=0,equal_nan=True))
            for j,feature in enumerate(FEATURES):
                delta=(bc-A)[:,j];delta=delta[np.isfinite(delta)]
                if len(delta) and np.isfinite(B[j]):
                    block_rows.append(dict(case_id=case,feature=feature,block_delta_p025=np.percentile(delta,2.5),
                        block_delta_median=np.median(delta),block_delta_p975=np.percentile(delta,97.5),
                        block_random_abs_tail_fraction=(1+sum(abs(delta)>=abs(B[j]-A[j])))/(len(delta)+1)))
        checks.append(dict(case_id=case,random_and_block_masks_valid=good,
                           baseline_and_shadow_match=same,first_random_replicates_reproduced=recompute))
        if i%50==0:print(f"Independent saved-output audit {i}",flush=True)
    table(out/"随机对照及重算核查.csv",checks)
    if not all(all(r[k] for k in r if k!="case_id") for r in checks):raise ValueError("saved output audit failed")
    table(out/"条件时间块随机对照汇总.csv",block_rows)
    table(out/"病例运行清单.csv",manifests)
    dictionary=pd.read_csv(out/"特征字典.csv").set_index("feature")
    pending=long[long.has_pending_grade3 & np.isfinite(long.original)].copy()
    pending=pending.join(dictionary[["family"]],on="feature")
    strata=summary[summary.stratum=="pending_grade3"].set_index("feature").loc[FEATURES].copy()
    strata["feature_id"]=[f"F{i+1:02}" for i in range(20)]
    rows=[]
    for feature,g in pending.groupby("feature",sort=False):
        block=g.get("block_random_abs_tail_fraction",pd.Series(np.nan,index=g.index))
        paired=g[np.isfinite(g.grade3_shadow)]
        finite_to_nan_n=int((~np.isfinite(g.grade3_shadow)).sum())
        rows.append(dict(feature_id=dictionary.loc[feature,"feature_id"],feature=feature,
            statistic=g.statistic.iloc[0],n_original_finite=len(g),n_paired_finite=len(paired),
            finite_to_nan_n=finite_to_nan_n,
            availability_retention_fraction=len(paired)/len(g) if len(g) else np.nan,
            signed_change_median=paired.signed_change.median() if len(paired) else np.nan,
            median_srd_pct=paired.srd_pct.median() if len(paired) else np.nan,
            p95_srd_pct=paired.srd_pct.quantile(.95) if len(paired) else np.nan,
            median_absolute_change=paired.absolute_change.median() if len(paired) else np.nan,
            random_upper_tail_screen_count=int(g.numerical_screen.sum()),
            block_tested_count=int(block.notna().sum()),block_upper_tail_screen_count=int((block<=.05).sum()),
            numerical_description="正式描述使用SRD、绝对变化、排序和finite→NaN；普通百分比变化仅保留审计",
            blind_review_status="pending",final_judgment="数值QC敏感性_不等同于伪影真值验证"))
    table(out/"逐特征证据与待复核结论.csv",rows)
    plt.rcParams.update({"font.size":9})
    fig,axes=plt.subplots(1,2,figsize=(13,7))
    labels=strata.feature_id.tolist()
    axes[0].barh(labels,strata.median_srd_pct,color=["#246b99" if x=="median" else "#d9792b" for x in strata.statistic])
    axes[0].set_xlabel("Median SRD (%)\nPatients with pending grade-3")
    axes[1].barh(labels,strata.spearman,color="#478872")
    axes[1].set_xlabel("Original vs shadow Spearman\nPatients with pending grade-3")
    axes[1].set_xlim(0,1)
    for ax in axes:ax.invert_yaxis();ax.grid(axis="x",alpha=.2)
    fig.tight_layout();fig.savefig(out/"feature_change_and_rank.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,fid in zip(axes,["F01","F02"]):
        g=long[(long.feature_id==fid)&long.has_pending_grade3]
        ax.scatter(g.original,g.grade3_shadow,s=18,alpha=.6)
        maximum=max(g.original.max(),g.grade3_shadow.max());ax.plot([0,maximum],[0,maximum],"k--",lw=1)
        ax.set(title=fid,xlabel="Original (/s)",ylabel="Grade-3 shadow (/s)")
    fig.tight_layout();fig.savefig(out/"RSR_original_vs_shadow.png",dpi=180);plt.close(fig)
    quality=pd.read_csv(out/"患者内质量差异_患者重采样汇总.csv")
    check_original=pd.read_csv(out/"Original复现检查.csv")
    integrity=json.loads((out/"source_integrity.json").read_text(encoding="utf-8"))
    selected=strata.loc[[FEATURES[i] for i in [0,1,5,7,9,15]],
        ["feature_id","n_original_finite","n_paired_finite","finite_to_nan_n",
         "availability_retention_fraction","median_srd_pct","p95_srd_pct",
         "spearman","max_abs_within_stratum_rank_change"]]
    compact=selected.to_csv(index=False)
    n_cases=long.case_id.nunique();n_pending=long.loc[long.has_pending_grade3,"case_id"].nunique()
    n_block=sum(m["block_status"]=="completed" for m in manifests)
    affected=long[long.has_pending_grade3 & np.isfinite(long.original) & np.isfinite(long.grade3_shadow)].copy()
    median_rows=affected[affected.statistic=="median"].srd_pct
    p95_rows=affected[affected.statistic=="p95"].srd_pct
    finite_to_nan_total=int((long.has_pending_grade3 & np.isfinite(long.original) & ~np.isfinite(long.grade3_shadow)).sum())
    pcc_row=quality[quality.metric=="delta_pair_PCC_mean"].iloc[0]
    fb_row=quality[quality.metric=="delta_pair_FB_p95"].iloc[0]
    report=f"""# 追踪质量敏感性验证：数值阶段结果

本轮使用 {n_cases} 例已有结果；其中 {n_pending} 例有待人工复核自动 Grade-3。原始结果、追踪算法和正式 QC 均未修改。
Original 复现：{len(check_original)} 项患者-特征核对全部通过（含匹配的缺失值）；比较容差 rtol=1e-7, atol=1e-10，仅用于数值复现。
源文件完整性：{integrity['files']} 个源文件 SHA256 核对未改变。

## 已执行的比较

A 是现有正式值；B 是在内存副本中排除 pending Grade-3 对应有效帧；C 每次独立从全部正式有效帧中等数量、无放回随机删除，允许抽中 Grade-3，每例500次。
第0帧不进入帧级比较或随机抽样池；A完整复现现有正式定义。时间轴和原始数值不重算，所有特征使用同一正式提取函数。
无Grade-3病例的随机分布退化为Original，并按同样500行记录。曲率缺失保持缺失，F15–F20仅对有物理标定者分析。
23例已有拓扑风险病例保留，只作分层。未读取任何临床结局。

最高5%帧按患者内帧级|RSR|P95排序选择，向上取整，同值按帧序；这是探索性抽样，不是正式质控阈值。
帧表同时提供正式有效域内PCC/FB和全部可观测点对的PCC/FB，点对表保存所有侧别与壁段，避免平均值遮蔽局部失配。
患者内差异先分别计算，再跨患者汇总；区间来自2000次患者重采样，不把帧当独立样本。

## 条件时间块对照

预先记录的计算触发：任一特征的Grade-3绝对变化在500次随机绝对变化的上尾比例<=0.05，且存在长度>1的连续候选片段。
这个0.05只触发额外计算，未经多重比较校正，不是显著性结论、临床标准或合格界限。
{n_block} 例完成条件时间块对照（每例500次）；保持原片段数、长度和非重叠，限定正式有效帧。
时间块采用依次均匀选择可行起点的随机放置；并非所有可能片段配置的严格均匀抽样。它用于检查时间聚集是否解释变化。
随机参考的上尾比例不是独立伪影真值，不用于证明因果。

## 重点特征：仅有pending Grade-3患者

跨患者-特征行描述：在Original与Grade-3 shadow均可计算的行中，median类SRD中位数为 {median_rows.median():.3f}%，第95百分位为 {median_rows.quantile(.95):.3f}%；P95类分别为 {p95_rows.median():.3f}% 和 {p95_rows.quantile(.95):.3f}%。此外，存在 {finite_to_nan_total} 个“Original可计算、Grade-3 shadow后不可计算”的患者-特征事件；这些事件单独作为availability下降报告，不再从汇总中静默删除。
患者内部高RSR帧相对普通帧的PCC差值中位数为 {pcc_row['median']:.6f}（患者重采样95%区间 {pcc_row['median_ci_low']:.6f} 至 {pcc_row['median_ci_high']:.6f}），FB error差值中位数为 {fb_row['median']:.6f}px（{fb_row['median_ci_low']:.6f} 至 {fb_row['median_ci_high']:.6f}）。这是时间对应关系，不能确定追踪变差与真实运动的因果方向。

下表为CSV格式；正式变化量以SRD、绝对变化、可计算性和排序为主。排名变化只在Original与shadow均为finite的患者中计算。

```csv
{compact}```

完整20项和全部病例、pending、实际删除、拓扑分层见“总体分层汇总.csv”。
随机500次全部特征值及删除帧索引在每例 random_control.npz；条件时间块结果在 block_random_control.npz。
每例frames.csv.gz为帧表、pairs.csv.gz为完整点对表，均保留原帧索引；side=0前壁、1后壁，section为从0开始的壁段索引。

## 当前判断边界与人工复核

当前结果属于数值QC敏感性分析：可以描述“更严格Grade-3屏蔽会让measurement改变多少、是否变为不可计算”，但不能独立认定被屏蔽帧一定是伪影，也不能把数值稳定直接解释成生理准确。
即使随机参考上尾很小，也要看实际变化幅度；很小的数值变化可比随机波动更有方向性。
自动Grade-3使用了RSR同步运动、PCC和FB信息，不能把它与这些量的关联解释成独立验证。
特征稳定也不等于算法整体有效性验证完成，更不证明因果方向或自动完成临床建模准入。

第一遍匿名视频与空白判断表将单独导出；研究者映射、抽样分组不交给第一遍复核者。
第一遍判断提交并锁定后才导出第二遍数值视频。主判断A/B/C/D，允许混合类别并记录把握程度。
定向抽样片段不能估计全队列伪影率。最终逐特征结合独立人工记录后才能形成稳健/敏感/证据不足判断。

## 与文献要求的关系

本轮是预先规定的验证设计；5%、500次、随机上尾计算触发均不冒充文献临床质控标准，也不改变正式PCC或Grade-3阈值。
Huang等2022研究的平均PCC>0.8是其记录级方法，不能直接套作本项目帧级删除阈值。
参考：https://doi.org/10.1109/TUFFC.2022.3165688
Wang等2022采用稳定探头、伪影处理和独立视觉评估，支持本轮人工核实的方向，并未验证我们的Grade-3分级。
参考：https://doi.org/10.3389/fphys.2022.983177
"""
    (out/"分析报告.md").write_text(report,encoding="utf-8")
    print(selected.to_string(index=False),flush=True)
    print(quality.to_string(index=False),flush=True)
    print(f"Patients={n_cases}; pending={n_pending}; block={n_block}; independent audit passed",flush=True)


if __name__=="__main__":main()
