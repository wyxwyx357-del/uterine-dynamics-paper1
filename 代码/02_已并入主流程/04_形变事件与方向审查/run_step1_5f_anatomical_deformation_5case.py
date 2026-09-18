#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract adjacent-frame anatomical deformation evidence in five cases."""

from __future__ import annotations

from peristalsis_pipeline.project_layout import output_dir

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "代码" / "01_底层算法"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from peristalsis_pipeline.anatomical_deformation_features import (
    anatomical_deformation_from_tracking,
)
from peristalsis_pipeline.formal_qc_contract import (
    FORMAL_QC_LINEAGE_ID_FIELD,
    FORMAL_QC_VERSION_FIELD,
    validate_case_id,
    validate_formal_qc_lineage,
)


CASES = (
    "CASE_003",
    "CASE_002",
    "CASE_001",
    "CASE_005",
    "CASE_004",
)
TRACKING_DIR = (
    output_dir(PROJECT_ROOT, "step1_4_p3_image_boundary_correction_v2_5case")
)
FOURWAY_DIR = output_dir(PROJECT_ROOT, "step1_5f_rsr_fourway_v1_5case")
DEFAULT_OUTPUT = (
    output_dir(PROJECT_ROOT, "step1_5f_anatomical_deformation_v1_5case")
)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def finite_abs_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.abs(np.asarray(values)[np.isfinite(values)])
    return float(np.percentile(finite, percentile)) if len(finite) else float("nan")


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    x = np.asarray(left[valid], dtype=np.float64)
    y = np.asarray(right[valid], dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_map_figure(path: Path, features: dict[str, np.ndarray], fps: float) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    maps = (
        ("腔宽应变率（负值=闭合）", features["qc_cavity_width_strain_rate_s"]),
        ("前后壁协调内收速度", features["qc_coordinated_inward_velocity_px_s"]),
        ("前壁沿壁应变率", features["qc_longitudinal_wall_strain_rate_s"][:, 0]),
        ("后壁沿壁应变率", features["qc_longitudinal_wall_strain_rate_s"][:, 1]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    duration_s = (len(maps[0][1]) - 1) / fps
    for axis, (title, values) in zip(axes.ravel(), maps):
        finite = np.abs(values[np.isfinite(values)])
        limit = max(float(np.percentile(finite, 99)) if len(finite) else 0.1, 1e-5)
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=(0.0, duration_s, 0.0, 1.0),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("宫颈→宫底归一化位置")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("解剖曲线/壁带形变证据（非正式传播）")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_case(case_id: str, output: Path) -> dict[str, Any]:
    tracking = load_npz(
        TRACKING_DIR / f"{case_id}_p3_image_boundary_evidence_v2.npz"
    )
    fourway = load_npz(FOURWAY_DIR / f"{case_id}_rsr_fourway_v1.npz")
    fourway_source = FOURWAY_DIR / f"{case_id}_rsr_fourway_v1.npz"
    validate_formal_qc_lineage(fourway, source=fourway_source)
    validate_case_id(fourway, expected_case_id=case_id, source=fourway_source)
    fps = float(tracking["fps"])
    sections = int(tracking["longitudinal_section_count"])
    features = anatomical_deformation_from_tracking(
        tracking, section_count=sections, fps=fps
    )
    pair_qc_valid = np.isfinite(
        fourway["current_qc_radial_strain_rate_s"]
    )
    cavity_qc_valid = pair_qc_valid[:, 0] & pair_qc_valid[:, 1]
    longitudinal_qc_valid = pair_qc_valid[:, :, :-1] & pair_qc_valid[:, :, 1:]
    curvature_qc_valid = np.zeros_like(pair_qc_valid)
    curvature_qc_valid[:, :, 1:-1] = (
        pair_qc_valid[:, :, :-2]
        & pair_qc_valid[:, :, 1:-1]
        & pair_qc_valid[:, :, 2:]
    )

    qc_wall_normal = features["wall_normal_velocity_px_s"].copy()
    qc_wall_inward = features["wall_inward_velocity_px_s"].copy()
    qc_cavity_strain = features["cavity_width_strain_rate_s"].copy()
    qc_closing = features["cavity_closing_rate_px_s"].copy()
    qc_coordinated = features["coordinated_inward_velocity_px_s"].copy()
    qc_asymmetry = features["inward_velocity_asymmetry_px_s"].copy()
    qc_longitudinal = features["longitudinal_wall_strain_rate_s"].copy()
    qc_curvature = features["wall_curvature_px_inv"].copy()
    qc_curvature_change_rate = features[
        "wall_curvature_change_rate_px_inv_s"
    ].copy()
    qc_wall_normal[~pair_qc_valid] = np.nan
    qc_wall_inward[~pair_qc_valid] = np.nan
    for values in (qc_cavity_strain, qc_closing, qc_coordinated, qc_asymmetry):
        values[~cavity_qc_valid] = np.nan
    qc_longitudinal[~longitudinal_qc_valid] = np.nan
    qc_curvature[~curvature_qc_valid] = np.nan
    qc_curvature_change_rate[~curvature_qc_valid] = np.nan
    output_fields = {
        **features,
        "case_id": np.asarray(case_id),
        "qc_wall_normal_velocity_px_s": qc_wall_normal,
        "qc_wall_inward_velocity_px_s": qc_wall_inward,
        "qc_cavity_width_strain_rate_s": qc_cavity_strain,
        "qc_cavity_closing_rate_px_s": qc_closing,
        "qc_coordinated_inward_velocity_px_s": qc_coordinated,
        "qc_inward_velocity_asymmetry_px_s": qc_asymmetry,
        "qc_longitudinal_wall_strain_rate_s": qc_longitudinal,
        "qc_wall_curvature_px_inv": qc_curvature,
        "qc_wall_curvature_change_rate_px_inv_s": qc_curvature_change_rate,
        "pair_qc_valid": pair_qc_valid,
        "cavity_qc_valid": cavity_qc_valid,
        "longitudinal_qc_valid": longitudinal_qc_valid,
        "normalized_cervix_to_fundus_position": np.linspace(
            0.0, 1.0, sections, dtype=np.float32
        ),
        "fps": np.asarray(fps, dtype=np.float32),
        "evidence_layer": np.asarray("roi_deformation_activity"),
        "qc_feature_branch": np.asarray(
            "formal_human_artifact_and_manual_p3_position_qc"
        ),
        "qc_feature_branch_is_formal": np.asarray(True),
        "formal_propagation_released": np.asarray(False),
        "current_qc_semantics": fourway["current_qc_semantics"],
        FORMAL_QC_VERSION_FIELD: fourway[FORMAL_QC_VERSION_FIELD],
        FORMAL_QC_LINEAGE_ID_FIELD: fourway[FORMAL_QC_LINEAGE_ID_FIELD],
    }
    np.savez_compressed(
        output / f"{case_id}_anatomical_deformation_v1.npz", **output_fields
    )
    write_map_figure(
        output / f"{case_id}_解剖形变证据图.png", output_fields, fps
    )
    finite_cavity = np.isfinite(qc_cavity_strain)
    return {
        "病例": case_id,
        "腔宽应变率有限百分比": round(100 * float(np.mean(finite_cavity)), 4),
        "腔宽应变率绝对值P95_1_s": round(
            finite_abs_percentile(qc_cavity_strain, 95), 6
        ),
        "协调内收速度绝对值P95_px_s": round(
            finite_abs_percentile(qc_coordinated, 95), 6
        ),
        "沿壁应变率绝对值P95_1_s": round(
            finite_abs_percentile(qc_longitudinal, 95), 6
        ),
        "前后壁内收速度相关系数": round(
            finite_correlation(qc_wall_inward[:, 0], qc_wall_inward[:, 1]), 6
        ),
        "同向内收或外扩比例": round(
            float(np.mean(features["inward_same_sign"][finite_cavity]))
            if np.any(finite_cavity)
            else float("nan"),
            6,
        ),
        "证据层": "ROI局部形变",
        "汇总质控分支": "正式人工伪影/人工P3位置质控",
        "汇总质控分支是否正式": True,
        "正式传播是否释放": False,
    }


def write_readme(path: Path) -> None:
    path.write_text(
        """# 解剖形变证据 v1

本分支从相邻帧LK测得的径向点对中心计算，不使用P3坐标自身的帧间变化作为形变。

输出包括：前后壁间腔宽应变率、腔闭合速度、两侧局部向内速度、协调内收速度、内收不对称性、沿壁相邻段应变率，以及前后壁局部曲率和曲率变化率。整体刚体平移在腔宽和沿壁距离差中抵消；单侧速度另减去中线估计的整体运动预测。

质控副本只应用当前正式授权的人工确认伪影排除和人工确认P3位置错误点对排除。LK红色影子结果仍完整保留在四路RSR输出中，作为独立敏感性/警告信息，不再用于本分支的硬性NaN掩膜；原始特征仍完整保留。

解剖形变结果仍只属于ROI局部形变证据，不能单独证明宫颈—宫底传播，也不释放正式传播。当前速度单位为缩放后像素/秒；在DICOM毫米标定接入前不能换算为mm/s。
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case_id, args.output_dir) for case_id in CASES]
    write_csv(args.output_dir / "五例解剖形变汇总.csv", rows)
    write_readme(args.output_dir / "README_中文说明.md")
    for row in rows:
        print(
            f"[{row['病例']}] cavity_p95={row['腔宽应变率绝对值P95_1_s']} 1/s"
        )


if __name__ == "__main__":
    main()
