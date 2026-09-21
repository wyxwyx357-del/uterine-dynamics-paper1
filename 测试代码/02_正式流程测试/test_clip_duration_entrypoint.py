from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "代码" / "04_运行入口" / "run_clip_duration_robustness.py"
SPEC = importlib.util.spec_from_file_location("clip_duration_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _row(case_id: str, percent: int, duration: float) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": case_id,
        "target_percent": percent,
        "target_fraction": percent / 100.0,
        "actual_fraction": percent / 100.0,
        "full_pair_frames": 100,
        "window_pair_frames": percent,
        "full_pair_duration_s": 50.0,
        "window_pair_duration_s": duration,
        "window_start_pair_frame": 0,
        "window_stop_pair_frame_exclusive": percent,
        "window_start_s": 0.0,
        "window_stop_s": duration,
        "rsr_valid_ratio": 0.9,
        "cavity_valid_ratio": 0.8,
        "longitudinal_valid_ratio": 0.7,
        "curvature_valid_ratio": 0.6,
        "dicom_physical_curvature_available": True,
        "dicom_physical_curvature_rate_available": True,
        "f01_f20_missing_codes": "",
        "f01_f20_all_evaluable": True,
    }
    for feature in RUNNER.BIOLOGICAL_FEATURE_COLUMNS:
        row[feature] = 1.0
    return row


def test_inventory_has_one_row_per_case_and_records_each_window():
    rows = [
        _row("CASE_A", 100, 50.0),
        _row("CASE_A", 75, 37.5),
        _row("CASE_A", 50, 25.0),
        _row("CASE_A", 25, 12.5),
        _row("CASE_B", 100, 40.0),
        _row("CASE_B", 75, 30.0),
        _row("CASE_B", 50, 20.0),
        _row("CASE_B", 25, 10.0),
    ]
    inventory = RUNNER._build_inventory(pd.DataFrame(rows))

    assert list(inventory["case_id"]) == ["CASE_A", "CASE_B"]
    assert inventory.loc[0, "p100_window_pair_duration_s"] == 50.0
    assert inventory.loc[0, "p025_window_pair_duration_s"] == 12.5
    assert inventory.loc[1, "p100_window_pair_duration_s"] == 40.0
    assert inventory.loc[1, "p025_window_pair_duration_s"] == 10.0
    assert inventory.loc[0, "p050_rsr_valid_ratio"] == 0.9
    assert inventory.loc[0, "p075_curvature_valid_ratio"] == 0.6


def test_inventory_rejects_missing_100_percent_reference():
    rows = [
        _row("CASE_A", 75, 37.5),
        _row("CASE_A", 50, 25.0),
        _row("CASE_A", 25, 12.5),
    ]
    try:
        RUNNER._build_inventory(pd.DataFrame(rows))
    except RuntimeError as exc:
        assert "100% reference" in str(exc)
    else:
        raise AssertionError("missing 100% reference must fail")


def test_missing_feature_codes_use_frozen_f01_f20_order():
    row = pd.Series({feature: 1.0 for feature in RUNNER.BIOLOGICAL_FEATURE_COLUMNS})
    row[RUNNER.BIOLOGICAL_FEATURE_COLUMNS[0]] = np.nan
    row[RUNNER.BIOLOGICAL_FEATURE_COLUMNS[-1]] = np.nan

    assert RUNNER._missing_feature_codes(row) == "F01|F20"
