from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "代码" / "04_运行入口" / "run_clip_duration_statistics.py"
SPEC = importlib.util.spec_from_file_location("clip_duration_statistics_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _make_long_table() -> pd.DataFrame:
    rows = []
    for case_index, case_id in enumerate(("CASE_A", "CASE_B", "CASE_C", "CASE_D"), start=1):
        base = float(case_index)
        for percent in (100, 75, 50, 25):
            scale = 1.0 + (100 - percent) / 1000.0
            row = {
                "case_id": case_id,
                "target_percent": percent,
                "actual_fraction": percent / 100.0,
                "full_pair_duration_s": 48.0 + case_index,
                "window_pair_duration_s": (48.0 + case_index) * percent / 100.0,
                "rsr_valid_ratio": 0.95 - (100 - percent) / 1000.0,
                "cavity_valid_ratio": 0.90,
                "longitudinal_valid_ratio": 0.85,
                "curvature_valid_ratio": 0.80,
                "f01_f20_all_evaluable": True,
            }
            for feature_index, feature in enumerate(
                RUNNER.BIOLOGICAL_FEATURE_COLUMNS, start=1
            ):
                row[feature] = base * feature_index * scale
            rows.append(row)
    return pd.DataFrame(rows)


def test_validate_long_table_and_output_shapes():
    table = _make_long_table()
    proportions = RUNNER.validate_long_table(table)

    assert proportions == (100, 75, 50, 25)

    summary = RUNNER.build_pairwise_summary(
        table,
        proportions,
        bootstrap_repetitions=50,
        seed=1,
    )
    errors = RUNNER.build_case_errors(table, proportions)
    qc = RUNNER.build_qc_summary(table, proportions)

    assert summary.shape[0] == 20 * 3
    assert errors.shape[0] == 4 * 20 * 3
    assert qc.shape[0] == 4
    assert set(summary["target_percent"]) == {75, 50, 25}
    assert (summary["reference_percent"] == 100).all()


def test_case_errors_use_shortened_minus_reference_direction():
    table = _make_long_table()
    proportions = RUNNER.validate_long_table(table)
    errors = RUNNER.build_case_errors(table, proportions)

    selected = errors.loc[
        (errors["case_id"] == "CASE_A")
        & (errors["target_percent"] == 75)
        & (errors["feature_code"] == "F01")
    ].iloc[0]

    assert selected["difference_short_minus_reference"] > 0
    assert selected["absolute_difference"] == abs(
        selected["difference_short_minus_reference"]
    )


def test_validate_long_table_rejects_incomplete_case():
    table = _make_long_table()
    table = table.loc[
        ~((table["case_id"] == "CASE_D") & (table["target_percent"] == 25))
    ].copy()

    try:
        RUNNER.validate_long_table(table)
    except ValueError as exc:
        assert "proportional rows are incomplete" in str(exc)
    else:
        raise AssertionError("incomplete patient proportions must fail")


def test_qc_summary_preserves_pair_duration_distribution():
    table = _make_long_table()
    proportions = RUNNER.validate_long_table(table)
    qc = RUNNER.build_qc_summary(table, proportions)

    row100 = qc.loc[qc["target_percent"] == 100].iloc[0]
    assert np.isclose(row100["window_pair_duration_s_median"], 50.5)
    assert row100["all_f01_f20_evaluable_n"] == 4
