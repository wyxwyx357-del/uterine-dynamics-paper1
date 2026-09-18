from __future__ import annotations

import numpy as np

from peristalsis_pipeline.clip_duration_statistics import (
    bootstrap_icc_absolute_agreement_single,
    icc_absolute_agreement_single,
    pairwise_duration_statistics,
)


def test_icc_a1_is_one_for_identical_nonconstant_conditions():
    values = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    matrix = np.column_stack([values, values])

    assert icc_absolute_agreement_single(matrix) == 1.0


def test_pairwise_statistics_are_zero_error_for_identical_values():
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
    result = pairwise_duration_statistics(
        values,
        values,
        bootstrap_repetitions=200,
        seed=123,
    )

    assert result["n_pairs"] == 6
    assert result["icc_a1"] == 1.0
    assert result["mean_difference_short_minus_reference"] == 0.0
    assert result["sd_difference"] == 0.0
    assert result["bland_altman_loa_lower"] == 0.0
    assert result["bland_altman_loa_upper"] == 0.0
    assert result["median_absolute_difference"] == 0.0
    assert result["p95_absolute_difference"] == 0.0
    assert result["median_absolute_relative_error_pct"] == 0.0
    assert result["spearman_rho"] == 1.0


def test_pairwise_statistics_use_only_complete_pairs_and_handle_zero_reference():
    reference = np.asarray([1.0, 0.0, np.nan, 4.0])
    shortened = np.asarray([1.1, 0.2, 3.0, np.nan])

    result = pairwise_duration_statistics(
        reference,
        shortened,
        bootstrap_repetitions=20,
        seed=1,
    )

    assert result["n_pairs"] == 2
    assert result["relative_error_evaluable_n"] == 1
    assert np.isclose(result["median_absolute_relative_error_pct"], 10.0)


def test_bootstrap_icc_is_deterministic_for_fixed_seed():
    reference = np.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0])
    shortened = reference * 1.01

    first = bootstrap_icc_absolute_agreement_single(
        np.column_stack([reference, shortened]),
        repetitions=300,
        seed=20260918,
    )
    second = bootstrap_icc_absolute_agreement_single(
        np.column_stack([reference, shortened]),
        repetitions=300,
        seed=20260918,
    )

    assert first == second
    assert np.isfinite(first.estimate)
    assert np.isfinite(first.ci_lower)
    assert np.isfinite(first.ci_upper)
