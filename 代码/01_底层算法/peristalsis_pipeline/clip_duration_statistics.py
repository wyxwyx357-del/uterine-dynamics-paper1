from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


DEFAULT_BOOTSTRAP_REPETITIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260918


@dataclass(frozen=True)
class ICCResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    bootstrap_valid_repetitions: int


def icc_absolute_agreement_single(matrix: np.ndarray) -> float:
    """Two-way absolute-agreement single-measure ICC, ICC(A,1).

    Rows are patients and columns are fixed duration conditions. The point
    estimate uses the standard two-way ANOVA absolute-agreement formula.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("ICC matrix must be two-dimensional")
    n, k = values.shape
    if n < 2 or k < 2:
        raise ValueError("ICC(A,1) requires at least two patients and conditions")
    if not np.all(np.isfinite(values)):
        raise ValueError("ICC matrix must be complete and finite")

    grand = float(np.mean(values))
    row_means = np.mean(values, axis=1)
    col_means = np.mean(values, axis=0)

    ss_rows = float(k * np.sum((row_means - grand) ** 2))
    ss_cols = float(n * np.sum((col_means - grand) ** 2))
    residual = (
        values
        - row_means[:, None]
        - col_means[None, :]
        + grand
    )
    ss_error = float(np.sum(residual**2))

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = (
        ms_rows
        + (k - 1) * ms_error
        + (k * (ms_cols - ms_error) / n)
    )
    if denominator == 0.0:
        return float("nan")
    return float((ms_rows - ms_error) / denominator)


def bootstrap_icc_absolute_agreement_single(
    matrix: np.ndarray,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ICCResult:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("ICC bootstrap requires a patient x condition matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("ICC bootstrap matrix must be complete and finite")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    estimate = icc_absolute_agreement_single(values)
    rng = np.random.default_rng(seed)
    sampled: list[float] = []
    n = values.shape[0]
    for _ in range(repetitions):
        indices = rng.integers(0, n, size=n)
        value = icc_absolute_agreement_single(values[indices])
        if np.isfinite(value):
            sampled.append(float(value))

    if len(sampled) < max(100, repetitions // 2):
        lower = float("nan")
        upper = float("nan")
    else:
        lower, upper = np.percentile(sampled, [2.5, 97.5])
        lower = float(lower)
        upper = float(upper)

    return ICCResult(
        estimate=float(estimate),
        ci_lower=lower,
        ci_upper=upper,
        bootstrap_valid_repetitions=len(sampled),
    )


def pairwise_duration_statistics(
    reference: Iterable[float],
    shortened: Iterable[float],
    *,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Summarize patient-level agreement of a shortened window vs 100%."""

    ref = np.asarray(list(reference), dtype=float)
    short = np.asarray(list(shortened), dtype=float)
    if ref.shape != short.shape:
        raise ValueError("reference and shortened vectors must have the same shape")

    finite = np.isfinite(ref) & np.isfinite(short)
    ref = ref[finite]
    short = short[finite]
    n = int(ref.size)

    result: dict[str, float | int] = {
        "n_pairs": n,
        "icc_a1": float("nan"),
        "icc_ci_lower": float("nan"),
        "icc_ci_upper": float("nan"),
        "icc_bootstrap_valid_repetitions": 0,
        "spearman_rho": float("nan"),
        "mean_difference_short_minus_reference": float("nan"),
        "sd_difference": float("nan"),
        "bland_altman_loa_lower": float("nan"),
        "bland_altman_loa_upper": float("nan"),
        "median_absolute_difference": float("nan"),
        "p95_absolute_difference": float("nan"),
        "relative_error_evaluable_n": 0,
        "median_absolute_relative_error_pct": float("nan"),
        "p95_absolute_relative_error_pct": float("nan"),
    }
    if n == 0:
        return result

    difference = short - ref
    absolute_difference = np.abs(difference)
    result["mean_difference_short_minus_reference"] = float(np.mean(difference))
    result["median_absolute_difference"] = float(np.median(absolute_difference))
    result["p95_absolute_difference"] = float(
        np.percentile(absolute_difference, 95)
    )

    if n >= 2:
        sd = float(np.std(difference, ddof=1))
        mean_difference = float(np.mean(difference))
        result["sd_difference"] = sd
        result["bland_altman_loa_lower"] = mean_difference - 1.96 * sd
        result["bland_altman_loa_upper"] = mean_difference + 1.96 * sd

        rho = spearmanr(ref, short).statistic
        result["spearman_rho"] = float(rho) if np.isfinite(rho) else float("nan")

        icc = bootstrap_icc_absolute_agreement_single(
            np.column_stack([ref, short]),
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
        result["icc_a1"] = icc.estimate
        result["icc_ci_lower"] = icc.ci_lower
        result["icc_ci_upper"] = icc.ci_upper
        result["icc_bootstrap_valid_repetitions"] = (
            icc.bootstrap_valid_repetitions
        )

    nonzero_reference = ref != 0.0
    relative = absolute_difference[nonzero_reference] / np.abs(
        ref[nonzero_reference]
    )
    result["relative_error_evaluable_n"] = int(relative.size)
    if relative.size:
        result["median_absolute_relative_error_pct"] = float(
            np.median(relative) * 100.0
        )
        result["p95_absolute_relative_error_pct"] = float(
            np.percentile(relative, 95) * 100.0
        )

    return result
