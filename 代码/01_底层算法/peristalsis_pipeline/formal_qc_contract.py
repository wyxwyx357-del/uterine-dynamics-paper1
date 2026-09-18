from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


FORMAL_QC_SEMANTICS = (
    "formal_human_artifact_and_manual_p3_position_exclusion"
)
FORMAL_QC_VERSION_FIELD = "artifact_qc_version"
FORMAL_QC_VERSION = "2.1"
FORMAL_QC_LINEAGE_ID_FIELD = "formal_qc_lineage_id"
CASE_ID_FIELD = "case_id"


def formal_analysis_eligibility(
    payload: Mapping[str, Any],
    *,
    eligible_field: str = "formal_feature_eligible",
    topology_field: str = "has_tracking_topology_risk",
) -> dict[str, object]:
    """Interpret existing upstream eligibility; never change features or QC masks.

    Both fields must be explicit booleans (or CSV/NPZ scalar equivalents).
    A missing or malformed field is unknown, not permission to enter analysis.
    This only checks imaging eligibility, not patient/cycle matching or outcomes.
    """
    flags: dict[str, bool] = {}
    unknown: list[str] = []
    for field in (eligible_field, topology_field):
        value = np.asarray(payload.get(field))
        if value.size != 1:
            unknown.append(field)
            continue
        scalar = value.reshape(-1)[0]
        if isinstance(scalar, (bool, np.bool_)):
            flags[field] = bool(scalar)
        elif isinstance(scalar, (int, float, np.integer, np.floating)) and scalar in (0, 1):
            flags[field] = bool(scalar)
        elif isinstance(scalar, str) and scalar.strip().lower() in ("true", "false", "0", "1"):
            flags[field] = scalar.strip().lower() in ("true", "1")
        else:
            unknown.append(field)

    if unknown:
        status, reason = "unknown", "missing_or_invalid:" + ",".join(unknown)
    elif flags[eligible_field] and flags[topology_field]:
        status, reason = "conflict", "eligible_true_with_tracking_topology_risk"
    elif not flags[eligible_field] or flags[topology_field]:
        status, reason = "ineligible", "upstream_formal_feature_ineligible"
        if flags[topology_field]:
            reason += ";tracking_topology_risk"
    else:
        status, reason = "eligible", "upstream_eligible_no_tracking_topology_risk"
    return {
        "formal_analysis_eligible": status == "eligible",
        "formal_analysis_status": status,
        "formal_analysis_reason": reason,
    }


def scalar_text(value: Any, *, field: str, source: object) -> str:
    """Read one auditable text scalar without guessing or coercing arrays."""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{source}: {field} must be a scalar text value")
    return str(array.reshape(-1)[0])


def validate_formal_qc_lineage(
    payload: Mapping[str, Any], *, source: object
) -> None:
    """Fail closed unless a payload carries the frozen formal-QC lineage."""

    expected = {
        "current_qc_semantics": FORMAL_QC_SEMANTICS,
        FORMAL_QC_VERSION_FIELD: FORMAL_QC_VERSION,
    }
    for field, expected_value in expected.items():
        if field not in payload:
            raise KeyError(f"{source}: required formal-QC field is missing: {field}")
        actual_value = scalar_text(payload[field], field=field, source=source)
        if actual_value != expected_value:
            raise ValueError(
                f"{source}: {field} mismatch; "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )
    if FORMAL_QC_LINEAGE_ID_FIELD not in payload:
        raise KeyError(
            f"{source}: required formal-QC field is missing: "
            f"{FORMAL_QC_LINEAGE_ID_FIELD}"
        )
    lineage_id = scalar_text(
        payload[FORMAL_QC_LINEAGE_ID_FIELD],
        field=FORMAL_QC_LINEAGE_ID_FIELD,
        source=source,
    )
    if not lineage_id.strip():
        raise ValueError(f"{source}: {FORMAL_QC_LINEAGE_ID_FIELD} must not be empty")


def validate_case_id(
    payload: Mapping[str, Any], *, expected_case_id: str, source: object
) -> None:
    """Fail closed unless a payload explicitly identifies the expected case."""

    if CASE_ID_FIELD not in payload:
        raise KeyError(f"{source}: required formal field is missing: {CASE_ID_FIELD}")
    actual_case_id = scalar_text(
        payload[CASE_ID_FIELD], field=CASE_ID_FIELD, source=source
    )
    if actual_case_id != expected_case_id:
        raise ValueError(
            f"{source}: {CASE_ID_FIELD} mismatch; "
            f"expected={expected_case_id!r}, actual={actual_case_id!r}"
        )


def validate_matching_formal_qc_lineage(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_source: object,
    right_source: object,
) -> None:
    """Fail closed unless two formal payloads share one exact lineage ID."""

    validate_formal_qc_lineage(left, source=left_source)
    validate_formal_qc_lineage(right, source=right_source)
    left_id = scalar_text(
        left[FORMAL_QC_LINEAGE_ID_FIELD],
        field=FORMAL_QC_LINEAGE_ID_FIELD,
        source=left_source,
    )
    right_id = scalar_text(
        right[FORMAL_QC_LINEAGE_ID_FIELD],
        field=FORMAL_QC_LINEAGE_ID_FIELD,
        source=right_source,
    )
    if left_id != right_id:
        raise ValueError(
            "formal_qc_lineage_id mismatch between formal inputs; "
            f"left={left_id!r}, right={right_id!r}"
        )


def validate_formal_qc_version(
    payload: Mapping[str, Any], *, source: object
) -> None:
    """Validate the frozen artifact-QC version before formal RSR is created."""

    if FORMAL_QC_VERSION_FIELD not in payload:
        raise KeyError(
            f"{source}: required formal-QC field is missing: "
            f"{FORMAL_QC_VERSION_FIELD}"
        )
    actual_value = scalar_text(
        payload[FORMAL_QC_VERSION_FIELD],
        field=FORMAL_QC_VERSION_FIELD,
        source=source,
    )
    if actual_value != FORMAL_QC_VERSION:
        raise ValueError(
            f"{source}: {FORMAL_QC_VERSION_FIELD} mismatch; "
            f"expected={FORMAL_QC_VERSION!r}, actual={actual_value!r}"
        )
