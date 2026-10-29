"""Internal frozen six-task classification rules; thresholds are unchanged."""
from __future__ import annotations
import math
from types import SimpleNamespace
from typing import Any
import numpy as np

RULES: dict[str, Any] = {
    "schema_version": "exploration-v2-completion-rules-20260807-cqt-pitch-v2",
    "generate": {"target_clap_strictly_greater_than": 0.08},
    "add": {"target_clap_strictly_greater_than": 0.04, "base_not_scored": True},
    "replace": {
        "class_presence_clap_strictly_greater_than": 0.08,
        "both_present": "larger cosine owns the region; exact tie is ambiguous",
        "success": "A=target and B=base",
    },
    "remove": {
        "A_removed_drop_db_greater_equal": 6.0,
        "B_removed_drop_db_greater_equal": 6.0,
        "B_changed_abs_drift_db_greater_equal": 3.0,
    },
    "volume": {
        "changed_abs_db_greater_equal": 1.5,
        "target_progress_inclusive": [0.8, 1.2],
    },
    "pitch": {
        "measurement": "paired multi-resolution CQT relative frequency shift",
        "bins_per_octave": [24, 36, 48],
        "max_bidirectional_error_semitone": 0.5,
        "max_multiresolution_range_semitone": 0.5,
        "max_frame_iqr_semitone": 1.0,
        "minimum_active_frames": 5,
        "changed_abs_semitone_greater_equal": 1.0,
        "target_progress_inclusive": [0.8, 1.2],
        "unmeasurable_error": "pitch_unmeasurable_after_edit",
        "unmeasurable_in_accuracy_denominator": False,
    },
}

def finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def explicit_true(value: Any) -> bool:
    """Accept only an actual true-like scalar; NaN is never true."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return bool(value)

def region_owner(target: Any, original: Any, target_name: str, original_name: str) -> str:
    threshold = RULES["replace"]["class_presence_clap_strictly_greater_than"]
    target_pass = finite(target) and float(target) > threshold
    original_pass = finite(original) and float(original) > threshold
    if target_pass and original_pass:
        if float(target) > float(original):
            return target_name
        if float(original) > float(target):
            return original_name
        return "ambiguous"
    if target_pass:
        return target_name
    if original_pass:
        return original_name
    return "unknown"

def classify_row(row: SimpleNamespace) -> dict[str, Any]:
    task = str(row.task).lower()
    result: dict[str, Any] = {
        "metric_prediction": False,
        "metric_in_denominator": True,
        "metric_measurable": True,
        "error_type": "other",
        "replace_A_owner": "",
        "replace_B_owner": "",
        "metric_a_progress": np.nan,
        "metric_b_progress": np.nan,
    }

    if task == "generate":
        success = finite(row.clap_target_global) and float(row.clap_target_global) > 0.08
        result.update(metric_prediction=success, error_type="none" if success else "other")
        return result

    if task == "add":
        success = finite(row.clap_target_global) and float(row.clap_target_global) > 0.04
        result.update(
            metric_prediction=success,
            error_type="none" if success else "target_not_added",
        )
        return result

    if task == "replace":
        a_owner = region_owner(
            row.clap_destination_edit, row.clap_source_edit, "target", "source"
        )
        b_owner = region_owner(
            row.clap_destination_preserve,
            row.clap_base_preserve_output,
            "target",
            "base",
        )
        result["replace_A_owner"] = a_owner
        result["replace_B_owner"] = b_owner
        if a_owner == "target" and b_owner == "base":
            result.update(metric_prediction=True, error_type="none")
        elif a_owner == "target" and b_owner == "target":
            result["error_type"] = "both_A_B_replaced"
        elif a_owner == "source" and b_owner == "target":
            result["error_type"] = "B_replaced_A_unchanged"
        elif a_owner == "source" and b_owner == "base":
            result["error_type"] = "neither_replaced"
        return result

    if task == "remove":
        a_removed = finite(row.remove_drop_db) and float(row.remove_drop_db) >= 6.0
        b_drop = (
            float(row.reference_preserve_db) - float(row.output_preserve_db)
            if finite(row.reference_preserve_db) and finite(row.output_preserve_db)
            else np.nan
        )
        b_removed = finite(b_drop) and float(b_drop) >= 6.0
        b_changed_not_removed = (
            finite(row.preserve_drift_db)
            and float(row.preserve_drift_db) >= 3.0
            and not b_removed
        )
        if a_removed and not b_removed and not b_changed_not_removed:
            result.update(metric_prediction=True, error_type="none")
        elif a_removed and b_removed:
            result["error_type"] = "A_removed_B_also_removed"
        elif not a_removed and not b_removed and not b_changed_not_removed:
            result["error_type"] = "A_not_removed_B_kept"
        elif not a_removed and b_removed:
            result["error_type"] = "A_not_removed_B_removed"
        return result

    if task == "volume":
        requested = float(row.requested_change) if finite(row.requested_change) else np.nan
        a_delta = float(row.edit_delta_db) if finite(row.edit_delta_db) else np.nan
        b_delta = (
            float(row.output_preserve_db) - float(row.reference_preserve_db)
            if finite(row.output_preserve_db) and finite(row.reference_preserve_db)
            else np.nan
        )
        a_progress = a_delta / requested if finite(requested) and abs(requested) > 1e-8 else np.nan
        b_progress = b_delta / requested if finite(requested) and abs(requested) > 1e-8 else np.nan
        result["metric_a_progress"] = a_progress
        result["metric_b_progress"] = b_progress
        a_target = finite(a_progress) and 0.8 <= a_progress <= 1.2
        b_target = finite(b_progress) and 0.8 <= b_progress <= 1.2
        a_unchanged = finite(a_delta) and abs(a_delta) < 1.5
        b_unchanged = finite(b_delta) and abs(b_delta) < 1.5
        if a_target and b_unchanged:
            result.update(metric_prediction=True, error_type="none")
        elif a_target and b_target:
            result["error_type"] = "A_progress_B_also_progress"
        elif a_unchanged and b_target:
            result["error_type"] = "A_unchanged_B_progress"
        elif a_unchanged and b_unchanged:
            result["error_type"] = "A_unchanged_B_unchanged"
        return result

    if task == "pitch":
        output_valid = explicit_true(row.output_edit_pitch_valid) and explicit_true(
            row.output_preserve_pitch_valid
        )
        if not output_valid:
            result.update(
                metric_in_denominator=False,
                metric_measurable=False,
                error_type="pitch_unmeasurable_after_edit",
            )
            return result
        requested = float(row.requested_change) if finite(row.requested_change) else np.nan
        a_delta = (
            float(row.edit_delta_semitone) if finite(row.edit_delta_semitone) else np.nan
        )
        b_delta = (
            float(row.preserve_delta_semitone)
            if finite(row.preserve_delta_semitone)
            else np.nan
        )
        a_progress = a_delta / requested if finite(requested) and abs(requested) > 1e-8 else np.nan
        b_progress = b_delta / requested if finite(requested) and abs(requested) > 1e-8 else np.nan
        result["metric_a_progress"] = a_progress
        result["metric_b_progress"] = b_progress
        a_target = finite(a_progress) and 0.8 <= a_progress <= 1.2
        b_target = finite(b_progress) and 0.8 <= b_progress <= 1.2
        a_unchanged = finite(a_delta) and abs(a_delta) < 1.0
        b_unchanged = finite(b_delta) and abs(b_delta) < 1.0
        if a_target and b_unchanged:
            result.update(metric_prediction=True, error_type="none")
        elif a_target and b_target:
            result["error_type"] = "A_progress_B_also_progress"
        elif a_unchanged and b_target:
            result["error_type"] = "A_unchanged_B_progress"
        elif a_unchanged and b_unchanged:
            result["error_type"] = "A_unchanged_B_unchanged"
        return result

    raise ValueError(f"unsupported task: {task!r}")
