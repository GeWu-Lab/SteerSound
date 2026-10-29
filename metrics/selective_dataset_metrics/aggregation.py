from __future__ import annotations
from collections import defaultdict
import math
import numpy as np

def _completion_values(values):
    values = [float('nan') if value is None else float(value) for value in values]
    if any((math.isfinite(value) and value not in (0.0, 1.0) for value in values)):
        raise ValueError('completion must be 0, 1, or an unmeasurable value (None/NaN)')
    return values

def summarize_completion(completions):
    """Success / measurable count; unmeasurable pitch is excluded from both."""
    values = _completion_values(completions)
    valid = [value for value in values if math.isfinite(value)]
    success_n = sum((value == 1.0 for value in valid))
    return {'total_n': len(values), 'valid_n': len(valid), 'excluded_n': len(values) - len(valid), 'success_n': success_n, 'failure_n': len(valid) - success_n, 'success_rate': success_n / len(valid) if valid else None}


def summarize_tasks(records):
    grouped=defaultdict(list)
    for record in records:grouped[str(record.get('task',record.get('task_name')))].append(record['completion'])
    return {task:summarize_completion(values) for task,values in grouped.items()}

def summarize_classifications(records):
    """Pool counts, excluding unmeasurable samples from both numerator and denominator."""
    records=list(records)
    valid=[r for r in records if r['metric_in_denominator']]
    success=sum(bool(r['metric_prediction']) for r in valid)
    errors=sum(r['error_type'] not in ('none','other') for r in valid)
    return dict(total_n=len(records),valid_n=len(valid),excluded_n=len(records)-len(valid),success_n=success,
                success_rate=success/len(valid) if valid else None,selective_error_n=errors,
                selective_error_rate=errors/len(valid) if valid else None)
