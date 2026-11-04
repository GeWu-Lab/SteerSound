"""Aggregate continuous per-audio scores without loading scoring models."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np


def summarize_metric(values):
    """Mean of finite values only; report the actual numerator and denominator.

    None, NaN and infinity are excluded. Empty/all-unmeasurable input has
    mean=None and denominator=0, rather than an artificial zero error.
    """
    values = [float('nan') if value is None else float(value) for value in values]
    valid = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return {
        'total_n': len(values),
        'valid_n': int(valid.size),
        'excluded_n': len(values) - int(valid.size),
        'sum': float(valid.sum()) if valid.size else 0.0,
        'mean': float(valid.mean()) if valid.size else None,
    }


def summarize_tasks(records, *, metric_names):
    """Group records by task; each named metric has its own valid denominator.

    Each record contains task (or legacy task_name) and per-audio metric values.
    Missing metric values are excluded, never imputed with zero.
    """
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    names = list(dict.fromkeys(metric_names))
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError('metric_names must contain nonempty metric names')
    groups = defaultdict(list)
    for record in records:
        task = record.get('task', record.get('task_name'))
        if task not in {'generate', 'add', 'remove', 'replace', 'pitch', 'volume'}:
            raise ValueError(f'unknown task: {task!r}')
        groups[task].append(record)
    return {
        task: {
            'total_n': len(rows),
            'metrics': {name: summarize_metric(row.get(name) for row in rows) for name in names},
        }
        for task, rows in sorted(groups.items())
    }


def tbg(current,baseline,tasks=None):
    """100 * (geomean of task-oriented ratios - 1); not an arithmetic average."""
    selected=list(tasks) if tasks is not None else list(current)
    if not selected:raise ValueError('No tasks to aggregate')
    ratios=[]
    for task in selected:
        a,b=float(current[task]),float(baseline[task])
        if not math.isfinite(a) or not math.isfinite(b) or a<=0 or b<=0:
            raise ValueError(f'TBG requires finite positive primary scores: {task}')
        ratios.append(b/a if task in ('pitch','volume') else a/b)
    return 100*(math.exp(sum(math.log(v) for v in ratios)/len(ratios))-1)

def tbg_all(current,baseline):return tbg(current,baseline)

def tbg_selective(current,baseline):
    return tbg(current,baseline,[t for t in current if t in ('remove','replace','pitch','volume')])
