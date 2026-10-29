"""Manifest I/O and validation for source-audio preparation (16 kHz timing)."""
import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    raise ValueError("Use a .csv or .parquet manifest: " + str(path))


def write_table(table, path):
    path = Path(path)
    if path.suffix not in (".csv", ".parquet"):
        raise ValueError("Output must end in .csv or .parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)


def missing(value):
    return value is None or (isinstance(value, str) and not value.strip()) or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    )


def spans(value, total_samples, field):
    """Parse full-file (start, length) annotations and validate their bounds."""
    if missing(value):
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if value is None:
        return []
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(field + " must be a list of (start_sample, length_samples)")
    out = []
    for pair in value:
        if not isinstance(pair, (list, tuple, np.ndarray)) or len(pair) != 2:
            raise ValueError(field + ": expected (start, length), not start/end")
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in pair):
            raise ValueError(field + ": coordinates must be integers")
        start, length = map(int, pair)
        if start < 0 or length <= 0 or start + length > total_samples:
            raise ValueError(f"{field}: {pair} outside [0, {total_samples}]")
        if field == "point_audio" and not 8000 <= length < 16000:
            raise ValueError("point_audio lengths must be 8000..15999 samples (0.5 to <1 s)")
        if field == "piece_audio" and length < 16000:
            raise ValueError("piece_audio lengths must be >=16000 samples (>=1 s)")
        out.append((start, length))
    return out


def normalize(table, audio_root, kind, require_timing=False, decode=False):
    """Normalize either source type to the same five-column whole-file schema."""
    text_column = "label" if kind == "labels" else "caption"
    required = {"path", text_column}
    if require_timing:
        required.update(("point_audio", "piece_audio"))
    absent = required - set(table.columns)
    if absent:
        raise ValueError("Missing columns: " + ", ".join(sorted(absent)))
    if table.empty:
        raise ValueError("Manifest contains no records")
    root = Path(audio_root).resolve()
    rows = []
    for index, row in table.iterrows():
        try:
            rel = row["path"]
            text = row[text_column]
            if not isinstance(rel, str) or not rel.strip():
                raise ValueError("path must be a nonempty relative filename")
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("path must be relative to --audio-root, without '..'")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(text_column + " must be nonempty text")
            info = sf.info(root / rel)
            full = info.frames / info.samplerate
            # Accept redundant legacy full-file bounds, but never reinterpret
            # excerpt-relative timings as full-file coordinates silently.
            start_value = row.get("chunk_start_second")
            end_value = row.get("chunk_end_second")
            start = 0.0 if missing(start_value) else float(start_value)
            end = full if missing(end_value) else float(end_value)
            if not (math.isfinite(start) and math.isfinite(end) and start == 0 and abs(end - full) <= 1 / SR):
                raise ValueError("Preparation uses whole audio files for both collections. "
                                 "Save the desired excerpt as a separate audio file and annotate from its beginning.")
            duration = full
            minimum_ok = duration >= 1 if kind == "labels" else duration > 1
            if not minimum_ok:
                raise ValueError("Label files need >=1 s; caption files need >1 s")
            supplied_duration = row.get("duration")
            if not missing(supplied_duration):
                declared = float(supplied_duration)
                if not math.isfinite(declared) or abs(declared - duration) > .002:
                    raise ValueError(f"duration {supplied_duration} disagrees with full audio ({duration:.6f} s)")
            total = math.ceil(info.frames * SR / info.samplerate)
            normalized = {"path": rel, text_column: text.strip(), "duration": duration}
            for field in ("point_audio", "piece_audio"):
                parsed = spans(row.get(field), total, field)
                normalized[field] = str(parsed) if parsed else None
            if decode:
                wave, _ = sf.read(root / rel, dtype="float32", always_2d=True)
                if not wave.size or not np.isfinite(wave).all():
                    raise ValueError("Audio is empty or contains non-finite samples")
            rows.append(normalized)
        except Exception as exc:
            raise ValueError(f"Row {index} ({row.get('path', '?')}): {exc}") from exc
    return pd.DataFrame(rows)


def buckets(table):
    point = table.point_audio.map(lambda x: not missing(x) and str(x) != "[]")
    piece = table.piece_audio.map(lambda x: not missing(x) and str(x) != "[]")
    return {"all": table, "point": table[point], "piece": table[piece],
            "point_piece": table[point & piece]}


def validate_pools(config, labels, captions=None, aliases=None):
    """Check pool coverage and category exclusions before sampling."""
    weights = config["task_probs"]
    if len(weights) != 6 or any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError("task_probs needs six nonnegative weights with positive sum")
    alias_to_key = {}
    for key, values in (aliases or {}).items():
        if key not in set(labels.label):
            raise ValueError("Alias dictionary key absent from labels: " + key)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("Alias values must be nonempty lists of category phrases")
        for value in [key] + values:
            value = value.strip().casefold()
            if value in alias_to_key and alias_to_key[value] != key:
                raise ValueError("Ambiguous category alias: " + value)
            alias_to_key[value] = key
    for label in labels.label.unique():
        alias_to_key.setdefault(label.strip().casefold(), label)
    def canonical(text):
        return alias_to_key.get(text.strip().casefold(), text)

    pool = buckets(labels)
    base_probs = config["base_audio_num_probs"]
    if len(base_probs) != 3 or any(not math.isfinite(x) or x < 0 for x in base_probs) or sum(base_probs) <= 0:
        raise ValueError("base_audio_num_probs needs three nonnegative weights with positive sum")
    max_base = max(i + 1 for i, weight in enumerate(base_probs) if weight > 0)
    has_base = config.get("use_base_prob", 1) > 0
    required = {"all": max_base if weights[0] > 0 else 1}
    label_sets = {name: set(table.label.map(canonical)) for name, table in pool.items()}

    def require(name, count):
        required[name] = max(required.get(name, 0), count)

    mode = config.get("specific_time")
    time_pools = {mode} if mode in ("point", "piece") else {"all"}
    if mode is None:
        time_pools = {name for name, weight in zip(("all", "point", "piece"), config["specific_time_probs"]) if weight > 0}
    if any(weights[i] > 0 for i in (1, 4, 5)):
        # These tasks draw the edited event after choosing the base categories.
        for name in time_pools:
            require(name, max_base + 1 if has_base else 1)
        if has_base:
            require("all", max_base + 1)
    caption_targets = set(config.get("required_long_prompt_tasks", [])) if captions is not None else set()
    for task in (0, 2):
        if weights[task] > 0 and task not in caption_targets:
            for name in time_pools:
                require(name, 1)
    if captions is not None:
        # The existing loader initializes all four caption buckets unconditionally.
        for name, subset in buckets(captions).items():
            if subset.empty:
                raise ValueError(f"Caption pool '{name}' is empty. Supply matching real timing annotations; "
                                 "do not invent short events. Use --labels-only to train without captions.")
    if weights[3] > 0:
        # Replace selects its source/destination first, then an independent base.
        replace_pools = {"point", "piece"} if mode in (None, "No") else {mode}
        for name in replace_pools:
            require(name, 1 if 3 in caption_targets else 2)
            if 3 in caption_targets:
                destination = set(buckets(captions)[name].caption.map(canonical))
                if len(destination) == 1 and destination <= label_sets[name]:
                    raise ValueError(f"Caption pool '{name}' needs a replacement destination distinct from every source category")
        if has_base:
            destination_overlaps = captions is None or bool(set(captions.caption.map(canonical)) & label_sets["all"])
            require("all", max_base + (2 if destination_overlaps else 1))
    for name, needed in required.items():
        count = len(label_sets[name])
        if count < needed:
            raise ValueError(f"Label pool '{name}' has {count} distinct categories; this setup needs at least {needed}. "
                             "Add annotated categories, or explicitly reduce base_audio_num_probs/task_probs. "
                             "Replace with specific_time='No' requires both point and piece pools.")
    return {"labels": {k: len(v) for k, v in pool.items()},
            "captions": None if captions is None else {k: len(v) for k, v in buckets(captions).items()}}
