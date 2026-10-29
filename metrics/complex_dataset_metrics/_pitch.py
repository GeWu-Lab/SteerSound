"""Bidirectional, multi-resolution CQT pitch measurement.

Package-local pitch-measurement core, preserving the
original measurability gates and numerical constants.
"""

from __future__ import annotations

import math

import librosa
import numpy as np


SR = 16_000
HOP = 256
RESOLUTIONS = (24, 36, 48)
MAX_SHIFT_SEMITONES = 6.0
MAX_BIDIRECTIONAL_ERROR = 0.5
MAX_MULTIRESOLUTION_RANGE = 0.5
MAX_FRAME_IQR = 1.0
MIN_ACTIVE_FRAMES = 5


def frequency_smooth(value: np.ndarray, width: int = 17) -> np.ndarray:
    kernel = np.ones(width, dtype=np.float64) / width
    return np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="same"), 0, value)


def cqt_features(wave: np.ndarray, bins_per_octave: int) -> np.ndarray:
    magnitude = np.abs(
        librosa.cqt(
            wave,
            sr=SR,
            hop_length=HOP,
            fmin=float(librosa.note_to_hz("C1")),
            n_bins=7 * bins_per_octave,
            bins_per_octave=bins_per_octave,
            pad_mode="reflect",
        )
    )
    value = np.log1p(100.0 * magnitude)
    return value - frequency_smooth(value)


def active_mask(value: np.ndarray) -> np.ndarray:
    energy = np.linalg.norm(value, axis=0)
    if not len(energy) or float(np.max(energy)) <= 1e-8:
        return np.zeros(len(energy), dtype=bool)
    return energy >= max(1e-6, 0.1 * float(np.max(energy)))


def directional_alignment(
    reference: np.ndarray,
    edited: np.ndarray,
    bins_per_octave: int,
    valid_frames: np.ndarray,
) -> tuple[float, float]:
    """Return global shift and per-frame shift IQR, in semitones."""
    bins_per_semitone = bins_per_octave / 12.0
    max_bins = int(round(MAX_SHIFT_SEMITONES * bins_per_semitone))
    frame_scores: list[np.ndarray] = []
    for offset in range(-max_bins, max_bins + 1):
        if offset > 0:
            left, right = reference[:-offset], edited[offset:]
        elif offset < 0:
            left, right = reference[-offset:], edited[:offset]
        else:
            left, right = reference, edited
        denominator = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0) + 1e-9
        frame_scores.append(np.sum(left * right, axis=0) / denominator)
    scores = np.stack(frame_scores, axis=0)
    if valid_frames.sum() < MIN_ACTIVE_FRAMES:
        return math.nan, math.nan
    aggregate = np.median(scores[:, valid_frames], axis=1)
    peak_index = int(np.argmax(aggregate))
    peak_bin = peak_index - max_bins
    fractional = 0.0
    if 0 < peak_index < len(aggregate) - 1:
        before, peak, after = aggregate[peak_index - 1 : peak_index + 2]
        curvature = before - 2.0 * peak + after
        if abs(curvature) > 1e-12:
            fractional = float(0.5 * (before - after) / curvature)
    global_shift = float((peak_bin + fractional) / bins_per_semitone)
    frame_bins = np.argmax(scores[:, valid_frames], axis=0) - max_bins
    frame_shifts = frame_bins.astype(np.float64) / bins_per_semitone
    frame_iqr = float(np.quantile(frame_shifts, 0.75) - np.quantile(frame_shifts, 0.25))
    return global_shift, frame_iqr


def region_measure(reference: np.ndarray, edited: np.ndarray) -> dict[str, object]:
    forward: list[float] = []
    reverse: list[float] = []
    frame_iqr: list[float] = []
    active_counts: list[int] = []
    for resolution in RESOLUTIONS:
        ref = cqt_features(reference, resolution)
        out = cqt_features(edited, resolution)
        frames = min(ref.shape[1], out.shape[1])
        ref, out = ref[:, :frames], out[:, :frames]
        valid = active_mask(ref) & active_mask(out)
        active_counts.append(int(valid.sum()))
        fwd, iqr = directional_alignment(ref, out, resolution, valid)
        rev, _ = directional_alignment(out, ref, resolution, valid)
        forward.append(fwd)
        reverse.append(rev)
        frame_iqr.append(iqr)

    values = np.asarray(forward + reverse + frame_iqr, dtype=np.float64)
    finite = bool(np.isfinite(values).all())
    bidirectional_errors = [abs(a + b) for a, b in zip(forward, reverse)] if finite else [math.inf]
    multiresolution_range = float(max(forward) - min(forward)) if finite else math.inf
    maximum_iqr = float(max(frame_iqr)) if finite else math.inf
    enough_frames = min(active_counts) >= MIN_ACTIVE_FRAMES
    measurable = bool(
        finite
        and enough_frames
        and max(bidirectional_errors) <= MAX_BIDIRECTIONAL_ERROR
        and multiresolution_range <= MAX_MULTIRESOLUTION_RANGE
        and maximum_iqr <= MAX_FRAME_IQR
    )
    return {
        "delta_semitone": float(np.median(forward)) if finite else math.nan,
        "forward_24": forward[0],
        "forward_36": forward[1],
        "forward_48": forward[2],
        "reverse_24": reverse[0],
        "reverse_36": reverse[1],
        "reverse_48": reverse[2],
        "max_bidirectional_error": float(max(bidirectional_errors)),
        "multiresolution_range": multiresolution_range,
        "max_frame_iqr": maximum_iqr,
        "minimum_active_frames": int(min(active_counts)),
        "measurable": measurable,
    }
