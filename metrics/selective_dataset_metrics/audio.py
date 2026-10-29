"""Package-local single-waveform reading, resampling, cropping and path validation."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable
import numpy as np
SAMPLE_RATE = 16000

def validate_audio_path(value, *, allowed_roots: Iterable[str | Path] | None=None) -> Path:
    if value is None or not str(value).strip() or str(value).strip().lower() in {'none', 'nan'}:
        raise ValueError('audio path must be nonempty')
    path = Path(value).expanduser().resolve()
    if allowed_roots is None and os.environ.get('STEERSOUND_AUDIO_ROOTS'):
        allowed_roots = os.environ['STEERSOUND_AUDIO_ROOTS'].split(os.pathsep)
    if allowed_roots is not None:
        roots = [Path(x).expanduser().resolve() for x in allowed_roots if str(x)]
        if not roots or not any((path.is_relative_to(root) for root in roots)):
            raise ValueError(f'audio path escapes allowed roots: {path}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def read_audio(audio, *, sample_rate: int=SAMPLE_RATE) -> np.ndarray:
    """Read one file or mono numpy/Torch waveform; return mono float32 at 16 kHz.

    File sampling rate is read from its header. sample_rate applies to arrays.
    A two-dimensional array is accepted only if one axis has length one.
    The waveform's duration is preserved, not padded or trimmed here.
    """
    if isinstance(audio, (str, Path)):
        import soundfile as sf
        wave, rate = sf.read(validate_audio_path(audio), dtype='float32', always_2d=False)
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
    else:
        if hasattr(audio, 'detach'):
            audio = audio.detach().cpu().float().numpy()
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim == 2 and 1 in wave.shape:
            wave = wave.reshape(-1)
        rate = sample_rate
    if wave.ndim != 1:
        raise ValueError('audio must be one mono waveform, not a batch')
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError('audio must be nonempty and finite')
    if isinstance(rate, bool) or not isinstance(rate, (int, np.integer)) or rate <= 0:
        raise ValueError('sample_rate must be a positive integer')
    if rate != SAMPLE_RATE:
        import torch
        import torchaudio
        wave = torchaudio.functional.resample(torch.from_numpy(np.asarray(wave, dtype=np.float32)), rate, SAMPLE_RATE).numpy()
    return np.ascontiguousarray(wave, dtype=np.float32)

def crop_audio(wave: np.ndarray, region=None) -> np.ndarray:
    """region is (start_seconds,end_seconds), evaluated after 16k resampling."""
    if region is None:
        return wave
    if len(region) != 2:
        raise ValueError('region must be (start_seconds,end_seconds)')
    begin, end = map(float, region)
    if not np.isfinite([begin, end]).all() or begin < 0 or end <= begin:
        raise ValueError('invalid region')
    a, b = (int(round(begin * SAMPLE_RATE)), int(round(end * SAMPLE_RATE)))
    if not 0 <= a < b <= len(wave):
        raise ValueError('region lies outside the audio duration')
    return wave[a:b]

def tensor_audio(audio, *, sample_rate=SAMPLE_RATE, region=None):
    import torch
    return torch.from_numpy(crop_audio(read_audio(audio, sample_rate=sample_rate), region).copy())

def exploration_audio(audio, *, sample_rate=SAMPLE_RATE):
    wave = read_audio(audio, sample_rate=sample_rate)
    return np.pad(wave, (0, max(0, 160000 - len(wave))))[:160000]
