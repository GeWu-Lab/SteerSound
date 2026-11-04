"""Independent per-audio editing features and frozen six-task completion metrics."""
from __future__ import annotations
from functools import lru_cache
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .audio import exploration_audio, crop_audio
from ._device import configure_device
from . import DEFAULT_CLAP
from ._rules import classify_row

def _rms(wave):
    return 20.0 * math.log10(math.sqrt(float(np.mean(np.asarray(wave, dtype=np.float64) ** 2))) + 1e-08)

def _paired(audio, reference_audio, sample_rate, region):
    return (crop_audio(exploration_audio(audio, sample_rate=sample_rate), region), crop_audio(exploration_audio(reference_audio, sample_rate=sample_rate), region))

def remove_drop_db(audio, reference_audio, *, edit_region, sample_rate=16000) -> float:
    """A-region input dB minus edited dB; positive means attenuation."""
    (out, ref) = _paired(audio, reference_audio, sample_rate, edit_region)
    return _rms(ref) - _rms(out)

def preserve_drift_db(audio, reference_audio, *, preserve_region, sample_rate=16000) -> float:
    """Absolute B-region loudness change relative to input."""
    (out, ref) = _paired(audio, reference_audio, sample_rate, preserve_region)
    return abs(_rms(out) - _rms(ref))

def volume_change_db(audio, reference_audio, *, edit_region, sample_rate=16000) -> float:
    """Signed A-region output-minus-input dB change."""
    (out, ref) = _paired(audio, reference_audio, sample_rate, edit_region)
    return _rms(out) - _rms(ref)

def _request(value):
    value = float(value)
    if not math.isfinite(value) or abs(value) <= 1e-08:
        raise ValueError('requested_change must be finite and nonzero')
    return value

def volume_progress(audio, reference_audio, *, edit_region, requested_change_db, sample_rate=16000) -> float:
    """Actual signed A-region dB change divided by requested dB change."""
    return volume_change_db(audio, reference_audio, edit_region=edit_region, sample_rate=sample_rate) / _request(requested_change_db)

def _pitch_delta(reference, edited):
    from ._pitch import region_measure
    try:
        result = region_measure(reference, edited)
    except (ValueError, RuntimeError, IndexError):
        return float('nan')
    return float(result['delta_semitone']) if result['measurable'] else float('nan')

def pitch_shift_semitone(audio, reference_audio, *, region, sample_rate=16000) -> float:
    """Signed paired-CQT output-minus-input pitch shift, or NaN if unmeasurable."""
    (out, ref) = _paired(audio, reference_audio, sample_rate, region)
    return _pitch_delta(ref, out)

def pitch_progress(audio, reference_audio, *, edit_region, requested_change_semitone, sample_rate=16000) -> float:
    """Actual signed A-region pitch shift divided by requested semitone shift."""
    return pitch_shift_semitone(audio, reference_audio, region=edit_region, sample_rate=sample_rate) / _request(requested_change_semitone)

@lru_cache(maxsize=8)
def _clap_context(path, device, dtype):
    device = configure_device(device)
    import torch
    from transformers import ClapModel, ClapProcessor
    if dtype not in {'float32', 'float16', 'bfloat16'}:
        raise ValueError('invalid CLAP dtype')
    processor = ClapProcessor.from_pretrained(path, local_files_only=True)
    model = ClapModel.from_pretrained(path, local_files_only=True).eval().requires_grad_(False).to(device, dtype=getattr(torch, dtype))
    return (processor, model, torch.device(device), getattr(torch, dtype), {})

def _clap_scalar(wave, label, region, clap_path, device, dtype):
    if not isinstance(label, str) or not label.strip():
        raise ValueError('label must be nonempty')
    import torch
    import torchaudio
    if region is not None:
        wave = crop_audio(wave, region)
    wave48 = torchaudio.functional.resample(torch.from_numpy(wave.copy()), 16000, 48000).numpy()
    if region is None:
        wave48 = np.pad(wave48, (0, max(0, 480000 - len(wave48))))[:480000]
    else:
        if len(wave48) > 480000:
            start = (len(wave48) - 480000) // 2
            wave48 = wave48[start:start + 480000]
        before = (480000 - len(wave48)) // 2
        wave48 = np.pad(wave48, (before, 480000 - len(wave48) - before))
    (processor, model, dev, dt, text_cache) = _clap_context(str(Path(clap_path).expanduser().resolve()), str(device), dtype)

    def move(batch):
        return {k: v.to(dev, dtype=dt if v.is_floating_point() else None) for (k, v) in batch.items()}
    with torch.inference_mode():
        if label not in text_cache:
            text_batch = processor(text=[label], padding=True, truncation=True, return_tensors='pt')
            text_cache[label] = torch.nn.functional.normalize(model.get_text_features(**move(text_batch)).float(), dim=-1).cpu()
        audio_batch = processor(audios=[np.asarray(wave48, dtype=np.float32)], sampling_rate=48000, return_tensors='pt', padding=True)
        if 'is_longer' not in audio_batch:
            raise ValueError('CLAP processor omitted is_longer')
        audio_batch['is_longer'].fill_(False)
        feature = torch.nn.functional.normalize(model.get_audio_features(**move(audio_batch)).float(), dim=-1).cpu()
        return float((feature * text_cache[label]).sum().item())

def clap_similarity(audio, label, *, region=None, sample_rate=16000, clap_path=DEFAULT_CLAP, device='cpu', dtype='bfloat16') -> float:
    """Frozen exploration CLAP cosine: full right-padding or local centered-padding."""
    return _clap_scalar(exploration_audio(audio, sample_rate=sample_rate), label, region, clap_path, device, dtype)

def _completion(**features):
    result = classify_row(SimpleNamespace(**features))
    if not result['metric_in_denominator']:
        return float('nan')
    return float(result['metric_prediction'])

def generate_completion(audio, target_label, *, sample_rate=16000, clap_path=DEFAULT_CLAP, device='cpu', dtype='bfloat16') -> float:
    """Generate completion: target CLAP strictly >0.08; return 1 or 0."""
    value = clap_similarity(audio, target_label, sample_rate=sample_rate, clap_path=clap_path, device=device, dtype=dtype)
    return _completion(task='generate', clap_target_global=value)

def add_completion(audio, target_label, *, sample_rate=16000, clap_path=DEFAULT_CLAP, device='cpu', dtype='bfloat16') -> float:
    """Add completion: target CLAP strictly >0.04; return 1 or 0."""
    value = clap_similarity(audio, target_label, sample_rate=sample_rate, clap_path=clap_path, device=device, dtype=dtype)
    return _completion(task='add', clap_target_global=value)

def remove_completion(audio, reference_audio, *, edit_region, preserve_region, sample_rate=16000) -> float:
    """Remove: A drop >=6dB and abs(B drift)<3dB; return 1 or 0."""
    (out_a, ref_a) = _paired(audio, reference_audio, sample_rate, edit_region)
    (out_b, ref_b) = _paired(audio, reference_audio, sample_rate, preserve_region)
    return _completion(task='remove', remove_drop_db=_rms(ref_a) - _rms(out_a), output_preserve_db=_rms(out_b), reference_preserve_db=_rms(ref_b), preserve_drift_db=abs(_rms(out_b) - _rms(ref_b)))

def replace_completion(audio, *, source_label, destination_label, base_labels, edit_region, preserve_region, sample_rate=16000, clap_path=DEFAULT_CLAP, device='cpu', dtype='bfloat16') -> float:
    """Replace: A is owned by destination, B by base, using fixed 0.08 presence gates."""
    wave = exploration_audio(audio, sample_rate=sample_rate)
    if isinstance(base_labels, str):
        base_labels = [base_labels]
    base_labels = list(base_labels)
    if not base_labels or not all((isinstance(x, str) and x.strip() for x in base_labels)):
        raise ValueError('base_labels must be nonempty labels')

    def sim(label, region):
        return _clap_scalar(wave, label, region, clap_path, device, dtype)
    return _completion(task='replace', clap_destination_edit=sim(destination_label, edit_region), clap_source_edit=sim(source_label, edit_region), clap_destination_preserve=sim(destination_label, preserve_region), clap_base_preserve_output=min((sim(label, preserve_region) for label in base_labels)))

def volume_completion(audio, reference_audio, *, edit_region, preserve_region, requested_change_db, sample_rate=16000) -> float:
    """Volume: requested A progress in [0.8,1.2], abs(B delta)<1.5dB."""
    requested = _request(requested_change_db)
    (out_a, ref_a) = _paired(audio, reference_audio, sample_rate, edit_region)
    (out_b, ref_b) = _paired(audio, reference_audio, sample_rate, preserve_region)
    return _completion(task='volume', requested_change=requested, edit_delta_db=_rms(out_a) - _rms(ref_a), output_preserve_db=_rms(out_b), reference_preserve_db=_rms(ref_b))

def pitch_completion(audio, reference_audio, *, edit_region, preserve_region, requested_change_semitone, sample_rate=16000) -> float:
    """Pitch: paired A/B must be measurable; A progress [0.8,1.2], abs(B)<1 semitone."""
    requested = _request(requested_change_semitone)
    (out_a, ref_a) = _paired(audio, reference_audio, sample_rate, edit_region)
    (out_b, ref_b) = _paired(audio, reference_audio, sample_rate, preserve_region)
    (a, b) = (_pitch_delta(ref_a, out_a), _pitch_delta(ref_b, out_b))
    return _completion(task='pitch', requested_change=requested, edit_delta_semitone=a, preserve_delta_semitone=b, output_edit_pitch_valid=math.isfinite(a), output_preserve_pitch_valid=math.isfinite(b))
