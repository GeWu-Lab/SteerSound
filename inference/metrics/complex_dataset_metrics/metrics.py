"""One function per continuous metric; no indices, DataFrames or job launching."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
from .audio import tensor_audio
from ._device import configure_device
from . import DEFAULT_CLAP, DEFAULT_TORCH_HOME

@lru_cache(maxsize=8)
def _clap(path, device):
    device = configure_device(device)
    from ._models import ClapScorer
    return ClapScorer(path, device=device)

def _torch_cache():
    import torch
    root = DEFAULT_TORCH_HOME
    torch.hub.set_dir(str(root / 'hub'))

@lru_cache(maxsize=8)
def _kl(device):
    device = configure_device(device)
    from ._models import KlScorer
    _torch_cache()
    return KlScorer(device=device)

def rms_db(audio, *, sample_rate=16000, region=None) -> float:
    """Single audio RMS level in dB; 20log10(max(sqrt(mean(x²)+1e-12),1e-12))."""
    from ._models import rms_db as compute
    return compute(tensor_audio(audio, sample_rate=sample_rate, region=region))

def volume_mae_db(audio, reference_audio, *, sample_rate=16000, region=None) -> float:
    """Absolute RMS dB difference against the desired target; lower is better."""
    return abs(rms_db(audio, sample_rate=sample_rate, region=region) - rms_db(reference_audio, sample_rate=sample_rate, region=region))

def lsd(audio, reference_audio, *, sample_rate=16000, region=None) -> float:
    """Log-spectral distance to target, n_fft=512/hop=160; <512 samples -> NaN."""
    from ._models import lsd as compute
    return compute(tensor_audio(audio, sample_rate=sample_rate, region=region), tensor_audio(reference_audio, sample_rate=sample_rate, region=region))

def pitch_mae_semitone(audio, reference_audio, *, sample_rate=16000, region=None) -> float:
    """Absolute paired-CQT pitch offset to target; unmeasurable pairs -> NaN."""
    from ._models import pitch_mae_semitone as compute
    return compute(tensor_audio(audio, sample_rate=sample_rate, region=region), tensor_audio(reference_audio, sample_rate=sample_rate, region=region))

def clap_task_score(audio, *, task, base_labels=(), add_labels=(), remove_labels=(), sample_rate=16000, region=None, clap_path=DEFAULT_CLAP, device='cpu') -> float:
    """Single-audio Task-Aware CLAP; label lists describe this one audio only."""
    if task not in {'generate', 'add', 'remove', 'replace'}:
        raise ValueError('task must be generate/add/remove/replace')

    def labels_for_group(labels):
        if isinstance(labels, str):
            labels = [labels]
        return [str(x) for x in labels if str(x).strip()]
    groups = {'base': labels_for_group(base_labels)}
    if task in {'generate', 'add', 'replace'}:
        groups['add'] = labels_for_group(add_labels)
    if task in {'remove', 'replace'}:
        groups['remove'] = labels_for_group(remove_labels)
    if not any(groups.values()):
        return float('nan')
    wave = tensor_audio(audio, sample_rate=sample_rate, region=region)
    scorer = _clap(str(Path(clap_path).expanduser().resolve()), str(device))
    scores = scorer.score_label_groups([wave], [groups])[0]
    if task == 'remove':
        (base, removed) = (scores['base'], scores['remove'])
        return float(base - removed) if np.isfinite(base) and np.isfinite(removed) else float('nan')
    vals = [value if name != 'remove' else -value for (name, value) in scores.items()]
    vals = [v for v in vals if np.isfinite(v)]
    return float(sum(vals)) if vals else float('nan')

def _kl_pair(audio, reference_audio, sample_rate, region, device):
    a = tensor_audio(audio, sample_rate=sample_rate, region=region)
    b = tensor_audio(reference_audio, sample_rate=sample_rate, region=region)
    model = _kl(str(device))
    (a, b) = model.logits_batch([a, b])
    return model.kl_pair(a, b)

def kl_softmax(audio, reference_audio, *, sample_rate=16000, region=None, device='cpu') -> float:
    """PaSST softmax KL(target || generated), clipped at zero; lower is better."""
    return _kl_pair(audio, reference_audio, sample_rate, region, device)
