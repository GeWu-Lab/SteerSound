"""Internal numerical formulas and lazy scorer implementations; use public metric functions."""
from __future__ import annotations
import math
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from ._pitch import region_measure
from .audio import validate_audio_path
SR = 16000
PASST_SR = 32000
TARGET_LEN = SR * 10
PASST_TARGET_LEN = PASST_SR * 10

def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float('nan')

def rms_db(audio: torch.Tensor) -> float:
    if audio is None or audio.numel() == 0:
        return float('nan')
    rms = torch.sqrt(torch.mean(audio.float() ** 2) + 1e-12).item()
    return 20.0 * math.log10(max(rms, 1e-12))

def lsd(audio_a: torch.Tensor, audio_b: torch.Tensor) -> float:
    if audio_a is None or audio_b is None or audio_a.numel() == 0 or (audio_b.numel() == 0):
        return float('nan')
    n = min(audio_a.numel(), audio_b.numel())
    a = audio_a[:n].float()
    b = audio_b[:n].float()
    if n < 512:
        return float('nan')
    win = torch.hann_window(512)
    spec_a = torch.stft(a, n_fft=512, hop_length=160, win_length=512, window=win, return_complex=True)
    spec_b = torch.stft(b, n_fft=512, hop_length=160, win_length=512, window=win, return_complex=True)
    log_a = torch.log10(torch.clamp(torch.abs(spec_a), min=1e-07))
    log_b = torch.log10(torch.clamp(torch.abs(spec_b), min=1e-07))
    return float(torch.sqrt(torch.mean((log_a - log_b) ** 2)).item())

def pitch_mae_semitone(audio_a: torch.Tensor, audio_b: torch.Tensor) -> float:
    """Absolute CQT pitch offset from target ``audio_b`` to generated ``audio_a``.

    The returned value is NaN when the exploration metric's bidirectional,
    multi-resolution consistency gates say that the pair is not measurable.
    Aggregators intentionally exclude such samples from the metric denominator.
    """
    if audio_a is None or audio_b is None or audio_a.numel() == 0 or (audio_b.numel() == 0):
        return float('nan')
    generated = audio_a.detach().cpu().float().reshape(-1).numpy().astype(np.float32, copy=False)
    target = audio_b.detach().cpu().float().reshape(-1).numpy().astype(np.float32, copy=False)
    if not np.isfinite(generated).all() or not np.isfinite(target).all():
        return float('nan')
    try:
        measurement = region_measure(reference=target, edited=generated)
    except Exception:
        return float('nan')
    delta = float(measurement['delta_semitone'])
    if not bool(measurement['measurable']) or not np.isfinite(delta):
        return float('nan')
    return abs(delta)

def _pad_or_trim(audio: torch.Tensor, length: int) -> torch.Tensor:
    if audio.numel() < length:
        return torch.nn.functional.pad(audio, (0, length - audio.numel()))
    return audio[:length]

class KlScorer:

    def __init__(self, device: str='cpu'):
        from ._passt import get_basic_model
        self.device = device
        with open(os.devnull, 'w', encoding='utf-8') as devnull, redirect_stdout(devnull):
            self.model = get_basic_model(mode='logits').eval().to(device)

    @torch.no_grad()
    def logits(self, audio: torch.Tensor) -> torch.Tensor:
        return self.logits_batch([audio])[0]

    @torch.no_grad()
    def logits_batch(self, audios: list[torch.Tensor], batch_size: int=8) -> list[torch.Tensor]:
        if not audios:
            return []
        import torchaudio
        wavs = []
        for audio in audios:
            wav = audio.detach().cpu().float().view(1, -1)
            wav = torchaudio.functional.resample(wav, orig_freq=SR, new_freq=PASST_SR)
            wavs.append(_pad_or_trim(wav.view(-1), PASST_TARGET_LEN))
        outputs = []
        stacked = torch.stack(wavs, dim=0)
        for start in range(0, stacked.shape[0], max(1, batch_size)):
            batch = stacked[start:start + max(1, batch_size)].to(self.device)
            with open(os.devnull, 'w', encoding='utf-8') as devnull, redirect_stdout(devnull):
                logits = self.model(batch)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            outputs.append(logits.detach().cpu().float())
        return [row.view(-1) for row in torch.cat(outputs, dim=0)]

    @staticmethod
    def kl_pair(gen_logits: torch.Tensor, target_logits: torch.Tensor) -> float:
        eps = 1e-06
        gen = gen_logits.view(1, -1)
        target = target_logits.view(1, -1)
        softmax = torch.nn.functional.kl_div((gen.softmax(dim=1) + eps).log(), target.softmax(dim=1), reduction='sum')
        return max(0.0, float(softmax.item()))

class ClapScorer:

    def __init__(self, clap_path: str | Path, device: str='cpu'):
        import librosa
        from transformers import AutoProcessor, ClapModel
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(clap_path), local_files_only=True)
        self.model = ClapModel.from_pretrained(str(clap_path), local_files_only=True).eval().to(device)
        self._text_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def audio_features_batch(self, audios: list[torch.Tensor], batch_size: int=32) -> list[torch.Tensor]:
        if not audios:
            return []
        import torchaudio
        wavs48 = []
        for audio in audios:
            wav = audio.detach().cpu().float().view(1, -1)
            wav48 = torchaudio.functional.resample(wav, orig_freq=SR, new_freq=48000)
            wavs48.append(wav48.view(-1).numpy())
        features = []
        for start in range(0, len(wavs48), max(1, batch_size)):
            batch_wavs = wavs48[start:start + max(1, batch_size)]
            inputs = self.processor(audios=batch_wavs, return_tensors='pt', padding=True, sampling_rate=48000)
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for (k, v) in inputs.items()}
            out = self.model.get_audio_features(**inputs)
            features.extend((row.detach().cpu().float() for row in out))
        return features

    @torch.no_grad()
    def text_features_batch(self, texts: list[str], batch_size: int=64) -> list[torch.Tensor]:
        ordered = [str(text) for text in texts]
        missing = [text for text in dict.fromkeys(ordered) if text and text not in self._text_cache]
        for start in range(0, len(missing), max(1, batch_size)):
            batch_texts = missing[start:start + max(1, batch_size)]
            inputs = self.processor(text=batch_texts, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for (k, v) in inputs.items()}
            out = self.model.get_text_features(**inputs)
            for (text, feature) in zip(batch_texts, out):
                self._text_cache[text] = feature.detach().cpu().float()
        return [self._text_cache[text] for text in ordered]

    def score_label_groups(self, audios: list[torch.Tensor], label_groups: list[dict[str, list[str]]], batch_size: int=32) -> list[dict[str, float]]:
        audio_features = self.audio_features_batch(audios, batch_size=batch_size)
        unique_texts = sorted({str(label) for groups in label_groups for labels in groups.values() for label in labels if str(label)})
        text_features = {text: feature for (text, feature) in zip(unique_texts, self.text_features_batch(unique_texts, batch_size=batch_size))}
        results = []
        for (audio_feature, groups) in zip(audio_features, label_groups):
            row = {}
            for (name, labels) in groups.items():
                valid = [str(label) for label in labels if str(label)]
                if not valid:
                    row[name] = float('nan')
                    continue
                texts_tensor = torch.stack([text_features[label] for label in valid], dim=0)
                audio_tensor = audio_feature.view(1, -1).expand(texts_tensor.shape[0], -1)
                sims = torch.nn.functional.cosine_similarity(texts_tensor, audio_tensor, dim=1)
                row[name] = float(sims.mean().item())
            results.append(row)
        return results

    @torch.no_grad()
    def score(self, audio: torch.Tensor, texts: list[str]) -> list[float]:
        valid = [str(t) for t in texts if str(t)]
        if not valid:
            return []
        return self.score_batch([audio for _ in valid], valid)

    @torch.no_grad()
    def score_batch(self, audios: list[torch.Tensor], texts: list[str], batch_size: int=32) -> list[float]:
        pairs = [(audio, str(text)) for (audio, text) in zip(audios, texts) if str(text)]
        if not pairs:
            return []
        audio_features = self.audio_features_batch([audio for (audio, _) in pairs], batch_size=batch_size)
        text_features = self.text_features_batch([text for (_, text) in pairs], batch_size=batch_size)
        return [float(torch.nn.functional.cosine_similarity(audio_feature, text_feature, dim=0).item()) for (audio_feature, text_feature) in zip(audio_features, text_features)]

    def score_mean(self, audio: torch.Tensor, labels: list[str]) -> float:
        return nanmean(self.score(audio, labels))
