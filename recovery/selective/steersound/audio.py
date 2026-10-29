"""The training-time 16 kHz, 10 second audio frontend."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import soundfile as sf
import torchaudio
from .frontend import TacotronSTFT
SAMPLE_RATE=16000
SAMPLES=160000
@lru_cache(maxsize=1)
def make_stft():
    return TacotronSTFT(1024,160,1024,64,16000,0,8000)
def load_audio(audio,sample_rate=None):
    if audio is None:
        return torch.zeros(1,1,SAMPLES)
    if isinstance(audio,(str,Path)):
        wave,sr=sf.read(str(audio),dtype='float32',always_2d=True)
        wave=torch.from_numpy(wave).mean(dim=1)
    else:
        if sample_rate is None:raise ValueError('sample_rate is required for array/tensor input')
        sr=sample_rate;wave=torch.as_tensor(audio,dtype=torch.float32)
        if wave.ndim==2:wave=wave.mean(dim=0)
        if wave.ndim!=1:raise ValueError('Expected mono waveform or channels x samples')
    if not torch.isfinite(wave).all():raise ValueError('Audio contains non-finite values')
    if sr!=SAMPLE_RATE:wave=torchaudio.functional.resample(wave,sr,SAMPLE_RATE)
    wave=wave[:SAMPLES]
    if len(wave)<SAMPLES:wave=torch.nn.functional.pad(wave,(0,SAMPLES-len(wave)))
    return wave.clamp(-1,1).reshape(1,1,SAMPLES)
