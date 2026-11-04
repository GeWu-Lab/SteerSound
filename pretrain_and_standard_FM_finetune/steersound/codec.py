# 包含函数：
# wav_to_mel_batch    : 将 batch 的 wav 转化成 batch 的 mel
# mel_to_wav_batch    : 将 batch 的 mel 转化为 batch 的 wav (走 SpeechT5HifiGan)
# vae_norm            : avae.encode 之前的 mel 归一化
# un_vae_norm         : avae.decode 之后的 mel 反归一化
# dit_norm            : avae.encode 之后到 DiT 之前的 latent 归一化
# un_dit_norm         : DiT 之后到 avae.decode 之前的 latent 反归一化
# vae_encode          : (联合功能) batch[ref/tar audio]  -> dit_norm 后的 latent, 写回 batch
# vae_decode          : (联合功能) latent              -> wav np.ndarray
"""
encode_and_decode_function.py — 全部 wav <-> latent 之间的变换

整体编码流程:
    wav -> wav_to_mel_batch -> vae_norm -> avae.encode -> dit_norm -> DiT
整体解码流程:
    DiT -> un_dit_norm -> avae.decode -> un_vae_norm -> mel_to_wav_batch -> wav

mel <-> wav 的具体配置:
    - TacotronSTFT (filter_length=1024, hop_length=160, win_length=1024,
      n_mel=64, sr=16000) — 与 dataset.py::fn_STFT 一致
    - HiFi-GAN: HuggingFace SpeechT5HifiGan, 在 trainer.build_models 中加载到 ModelBundle.vocoder
      upsample = [5,4,2,2,2] -> 160 (= TacotronSTFT.hop_length), n_mels=64
"""
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# wav <-> mel
# ============================================================

def wav_to_mel_batch(wave_batch: torch.Tensor, fn_STFT, target_length: int) -> torch.Tensor:
    """
    将 batch 的 wav 转化成 batch 的 mel 图.

    输入:
        wave_batch:    Tensor [B, 1, T]  (16kHz waveform)
        fn_STFT:       TacotronSTFT 实例
        target_length: 目标时间帧数 (= duration*100 + 1, 例如 10s -> 1001)

    返回:
        mel: Tensor [B, 1, target_length, n_mel]   (e.g. [B, 1, 1001, 64])
    """
    device = wave_batch.device
    fn_STFT.to(device)

    wave = wave_batch.squeeze(1).float().to(device, non_blocking=True)
    wave = torch.clamp(wave, -1.0, 1.0)

    mel, _, _ = fn_STFT.mel_spectrogram(wave)        # [B, n_mel, T_frames]
    mel = mel.transpose(1, 2)                         # [B, T_frames, n_mel]

    n_frames = mel.shape[1]
    if n_frames < target_length:
        mel = F.pad(mel, (0, 0, 0, target_length - n_frames))
    elif n_frames > target_length:
        mel = mel[:, :target_length, :]

    if mel.size(-1) % 2 != 0:
        mel = mel[..., :-1]

    return mel.unsqueeze(1)                           # [B, 1, target_length, n_mel]


@torch.no_grad()
def mel_to_wav_batch(output_mel_batch, vocoder, sr: int = 16000) -> np.ndarray:
    """
    将 batch 的 mel 转化成 batch 的 wav (HiFi-GAN 神经声码器).

    输入:
        output_mel_batch: Tensor 或 np.ndarray, [B, H, W] 或 [B, 1, H, W]
                          (avae.decode 后再做 un_vae_norm 的 log-magnitude mel)
        vocoder:          SpeechT5HifiGan 实例 (由 trainer.build_models 加载,
                          通过 ModelBundle.vocoder 传入)
        sr:               采样率, 默认 16000

    返回:
        np.ndarray [B, T]   (T = sr * 10 = 160000)

    HiFi-GAN.forward(spectrogram) 期望 [B, T_mel, n_mels=64], 输出 [B, T_mel*160].
    """
    if isinstance(output_mel_batch, np.ndarray):
        mel = torch.from_numpy(output_mel_batch)
    else:
        mel = output_mel_batch
    if mel.dim() == 4:
        mel = mel.squeeze(1)
    assert mel.dim() == 3, f"expect [B, T_mel, n_mel], got {mel.shape}"

    voc_dtype = next(vocoder.parameters()).dtype
    voc_device = next(vocoder.parameters()).device
    mel = mel.to(device=voc_device, dtype=voc_dtype)

    wav = vocoder(mel)                                # [B, T_mel * 160]
    wav = wav.float().cpu().numpy()

    target_len = sr * 10
    out = np.zeros((wav.shape[0], target_len), dtype=np.float32)
    for i, w in enumerate(wav):
        n = min(len(w), target_len)
        out[i, :n] = w[:n]
    return out


# ============================================================
# 标准化 / 反标准化
# ============================================================

def vae_norm(batch: torch.Tensor) -> torch.Tensor:
    dtype = batch.dtype
    batch = batch.to(torch.float32)
    batch = (batch + 4.539258003234863) / 1.1940251588821411
    return batch.to(dtype)


def un_vae_norm(batch: torch.Tensor) -> torch.Tensor:
    dtype = batch.dtype
    batch = batch.to(torch.float32)
    batch = batch * 1.1940251588821411 - 4.539258003234863
    return batch.to(dtype)


def dit_norm(batch: torch.Tensor) -> torch.Tensor:
    dtype = batch.dtype
    batch = batch.to(torch.float32)
    batch = (batch + 0.02288602478802204) / 1.0058698654174805
    return batch.to(dtype)


def un_dit_norm(batch: torch.Tensor) -> torch.Tensor:
    dtype = batch.dtype
    batch = batch.to(torch.float32)
    batch = batch * 1.0058698654174805 - 0.02288602478802204
    return batch.to(dtype)


# ============================================================
# 联合功能函数: vae_encode / vae_decode
# ============================================================

def vae_encode(batch, avae, config, device, dtype, generator=None):
    """
    将 batch 中的 ref/tar audio (wav) 编码为 DiT 输入空间的 latent, 直接修改 batch.

    流程: wav -> wav_to_mel_batch -> vae_norm -> avae.encode().sample() -> dit_norm
    优化:

    输入:
        batch:    含 'ref_audio' / 'tar_audio' (Tensor [B, 1, T]) 的 dict
        avae:     AutoEncoder 实例 (eval, frozen)
        config:   带 .duration 字段
        device, dtype

    返回: 同一个 batch (in-place 写入 dit_norm 后的 latent), 形状 [B, 16, 125, 8]
    """
    from .audio import make_stft
    fn_STFT = make_stft().to(device)

    target_length = int(config.duration * 100 + 1)
    fn_STFT.to(device)

    ref_mel = wav_to_mel_batch(batch["ref_audio"].to(device), fn_STFT, target_length)
    tar_mel = wav_to_mel_batch(batch["tar_audio"].to(device), fn_STFT, target_length)

    bs_ref = ref_mel.shape[0]
    both = torch.cat([ref_mel, tar_mel], dim=0).to(device, dtype)
    # Training intentionally leaves ``generator`` unset and keeps posterior
    # sampling stochastic.  Formal generation passes the per-sample generator
    # so both the AVAE posterior sample and the diffusion noise are repeatable.
    # Previously only the latter was seeded, which made same-seed WAVs differ.
    both_latent = avae.encode(vae_norm(both)).sample(generator=generator)

    batch["ref_audio"] = dit_norm(both_latent[:bs_ref])
    batch["tar_audio"] = dit_norm(both_latent[bs_ref:])
    return batch


@torch.no_grad()
def vae_decode(latent: torch.Tensor, avae, vocoder) -> np.ndarray:
    """
    将 DiT 推理得到的 latent 解码回 wav.

    流程: un_dit_norm -> avae.decode -> un_vae_norm -> mel_to_wav_batch -> wav

    输入:
        latent:  Tensor [B, 16, 125, 8]   (dit_norm 域)
        avae:    AutoEncoder 实例 (eval)
        vocoder: SpeechT5HifiGan 实例 (由 trainer.build_models 加载, ModelBundle.vocoder)

    返回: np.ndarray [B, 160000]
    """
    latent_unnorm = un_dit_norm(latent)
    output_mel = avae.decode(latent_unnorm)            # [B, 1, H, W]
    output_mel = un_vae_norm(output_mel)
    output_mel_np = output_mel[:, 0].cpu().float().numpy()   # [B, H, W]
    return mel_to_wav_batch(output_mel_np, vocoder=vocoder, sr=16000)
