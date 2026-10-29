"""训练整体模型 - 多任务音频编辑数据集。

任务编码: 0 添加 / 1 去除 / 2 生成 / 3 替换 / 4 音量变化 / 5 音调变化。
所有 ref_audio / tar_audio 严格 (1, 160000), 即 16k * 10s。

结构概览:
- 顶部工具函数:
    fast_pitch_shift, _parse_timing, _shift_and_clamp_segs,
    _mask_to_sounding_only, _pair_replace_piece, _pair_replace_point,
    Random_Crop, Random_Pad, AudioCropPad, prompt 拼装系列等。
- 增强组件:
    Disrupt_base_audio (base 路径专用, 用 fast_pitch_shift)
    Disrupt_audio      (非 base 路径, 正规 pitch_shift; jump=True 时全跳过)
- AudioDataset:
    parquet 多桶抽样, base 多源混合 (1/2/3 = 4.5:4.5:1), 六任务分支。
"""

import os
import ast
import random
import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as audioF
import torchaudio.transforms as audioT
from torch.utils.data import Dataset

from steersound.audio import make_stft


SR = 16000
TARGET_LEN = SR * 10  # 160000
DEFAULT_LABEL_DICT_PATH = None

fn_STFT = make_stft()


# =============================================================================
# DataLoader collate
# =============================================================================

def collate_fn(batch):
    """Collate training samples.

    Tensor fields are stacked and plain T5 prompt strings are kept as lists.
    """
    out = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if key in {"input_features", "feature_attention_mask"}:
            values = [v.detach() for v in values]
            out[key] = torch.cat(values, dim=0)
        elif key in {"input_ids", "attention_mask"}:
            values = [v.detach() for v in values]
            stacked = torch.stack(values)
            out[key] = stacked.squeeze(1) if stacked.ndim >= 3 and stacked.shape[1] == 1 else stacked
        elif all(isinstance(v, torch.Tensor) for v in values):
            values = [v.detach() for v in values]
            try:
                out[key] = torch.stack(values)
            except RuntimeError:
                out[key] = values
        else:
            out[key] = values
    return out


# =============================================================================
# 1. 通用工具
# =============================================================================

def remove_period(text: str) -> str:
    if isinstance(text, str) and text.endswith("."):
        return text[:-1]
    return text


def fast_pitch_shift(audio: torch.Tensor, n_steps: float, sample_rate: int) -> torch.Tensor:
    """通过 time-shift 模拟变调; 会改变长度, 调用方需要后接 crop+pad 归一长度。"""
    ratio = 2.0 ** (n_steps / 12.0)
    target_len = max(1, int(audio.shape[-1] / ratio))
    shifted = F.interpolate(
        audio.unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)
    return shifted


def _crop_or_pad_to(wave: torch.Tensor, target_length: int) -> torch.Tensor:
    """防御性长度归一: 仅在长度不等时左裁右补/右截。"""
    cur = wave.shape[-1]
    if cur == target_length:
        return wave
    if cur > target_length:
        return wave[..., :target_length]
    return F.pad(wave, (0, target_length - cur))


def _quantize_pitch_step(n_step: float, pitch_range: float, step: Optional[float]) -> float:
    """按固定 pitch step 量化; step=None/<=0 时不量化。"""
    if step is None or step <= 0:
        return float(n_step)
    n_step = round(float(n_step) / step) * step
    n_step = max(-pitch_range, min(pitch_range, n_step))
    if abs(n_step) < 1e-12:
        n_step = 0.0
    return float(n_step)


def _sample_quantized_pitch_step(pitch_range: float, step: Optional[float]) -> float:
    """等概率采样量化 pitch 档位。

    0.5 量化时 raw 采样范围扩展到 [-range-0.25, range+0.25],
    这样端点档位和中间档位都有同样的 0.5 宽度。
    """
    if step is None or step <= 0:
        return random.uniform(-pitch_range, pitch_range)
    half_step = step / 2.0
    raw = random.uniform(-pitch_range - half_step, pitch_range + half_step)
    return _quantize_pitch_step(raw, pitch_range, step)


def _pitch_step_values(pitch_range: float, step: Optional[float]) -> List[float]:
    if step is None or step <= 0:
        return []
    n = int(round((2 * pitch_range) / step))
    vals = []
    for i in range(n + 1):
        v = -pitch_range + i * step
        v = max(-pitch_range, min(pitch_range, v))
        if abs(v) < 1e-12:
            v = 0.0
        vals.append(round(float(v), 6))
    return sorted(set(vals))


# ---- timing 解析 / 平移 ----

def _parse_timing(x):
    """None / NaN / str / list -> list[(start, dur)] 或 None。"""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "none":
            return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return None
    if isinstance(x, list):
        return x if len(x) > 0 else None
    if isinstance(x, np.ndarray):
        return x.tolist() if len(x) > 0 else None
    return None


def _shift_and_clamp_segs(segs, offset: int, T: int):
    """把 timing 段从原 chunk 坐标系平移到对齐后坐标系, clamp 到 [0, T], 丢弃完全越界。"""
    if not segs:
        return []
    out = []
    for (s, d) in segs:
        s_new = int(s) + int(offset)
        e_new = s_new + int(d)
        s_clip = max(0, s_new)
        e_clip = min(T, e_new)
        d_clip = e_clip - s_clip
        if d_clip <= 0:
            continue
        out.append((s_clip, d_clip))
    return out


def _mask_to_sounding_only(audio: torch.Tensor,
                           points: list,
                           pieces: list,
                           point_keep_lens: Optional[list] = None,
                           piece_keep_lens: Optional[list] = None) -> torch.Tensor:
    """只保留 audio 中 (points ∪ pieces) 覆盖的样本, 其它置零。"""
    masked = torch.zeros_like(audio)
    T = audio.shape[1]
    for i, (s, d) in enumerate(points or []):
        keep = int(d) if point_keep_lens is None else int(point_keep_lens[i])
        s = max(0, int(s)); e = min(s + keep, T)
        if e > s:
            masked[:, s:e] = audio[:, s:e]
    for i, (s, d) in enumerate(pieces or []):
        keep = int(d) if piece_keep_lens is None else int(piece_keep_lens[i])
        s = max(0, int(s)); e = min(s + keep, T)
        if e > s:
            masked[:, s:e] = audio[:, s:e]
    return masked


def _pair_replace_piece(tar: torch.Tensor, b_segs: list, c_segs: list, c_audio: torch.Tensor):
    """piece 模式替换: 写入长度 = min(b_dur, c_dur), 整段先清零。"""
    T_b = tar.shape[1]
    T_c = c_audio.shape[1]
    nb = len(b_segs); nc = len(c_segs)
    infos = []
    for i in range(nb):
        bs, bd = b_segs[i]
        bs = max(0, int(bs))
        be = min(bs + int(bd), T_b)
        if be <= bs:
            continue
        tar[:, bs:be] = 0
        if i < nc:
            cs, cd = c_segs[i]
            cs = max(0, int(cs))
            cand = min(int(bd), int(cd))
            write_len = max(0, min(cand, T_b - bs, T_c - cs))
            if write_len > 0:
                tar[:, bs:bs + write_len] = c_audio[:, cs:cs + write_len]
        else:
            write_len = 0
        b_dur_eff = be - bs
        infos.append({
            "b_start_sec":      bs / SR,
            "b_end_sec":        be / SR,
            "c_write_len_sec":  write_len / SR,
            "b_zeroed_len_sec": max(0, b_dur_eff - write_len) / SR,
        })
    return tar, infos


def _pair_replace_point(tar: torch.Tensor, b_segs: list, c_segs: list, c_audio: torch.Tensor):
    """point 模式替换: 先清空整个 b 段, 再贴 c 的整段 (仅超 10s 才截)。"""
    T_b = tar.shape[1]
    T_c = c_audio.shape[1]
    nb = len(b_segs); nc = len(c_segs)
    infos = []
    for i in range(nb):
        bs, bd = b_segs[i]
        bs = max(0, int(bs))
        be = min(bs + int(bd), T_b)
        if be <= bs:
            continue
        tar[:, bs:be] = 0
        if i < nc:
            cs, cd = c_segs[i]
            cs = max(0, int(cs))
            write_len = max(0, min(int(cd), T_b - bs, T_c - cs))
            if write_len > 0:
                tar[:, bs:bs + write_len] = c_audio[:, cs:cs + write_len]
            zeroed = max(0, (be - bs) - write_len)
        else:
            write_len = 0
            zeroed = be - bs
        infos.append({
            "b_start_sec":      bs / SR,
            "b_end_sec":        be / SR,
            "c_write_len_sec":  write_len / SR,
            "b_zeroed_len_sec": zeroed / SR,
        })
    return tar, infos


# =============================================================================
# 替换任务专用函数: 按 specific_time 走三条独立逻辑
#   - _replace_no_combined: No 模式, 仅留 b 发声段 + b 段位置写 c 内容(piece+point 综合)
#   - _replace_point_target: point 模式, 仅替换 b 上"指定 target point"位置, 其它位置不动
#   - _replace_piece_target: piece 模式, 仅替换 b 上"指定 target piece"位置, 其它位置不动;
#                            含 swap (确保 c 段 ≥ b 段, 与第三阶段一致).
# =============================================================================

def _replace_no_combined(
    wave_b: torch.Tensor,
    b_points: list, b_pieces: list,
    wave_c: torch.Tensor,
    c_points: list, c_pieces: list,
) -> torch.Tensor:
    """No 模式综合替换: 对齐替换损失实验 dataset.ipynb 的构造.

    No 模式没有指定时间, 因此先只保留 b 中确实会被 c 替换的长度:
      - point: 第 i 个 b_point 只有在 c_points[i] 存在时才保留完整 b_point;
      - piece: 按时长降序配对, 第 i 个 b_piece 只保留 min(b_dur, c_dur);
      - 配不到的 b 段保留长度为 0, 避免 ref 留下未被 tar 对应替换的 b 原声.

    之后 tar 从 ref 克隆出来, 先执行 piece 替换, 再执行 point 替换。
    这里不做 point/piece 重叠过滤, 保持替换损失实验 ipynb 的顺序语义。

    返回: ref, tar (两者均 [1, T])
    """
    point_keep_lens = [
        int(d) if i < len(c_points) else 0
        for i, (_, d) in enumerate(b_points)
    ]

    if b_pieces:
        b_pieces_sorted = sorted(b_pieces, key=lambda x: -x[1])
        c_pieces_sorted = sorted(c_pieces, key=lambda x: -x[1])
        piece_keep_lens = [
            min(int(bd), int(c_pieces_sorted[i][1])) if i < len(c_pieces_sorted) else 0
            for i, (_, bd) in enumerate(b_pieces_sorted)
        ]
    else:
        b_pieces_sorted = []
        c_pieces_sorted = sorted(c_pieces, key=lambda x: -x[1]) if c_pieces else []
        piece_keep_lens = []

    ref = _mask_to_sounding_only(
        wave_b, b_points, b_pieces_sorted,
        point_keep_lens=point_keep_lens,
        piece_keep_lens=piece_keep_lens,
    )
    tar = ref.clone()

    if b_pieces_sorted:
        tar, _ = _pair_replace_piece(tar, b_pieces_sorted, c_pieces_sorted, wave_c)

    tar, _ = _pair_replace_point(tar, b_points, c_points, wave_c)
    return ref, tar


def _replace_point_target(
    wave_b: torch.Tensor,
    target_point_b,                 # tensor([s, e]) 或 (s, e) — b 上要被替换的位置
    wave_c: torch.Tensor,
    source_point_c,                 # tensor([s, e]) 或 (s, e) — c 上提供内容的位置
) -> torch.Tensor:
    """point 模式: 仅替换 b 上 target_point_b 位置的内容, b 其余位置不动.

    与第三阶段 split_audio + trans_audio_start_time 等价:
      - 把 b 上 target_point_b 区间清零;
      - 把 c[source_point_c.s : source_point_c.s + min(c_dur, T - target_b.s)] 贴到 b 的 target_b.s 位置.
        即 c 段过长时按"target_b 起点到 T 的距离"截掉.

    返回: 处理后的 wave_b' (1, T) — b 上仅 target_point 区间被换, 其余原样保留.
    """
    T = wave_b.shape[1]
    bs = int(target_point_b[0]); be = int(target_point_b[1])
    cs = int(source_point_c[0]); ce = int(source_point_c[1])
    cd = ce - cs

    out = wave_b.clone()
    # 清零 b target 区
    bs_clip = max(0, bs); be_clip = min(be, T)
    if be_clip > bs_clip:
        out[:, bs_clip:be_clip] = 0
    # c 段往 b 上贴; 超出边界就截到 T - bs
    if bs < T and cd > 0:
        write_len = max(0, min(cd, T - bs, wave_c.shape[1] - cs))
        if write_len > 0:
            out[:, bs:bs + write_len] = wave_c[:, cs:cs + write_len]
    return out


def _replace_piece_target(
    wave_b: torch.Tensor,
    target_piece_b,                 # tensor([s, e]) 或 (s, e) — b 上要被替换的位置
    wave_c: torch.Tensor,
    source_piece_c,                 # tensor([s, e]) 或 (s, e) — c 上提供内容的位置
) -> torch.Tensor:
    """piece 模式: 仅替换 b 上 target_piece_b 位置的内容, b 其余位置不动.

    前提: 调用方已保证 c 段时长 >= b 段时长 (第三阶段做法: 若不满足则 swap b/c).
    实际写入长度 = b 段时长 = target_piece_b[1] - target_piece_b[0].

    返回: 处理后的 wave_b' (1, T) — b 上 target_piece 区间被 c 对应起点开始的同长度内容替换.
    """
    T = wave_b.shape[1]
    bs = int(target_piece_b[0]); be = int(target_piece_b[1])
    cs = int(source_piece_c[0])
    bd = be - bs

    out = wave_b.clone()
    bs_clip = max(0, bs); be_clip = min(be, T)
    if be_clip > bs_clip:
        out[:, bs_clip:be_clip] = 0
    write_len = max(0, min(bd, T - bs, wave_c.shape[1] - cs))
    if write_len > 0:
        out[:, bs:bs + write_len] = wave_c[:, cs:cs + write_len]
    return out


# ---- chunk 切片读取 ----

def load_chunk_aligned_10s(audio_base_path: str,
                           rel_path: str,
                           start_sec: float,
                           end_sec: float) -> Optional[Tuple[torch.Tensor, int]]:
    """读 chunk -> mono 16k -> 随机 pad/crop 到 (1, 160000)。返回 (wave, offset)。

    offset 用来把原 chunk 坐标系的 timing 平移到对齐后坐标系:
        s_new = s_old + offset
        随机 pad   -> offset = +pad_left
        随机 crop  -> offset = -crop_start
        等长       -> offset = 0
    加载失败返回 None, 上层应跳过。
    """
    abs_path = os.path.join(audio_base_path, rel_path)
    try:
        info = torchaudio.info(abs_path)
        sr = info.sample_rate
        frame_offset = int(start_sec * sr)
        num_frames = int((end_sec - start_sec) * sr)
        wave, src_sr = torchaudio.load(abs_path, frame_offset=frame_offset, num_frames=num_frames)
    except Exception:
        try:
            wave, src_sr = torchaudio.load(abs_path)
        except Exception:
            return None
    wave = wave[:1, :]
    if src_sr != SR:
        wave = audioF.resample(wave, src_sr, SR)
    raw_len = wave.shape[1]
    if raw_len == 0:
        return None
    if raw_len >= TARGET_LEN:
        crop_start = random.randint(0, raw_len - TARGET_LEN)
        wave = wave[:, crop_start:crop_start + TARGET_LEN]
        offset = -crop_start
    else:
        pad_left = random.randint(0, TARGET_LEN - raw_len)
        pad_right = TARGET_LEN - raw_len - pad_left
        wave = F.pad(wave, (pad_left, pad_right))
        offset = pad_left
    return wave, offset


def load_chunk_full(audio_base_path: str,
                    rel_path: str,
                    start_sec: float,
                    end_sec: float) -> torch.Tensor:
    """读 chunk -> mono 16k, 不做 crop/pad. 返回 wave [1, T_raw].

    IO 由 dataset.__init__ 阶段的 verify_io 预检保证成功;
    此函数假设文件可读, 若失败直接抛异常.
    """
    abs_path = os.path.join(audio_base_path, rel_path)
    info = torchaudio.info(abs_path)
    sr = info.sample_rate
    frame_offset = int(start_sec * sr)
    num_frames = int((end_sec - start_sec) * sr)
    wave, src_sr = torchaudio.load(abs_path, frame_offset=frame_offset, num_frames=num_frames)
    wave = wave[:1, :]
    if src_sr != SR:
        wave = audioF.resample(wave, src_sr, SR)
    if wave.shape[1] == 0:
        raise RuntimeError(
            f"load_chunk_full got empty wave: {abs_path} [{start_sec},{end_sec}]"
        )
    return wave


def load_full_audio(audio_base_path: str, rel_path: str) -> torch.Tensor:
    """Read a full audio file as mono 16 kHz without chunk slicing or crop/pad."""
    abs_path = os.path.join(audio_base_path, rel_path)
    wave, src_sr = torchaudio.load(abs_path)
    wave = wave[:1, :]
    if src_sr != SR:
        wave = audioF.resample(wave, src_sr, SR)
    if wave.shape[1] == 0:
        raise RuntimeError(f"load_full_audio got empty wave: {abs_path}")
    return wave


def _unwrap_lp_timing(x) -> List[Tuple[int, int]]:
    """Parse long-prompt timing and unwrap one legacy outer list when present."""
    parsed = _parse_timing(x)
    if not parsed:
        return []
    if isinstance(parsed, list) and isinstance(parsed[0], list):
        return list(parsed[0])
    return list(parsed)


def _clamp_segs_to_len(segs, T: int) -> List[Tuple[int, int]]:
    """把 [(start, duration), ...] clamp 到 [0, T] (不 shift):

    - start<0 时把 start 拉到 0, duration 相应缩短.
    - start+duration>T 时把 duration 截短到 T-start.
    - 截短后 duration<=0 或 start>=T 的段丢弃.
    """
    if not segs:
        return []
    out: List[Tuple[int, int]] = []
    for (s, d) in segs:
        s = int(s); d = int(d)
        if d <= 0:
            continue
        e = s + d
        if e <= 0 or s >= T:
            continue
        s_c = max(0, s)
        e_c = min(T, e)
        d_c = e_c - s_c
        if d_c <= 0:
            continue
        out.append((s_c, d_c))
    return out


def _shift_and_drop_oob(segs_t: torch.Tensor, bias: int, T: int) -> torch.Tensor:
    """把 segs (Tensor[N,2] s/e) 平移 bias, 然后 clamp 到 [0, T]:

    - 部分越界: 端点 clamp 到边界, 段被截短保留.
    - 完全越界 (clamp 后 e<=s): 丢弃.

    仅用于 replace-No 场景, 处理保护裁切后 target 之外的 point/piece 段.
    """
    if segs_t.shape[0] == 0:
        return segs_t
    s = segs_t[:, 0] + int(bias)
    e = segs_t[:, 1] + int(bias)
    s_c = torch.clamp(s, 0, T)
    e_c = torch.clamp(e, 0, T)
    keep = e_c > s_c
    if not keep.any():
        return torch.zeros((0, 2), dtype=segs_t.dtype)
    return torch.stack([s_c[keep], e_c[keep]], dim=1)


# =============================================================================
# 2. 长度归一组件 (旧版同款)
# =============================================================================

class Random_Crop:
    """长度 > target 时随机裁剪到 target; 否则原样返回 (no-op)。"""
    def __call__(self, waveform: torch.Tensor, target_point_length: int) -> torch.Tensor:
        _, length = waveform.shape
        if length <= target_point_length:
            return waveform
        start = random.randint(0, length - target_point_length)
        return waveform[:, start:start + target_point_length]


class Random_Pad:
    """长度 < target 时随机左右补零到 target; 否则原样返回。"""
    def __call__(self, waveform: torch.Tensor, target_point_length: int) -> torch.Tensor:
        _, length = waveform.shape
        diff = target_point_length - length
        if diff < 0:
            return waveform
        if diff == 0:
            return waveform
        pad_left = random.randint(0, diff)
        pad_right = diff - pad_left
        return F.pad(waveform, (pad_left, pad_right), mode="constant", value=0.0)


class AudioCropPad:
    """裁/补到固定 target_length, 同时返回 bias 用于平移 piece 坐标。

    bias 语义:
        padding -> +pad_left
        cropping -> -start
    """
    def __init__(self, target_length: int):
        self.target_length = target_length

    def __call__(self, waveform: torch.Tensor, piece: Optional[torch.Tensor] = None):
        _, length = waveform.shape
        if length < self.target_length:
            return self._pad(waveform)
        if length > self.target_length:
            return self._crop(waveform, piece)
        return waveform, self.target_length / SR, "padding", 0

    def _pad(self, waveform: torch.Tensor):
        padding = self.target_length - waveform.size(1)
        pad_left = random.randint(0, padding)
        pad_right = padding - pad_left
        return (
            F.pad(waveform, (pad_left, pad_right)),
            self.target_length / SR,
            "padding",
            pad_left,
        )

    def _crop(self, waveform: torch.Tensor, piece):
        if piece is None:
            length = waveform.size(1)
            start = torch.randint(0, length - self.target_length, (1,)).item()
        else:
            total_len = waveform.shape[1]
            start_point = int(piece[0])
            end_point = int(piece[1])
            piece_len = end_point - start_point
            if piece_len >= self.target_length:
                start = min(start_point, total_len - self.target_length)
            else:
                low = max(0, end_point - self.target_length)
                high = min(start_point, total_len - self.target_length)
                if low > high:
                    low, high = max(0, total_len - self.target_length), max(0, total_len - self.target_length)
                start = random.randint(low, high)
        return (
            waveform[:, start:start + self.target_length],
            self.target_length / SR,
            "cropping",
            -start,
        )


# =============================================================================
# 3. 读取期增强
# =============================================================================

class Disrupt_base_audio:
    """base 路径专用: 用 fast_pitch_shift (会改长度) + dB 缩放, 不动 timing。

    fast_pitch 会把 base_audio 整体压/拉, 但因为 base 不参与发声段定位,
    长度变化由后续 random_crop + random_pad 归一回 TARGET_LEN。
    """

    def __init__(
        self,
        sr: int = SR,
        pitch_range: float = 3.0,
        volume_range: float = 3.0,
        pitch_change_prob: float = 0.5,
        volume_change_prob: float = 0.5,
    ):
        self.sr = sr
        self.pitch_range = pitch_range
        self.volume_range = volume_range
        self.pitch_change_prob = pitch_change_prob
        self.volume_change_prob = volume_change_prob

    def __call__(self, audio_wave: torch.Tensor) -> torch.Tensor:
        assert audio_wave.ndim == 2 and audio_wave.shape[0] == 1
        if random.random() < self.pitch_change_prob:
            n_step = random.uniform(-self.pitch_range, self.pitch_range)
            audio_wave = fast_pitch_shift(audio_wave, n_step, self.sr)
        if random.random() < self.volume_change_prob:
            db = random.uniform(-self.volume_range, self.volume_range)
            audio_wave = audio_wave * (10 ** (db / 20))
        return audio_wave


class Disrupt_audio:
    """非 base 路径专用: 正规 pitch_shift + dB 缩放。

    与之前版本的差异: **删除 time_fake 路径**(随机搬移发声段位置), timing 始终保持原 chunk
    位置, 只做格式转换 (start, duration) -> (start, end)。pitch / volume 增强仍保留。

    jump=True 时不做任何增强, 仅做 timing 格式转换。

    fast 版差异:
      - 读取期 pitch 增强把连续 n_steps 量化到固定档位;
      - 复用 torchaudio.transforms.PitchShift 实例, 避免每条样本重复初始化重型 kernel.
    """

    def __init__(
        self,
        pitch_fake_prob: float = 0.5,
        volume_fake_prob: float = 0.5,
        pitch_range: float = 3.0,
        volume_range: float = 3.0,
        sr: int = SR,
        pitch_step_quantization: Optional[float] = 0.5,
        pitch_cache_size: int = 32,
        pitch_shift_threads: Optional[int] = 16,
        pitch_shift_max_seconds: float = 12.0,
    ):
        self.pitch_fake_prob = pitch_fake_prob
        self.volume_fake_prob = volume_fake_prob
        self.pitch_range = pitch_range
        self.volume_range = volume_range
        self.sr = sr
        # 音频超过该秒数时跳过 pitch_shift, 防 OOM. 通过 config 暴露以支持不同硬件调节.
        self.longset_pitch_shift_point_duration = int(self.sr * float(pitch_shift_max_seconds))
        self.pitch_step_quantization = pitch_step_quantization
        self.pitch_cache_size = int(pitch_cache_size)
        self.pitch_shift_threads = pitch_shift_threads
        self._pitch_shift_cache: Dict[float, audioT.PitchShift] = {}
        self._pitch_shift_cache_order: List[float] = []

    def _pitch_step_values(self) -> List[float]:
        return _pitch_step_values(self.pitch_range, self.pitch_step_quantization)

    def _sample_pitch_step(self) -> float:
        return _sample_quantized_pitch_step(self.pitch_range, self.pitch_step_quantization)

    def _get_pitch_shifter(self, n_step: float) -> audioT.PitchShift:
        key = round(float(n_step), 6)
        cached = self._pitch_shift_cache.get(key)
        if cached is not None:
            return cached
        shifter = audioT.PitchShift(sample_rate=self.sr, n_steps=key)
        self._pitch_shift_cache[key] = shifter
        self._pitch_shift_cache_order.append(key)
        if self.pitch_cache_size > 0:
            while len(self._pitch_shift_cache_order) > self.pitch_cache_size:
                old_key = self._pitch_shift_cache_order.pop(0)
                self._pitch_shift_cache.pop(old_key, None)
        return shifter

    def _apply_pitch_shift(self, audio_wave: torch.Tensor, n_step: float) -> torch.Tensor:
        shifter = self._get_pitch_shifter(n_step)
        num_threads = self.pitch_shift_threads
        if num_threads is None or num_threads <= 0:
            return shifter(audio_wave)

        prev_threads = torch.get_num_threads()
        target_threads = int(num_threads)
        if prev_threads == target_threads:
            return shifter(audio_wave)

        torch.set_num_threads(target_threads)
        try:
            return shifter(audio_wave)
        finally:
            torch.set_num_threads(prev_threads)

    def warmup_pitch_cache(
        self,
        length: int = TARGET_LEN,
        num_threads: Optional[int] = None,
    ) -> None:
        """预热量化后的 PitchShift cache, 避免训练/预览时首个样本触发长尾."""
        values = self._pitch_step_values()
        if not values:
            return
        prev_threads = None
        if num_threads is not None and num_threads > 0:
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(int(num_threads))
        warm = torch.zeros(1, int(length))
        try:
            with torch.no_grad():
                for n_step in values:
                    if abs(n_step) > 1e-12:
                        _ = self._get_pitch_shifter(n_step)(warm)
        finally:
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)

    def __call__(
        self,
        audio_wave: torch.Tensor,
        point: list,
        piece: list,
        jump: bool = False,
    ):
        """
        point/piece: list[(start, duration)] 或空列表 (来自 _shift_and_clamp_segs)。
        返回:
            audio_wave: (1, T)
            point:      Tensor[N_pt, 2]  (start, end), 不存在时为 shape [0, 2] 空 Tensor
            piece:      Tensor[N_pc, 2]  (start, end)
        无论 jump 是否为 True, timing 都做 (start, duration) -> (start, end) 转换.
        """
        assert audio_wave.ndim == 2 and audio_wave.shape[0] == 1

        n_pt = len(point or [])
        n_pc = len(piece or [])
        sound_piece = torch.tensor(list(point or []) + list(piece or []), dtype=torch.long) \
            if (n_pt + n_pc) > 0 else torch.zeros((0, 2), dtype=torch.long)

        # 仅做 (start, duration) -> (start, end) 格式转换; 不再随机搬移
        if sound_piece.shape[0] == 0:
            point_t = torch.zeros((0, 2), dtype=torch.long)
            piece_t = torch.zeros((0, 2), dtype=torch.long)
        else:
            new_sp = torch.stack([sound_piece[:, 0], sound_piece[:, 0] + sound_piece[:, 1]], dim=1)
            point_t = new_sp[:n_pt]
            piece_t = new_sp[n_pt:]

        if jump:
            return audio_wave, point_t, piece_t

        # ---- pitch_shift (正规版, 保长度但 11s+ 跳过防 OOM) ----
        if (
            random.random() < self.pitch_fake_prob
            and audio_wave.shape[1] < self.longset_pitch_shift_point_duration
        ):
            n_step = self._sample_pitch_step()
            if abs(n_step) > 1e-12:
                audio_wave = self._apply_pitch_shift(audio_wave, n_step)

        # ---- volume ----
        if random.random() < self.volume_fake_prob:
            db = random.uniform(-self.volume_range, self.volume_range)
            audio_wave = audio_wave * (10 ** (db / 20))

        return audio_wave, point_t, piece_t


# =============================================================================
# 4. Prompt 拼装
# =============================================================================

def get_describe(feature: list) -> str:
    """feature: [text|False, has_audio_bool]"""
    assert isinstance(feature, list) and len(feature) == 2
    has_text = feature[0] is not False
    has_audio = feature[1] is not False
    if has_text and has_audio:
        return f"of {feature[0]} like <|audio_bos|><|AUDIO|><|audio_eos|>"
    if has_text and not has_audio:
        return f"of {feature[0]}"
    if not has_text and has_audio:
        return "like <|audio_bos|><|AUDIO|><|audio_eos|>"
    raise KeyError("get_describe: feature 不能同时为 False")


def get_time_describe(specific_time: str, target_point_piece) -> str:
    if specific_time == "No":
        return ""
    if specific_time == "point":
        return f" at the {float(target_point_piece[0]) / SR:.1f}th second"
    if specific_time == "piece":
        return (
            f" from the {float(target_point_piece[0]) / SR:.1f}th"
            f" to the {float(target_point_piece[1]) / SR:.1f}th second"
        )
    raise KeyError(f"输入了错误的specific_time:{specific_time}")


def add_special_token(task: int, task_text, specific_time: str, target_point_piece) -> List[str]:
    """task: 0 添加 / 1 去除 / 2 生成 / 3 替换 / 4 音量 / 5 音调"""
    prompt = ""  # The model text encoder adds its own control tokens.
    time_str = get_time_describe(specific_time, target_point_piece)
    if task == 0:
        prompt += " Add the sound " + get_describe(task_text) + time_str + "."
    elif task == 1:
        prompt += " Remove the sound " + get_describe(task_text) + time_str + "."
    elif task == 2:
        prompt += " Generate the sound " + get_describe(task_text) + time_str + "."
    elif task == 3:
        prompt += (
            " Replace the sound " + get_describe(task_text[0]) + time_str
            + " by the sound " + get_describe(task_text[1]) + "."
        )
    elif task == 4:
        verb = "Increase" if task_text[1] >= 0 else "Reduce"
        prompt += (
            f" {verb} the volume of the sound " + get_describe(task_text[0]) + time_str
            + " by {:.1f}".format(abs(float(task_text[1]))) + "."
        )
    elif task == 5:
        verb = "Increase" if task_text[1] >= 0 else "Reduce"
        prompt += (
            f" {verb} the pitch of the sound " + get_describe(task_text[0]) + time_str
            + " by {:.1f}".format(abs(float(task_text[1]))) + "."
        )
    else:
        raise KeyError(f"未知 task: {task}")
    return [prompt.strip()]


# =============================================================================
# 5. AudioDataset
# =============================================================================

class _Transform:
    """把所有变换组件挂到一起, 不做额外逻辑。"""
    def __init__(self, random_crop, random_pad, disrupt_base, crop_pad, disrupt):
        self.random_crop = random_crop
        self.random_pad = random_pad
        self.disrupt_base = disrupt_base
        self.crop_pad = crop_pad
        self.disrupt = disrupt


class AudioDataset(Dataset):
    """多任务音频编辑数据集。

    顶层接口同旧版 (__getitem__ 接受可选的 task / specific_time / use_base 等覆盖参数)。
    主数据源使用 train_with_timing.parquet；add/generate 可按配置从 long-prompt CSV 抽目标音频。
    """

    def __init__(
        self,
        parquet_path: str,
        audio_base_path: str,
        duration: float = 10.0,
        task_specific: Optional[torch.Tensor] = None,
        task_probs: Tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        specific_time: Optional[str] = None,
        # ---- 抽样比例 ----
        base_audio_num_probs: Tuple[float, float, float] = (4.5, 4.5, 1.0), # 1条/2条/3条 作为base的相对概率
        specific_time_probs: Tuple[float, float, float] = (0.2, 0.4, 0.4),  # No / point / piece
        condition_type_probs: Tuple[float, float, float] = (0.7, 0.15, 0.15),  # t / a / ta
        use_base_prob: float = 0.9,
        # ---- 增强超参 ----
        pitch_fake_prob: float = 0.5,
        volume_fake_prob: float = 0.5,
        read_pitch_range: float = 3.0,
        read_pitch_step_quantization: Optional[float] = 0.5,
        pitch_shift_cache_size: int = 32,
        pitch_shift_prewarm: bool = False,
        pitch_shift_warmup_threads: Optional[int] = None,
        read_pitch_shift_threads: Optional[int] = 16,
        read_pitch_shift_max_seconds: float = 12.0,
        read_volume_range: float = 3.0,
        edit_volume_range: float = 5.0,
        edit_pitch_range: float = 4.0,
        edit_pitch_step_quantization: Optional[float] = 0.5,
        edit_pitch_shift_cache_size: int = 16,
        edit_pitch_shift_threads: Optional[int] = None,
        label_dict_path: Optional[str] = DEFAULT_LABEL_DICT_PATH,
        # ---- processor / prompt ----
        processor_max_length: int = 425,
        # ---- 行为开关 ----
        verbose_errors: bool = False,
        verify_io: bool = True,
        # ---- long-prompt 添加/生成数据源 ----
        long_prompt_csv_path: Optional[str] = None,
        long_prompt_audio_base_path: str = ".",
        long_prompt_prob: float = 0.5,
        long_prompt_tasks: Tuple[int, ...] = (0, 2),
        required_long_prompt_tasks: Tuple[int, ...] = (),
        audit_provenance: bool = False,
    ):
        self.task_probs = [float(weight) for weight in task_probs]
        if len(self.task_probs) != 6:
            raise ValueError(f"task_probs 必须包含 6 个权重, 收到 {len(self.task_probs)} 个")
        if any(weight < 0.0 for weight in self.task_probs):
            raise ValueError(f"task_probs 不能包含负数, 收到 {self.task_probs}")
        if sum(self.task_probs) <= 0.0:
            raise ValueError("task_probs 的总权重必须大于 0")

        df = pd.read_parquet(parquet_path)
        df = df[df["duration"] >= 1].reset_index(drop=True)
        # parquet 必须有 label 列与 timing 列 (point_audio / piece_audio); 缺失就 raise,
        # 避免训练时静默 fallback 成 No-time 模式.
        for col in ("label", "point_audio", "piece_audio"):
            if col not in df.columns:
                raise KeyError(
                    f"AudioDataset: parquet '{parquet_path}' 缺少必要列 '{col}'. "
                    f"请确认使用的是 *_with_timing.parquet."
                )
        df["label"] = df["label"].fillna("").astype(str)
        df["point_audio"] = df["point_audio"].apply(_parse_timing)
        df["piece_audio"] = df["piece_audio"].apply(_parse_timing)

        self.df = df
        self.label_dict = self._load_label_dict(label_dict_path)
        self._label_alias_to_key = self._build_label_alias_to_key(self.label_dict)
        for raw_label in df["label"].unique():
            raw_label = str(raw_label)
            canonical = self._label_alias_to_key.get(self._label_norm(raw_label), raw_label)
            self._label_alias_to_key.setdefault(self._label_norm(raw_label), canonical)
            self._label_alias_to_key.setdefault(
                self._label_norm(self._lower_first_label(raw_label)),
                canonical,
            )
        self.audio_base_path = audio_base_path
        self.long_prompt_csv_path = long_prompt_csv_path
        self.long_prompt_audio_base_path = long_prompt_audio_base_path
        self.long_prompt_prob = float(long_prompt_prob)
        if not 0.0 <= self.long_prompt_prob <= 1.0:
            raise ValueError(f"long_prompt_prob 必须在 [0, 1] 内, 收到 {long_prompt_prob}")
        self.long_prompt_tasks = {int(task_id) for task_id in long_prompt_tasks}
        invalid_long_prompt_tasks = self.long_prompt_tasks - {0, 2}
        if invalid_long_prompt_tasks:
            raise ValueError(
                "long_prompt_tasks 只允许 add=0 / generate=2, "
                f"收到 {sorted(invalid_long_prompt_tasks)}"
            )
        self.required_long_prompt_tasks = {
            int(task_id) for task_id in required_long_prompt_tasks
        }
        invalid_required_long_prompt_tasks = self.required_long_prompt_tasks - {0, 2, 3}
        if invalid_required_long_prompt_tasks:
            raise ValueError(
                "required_long_prompt_tasks 只允许 add=0 / generate=2 / replace=3, "
                f"收到 {sorted(invalid_required_long_prompt_tasks)}"
            )
        self.audit_provenance = bool(audit_provenance)
        self._endpoint_audit: List[Dict] = []
        self._audit_task_name = ""
        self.duration = duration
        self.target_audio_size = (1, TARGET_LEN)
        self.sr = SR

        self.target_length = TARGET_LEN
        self.verbose_errors = verbose_errors
        self.processor_max_length = int(processor_max_length)

        self.base_audio_num_probs = list(base_audio_num_probs)
        self.specific_time_probs = list(specific_time_probs)
        self.condition_type_probs = list(condition_type_probs)
        self.use_base_prob = use_base_prob
        self.edit_volume_range = edit_volume_range
        self.edit_pitch_range = edit_pitch_range
        self.edit_pitch_step_quantization = edit_pitch_step_quantization
        self.edit_pitch_shift_cache_size = int(edit_pitch_shift_cache_size)
        self.read_pitch_shift_threads = read_pitch_shift_threads
        self.edit_pitch_shift_threads = (
            read_pitch_shift_threads if edit_pitch_shift_threads is None
            else edit_pitch_shift_threads
        )
        self._edit_pitch_shift_cache: Dict[float, audioT.PitchShift] = {}
        self._edit_pitch_shift_cache_order: List[float] = []

        # ---- 4 个数据桶 ----
        has_point = df["point_audio"].apply(lambda x: x is not None and len(x) > 0)
        has_piece = df["piece_audio"].apply(lambda x: x is not None and len(x) > 0)
        self.dataframe = df
        self.point_dataframe = df[has_point].reset_index(drop=True)
        self.piece_dataframe = df[has_piece].reset_index(drop=True)
        self.point_piece_dataframe = df[has_point & has_piece].reset_index(drop=True)

        self.dataframe_len = len(self.dataframe)
        self.point_dataframe_len = len(self.point_dataframe)
        self.piece_dataframe_len = len(self.piece_dataframe)
        self.point_piece_dataframe_len = len(self.point_piece_dataframe)

        print("总体训练音频数量: ", self.dataframe_len)
        print("含有point的音频数据量: ", self.point_dataframe_len)
        print("含有piece的音频数据量: ", self.piece_dataframe_len)
        print("同时含有point与piece的音频数量", self.point_piece_dataframe_len)

        # ---- bucket 索引 (用于 _pick_row_excluding 的 forbidden 预过滤) ----
        self._bucket_dfs: Dict[str, pd.DataFrame] = {
            "all":         self.dataframe,
            "point":       self.point_dataframe,
            "piece":       self.piece_dataframe,
            "point_piece": self.point_piece_dataframe,
        }
        self._bucket_canonical: Dict[str, pd.Series] = {
            k: v["label"].map(self._canonical_label).reset_index(drop=True)
            for k, v in self._bucket_dfs.items()
        }
        self._bucket_unique_labels: Dict[str, set] = {
            k: set(arr.unique().tolist())
            for k, arr in self._bucket_canonical.items()
        }

        # ---- long-prompt 数据桶; probability=0 / tasks=() / path=None 时不加载 ----
        long_prompt_enabled = bool(
            self.long_prompt_csv_path
            and self.long_prompt_prob > 0.0
            and (self.long_prompt_tasks or self.required_long_prompt_tasks)
        )
        if long_prompt_enabled:
            lp_df = pd.read_csv(self.long_prompt_csv_path)
            if "Unnamed: 0" in lp_df.columns:
                lp_df = lp_df.drop(columns=["Unnamed: 0"])
            for col in ("path", "duration", "label", "point_audio", "piece_audio"):
                if col not in lp_df.columns:
                    raise KeyError(
                        f"AudioDataset: long_prompt csv '{self.long_prompt_csv_path}' "
                        f"缺少必要列 '{col}'."
                    )
            lp_df = lp_df[lp_df["duration"] > 1].reset_index(drop=True)
            lp_df["label"] = lp_df["label"].fillna("").astype(str)
            lp_df["point_audio"] = lp_df["point_audio"].apply(_parse_timing)
            lp_df["piece_audio"] = lp_df["piece_audio"].apply(_parse_timing)

            lp_has_point = lp_df["point_audio"].apply(lambda x: x is not None and len(x) > 0)
            lp_has_piece = lp_df["piece_audio"].apply(lambda x: x is not None and len(x) > 0)
            self.long_prompt_df = lp_df
            self._long_prompt_bucket_dfs: Optional[Dict[str, pd.DataFrame]] = {
                "all": lp_df,
                "point": lp_df[lp_has_point].reset_index(drop=True),
                "piece": lp_df[lp_has_piece].reset_index(drop=True),
                "point_piece": lp_df[lp_has_point & lp_has_piece].reset_index(drop=True),
            }
            self._long_prompt_bucket_canonical: Optional[Dict[str, pd.Series]] = {
                k: v["label"].map(self._canonical_label).reset_index(drop=True)
                for k, v in self._long_prompt_bucket_dfs.items()
            }
            for bucket_kind, bucket_df in self._long_prompt_bucket_dfs.items():
                if len(bucket_df) == 0:
                    raise ValueError(f"long_prompt bucket '{bucket_kind}' 为空")
            print(
                f"[dataset] long_prompt 桶大小: all={len(self._long_prompt_bucket_dfs['all'])} "
                f"point={len(self._long_prompt_bucket_dfs['point'])} "
                f"piece={len(self._long_prompt_bucket_dfs['piece'])} "
                f"point_piece={len(self._long_prompt_bucket_dfs['point_piece'])}"
            )
        else:
            self.long_prompt_df = None
            self._long_prompt_bucket_dfs = None
            self._long_prompt_bucket_canonical = None

        # ---- task_specific ----
        if task_specific is not None:
            assert isinstance(task_specific, torch.Tensor), "task需要是tensor!"
        self.task_specific = task_specific

        if specific_time is not None:
            assert specific_time in ["No", "point", "piece"]
        self.specific_time = specific_time

        # ---- 变换组件 ----
        self.transform = _Transform(
            random_crop=Random_Crop(),
            random_pad=Random_Pad(),
            disrupt_base=Disrupt_base_audio(
                sr=SR,
                pitch_range=read_pitch_range,
                volume_range=read_volume_range,
                pitch_change_prob=pitch_fake_prob,
                volume_change_prob=volume_fake_prob,
            ),
            crop_pad=AudioCropPad(target_length=TARGET_LEN),
            disrupt=Disrupt_audio(
                pitch_fake_prob=pitch_fake_prob,
                volume_fake_prob=volume_fake_prob,
                pitch_range=read_pitch_range,
                volume_range=read_volume_range,
                sr=SR,
                pitch_step_quantization=read_pitch_step_quantization,
                pitch_cache_size=pitch_shift_cache_size,
                pitch_shift_threads=read_pitch_shift_threads,
                pitch_shift_max_seconds=read_pitch_shift_max_seconds,
            ),
        )
        if pitch_shift_prewarm:
            self.transform.disrupt.warmup_pitch_cache(
                length=TARGET_LEN,
                num_threads=pitch_shift_warmup_threads,
            )
            self._warmup_edit_pitch_cache(
                length=TARGET_LEN,
                num_threads=pitch_shift_warmup_threads,
            )

        # Experiment 3 is text-only. Frozen T5 tokenization happens in the
        # training process after collation; no multimodal processor is built.

        # ---- IO 预检 ----
        # 训练时默认开; 若一次性把所有音频都过一遍 torchaudio.info 全通过, 后续 read_typing_data
        # / _load_base_one 就无需运行时 IO 重试. 若失败直接 raise, 让人在启动时定位.
        if verify_io:
            self._verify_all_io()

    def __len__(self):
        return len(self.dataframe)

    # -----------------------------------------------------------------------
    # IO 预检
    # -----------------------------------------------------------------------
    def _verify_all_io(self) -> None:
        """启动阶段过一遍主数据源和启用的 long-prompt 数据源.

        通过后, 后续 load_chunk_full / load_chunk_aligned_10s / load_full_audio 可视为不会 IO 失败,
        无需运行时重试.
        """
        groups = [(self.audio_base_path, self.dataframe["path"].unique())]
        if self.long_prompt_df is not None:
            groups.append(
                (self.long_prompt_audio_base_path, self.long_prompt_df["path"].unique())
            )
        total = sum(len(paths) for _, paths in groups)
        print(f"[dataset] IO 预检: {total} 条唯一音频 ({len(groups)} base_paths)...")
        t0 = time.time()
        bad: List[Tuple[str, str]] = []
        for base_path, paths in groups:
            for p in paths:
                ap = os.path.join(base_path, p)
                try:
                    _ = torchaudio.info(ap)
                except Exception as ex:
                    bad.append((ap, f"{type(ex).__name__}: {ex}"))
        elapsed = time.time() - t0
        if bad:
            head = "\n".join(f"  {ap}: {msg}" for ap, msg in bad[:20])
            raise RuntimeError(
                f"[dataset] IO 预检失败: {len(bad)} 条读不到 (前 20 条):\n{head}"
            )
        print(f"[dataset] IO 预检通过 ({elapsed:.1f}s)")

    # -----------------------------------------------------------------------
    # 抽样辅助
    # -----------------------------------------------------------------------
    def _sample_task(self, task: Optional[torch.Tensor]) -> torch.Tensor:
        if task is None and self.task_specific is None:
            return torch.tensor(random.choices(range(6), weights=self.task_probs, k=1)[0])
        if task is not None:
            assert isinstance(task, torch.Tensor), "task需要是tensor!"
            return task
        return self.task_specific

    def _sample_specific_time(self, specific_time: Optional[str]) -> str:
        if specific_time is None and self.specific_time is None:
            r = random.random()
            p_no = self.specific_time_probs[0]
            p_pt = p_no + self.specific_time_probs[1]
            if r < p_no:
                return "No"
            if r < p_pt:
                return "point"
            return "piece"
        if specific_time is not None:
            assert specific_time in ["No", "point", "piece"]
            return specific_time
        return self.specific_time

    def _sample_condition_type(self, ct: Optional[str]) -> str:
        if ct is not None:
            return ct
        r = random.random()
        p_t = self.condition_type_probs[0]
        p_a = p_t + self.condition_type_probs[1]
        if r < p_t:
            return "t"
        if r < p_a:
            return "a"
        return "ta"

    def _sample_use_base(self, use_base: Optional[bool]) -> bool:
        if use_base is not None:
            return bool(use_base)
        return random.random() < self.use_base_prob

    def _sample_base_count(self) -> int:
        return random.choices([1, 2, 3], weights=self.base_audio_num_probs, k=1)[0]

    @staticmethod
    def _audit_timing_list(value) -> List[List[int]]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        return [[int(item[0]), int(item[1])] for item in value]

    def _record_endpoint_audit(
        self,
        *,
        role: str,
        origin: str,
        path: Optional[str],
        raw_label: str,
        display_label: str,
        wave: torch.Tensor,
        points,
        pieces,
        chunk_start_second: Optional[float] = None,
        chunk_end_second: Optional[float] = None,
        timing_format: str = "start_end_post_transform_samples",
    ) -> None:
        if not self.audit_provenance:
            return
        self._endpoint_audit.append(
            {
                "task": self._audit_task_name,
                "role": role,
                "origin": origin,
                "path": path,
                "raw_label": str(raw_label),
                "display_label": str(display_label),
                "canonical_label": self._canonical_label(str(raw_label)),
                "empty_label": str(raw_label) == "",
                "point_timing_samples": self._audit_timing_list(points),
                "piece_timing_samples": self._audit_timing_list(pieces),
                "timing_format": timing_format,
                "chunk_start_second": chunk_start_second,
                "chunk_end_second": chunk_end_second,
                "decodable": bool(wave.numel() > 0),
                "finite": bool(torch.isfinite(wave).all().item()),
                "duration_seconds": float(wave.shape[-1] / self.sr),
                "final_prompt": "",
            }
        )

    def _attach_endpoint_audit(self, ret: Dict, prompt: str) -> Dict:
        if not self.audit_provenance:
            return ret
        for endpoint in self._endpoint_audit:
            endpoint["final_prompt"] = str(prompt)
        ret["endpoint_audit"] = [dict(item) for item in self._endpoint_audit]
        ret["audit_final_prompt"] = str(prompt)
        return ret

    def _read_add_target_data(
        self,
        task_int: int,
        condition_type: str,
        specific_time: str,
    ):
        use_long_prompt = (
            self._long_prompt_bucket_dfs is not None
            and task_int in self.long_prompt_tasks
            and (
                self.long_prompt_prob >= 1.0
                or random.random() < self.long_prompt_prob
            )
        )
        if use_long_prompt:
            return self.read_long_prompt_data(
                condition_type,
                specific_time,
                jump=True,
                endpoint_role="destination",
            )
        if task_int in self.required_long_prompt_tasks:
            raise RuntimeError(
                f"task={task_int} 要求 100% long-prompt，但本次未命中；"
                "禁止回退到普通 filtered train"
            )
        return self.read_typing_data(
            condition_type,
            specific_time,
            jump=True,
            endpoint_role="destination",
        )

    def _get_edit_pitch_shifter(self, n_step: float) -> audioT.PitchShift:
        key = round(float(n_step), 6)
        cached = self._edit_pitch_shift_cache.get(key)
        if cached is not None:
            return cached
        shifter = audioT.PitchShift(sample_rate=self.sr, n_steps=key)
        self._edit_pitch_shift_cache[key] = shifter
        self._edit_pitch_shift_cache_order.append(key)
        if self.edit_pitch_shift_cache_size > 0:
            while len(self._edit_pitch_shift_cache_order) > self.edit_pitch_shift_cache_size:
                old_key = self._edit_pitch_shift_cache_order.pop(0)
                self._edit_pitch_shift_cache.pop(old_key, None)
        return shifter

    def _apply_edit_pitch_shift(self, audio_wave: torch.Tensor, n_step: float) -> torch.Tensor:
        if abs(float(n_step)) <= 1e-12:
            return audio_wave
        shifter = self._get_edit_pitch_shifter(n_step)
        num_threads = self.edit_pitch_shift_threads
        if num_threads is None or num_threads <= 0:
            return shifter(audio_wave)

        prev_threads = torch.get_num_threads()
        target_threads = int(num_threads)
        if prev_threads == target_threads:
            return shifter(audio_wave)

        torch.set_num_threads(target_threads)
        try:
            return shifter(audio_wave)
        finally:
            torch.set_num_threads(prev_threads)

    def _warmup_edit_pitch_cache(
        self,
        length: int = TARGET_LEN,
        num_threads: Optional[int] = None,
    ) -> None:
        values = _pitch_step_values(self.edit_pitch_range, self.edit_pitch_step_quantization)
        if not values:
            return
        prev_threads = None
        if num_threads is not None and num_threads > 0:
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(int(num_threads))
        warm = torch.zeros(1, int(length))
        try:
            with torch.no_grad():
                for n_step in values:
                    if abs(n_step) > 1e-12:
                        _ = self._get_edit_pitch_shifter(n_step)(warm)
        finally:
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)

    # -----------------------------------------------------------------------
    # 时间字段
    # -----------------------------------------------------------------------
    def get_start_time(self, target_point_piece, specific_time: str) -> torch.Tensor:
        if specific_time == "No":
            return torch.tensor([-1.0])
        if specific_time in ["point", "piece"]:
            return torch.tensor([float(target_point_piece[0]) / SR])
        raise KeyError()

    def get_end_time(self, target_point_piece, specific_time: str) -> torch.Tensor:
        if specific_time == "No":
            return torch.tensor([-1.0])
        if specific_time in ["point", "piece"]:
            return torch.tensor([float(target_point_piece[1]) / SR])
        raise KeyError()

    @staticmethod
    def _split_audio(target_point_piece, audio: torch.Tensor):
        """把 audio 在 target_point_piece 内的部分切出来, 原音频对应区间清零。

        返回 (剩余音频, 目标段音频), 两者形状都和 audio 相同。
        """
        new_audio = torch.zeros_like(audio)
        s = int(target_point_piece[0]); e = int(target_point_piece[1])
        new_audio[:, s:e] = audio[:, s:e]
        audio[:, s:e] = 0
        return audio, new_audio

    @staticmethod
    def _updata_piece(piece: torch.Tensor, bias: int, total_length: int) -> torch.Tensor:
        piece_shift = piece + bias
        return torch.clamp(piece_shift, min=0, max=total_length)

    # -----------------------------------------------------------------------
    # 数据帧选择
    # -----------------------------------------------------------------------
    @staticmethod
    def _bucket_kind_for(condition_type: str, specific_time: str) -> str:
        """按 (condition_type, specific_time) 映射到 4 个数据桶名之一.

        - t + No     -> all
        - t + point  -> point
        - t + piece  -> piece
        - a/ta + No     -> point   (feature_audio 需要 point)
        - a/ta + point  -> point
        - a/ta + piece  -> point_piece   (target=piece + feature=point)
        """
        if condition_type == "t":
            if specific_time == "No":
                return "all"
            if specific_time == "point":
                return "point"
            if specific_time == "piece":
                return "piece"
        elif condition_type in ("a", "ta"):
            if specific_time == "No":
                return "point"
            if specific_time == "point":
                return "point"
            if specific_time == "piece":
                return "point_piece"
        raise NotImplementedError(
            f"Wrong audio typing: condition_type={condition_type}, specific_time={specific_time}"
        )

    def _pick_row_excluding(self, bucket_kind: str, forbidden: set) -> pd.Series:
        """从指定桶随机抽 row, 一次性预过滤掉 canonical label 在 forbidden 中的行.

        forbidden 差集 assert 非空: 数据核实各桶 unique labels >= 159, forbidden <= 3, 不会掏空.
        """
        df = self._bucket_dfs[bucket_kind]
        if not forbidden:
            return df.iloc[random.randint(0, len(df) - 1)]
        remaining = self._bucket_unique_labels[bucket_kind] - forbidden
        assert remaining, (
            f"bucket {bucket_kind} 所有 label 都在 forbidden={forbidden} 中, "
            f"数据 assumption 破了; 请检查数据."
        )
        mask = ~self._bucket_canonical[bucket_kind].isin(forbidden)
        filtered = df[mask.values]
        return filtered.iloc[random.randint(0, len(filtered) - 1)]

    def _pick_typing_row(
        self,
        condition_type: str,
        specific_time: str,
        forbidden: Optional[set] = None,
    ) -> pd.Series:
        bucket_kind = self._bucket_kind_for(condition_type, specific_time)
        return self._pick_row_excluding(bucket_kind, forbidden or set())

    def _row_path_chunks(self, row: pd.Series) -> Tuple[str, float, float]:
        path = row["path"]
        s = float(row.get("chunk_start_second", 0.0)) if not pd.isna(row.get("chunk_start_second", 0.0)) else 0.0
        e_default = s + float(row.get("duration", self.duration))
        e = float(row.get("chunk_end_second", e_default)) if not pd.isna(row.get("chunk_end_second", e_default)) else e_default
        return path, s, e

    # -----------------------------------------------------------------------
    # label 展示文本 / 同义词增强
    # -----------------------------------------------------------------------
    @staticmethod
    def _lower_first_label(label: str) -> str:
        label = remove_period(str(label)).strip()
        if not label:
            return label
        return label[0].lower() + label[1:]

    @staticmethod
    def _label_norm(label: str) -> str:
        return str(label).strip().casefold()

    @staticmethod
    def _label_values(values) -> List[str]:
        if values is None:
            return []
        if isinstance(values, (list, tuple, set)):
            return [str(v) for v in values if str(v).strip()]
        return [str(values)] if str(values).strip() else []

    @classmethod
    def _load_label_dict(cls, label_dict_path: Optional[str]) -> Dict[str, List[str]]:
        if label_dict_path is None:
            return {}
        raw = torch.load(label_dict_path, map_location="cpu")
        if not isinstance(raw, dict):
            raise TypeError(f"label_dict_path 必须加载为 dict, 实际为 {type(raw)!r}: {label_dict_path}")
        return {str(k): cls._label_values(v) for k, v in raw.items()}

    @classmethod
    def _build_label_alias_to_key(cls, label_dict: Dict[str, List[str]]) -> Dict[str, str]:
        alias_to_key: Dict[str, str] = {}
        for key, values in label_dict.items():
            candidates = [str(key)] + cls._label_values(values)
            for cand in candidates:
                alias_to_key[cls._label_norm(cand)] = str(key)
                alias_to_key[cls._label_norm(cls._lower_first_label(cand))] = str(key)
        return alias_to_key

    def _canonical_label(self, label: str) -> str:
        raw = str(label).strip()
        return self._label_alias_to_key.get(self._label_norm(raw), raw)

    def _canonical_label_set(self, labels) -> set:
        return {self._canonical_label(x) for x in (labels or [])}

    def _sample_label_text(self, raw_label: str) -> str:
        key = self._canonical_label(raw_label)
        values = self.label_dict.get(key, [])
        candidates = [key] + list(values)
        picked = random.choice(candidates) if candidates else str(raw_label)
        return self._lower_first_label(picked)

    # -----------------------------------------------------------------------
    # info / feature_audio 构造
    # -----------------------------------------------------------------------
    @staticmethod
    def _get_info(audio_wave: torch.Tensor, condition_type: str, text: str, point: torch.Tensor):
        """返回 (label_feature, feature_audio_np 或 None).

        - condition_type == "t": 直接短路 (不需要 feature audio, 也不检查 point 是否存在).
          这是必须的, 因为 t + specific_time=No 从 all 桶抽, row 可能没 point.
        - condition_type ∈ {a, ta}: 桶保证 point 非空, 用 assert 兜底而非静默退化.
        """
        if condition_type == "t":
            return [remove_period(text), False], None
        assert point is not None and point.shape[0] > 0, \
            f"condition_type={condition_type} 但 point 为空 —— 桶应保证有 point"
        idx = random.randint(0, point.shape[0] - 1)
        seg = point[idx]
        s, e = int(seg[0].item()), int(seg[1].item())
        assert e > s, f"point 段无效 s={s}, e={e}"
        feat = audio_wave[0, s:e].detach().cpu().numpy()
        assert feat.size > 0, "feature audio 切片为空"
        if condition_type == "a":
            return [False, True], feat
        return [remove_period(text), True], feat

    # -----------------------------------------------------------------------
    # base_audio 读取 (1/2/3 条混合)
    # -----------------------------------------------------------------------
    def read_base_audio(
        self,
        idx: int,
        forbidden_labels: Optional[set] = None,
    ) -> Tuple[torch.Tensor, List[str]]:
        """返回 (mix[1, TARGET_LEN], base_labels: List[str])。

        - 第 1 条用 idx 处的样本 (若 idx label 在 forbidden 中则重新按 forbidden 过滤抽)。
        - k>=2 条按 (4.5, 4.5, 1) 概率抽, 每条用 random_crop 到子段长度 (3-10s) 后 random_pad 到 TARGET_LEN。
        - forbidden_labels 用于替换任务: 强制 base.label 不在该集合里。
        - 返回的 base_labels 是随机同义词增强后的展示 label, 且首字母小写。
        """
        forbidden = self._canonical_label_set(forbidden_labels)
        n = self._sample_base_count()
        labels: List[str] = []
        mix = torch.zeros(1, TARGET_LEN)

        # 第 1 条: 优先用 idx
        wave0, raw_label0, label0 = self._load_base_one(idx, forbidden)
        wave0 = self.transform.disrupt_base(wave0)
        wave0 = self.transform.random_crop(wave0, TARGET_LEN)
        wave0 = self.transform.random_pad(wave0, TARGET_LEN)
        mix = mix + wave0
        labels.append(label0)
        forbidden.add(self._canonical_label(raw_label0))

        # 第 2~n 条: 从 all 桶按 forbidden 过滤后抽
        for _ in range(n - 1):
            wave_k, raw_label_k, label_k = self._load_base_one(None, forbidden)
            wave_k = self.transform.disrupt_base(wave_k)
            sub_len = int(random.uniform(3, 10) * SR)
            wave_k = self.transform.random_crop(wave_k, sub_len)
            wave_k = self.transform.random_pad(wave_k, TARGET_LEN)
            mix = mix + wave_k
            labels.append(label_k)
            forbidden.add(self._canonical_label(raw_label_k))

        # 防御性长度归一
        mix = _crop_or_pad_to(mix, TARGET_LEN)
        return mix, labels

    def _load_base_one(
        self,
        idx: Optional[int],
        forbidden: set,
    ) -> Tuple[torch.Tensor, str, str]:
        """加载一条 base 用音频; idx=None 时从 all 桶按 forbidden 预过滤后抽.

        - idx 优先语义保留: 若 idx 提供且其 row 的 canonical_label ∉ forbidden, 直接用.
        - 否则走 _pick_row_excluding("all", forbidden), 一次抽一条 label ∉ forbidden 的 row.
        - IO 由 __init__ verify_io 保证; load_chunk_aligned_10s 极端情况下若返回 None
          (例: chunk 边界得到空 wave) 则 raise, 让人定位数据 bug.

        返回 (wave[1, TARGET_LEN], raw_label, display_label). 不返回 None.
        """
        row = None
        if idx is not None:
            candidate = self.df.iloc[idx]
            if self._canonical_label(str(candidate["label"])) not in forbidden:
                row = candidate
        if row is None:
            row = self._pick_row_excluding("all", forbidden)

        raw_label = str(row["label"])
        path, s, e = self._row_path_chunks(row)
        loaded = load_chunk_aligned_10s(self.audio_base_path, path, s, e)
        if loaded is None:
            raise RuntimeError(
                f"_load_base_one: load_chunk_aligned_10s 返回 None, path={path} "
                f"chunk=[{s},{e}]. verify_io 已通过, 请检查此 chunk 边界."
            )
        wave, _off = loaded
        # Keep the training RNG contract identical with audit_provenance=False:
        # label display sampling must happen exactly once per endpoint.
        display_label = self._sample_label_text(raw_label)
        self._record_endpoint_audit(
            role="base",
            origin="filtered_train",
            path=str(path),
            raw_label=raw_label,
            display_label=display_label,
            wave=wave,
            points=row["point_audio"],
            pieces=row["piece_audio"],
            chunk_start_second=float(s),
            chunk_end_second=float(e),
            timing_format="start_duration_source_row_samples",
        )
        return wave, raw_label, display_label

    # -----------------------------------------------------------------------
    # 非 base 路径读取
    # -----------------------------------------------------------------------
    def read_typing_data(
        self,
        condition_type: str,
        specific_time: str,
        jump: bool = False,
        forbidden_labels: Optional[set] = None,
        return_full_timing: bool = False,
        endpoint_role: str = "source",
    ):
        """返回 (wave[1, TARGET_LEN], info, target_point_piece, label_text).

        新流程 (对齐第四阶段 dataset):
          1. 按 (condition_type, specific_time) 选桶
          2. 用 forbidden 一次性 vectorized 过滤, 直接抽 row
          3. load_chunk_full: 只读 chunk 段, 不 crop/pad
          4. Disrupt_audio: pitch(<12s时) + volume 加强, 长度不变
          5. _get_info: 从完整 disrupt 后 wave 上切 feature audio
             (⇒ feature audio 与最终 10s 里的目标声音共享同一次 pitch/volume 加强)
          6. specific_time in {point, piece}: 从完整 timing tensor 里选 target
          7. AudioCropPad(piece=target): 保护裁切到 10s
          8. bias 平移 target 与 (可选) 全部 timing 到 10s 坐标系
          9. 全部 timing: 部分越界截断, 完全越界丢弃 (_shift_and_drop_oob)

        forbidden_labels: 替换/去除等任务用; 桶差集非空由数据 assumption 保证 (assert 兜底).
        return_full_timing: True 时额外返回 (full_points_t, full_pieces_t) — 已平移到 10s
            坐标系并处理越界. 供 replace-No 分支使用.

        无内部重试. IO 由 __init__ 阶段 verify_io 保证; forbidden 差集非空由数据保证.
        任何失败都是数据 bug, 直接 raise 让人定位.
        """
        forbidden = self._canonical_label_set(forbidden_labels)
        bucket_kind = self._bucket_kind_for(condition_type, specific_time)
        row = self._pick_row_excluding(bucket_kind, forbidden)

        raw_label = str(row["label"])
        label = self._sample_label_text(raw_label)
        path, s_sec, e_sec = self._row_path_chunks(row)

        wave = load_chunk_full(self.audio_base_path, path, s_sec, e_sec)  # [1, T_raw]
        T_raw = wave.shape[1]

        # 原 timing (list of (start, duration)); clamp 到 [0, T_raw] 防表格越界; 不 shift.
        points_raw = _clamp_segs_to_len(row["point_audio"], T_raw)
        pieces_raw = _clamp_segs_to_len(row["piece_audio"], T_raw)

        # Disrupt: pitch(<12s) + vol; 内部完成 (s,d)->(s,e) 格式转换
        wave, point_t, piece_t = self.transform.disrupt(
            wave, points_raw, pieces_raw, jump=jump,
        )

        # feature audio (在完整 wave 上切; t 分支短路不用 point)
        info = self._get_info(wave, condition_type, label, point_t)

        # target 选择 (从完整 wave 的 timing 里选)
        if specific_time == "point":
            assert point_t.shape[0] > 0, \
                f"point 桶抽到无 point 的 row: {path}"
            target_point = point_t[random.randint(0, point_t.shape[0] - 1)]
        elif specific_time == "piece":
            assert piece_t.shape[0] > 0, \
                f"piece 桶抽到无 piece 的 row: {path}"
            target_point = piece_t[random.randint(0, piece_t.shape[0] - 1)]
        else:
            target_point = None

        # 保护裁切到 10s (AudioCropPad 内部对 piece 参数已实现保护 crop; wave<=10s 走 pad)
        wave_out, _, _, bias = self.transform.crop_pad(wave, target_point)
        wave_out = _crop_or_pad_to(wave_out, TARGET_LEN)

        if target_point is not None:
            target_point = self._updata_piece(target_point, bias, TARGET_LEN)

        full_points = _shift_and_drop_oob(point_t, bias, TARGET_LEN)
        full_pieces = _shift_and_drop_oob(piece_t, bias, TARGET_LEN)
        self._record_endpoint_audit(
            role=endpoint_role,
            origin="filtered_train",
            path=str(path),
            raw_label=raw_label,
            display_label=label,
            wave=wave_out,
            points=full_points,
            pieces=full_pieces,
            chunk_start_second=float(s_sec),
            chunk_end_second=float(e_sec),
        )
        if return_full_timing:
            # replace-No 用: 除 target 外的 point/piece 段可能部分/完全越界, 用 shift+drop_oob 处理
            return wave_out, info, target_point, label, full_points, full_pieces
        return wave_out, info, target_point, label

    # -----------------------------------------------------------------------
    # long-prompt 添加/生成目标音频
    # -----------------------------------------------------------------------
    def _pick_long_prompt_row(
        self,
        bucket_kind: str,
        forbidden_labels: Optional[set] = None,
    ) -> pd.Series:
        if self._long_prompt_bucket_dfs is None:
            raise RuntimeError("long_prompt 数据源未启用")
        df = self._long_prompt_bucket_dfs[bucket_kind]
        forbidden = self._canonical_label_set(forbidden_labels)
        if not forbidden:
            return df.iloc[random.randint(0, len(df) - 1)]
        canonical = self._long_prompt_bucket_canonical[bucket_kind]
        mask = ~canonical.isin(forbidden)
        filtered = df[mask.values]
        if len(filtered) == 0:
            raise RuntimeError(
                f"long_prompt bucket {bucket_kind} 在 canonical forbidden={forbidden} "
                "下没有合法 destination；禁止回退普通 train"
            )
        return filtered.iloc[random.randint(0, len(filtered) - 1)]

    @staticmethod
    def _long_prompt_label(raw_label: str) -> str:
        """Strip, remove a trailing period, and lowercase only the first character."""
        label = remove_period(str(raw_label).strip())
        if not label:
            return label
        return label[0].lower() + label[1:]

    def read_long_prompt_data(
        self,
        condition_type: str,
        specific_time: str,
        jump: bool = True,
        forbidden_labels: Optional[set] = None,
        return_full_timing: bool = False,
        endpoint_role: str = "destination",
    ):
        """Read a long-prompt target; optional arguments preserve the old call result."""
        bucket_kind = self._bucket_kind_for(condition_type, specific_time)
        row = self._pick_long_prompt_row(bucket_kind, forbidden_labels)

        label = self._long_prompt_label(str(row["label"]))
        path = row["path"]
        wave = load_full_audio(self.long_prompt_audio_base_path, path)
        raw_length = wave.shape[1]

        points_raw = _clamp_segs_to_len(_unwrap_lp_timing(row["point_audio"]), raw_length)
        pieces_raw = _clamp_segs_to_len(_unwrap_lp_timing(row["piece_audio"]), raw_length)
        wave, point_t, piece_t = self.transform.disrupt(
            wave, points_raw, pieces_raw, jump=jump,
        )
        info = self._get_info(wave, condition_type, label, point_t)

        if specific_time == "point":
            assert point_t.shape[0] > 0, f"long_prompt point 桶抽到无 point 的 row: {path}"
            target_point = point_t[random.randint(0, point_t.shape[0] - 1)]
        elif specific_time == "piece":
            assert piece_t.shape[0] > 0, f"long_prompt piece 桶抽到无 piece 的 row: {path}"
            target_point = piece_t[random.randint(0, piece_t.shape[0] - 1)]
        else:
            target_point = None

        wave_out, _, _, bias = self.transform.crop_pad(wave, target_point)
        wave_out = _crop_or_pad_to(wave_out, TARGET_LEN)
        if target_point is not None:
            target_point = self._updata_piece(target_point, bias, TARGET_LEN)
        full_points = _shift_and_drop_oob(point_t, bias, TARGET_LEN)
        full_pieces = _shift_and_drop_oob(piece_t, bias, TARGET_LEN)
        self._record_endpoint_audit(
            role=endpoint_role,
            origin="training_only_long_prompt",
            path=str(path),
            raw_label=str(row["label"]),
            display_label=label,
            wave=wave_out,
            points=full_points,
            pieces=full_pieces,
        )
        if return_full_timing:
            return wave_out, info, target_point, label, full_points, full_pieces
        return wave_out, info, target_point, label

    # -----------------------------------------------------------------------
    # 替换任务用: b/c 二元读取 (走 read_typing_data 通用路径)
    # -----------------------------------------------------------------------
    def _read_replace_pair(
        self,
        condition_type_b: str,
        condition_type_c: str,
        pool_kind: str,                       # "piece" / "point" — 限定 b/c 数据来源
        forbidden_labels: set,
    ):
        """读取一对 b/c 音频用于替换任务.

        用法:
          - specific_time == "No":      调用方传 pool_kind ∈ {"piece","point"} 决定从哪个桶抽;
          - specific_time == "point":   pool_kind="point"
          - specific_time == "piece":   pool_kind="piece"

        b 走 read_typing_data(jump=False)  — 走完整 disrupt(pitch+volume).
        c 走 read_typing_data(jump=True)   — 不做 disrupt, 保持原音频.
        feature_audio 在 read_typing_data 内 _get_info 取得, 都基于"替换前"的 raw b/c.

        返回:
            wave_b, b_points (Tensor[*,2] s/e), b_pieces (Tensor[*,2] s/e),
                info_b, target_pp_b (Tensor[2] s/e), label_b,
            wave_c, c_points (Tensor[*,2] s/e), c_pieces (Tensor[*,2] s/e),
                info_c, target_pp_c (Tensor[2] s/e), label_c
        """
        assert pool_kind in ("piece", "point"), f"pool_kind 必须是 piece/point, 收到 {pool_kind}"

        # ---- b ----
        wave_b, info_b, target_pp_b, label_b, b_points_full, b_pieces_full = self.read_typing_data(
            condition_type_b, pool_kind, jump=False,
            forbidden_labels=forbidden_labels,
            return_full_timing=True,
            endpoint_role="source",
        )

        # ---- c ----
        forbidden_for_c = set(forbidden_labels) | {label_b}
        if 3 in self.required_long_prompt_tasks:
            wave_c, info_c, target_pp_c, label_c, c_points_full, c_pieces_full = self.read_long_prompt_data(
                condition_type_c, pool_kind, jump=True,
                forbidden_labels=forbidden_for_c,
                return_full_timing=True,
                endpoint_role="destination",
            )
        else:
            wave_c, info_c, target_pp_c, label_c, c_points_full, c_pieces_full = self.read_typing_data(
                condition_type_c, pool_kind, jump=True,
                forbidden_labels=forbidden_for_c,
                return_full_timing=True,
                endpoint_role="destination",
            )

        return (
            wave_b, b_points_full, b_pieces_full, info_b, target_pp_b, label_b,
            wave_c, c_points_full, c_pieces_full, info_c, target_pp_c, label_c,
        )

    # -----------------------------------------------------------------------
    # processor 调用
    # -----------------------------------------------------------------------
    def _call_processor(
        self,
        text: List[str],
        ref_audio: torch.Tensor,
        feature_audio: Optional[np.ndarray] = None,
        feature_audio2: Optional[np.ndarray] = None,
    ) -> Dict:
        del ref_audio, feature_audio, feature_audio2
        if len(text) != 1 or not isinstance(text[0], str):
            raise ValueError(f"expected one plain-text prompt, got {text!r}")
        return {"prompt_text": text[0]}

    # -----------------------------------------------------------------------
    # 统一 ret 构造
    # -----------------------------------------------------------------------
    @staticmethod
    def _empty_feat_tensor() -> torch.Tensor:
        return torch.tensor([[0.0]])

    @staticmethod
    def _wrap_feat(feat: Optional[np.ndarray]) -> torch.Tensor:
        if feat is None:
            return torch.tensor([[0.0]])
        t = torch.tensor(feat, dtype=torch.float32)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t.detach()

    # -----------------------------------------------------------------------
    # __getitem__
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def __getitem__(
        self,
        idx,
        task: Optional[torch.Tensor] = None,
        change_vol: Optional[float] = None,
        change_Pitch: Optional[float] = None,
        use_base: Optional[bool] = None,
        specific_time: Optional[str] = None,
        condition_type: Optional[str] = None,
        condition_type2: Optional[str] = None,
    ):
        task = self._sample_task(task)
        specific_time = self._sample_specific_time(specific_time)
        condition_type = self._sample_condition_type(condition_type)
        condition_type2 = self._sample_condition_type(condition_type2)
        use_base = self._sample_use_base(use_base)

        task_int = int(task.item()) if isinstance(task, torch.Tensor) else int(task)
        if self.audit_provenance:
            self._endpoint_audit = []
            self._audit_task_name = {
                0: "add",
                1: "remove",
                2: "generate",
                3: "replace",
                4: "volume",
                5: "pitch",
            }[task_int]

        # -------------------------------------------------------------------
        # task 0: 添加
        # -------------------------------------------------------------------
        if task_int == 0:
            wave, base_labels = self.read_base_audio(idx)
            wave2, info, target_point_piece, add_label = self._read_add_target_data(
                task_int, condition_type, specific_time,
            )
            if specific_time != "No":
                _, wave2 = self._split_audio(target_point_piece, wave2)
            ref_audio = wave
            tar_audio = wave + wave2

            text = add_special_token(0, info[0], specific_time, target_point_piece)
            feat = info[1] if condition_type in {"a", "ta"} else None
            features = self._call_processor(text, ref_audio, feat, None)
            ret = self._build_ret(
                task=task,
                tar_audio=tar_audio, ref_audio=ref_audio,
                add_labels=[add_label], remove_labels=[], base_labels=base_labels,
                pitch_change_label=[], volume_change_label=[],
                change_vol=0.0, change_Pitch=0.0,
                feature_audio=feat, feature_audio2=None,
                target_point_piece=target_point_piece, specific_time=specific_time,
                use_base=True, type1=condition_type, type2=None,
            )
            ret.update(features)
            return self._attach_endpoint_audit(ret, text[0])

        # -------------------------------------------------------------------
        # task 1: 去除
        # -------------------------------------------------------------------
        if task_int == 1:
            if not use_base:
                wave = torch.zeros(self.target_audio_size)
                base_labels: List[str] = []
            else:
                wave, base_labels = self.read_base_audio(idx)
            wave2, info, target_point_piece, remove_label = self.read_typing_data(
                condition_type, specific_time, jump=False,
                forbidden_labels=set(base_labels),
            )
            if specific_time != "No":
                remaining, wave2 = self._split_audio(target_point_piece, wave2)
                wave = wave + remaining
            ref_audio = wave + wave2
            tar_audio = wave

            text = add_special_token(1, info[0], specific_time, target_point_piece)
            feat = info[1] if condition_type in {"a", "ta"} else None
            features = self._call_processor(text, ref_audio, feat, None)
            ret = self._build_ret(
                task=task,
                tar_audio=tar_audio, ref_audio=ref_audio,
                add_labels=[], remove_labels=[remove_label], base_labels=base_labels,
                pitch_change_label=[], volume_change_label=[],
                change_vol=0.0, change_Pitch=0.0,
                feature_audio=feat, feature_audio2=None,
                target_point_piece=target_point_piece, specific_time=specific_time,
                use_base=use_base, type1=condition_type, type2=None,
            )
            ret.update(features)
            return self._attach_endpoint_audit(ret, text[0])

        # -------------------------------------------------------------------
        # task 2: 生成
        # -------------------------------------------------------------------
        if task_int == 2:
            wave2, info, target_point_piece, add_label = self._read_add_target_data(
                task_int, condition_type, specific_time,
            )
            if specific_time != "No":
                _, wave2 = self._split_audio(target_point_piece, wave2)
            tar_audio = wave2
            ref_audio = torch.zeros_like(wave2)
            self._record_endpoint_audit(
                role="reference",
                origin="synthetic_silence",
                path=None,
                raw_label="synthetic_silence",
                display_label="synthetic_silence",
                wave=ref_audio,
                points=[],
                pieces=[],
                timing_format="none",
            )

            text = add_special_token(2, info[0], specific_time, target_point_piece)
            feat = info[1] if condition_type in {"a", "ta"} else None
            features = self._call_processor(text, ref_audio, feat, None)
            ret = self._build_ret(
                task=task,
                tar_audio=tar_audio, ref_audio=ref_audio,
                add_labels=[add_label], remove_labels=[], base_labels=[],
                pitch_change_label=[], volume_change_label=[],
                change_vol=0.0, change_Pitch=0.0,
                feature_audio=feat, feature_audio2=None,
                target_point_piece=target_point_piece, specific_time=specific_time,
                use_base=False, type1=condition_type, type2=None,
            )
            ret.update(features)
            return self._attach_endpoint_audit(ret, text[0])

        # -------------------------------------------------------------------
        # task 3: 替换
        #   No  : 综合替换 (mask + piece+point 一起对位贴 c). b/c 同来自 piece 桶
        #         或 point 桶 (1:1). ref/tar 都先做 _mask_to_sounding_only.
        #   point: 只换 b 的指定 target point 位置, 其它位置保留. 不归零非 target 区.
        #   piece: 只换 b 的指定 target piece 位置, 其它位置保留. 不归零非 target 区.
        #          含 swap (确保 c 段 ≥ b 段, 与第三阶段一致).
        # 所有 feature_audio 都基于"替换前"的 b/c raw 取得; swap 时一起对调.
        # -------------------------------------------------------------------
        if task_int == 3:
            # ---- 1) 决定 pool_kind 与是否 mask ----
            #   No 模式: 1:1 从 piece/point 桶随机选一个;
            #   point/piece 模式: 强制对应桶.
            if specific_time == "No":
                pool_kind = "piece" if random.random() < 0.5 else "point"
            elif specific_time == "point":
                pool_kind = "point"
            else:  # piece
                pool_kind = "piece"

            # ---- 2) 先读 b/c (base 还没读, forbidden=空; base 后置抽取) ----
            (
                wave_b, b_points, b_pieces, info_b, target_pp_b, label_b,
                wave_c, c_points, c_pieces, info_c, target_pp_c, label_c,
            ) = self._read_replace_pair(
                condition_type, condition_type2, pool_kind, forbidden_labels=set(),
            )

            # ---- 3) 抽 base 音频; forbidden 包含 b/c 的 label ----
            if not use_base:
                wave = torch.zeros(self.target_audio_size)
                base_labels: List[str] = []
            else:
                wave, base_labels = self.read_base_audio(
                    idx, forbidden_labels={label_b, label_c},
                )

            # ---- 4) 三模式分支 ----
            if specific_time == "No":
                # 综合替换: mask + 配对写 c
                b_points_sd = [(int(b_points[i, 0].item()),
                                int(b_points[i, 1].item() - b_points[i, 0].item()))
                               for i in range(b_points.shape[0])]
                b_pieces_sd = [(int(b_pieces[i, 0].item()),
                                int(b_pieces[i, 1].item() - b_pieces[i, 0].item()))
                               for i in range(b_pieces.shape[0])]
                c_points_sd = [(int(c_points[i, 0].item()),
                                int(c_points[i, 1].item() - c_points[i, 0].item()))
                               for i in range(c_points.shape[0])]
                c_pieces_sd = [(int(c_pieces[i, 0].item()),
                                int(c_pieces[i, 1].item() - c_pieces[i, 0].item()))
                               for i in range(c_pieces.shape[0])]
                ref_clean, tar_clean = _replace_no_combined(
                    wave_b, b_points_sd, b_pieces_sd,
                    wave_c, c_points_sd, c_pieces_sd,
                )
                ref_audio = wave + ref_clean
                tar_audio = wave + tar_clean
                t_pp = None  # No 模式没有时间戳
                feat_b_for_text = info_b[0]
                feat_c_for_text = info_c[0]
                feat_b_audio = info_b[1]
                feat_c_audio = info_c[1]
                ct1, ct2 = condition_type, condition_type2
                lab_remove, lab_add = label_b, label_c

            elif specific_time == "point":
                # 只替换 b 上 target_pp_b 这一段; b 其余位置原样.
                # c 段过长截到 b 边界 (与第三阶段 trans_audio_start_time(point) 一致).
                wave_b_replaced = _replace_point_target(
                    wave_b, target_pp_b, wave_c, target_pp_c,
                )
                ref_audio = wave + wave_b
                tar_audio = wave + wave_b_replaced
                t_pp = target_pp_b
                feat_b_for_text = info_b[0]
                feat_c_for_text = info_c[0]
                feat_b_audio = info_b[1]
                feat_c_audio = info_c[1]
                ct1, ct2 = condition_type, condition_type2
                lab_remove, lab_add = label_b, label_c

            else:  # piece
                # 第三阶段做法: 先比 b/c 时长, 若 b_dur >= c_dur 则 swap b/c
                # (保证写入时不溢出, 始终用较短的当被替换段).
                # swap 时 wave / target / info / label / condition_type 全部对调.
                bd = int(target_pp_b[1] - target_pp_b[0])
                cd = int(target_pp_c[1] - target_pp_c[0])
                if bd >= cd:
                    wave_b, wave_c = wave_c, wave_b
                    target_pp_b, target_pp_c = target_pp_c, target_pp_b
                    info_b, info_c = info_c, info_b
                    label_b, label_c = label_c, label_b
                    ct1, ct2 = condition_type2, condition_type
                else:
                    ct1, ct2 = condition_type, condition_type2

                wave_b_replaced = _replace_piece_target(
                    wave_b, target_pp_b, wave_c, target_pp_c,
                )
                ref_audio = wave + wave_b
                tar_audio = wave + wave_b_replaced
                t_pp = target_pp_b
                feat_b_for_text = info_b[0]
                feat_c_for_text = info_c[0]
                feat_b_audio = info_b[1]
                feat_c_audio = info_c[1]
                lab_remove, lab_add = label_b, label_c

            # ---- 5) prompt + processor + ret ----
            text = add_special_token(3, [feat_b_for_text, feat_c_for_text], specific_time, t_pp)
            feat_b_proc = feat_b_audio if ct1 in {"a", "ta"} else None
            feat_c_proc = feat_c_audio if ct2 in {"a", "ta"} else None
            features = self._call_processor(text, ref_audio, feat_b_proc, feat_c_proc)

            ret = self._build_ret(
                task=task,
                tar_audio=tar_audio, ref_audio=ref_audio,
                add_labels=[lab_add], remove_labels=[lab_remove], base_labels=base_labels,
                pitch_change_label=[], volume_change_label=[],
                change_vol=0.0, change_Pitch=0.0,
                feature_audio=feat_b_proc, feature_audio2=feat_c_proc,
                target_point_piece=t_pp, specific_time=specific_time,
                use_base=use_base, type1=ct1, type2=ct2,
            )
            ret.update(features)
            return self._attach_endpoint_audit(ret, text[0])

        # -------------------------------------------------------------------
        # task 4: 音量变化
        # -------------------------------------------------------------------
        if task_int == 4:
            if not use_base:
                wave = torch.zeros(self.target_audio_size)
                base_labels = []
            else:
                wave, base_labels = self.read_base_audio(idx)

            wave2, info, target_point_piece, change_label = self.read_typing_data(
                condition_type, specific_time, jump=False,
                forbidden_labels=set(base_labels),
            )
            if specific_time != "No":
                remaining, wave2 = self._split_audio(target_point_piece, wave2)
                wave = wave + remaining

            if change_vol is None:
                change = random.uniform(-self.edit_volume_range, self.edit_volume_range)
            else:
                assert isinstance(change_vol, float), "音量变化以float类型输入"
                change = change_vol
            gain = 10 ** (change / 20)
            wave2_changed = wave2 * gain

            ref_audio = wave + wave2
            tar_audio = wave + wave2_changed

            text = add_special_token(4, [info[0], change], specific_time, target_point_piece)
            feat = info[1] if condition_type in {"a", "ta"} else None
            features = self._call_processor(text, ref_audio, feat, None)
            ret = self._build_ret(
                task=task,
                tar_audio=tar_audio, ref_audio=ref_audio,
                add_labels=[], remove_labels=[], base_labels=base_labels,
                pitch_change_label=[], volume_change_label=[change_label],
                change_vol=float(change), change_Pitch=0.0,
                feature_audio=feat, feature_audio2=None,
                target_point_piece=target_point_piece, specific_time=specific_time,
                use_base=use_base, type1=condition_type, type2=None,
            )
            ret.update(features)
            return self._attach_endpoint_audit(ret, text[0])

        # -------------------------------------------------------------------
        # task 5: 音调变化
        # -------------------------------------------------------------------
        if task_int == 5:
            if not use_base:
                wave = torch.zeros(self.target_audio_size)
                base_labels = []
            else:
                wave, base_labels = self.read_base_audio(idx)

            wave2, info, target_point_piece, change_label = self.read_typing_data(
                condition_type, specific_time, jump=False,
                forbidden_labels=set(base_labels),
            )
            if specific_time != "No":
                remaining, wave2 = self._split_audio(target_point_piece, wave2)
                wave = wave + remaining

            if change_Pitch is None:
                change = _sample_quantized_pitch_step(
                    self.edit_pitch_range,
                    self.edit_pitch_step_quantization,
                )
            else:
                assert isinstance(change_Pitch, float), "音调变化以float类型输入"
                change = _quantize_pitch_step(
                    change_Pitch,
                    self.edit_pitch_range,
                    self.edit_pitch_step_quantization,
                )

            wave2_changed = self._apply_edit_pitch_shift(wave2, change)
            wave2_changed = _crop_or_pad_to(wave2_changed, TARGET_LEN)

            ref_audio = wave + wave2
            tar_audio = wave + wave2_changed

            text = add_special_token(5, [info[0], change], specific_time, target_point_piece)
            feat = info[1] if condition_type in {"a", "ta"} else None
            features = self._call_processor(text, ref_audio, feat, None)
            ret = self._build_ret(
                task=task,
                tar_audio=tar_audio, ref_audio=ref_audio,
                add_labels=[], remove_labels=[], base_labels=base_labels,
                pitch_change_label=[change_label], volume_change_label=[],
                change_vol=0.0, change_Pitch=float(change),
                feature_audio=feat, feature_audio2=None,
                target_point_piece=target_point_piece, specific_time=specific_time,
                use_base=use_base, type1=condition_type, type2=None,
            )
            ret.update(features)
            return self._attach_endpoint_audit(ret, text[0])

        raise KeyError(f"未知 task: {task_int}")

    # -----------------------------------------------------------------------
    # ret 工厂
    # -----------------------------------------------------------------------
    def _build_ret(
        self,
        task,
        tar_audio: torch.Tensor,
        ref_audio: torch.Tensor,
        add_labels: List[str],
        remove_labels: List[str],
        base_labels: List[str],
        pitch_change_label: List[str],
        volume_change_label: List[str],
        change_vol: float,
        change_Pitch: float,
        feature_audio: Optional[np.ndarray],
        feature_audio2: Optional[np.ndarray],
        target_point_piece,
        specific_time: str,
        use_base: bool,
        type1: str,
        type2: Optional[str],
    ) -> Dict:
        # 严格防御性长度归一
        ref_audio = _crop_or_pad_to(ref_audio, TARGET_LEN).detach().contiguous()
        tar_audio = _crop_or_pad_to(tar_audio, TARGET_LEN).detach().contiguous()

        if specific_time == "No" or target_point_piece is None:
            start_time = torch.tensor([-1.0])
            end_time = torch.tensor([-1.0])
        else:
            start_time = self.get_start_time(target_point_piece, specific_time).detach()
            end_time = self.get_end_time(target_point_piece, specific_time).detach()

        return {
            "task": task.detach() if isinstance(task, torch.Tensor) else task,
            "tar_audio": tar_audio,
            "ref_audio": ref_audio,
            "add_labels": add_labels,
            "remove_labels": remove_labels,
            "base_labels": base_labels,
            "pitch_change_label": pitch_change_label,
            "volume_change_label": volume_change_label,
            "change_vol": torch.tensor(float(change_vol)),
            "change_Pitch": torch.tensor(float(change_Pitch)),
            "feature_audio": self._wrap_feat(feature_audio),
            "feature_audio2": self._wrap_feat(feature_audio2),
            "start_time": start_time,
            "end_time": end_time,
            "specific_time": specific_time,
            "use_base": use_base,
            "type1": type1,
            "type2": type2,
        }
