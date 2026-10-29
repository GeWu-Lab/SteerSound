from __future__ import annotations

import torch
from einops import rearrange


def standard_fm_mse(v_pred_tokens, target, noise):
    """The only optimization loss in Experiment 3."""
    _, _, height, _ = target.shape
    prediction = rearrange(
        v_pred_tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=(height + 1) // 2,
        ph=2,
        pw=2,
    )
    prediction = prediction[:, :, : target.shape[2], : target.shape[3]].float()
    velocity_target = noise.float() - target.float()
    return torch.mean((prediction - velocity_target) ** 2)
