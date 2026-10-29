#!/usr/bin/env python3
"""Frozen user-specified 50-step Euler/CFG inference for Stage-1.

The conditional and null branches share the same reference latent, task,
change amount, noisy latent and timestep.  Only their text differs. Guidance
is always ``v_null + scale * (v_cond - v_null)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from einops import rearrange


NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 50
EDIT_VOLUME_RANGE_DB = 5.0
EDIT_PITCH_RANGE_SEMITONES = 4.0
PITCH_QUANTIZATION_STEP = 0.5
CFG_SCALES = (3.5, 8.5)
CFG_FORMULAS = ("standard",)
CFG_NULL_CONTRACTS = ("same_task_same_reference",)
TRACE_STEPS = frozenset((0, 1, 4, 9, 24, 49))


class ContractError(RuntimeError):
    """Raised when a frozen inference contract is violated."""


@dataclass(frozen=True)
class BranchInputs:
    ref_img: torch.Tensor
    text: torch.Tensor
    text_mask: torch.Tensor
    task: torch.Tensor
    change_vol: torch.Tensor
    change_pitch: torch.Tensor


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def array_sha256(value: Sequence[float]) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def quantize_pitch_tensor(
    value: torch.Tensor,
    pitch_range: float = EDIT_PITCH_RANGE_SEMITONES,
    step: float = PITCH_QUANTIZATION_STEP,
) -> torch.Tensor:
    if step <= 0:
        return value
    return torch.clamp(torch.round(value / step) * step, -pitch_range, pitch_range)


def pad_image(image: torch.Tensor) -> torch.Tensor:
    height_dimension = 2
    needed = (2 - (image.shape[height_dimension] % 2)) % 2
    paddings = [0] * (image.dim() * 2)
    paddings[2 * (image.dim() - height_dimension) - 1] = needed
    return F.pad(image, tuple(paddings), mode="constant", value=0)


def _prepare_branch(
    branch: BranchInputs,
    *,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if branch.ref_img.shape[0] != batch_size:
        raise ContractError(
            f"reference batch {branch.ref_img.shape[0]} != noise batch {batch_size}"
        )
    if branch.text.shape[0] != batch_size:
        raise ContractError("text condition batch does not match noise batch")
    ref = pad_image(branch.ref_img.to(device=device, dtype=dtype))
    ref = rearrange(
        ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
    )
    txt = branch.text.to(device=device, dtype=dtype)
    txt_mask = branch.text_mask.to(device=device, dtype=torch.bool)
    txt_ids = torch.zeros(batch_size, txt.shape[1], 3, device=device)
    task = branch.task.to(device=device, dtype=dtype) / 5.0
    volume = (
        branch.change_vol.to(device=device, dtype=dtype) + EDIT_VOLUME_RANGE_DB
    ) / (2.0 * EDIT_VOLUME_RANGE_DB)
    pitch = quantize_pitch_tensor(
        branch.change_pitch.to(device=device, dtype=dtype)
    )
    pitch = (pitch + EDIT_PITCH_RANGE_SEMITONES) / (
        2.0 * EDIT_PITCH_RANGE_SEMITONES
    )
    return {
        "ref_img": ref,
        "txt": txt,
        "txt_ids": txt_ids,
        "txt_mask": txt_mask,
        "task": task,
        "change_vol": volume,
        "change_Pitch": pitch,
    }


def _img_ids(
    *, batch_size: int, height: int, width: int, device: str
) -> torch.Tensor:
    value = torch.zeros((height + 1) // 2, width // 2, 3, device=device)
    value[..., 1] += torch.arange((height + 1) // 2, device=device)[:, None]
    value[..., 2] += torch.arange(width // 2, device=device)[None, :]
    return value.unsqueeze(0).repeat(batch_size, 1, 1, 1).reshape(
        batch_size, -1, 3
    )


def _unpack_velocity(
    velocity: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    return rearrange(
        velocity,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=(height + 1) // 2,
        ph=2,
        pw=2,
    )[:, :, :height, :width]


def _rms_per_sample(value: torch.Tensor) -> list[float]:
    flattened = value.detach().float().reshape(value.shape[0], -1)
    return [float(item) for item in flattened.square().mean(dim=1).sqrt().cpu()]


@torch.no_grad()
def inference_dit_explicit(
    dit: torch.nn.Module,
    conditional: BranchInputs,
    initial_noise: torch.Tensor,
    *,
    cfg_scale: float,
    null: BranchInputs | None,
    cfg_formula: str = "standard",
    cfg_null_contract: str = "same_task_same_reference",
    device: str,
    dtype: torch.dtype,
    num_steps: int = NUM_INFERENCE_STEPS,
    collect_trace: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run frozen Euler inference using caller-provided noise.

    The matched null contract keeps task/reference/edit scalars fixed and
    changes only the T5 instruction.
    """
    if num_steps != NUM_INFERENCE_STEPS:
        raise ContractError(f"formal inference requires 50 steps, got {num_steps}")
    if cfg_scale not in CFG_SCALES:
        raise ContractError(f"unsupported frozen CFG scale: {cfg_scale}")
    if cfg_formula not in CFG_FORMULAS:
        raise ContractError(f"unsupported CFG formula: {cfg_formula}")
    if cfg_null_contract not in CFG_NULL_CONTRACTS:
        raise ContractError(f"unsupported CFG null contract: {cfg_null_contract}")
    if null is None:
        raise ContractError("CFG arm requires the explicit empty null branch")
    if not torch.equal(conditional.ref_img.detach().cpu(), null.ref_img.detach().cpu()):
        raise ContractError("matched CFG null reference differs from conditional")
    if not torch.equal(conditional.task.detach().cpu(), null.task.detach().cpu()):
        raise ContractError("matched CFG null task differs from conditional")
    if not torch.equal(
        conditional.change_vol.detach().cpu(), null.change_vol.detach().cpu()
    ):
        raise ContractError("matched CFG null volume differs from conditional")
    if not torch.equal(
        conditional.change_pitch.detach().cpu(), null.change_pitch.detach().cpu()
    ):
        raise ContractError("matched CFG null pitch differs from conditional")

    if initial_noise.ndim != 4 or not torch.isfinite(initial_noise).all():
        raise ContractError("initial noise must be a finite BCHW tensor")
    batch_size, _channels, height, width = initial_noise.shape
    cond = _prepare_branch(
        conditional, batch_size=batch_size, device=device, dtype=dtype
    )
    null_values = (
        None
        if null is None
        else _prepare_branch(null, batch_size=batch_size, device=device, dtype=dtype)
    )
    image_ids = _img_ids(
        batch_size=batch_size, height=height, width=width, device=device
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS
    )
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    sigmas = [float(value.item()) / 1000.0 for value in timesteps] + [0.0]
    latent = initial_noise.to(device=device, dtype=dtype).clone()
    trace: list[dict[str, Any]] = []

    for index, timestep in enumerate(timesteps):
        sigma = float(timestep.item()) / 1000.0
        packed_noisy = rearrange(
            pad_image(latent),
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=2,
            pw=2,
        )
        timestep_batch = torch.full(
            (batch_size,), sigma, device=device, dtype=dtype
        )
        common = {
            "noi_img": packed_noisy,
            "img_ids": image_ids,
            "timesteps": timestep_batch,
        }
        velocity_cond = dit(**cond, **common)
        if null_values is None:
            velocity = velocity_cond
            velocity_null = None
        else:
            velocity_null = dit(**null_values, **common)
            delta = velocity_cond - velocity_null
            velocity = velocity_null + float(cfg_scale) * delta
        velocity_image = _unpack_velocity(
            velocity, height=height, width=width
        )
        if collect_trace and index in TRACE_STEPS:
            entry: dict[str, Any] = {
                "step_index": index,
                "sigma": sigma,
                "conditional_velocity_rms": _rms_per_sample(velocity_cond),
                "guided_velocity_rms": _rms_per_sample(velocity),
                "latent_rms_before": _rms_per_sample(latent),
            }
            if velocity_null is not None:
                entry["null_velocity_rms"] = _rms_per_sample(velocity_null)
                entry["conditional_minus_null_rms"] = _rms_per_sample(
                    velocity_cond - velocity_null
                )
            trace.append(entry)
        latent = latent - (sigma - sigmas[index + 1]) * velocity_image

    audit = {
        "scheduler": "diffusers.FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
        "num_inference_steps": num_steps,
        "sigma_sha256_float64": array_sha256(sigmas),
        "cfg_enabled": True,
        "cfg_scale": float(cfg_scale),
        "cfg_formula_name": cfg_formula,
        "cfg_formula": "v_null + cfg_scale * (v_cond - v_null)",
        "cfg_null_contract": cfg_null_contract,
        "true_model_batch_size": batch_size,
        "dit_call_count": num_steps * (2 if cfg_scale != 0.0 else 1),
        "initial_noise_sha256": tensor_sha256(initial_noise),
        "final_latent_sha256": tensor_sha256(latent),
        "trace": trace,
    }
    return latent, audit


def repeat_branch(branch: BranchInputs, count: int) -> BranchInputs:
    """Repeat one sample for the batch-equivalence audit."""
    if count <= 0 or branch.ref_img.shape[0] != 1:
        raise ContractError("repeat_branch requires one source sample and count > 0")
    return BranchInputs(
        ref_img=branch.ref_img.repeat(count, 1, 1, 1),
        text=branch.text.repeat(count, 1, 1),
        text_mask=branch.text_mask.repeat(count, 1),
        task=branch.task.repeat(count),
        change_vol=branch.change_vol.repeat(count),
        change_pitch=branch.change_pitch.repeat(count),
    )


def branch_from_prepared(prepared_inputs: Any) -> BranchInputs:
    return BranchInputs(
        ref_img=prepared_inputs.ref_img,
        text=prepared_inputs.text,
        text_mask=prepared_inputs.text_mask,
        task=prepared_inputs.task,
        change_vol=prepared_inputs.change_vol,
        change_pitch=prepared_inputs.change_pitch,
    )
