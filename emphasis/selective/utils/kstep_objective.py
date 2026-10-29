"""Auditable fixed-scheduler k-step corrective-flow objective.

The 50-point sigma grid is taken directly from Diffusers' formal inference
scheduler. Rollout states and target construction are detached. After reaching
``z_k``, callers evaluate the trainable model at ``(z_k, sigma_k)`` against the
instruction-defined corrective target.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


NUM_INFERENCE_STEPS = 50
NUM_TRAIN_TIMESTEPS = 1000
MAX_TRAINED_K = 15
GENERATIVE_TASK_IDS = (0, 2)
LOCALIZATION_TASK_IDS = (1, 3, 4, 5)


@dataclass(frozen=True)
class HybridSigmaBatch:
    sigma: torch.Tensor
    used_logit_normal: torch.Tensor
    used_corrective_rollout: torch.Tensor


class HybridTaskSigmaSampler:
    """Route Add/Generate to Logit-Normal and editing rows to sigma=1.

    The four localization-oriented tasks start from k=0 and are subsequently
    rolled out to the fixed training position k.  Add and Generate retain the
    standard flow-matching noise distribution from the original training.
    """

    def __init__(
        self,
        *,
        seed: int,
        rank: int,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> None:
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + int(rank) * 1_000_003 + 311_213)
        self.logit_count_by_task = torch.zeros(6, dtype=torch.int64)
        self.corrective_count_by_task = torch.zeros(6, dtype=torch.int64)

    def sample(
        self,
        tasks: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> HybridSigmaBatch:
        task_ids = tasks.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if task_ids.numel() == 0 or bool(((task_ids < 0) | (task_ids > 5)).any()):
            raise ValueError("task ids must be non-empty and in [0,5]")
        normal = torch.randn(
            tuple(task_ids.shape), generator=self.generator, dtype=torch.float64
        )
        logit_sigma = torch.sigmoid(self.logit_mean + self.logit_std * normal)
        use_logit = torch.zeros_like(task_ids, dtype=torch.bool)
        for task_id in GENERATIVE_TASK_IDS:
            use_logit |= task_ids == task_id
        use_corrective = ~use_logit
        sigma = torch.where(use_logit, logit_sigma, torch.ones_like(logit_sigma))
        for task_id in range(6):
            task_mask = task_ids == task_id
            self.logit_count_by_task[task_id] += (use_logit & task_mask).sum()
            self.corrective_count_by_task[task_id] += (
                use_corrective & task_mask
            ).sum()
        return HybridSigmaBatch(
            sigma=sigma.to(device=device, dtype=dtype),
            used_logit_normal=use_logit.to(device=device),
            used_corrective_rollout=use_corrective.to(device=device),
        )

    def state_dict(self) -> dict:
        return {
            "schema": "powersound_hybrid_task_sigma_v1",
            "generator_state": self.generator.get_state(),
            "logit_count_by_task": self.logit_count_by_task.clone(),
            "corrective_count_by_task": self.corrective_count_by_task.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema") != "powersound_hybrid_task_sigma_v1":
            raise ValueError("hybrid sigma sampler checkpoint schema mismatch")
        self.generator.set_state(state["generator_state"])
        self.logit_count_by_task.copy_(state["logit_count_by_task"])
        self.corrective_count_by_task.copy_(state["corrective_count_by_task"])

    def audit(self) -> dict:
        names = ("add", "remove", "generate", "replace", "volume", "pitch")
        return {
            "schema": "powersound_hybrid_task_sigma_audit_v1",
            "logit_normal_tasks": ["add", "generate"],
            "corrective_rollout_tasks": ["remove", "replace", "volume", "pitch"],
            "route_counts_by_task": {
                name: {
                    "logit_normal": int(self.logit_count_by_task[index]),
                    "corrective_rollout_from_sigma_1": int(
                        self.corrective_count_by_task[index]
                    ),
                }
                for index, name in enumerate(names)
            },
        }


def scheduler_sigmas() -> tuple[float, ...]:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)
    values = tuple(float(item.item()) / 1000.0 for item in scheduler.timesteps)
    values += (0.0,)
    if len(values) != NUM_INFERENCE_STEPS + 1 or values[0] != 1.0:
        raise RuntimeError(f"unexpected scheduler sigma grid: {values[:3]}")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise RuntimeError("scheduler sigmas must be strictly descending")
    return values


SIGMAS = scheduler_sigmas()


def validate_k(k: int) -> None:
    if not 1 <= int(k) <= MAX_TRAINED_K:
        raise ValueError(f"k must be in [1,{MAX_TRAINED_K}], got {k}")


def corrective_velocity_target(
    z_k: torch.Tensor, z_0: torch.Tensor, sigma_k: float | torch.Tensor
) -> torch.Tensor:
    """Return instruction-defined ``(z_k-z_0)/sigma_k`` without a graph."""
    sigma = torch.as_tensor(sigma_k, device=z_k.device, dtype=z_k.dtype)
    if sigma.ndim == 0:
        sigma = sigma.repeat(z_k.shape[0])
    while sigma.ndim < z_k.ndim:
        sigma = sigma.unsqueeze(-1)
    if bool(torch.any(sigma <= 0)):
        raise ValueError("corrective target requires sigma_k > 0")
    return ((z_k.detach() - z_0.detach()) / sigma).detach()


@torch.no_grad()
def detached_rollout_to_k(
    *,
    initial_noise: torch.Tensor,
    k: int,
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigmas: Sequence[float] = SIGMAS,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Roll out k Euler steps and return the detached reached state ``z_k``.

    The model is evaluated at ``z_{j-1}, sigma_{j-1}``, then integrated as
    ``z_j = z_{j-1} - (sigma_{j-1}-sigma_j) * v``. The rollout is graph-free;
    the separate trainable evaluation at ``z_k`` is deliberately not done here.
    """
    validate_k(k)
    if len(sigmas) < k + 1:
        raise ValueError("sigma grid is shorter than requested rollout")
    state = initial_noise.detach()
    trace: list[dict[str, float]] = []
    for step in range(1, k + 1):
        step_input = state.detach()
        sigma_from = float(sigmas[step - 1])
        sigma_to = float(sigmas[step])
        sigma_batch = torch.full(
            (state.shape[0],), sigma_from, device=state.device, dtype=state.dtype
        )
        velocity = velocity_fn(step_input, sigma_batch).detach()
        state = (step_input - (sigma_from - sigma_to) * velocity).detach()
        trace.append({
            "step": step,
            "sigma_from": sigma_from,
            "sigma_to": sigma_to,
            "delta_sigma": sigma_from - sigma_to,
        })
    if state.requires_grad:
        raise RuntimeError("rollout tensors unexpectedly retain gradients")
    return state, trace
