"""Task-routed sigma sampling with isolated, reproducible RNG state."""
from __future__ import annotations

from dataclasses import dataclass

import torch


TASK_NAMES = ("add", "remove", "generate", "replace", "volume", "pitch")
GENERATIVE_TASK_IDS = (0, 2)
LOCALIZATION_TASK_IDS = (1, 3, 4, 5)
ROUTED_ARM = "routed"
FRONT030_ARM = "front030"
FRONT005_ARM = "front005"
ACTIVE_ARMS = ("logit", ROUTED_ARM, FRONT030_ARM, FRONT005_ARM)
ARM_TO_BETA = {
    "logit": None,
    ROUTED_ARM: None,
    FRONT030_ARM: None,
    FRONT005_ARM: None,
}
ARM_TO_HIGH_UNIFORM_PROBABILITY = {
    "logit": 0.0,
    ROUTED_ARM: 1.0,
    FRONT030_ARM: 1.0,
    FRONT005_ARM: 1.0,
}


def uniform_interval_by_task(arm: str) -> dict[str, list[float] | None] | None:
    """Return the single authoritative task-to-sigma contract for an arm."""

    if arm == "logit":
        return None
    lower_by_task = {
        ROUTED_ARM: {"remove": 0.97, "replace": 0.71, "volume": 0.71, "pitch": 0.71},
        FRONT030_ARM: {"remove": 0.71, "replace": 0.71, "volume": 0.71, "pitch": 0.71},
        FRONT005_ARM: {"remove": 0.97, "replace": 0.97, "volume": 0.97, "pitch": 0.97},
    }
    if arm not in lower_by_task:
        raise ValueError(f"unknown sigma arm {arm!r}; expected {ACTIVE_ARMS}")
    return {
        name: (None if name in ("add", "generate") else [lower_by_task[arm][name], 1.0])
        for name in TASK_NAMES
    }


@dataclass(frozen=True)
class SigmaBatch:
    sigma: torch.Tensor
    used_logit_normal: torch.Tensor
    used_beta: torch.Tensor
    used_high_uniform: torch.Tensor


class TaskAwareSigmaSampler:
    """Sample sigma per row according to task and experiment arm.

    Both a normal candidate and a uniform candidate are consumed for every
    sample in every arm.  Selecting another arm therefore does not change RNG
    consumption or the independent FM-noise stream.
    """

    def __init__(
        self,
        arm: str,
        *,
        seed: int,
        rank: int,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> None:
        if arm not in ARM_TO_BETA:
            raise ValueError(f"unknown sigma arm {arm!r}; expected {tuple(ARM_TO_BETA)}")
        self.arm = arm
        self.beta_k = ARM_TO_BETA[arm]
        self.high_uniform_probability = ARM_TO_HIGH_UNIFORM_PROBABILITY[arm]
        self.logit_normal_probability = (
            1.0 - self.high_uniform_probability
            if self.high_uniform_probability
            else (0.0 if self.beta_k is not None else 1.0)
        )
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) + int(rank) * 1_000_003 + 74_021)
        # Keep mixture routing on an independent stream.  Adding the new arms
        # must not perturb the pre-existing normal/uniform candidate stream or
        # the separately seeded FM-noise stream.
        self.mixture_generator = torch.Generator(device="cpu")
        self.mixture_generator.manual_seed(
            int(seed) + int(rank) * 1_000_003 + 193_117
        )
        self.count_by_task = torch.zeros(6, dtype=torch.int64)
        self.sum_by_task = torch.zeros(6, dtype=torch.float64)
        # Fifty bins keep the prefix-denoising interpretation auditable.
        self.histogram_by_task = torch.zeros(6, 50, dtype=torch.int64)
        self.logit_count_by_task = torch.zeros(6, dtype=torch.int64)
        self.beta_count_by_task = torch.zeros(6, dtype=torch.int64)
        self.high_uniform_count_by_task = torch.zeros(6, dtype=torch.int64)

    def sample(
        self,
        tasks: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> SigmaBatch:
        task_ids = tasks.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if task_ids.numel() == 0 or bool(((task_ids < 0) | (task_ids > 5)).any()):
            raise ValueError(f"task ids must be non-empty and in [0,5], got {task_ids.tolist()}")
        shape = tuple(task_ids.shape)
        normal = torch.randn(shape, generator=self.generator, dtype=torch.float64)
        uniform = torch.rand(shape, generator=self.generator, dtype=torch.float64)
        mixture_selector = torch.rand(
            shape, generator=self.mixture_generator, dtype=torch.float64
        )
        logit_sigma = torch.sigmoid(self.logit_mean + self.logit_std * normal)
        beta_sigma = (
            logit_sigma
            if self.beta_k is None
            else uniform.pow(1.0 / float(self.beta_k))
        )
        localization = torch.zeros_like(task_ids, dtype=torch.bool)
        for task_id in LOCALIZATION_TASK_IDS:
            localization |= task_ids == task_id
        use_beta = localization & (self.beta_k is not None)
        if self.arm in (ROUTED_ARM, FRONT030_ARM, FRONT005_ARM):
            use_high_uniform = localization
        else:
            use_high_uniform = (
                localization
                & (self.high_uniform_probability > 0.0)
                & (mixture_selector < self.high_uniform_probability)
            )
        use_logit_normal = ~(use_beta | use_high_uniform)
        sigma_cpu = torch.where(use_beta, beta_sigma, logit_sigma)
        intervals = uniform_interval_by_task(self.arm)
        if intervals is not None:
            for task_id in LOCALIZATION_TASK_IDS:
                lower = intervals[TASK_NAMES[task_id]][0]
                task_uniform = lower + (1.0 - lower) * uniform
                sigma_cpu = torch.where(task_ids == task_id, task_uniform, sigma_cpu)

        for task_id in range(6):
            selected = sigma_cpu[task_ids == task_id]
            if selected.numel() == 0:
                continue
            self.count_by_task[task_id] += selected.numel()
            self.sum_by_task[task_id] += selected.sum()
            bins = torch.clamp((selected * 50).to(torch.int64), 0, 49)
            self.histogram_by_task[task_id] += torch.bincount(bins, minlength=50)
            task_mask = task_ids == task_id
            self.logit_count_by_task[task_id] += use_logit_normal[task_mask].sum()
            self.beta_count_by_task[task_id] += use_beta[task_mask].sum()
            self.high_uniform_count_by_task[task_id] += (
                use_high_uniform[task_mask].sum()
            )

        return SigmaBatch(
            sigma=sigma_cpu.to(device=device, dtype=dtype),
            used_logit_normal=use_logit_normal.to(device=device),
            used_beta=use_beta.to(device=device),
            used_high_uniform=use_high_uniform.to(device=device),
        )

    def state_dict(self) -> dict:
        return {
            "arm": self.arm,
            "generator_state": self.generator.get_state(),
            "mixture_generator_state": self.mixture_generator.get_state(),
            "count_by_task": self.count_by_task.clone(),
            "sum_by_task": self.sum_by_task.clone(),
            "histogram_by_task": self.histogram_by_task.clone(),
            "logit_count_by_task": self.logit_count_by_task.clone(),
            "beta_count_by_task": self.beta_count_by_task.clone(),
            "high_uniform_count_by_task": self.high_uniform_count_by_task.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["arm"] != self.arm:
            raise ValueError(f"sigma arm mismatch: {state['arm']} != {self.arm}")
        self.generator.set_state(state["generator_state"])
        if "mixture_generator_state" in state:
            self.mixture_generator.set_state(state["mixture_generator_state"])
        elif self.high_uniform_probability:
            raise ValueError("mixture checkpoint is missing mixture_generator_state")
        self.count_by_task.copy_(state["count_by_task"])
        self.sum_by_task.copy_(state["sum_by_task"])
        self.histogram_by_task.copy_(state["histogram_by_task"])
        for name in (
            "logit_count_by_task",
            "beta_count_by_task",
            "high_uniform_count_by_task",
        ):
            if name in state:
                getattr(self, name).copy_(state[name])

    def audit(self) -> dict:
        counts = self.count_by_task.tolist()
        sums = self.sum_by_task.tolist()
        return {
            "arm": self.arm,
            "beta_k": self.beta_k,
            "logit_normal_probability": (
                None if self.arm in (ROUTED_ARM, FRONT030_ARM, FRONT005_ARM)
                else self.logit_normal_probability
            ),
            "high_uniform_probability": (
                None if self.arm in (ROUTED_ARM, FRONT030_ARM, FRONT005_ARM)
                else self.high_uniform_probability
            ),
            "high_uniform_interval": None,
            "high_uniform_interval_by_task": uniform_interval_by_task(self.arm),
            "task_counts": dict(zip(TASK_NAMES, counts)),
            "task_sigma_means": {
                name: (sums[index] / counts[index] if counts[index] else None)
                for index, name in enumerate(TASK_NAMES)
            },
            "histogram_edges": [index / 50 for index in range(51)],
            "histogram_by_task": {
                name: self.histogram_by_task[index].tolist()
                for index, name in enumerate(TASK_NAMES)
            },
            "route_counts_by_task": {
                name: {
                    "logit_normal": int(self.logit_count_by_task[index]),
                    "beta": int(self.beta_count_by_task[index]),
                    "high_uniform": int(
                        self.high_uniform_count_by_task[index]
                    ),
                    "uniform_0p71_1p0": int(
                        self.high_uniform_count_by_task[index]
                        if uniform_interval_by_task(self.arm)
                        and uniform_interval_by_task(self.arm)[name] == [0.71, 1.0]
                        else 0
                    ),
                    "uniform_0p97_1p0": int(
                        self.high_uniform_count_by_task[index]
                        if uniform_interval_by_task(self.arm)
                        and uniform_interval_by_task(self.arm)[name] == [0.97, 1.0]
                        else 0
                    ),
                }
                for index, name in enumerate(TASK_NAMES)
            },
        }
