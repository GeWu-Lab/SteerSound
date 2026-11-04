from __future__ import annotations
import torch
from torch.utils.data import Sampler
FORMAL_TASKS=("add","remove","generate","replace","volume","pitch")

class BalancedSixTaskDistributedSampler(Sampler[int]):
    """Give every rank exactly five rows per task in every batch (batch=30)."""

    def __init__(
        self,
        frame,
        *,
        rank: int,
        world_size: int,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size % len(FORMAL_TASKS):
            raise ValueError("batch size must be divisible by six tasks")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size = int(batch_size)
        self.rows_per_task_per_batch = batch_size // len(FORMAL_TASKS)
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_task = {
            task: torch.tensor(
                frame.index[frame["task"].astype(str) == task].tolist(),
                dtype=torch.int64,
            )
            for task in FORMAL_TASKS
        }
        counts = {task: len(value) for task, value in self.indices_by_task.items()}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"six-task sampler requires balanced input: {counts}")
        global_rows_per_task_batch = (
            self.world_size * self.rows_per_task_per_batch
        )
        self.batches = min(counts.values()) // global_rows_per_task_batch
        if self.batches <= 0:
            raise RuntimeError("balanced sampler has no complete global batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches * self.batch_size

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch * 10_000_019)
        per_task_rank_batches = {}
        needed = self.batches * self.world_size * self.rows_per_task_per_batch
        for task, indices in self.indices_by_task.items():
            selected = indices[torch.randperm(len(indices), generator=generator)[:needed]]
            selected = selected.reshape(
                self.batches, self.world_size, self.rows_per_task_per_batch
            )
            per_task_rank_batches[task] = selected[:, self.rank, :]
        ordered = []
        for batch_index in range(self.batches):
            batch = torch.cat([
                per_task_rank_batches[task][batch_index] for task in FORMAL_TASKS
            ])
            batch = batch[torch.randperm(len(batch), generator=generator)]
            ordered.extend(int(value) for value in batch.tolist())
        return iter(ordered)
