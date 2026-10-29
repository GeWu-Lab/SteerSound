"""Deterministic spawned DataLoader workers for formal Main C training."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


FORMAL_NUM_WORKERS = 2
FORMAL_PREFETCH_FACTOR = 2
FORMAL_MULTIPROCESSING_CONTEXT = "spawn"


def derive_worker_seed(
    training_seed: int,
    rank: int,
    worker_id: int,
    epoch: int,
) -> int:
    payload = (
        f"{int(training_seed)}\0{int(rank)}\0"
        f"{int(worker_id)}\0{int(epoch)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**32
    )


def derive_sample_seed(worker_seed: int, index: Any) -> int:
    """Make worker prefetch timing irrelevant to a sample's random stream."""
    payload = (
        f"{int(worker_seed)}\0{type(index).__qualname__}\0{index!r}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**32
    )


def seed_python_numpy_torch(seed: int) -> None:
    normalized = int(seed) % (2**32)
    random.seed(normalized)
    np.random.seed(normalized)
    torch.manual_seed(normalized)


class EpochSeededDataset(Dataset):
    """Reseed each persistent worker exactly once per formal epoch."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        training_seed: int,
        rank: int,
        epoch_state: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.training_seed = int(training_seed)
        self.rank = int(rank)
        context = mp.get_context(FORMAL_MULTIPROCESSING_CONTEXT)
        self.epoch_state = (
            epoch_state
            if epoch_state is not None
            else context.Value("q", 0, lock=True)
        )
        self._seeded_context: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        value = int(epoch)
        if value < 0:
            raise ValueError("epoch must be nonnegative")
        with self.epoch_state.get_lock():
            self.epoch_state.value = value
        self._seeded_context = None

    def seed_worker(self, worker_id: int) -> int:
        with self.epoch_state.get_lock():
            epoch = int(self.epoch_state.value)
        seed = derive_worker_seed(
            self.training_seed,
            self.rank,
            int(worker_id),
            epoch,
        )
        seed_python_numpy_torch(seed)
        self._seeded_context = (int(worker_id), epoch)
        return seed

    def __getitem__(self, index):
        info = get_worker_info()
        worker_id = 0 if info is None else int(info.id)
        with self.epoch_state.get_lock():
            epoch = int(self.epoch_state.value)
        if self._seeded_context != (worker_id, epoch):
            self.seed_worker(worker_id)
        # Persistent DataLoader workers may execute stale prefetched requests
        # after the main process advances the epoch.  Reseeding from the
        # four-part worker root plus the sample index makes those requests
        # unable to consume randomness belonging to a later formal sample.
        sample_seed = derive_sample_seed(
            derive_worker_seed(
                self.training_seed,
                self.rank,
                worker_id,
                epoch,
            ),
            index,
        )
        seed_python_numpy_torch(sample_seed)
        return self.dataset[index]

    def __getattr__(self, name: str):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def formal_worker_init_fn(worker_id: int) -> None:
    info = get_worker_info()
    if info is None or not isinstance(info.dataset, EpochSeededDataset):
        raise RuntimeError(
            "formal worker_init_fn requires EpochSeededDataset"
        )
    worker_threads = os.environ.get("EXP3_WORKER_TORCH_THREADS")
    if worker_threads is not None:
        parsed_threads = int(worker_threads)
        if parsed_threads <= 0:
            raise ValueError(
                "EXP3_WORKER_TORCH_THREADS must be a positive integer"
            )
        # Setting this once makes PitchShift's steady-state path avoid
        # per-sample torch.set_num_threads()/restore calls when the configured
        # PitchShift thread count is identical.
        torch.set_num_threads(parsed_threads)
    info.dataset.seed_worker(int(worker_id))


def dataloader_generator(training_seed: int, rank: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(
        derive_worker_seed(training_seed, rank, 0, 0)
    )
    return generator
