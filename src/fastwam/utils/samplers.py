from typing import Iterator, List, Optional, Sized

import torch
from torch.utils.data import ConcatDataset, Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class InterleavedEmbodimentSampler(Sampler[int]):
    """Batch-homogeneous, ratio-controlled interleaving across a `ConcatDataset` of
    per-embodiment sub-datasets with incompatible per-sample tensor shapes (e.g. LIBERO's
    2-camera 224x448 video vs. RoboTwin's 3-camera 384x320 video) that cannot be mixed
    *within* one batch via default collation.

    Each yielded batch (a contiguous `batch_size`-sized run of indices, matching how
    `DataLoader(..., sampler=this, shuffle=False)` groups a plain `Sampler`'s output)
    is drawn entirely from ONE sub-dataset, so it collates correctly. The SEQUENCE of
    batches interleaves across sub-datasets according to `ratios`, giving mixing at the
    training-step level instead of the sample level. Sub-datasets are cycled (reshuffled
    on exhaustion) independently, so a much smaller sub-dataset (e.g. a LIBERO replay set)
    is revisited many times per pass over a much larger one (e.g. RoboTwin).

    Shares `ResumableEpochSampler`'s epoch/seed/offset resume contract so it's a drop-in
    replacement wherever a multi-embodiment `ConcatDataset` is used.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        seed: int,
        batch_size: int,
        num_processes: int,
        ratios: Optional[List[float]] = None,
    ):
        if not hasattr(dataset, "cumulative_sizes"):
            raise TypeError(
                "InterleavedEmbodimentSampler requires a torch.utils.data.ConcatDataset "
                "(or a dataset exposing `.cumulative_sizes`)."
            )
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

        starts = [0] + list(dataset.cumulative_sizes[:-1])
        self.ranges = list(zip(starts, dataset.cumulative_sizes))
        for start, end in self.ranges:
            if end - start < 1:
                raise ValueError(f"Every sub-dataset must be non-empty, got range ({start}, {end}).")

        n = len(self.ranges)
        ratios = ratios if ratios is not None else [1.0] * n
        if len(ratios) != n:
            raise ValueError(f"`ratios` must have one entry per sub-dataset ({n}), got {len(ratios)}.")
        total_ratio = sum(ratios)
        if total_ratio <= 0:
            raise ValueError(f"`ratios` must sum to a positive value, got {ratios}.")
        self.ratios = [r / total_ratio for r in ratios]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _shuffled_cycle(self, sub_idx: int, reshuffle_salt: int) -> List[int]:
        start, end = self.ranges[sub_idx]
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset + sub_idx * 1_000_003 + reshuffle_salt)
        perm = torch.randperm(end - start, generator=g)
        return (perm + start).tolist()

    def _batch_owner_sequence(self, num_batches: int) -> List[int]:
        """Proportional round-robin (Bresenham-style) interleaving: at each step, pick
        the sub-dataset with the smallest (batches_emitted_so_far + 1) / ratio, i.e. the
        one "most overdue" relative to its target ratio. Deterministic given `ratios`
        and `num_batches`."""
        n = len(self.ratios)
        counts = [round(r * num_batches) for r in self.ratios]
        counts[0] += num_batches - sum(counts)  # fix rounding drift on the first sub-dataset
        remaining = counts[:]
        progress = [0] * n
        sequence = []
        for _ in range(num_batches):
            candidates = [i for i in range(n) if remaining[i] > 0]
            owner = min(candidates, key=lambda i: (progress[i] + 1) / self.ratios[i])
            sequence.append(owner)
            progress[owner] += 1
            remaining[owner] -= 1
        return sequence

    def __iter__(self) -> Iterator[int]:
        total_len = self.ranges[-1][1]
        num_batches = max(total_len // self.batch_size, 1)
        owner_sequence = self._batch_owner_sequence(num_batches)

        cycles = [self._shuffled_cycle(i, reshuffle_salt=0) for i in range(len(self.ranges))]
        cursors = [0] * len(self.ranges)
        reshuffle_counts = [0] * len(self.ranges)

        indices: List[int] = []
        for owner in owner_sequence:
            for _ in range(self.batch_size):
                if cursors[owner] >= len(cycles[owner]):
                    reshuffle_counts[owner] += 1
                    cycles[owner] = self._shuffled_cycle(owner, reshuffle_salt=reshuffle_counts[owner])
                    cursors[owner] = 0
                indices.append(cycles[owner][cursors[owner]])
                cursors[owner] += 1

        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
