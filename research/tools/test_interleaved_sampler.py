"""Correctness tests for fastwam.utils.samplers.InterleavedEmbodimentSampler.

Standalone script, only needs torch. Run:
    python research/tools/test_interleaved_sampler.py
"""
from collections import Counter

import torch
from torch.utils.data import ConcatDataset, Dataset

from fastwam.utils.samplers import InterleavedEmbodimentSampler


class TinyDataset(Dataset):
    def __init__(self, n, tag):
        self.n = n
        self.tag = tag

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.tag, idx)


def _owner_of(index, ranges):
    for i, (start, end) in enumerate(ranges):
        if start <= index < end:
            return i
    raise AssertionError(index)


def test_batches_are_homogeneous():
    ds = ConcatDataset([TinyDataset(20, "libero"), TinyDataset(2000, "robotwin")])
    ranges = [(0, 20), (20, 2020)]
    sampler = InterleavedEmbodimentSampler(ds, seed=0, batch_size=4, num_processes=1, ratios=[1.0, 1.0])
    indices = list(iter(sampler))
    assert len(indices) % 4 == 0
    for start in range(0, len(indices), 4):
        chunk = indices[start : start + 4]
        owners = {_owner_of(i, ranges) for i in chunk}
        assert len(owners) == 1, f"batch at {start} mixes sub-datasets: {chunk} -> owners {owners}"
    print(f"PASS: all {len(indices)//4} batches are homogeneous (single sub-dataset each)")


def test_small_dataset_is_cycled_not_exhausted():
    """LIBERO (20 samples) must be revisited many times to keep pace with RoboTwin (2000)
    at a 1:1 batch ratio, batch_size=4 -> 5 LIBERO indices needed per LIBERO batch cycle
    of 20/4=5 batches before a reshuffle; over ~250 RoboTwin-sized batches this must not
    crash or silently truncate."""
    ds = ConcatDataset([TinyDataset(20, "libero"), TinyDataset(2000, "robotwin")])
    sampler = InterleavedEmbodimentSampler(ds, seed=1, batch_size=4, num_processes=1, ratios=[1.0, 1.0])
    indices = list(iter(sampler))
    libero_indices = [i for i in indices if i < 20]
    # with num_batches = 2020//4 = 505, ~half go to libero -> ~252 batches * 4 = ~1008 draws
    # from a 20-item pool -> each item drawn roughly 1008/20 ≈ 50 times
    counts = Counter(libero_indices)
    assert len(counts) == 20, f"not all 20 libero indices were ever drawn: {sorted(counts)}"
    assert min(counts.values()) >= 30, f"cycling too uneven: min count {min(counts.values())}"
    print(f"PASS: small sub-dataset cycled correctly (each of 20 indices drawn {min(counts.values())}-{max(counts.values())} times)")


def test_ratio_is_approximately_respected():
    ds = ConcatDataset([TinyDataset(500, "libero"), TinyDataset(2000, "robotwin")])
    ranges = [(0, 500), (500, 2500)]
    sampler = InterleavedEmbodimentSampler(ds, seed=2, batch_size=4, num_processes=1, ratios=[1.0, 2.0])
    indices = list(iter(sampler))
    owner_per_batch = [_owner_of(indices[s], ranges) for s in range(0, len(indices), 4)]
    frac_libero = owner_per_batch.count(0) / len(owner_per_batch)
    # target ratio 1:2 -> libero should get ~1/3 of batches
    assert abs(frac_libero - 1 / 3) < 0.02, frac_libero
    print(f"PASS: 1:2 ratio approximately respected (libero batch fraction = {frac_libero:.3f}, target 0.333)")


def test_resume_offset_skips_correctly():
    ds = ConcatDataset([TinyDataset(20, "libero"), TinyDataset(2000, "robotwin")])
    sampler_full = InterleavedEmbodimentSampler(ds, seed=3, batch_size=4, num_processes=1, ratios=[1.0, 1.0])
    full = list(iter(sampler_full))

    sampler_resumed = InterleavedEmbodimentSampler(ds, seed=3, batch_size=4, num_processes=1, ratios=[1.0, 1.0])
    sampler_resumed.set_resume_batch_offset(10)
    resumed = list(iter(sampler_resumed))

    assert resumed == full[10 * 4 :]
    print("PASS: resume_batch_offset correctly skips the same prefix a fresh iterator would produce")


def test_deterministic_given_seed_and_epoch():
    ds = ConcatDataset([TinyDataset(20, "libero"), TinyDataset(2000, "robotwin")])
    s1 = InterleavedEmbodimentSampler(ds, seed=5, batch_size=4, num_processes=1, ratios=[1.0, 1.0])
    s2 = InterleavedEmbodimentSampler(ds, seed=5, batch_size=4, num_processes=1, ratios=[1.0, 1.0])
    assert list(iter(s1)) == list(iter(s2))
    s1.set_epoch(1)
    assert list(iter(s1)) != list(iter(s2))
    print("PASS: deterministic given (seed, epoch); differs across epochs")


if __name__ == "__main__":
    test_batches_are_homogeneous()
    test_small_dataset_is_cycled_not_exhausted()
    test_ratio_is_approximately_respected()
    test_resume_offset_skips_correctly()
    test_deterministic_given_seed_and_epoch()
    print("\nALL TESTS PASSED")
