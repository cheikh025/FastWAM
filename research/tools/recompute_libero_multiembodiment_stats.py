#!/usr/bin/env python3
"""Recompute LIBERO's normalization stats for the multi-embodiment candidate.

`fastwam.datasets.lerobot.multi_embodiment.build_multi_embodiment_dataset` builds
each embodiment's RobotVideoDataset independently, but both LIBERO's (freshly
computed, pretrained_norm_stats=None) and RoboTwin's (loaded from
data/robotwin2.0/dataset_stats.json) save to the SAME shared
`{work_dir}/dataset_stats.json` -- RoboTwin's save (constructed second in the
embodiments list) silently overwrites LIBERO's freshly-computed stats before they
were ever durably persisted. Training itself was unaffected (each embodiment's own
processor.set_normalizer_from_stats() call used the correct in-memory stats at
construction time) but the stats needed for faithful evaluation were lost.

This script recomputes LIBERO's stats via the exact same deterministic pass
(RobotVideoDataset.get_dataset_stats -> BaseLerobotDataset.get_dataset_stats, a full,
non-random iteration over every episode) as was used during exp0001 training, so
evaluation uses numerically-identical normalization to what the model was actually
trained under.

Usage:
    python research/tools/recompute_libero_multiembodiment_stats.py \\
        --output runs/reweighted_multiembodiment/exp0001_padded_baseline_v2/libero_dataset_stats.json
"""
import argparse

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import os

from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["task=libero_uncond_2cam224_multiembodiment_eval"])

    ds_cfg = cfg.data.train
    processor = instantiate(ds_cfg.processor)
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from omegaconf import OmegaConf

    lerobot_dataset = BaseLerobotDataset(
        dataset_dirs=list(ds_cfg.dataset_dirs),
        shape_meta=OmegaConf.to_container(ds_cfg.shape_meta, resolve=True),
        obs_size=int(ds_cfg.num_frames),
        action_size=int(ds_cfg.num_frames) - 1,
        val_set_proportion=float(ds_cfg.val_set_proportion),
        is_training_set=True,
        global_sample_stride=int(ds_cfg.global_sample_stride),
    )
    stats = lerobot_dataset.get_dataset_stats(processor)
    save_dataset_stats_to_json(stats, args.output)
    print(f"Saved recomputed LIBERO stats: {args.output}")


if __name__ == "__main__":
    main()
