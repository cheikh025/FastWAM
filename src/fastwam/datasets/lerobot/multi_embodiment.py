from typing import Any, List

from torch.utils.data import ConcatDataset

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def build_multi_embodiment_dataset(embodiments: List[Any], **kwargs: Any) -> ConcatDataset:
    """Combine several per-embodiment `RobotVideoDataset`s into one `ConcatDataset`,
    tagged with the per-embodiment sampling ratios `Wan22Trainer._build_loader` reads
    to switch to `InterleavedEmbodimentSampler` (batch-homogeneous, ratio-controlled
    interleaving — see that class's docstring for why sample-level mixing isn't used).

    Each entry in `embodiments` is expected to have `name: str`, `ratio: float`, and
    `dataset`. Hydra's `instantiate()` recursively instantiates nested `_target_`
    configs by default, so by the time this function runs, `entry["dataset"]` is
    already a real, constructed `RobotVideoDataset` instance — NOT a config to
    instantiate again (calling `instantiate()` on it a second time raises
    `InstantiationException: Cannot instantiate config of type RobotVideoDataset`,
    since it's no longer a DictConfig at that point).

    `**kwargs` absorbs any extra kwargs `runtime.build_datasets` might forward to a
    single-dataset target (e.g. `pretrained_norm_stats` when building the val split) —
    intentionally ignored here since each sub-dataset already carries its own
    per-embodiment `pretrained_norm_stats` in its own config.
    """
    if not embodiments:
        raise ValueError("`embodiments` must be a non-empty list.")

    names, ratios, datasets = [], [], []
    for entry in embodiments:
        name = str(entry["name"])
        ratio = float(entry["ratio"])
        ds = entry["dataset"]  # already instantiated by Hydra's recursive instantiate()
        logger.info("Built embodiment dataset '%s': %d samples, ratio=%.3f", name, len(ds), ratio)
        names.append(name)
        ratios.append(ratio)
        datasets.append(ds)

    concat = ConcatDataset(datasets)
    concat.embodiment_names = names
    concat.embodiment_ratios = ratios
    return concat
