#!/usr/bin/env python3
"""Precompute T5 text-embedding caches for the multi-embodiment candidate.

`scripts/precompute_text_embeds.py` (the shared, existing tool) reads raw `task`
strings straight from each dataset's `meta/tasks.jsonl` and hashes
`DEFAULT_PROMPT.format(task=<raw task>)` -- it does NOT go through
`FastWAMProcessor.augment_instruction()`, and it writes every computed embedding into
EVERY discovered `text_embedding_cache_dir` (a flat prompt-list x cache-dir-list
design, fine when every dataset node wants the identical prompt set, but not when
different embodiments need different, embodiment-prefixed prompts cached to
different directories).

This candidate's `augment_instruction()` prepends a per-embodiment
`embodiment_description` (Qwen-VLA-style textual conditioning) to the instruction
before it's used as the cache lookup key at runtime
(`RobotVideoDataset._get_cached_text_context`) -- so the shared script's caches would
silently miss. This script replicates `augment_instruction()`'s exact (deterministic,
for our configs: `drop_high_level_prob=1.0`, `use_zh_instruction=False`) string-building
logic per embodiment, and writes each embodiment's caches only to its own
`text_embedding_cache_dir`.

Deliberately does NOT modify scripts/precompute_text_embeds.py, to avoid any risk of
regressing the existing single-embodiment LIBERO/RoboTwin training paths.

Usage:
    python research/tools/precompute_multiembodiment_text_embeds.py \\
        task=multiembodiment_libero_robotwin_3e-5
"""
import hashlib
import json
import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm

from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from fastwam.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
ENC_ID = "wan22ti2v5b"  # must match the literal hardcoded in RobotVideoDataset._get_cached_text_context
DEFAULT_BATCH_SIZE = 16


def _init_distributed():
    """Mirrors scripts/precompute_text_embeds.py::_init_distributed exactly."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _augmented_instruction(raw_task: str, embodiment_description: str | None, use_zh_instruction: bool = False) -> str:
    """Mirrors FastWAMProcessor.augment_instruction() exactly for our configs
    (drop_high_level_prob=1.0 -> always the low-level-only branch, so this is
    deterministic and doesn't need `coarse_task`)."""
    low_level_instruction = raw_task
    if "@" in low_level_instruction:
        zh, eng = low_level_instruction.split("@")
        low_level_instruction = zh if use_zh_instruction else eng
    instruction = low_level_instruction
    if embodiment_description:
        instruction = f"{embodiment_description} {instruction}"
    return instruction


def _read_raw_tasks(dataset_dirs: list[str]) -> list[str]:
    tasks: list[str] = []
    seen = set()
    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                task = str(record["task"])
                if task not in seen:
                    seen.add(task)
                    tasks.append(task)
    return tasks


def _atomic_torch_save(payload: dict, output_path: Path):
    import os
    import uuid

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    is_distributed, rank, world_size, local_rank = _init_distributed()
    setup_logging(log_level=logging.INFO, is_main_process=(rank == 0))
    if is_distributed and rank == 0:
        logger.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d "
            "research/tools/precompute_multiembodiment_text_embeds.py",
            torch.cuda.device_count(),
        )

    model_cfg = cfg.model
    model_id = str(model_cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"))
    tokenizer_model_id = str(model_cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"))
    redirect_common_files = bool(model_cfg.get("redirect_common_files", True))

    embodiments = cfg.data.train.embodiments
    context_lens = {int(e.dataset.context_len) for e in embodiments}
    if len(context_lens) != 1:
        raise ValueError(f"Inconsistent context_len across embodiments: {context_lens}")
    context_len = next(iter(context_lens))

    # Collect (prompt -> set of cache_dirs) so an identical augmented instruction
    # shared across embodiments (unlikely here, but possible) is only encoded once.
    prompt_to_cache_dirs: dict[str, set[Path]] = {}
    for e in embodiments:
        ds_cfg = e.dataset
        dataset_dirs = list(ds_cfg.dataset_dirs)
        cache_dir = Path(str(ds_cfg.text_embedding_cache_dir)).expanduser()
        embodiment_description = ds_cfg.processor.get("embodiment_description", None)
        use_zh_instruction = bool(ds_cfg.processor.get("use_zh_instruction", False))

        raw_tasks = _read_raw_tasks(dataset_dirs)
        logger.info("Embodiment '%s': %d unique raw tasks, embodiment_description=%r", e.name, len(raw_tasks), embodiment_description)
        for raw_task in raw_tasks:
            prompt = DEFAULT_PROMPT.format(
                task=_augmented_instruction(raw_task, embodiment_description, use_zh_instruction)
            )
            prompt_to_cache_dirs.setdefault(prompt, set()).add(cache_dir)

    all_prompts = list(prompt_to_cache_dirs.keys())  # identical across all ranks (deterministic build)
    if rank == 0:
        logger.info("Total unique augmented prompts across all embodiments: %d", len(all_prompts))

    # Skip prompts already fully cached (resumable — safe to re-run/interrupt).
    def _fully_cached(prompt: str) -> bool:
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for cache_dir in prompt_to_cache_dirs[prompt]:
            if not (cache_dir / f"{hashed}.t5_len{context_len}.{ENC_ID}.pt").exists():
                return False
        return True

    pending_prompts = [p for p in all_prompts if not _fully_cached(p)]
    if rank == 0:
        logger.info(
            "Already cached: %d, remaining to encode: %d", len(all_prompts) - len(pending_prompts), len(pending_prompts)
        )

    # Shard the REMAINING work across ranks (not the full list) so a resumed run
    # rebalances rather than always giving rank 0 a stale head-start.
    prompts = pending_prompts[rank::world_size] if is_distributed else pending_prompts

    device = f"cuda:{local_rank}" if (is_distributed and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id, tokenizer_model_id=tokenizer_model_id, redirect_common_files=redirect_common_files
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path, "wan_video_text_encoder", torch_dtype=torch_dtype, device=device
    ).eval()
    tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=context_len, clean="whitespace")

    new_count = 0
    with tqdm(
        total=len(prompts),
        desc=f"Encoding (rank {rank}/{world_size})" if is_distributed else "Encoding augmented prompts",
        unit="prompt",
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
                batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    context_i = context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                    mask_i = mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous()
                    payload = {"context": context_i, "mask": mask_i}

                    for cache_dir in prompt_to_cache_dirs[prompt]:
                        cache_path = cache_dir / f"{hashed}.t5_len{context_len}.{ENC_ID}.pt"
                        if cache_path.exists():
                            continue
                        _atomic_torch_save(payload, cache_path)
                        new_count += 1
                pbar.update(len(batch_prompts))

    logger.info("Rank %d done. new=%d assigned=%d", rank, new_count, len(prompts))
    if is_distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
