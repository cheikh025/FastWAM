#!/usr/bin/env python3
"""Expand a FastWAM checkpoint's action_encoder/head/proprio_encoder for the shared
padded multi-embodiment action interface (see research/RUNBOOK.md "Shared padded
action representation" and research/progress/PROGRESS_0001_padded_multiembodiment_baseline.md).

Grows:
  - mixtures.action.action_encoder: Linear(old_action_dim -> hidden) to Linear(new_action_dim -> hidden)
  - mixtures.action.head:           Linear(hidden -> old_action_dim) to Linear(hidden -> new_action_dim)
  - proprio_encoder:                Linear(old_proprio_dim -> text_dim) to Linear(new_proprio_dim -> text_dim)

preserving the inherited weights EXACTLY at their original index positions.

Initialization of new parameters (deliberate, not incidental):
  - New encoder INPUT columns (action_encoder, proprio_encoder): drawn from a freshly
    constructed nn.Linear of the new full input size, seeded -- i.e. exactly what a
    from-scratch model of the new size would have for those columns. The network needs
    a real, statistically-appropriate init to learn to read the newly-added channels
    (e.g. RoboTwin's second-arm channels for LIBERO's widened action_encoder).
  - New head OUTPUT rows: zero-initialized. An unlearned/new output channel starts at a
    well-defined 0 in the flow-matching target space rather than injecting large random
    noise into predictions the model has never been trained to make.

Usage:
    python research/tools/expand_checkpoint_for_multiembodiment.py \\
        --input checkpoints/exp0019_parent_hf/promoted/0019_spatial_weak_task_oversampling/step_005000.pt \\
        --output checkpoints/exp0019_expanded_k14/step_005000.pt \\
        --new-action-dim 14 --new-proprio-dim 14 --seed 0
"""
import argparse
import os

import torch
import torch.nn as nn


def expand_linear_input(weight: torch.Tensor, bias: torch.Tensor | None, new_in_dim: int, seed: int):
    """Grow a Linear's INPUT dimension: weight (out,in)->(out,new_in). Bias is unchanged
    (bias is per-output-neuron, independent of input width)."""
    old_out, old_in = weight.shape
    if new_in_dim < old_in:
        raise ValueError(f"new_in_dim ({new_in_dim}) must be >= old_in ({old_in})")
    if new_in_dim == old_in:
        return weight.clone(), (bias.clone() if bias is not None else None)

    torch.manual_seed(seed)
    fresh = nn.Linear(new_in_dim, old_out)
    new_weight = weight.new_zeros((old_out, new_in_dim))
    new_weight[:, :old_in] = weight
    new_weight[:, old_in:] = fresh.weight.detach().to(dtype=weight.dtype)[:, old_in:]
    new_bias = bias.clone() if bias is not None else None
    return new_weight, new_bias


def expand_linear_output(weight: torch.Tensor, bias: torch.Tensor | None, new_out_dim: int):
    """Grow a Linear's OUTPUT dimension: weight (out,in)->(new_out,in), bias (out,)->(new_out,).
    New rows/entries are zero-initialized."""
    old_out, in_dim = weight.shape
    if new_out_dim < old_out:
        raise ValueError(f"new_out_dim ({new_out_dim}) must be >= old_out ({old_out})")
    if new_out_dim == old_out:
        return weight.clone(), (bias.clone() if bias is not None else None)

    new_weight = weight.new_zeros((new_out_dim, in_dim))
    new_weight[:old_out, :] = weight
    new_bias = None
    if bias is not None:
        new_bias = bias.new_zeros((new_out_dim,))
        new_bias[:old_out] = bias
    return new_weight, new_bias


def expand_checkpoint(payload: dict, new_action_dim: int, new_proprio_dim: int, seed: int) -> dict:
    payload = dict(payload)  # shallow copy of the top-level payload
    mot = dict(payload["mot"])  # shallow copy so we don't mutate the caller's dict in place

    ae_w, ae_b = mot["mixtures.action.action_encoder.weight"], mot["mixtures.action.action_encoder.bias"]
    old_action_dim = ae_w.shape[1]
    new_ae_w, new_ae_b = expand_linear_input(ae_w, ae_b, new_action_dim, seed=seed)
    mot["mixtures.action.action_encoder.weight"] = new_ae_w
    mot["mixtures.action.action_encoder.bias"] = new_ae_b

    head_w, head_b = mot["mixtures.action.head.weight"], mot["mixtures.action.head.bias"]
    assert head_w.shape[0] == old_action_dim, (head_w.shape, old_action_dim)
    new_head_w, new_head_b = expand_linear_output(head_w, head_b, new_action_dim)
    mot["mixtures.action.head.weight"] = new_head_w
    mot["mixtures.action.head.bias"] = new_head_b

    payload["mot"] = mot

    if "proprio_encoder" in payload:
        proprio = dict(payload["proprio_encoder"])
        pe_w, pe_b = proprio["weight"], proprio["bias"]
        new_pe_w, new_pe_b = expand_linear_input(pe_w, pe_b, new_proprio_dim, seed=seed + 1)
        proprio["weight"] = new_pe_w
        proprio["bias"] = new_pe_b
        payload["proprio_encoder"] = proprio

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--new-action-dim", type=int, required=True)
    parser.add_argument("--new-proprio-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.input}")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)

    old_action_dim = payload["mot"]["mixtures.action.action_encoder.weight"].shape[1]
    old_proprio_dim = payload["proprio_encoder"]["weight"].shape[1] if "proprio_encoder" in payload else None
    print(f"Original: action_dim={old_action_dim}, proprio_dim={old_proprio_dim}")
    print(f"Target:   action_dim={args.new_action_dim}, proprio_dim={args.new_proprio_dim}")

    expanded = expand_checkpoint(payload, args.new_action_dim, args.new_proprio_dim, seed=args.seed)

    new_ae_w = expanded["mot"]["mixtures.action.action_encoder.weight"]
    new_head_w = expanded["mot"]["mixtures.action.head.weight"]
    print(f"New action_encoder.weight shape: {tuple(new_ae_w.shape)}")
    print(f"New head.weight shape: {tuple(new_head_w.shape)}")
    if "proprio_encoder" in expanded:
        print(f"New proprio_encoder.weight shape: {tuple(expanded['proprio_encoder']['weight'].shape)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(expanded, args.output)
    print(f"Saved expanded checkpoint: {args.output}")


if __name__ == "__main__":
    main()
