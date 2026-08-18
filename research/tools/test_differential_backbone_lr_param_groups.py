"""Correctness test for the `dit_with_backbone_low_lr` optimizer parameter-group
split in Trainer.__init__ (PROGRESS_0005).

Standalone script (not pytest), no model/GPU/data needed. Run:
    python research/tools/test_differential_backbone_lr_param_groups.py

Exercises the exact id-based set-difference logic used in trainer.py against a toy
module tree shaped like the real one (`model.dit` containing a nested
`mixtures["action"].action_encoder`/`.head`), verifying:
  - backbone_params and projection_params are disjoint (no double-counted parameter)
  - their union equals every parameter in `model.dit`
  - each group is assigned the intended learning rate
"""
import torch
import torch.nn as nn


class ToyActionMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = nn.Linear(21, 64)
        self.head = nn.Linear(64, 21)
        self.other_action_block = nn.Linear(64, 64)  # part of the backbone, not a projection


class ToyMoT(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_block = nn.Linear(32, 32)
        self.mixtures = nn.ModuleDict({"action": ToyActionMixture()})


def split_backbone_and_projection_params(dit: nn.Module):
    """Mirrors Trainer.__init__'s `dit_with_backbone_low_lr` branch exactly."""
    action_mixture = dit.mixtures["action"]
    projection_params = list(action_mixture.action_encoder.parameters()) + list(
        action_mixture.head.parameters()
    )
    projection_param_ids = {id(p) for p in projection_params}
    backbone_params = [p for p in dit.parameters() if id(p) not in projection_param_ids]
    return backbone_params, projection_params


def test_disjoint_and_union_covers_all_dit_params():
    dit = ToyMoT()
    backbone_params, projection_params = split_backbone_and_projection_params(dit)

    backbone_ids = {id(p) for p in backbone_params}
    projection_ids = {id(p) for p in projection_params}
    all_dit_ids = {id(p) for p in dit.parameters()}

    assert backbone_ids.isdisjoint(projection_ids), "backbone/projection params overlap"
    assert backbone_ids | projection_ids == all_dit_ids, "union does not cover all dit params"
    # action_encoder (weight+bias) + head (weight+bias) = 4 tensors
    assert len(projection_params) == 4, len(projection_params)
    # video_block (2) + other_action_block (2) = 4 tensors
    assert len(backbone_params) == 4, len(backbone_params)
    print("PASS: backbone/projection param groups are disjoint and cover all of model.dit")


def test_optimizer_assigns_correct_lr_per_group():
    dit = ToyMoT()
    backbone_params, projection_params = split_backbone_and_projection_params(dit)
    backbone_lr, projection_lr = 3e-6, 3e-5

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": projection_params, "lr": projection_lr},
        ],
        weight_decay=1e-2,
        betas=(0.9, 0.95),
    )

    assert optimizer.param_groups[0]["lr"] == backbone_lr
    assert optimizer.param_groups[1]["lr"] == projection_lr
    assert len(optimizer.param_groups[0]["params"]) == 4
    assert len(optimizer.param_groups[1]["params"]) == 4
    print("PASS: optimizer assigns the correct per-group learning rate")


def test_backbone_group_actually_updates_slower_than_projection_group():
    """End-to-end sanity check: after one optimizer step on identical-magnitude
    gradients, the backbone group's parameters move less than the projection
    group's, proportional to the LR ratio (not just that the LR values are stored
    correctly, but that they actually take effect during optimization)."""
    torch.manual_seed(0)
    dit = ToyMoT()
    backbone_params, projection_params = split_backbone_and_projection_params(dit)
    backbone_lr, projection_lr = 3e-6, 3e-5

    # weight_decay=0 here (unlike the other tests): AdamW's decoupled weight decay
    # adds a term proportional to each parameter's own (differently-initialized)
    # value, which would add noise to the ratio this test checks. Zeroing it isolates
    # the gradient-driven update this test cares about; the per-group LR values
    # themselves are already verified against the real weight_decay in the test above.
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": projection_params, "lr": projection_lr},
        ],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    backbone_before = [p.detach().clone() for p in backbone_params]
    projection_before = [p.detach().clone() for p in projection_params]

    for p in backbone_params + projection_params:
        p.grad = torch.ones_like(p)
    optimizer.step()

    # Mean per-ELEMENT delta, not summed delta -- the two groups have different total
    # parameter-element counts (backbone: video_block + other_action_block; projection:
    # action_encoder + head), so a raw sum would confound element count with the LR
    # effect this test is actually checking.
    backbone_delta = sum((p - b).abs().sum().item() for p, b in zip(backbone_params, backbone_before))
    backbone_numel = sum(p.numel() for p in backbone_params)
    projection_delta = sum(
        (p - b).abs().sum().item() for p, b in zip(projection_params, projection_before)
    )
    projection_numel = sum(p.numel() for p in projection_params)
    ratio = (projection_delta / projection_numel) / (backbone_delta / backbone_numel)
    expected_ratio = projection_lr / backbone_lr
    assert abs(ratio - expected_ratio) / expected_ratio < 0.05, (ratio, expected_ratio)
    print(
        f"PASS: projection group moved {ratio:.2f}x more than backbone group after one step "
        f"(expected ~{expected_ratio:.2f}x from the LR ratio)"
    )


if __name__ == "__main__":
    test_disjoint_and_union_covers_all_dit_params()
    test_optimizer_assigns_correct_lr_per_group()
    test_backbone_group_actually_updates_slower_than_projection_group()
    print("\nALL TESTS PASSED")
