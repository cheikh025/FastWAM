"""Correctness tests for ConcatLeftAlign's per-embodiment column offset (PROGRESS_0003).

Standalone script (not pytest) so it can run with just torch, no model/GPU/data
needed. Run: python research/tools/test_action_state_merger_offset.py

The central claim under test: placing two embodiments' natural-dim channels at
disjoint offsets within a shared K-wide tensor (instead of both left-aligned at 0)
makes a SHARED nn.Linear's weight columns/rows receive gradient from only one
embodiment's batches -- eliminating the exp0001/exp0002 interference mechanism
without separate per-embodiment weight matrices.
"""
import torch
import torch.nn as nn

from fastwam.datasets.lerobot.transforms.action_state_merger import ConcatLeftAlign
from fastwam.utils.losses import masked_action_loss


def _shape_meta(dim):
    return [{"key": "default", "shape": dim}]


def test_default_offset_matches_old_left_aligned_behavior():
    """offset=0 (the default) must reproduce the exact pre-PROGRESS_0003 padding
    behavior -- this is what keeps LIBERO's single-embodiment configs and the
    inherited exp0019 weight positions unchanged."""
    torch.manual_seed(0)
    merger = ConcatLeftAlign(action_target_dim=14, state_target_dim=14)
    merger.set_shape_meta({"action": _shape_meta(7), "state": _shape_meta(8)})
    x = torch.randn(3, 7)
    padded, mask = merger._pad(x, 14, offset=0)
    assert torch.equal(padded[:, :7], x)
    assert torch.all(padded[:, 7:] == 0.0)
    assert torch.equal(mask, torch.cat([torch.zeros(7, dtype=torch.bool), torch.ones(7, dtype=torch.bool)]))
    print("PASS: offset=0 matches the original left-aligned padding behavior")


def test_offset_places_data_at_correct_columns():
    torch.manual_seed(1)
    x = torch.randn(2, 14)  # RoboTwin's natural action dim
    padded, mask = ConcatLeftAlign._pad(x, 21, offset=7)
    assert torch.all(padded[:, :7] == 0.0)
    assert torch.equal(padded[:, 7:21], x)
    expected_mask = torch.zeros(21, dtype=torch.bool)
    expected_mask[:7] = True
    assert torch.equal(mask, expected_mask)
    print("PASS: offset=7 places RoboTwin's 14 channels at columns [7:21]")


def test_forward_backward_roundtrip_with_offset():
    merger_libero = ConcatLeftAlign(action_target_dim=21, state_target_dim=22, action_offset=0, state_offset=0)
    merger_libero.set_shape_meta({"action": _shape_meta(7), "state": _shape_meta(8)})
    merger_robotwin = ConcatLeftAlign(action_target_dim=21, state_target_dim=22, action_offset=7, state_offset=8)
    merger_robotwin.set_shape_meta({"action": _shape_meta(14), "state": _shape_meta(14)})

    torch.manual_seed(2)
    libero_action = torch.randn(4, 7)
    robotwin_action = torch.randn(4, 14)

    libero_batch = merger_libero.forward({"action": {"default": libero_action}, "state": {"default": torch.randn(4, 8)}})
    robotwin_batch = merger_robotwin.forward({"action": {"default": robotwin_action}, "state": {"default": torch.randn(4, 14)}})

    assert libero_batch["action"].shape == (4, 21)
    assert robotwin_batch["action"].shape == (4, 21)
    # Disjoint occupied ranges: LIBERO's real data is in [0:7], zero elsewhere;
    # RoboTwin's real data is in [7:21], zero elsewhere.
    assert torch.all(libero_batch["action"][:, 7:] == 0.0)
    assert torch.all(robotwin_batch["action"][:, :7] == 0.0)

    # backward() must recover the original per-embodiment tensors exactly.
    libero_action_3d = libero_batch["action"].unsqueeze(1)  # [B, T=1, K]
    cropped = ConcatLeftAlign._crop(libero_action_3d, merger_libero.action_meta, offset=0)
    assert torch.allclose(cropped.squeeze(1), libero_action)

    robotwin_action_3d = robotwin_batch["action"].unsqueeze(1)
    cropped_rt = ConcatLeftAlign._crop(robotwin_action_3d, merger_robotwin.action_meta, offset=7)
    assert torch.allclose(cropped_rt.squeeze(1), robotwin_action)
    print("PASS: forward/backward roundtrip recovers exact per-embodiment values under disjoint offsets")


def test_shared_linear_gradient_isolation_across_embodiments():
    """The actual claim exp0003 is built on: with disjoint offsets, a SHARED
    action_encoder/head Linear's weight columns/rows for one embodiment's range
    receive exactly zero gradient from the other embodiment's batches -- reproducing
    the interference-free property of fully separate per-embodiment weights, without
    needing them. This directly demonstrates the fix for the exp0001/exp0002 failure
    mode (RoboTwin's batches previously overwrote weight columns [0:7] that LIBERO's
    batches also used, since both were left-aligned at offset 0)."""
    torch.manual_seed(3)
    K, hidden = 21, 32
    action_encoder = nn.Linear(K, hidden)
    head = nn.Linear(hidden, K)

    def run_batch(action_native, offset, native_dim, K):
        padded, dim_is_pad = ConcatLeftAlign._pad(action_native, K, offset=offset)
        dim_is_pad = dim_is_pad.unsqueeze(0).expand(action_native.shape[0], -1)
        hidden_out = action_encoder(padded)
        pred = head(hidden_out).unsqueeze(1)  # [B, T=1, K]
        target = torch.randn_like(pred)
        loss = masked_action_loss(pred, target, None, dim_is_pad).sum()
        loss.backward()

    action_encoder.zero_grad()
    head.zero_grad()

    # RoboTwin batch: occupies columns [7:21].
    robotwin_action = torch.randn(5, 14, requires_grad=False)
    run_batch(robotwin_action, offset=7, native_dim=14, K=K)

    ae_grad_libero_cols = action_encoder.weight.grad[:, :7].clone()
    head_grad_libero_rows = head.weight.grad[:7, :].clone()
    assert torch.all(ae_grad_libero_cols == 0.0), (
        "action_encoder weight columns [0:7] (LIBERO's range) received nonzero "
        "gradient from a RoboTwin-only batch -- the interference bug is NOT fixed."
    )
    assert torch.all(head_grad_libero_rows == 0.0), (
        "head weight rows [0:7] (LIBERO's range) received nonzero gradient from a "
        "RoboTwin-only batch -- the interference bug is NOT fixed."
    )
    assert torch.any(action_encoder.weight.grad[:, 7:21] != 0.0)
    assert torch.any(head.weight.grad[7:21, :] != 0.0)
    print("PASS: a RoboTwin-only batch leaves LIBERO's shared weight columns/rows at exactly zero gradient")

    action_encoder.zero_grad()
    head.zero_grad()

    # LIBERO batch: occupies columns [0:7].
    libero_action = torch.randn(5, 7, requires_grad=False)
    run_batch(libero_action, offset=0, native_dim=7, K=K)

    ae_grad_robotwin_cols = action_encoder.weight.grad[:, 7:21].clone()
    head_grad_robotwin_rows = head.weight.grad[7:21, :].clone()
    assert torch.all(ae_grad_robotwin_cols == 0.0), (
        "action_encoder weight columns [7:21] (RoboTwin's range) received nonzero "
        "gradient from a LIBERO-only batch -- the interference bug is NOT fixed."
    )
    assert torch.all(head_grad_robotwin_rows == 0.0), (
        "head weight rows [7:21] (RoboTwin's range) received nonzero gradient from a "
        "LIBERO-only batch -- the interference bug is NOT fixed."
    )
    assert torch.any(action_encoder.weight.grad[:, :7] != 0.0)
    assert torch.any(head.weight.grad[:7, :] != 0.0)
    print("PASS: a LIBERO-only batch leaves RoboTwin's shared weight columns/rows at exactly zero gradient")


def test_overlapping_offset_reproduces_old_interference_for_contrast():
    """Sanity check the isolation test's own premise: with the OLD offset=0-for-both
    layout, a RoboTwin-only batch DOES produce nonzero gradient in LIBERO's columns --
    confirming the isolation property above is actually due to the offset, not
    something else (e.g. an unrelated masking bug already preventing all cross-talk)."""
    torch.manual_seed(4)
    K, hidden = 14, 16
    action_encoder = nn.Linear(K, hidden)
    head = nn.Linear(hidden, K)

    robotwin_action = torch.randn(5, 14, requires_grad=False)  # offset=0, fills all of K=14
    padded, dim_is_pad = ConcatLeftAlign._pad(robotwin_action, 14, offset=0)
    dim_is_pad = dim_is_pad.unsqueeze(0).expand(5, -1)
    pred = head(action_encoder(padded)).unsqueeze(1)
    target = torch.randn_like(pred)
    loss = masked_action_loss(pred, target, None, dim_is_pad).sum()
    loss.backward()

    assert torch.any(action_encoder.weight.grad[:, :7] != 0.0), (
        "Expected the old offset=0-for-both layout to reproduce nonzero gradient in "
        "LIBERO's columns from a RoboTwin-only batch (this is the bug PROGRESS_0003 fixes)."
    )
    print("PASS: confirmed the old offset=0-for-both layout does leak gradient into the other embodiment's columns")


if __name__ == "__main__":
    test_default_offset_matches_old_left_aligned_behavior()
    test_offset_places_data_at_correct_columns()
    test_forward_backward_roundtrip_with_offset()
    test_shared_linear_gradient_isolation_across_embodiments()
    test_overlapping_offset_reproduces_old_interference_for_contrast()
    print("\nALL TESTS PASSED")
