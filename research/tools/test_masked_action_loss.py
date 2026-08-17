"""Correctness tests for fastwam.utils.losses.masked_action_loss.

Standalone script (not pytest) so it can run with just torch, no model/GPU/data
needed. Run: python research/tools/test_masked_action_loss.py
"""
import torch

from fastwam.utils.losses import masked_action_loss


def test_no_masks_matches_flat_mean():
    torch.manual_seed(0)
    pred = torch.randn(4, 8, 14)
    target = torch.randn(4, 8, 14)
    got = masked_action_loss(pred, target, None, None)
    expected = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print("PASS: no masks matches flat mean")


def test_all_false_dim_mask_matches_no_mask():
    torch.manual_seed(1)
    B, T, K = 3, 6, 14
    pred = torch.randn(B, T, K)
    target = torch.randn(B, T, K)
    action_is_pad = torch.zeros(B, T, dtype=torch.bool)
    action_dim_is_pad = torch.zeros(B, K, dtype=torch.bool)
    got = masked_action_loss(pred, target, action_is_pad, action_dim_is_pad)
    expected = masked_action_loss(pred, target, None, None)
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    print("PASS: all-False dim mask matches no mask")


def test_padded_channels_do_not_affect_loss_value():
    """The exact correctness property required before training: padding a LIBERO-shaped
    (7-valid-of-14) sample must reproduce the identical loss to running the ORIGINAL
    unwidened 7-channel-only computation, regardless of what garbage values sit in the
    padded channels' pred/target."""
    torch.manual_seed(2)
    B, T, K_valid = 2, 5, 7
    K_full = 14
    pred_valid = torch.randn(B, T, K_valid)
    target_valid = torch.randn(B, T, K_valid)

    # Reference: the original unwidened (K=7) computation with no dim-masking at all.
    reference = torch.nn.functional.mse_loss(pred_valid, target_valid, reduction="none").mean(dim=(1, 2))

    # Padded version: channels 7..13 filled with arbitrary large-scale garbage in both
    # pred AND target (mismatched between pred/target, so if the mask failed to exclude
    # them the loss value would change substantially).
    pred_full = torch.cat([pred_valid, 1000.0 * torch.randn(B, T, K_full - K_valid)], dim=-1)
    target_full = torch.cat([target_valid, -1000.0 * torch.randn(B, T, K_full - K_valid)], dim=-1)
    action_dim_is_pad = torch.zeros(B, K_full, dtype=torch.bool)
    action_dim_is_pad[:, K_valid:] = True

    got = masked_action_loss(pred_full, target_full, None, action_dim_is_pad)
    assert torch.allclose(got, reference, atol=1e-5), (got, reference)
    print("PASS: padded channels (even with huge garbage values) do not affect the loss")


def test_padded_channels_do_not_affect_gradient():
    """Confirms padded channels contribute exactly zero gradient to pred_action."""
    torch.manual_seed(3)
    B, T, K_valid, K_full = 2, 4, 7, 14
    pred_valid = torch.randn(B, T, K_valid, requires_grad=False)
    target = torch.randn(B, T, K_full)
    pred_full = torch.cat([pred_valid, torch.randn(B, T, K_full - K_valid)], dim=-1).requires_grad_(True)
    action_dim_is_pad = torch.zeros(B, K_full, dtype=torch.bool)
    action_dim_is_pad[:, K_valid:] = True

    loss = masked_action_loss(pred_full, target, None, action_dim_is_pad).sum()
    loss.backward()
    grad_valid = pred_full.grad[:, :, :K_valid]
    grad_pad = pred_full.grad[:, :, K_valid:]
    assert torch.all(grad_pad == 0.0), grad_pad
    assert torch.any(grad_valid != 0.0)
    print("PASS: padded channels receive exactly zero gradient")


def test_temporal_and_dim_masks_compose_correctly():
    """Both mask levels active at once: verify against a manual per-sample computation."""
    torch.manual_seed(4)
    B, T, K = 2, 4, 6
    pred = torch.randn(B, T, K)
    target = torch.randn(B, T, K)
    action_is_pad = torch.tensor([[False, False, True, True], [False, True, True, True]])
    action_dim_is_pad = torch.tensor([[False, False, False, True, True, True], [False, False, True, True, True, True]])

    got = masked_action_loss(pred, target, action_is_pad, action_dim_is_pad)

    err = torch.nn.functional.mse_loss(pred, target, reduction="none")  # [B,T,K]
    manual = torch.zeros(B)
    for b in range(B):
        valid_t = (~action_is_pad[b]).nonzero().flatten().tolist()
        valid_k = (~action_dim_is_pad[b]).nonzero().flatten().tolist()
        channel_means = []
        for k in valid_k:
            vals = [err[b, t, k].item() for t in valid_t]
            channel_means.append(sum(vals) / len(vals))
        manual[b] = sum(channel_means) / len(channel_means)

    assert torch.allclose(got, manual, atol=1e-5), (got, manual)
    print("PASS: temporal and per-channel masks compose correctly (matches manual reference)")


if __name__ == "__main__":
    test_no_masks_matches_flat_mean()
    test_all_false_dim_mask_matches_no_mask()
    test_padded_channels_do_not_affect_loss_value()
    test_padded_channels_do_not_affect_gradient()
    test_temporal_and_dim_masks_compose_correctly()
    print("\nALL TESTS PASSED")
