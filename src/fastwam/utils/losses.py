from typing import Optional

import torch
import torch.nn.functional as F


def masked_action_loss(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    action_is_pad: Optional[torch.Tensor],
    action_dim_is_pad: Optional[torch.Tensor],
) -> torch.Tensor:
    """Per-sample action MSE with a two-level masked average (Qwen-VLA-style).

    Supports the shared padded multi-embodiment action interface: `pred_action`/
    `target_action` may contain channels padded to a shared width `K` that are not
    valid for every embodiment (e.g. K=14 with only the first 7 valid for LIBERO).

    (1) Per-channel MSE is averaged over valid timesteps only (`action_is_pad`
        marks whole timesteps invalid identically across all channels).
    (2) The result is then averaged uniformly over the `c` *valid* channels only
        (`action_dim_is_pad`), not over the full padded `K`, so an embodiment with
        more valid channels does not implicitly receive more loss weight than one
        with fewer.

    When both masks are `None` (or `action_dim_is_pad` is all-False), this is
    numerically identical to a flat `.mean(dim=(1, 2))` over the unpadded tensor.

    Args:
        pred_action: [B, T, K]
        target_action: [B, T, K]
        action_is_pad: [B, T] bool, True = padded/invalid timestep, or None.
        action_dim_is_pad: [B, K] bool, True = padded/invalid channel, or None.

    Returns:
        [B] per-sample loss.
    """
    action_loss_elem = F.mse_loss(pred_action.float(), target_action.float(), reduction="none")  # [B, T, K]

    if action_is_pad is not None:
        t_valid = (~action_is_pad).to(device=action_loss_elem.device, dtype=action_loss_elem.dtype)  # [B, T]
        t_valid_sum = t_valid.sum(dim=1).clamp(min=1.0)  # [B]
        channel_loss = (action_loss_elem * t_valid.unsqueeze(-1)).sum(dim=1) / t_valid_sum.unsqueeze(-1)  # [B, K]
    else:
        channel_loss = action_loss_elem.mean(dim=1)  # [B, K]

    if action_dim_is_pad is not None:
        c_valid = (~action_dim_is_pad).to(device=channel_loss.device, dtype=channel_loss.dtype)  # [B, K]
        c_valid_sum = c_valid.sum(dim=1).clamp(min=1.0)  # [B]
        action_loss_per_sample = (channel_loss * c_valid).sum(dim=1) / c_valid_sum  # [B]
    else:
        action_loss_per_sample = channel_loss.mean(dim=1)  # [B]

    return action_loss_per_sample
