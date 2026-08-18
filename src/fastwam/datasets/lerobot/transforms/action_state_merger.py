from typing import Dict

import torch
from torch.nn.functional import pad


class ConcatLeftAlign:
    """Pad each embodiment's natural action/state vector into a shared K-wide tensor.

    `action_offset`/`state_offset` place the natural-dim data at a fixed column
    offset within the K-wide tensor instead of always at column 0. This matters for
    multi-embodiment training: if two embodiments are both left-aligned at offset 0
    (the default), and one embodiment's natural dim is >= the other's, its real
    (non-padded) forward values occupy the *same* shared-weight columns the smaller
    embodiment's channels depend on -- so its gradient directly overwrites those
    weight positions regardless of loss masking (see
    research/progress/PROGRESS_0002_frozen_backbone_warmup.md Section 10 and
    PROGRESS_0003). Assigning each embodiment a disjoint `[offset, offset+dim)`
    column range makes every other column's input provably zero for that
    embodiment's batches, so its gradient at those columns is exactly zero and the
    weight-sharing becomes interference-free without needing separate per-embodiment
    weight matrices.
    """

    def __init__(
        self,
        action_target_dim: int | None = None,
        state_target_dim: int | None = None,
        action_offset: int = 0,
        state_offset: int = 0,
    ):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim
        self.action_offset = action_offset
        self.state_offset = state_offset

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        if "action" in batch:
            batch["action"] = self._concat(batch["action"], self.action_meta)
            batch["action"], batch["action_dim_is_pad"] = self._pad(
                batch["action"], self.action_target_dim, self.action_offset
            )

        batch["state"] = self._concat(batch["state"], self.state_meta)
        batch["state"], batch["state_dim_is_pad"] = self._pad(
            batch["state"], self.state_target_dim, self.state_offset
        )

        return batch

    def backward(self, batch):
        if self.state_target_dim is not None:
            assert batch["state"].shape[-1] == self.state_target_dim
        batch["state"] = self._crop(batch["state"], self.state_meta, self.state_offset)
        batch["state"] = self._split(batch["state"], self.state_meta)

        if self.action_target_dim is not None:
            assert batch["action"].shape[-1] == self.action_target_dim
        batch["action"] = self._crop(batch["action"], self.action_meta, self.action_offset)
        batch["action"] = self._split(batch["action"], self.action_meta)

        return batch

    @staticmethod
    def _pad(x: torch.Tensor, dim: int, offset: int = 0):
        if dim is None:
            dim = x.shape[-1]

        assert x.ndim == 2
        native_dim = x.shape[-1]
        assert offset >= 0 and offset + native_dim <= dim, (
            f"offset ({offset}) + native_dim ({native_dim}) must be <= target dim ({dim})"
        )
        left_pad = offset
        right_pad = dim - native_dim - offset
        x_padded = pad(x, (left_pad, right_pad))
        mask = torch.zeros_like(x[0]).bool()
        mask = pad(mask, (left_pad, right_pad), value=True)
        return x_padded, mask

    @staticmethod
    def _crop(x: torch.Tensor, meta: int, offset: int = 0):
        assert x.ndim == 3
        dim = sum([m["shape"] for m in meta])
        x = x[:, :, offset:offset + dim]
        return x
    
    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        x = torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        assert x.ndim == 3
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            y[key] = x[:, :, idx: idx + dim]
            idx += dim

        return y