"""Correctness tests for research/tools/expand_checkpoint_for_multiembodiment.py.

Standalone script (not pytest), only needs torch. Run:
    python research/tools/test_expand_checkpoint.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from expand_checkpoint_for_multiembodiment import expand_checkpoint, expand_linear_input, expand_linear_output


def test_expand_linear_input_preserves_output_on_zero_padded_input():
    """The core property: an expanded encoder fed a zero-padded version of the
    original input must reproduce the ORIGINAL encoder's output exactly, for any
    input (not just a hand-picked one)."""
    torch.manual_seed(0)
    old_in, out_dim = 7, 1024
    orig = nn.Linear(old_in, out_dim)

    new_in = 14
    new_w, new_b = expand_linear_input(orig.weight.data, orig.bias.data, new_in, seed=42)
    expanded = nn.Linear(new_in, out_dim)
    with torch.no_grad():
        expanded.weight.copy_(new_w)
        expanded.bias.copy_(new_b)

    x = torch.randn(5, old_in)
    x_padded = torch.cat([x, torch.zeros(5, new_in - old_in)], dim=-1)

    y_orig = orig(x)
    y_expanded = expanded(x_padded)
    assert torch.allclose(y_orig, y_expanded, atol=1e-5), (y_orig - y_expanded).abs().max()
    print("PASS: expanded input-layer on zero-padded input reproduces original output exactly")


def test_expand_linear_input_new_columns_are_not_dead():
    """New input columns must be non-trivially initialized (not all-zero), so the
    encoder can actually learn to read the new channels."""
    torch.manual_seed(1)
    orig = nn.Linear(7, 1024)
    new_w, _ = expand_linear_input(orig.weight.data, orig.bias.data, 14, seed=7)
    new_cols = new_w[:, 7:]
    assert new_cols.abs().sum() > 0
    assert not torch.allclose(new_cols, torch.zeros_like(new_cols))
    # sanity: statistics roughly match a fresh nn.Linear(14, 1024)'s init scale
    torch.manual_seed(7)
    fresh = nn.Linear(14, 1024)
    assert abs(new_cols.std().item() - fresh.weight.data[:, 7:].std().item()) < 1e-6
    print("PASS: new input columns are non-trivially, reproducibly initialized")


def test_expand_linear_output_preserves_original_rows_and_zeros_new_ones():
    torch.manual_seed(2)
    old_out, in_dim = 7, 1024
    orig = nn.Linear(in_dim, old_out)

    new_out = 14
    new_w, new_b = expand_linear_output(orig.weight.data, orig.bias.data, new_out)

    assert torch.allclose(new_w[:old_out], orig.weight.data)
    assert torch.allclose(new_b[:old_out], orig.bias.data)
    assert torch.all(new_w[old_out:] == 0.0)
    assert torch.all(new_b[old_out:] == 0.0)

    x = torch.randn(5, in_dim)
    y_orig = orig(x)
    expanded = nn.Linear(in_dim, new_out)
    with torch.no_grad():
        expanded.weight.copy_(new_w)
        expanded.bias.copy_(new_b)
    y_expanded = expanded(x)
    assert torch.allclose(y_orig, y_expanded[:, :old_out], atol=1e-5)
    assert torch.allclose(y_expanded[:, old_out:], torch.zeros(5, new_out - old_out), atol=1e-6)
    print("PASS: expanded output-layer preserves original rows exactly, new rows/outputs are zero")


def test_full_checkpoint_expansion_preserves_action_dit_forward():
    """End-to-end: build a tiny synthetic 'exp0019-shaped' checkpoint payload, run it
    through expand_checkpoint, then verify the composed action_encoder->head path
    (with a nonlinearity in between, mimicking the real ActionDiT) reproduces the
    original model's forward pass exactly on the original valid channels, for a
    zero-padded input."""
    torch.manual_seed(3)
    hidden_dim, old_action_dim, old_proprio_dim = 64, 7, 8
    action_encoder = nn.Linear(old_action_dim, hidden_dim)
    head = nn.Linear(hidden_dim, old_action_dim)
    proprio_encoder = nn.Linear(old_proprio_dim, 128)

    payload = {
        "mot": {
            "mixtures.action.action_encoder.weight": action_encoder.weight.data.clone(),
            "mixtures.action.action_encoder.bias": action_encoder.bias.data.clone(),
            "mixtures.action.head.weight": head.weight.data.clone(),
            "mixtures.action.head.bias": head.bias.data.clone(),
            "mixtures.video.some_unrelated_key": torch.randn(3, 3),  # must pass through untouched
        },
        "proprio_encoder": {
            "weight": proprio_encoder.weight.data.clone(),
            "bias": proprio_encoder.bias.data.clone(),
        },
        "step": 5000,
    }

    new_action_dim, new_proprio_dim = 14, 14
    expanded = expand_checkpoint(payload, new_action_dim, new_proprio_dim, seed=99)

    # unrelated keys/metadata pass through untouched
    assert torch.equal(expanded["mot"]["mixtures.video.some_unrelated_key"], payload["mot"]["mixtures.video.some_unrelated_key"])
    assert expanded["step"] == 5000

    action_encoder_new = nn.Linear(new_action_dim, hidden_dim)
    head_new = nn.Linear(hidden_dim, new_action_dim)
    with torch.no_grad():
        action_encoder_new.weight.copy_(expanded["mot"]["mixtures.action.action_encoder.weight"])
        action_encoder_new.bias.copy_(expanded["mot"]["mixtures.action.action_encoder.bias"])
        head_new.weight.copy_(expanded["mot"]["mixtures.action.head.weight"])
        head_new.bias.copy_(expanded["mot"]["mixtures.action.head.bias"])

    def forward(enc, hd, x):
        return hd(torch.relu(enc(x)))

    x_valid = torch.randn(6, old_action_dim)
    x_padded = torch.cat([x_valid, torch.zeros(6, new_action_dim - old_action_dim)], dim=-1)

    y_orig = forward(action_encoder, head, x_valid)
    y_new = forward(action_encoder_new, head_new, x_padded)

    assert torch.allclose(y_orig, y_new[:, :old_action_dim], atol=1e-4), (y_orig - y_new[:, :old_action_dim]).abs().max()
    print("PASS: full checkpoint expansion preserves the original action_dit forward pass exactly")


if __name__ == "__main__":
    test_expand_linear_input_preserves_output_on_zero_padded_input()
    test_expand_linear_input_new_columns_are_not_dead()
    test_expand_linear_output_preserves_original_rows_and_zeros_new_ones()
    test_full_checkpoint_expansion_preserves_action_dit_forward()
    print("\nALL TESTS PASSED")
