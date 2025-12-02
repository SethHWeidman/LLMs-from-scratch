#!/usr/bin/env python3
"""Command-line walkthrough of why PyTorch buffers matter for causal attention."""

# In PyTorch, a "buffer" is a tensor that is part of a module's state (it moves with
# `.to(device)` and is saved in `state_dict`) but is not a learnable parameter (no
# gradients, no optimizer updates). We use `register_buffer` here for the causal mask to
# show this behavior.

import pathlib

import torch
from torch import nn


def _section(title: str) -> None:
    """Print a lightweight section header for readability."""

    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def build_demo_batch() -> torch.Tensor:
    """Create the small batch from the notebook (two identical sequences)."""

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # "Your"
            [0.55, 0.87, 0.66],  # "journey"
            [0.57, 0.85, 0.64],  # "starts"
            [0.22, 0.58, 0.33],  # "with"
            [0.77, 0.25, 0.10],  # "one"
            [0.05, 0.80, 0.55],  # "step"
        ],
        dtype=torch.float32,
    )
    return torch.stack((inputs, inputs), dim=0)


class BaseCausalAttention(nn.Module):
    """Shared causal attention implementation with configurable mask storage."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
        *,
        mask_as_buffer: bool,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        if mask_as_buffer:
            # Buffers automatically move with the module and land in the state_dict.
            self.register_buffer("mask", mask)
        else:
            # Plain tensor attribute—good for showing the baseline problem.
            self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class CausalAttentionWithoutBuffers(BaseCausalAttention):
    """Concrete module whose mask stays a plain tensor attribute."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_as_buffer=False,
        )


class CausalAttentionWithBuffer(BaseCausalAttention):
    """Same module, but the mask is registered as a buffer."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            qkv_bias=qkv_bias,
            mask_as_buffer=True,
        )


def _pick_device() -> tuple[torch.device, bool]:
    """Choose the 'best' available device and indicate if it is actually a GPU."""

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

    if has_cuda:
        device = torch.device("cuda")
    elif has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device, bool(has_cuda or has_mps)


def main() -> None:
    torch.manual_seed(251201)
    demo_batch = build_demo_batch()
    batch_size, seq_len, d_in = demo_batch.shape
    d_out = 2
    print(f"Demo batch shape: {demo_batch.shape} (batch={batch_size}, tokens={seq_len})")

    section("Causal attention without buffers (CPU run)")
    ca_no_buffer = CausalAttentionWithoutBuffers(
        d_in=d_in, d_out=d_out, context_length=seq_len, dropout=0.0
    )
    with torch.no_grad():
        cpu_context = ca_no_buffer(demo_batch)
    print("Context vectors (CPU, no buffers):")
    print(cpu_context)

    section("Moving module to GPU without buffers")
    device, has_gpu = _pick_device()
    print(f"Selected device: {device} (GPU available: {has_gpu})")

    if has_gpu:
        batch_on_device = demo_batch.to(device)
        ca_gpu = CausalAttentionWithoutBuffers(
            d_in=d_in, d_out=d_out, context_length=seq_len, dropout=0.0
        ).to(device)
        print("Attempting forward pass without moving the mask...")
        try:
            with torch.no_grad():
                _ = ca_gpu(batch_on_device)
        except RuntimeError as err:
            print("As expected, the mixed-device computation fails:")
            print(err)

        print("W_query is on:", ca_gpu.W_query.weight.device)
        print("mask is on:", ca_gpu.mask.device)
        print("mask type:", type(ca_gpu.mask))

        print("\nManually moving the mask fixes the issue:")
        ca_gpu.mask = ca_gpu.mask.to(device)
        with torch.no_grad():
            fixed_context = ca_gpu(batch_on_device)
        print(fixed_context)
    else:
        print(
            "No GPU found — skipping the failing forward-pass demo, "
            "but the code above shows the exact steps."
        )

    _section("Using register_buffer to keep the mask with the module")
    ca_with_buffer = CausalAttentionWithBuffer(
        d_in=d_in, d_out=d_out, context_length=seq_len, dropout=0.0
    )
    ca_with_buffer_device = ca_with_buffer.to(device)
    print("W_query (with buffer) is on:", ca_with_buffer_device.W_query.weight.device)
    print("mask (with buffer) is on:", ca_with_buffer_device.mask.device)
    with torch.no_grad():
        buffer_context = ca_with_buffer_device(demo_batch.to(device))
    print("Context vectors with register_buffer (on selected device):")
    print(buffer_context.to("cpu"))

    _section("Buffers also show up inside the state_dict")
    no_buffer_keys = list(ca_no_buffer.state_dict().keys())
    with_buffer_keys = list(ca_with_buffer.state_dict().keys())
    print("State dict keys (no buffer):", no_buffer_keys)
    print("State dict keys (with buffer):", with_buffer_keys)

    checkpoint_path = pathlib.Path(__file__).with_suffix(".pth")

    print("\nTweaking the buffer and saving / loading the model...")
    ca_with_buffer.mask[ca_with_buffer.mask == 1.0] = 2.0
    torch.save(ca_with_buffer.state_dict(), checkpoint_path)
    new_model = CausalAttentionWithBuffer(
        d_in=d_in, d_out=d_out, context_length=seq_len, dropout=0.0
    )
    new_model.load_state_dict(torch.load(checkpoint_path))
    print("Loaded mask (should contain 2.0 values):")
    print(new_model.mask)

    print("\nDoing the same for the module without buffers (mask is not restored):")
    ca_no_buffer.mask[ca_no_buffer.mask == 1.0] = 2.0
    torch.save(ca_no_buffer.state_dict(), checkpoint_path)
    new_model_no_buffer = CausalAttentionWithoutBuffers(
        d_in=d_in, d_out=d_out, context_length=seq_len, dropout=0.0
    )
    new_model_no_buffer.load_state_dict(torch.load(checkpoint_path))
    print("Loaded mask (notice the original 1.0 values):")
    print(new_model_no_buffer.mask)

    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
