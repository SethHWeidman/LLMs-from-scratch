import torch
from torch import testing


def main() -> None:
    # Generate a random tensor of shape (b, num_heads, num_tokens, head_dim)
    # Here: b = 1, num_heads = 2, num_tokens = 3, head_dim = 4.
    torch.manual_seed(251128)
    a = torch.rand(1, 2, 3, 4, dtype=torch.float32)

    # Batched multi-head version: for each (batch, head), this computes
    # (num_tokens, head_dim) @ (head_dim, num_tokens) -> (num_tokens, num_tokens).
    b = a @ a.transpose(2, 3)  # shape: (1, 2, 3, 3)

    # Per-head “manual” version: compute head @ head.T for each head.
    first_head = a[0, 0, :, :]  # (3, 4)
    first_res = first_head @ first_head.T  # (3, 3)

    second_head = a[0, 1, :, :]  # (3, 4)
    second_res = second_head @ second_head.T  # (3, 3)

    # Stack per-head results: (2, 3, 3)
    stacked = torch.stack([first_res, second_res], dim=0)
    # Add the batch dimension back: (1, 2, 3, 3)
    c = stacked.unsqueeze(0)

    # Check both ways of computing give the same result
    testing.assert_close(b, c)
    print("OK: batched result b matches manual per-head result c")


if __name__ == "__main__":
    main()
