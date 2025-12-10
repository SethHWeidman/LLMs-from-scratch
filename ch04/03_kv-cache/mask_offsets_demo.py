"""Small demo script to illustrate the cached causal mask construction.

This focuses on the logic in `gpt_with_kv_cache_reference.MultiHeadAttention` that builds
a causal mask when some keys are already stored in the KV cache.
"""

import torch


def build_cached_causal_mask(num_new_tokens: int, num_keys: int) -> torch.Tensor:
    """Reproduce the cached-branch causal mask for toy sizes."""
    offset = num_keys - num_new_tokens
    row_idx = torch.arange(num_new_tokens).unsqueeze(1)  # (num_new_tokens, 1)
    col_idx = torch.arange(num_keys).unsqueeze(0)  # (1, num_keys)
    return row_idx + offset < col_idx


def main() -> None:
    # Suppose we already cached 4 tokens and now process a chunk of 3 new tokens.
    # That means the global positions of the new queries are 4, 5, and 6, while
    # the keys span positions 0..6 (4 cached, 3 new).
    num_cached = 4
    num_new_tokens = 3
    num_keys = num_cached + num_new_tokens

    mask = build_cached_causal_mask(num_new_tokens=num_new_tokens, num_keys=num_keys)
    offset = num_keys - num_new_tokens

    print("num_cached:", num_cached)
    print("num_new_tokens:", num_new_tokens)
    print("num_keys (cached + new):", num_keys)
    print("offset (cached tokens before new chunk):", offset)
    print("\nrow = query index within new chunk")
    print("col = key index within cached window\n")
    print("Causal mask (1 = masked / future, 0 = allowed / past-or-self):")
    print(mask.to(torch.int))  # print 0/1 instead of False/True


if __name__ == "__main__":
    main()
