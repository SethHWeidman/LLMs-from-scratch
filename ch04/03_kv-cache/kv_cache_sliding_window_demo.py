import torch


batch_size = 1
num_heads = 1
window_size = 5
head_dim = 2

# pretend these are our KV caches
cache_k = torch.zeros(batch_size, num_heads, window_size, head_dim)
ptr_cur = 0


def append_chunk(
    cache: torch.Tensor, ptr_cur: int, chunk: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Append a new time chunk into a fixed-size sliding-window cache, dropping oldest
    entries if necessary."""
    num_tokens = chunk.size(-2)
    if ptr_cur + num_tokens > cache.size(-2):
        overflow = ptr_cur + num_tokens - cache.size(-2)
        cache[:, :, :-overflow, :] = cache[:, :, overflow:, :].clone()
        ptr_cur -= overflow
    cache[:, :, ptr_cur : ptr_cur + num_tokens, :] = chunk
    ptr_cur += num_tokens
    return cache, ptr_cur


def show(msg: str, cache: torch.Tensor, ptr: int) -> None:
    print(msg)
    print("ptr_cur:", ptr)
    print(cache[0, 0])  # show time x head_dim
    print("-" * 20)


# first chunk: 2 tokens
chunk1 = torch.tensor([[[[1.0, 1.0], [2.0, 2.0]]]])  # (1,1,2,2)
cache_k, ptr_cur = append_chunk(cache_k, ptr_cur, chunk1)
show("After chunk1", cache_k, ptr_cur)

# second chunk: 3 tokens
chunk2 = torch.tensor([[[[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]]])  # (1,1,3,2)
cache_k, ptr_cur = append_chunk(cache_k, ptr_cur, chunk2)
show("After chunk2", cache_k, ptr_cur)

# third chunk: 2 tokens → causes overflow (window_size=5)
chunk3 = torch.tensor([[[[6.0, 6.0], [7.0, 7.0]]]])  # (1,1,2,2)
cache_k, ptr_cur = append_chunk(cache_k, ptr_cur, chunk3)
show("After chunk3 (with overflow)", cache_k, ptr_cur)
