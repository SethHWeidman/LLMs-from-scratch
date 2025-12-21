import time
import typing

import tiktoken
import torch
import torch.nn as nn
from torch import cuda

import attention_helpers


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        num_kv_groups: int,
        dtype: typing.Optional[torch.dtype] = None,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype
        )
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values_new = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        # Always use the KV cache path: append new keys/values along the time
        # dimension and reuse all cached entries.
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
        keys_base, values_base = self.cache_k, self.cache_v

        keys = keys_base.repeat_interleave(self.group_size, dim=1)
        values = values_base.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)

        num_tokens_q = queries.shape[-2]
        num_tokens_k = keys.shape[-2]
        device = queries.device

        # Causal Masking with a KV Cache
        # ------------------------------
        # To mask correctly, we must align the Query and Key tensors using their
        # "Absolute Positions" in the full text sequence.
        #
        # 1. Queries: The new tokens start at `self.ptr_current_pos`.
        #
        # 2. Keys: In this infinite-cache implementation, the cache always begins
        #    at Absolute Position 0.
        #
        #    (Note: If we were using a sliding window, we would calculate the start
        #    position as `total_tokens_processed - current_cache_size`).
        q_positions = torch.arange(
            self.ptr_current_pos,
            self.ptr_current_pos + num_tokens_q,
            device=device,
            dtype=torch.long,
        )
        self.ptr_current_pos += num_tokens_q
        k_positions = torch.arange(num_tokens_k, device=device, dtype=torch.long)
        mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

    def reset_cache(self) -> None:
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = attention_helpers.FeedForward(cfg)
        self.norm1 = attention_helpers.LayerNorm(cfg["emb_dim"])
        self.norm2 = attention_helpers.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.current_pos = 0

        self.final_norm = attention_helpers.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_ids = torch.arange(
            self.current_pos,
            self.current_pos + seq_len,
            device=in_idx.device,
            dtype=torch.long,
        )
        self.current_pos += seq_len
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            x = blk(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self) -> None:
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0


def generate_text_simple_cached(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: typing.Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        model.reset_kv_cache()
        logits = model(idx[:, -ctx_len:])

        for _ in range(max_new_tokens):
            next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_idx], dim=1)
            logits = model(next_idx)

    return idx


def main() -> None:
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
        "n_kv_groups": 2,  # Number of key/value groups
    }
    torch.manual_seed(251209)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if cuda.is_available():
        cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model, idx=encoded_tensor, max_new_tokens=200
    )

    if cuda.is_available():
        cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if cuda.is_available():
        max_mem_bytes = cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024**3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()
