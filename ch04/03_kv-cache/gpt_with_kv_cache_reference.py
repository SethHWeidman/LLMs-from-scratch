import time
import typing

import tiktoken
import torch
import torch.nn as nn
from torch import cuda

import attention_helpers

# Reference KV-cache implementation adapted from
# https://github.com/rasbt/LLMs-from-scratch


class MultiHeadAttention(attention_helpers.MultiHeadAttentionBase):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
        max_seq_len: typing.Optional[int] = None,
        kv_window_size: typing.Optional[int] = None,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.max_seq_len = max_seq_len or context_length
        # Maximum number of past time steps (tokens) to keep in the KV cache per head;
        # older entries are dropped once this sliding window is full.
        self.kv_window_size = kv_window_size or self.max_seq_len
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_new_tokens, _ = x.shape

        # Use the shared projection helper to obtain per-head Q, K, V in (b, num_heads,
        # num_new_tokens, head_dim) layout.
        queries, keys_new, values_new = self._project_qkv(x)

        # Always use the KV cache path: append the new keys/values into the fixed-size
        # sliding window buffer, dropping the oldest entries if necessary.
        if self.cache_k is None or self.cache_k.size(0) != b:
            self.cache_k = torch.zeros(
                b, self.num_heads, self.kv_window_size, self.head_dim, device=x.device
            )
            self.cache_v = torch.zeros_like(self.cache_k)
            self.ptr_cur = 0  # pointer to next free slot

        # if incoming chunk would overflow discard oldest tokens
        if self.ptr_cur + num_new_tokens > self.kv_window_size:
            overflow = self.ptr_cur + num_new_tokens - self.kv_window_size
            # shift everything left by `overflow` (cheap view-copy)
            # (drop the oldest `overflow` timesteps to make room)
            self.cache_k[:, :, :-overflow, :] = self.cache_k[:, :, overflow:, :].clone()
            self.cache_v[:, :, :-overflow, :] = self.cache_v[:, :, overflow:, :].clone()
            # pointer after shift
            self.ptr_cur -= overflow

        # Write the new keys/values into the cache at positions [ptr_cur, ptr_cur +
        # num_new_tokens) along the time dimension.
        # shapes:
        #   cache_k:  (b, n_heads, kv_window_size, head_dim)
        #   keys_new: (b, n_heads, num_new_tokens, head_dim)
        self.cache_k[:, :, self.ptr_cur : self.ptr_cur + num_new_tokens, :] = keys_new
        self.cache_v[:, :, self.ptr_cur : self.ptr_cur + num_new_tokens, :] = values_new
        self.ptr_cur += num_new_tokens

        keys = self.cache_k[:, :, : self.ptr_cur, :]
        values = self.cache_v[:, :, : self.ptr_cur, :]

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # num_keys is the total number of keys this query can attend to: cached
        # keys from previous steps plus the current `num_tokens` keys.
        num_keys = attn_scores.size(-1)

        if num_new_tokens == num_keys:
            # No cache case: queries and keys span the same sequence, so we can just use
            # the standard upper‑triangular causal mask where positions strictly above
            # the main diagonal are masked out.
            causal_mask = torch.triu(
                torch.ones(num_new_tokens, num_keys, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            # Cached case: we only have `num_tokens` fresh queries, but there are
            # `num_keys` keys in total. The first (num_keys - num_tokens) keys come from
            # earlier timesteps stored in the KV cache, so each query i in this chunk
            # corresponds to a *global* position (offset + i), where offset = number of
            # cached tokens before this chunk.
            #
            # A key at column j is in the "future" of query (offset + i) if j > offset +
            # i, which we encode via the inequality below.
            offset = num_keys - num_new_tokens
            # Row indices i = 0..num_new_tokens-1, shaped (num_new_tokens, 1)
            row_idx = torch.arange(num_new_tokens, device=x.device).unsqueeze(1)
            # Column indices j = 0..num_keys-1, shaped (1, num_keys)
            col_idx = torch.arange(num_keys, device=x.device).unsqueeze(0)
            # True wherever j > offset + i, i.e., where the key would be a "future"
            # position for this query and must be masked out.
            causal_mask = row_idx + offset < col_idx

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_new_tokens, num_heads, head_dim) after transpose
        context = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context = context.contiguous().view(b, num_new_tokens, self.d_out)
        return self.out_proj(context)

    def reset_cache(self) -> None:
        self.cache_k, self.cache_v = None, None


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            kv_window_size=cfg["kv_window_size"],
        )
        self.ff = attention_helpers.FeedForward(cfg)
        self.norm1 = attention_helpers.LayerNorm(cfg["emb_dim"])
        self.norm2 = attention_helpers.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

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

        self.ptr_current_pos = 0

        self.final_norm = attention_helpers.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_ids = torch.arange(
            self.ptr_current_pos,
            self.ptr_current_pos + seq_len,
            device=in_idx.device,
            dtype=torch.long,
        )
        self.ptr_current_pos += seq_len
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            x = blk(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def reset_kv_cache(self) -> None:
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0


def generate_text_simple_cached(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: typing.Optional[int] = None,
) -> torch.Tensor:
    """Generate `max_new_tokens` new tokens autoregressively, starting from the
    prompt token indices `idx` using the model's KV cache."""
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
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
        "kv_window_size": 1024,  # NEW: KV cache window size
    }

    torch.manual_seed(251209)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # disable dropout

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
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
