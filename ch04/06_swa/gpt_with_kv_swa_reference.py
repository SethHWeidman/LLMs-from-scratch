import time
import typing

import tiktoken
import torch
import torch.nn as nn
from torch import cuda

import attention_helpers


class MultiHeadAttentionWithSWA(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        sliding_window_size: int,
        dtype: typing.Optional[torch.dtype] = None,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert sliding_window_size > 0, "sliding_window_size must be positive"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.sliding_window_size = int(sliding_window_size)

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        keys_new = self.W_key(x)
        values_new = self.W_value(x)
        queries = self.W_query(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys_new = keys_new.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        values_new = values_new.view(
            b, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 1. Update the Cache
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=2)

        # 2. Apply Sliding Window (Truncate)
        #
        #    We check the current size (after adding new tokens).
        #
        #    If it exceeds the window, we physically delete the oldest tokens.
        if self.cache_k.size(2) > self.sliding_window_size:
            self.cache_k = self.cache_k[:, :, -self.sliding_window_size :, :]
            self.cache_v = self.cache_v[:, :, -self.sliding_window_size :, :]

        keys, values = self.cache_k, self.cache_v

        attn_scores = queries @ keys.transpose(2, 3)

        num_tokens_q = queries.shape[-2]
        num_tokens_k = keys.shape[-2]
        device = queries.device

        # 3. Calculate Absolute Positions for Masking
        #
        #    We need the absolute index of the tokens in the full text sequence (0, 1, 2,
        #    ... 500) to ensure the mask works correctly.
        #
        # The 'right edge' of our cache is the total number of tokens processed so far.
        current_absolute_end = self.ptr_current_pos + num_tokens

        # The 'left edge' is simply the end minus the current cache size.
        # This gives us the Absolute Position of keys[:, :, 0, :].
        k_start_absolute = current_absolute_end - num_tokens_k

        # The queries start wherever the previous batch ended.
        q_start_absolute = self.ptr_current_pos

        q_positions = torch.arange(
            q_start_absolute,
            q_start_absolute + num_tokens_q,
            device=device,
            dtype=torch.long,
        )

        k_positions = torch.arange(
            k_start_absolute,
            k_start_absolute + num_tokens_k,
            device=device,
            dtype=torch.long,
        )

        # 4. Create and Apply Mask
        diff = q_positions.unsqueeze(-1) - k_positions.unsqueeze(0)

        # (diff < 0) -> Causal Mask (prevent looking at future)
        #
        # (diff >= window) -> Window Mask (prevent looking too far back)
        mask = (diff < 0) | (diff >= self.sliding_window_size)

        self.ptr_current_pos += num_tokens_q

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
        self.att = MultiHeadAttentionWithSWA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            sliding_window_size=cfg["sliding_window_size"],
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
        "sliding_window_size": 256,  # SWA window size W
    }
    torch.manual_seed(251221)
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
