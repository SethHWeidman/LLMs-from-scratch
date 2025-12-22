import time
import typing

import tiktoken
import torch
import torch.nn as nn
from torch import cuda

import attention_helpers
import gpt_reference_helpers


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        latent_dim: typing.Optional[int] = None,
        dtype: typing.Optional[torch.dtype] = None,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)

        # Compress K/V down to a latent space, cache that, then up-project for attention
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.W_down = nn.Linear(d_in, self.latent_dim, bias=qkv_bias, dtype=dtype)
        self.W_up_k = nn.Linear(self.latent_dim, d_out, bias=qkv_bias, dtype=dtype)
        self.W_up_v = nn.Linear(self.latent_dim, d_out, bias=qkv_bias, dtype=dtype)

        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("cache_latent", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        queries_all = self.W_query(x)
        latent_new = self.W_down(x)

        if self.cache_latent is None:
            self.cache_latent = latent_new
        else:
            self.cache_latent = torch.cat([self.cache_latent, latent_new], dim=1)
        latent_total = self.cache_latent

        keys_all = self.W_up_k(latent_total)
        values_all = self.W_up_v(latent_total)

        queries = queries_all.view(
            b, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys = keys_all.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values_all.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        num_tokens_q = queries.shape[-2]
        num_tokens_k = keys.shape[-2]
        device = queries.device
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
        self.cache_latent = None
        self.ptr_current_pos = 0


def make_block(cfg: dict[str, typing.Any]) -> nn.Module:
    att = MultiHeadLatentAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        num_heads=cfg["n_heads"],
        dropout=cfg["drop_rate"],
        qkv_bias=cfg["qkv_bias"],
        latent_dim=cfg["latent_dim"],
    )
    ff = attention_helpers.FeedForward(cfg)
    return gpt_reference_helpers.PreNormTransformerBlock(
        att=att, ff=ff, emb_dim=cfg["emb_dim"], drop_rate=cfg["drop_rate"]
    )


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
        "latent_dim": 192,  # Latent dim for MLA compression
    }
    torch.manual_seed(251221)
    model = gpt_reference_helpers.GPTReferenceModel(
        GPT_CONFIG_124M, block_factory=make_block
    )
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

    token_ids = gpt_reference_helpers.generate_text_simple_cached(
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
