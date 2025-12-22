import time
import typing

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import cuda

import gpt_reference_helpers


class CausalSelfAttentionKV(nn.Module):
    """Standard self-attention with an append-only KV cache.

    This is local to the MoE reference since the other ch04 reference scripts use
    custom attention variants (GQA/MLA/SWA) instead of plain MHA.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float,
        num_heads: int,
        dtype: typing.Optional[torch.dtype] = None,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

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

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys_new = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        values_new = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
        keys, values = self.cache_k, self.cache_v

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
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class SwiGLUExpert(nn.Module):
    def __init__(
        self, emb_dim: int, hidden_dim: int, dtype: typing.Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(functional.silu(self.fc1(x)) * self.fc2(x))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        dtype: typing.Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert num_experts > 0, "num_experts must be positive"
        assert 0 < top_k <= num_experts, "top_k must satisfy 0 < top_k <= num_experts"

        self.emb_dim = emb_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(emb_dim, num_experts, bias=False, dtype=dtype)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(emb_dim, hidden_dim, dtype=dtype) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shapes:
        # - x:      (batch, seq_len, emb_dim)
        # - scores: (batch, seq_len, num_experts)
        #
        # Routing flow:
        # 1) gate(x) gives logits over experts per token
        # 2) top-k picks the active experts per token
        # 3) softmax over the selected experts gives mixture weights
        # 4) run each selected expert on its assigned tokens
        # 5) scatter-add the weighted expert outputs back per token
        b, seq_len, _ = x.shape

        scores = self.gate(x)  # (b, seq_len, num_experts)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        x_flat = x.reshape(b * seq_len, -1)
        out_flat = torch.zeros(b * seq_len, self.emb_dim, device=x.device, dtype=x.dtype)

        topk_indices_flat = topk_indices.reshape(-1, self.top_k)
        topk_probs_flat = topk_probs.reshape(-1, self.top_k)

        # Only run experts that were selected by at least one token.
        for expert_id_tensor in torch.unique(topk_indices_flat):
            expert_id = int(expert_id_tensor.item())
            expert_mask = topk_indices_flat == expert_id
            if not expert_mask.any():
                continue

            # Find all flattened token indices that route to this expert.
            token_mask = expert_mask.any(dim=-1)
            selected_idx = token_mask.nonzero(as_tuple=False).squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            # Extract only the tokens that want this expert
            expert_input = x_flat.index_select(0, selected_idx)

            # Run the expert on that batch of tokens
            expert_out = self.experts[expert_id](expert_input)

            # Each token has exactly one slot that equals this expert; use that slot to
            # pick the corresponding routing probability.
            mask_selected = expert_mask[selected_idx]
            slot_indices = mask_selected.int().argmax(dim=-1, keepdim=True)
            selected_probs = torch.gather(
                topk_probs_flat.index_select(0, selected_idx), dim=-1, index=slot_indices
            ).squeeze(-1)

            out_flat.index_add_(
                0, selected_idx, expert_out * selected_probs.unsqueeze(-1)
            )

        return out_flat.reshape(b, seq_len, self.emb_dim)


def make_block(cfg: dict[str, typing.Any]) -> nn.Module:
    att = CausalSelfAttentionKV(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        dropout=cfg["drop_rate"],
        num_heads=cfg["n_heads"],
        qkv_bias=cfg["qkv_bias"],
    )
    ff = MoEFeedForward(
        emb_dim=cfg["emb_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_experts=cfg["num_experts"],
        top_k=cfg["num_experts_per_tok"],
    )
    return gpt_reference_helpers.PreNormTransformerBlock(
        att=att, ff=ff, emb_dim=cfg["emb_dim"], drop_rate=cfg["drop_rate"]
    )


def main() -> None:
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "hidden_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "num_experts": 8,
        "num_experts_per_tok": 2,
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
        model=model, idx=encoded_tensor, max_new_tokens=50
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
