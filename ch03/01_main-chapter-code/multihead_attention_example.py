import torch
from torch import nn

import attention_helpers


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # queries, keys: (b, num_tokens, d_out)
        # keys.transpose(1, 2): (b, d_out, num_tokens)
        # This computes Q K^T with shape (b, num_tokens, num_tokens) so that
        #   attn_scores[b, i, j] = q_i · k_j
        # and each row i corresponds to query token i.
        attn_scores = queries @ keys.transpose(1, 2)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec


class MultiHeadAttention(attention_helpers.MultiHeadAttentionBase):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            dropout=dropout,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries, keys, values = self._project_qkv(x)
        context_vectors = self._compute_context_vectors(queries, keys, values)
        output_vectors = self.out_proj(context_vectors)
        return output_vectors


def run_demo() -> None:
    torch.manual_seed(251128)
    batch = torch.rand((2, 6, 3), dtype=torch.float32)
    batch_size, seq_len, d_in = batch.shape
    context_length = seq_len
    d_out = 4  # output feature dimension
    num_heads = 2  # two heads of size d_out / num_heads = 2

    print(
        f"Batch shape: {batch.shape} "
        f"(batch={batch_size}, tokens={seq_len}, embedding_dim={d_in})"
    )

    ca = SingleHeadAttention(d_in, d_out, context_length, dropout=0.0)
    ca_context = ca(batch)
    print(
        "SingleHeadAttention output shape:"
        f" {ca_context.shape} (batch={ca_context.shape[0]},"
        f" tokens={ca_context.shape[1]},"
        f" features={ca_context.shape[2]} = d_out({d_out}))"
    )

    mha = MultiHeadAttention(
        d_in, d_out, context_length, dropout=0.0, num_heads=num_heads
    )
    mha_context = mha(batch)
    print(
        "MultiHeadAttention output shape:"
        f" {mha_context.shape} (batch={mha_context.shape[0]},"
        f" tokens={mha_context.shape[1]},"
        f" features={mha_context.shape[2]} = d_out({d_out}))"
    )


if __name__ == "__main__":
    run_demo()
