import torch
from torch import nn


def create_demo_batch() -> torch.Tensor:
    """Create a fixed demo batch of shape (2, 6, 3).

    Values are taken from a torch RNG run with `torch.manual_seed(251128)` and then
    rounded to 2 decimals.
    """

    inputs = torch.tensor(
        [
            [
                [0.19, 0.28, 0.63],
                [0.33, 0.61, 0.36],
                [0.38, 0.99, 0.60],
                [0.56, 0.24, 0.57],
                [0.88, 0.92, 0.81],
                [0.06, 0.97, 0.39],
            ],
            [
                [0.64, 0.56, 0.78],
                [0.02, 0.74, 0.51],
                [0.03, 0.69, 0.76],
                [0.70, 0.42, 0.07],
                [0.64, 0.78, 0.61],
                [0.75, 0.11, 0.73],
            ],
        ],
        dtype=torch.float32,
    )
    return inputs


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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            msg = "d_out must be divisible by num_heads"
            raise ValueError(msg)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # queries, keys: (b, num_heads, num_tokens, head_dim)
        # keys.transpose(2, 3): (b, num_heads, head_dim, num_tokens)
        # Batched Q K^T per (batch, head), yielding
        #   attn_scores: (b, num_heads, num_tokens, num_tokens)
        # with attn_scores[b, h, i, j] = q_i · k_j.
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


def run_demo() -> None:
    torch.manual_seed(251128)
    batch = create_demo_batch()
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
