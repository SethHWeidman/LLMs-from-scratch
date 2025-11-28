# Chapter 3: Coding Attention Mechanisms

### Main Chapter Code

- [ch03.ipynb](ch03.ipynb) contains all the code as it appears in the chapter

### Optional Code

- [multihead-attention.ipynb](multihead-attention.ipynb) is a minimal notebook with the main data loading pipeline implemented in this chapter

### Additional Scripts

- `multihead_attention_example.py` is a standalone Python script that mirrors the core
  single-head (causal) and multi-head self-attention examples from the notebook, with
  type hints and explicit shape printouts.

  Example output:

  ```bash
  $ python multihead_attention_example.py
  Batch shape: torch.Size([2, 6, 3]) (batch=2, tokens=6, embedding_dim=3)
  SingleHeadAttention output shape: torch.Size([2, 6, 4]) (batch=2, tokens=6, features=4 = d_out(4))
  MultiHeadAttention output shape: torch.Size([2, 6, 4]) (batch=2, tokens=6, features=4 = d_out(4))
  ```

- `transpose.py` is a small utility script that demonstrates how the batched multi-head
  attention score computation `a @ a.transpose(2, 3)` matches doing `head @ head.T` for
  each head separately.

  Example output:

  ```bash
  $ python transpose.py
  OK: batched result b matches manual per-head result c
  ```
