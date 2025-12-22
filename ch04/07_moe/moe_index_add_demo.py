"""
Demo: The Mechanics of Mixture-of-Experts (MoE).

This script isolates the data movement (Gather -> Process -> Scatter) used to parallelize
expert computation.

Key Concepts:
1. Routing:  Deciding which tokens go to which expert.
2. Gather:   Grouping tokens that need the SAME expert into a contiguous batch.
3. Process:  Running the expert (FFN) on that batch.
4. Scatter:  Putting the results back into the original token sequence order.
"""

import torch
import torch.nn.functional as F


def main() -> None:
    # --- 0. SETUP ---
    # We have 3 tokens (Batch=1, Seq=3), embedding dimension of 4.
    # Token 0: "The", Token 1: "Cat", Token 2: "Sat"
    num_tokens = 3
    emb_dim = 4

    # Input activations (from the Attention layer).
    # [1,1..] tracks Token 0, [2,2..] tracks Token 1, etc.
    inputs = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
    )

    # The accumulator for the final output. Starts as zeros.
    # We will add expert contributions into this tensor.
    final_output = torch.zeros(num_tokens, emb_dim)

    print(f"--- Initial State ---\nInputs:\n{inputs}\n")

    # --- 1. MOCK ROUTING (The "Gate") ---
    # In a real model, this comes from: scores = self.gate(inputs)
    # Here, we simulate that Token 0 needs Expert A & B, Token 1 needs A, Token 2 needs
    # B.

    # Expert A is selected by Token 0 and Token 1
    indices_a = torch.tensor([0, 1])
    # The router assigned these probabilities (weights) to Expert A for these tokens.
    # e.g., Token 0 relies 60% on Expert A.
    weights_a = torch.tensor([0.6, 1.0]).unsqueeze(-1)

    # Expert B is selected by Token 0 and Token 2
    indices_b = torch.tensor([0, 2])
    # The router assigned these probabilities to Expert B.
    # e.g., Token 0 relies 40% on Expert B.
    weights_b = torch.tensor([0.4, 1.0]).unsqueeze(-1)

    # --- 2. EXPERT A EXECUTION ---
    print(f"--- Processing Expert A ---")

    # GATHER: Collect only the tokens relevant to Expert A
    # We copy tokens 0 and 1 from 'inputs' to a temporary buffer.
    expert_a_input = inputs.index_select(0, indices_a)

    # PROCESS: Run the expert's Feed-Forward Network (Simulated here by multiplying by
    # 10)
    expert_a_output = expert_a_input * 10

    # SCATTER: Accumulate results back to the global output
    # We multiply the output by the router's weight (0.6 for token 0, 1.0 for token 1)
    # Then we add it to the specific slots in final_output.
    weighted_out_a = expert_a_output * weights_a
    final_output.index_add_(0, indices_a, weighted_out_a)

    print(f"Accumulator after Expert A:\n{final_output}\n")

    # --- 3. EXPERT B EXECUTION ---
    print(f"--- Processing Expert B ---")

    # GATHER: Collect tokens relevant to Expert B (Token 0 and Token 2)
    expert_b_input = inputs.index_select(0, indices_b)

    # PROCESS: Run Expert B (Simulated by multiplying by 100)
    expert_b_output = expert_b_input * 100

    # SCATTER: Accumulate results back to global output
    # Note: Token 0 is processed again here. Since we use index_add_, this value (20 *
    # 0.4) is ADDED to the existing value from Expert A.
    weighted_out_b = expert_b_output * weights_b
    final_output.index_add_(0, indices_b, weighted_out_b)

    print(f"Accumulator after Expert B:\n{final_output}\n")

    # --- 4. FINAL ANALYSIS ---
    print(f"--- Final Result ---")
    print("Token 0 (Mixed):", final_output[0])
    print("  -> (1.0 * 10 * 0.6) + (1.0 * 100 * 0.4) = 6.0 + 40.0 = 46.0")
    print("Token 1 (Expert A only):", final_output[1])
    print("Token 2 (Expert B only):", final_output[2])


if __name__ == "__main__":
    main()
