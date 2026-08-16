"""
Phase 1 - Example 1: Tensors

A tensor is just an array of numbers with a shape. That's it.

Run:
    uv run python phase-01-foundations/01_tensors.py

Reference:
    https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
"""

import torch


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    section("1. Creating tensors")

    scalar = torch.tensor(5.0)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Shape is the single most useful thing to print when debugging.
    print(f"scalar  {scalar.shape}  <- empty shape means a single number")
    print(f"vector  {vector.shape}  <- 3 numbers in a row")
    print(f"matrix  {matrix.shape}  <- 2 rows, 3 columns")

    section("2. The shape LLMs use")

    # Real models work on batches of sequences of vectors.
    batch, seq_len, hidden = 2, 4, 8
    fake_activations = torch.randn(batch, seq_len, hidden)

    print(f"shape: {tuple(fake_activations.shape)}")
    print(f"  {batch} = sentences processed at once (batch)")
    print(f"  {seq_len} = tokens in each sentence (sequence length)")
    print(f"  {hidden} = numbers representing each token (hidden size)")
    print("\nA real 1B model looks the same, just hidden=2048 instead of 8.")

    section("3. Indexing")

    print(f"matrix:\n{matrix}")
    print(f"\nmatrix[0]     -> {matrix[0]}      (first row)")
    print(f"matrix[:, 0]  -> {matrix[:, 0]}      (first column)")
    print(f"matrix[1, 2]  -> {matrix[1, 2]}            (row 1, col 2)")

    section("4. Math happens element-by-element")

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([10.0, 20.0, 30.0])

    print(f"a       = {a}")
    print(f"b       = {b}")
    print(f"a + b   = {a + b}")
    print(f"a * b   = {a * b}   <- element-wise, NOT matrix multiply")
    print(f"a * 2   = {a * 2}   <- scalar applies to every element")

    section("5. Matrix multiplication")

    # This is THE operation inside a neural network layer.
    # Rule: (n, k) @ (k, m) -> (n, m). The inner numbers must match.
    x = torch.randn(2, 3)
    w = torch.randn(3, 4)
    out = x @ w

    print(f"x     {tuple(x.shape)}")
    print(f"w     {tuple(w.shape)}")
    print(f"x @ w {tuple(out.shape)}   <- the 3s cancel out")
    print("\nThis is what a Linear layer does: multiply by weights, add bias.")

    # Shape errors are the #1 beginner bug, so here's what one looks like.
    try:
        _ = x @ torch.randn(5, 4)
    except RuntimeError as e:
        print(f"\nMismatched shapes give you:\n  {str(e).splitlines()[0]}")

    section("6. Broadcasting")

    # Smaller tensors get stretched automatically to fit.
    m = torch.ones(2, 3)
    row = torch.tensor([10.0, 20.0, 30.0])

    print(f"ones(2,3) + [10,20,30]:\n{m + row}")
    print("\nThe row was reused for both rows. That's broadcasting.")

    section("7. Where the tensor lives")

    # CPU always works. MPS is Apple's GPU. CUDA is NVIDIA.
    if torch.backends.mps.is_available():
        device = "mps"
        note = "Apple Silicon GPU"
    elif torch.cuda.is_available():
        device = "cuda"
        note = "NVIDIA GPU"
    else:
        device = "cpu"
        note = "no GPU found"

    print(f"Using: {device}  ({note})")

    moved = vector.to(device)
    print(f"vector is now on: {moved.device}")
    print("\nModels and their inputs must be on the SAME device,")
    print("or you get a 'expected all tensors to be on the same device' error.")

    print("\nDone. Next: 02_training_loop.py")


if __name__ == "__main__":
    main()
