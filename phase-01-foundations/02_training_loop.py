"""
Phase 1 - Example 2: The training loop

We teach a model  y = 2x + 1  twice:
  Part A - by hand, computing gradients ourselves
  Part B - with PyTorch doing it for us

Both land on the same answer. Part A shows what Part B hides.

Run:
    uv run python phase-01-foundations/02_training_loop.py

Reference:
    https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

import torch
import torch.nn as nn

# Try changing these! See "Try it yourself" in the README.
LEARNING_RATE = 0.05
STEPS = 200

TRUE_W, TRUE_B = 2.0, 1.0


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def make_data():
    """y = 2x + 1, no noise, so we know the exact right answer."""
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = TRUE_W * x + TRUE_B
    return x, y


def train_by_hand(x, y):
    """The same 5 steps, written out manually."""
    w = torch.tensor(0.0)
    b = torch.tensor(0.0)
    n = len(x)

    for step in range(STEPS):
        # 1. forward: guess
        pred = w * x + b

        # 2. loss: how wrong (mean squared error)
        error = pred - y
        loss = (error**2).mean()

        # 3. backward: which way should w and b move?
        #    These are the derivatives of MSE, worked out by hand.
        grad_w = (2 * error * x).sum() / n
        grad_b = (2 * error).sum() / n

        # 4. step: nudge the numbers downhill
        w = w - LEARNING_RATE * grad_w
        b = b - LEARNING_RATE * grad_b

        # 5. no gradients stored, so nothing to zero here

        if step % 40 == 0 or step == STEPS - 1:
            print(f"  step {step:3d} | loss {loss.item():8.4f} | w {w.item():.3f} | b {b.item():.3f}")

    return w.item(), b.item()


def train_with_pytorch(x, y):
    """Identical logic, but PyTorch computes the gradients."""
    torch.manual_seed(0)

    model = nn.Linear(1, 1)          # one input, one output: w*x + b
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for step in range(STEPS):
        # 1. forward
        pred = model(x)

        # 2. loss
        loss = loss_fn(pred, y)

        # 3. backward - PyTorch works out every gradient for us
        loss.backward()

        # 4. step
        optimizer.step()

        # 5. zero the gradients
        #    Without this they ADD UP across steps and training breaks.
        #    This is the most common beginner bug.
        optimizer.zero_grad()

        if step % 40 == 0 or step == STEPS - 1:
            w = model.weight.item()
            b = model.bias.item()
            print(f"  step {step:3d} | loss {loss.item():8.4f} | w {w:.3f} | b {b:.3f}")

    return model.weight.item(), model.bias.item()


def main():
    x, y = make_data()

    print(f"Learning  y = {TRUE_W}x + {TRUE_B}  from {len(x)} examples")
    print(f"learning rate = {LEARNING_RATE}, steps = {STEPS}")

    section("Part A: manual gradients")
    w1, b1 = train_by_hand(x, y)

    section("Part B: PyTorch autograd")
    w2, b2 = train_with_pytorch(x, y)

    section("Result")
    print(f"  target : w = {TRUE_W:.3f}  b = {TRUE_B:.3f}")
    print(f"  by hand: w = {w1:.3f}  b = {b1:.3f}")
    print(f"  pytorch: w = {w2:.3f}  b = {b2:.3f}")

    if torch.isnan(torch.tensor(w1)):
        print("\n  w is nan -> learning rate is too high. Lower LEARNING_RATE.")
    elif abs(w1 - TRUE_W) > 0.1:
        print("\n  Not converged yet -> raise STEPS or LEARNING_RATE.")
    else:
        print("\n  Both found the answer. Same loop, one just writes itself.")

    print("\nThis exact loop trains a 1B-parameter LLM too.")
    print("Only the model and the data get bigger.")
    print("\nDone. Next: 03_overfitting.py")


if __name__ == "__main__":
    main()
