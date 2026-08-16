"""
Phase 1 - Example 3: Overfitting

Overfitting = the model memorizes your training data instead of learning
the pattern. You can only SEE it if you hold out data it never trains on.

We give a big model very little noisy data, which makes overfitting obvious.

Run:
    uv run python phase-01-foundations/03_overfitting.py

Reference:
    https://developers.google.com/machine-learning/crash-course/overfitting/overfitting
"""

import torch
import torch.nn as nn

# Try changing these! See "Try it yourself" in the README.
N_TRAIN = 12        # tiny training set -> easy to memorize
HIDDEN_SIZE = 64    # big model for such a small task
EPOCHS = 3000
NOISE = 0.3


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def make_data():
    """A wavy curve plus random noise. The noise is the trap."""
    torch.manual_seed(0)

    x_train = torch.linspace(-3, 3, N_TRAIN).unsqueeze(1)
    y_train = torch.sin(x_train) + torch.randn_like(x_train) * NOISE

    # Validation data: same underlying curve, different points, different noise.
    x_val = torch.linspace(-3, 3, 50).unsqueeze(1)
    y_val = torch.sin(x_val) + torch.randn_like(x_val) * NOISE

    return x_train, y_train, x_val, y_val


def build_model():
    return nn.Sequential(
        nn.Linear(1, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, 1),
    )


def main():
    x_train, y_train, x_val, y_val = make_data()

    print(f"train examples: {len(x_train)}   (deliberately tiny)")
    print(f"val examples:   {len(x_val)}   (never trained on)")
    print(f"model hidden size: {HIDDEN_SIZE}")

    torch.manual_seed(0)
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} for {N_TRAIN} data points -> plenty of room to memorize")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    section("Watch the two losses drift apart")
    print(f"{'epoch':>6} | {'train loss':>10} | {'val loss':>10} |")
    print(f"{'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-")

    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(EPOCHS + 1):
        # --- training step ---
        model.train()
        pred = model(x_train)
        train_loss = loss_fn(pred, y_train)

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # --- check on held-out data ---
        # no_grad = "just measuring, don't learn from this"
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val)

        history.append((train_loss.item(), val_loss.item()))

        if val_loss.item() < best_val:
            best_val = val_loss.item()
            best_epoch = epoch

        if epoch % 300 == 0:
            flag = ""
            if epoch > 0 and val_loss.item() > best_val * 1.05:
                flag = "  <- val is rising: overfitting"
            print(f"{epoch:>6} | {train_loss.item():>10.4f} | {val_loss.item():>10.4f} |{flag}")

    final_train, final_val = history[-1]

    section("What happened")
    print(f"  best val loss  : {best_val:.4f}  at epoch {best_epoch}")
    print(f"  final val loss : {final_val:.4f}  at epoch {EPOCHS}")
    print(f"  final train loss: {final_train:.4f}")

    gap = final_val / max(final_train, 1e-9)
    print(f"\n  val / train ratio: {gap:.1f}x")

    if gap > 3:
        print("\n  Classic overfitting:")
        print("  train loss kept dropping (memorizing the noise),")
        print("  while val loss stopped improving and got worse.")
        print(f"\n  The model was at its BEST around epoch {best_epoch}.")
        print("  Training longer past that point made it worse, not better.")
    else:
        print("\n  Not much overfitting this run - try lowering N_TRAIN.")

    section("How to fix it")
    print("  1. More data          <- best fix by far")
    print("  2. Stop early         <- keep the epoch with the lowest val loss")
    print("  3. Smaller model      <- fewer parameters to memorize with")
    print("  4. Regularization     <- dropout, weight decay")

    print("\n  In Phase 8 you'll watch this same table while fine-tuning.")
    print("  Rising val loss means: stop training and fix your data.")

    print("\nDone. Phase 1 complete -> phase-02-transformers/")


if __name__ == "__main__":
    main()
