"""
Phase 6 - Example 3: Splits, epochs, and overfitting

This is Phase 1's overfitting lesson, now on a real language model.

We deliberately overfit a small model on a handful of examples so you
can watch validation loss turn upward - the signal that tells you to
stop training and go fix your data.

Takes about a minute on Apple Silicon.

Run:
    uv run python phase-06-finetuning-basics/03_split_and_overfit.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from llmkit import TrainingMonitor, pick_device, section  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Deliberately tiny, so overfitting shows up quickly.
EPOCHS = 40
LEARNING_RATE = 5e-5

# A style the base model does not have: terse incident notes.
EXAMPLES = [
    "Ticket 401 resolved. Cause: expired token. Action: reissued.",
    "Ticket 402 resolved. Cause: disk full. Action: cleared logs.",
    "Ticket 403 resolved. Cause: bad config. Action: reverted change.",
    "Ticket 404 resolved. Cause: DNS timeout. Action: switched resolver.",
    "Ticket 405 resolved. Cause: stale cache. Action: purged keys.",
    "Ticket 406 resolved. Cause: rate limit. Action: raised quota.",
    "Ticket 407 resolved. Cause: memory leak. Action: restarted worker.",
    "Ticket 408 resolved. Cause: cert expiry. Action: renewed cert.",
]

# Held out. The model NEVER trains on these.
VALIDATION = [
    "Ticket 501 resolved. Cause: queue backlog. Action: added consumers.",
    "Ticket 502 resolved. Cause: bad migration. Action: rolled back.",
    "Ticket 503 resolved. Cause: clock drift. Action: resynced NTP.",
]


def batch_loss(model, tokenizer, texts, device):
    """Average loss over a list of texts. No gradients - just measuring."""
    total = 0.0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").to(device)
            total += model(**ids, labels=ids["input_ids"]).loss.item()
    return total / len(texts)


def main():
    device = pick_device(verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    section("1. The three-way split")

    print(f"  train      : {len(EXAMPLES)} examples  (the model learns from these)")
    print(f"  validation : {len(VALIDATION)} examples  (checked, never learned from)")
    print(f"  test       : held back entirely until the very end")
    print("\n  Validation is how you catch overfitting WHILE training.")
    print("  Test is touched once, at the end, so it stays honest -")
    print("  if you tune against it, it stops being a fair measure.")

    section("2. Before training")

    train_before = batch_loss(model, tokenizer, EXAMPLES, device)
    val_before = batch_loss(model, tokenizer, VALIDATION, device)

    print(f"  train loss      : {train_before:.4f}")
    print(f"  validation loss : {val_before:.4f}")
    print("\n  Both high - this writing style is unfamiliar to the model.")

    section("3. Training, watching both losses")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    monitor = TrainingMonitor("overfit-demo")

    print(f"  {EPOCHS} epochs on {len(EXAMPLES)} examples, lr={LEARNING_RATE}")
    print("  (an epoch = one pass over all training examples)\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for text in EXAMPLES:
            ids = tokenizer(text, return_tensors="pt").to(device)
            loss = model(**ids, labels=ids["input_ids"]).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 4 == 0 or epoch == 1:
            model.eval()
            train_loss = batch_loss(model, tokenizer, EXAMPLES, device)
            val_loss = batch_loss(model, tokenizer, VALIDATION, device)
            monitor.log(epoch, train_loss, val_loss)

    section("4. The loss curve")

    monitor.plot()
    monitor.summary()

    section("5. Reading this correctly")

    best_val, best_epoch = monitor.best_val
    final_val = [v for v in monitor.val if v is not None][-1]
    final_train = monitor.train[-1]

    print(f"  train loss went {monitor.train[0]:.3f} -> {final_train:.3f}")
    print(f"  validation bottomed out at {best_val:.3f} (epoch {best_epoch})")
    print(f"  validation ended at {final_val:.3f}")

    if final_val > best_val * 1.05:
        print(f"\n  Validation got WORSE after epoch {best_epoch}, while training")
        print("  loss kept falling. The model stopped learning the pattern and")
        print("  started memorising these 8 exact sentences.")
        print(f"\n  The right model was the one at epoch {best_epoch}.")
        print("  Everything after that was wasted compute making it worse.")
    else:
        print("\n  Validation still improving - this run could train longer.")

    print("\n  Train loss alone would have told you everything was great.")
    print("  It always looks great. That is why you need the split.")

    section("6. What to do about it")

    print("  1. More and more varied data      <- by far the best fix")
    print("  2. Stop at the best validation epoch (early stopping)")
    print("  3. Fewer epochs - 1 to 3 is normal for real fine-tuning")
    print("  4. Lower learning rate")
    print("  5. LoRA, which freezes the base model (Phase 10)")

    print("\n  Notice that 8 examples and 40 epochs is an absurd setup.")
    print("  It was chosen to make overfitting obvious in one minute.")
    print("  Real runs use 500+ examples and 1-3 epochs.")

    section("7. Choosing epochs and learning rate")

    rows = [
        ("epochs", "1-3", "more than 3 usually memorises"),
        ("learning rate (full)", "1e-5 to 5e-5", "small - you are nudging, not teaching"),
        ("learning rate (LoRA)", "1e-4 to 2e-4", "higher, since only adapters move"),
        ("batch size", "1-8 on small GPUs", "raise until you run out of memory"),
    ]

    print(f"  {'setting':<22} {'typical':<16} note")
    print(f"  {'-' * 22} {'-' * 16} {'-' * 34}")
    for name, typical, note in rows:
        print(f"  {name:<22} {typical:<16} {note}")

    print("\n  Start with the defaults. Change ONE thing at a time,")
    print("  and only when validation loss tells you to.")

    print("\nDone. Phase 6 complete -> phase-07-datasets/")


if __name__ == "__main__":
    main()
