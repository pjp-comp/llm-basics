"""
Phase 6 - Example 2: What training actually changes

Fine-tuning is the Phase 1 training loop, applied to a language model.
Same five steps. The only new idea is what "loss" means for text.

Nothing is downloaded here beyond the small model you already have.

Run:
    uv run python phase-06-finetuning-basics/02_what_training_does.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from llmkit import pick_device, print_model_size, section  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    device = pick_device(verbose=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    section("1. What is actually inside a model")

    print_model_size(model, "SmolLM2-135M")

    print("\n  Those 134 million numbers ARE the model.")
    print("  Training = changing some of them. That is the whole idea.")

    section("2. Loss for a language model")

    # The model predicts the next token at every position. Loss measures
    # how surprised it was by the token that actually came next.
    sentences = [
        "The capital of France is Paris",     # true, common
        "The capital of France is Berlin",    # false
        "The capital of France is banana",    # nonsense
    ]

    print("  Loss = how SURPRISED the model was by the real next token.")
    print("  Low loss means it expected that text.\n")
    print(f"  {'sentence':<42} {'loss':>7}")
    print(f"  {'-' * 42} {'-' * 7}")

    for text in sentences:
        ids = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            # Passing labels makes the model compute loss for us.
            # It shifts internally: predict token 2 from 1, 3 from 1-2, etc.
            out = model(**ids, labels=ids["input_ids"])

        print(f"  {text:<42} {out.loss.item():>7.3f}")

    print("\n  The model is least surprised by the true sentence.")
    print("  Training pushes loss DOWN on your examples - which means")
    print("  making your text the thing it expects to see.")

    section("3. Loss on YOUR data is the whole game")

    # Style the base model has never seen.
    target_style = "Ticket resolved. Root cause: expired token. Action: reissued."
    generic = "I have resolved your ticket for you. Let me know if you need anything else!"

    print("  Say you want terse, structured replies.\n")

    for label, text in (("your style", target_style), ("generic style", generic)):
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            loss = model(**ids, labels=ids["input_ids"]).loss.item()
        print(f"  {label:<16} loss {loss:>6.3f}   {text[:52]}")

    print("\n  Higher loss on your style = the model finds it unusual.")
    print("  Fine-tuning lowers that number, and the style becomes its default.")

    section("4. One training step, in full")

    # This is a REAL gradient update. We do a single step so you can see
    # the loss move, then throw the change away.
    text = target_style
    ids = tokenizer(text, return_tensors="pt").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print(f"  training on: {text!r}\n")
    print(f"  {'step':>5}  {'loss':>8}")
    print(f"  {'-' * 5}  {'-' * 8}")

    for step in range(6):
        # 1. forward
        out = model(**ids, labels=ids["input_ids"])
        loss = out.loss

        print(f"  {step:>5}  {loss.item():>8.4f}")

        # 2. backward
        loss.backward()
        # 3. update
        optimizer.step()
        # 4. clear gradients  <- forget this and training breaks
        optimizer.zero_grad()

    print("\n  The loss dropped. The model now finds that sentence normal.")
    print("  That is fine-tuning. Everything else is scale and plumbing.")

    section("5. Why one example is not enough")

    print("  We just trained on ONE sentence, six times. The model now")
    print("  strongly expects that exact text - and has got slightly worse")
    print("  at everything else. That is called CATASTROPHIC FORGETTING.")
    print("\n  Real fine-tuning avoids it with:")
    print("    - hundreds or thousands of varied examples")
    print("    - a small learning rate (1e-5 to 2e-4)")
    print("    - few epochs (1-3), so it generalises instead of memorising")
    print("    - LoRA, which freezes the original weights entirely (Phase 10)")

    section("6. The vocabulary you now understand")

    rows = [
        ("loss", "how surprised the model is by your text"),
        ("epoch", "one pass through your whole dataset"),
        ("batch size", "examples processed before each update"),
        ("learning rate", "how big each adjustment is"),
        ("gradient", "which direction each number should move"),
        ("checkpoint", "saved copy of the weights mid-training"),
        ("overfitting", "memorising your examples, worse on new ones"),
    ]

    print(f"  {'term':<16} meaning")
    print(f"  {'-' * 16} {'-' * 48}")
    for term, meaning in rows:
        print(f"  {term:<16} {meaning}")

    print("\n  Every one of these appeared in Phase 1 with a 2-parameter model.")
    print("  Nothing new was introduced - only the model got bigger.")

    print("\nDone. Next: 03_split_and_overfit.py")


if __name__ == "__main__":
    main()
