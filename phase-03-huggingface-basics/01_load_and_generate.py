"""
Phase 3 - Example 1: Loading a model and generating text

Shows what each generation setting actually does, by running the
same prompt with different settings.

First run downloads SmolLM2-135M (~270 MB) and caches it.

Run:
    uv run python phase-03-huggingface-basics/01_load_and_generate.py

Reference:
    https://huggingface.co/docs/transformers/generation_strategies
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT = "The best thing about learning to code is"

# Try changing this! See "Try it yourself" in the README.
TEMPERATURE = 1.5


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate(model, tokenizer, device, **kwargs):
    """Tokenize -> generate -> decode. Returns only the NEW text."""
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs,
        )

    # output includes the prompt, so slice it off to see just the completion
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = pick_device()
    print(f"Loading {MODEL} on {device}...")
    print("(first run downloads ~270 MB, then it's cached)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. {n_params / 1e6:.0f}M parameters.")
    print(f"\nPrompt: {PROMPT!r}")

    section("1. Greedy: always pick the most likely token")

    # do_sample=False means no randomness at all.
    for run in (1, 2):
        text = generate(model, tokenizer, device, max_new_tokens=25, do_sample=False)
        print(f"  run {run}: {text}")

    print("\n  Both runs are IDENTICAL - greedy is deterministic.")
    print("  Good for reproducibility, but can get repetitive.")

    section("2. Sampling: pick randomly, weighted by probability")

    torch.manual_seed(0)
    for run in (1, 2):
        text = generate(
            model, tokenizer, device,
            max_new_tokens=25, do_sample=True, temperature=0.8, top_p=0.9,
        )
        print(f"  run {run}: {text}")

    print("\n  Different each time - that's the randomness.")

    section("3. Temperature: how random?")

    for temp in (0.1, 0.7, TEMPERATURE):
        torch.manual_seed(0)
        text = generate(
            model, tokenizer, device,
            max_new_tokens=25, do_sample=True, temperature=temp, top_p=1.0,
        )
        label = {0.1: "focused", 0.7: "balanced"}.get(temp, "wild")
        print(f"  temp {temp:<4} ({label:<8}): {text}")

    print("\n  Low  = safe, predictable, can loop")
    print("  High = creative, and eventually nonsense")

    section("4. Length: max_new_tokens")

    for limit in (5, 15, 40):
        text = generate(model, tokenizer, device, max_new_tokens=limit, do_sample=False)
        print(f"  {limit:>2} tokens: {text}")

    print("\n  It stops mid-sentence because it hit the limit,")
    print("  not because it finished a thought.")

    section("5. Generation is a loop")

    # Do it manually, one token at a time, to make the loop visible.
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    ids = inputs["input_ids"]

    print(f"  start: {PROMPT!r}")
    for step in range(5):
        with torch.no_grad():
            logits = model(ids).logits

        # logits[0, -1] = scores for the token after the LAST position
        next_id = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        piece = tokenizer.decode(next_id[0])

        print(f"  step {step + 1}: predicted {piece!r}")
        ids = torch.cat([ids, next_id], dim=1)  # append and go again

    print(f"\n  result: {tokenizer.decode(ids[0], skip_special_tokens=True)!r}")
    print("\n  That's all .generate() does, in a faster loop.")

    print("\nDone. Next: 02_ollama_basics.py")


if __name__ == "__main__":
    main()
