"""
Phase 2 - Example 1: Tokenization

Models never see text. They see numbers. This shows the conversion,
and a few things about it that surprise people.

Downloads only the tokenizer (~2 MB), not the model.

Run:
    uv run python phase-02-transformers/01_tokenizer.py

Reference:
    https://huggingface.co/docs/transformers/tokenizer_summary
"""

from transformers import AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

WORDS = [
    "cat",
    " cat",          # leading space -> different token!
    "Cat",           # capital -> different token!
    "cats",
    "antidisestablishmentarianism",
    "hello world",
    "def main():",
    "नमस्ते",          # non-English uses more tokens
    "🚀",
]


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    print(f"Loading tokenizer: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")

    section("1. Text to numbers and back")

    text = "Transformers are powerful."
    ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([i]) for i in ids]

    print(f"text : {text!r}\n")
    print(f"{'token':<16} {'id':>8}")
    print(f"{'-' * 16} {'-' * 8}")
    for piece, i in zip(pieces, ids):
        print(f"{piece!r:<16} {i:>8}")

    print(f"\ndecoded back: {tokenizer.decode(ids)!r}")

    section("2. Token count is not word count")

    print(f"{'text':<32} {'tokens':>7}  pieces")
    print(f"{'-' * 32} {'-' * 7}  {'-' * 28}")

    for word in WORDS:
        ids = tokenizer.encode(word)
        pieces = [tokenizer.decode([i]) for i in ids]
        shown = " | ".join(repr(p) for p in pieces)
        if len(shown) > 28:
            shown = shown[:25] + "..."
        print(f"{word!r:<32} {len(ids):>7}  {shown}")

    print("\nNotice:")
    print("  'cat' and ' cat' are DIFFERENT tokens (the space belongs to the token)")
    print("  long rare words split into many pieces")
    print("  non-English text costs more tokens for the same meaning")
    print("  the emoji shows as '�' because each token is only PART of it -")
    print("    you need all 3 tokens together to rebuild the character")

    section("3. What the model actually receives")

    batch = tokenizer("Hello world", return_tensors="pt")

    for key, value in batch.items():
        print(f"  {key:<16} shape {tuple(value.shape)}  {value.tolist()}")

    print("\n  input_ids      = the token numbers")
    print("  attention_mask = 1 means 'real token', 0 means 'padding, ignore'")

    section("4. Padding: making inputs the same length")

    # Models process batches as rectangles, so shorter texts get padded.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(
        ["Hi", "A much longer sentence here"],
        padding=True,
        return_tensors="pt",
    )

    print(f"  input_ids shape: {tuple(batch['input_ids'].shape)}  (2 texts, padded to same length)\n")
    for i in range(2):
        ids = batch["input_ids"][i]
        mask = batch["attention_mask"][i]
        print(f"  text {i}: ids  {ids.tolist()}")
        print(f"          mask {mask.tolist()}  <- 0s are padding")

    print("\n  The mask tells the model to ignore the padded positions.")
    print("  Forgetting pad_token is a very common error - you'll see it in Phase 8.")

    section("5. Different models, different tokenizers")

    others = [
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "gpt2",
    ]
    sample = "Fine-tuning is fun!"

    print(f"text: {sample!r}\n")
    for name in others:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            ids = tok.encode(sample)
            print(f"  {name:<40} {len(ids):>2} tokens  vocab {tok.vocab_size:,}")
        except Exception as e:
            print(f"  {name:<40} skipped ({type(e).__name__})")

    print("\n  Same text, different token counts.")
    print("  This is why you must use the tokenizer that matches your model.")

    print("\nDone. Next: 02_inside_model.py")


if __name__ == "__main__":
    main()
