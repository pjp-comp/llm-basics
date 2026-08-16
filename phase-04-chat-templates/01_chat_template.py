"""
Phase 4 - Example 1: Chat templates

Shows the ACTUAL string that gets sent to a model, with all the
special markers visible. This is the thing you can't see otherwise.

Run:
    uv run python phase-04-chat-templates/01_chat_template.py

Reference:
    https://huggingface.co/docs/transformers/chat_templating
"""

from transformers import AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def show(text):
    """Print with visible line breaks so markers are easy to see."""
    print("  " + "-" * 56)
    for line in text.split("\n"):
        print(f"  | {line}")
    print("  " + "-" * 56)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print(f"Model: {MODEL}")

    section("1. One user message")

    messages = [{"role": "user", "content": "What is 2+2?"}]

    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("  You wrote:")
    print(f"    {messages}\n")
    print("  The model actually receives:")
    show(formatted)
    print("\n  Those <|im_start|> markers tell the model whose turn it is.")
    print("\n  Notice a system message appeared that you never wrote!")
    print("  Some models add a default one. Another good reason to PRINT")
    print("  the template instead of assuming what it contains.")

    section("2. add_generation_prompt: the important flag")

    with_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    without = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    print("  add_generation_prompt=True  (correct for asking a question):")
    show(with_prompt)
    print("\n  add_generation_prompt=False:")
    show(without)

    extra = with_prompt[len(without):]
    print(f"\n  The difference is: {extra!r}")
    print("  That opens the assistant's turn, telling the model to answer NOW.")
    print("  Without it, the model might keep writing as the USER instead.")

    section("3. Adding a system message")

    with_system = [
        {"role": "system", "content": "You are a maths tutor. Be brief."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    show(tokenizer.apply_chat_template(
        with_system, tokenize=False, add_generation_prompt=True
    ))
    print("\n  The system message goes FIRST and sets standing rules.")

    section("4. A full conversation")

    conversation = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Nice to meet you, Sam!"},
        {"role": "user", "content": "What is my name?"},
    ]

    show(tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    ))
    print("\n  The whole history is one long string with markers between turns.")
    print("  The model has no memory - this text IS its memory.")

    section("5. The hand-written version people try")

    manual = "User: What is 2+2?\nAssistant:"

    print("  What beginners often write:")
    show(manual)
    print("\n  What the model was trained on:")
    show(with_prompt)
    print("\n  Different. The model usually still replies, but worse -")
    print("  and you get no error telling you anything is wrong.")

    section("6. Every model family differs")

    others = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
    simple = [{"role": "user", "content": "Hi"}]

    for name in others:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            text = tok.apply_chat_template(
                simple, tokenize=False, add_generation_prompt=True
            )
            print(f"\n  {name}:")
            print(f"    {text!r}")
        except Exception as e:
            print(f"\n  {name}: skipped ({type(e).__name__})")

    print("\n  Same messages, different formats.")
    print("  This is why you let the tokenizer do it instead of hardcoding.")

    section("7. What actually goes to the model: numbers")

    # Note: with tokenize=True this returns a BatchEncoding, not a plain list,
    # so we pull out input_ids. Printing types you don't expect is normal -
    # library APIs change between versions.
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    if hasattr(encoded, "input_ids"):
        ids = encoded["input_ids"]
        if hasattr(ids[0], "__iter__"):  # batched: [[...]]
            ids = ids[0]
    else:
        ids = encoded

    ids = list(ids)

    print(f"  token count: {len(ids)}")
    print(f"  first 12 ids: {ids[:12]}\n")
    print(f"  {'id':>8}  token")
    print(f"  {'-' * 8}  {'-' * 24}")
    for tid in ids[:10]:
        print(f"  {tid:>8}  {tokenizer.decode([tid])!r}")

    print("\n  The markers are single tokens, not typed-out text.")
    print("\nDone. Next: 02_conversation.py")


if __name__ == "__main__":
    main()
