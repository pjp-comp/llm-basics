"""
Phase 6 - Example 1: Do you actually need fine-tuning?

Before spending a week training, rule out the cheaper options.
This runs the SAME task four ways and shows what each one fixes.

Needs Ollama:  ollama pull llama3.2:3b

Run:
    uv run python phase-06-finetuning-basics/01_when_to_finetune.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import ollama  # noqa: E402

from llmkit import section  # noqa: E402

MODEL = "llama3.2:3b"

# The task: turn a support message into strict JSON.
TICKETS = [
    "Hi, I can't log in since yesterday. This is really urgent, our whole team is blocked.",
    "The invoice for March seems wrong, it shows £340 but we only have 12 seats.",
    "Just wanted to say the new dashboard is lovely. No issue, keep it up!",
]


def ask(messages, num_predict=120, temperature=0.0):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"num_predict": num_predict, "temperature": temperature},
        )
    except Exception:
        print(f"\nCannot reach Ollama. Start it:  ollama serve")
        print(f"Then:  ollama pull {MODEL}")
        sys.exit(1)
    return (response["message"].get("content") or "").strip()


CATEGORIES = {"access", "billing", "feedback"}
URGENCIES = {"low", "medium", "high"}


def looks_like_json(text):
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def check_schema(text):
    """
    Valid JSON is not the same as CORRECT JSON. This checks that the
    values are from the allowed sets, and that no extra keys appeared.
    Returns (ok, list_of_problems).
    """
    import json

    try:
        data = json.loads(text.strip())
    except Exception:
        return False, ["not parseable as JSON"]

    problems = []

    if data.get("category") not in CATEGORIES:
        problems.append(f"category={data.get('category')!r} not in {sorted(CATEGORIES)}")
    if data.get("urgency") not in URGENCIES:
        problems.append(f"urgency={data.get('urgency')!r} not in {sorted(URGENCIES)}")

    extra = set(data) - {"category", "urgency", "summary"}
    if extra:
        problems.append(f"unexpected keys: {sorted(extra)}")

    return not problems, problems


def main():
    print(f"Using: {MODEL}")
    print("\nTask: turn a support ticket into JSON with fields")
    print("      category, urgency, summary")

    section("Option 1: a vague prompt")

    ticket = TICKETS[0]
    out = ask(f"Categorise this support ticket: {ticket}")

    print(f"  ticket: {ticket[:70]}...\n")
    print(f"  output:\n    {out[:250]}")
    print(f"\n  valid JSON? {'YES' if looks_like_json(out) else 'NO'}")
    print("  Readable, but not machine-usable. No fixed shape.")

    section("Option 2: a specific prompt")

    out = ask(
        "Extract from this support ticket and reply with ONLY a JSON object "
        'with keys "category", "urgency", "summary". '
        'category must be one of: access, billing, feedback. '
        'urgency must be one of: low, medium, high. '
        f"summary must be under 10 words.\n\nTicket: {ticket}"
    )

    print(f"  output:\n    {out[:250]}")
    print(f"\n  valid JSON? {'YES' if looks_like_json(out) else 'NO'}")
    print("  Much better, and it cost nothing but a clearer instruction.")

    section("Option 3: few-shot (show, don't tell)")

    few_shot = [
        {
            "role": "system",
            "content": "You convert support tickets to JSON. Reply with JSON only.",
        },
        {"role": "user", "content": "My card was charged twice this month."},
        {
            "role": "assistant",
            "content": '{"category": "billing", "urgency": "high", "summary": "Duplicate charge on card"}',
        },
        {"role": "user", "content": "Love the new export button, very handy."},
        {
            "role": "assistant",
            "content": '{"category": "feedback", "urgency": "low", "summary": "Positive feedback on export"}',
        },
        {"role": "user", "content": ticket},
    ]

    out = ask(few_shot)
    print(f"  output:\n    {out[:250]}")
    print(f"\n  valid JSON? {'YES' if looks_like_json(out) else 'NO'}")
    print("  Showing two examples locks the format harder than describing it.")

    section("Valid JSON is not the same as CORRECT JSON")

    valid = 0
    schema_ok = 0

    for t in TICKETS:
        messages = few_shot[:-1] + [{"role": "user", "content": t}]
        out = ask(messages, num_predict=80)

        is_json = looks_like_json(out)
        ok, problems = check_schema(out)
        valid += is_json
        schema_ok += ok

        print(f"\n  ticket : {t[:58]}...")
        print(f"  output : {out[:105]}")
        print(f"  json   : {'YES' if is_json else 'NO'}    schema: {'OK' if ok else 'FAILED'}")
        for problem in problems:
            print(f"           - {problem}")

    print(f"\n  {valid}/{len(TICKETS)} produced valid JSON")
    print(f"  {schema_ok}/{len(TICKETS)} matched the required schema")

    if schema_ok < valid:
        print("\n  This is the interesting failure. The model produced")
        print("  well-formed JSON while inventing values you never allowed.")
        print("  Prompts REQUEST a schema. They cannot ENFORCE one.")

    section("Option 4: fine-tuning - and when it is worth it")

    print("  Fine-tuning would help here if:")
    print("    - few-shot keeps drifting outside your allowed values (see above)")
    print("    - you process millions of tickets and want to drop the examples")
    print("      from every prompt (they cost tokens on every single call)")
    print("    - you need a smaller, cheaper model to match a big one's quality")
    print("    - your categories are unusual and not in any pretraining data")
    print("\n  Fine-tuning would NOT help if:")
    print("    - you want the model to know your product catalogue  -> RAG")
    print("    - you want current pricing or docs                   -> RAG")
    print("    - the prompt is simply vague                         -> fix the prompt")
    print("\n  Worth knowing: for strict schemas there is a third option -")
    print("  CONSTRAINED DECODING, where the tool only lets the model emit")
    print("  tokens that keep the output valid. Ollama supports this with")
    print("  its `format` parameter. Often cheaper than fine-tuning.")

    section("The decision, in order")

    print("  1. Write a clearer prompt        (minutes, free)")
    print("  2. Add 2-5 few-shot examples     (minutes, costs tokens per call)")
    print("  3. Use RAG if it needs facts     (hours, no training)")
    print("  4. Fine-tune                     (days, needs data + GPU)")
    print("\n  Each step is roughly 10x more expensive than the one above.")
    print("  Most projects stop at step 2 and never need to train anything.")

    print("\n  If you DO fine-tune, few-shot output is a great source of")
    print("  training data - generate it, correct it by hand, train on it.")

    print("\nDone. Next: 02_what_training_does.py")


if __name__ == "__main__":
    main()
