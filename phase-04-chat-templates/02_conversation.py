"""
Phase 4 - Example 2: Multi-turn conversations

The model has NO memory. A conversation only works because you
resend the whole history every single time.

Needs Ollama:  ollama pull llama3.2:3b

Run:
    uv run python phase-04-chat-templates/02_conversation.py

Reference:
    https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import sys

import ollama

MODEL = "llama3.2:3b"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def ask(messages, num_predict=60):
    try:
        response = ollama.chat(
            model=MODEL, messages=messages, options={"num_predict": num_predict}
        )
    except Exception:
        print(f"\nCannot reach Ollama or model '{MODEL}' is missing.")
        print(f"  ollama serve")
        print(f"  ollama pull {MODEL}")
        sys.exit(1)

    return (response["message"].get("content") or "").strip().replace("\n", " ")


def main():
    print(f"Using: {MODEL}")

    section("1. Without history: the model has no idea")

    # Each call is completely independent.
    ask([{"role": "user", "content": "My name is Sam."}])
    reply = ask([{"role": "user", "content": "What is my name?"}])

    print("  Call 1: 'My name is Sam.'")
    print("  Call 2: 'What is my name?'   <- sent on its own")
    print(f"\n  Reply: {reply[:140]}")
    print("\n  It can't know. The first call left no trace.")

    section("2. With history: it works")

    history = [
        {"role": "user", "content": "My name is Sam."},
        {"role": "assistant", "content": "Nice to meet you, Sam!"},
        {"role": "user", "content": "What is my name?"},
    ]

    reply = ask(history)
    print(f"  Sent all {len(history)} messages together.")
    print(f"\n  Reply: {reply[:140]}")
    print("\n  Same model. The difference is entirely what we sent.")

    section("3. Building a conversation turn by turn")

    conversation = [
        {"role": "system", "content": "You are a concise assistant. One short sentence."}
    ]

    turns = [
        "I'm learning Python.",
        "What should I learn after the basics?",
        "Why that one?",  # only makes sense with context
    ]

    for turn in turns:
        conversation.append({"role": "user", "content": turn})
        answer = ask(conversation, num_predict=50)
        conversation.append({"role": "assistant", "content": answer})

        print(f"\n  You: {turn}")
        print(f"  Bot: {answer[:130]}")
        print(f"       (history is now {len(conversation)} messages)")

    print("\n  'Why that one?' only worked because the earlier turns were there.")

    section("4. The cost of remembering")

    # Rough estimate: ~4 characters per token.
    total_chars = sum(len(m["content"]) for m in conversation)
    print(f"  messages: {len(conversation)}")
    print(f"  characters resent on the LAST call: {total_chars}")
    print(f"  roughly {total_chars // 4} tokens\n")
    print("  Every new turn resends everything before it.")
    print("  Long chats get slower and more expensive - which is why real")
    print("  apps eventually summarize or drop the oldest messages.")

    section("5. The system message steers everything")

    question = "How do I sort a list in Python?"

    for system in (
        "You are a beginner-friendly tutor. Explain simply.",
        "You are a terse senior engineer. Code only, no explanation.",
    ):
        answer = ask(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            num_predict=70,
        )
        print(f"\n  system: {system}")
        print(f"  reply : {answer[:160]}")

    print("\n  Same question, very different answers.")
    print("  The system message is the cheapest way to control a model.")

    print("\nDone. Next: 03_prompt_techniques.py")


if __name__ == "__main__":
    main()
