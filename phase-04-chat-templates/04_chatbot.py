"""
Phase 4 - Example 4: A working chatbot

Everything from this phase in one small program. The entire trick
is the `messages` list: append every turn, resend it each time.

Needs Ollama:  ollama pull llama3.2:3b

Run:
    uv run python phase-04-chat-templates/04_chatbot.py

Commands:  /history   show the message list
           /reset     clear memory
           /system    change the system prompt
           /quit      exit

Reference:
    https://github.com/ollama/ollama-python
"""

import sys

import ollama

MODEL = "llama3.2:3b"

# Try changing this! See "Try it yourself" in the README.
SYSTEM_PROMPT = "You are a helpful assistant. Keep answers under 3 sentences."


def check():
    try:
        ollama.list()
    except Exception:
        print("Cannot reach Ollama. Start it with:  ollama serve")
        sys.exit(1)


def show_history(messages):
    print(f"\n  {len(messages)} messages currently being resent:")
    print(f"  {'-' * 56}")
    for i, m in enumerate(messages):
        text = m["content"].replace("\n", " ")
        if len(text) > 44:
            text = text[:41] + "..."
        print(f"  {i:>2}. {m['role']:<10} {text}")
    print(f"  {'-' * 56}")

    chars = sum(len(m["content"]) for m in messages)
    print(f"  ~{chars // 4} tokens resent on every message you send.\n")


def main():
    check()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Chatbot using {MODEL}")
    print(f"System prompt: {SYSTEM_PROMPT}")
    print("\nCommands: /history  /reset  /system  /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye.")
            break

        if user_input == "/history":
            show_history(messages)
            continue

        if user_input == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  Memory cleared. The bot now knows nothing about this chat.\n")
            continue

        if user_input == "/system":
            new_prompt = input("  New system prompt: ").strip()
            if new_prompt:
                messages[0] = {"role": "system", "content": new_prompt}
                print("  Updated. It applies from your next message.\n")
            continue

        # 1. add what the user said
        messages.append({"role": "user", "content": user_input})

        # 2. send the ENTIRE history and stream the reply
        print("Bot: ", end="", flush=True)
        reply = ""
        try:
            for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
                piece = chunk["message"].get("content") or ""
                print(piece, end="", flush=True)
                reply += piece
        except Exception as e:
            print(f"\n  Error: {e}")
            messages.pop()  # undo the user message so history stays clean
            continue

        print("\n")

        # 3. remember what the bot said, so the next turn has context
        messages.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()
