"""
Phase 3 - Example 2: Ollama

Same ideas as the previous examples, but using models already
installed on your machine. No download, much faster, bigger models.

Needs Ollama running. Check with:  ollama list
If a model is missing:             ollama pull llama3.2:3b
                                   ollama pull nomic-embed-text

Run:
    uv run python phase-03-huggingface-basics/02_ollama_basics.py

Reference:
    https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import sys

import ollama

# Try changing this to any model from `ollama list`.
#
# Note: "reasoning" models like qwen3 think first, in a separate hidden field,
# before answering. With a small num_predict they can spend the whole budget
# thinking and return an EMPTY answer. llama3.2 doesn't do this, so it's a
# simpler default for learning. See reply_text() below.
MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def reply_text(response):
    """
    Get the visible answer from a response.

    Reasoning models put their scratch work in message.thinking and the real
    answer in message.content. If content is empty, the model was still
    thinking when it hit the token limit.
    """
    message = response["message"]
    content = (message.get("content") or "").strip()

    if content:
        return content

    if (message.get("thinking") or "").strip():
        return "[model was still thinking - raise num_predict, or use a non-reasoning model]"

    return "[empty response]"


def check_ollama():
    """Fail with a useful message instead of a stack trace."""
    try:
        installed = [m.model for m in ollama.list().models]
    except Exception:
        print("Cannot reach Ollama. Is it running?")
        print("  Start it with:  ollama serve")
        sys.exit(1)

    print("Models on this machine:")
    for name in installed:
        print(f"  - {name}")

    missing = [
        m for m in (MODEL, EMBED_MODEL)
        if not any(name.startswith(m.split(":")[0]) for name in installed)
    ]
    if missing:
        print(f"\nMissing: {missing}")
        print(f"  Get them with:  ollama pull {missing[0]}")
        sys.exit(1)

    return installed


def main():
    check_ollama()
    print(f"\nUsing: {MODEL}")

    section("1. A simple chat")

    # Note the SAME message format as Hugging Face chat templates (Phase 4).
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is a tokenizer? One sentence."}],
    )
    print(f"  {reply_text(response)}")

    section("2. A system message changes the behaviour")

    for system in ("You are a pirate. Stay in character.",
                   "You are a formal academic. Be precise."):
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "What is machine learning?"},
            ],
            options={"num_predict": 60},
        )
        text = reply_text(response).replace("\n", " ")
        print(f"  system: {system[:38]}...")
        print(f"  reply : {text[:150]}\n")

    section("3. Temperature, same as Phase 3 example 1")

    prompt = "Write 5 words describing the ocean."
    for temp in (0.0, 1.5):
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temp, "num_predict": 60},
        )
        text = reply_text(response).replace("\n", " ")
        label = "focused" if temp == 0.0 else "creative"
        print(f"  temp {temp} ({label}): {text[:150]}")

    section("4. Streaming: tokens as they arrive")

    print("  ", end="", flush=True)
    for chunk in ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 10."}],
        stream=True,
        options={"num_predict": 80},
    ):
        print(chunk["message"]["content"], end="", flush=True)
    print("\n\n  Each piece arrived as the model produced it.")
    print("  This is why chat apps feel fast - you see the first word immediately.")

    section("5. Multi-turn: the model has no memory")

    # You must send the whole history every time. The model is stateless.
    history = [
        {"role": "user", "content": "My favourite colour is green."},
        {"role": "assistant", "content": "Green is a lovely colour!"},
        {"role": "user", "content": "What is my favourite colour?"},
    ]

    response = ollama.chat(model=MODEL, messages=history, options={"num_predict": 50})
    print(f"  Sent {len(history)} messages as history.")
    print(f"  Reply: {reply_text(response)[:120]}")
    print("\n  It only knew because we RESENT the earlier messages.")
    print("  Every 'conversation' works this way.")

    section("6. Embeddings: text as a vector")

    texts = ["I love cats", "I adore kittens", "The stock market fell"]
    result = ollama.embed(model=EMBED_MODEL, input=texts)
    vectors = result["embeddings"]

    print(f"  model: {EMBED_MODEL}")
    print(f"  {len(vectors)} texts -> {len(vectors[0])} numbers each\n")

    def similarity(a, b):
        """Cosine similarity: 1.0 = identical meaning, 0.0 = unrelated."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb)

    print(f"  {'comparison':<44} {'score':>6}")
    print(f"  {'-' * 44} {'-' * 6}")
    for i, j in ((0, 1), (0, 2)):
        score = similarity(vectors[i], vectors[j])
        print(f"  {texts[i]!r} vs {texts[j]!r:<22} {score:>6.3f}")

    print("\n  Similar meanings score higher, even with different words.")
    print("  This is the foundation of search and RAG.")

    print("\nDone. Phase 3 complete -> phase-04-chat-templates/")


if __name__ == "__main__":
    main()
