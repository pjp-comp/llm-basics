"""
Phase 5 - Example 1: Embeddings

An embedding turns text into a list of numbers, positioned so that
similar meanings land near each other. Everything in RAG rests on this.

First run downloads all-MiniLM-L6-v2 (~90 MB), then it's cached.

Run:
    uv run python phase-05-rag/01_embeddings.py

Reference:
    https://www.sbert.net/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from llmkit import Timer, section  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def cosine(a, b):
    """
    Cosine similarity: the angle between two vectors.

     1.0 = same direction (same meaning)
     0.0 = unrelated
    -1.0 = opposite

    We use the angle, not the distance, because vector LENGTH tends to
    reflect text length rather than meaning.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    print(f"Loading embedding model: {MODEL}")
    with Timer("load"):
        model = SentenceTransformer(MODEL)

    # API name differs across sentence-transformers versions.
    dims = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    print(f"Each piece of text becomes {dims} numbers.")

    section("1. What an embedding looks like")

    vec = model.encode("The cat sat on the mat")
    print(f"  text  : 'The cat sat on the mat'")
    print(f"  shape : {vec.shape}")
    print(f"  first 8 numbers: {[round(float(x), 3) for x in vec[:8]]}")
    print("\n  No single number means anything on its own.")
    print("  The meaning lives in the whole pattern.")

    section("2. Similar meanings score higher")

    pairs = [
        ("I love cats", "I adore kittens"),
        ("I love cats", "Felines are wonderful"),
        ("I love cats", "The stock market fell today"),
        ("How do I reset my password?", "I forgot my login details"),
        ("How do I reset my password?", "What is the capital of France?"),
    ]

    print(f"  {'text A':<32} {'text B':<32} {'score':>6}")
    print(f"  {'-' * 32} {'-' * 32} {'-' * 6}")

    for a, b in pairs:
        score = cosine(model.encode(a), model.encode(b))
        print(f"  {a[:30]:<32} {b[:30]:<32} {score:>6.3f}")

    print("\n  Notice: 'I love cats' and 'Felines are wonderful' score high")
    print("  despite sharing NO words. That is the whole point - embeddings")
    print("  capture meaning, not spelling. Keyword search would score 0.")

    section("3. Word overlap is not meaning")

    tricky = [
        ("The bank raised interest rates", "I sat on the river bank"),
        ("The bank raised interest rates", "The central bank hiked borrowing costs"),
    ]

    for a, b in tricky:
        score = cosine(model.encode(a), model.encode(b))
        print(f"\n  A: {a}")
        print(f"  B: {b}")
        print(f"  score: {score:.3f}")

    print("\n  Same word 'bank', very different scores.")
    print("  The model uses context, not the word alone.")

    section("4. Ranking: what retrieval actually does")

    # This IS retrieval, in miniature. Compare a question against
    # everything you have, then sort by score.
    documents = [
        "Nimbus Deploy supports Python, Node.js, Go, Ruby, and Java.",
        "Holiday allowance is 28 days per year plus public holidays.",
        "The Team plan costs 18 pounds per user per month.",
        "Rollbacks take about 20 seconds and reuse the previous image.",
        "Core working hours are 10:00 to 16:00 local time.",
    ]

    question = "How much does the paid plan cost?"

    doc_vectors = model.encode(documents)
    q_vector = model.encode(question)

    scored = sorted(
        ((cosine(q_vector, dv), doc) for dv, doc in zip(doc_vectors, documents)),
        reverse=True,
    )

    print(f"  question: {question!r}\n")
    print(f"  {'score':>6}  document")
    print(f"  {'-' * 6}  {'-' * 52}")
    for score, doc in scored:
        marker = "  <- best match" if score == scored[0][0] else ""
        print(f"  {score:>6.3f}  {doc[:50]}{marker}")

    print("\n  The pricing sentence won, without sharing the word 'cost'.")
    print("  Take the top 1-3 of these, paste into a prompt, and you have RAG.")

    section("5. Encoding many texts at once is much faster")

    many = documents * 40  # 200 texts

    with Timer("one at a time", quiet=True) as t1:
        for text in many[:40]:
            model.encode(text)

    with Timer("all at once", quiet=True) as t2:
        model.encode(many[:40])

    print(f"  40 texts one by one : {t1.seconds:.2f}s")
    print(f"  40 texts in a batch : {t2.seconds:.2f}s")
    if t2.seconds > 0:
        print(f"  batching was {t1.seconds / t2.seconds:.1f}x faster")
    print("\n  Always pass a list when you can.")

    print("\nDone. Next: 02_chunking.py")


if __name__ == "__main__":
    main()
