"""
Phase 5 - Example 5: Measuring RAG quality

Most bad RAG systems are bad at RETRIEVAL, not generation. If the
right chunk never reaches the model, no model can save you.

This measures retrieval separately from generation, which is the
only way to know which half to fix.

Run:
    uv run python phase-05-rag/05_rag_quality.py

Reference:
    https://docs.ragas.io/en/stable/concepts/metrics/
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from llmkit import section  # noqa: E402

DOCS = Path(__file__).parent / "docs"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Set COMPARE_EMBEDDERS=1 to also download and benchmark a larger
# embedding model. Off by default to keep this script fast.
COMPARE_EMBEDDERS = os.environ.get("COMPARE_EMBEDDERS") == "1"

# A test set: question -> a string that MUST appear in the retrieved chunk.
# Writing this by hand is the work. There is no shortcut, and it is
# the only way to measure retrieval honestly.
TEST_SET = [
    ("How many holiday days do I get?", "28 days"),
    ("What happens after three years of employment?", "2 extra days"),
    ("How much is the Team plan?", "£18 per user"),
    ("What is the equipment budget for new joiners?", "£2,000"),
    ("How long do I have to submit expenses?", "45 days"),
    ("Which regions can I deploy to?", "London"),
    ("Why did the cache stampede happen?", "identical TTLs"),
    ("What is the rate limit for Hobby keys?", "10 requests per second"),
    ("How long is the notice period for senior engineers?", "three months"),
    ("What databases are available?", "PostgreSQL"),
]


def split_headings(text, source):
    chunks, current = [], []
    for line in text.split("\n"):
        if line.startswith("## ") and current:
            body = "\n".join(current).strip()
            if body:
                chunks.append({"text": body, "source": source})
            current = [line]
        else:
            current.append(line)
    body = "\n".join(current).strip()
    if body:
        chunks.append({"text": body, "source": source})
    return chunks


def split_fixed(text, source, size=200):
    return [
        {"text": text[i:i + size].strip(), "source": source}
        for i in range(0, len(text), size)
        if text[i:i + size].strip()
    ]


def build(chunks, model):
    vectors = np.asarray(
        model.encode([c["text"] for c in chunks], show_progress_bar=False),
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search(index, chunks, model, question, k):
    q = np.asarray(model.encode([question]), dtype="float32")
    faiss.normalize_L2(q)
    scores, ids = index.search(q, k)
    return [chunks[i] for i in ids[0]]


def recall_at_k(index, chunks, model, k):
    """
    Fraction of questions where the answer appears in the top k chunks.

    This is THE retrieval metric. If recall is low, fixing your prompt
    or switching to a bigger model will not help.
    """
    hits = 0
    misses = []

    for question, needle in TEST_SET:
        retrieved = search(index, chunks, model, question, k)
        found = any(needle.lower() in c["text"].lower() for c in retrieved)
        if found:
            hits += 1
        else:
            misses.append((question, needle))

    return hits / len(TEST_SET), misses


def main():
    model = SentenceTransformer(MODEL)

    texts = [(p.read_text(), p.name) for p in sorted(DOCS.glob("*.md"))]

    heading_chunks = [c for t, s in texts for c in split_headings(t, s)]
    fixed_chunks = [c for t, s in texts for c in split_fixed(t, s)]

    section("1. Recall@k: does the answer even reach the model?")

    print(f"  test set: {len(TEST_SET)} questions with known answers\n")

    heading_index = build(heading_chunks, model)
    fixed_index = build(fixed_chunks, model)

    print(f"  {'strategy':<20} {'chunks':>7} {'k=1':>7} {'k=3':>7} {'k=5':>7}")
    print(f"  {'-' * 20} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")

    for name, chunks, index in (
        ("heading split", heading_chunks, heading_index),
        ("fixed 200 chars", fixed_chunks, fixed_index),
    ):
        scores = [recall_at_k(index, chunks, model, k)[0] for k in (1, 3, 5)]
        print(
            f"  {name:<20} {len(chunks):>7} "
            + " ".join(f"{s:>6.0%}" for s in scores)
        )

    print("\n  Recall@k rises with k - retrieve more, miss less.")
    print("  But every extra chunk costs tokens and adds distraction.")

    section("2. Which questions fail, and why")

    _, misses = recall_at_k(heading_index, heading_chunks, model, 3)

    if not misses:
        print("  Nothing missed at k=3 with heading chunks.")
    else:
        print(f"  {len(misses)} of {len(TEST_SET)} questions missed at k=3:\n")
        for question, needle in misses:
            print(f"    Q: {question}")
            print(f"       expected to find: {needle!r}")

            retrieved = search(heading_index, heading_chunks, model, question, 3)
            for i, chunk in enumerate(retrieved, 1):
                first = chunk['text'].split(chr(10))[0][:46]
                print(f"       got {i}: {first}")
            print()

    print("  This is the loop that improves a RAG system:")
    print("  find the failures, look at what came back instead, fix the chunking.")

    section("3. Embedding model quality matters too")

    small_recall, _ = recall_at_k(heading_index, heading_chunks, model, 3)

    print(f"  current model: all-MiniLM-L6-v2  (384 dims, ~90 MB)")
    print(f"  recall@3     : {small_recall:.0%}\n")

    if not COMPARE_EMBEDDERS:
        print("  A larger embedding model usually retrieves better, at the cost")
        print("  of speed and disk. To measure it here, re-run with:\n")
        print("      COMPARE_EMBEDDERS=1 uv run python phase-05-rag/05_rag_quality.py\n")
        print("  That downloads all-mpnet-base-v2 (~420 MB, 768 dims) and")
        print("  compares recall@3 directly. It is off by default so this")
        print("  script stays quick to run.")
    else:
        print("  Loading all-mpnet-base-v2 (~420 MB on first run)...")
        bigger = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        big_index = build(heading_chunks, bigger)
        big_recall, _ = recall_at_k(big_index, heading_chunks, bigger, 3)

        print(f"\n  {'model':<38} {'dims':>6} {'recall@3':>10}")
        print(f"  {'-' * 38} {'-' * 6} {'-' * 10}")
        print(f"  {'all-MiniLM-L6-v2 (fast)':<38} {384:>6} {small_recall:>9.0%}")
        print(f"  {'all-mpnet-base-v2 (better)':<38} {768:>6} {big_recall:>9.0%}")
        print("\n  Measure before you upgrade - on small document sets the")
        print("  difference is often not worth the extra time and memory.")

    section("4. What to fix, in order")

    print("  1. Measure recall@k first. If it is low, nothing downstream matters.")
    print("  2. Fix chunking - split on structure, keep headings, add overlap.")
    print("  3. Raise k, and check whether answers improve or just get longer.")
    print("  4. Try a stronger embedding model.")
    print("  5. Only then touch the generation prompt.")
    print("\n  Beginners do this list backwards, starting with the prompt.")

    section("5. Building your own test set")

    print("  You need 20-50 questions with known answers. To get them:")
    print("    - write questions you actually expect people to ask")
    print("    - include some the documents CANNOT answer, to test refusal")
    print("    - include a few awkward phrasings and synonyms")
    print("    - record the exact string that must appear in the right chunk")
    print("\n  It is an afternoon of boring work, and it converts")
    print("  'the RAG feels bad' into a number you can improve.")

    print("\nDone. Phase 5 complete -> phase-06-finetuning-basics/")


if __name__ == "__main__":
    main()
