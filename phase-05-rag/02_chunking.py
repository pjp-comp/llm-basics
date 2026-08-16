"""
Phase 5 - Example 2: Chunking

You cannot embed a whole document usefully - you have to cut it up.
HOW you cut it decides whether retrieval works. This is the part
beginners skip, and it is the part that breaks their RAG system.

Run:
    uv run python phase-05-rag/02_chunking.py

Reference:
    https://www.pinecone.io/learn/chunking-strategies/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from llmkit import section  # noqa: E402

DOCS = Path(__file__).parent / "docs"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# chunking strategies
# ---------------------------------------------------------------------------


def chunk_fixed(text, size=200, overlap=0):
    """
    Cut every `size` characters. Simple, and it cuts through the middle
    of words and sentences - which is exactly the problem.
    """
    chunks = []
    step = size - overlap
    for start in range(0, len(text), step):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(piece)
    return chunks


def chunk_paragraphs(text):
    """Split on blank lines. Respects the author's own structure."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_headings(text):
    """
    Split on markdown headings, keeping the heading with its content.

    Usually the best option for structured documents, because a heading
    tells you what its section is about - useful context to keep attached.
    """
    chunks, current = [], []

    for line in text.split("\n"):
        if line.startswith("## ") and current:
            chunks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c]


def main():
    text = (DOCS / "company_handbook.md").read_text()

    print(f"Document: company_handbook.md")
    print(f"Length: {len(text)} characters")

    section("1. Why not embed the whole document?")

    print(f"  The document is {len(text)} characters.")
    print("  Embedding it gives ONE vector for everything: holidays,")
    print("  expenses, notice periods, equipment - all averaged together.")
    print("\n  Ask 'how many holiday days?' and that vector is a blurry")
    print("  average of every topic. It matches everything and nothing.")
    print("\n  So we cut the document into focused pieces.")

    section("2. Fixed-size chunks: simple and bad")

    fixed = chunk_fixed(text, size=200)
    print(f"  {len(fixed)} chunks of 200 characters\n")

    for i, chunk in enumerate(fixed[:3]):
        preview = chunk.replace("\n", " ")
        print(f"  chunk {i}: {preview[:95]}...")

    print("\n  Look at where these end - mid-sentence, mid-word.")
    print("  A chunk that starts halfway through a thought is hard to")
    print("  match against a question, and confusing when shown to the model.")

    section("3. Overlap softens the damage")

    no_overlap = chunk_fixed(text, size=200, overlap=0)
    with_overlap = chunk_fixed(text, size=200, overlap=50)

    print(f"  without overlap: {len(no_overlap)} chunks")
    print(f"  with 50-char overlap: {len(with_overlap)} chunks")
    print("\n  Overlap repeats the end of one chunk at the start of the next,")
    print("  so a sentence split down the middle still appears whole somewhere.")
    print("  Costs more storage. Usually worth it.")

    section("4. Paragraph chunks: respect the structure")

    paras = chunk_paragraphs(text)
    print(f"  {len(paras)} chunks\n")
    for i, chunk in enumerate(paras[:3]):
        preview = chunk.replace("\n", " ")
        print(f"  chunk {i}: {preview[:95]}...")

    print("\n  These end where the author ended a thought. Much better.")

    section("5. Heading chunks: best for structured docs")

    sections = chunk_headings(text)
    print(f"  {len(sections)} chunks\n")

    print(f"  {'chunk':>5}  {'chars':>6}  first line")
    print(f"  {'-' * 5}  {'-' * 6}  {'-' * 40}")
    for i, chunk in enumerate(sections):
        first = chunk.split("\n")[0][:40]
        print(f"  {i:>5}  {len(chunk):>6}  {first}")

    print("\n  Each chunk is one topic, and carries its own heading.")

    section("6. Does it actually change the answer?")

    print("  Loading embedding model...\n")
    model = SentenceTransformer(MODEL)

    question = "How many holiday days do I get after 4 years?"
    q_vec = model.encode(question)

    print(f"  question: {question!r}\n")

    for name, chunks in (
        ("fixed 200 chars", chunk_fixed(text, 200)),
        ("paragraphs", chunk_paragraphs(text)),
        ("headings", chunk_headings(text)),
    ):
        vectors = model.encode(chunks)
        scores = [cosine(q_vec, v) for v in vectors]
        best = int(np.argmax(scores))

        preview = chunks[best].replace("\n", " ")[:78]
        correct = "33" in chunks[best] or "three years" in chunks[best]

        print(f"  --- {name} ---")
        print(f"  best score : {scores[best]:.3f}")
        print(f"  retrieved  : {preview}...")
        print(f"  has answer : {'YES' if correct else 'NO'}\n")

    print("  The answer needs BOTH the '28 days' rule and the '+2 after")
    print("  three years' rule. A strategy that separates them loses.")
    print("\n  Look closely: the heading strategy scored LOWER than the")
    print("  others, and was still the only one that found the answer.")
    print("\n  Similarity score measures 'does this look related', not")
    print("  'does this contain the answer'. A high score on a chunk that")
    print("  got cut before the key sentence is confidently useless.")
    print("  Judge retrieval by whether the answer is there - not by the score.")

    section("7. Rules of thumb")

    print("  - Split on structure (headings, paragraphs) before falling back to size")
    print("  - 200-500 words per chunk suits most documents")
    print("  - Add 10-20% overlap when using fixed sizes")
    print("  - Keep headings attached to their content")
    print("  - Never split a table or code block down the middle")
    print("\n  And the one that matters most:")
    print("  READ your chunks. Print 20 of them. If they read as nonsense")
    print("  to you, they are nonsense to the retriever too.")

    print("\nDone. Next: 03_vector_store.py")


if __name__ == "__main__":
    main()
