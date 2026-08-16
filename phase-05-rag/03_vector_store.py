"""
Phase 5 - Example 3: Vector stores

Comparing a question against every chunk works fine for 100 chunks
and falls apart at 100,000. A vector store makes search fast, and
lets you save the index instead of re-embedding every time.

Run:
    uv run python phase-05-rag/03_vector_store.py

Reference:
    https://github.com/facebookresearch/faiss/wiki
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from llmkit import Timer, section  # noqa: E402

DOCS = Path(__file__).parent / "docs"
INDEX_DIR = Path(__file__).parent / ".index"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_headings(text, source):
    """Split markdown on ## headings. Keep the source filename with each chunk."""
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


def load_chunks():
    chunks = []
    for path in sorted(DOCS.glob("*.md")):
        chunks.extend(chunk_headings(path.read_text(), path.name))
    return chunks


def main():
    section("1. Load and chunk every document")

    chunks = load_chunks()
    by_source = {}
    for c in chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1

    print(f"  {len(chunks)} chunks from {len(by_source)} files\n")
    for source, count in by_source.items():
        print(f"    {source:<28} {count:>3} chunks")

    section("2. Embed them")

    model = SentenceTransformer(MODEL)

    with Timer("embedding", quiet=True) as t:
        vectors = model.encode([c["text"] for c in chunks], show_progress_bar=False)

    vectors = np.asarray(vectors, dtype="float32")
    print(f"  {vectors.shape[0]} chunks -> {vectors.shape[1]} numbers each")
    print(f"  took {t.seconds:.2f}s")
    print(f"  memory: {vectors.nbytes / 1024:.0f} KB")

    section("3. Build a FAISS index")

    # Normalising to unit length turns inner product into cosine similarity.
    # This is the standard trick: FAISS has no cosine index, but normalised
    # vectors + IndexFlatIP gives exactly that.
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print(f"  index type : IndexFlatIP (exact search, inner product)")
    print(f"  vectors    : {index.ntotal}")
    print("\n  'Flat' means it checks every vector - exact, but linear.")
    print("  Above ~1M vectors you'd switch to an approximate index (IVF, HNSW)")
    print("  which is faster but can miss results. Flat is right for learning.")

    section("4. Search")

    def search(question, k=3):
        q = np.asarray(model.encode([question]), dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = index.search(q, k)
        return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0])]

    for question in (
        "How much holiday do I get?",
        "What happens when a deployment fails and I need the old version?",
        "Why did the cache break?",
    ):
        print(f"\n  Q: {question}")
        for rank, (score, chunk) in enumerate(search(question), 1):
            first_line = chunk["text"].split("\n")[0][:52]
            print(f"    {rank}. [{score:.3f}] {chunk['source']:<24} {first_line}")

    print("\n  Each question found chunks from the RIGHT document,")
    print("  without us telling it which file to look in.")

    section("5. Saving and loading")

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "docs.faiss"))
    (INDEX_DIR / "chunks.pkl").write_bytes(pickle.dumps(chunks))

    size_kb = (INDEX_DIR / "docs.faiss").stat().st_size / 1024
    print(f"  saved index  -> {INDEX_DIR.name}/docs.faiss ({size_kb:.0f} KB)")
    print(f"  saved chunks -> {INDEX_DIR.name}/chunks.pkl")

    with Timer("reload", quiet=True) as t:
        reloaded = faiss.read_index(str(INDEX_DIR / "docs.faiss"))

    print(f"\n  reloaded {reloaded.ntotal} vectors in {t.seconds * 1000:.0f} ms")
    print("  No re-embedding needed. Embed once, search forever.")
    print("  Re-embed only when your documents change.")

    section("6. Speed: does the index actually help?")

    q = np.asarray(model.encode(["holiday allowance"]), dtype="float32")
    faiss.normalize_L2(q)

    with Timer("brute force", quiet=True) as t1:
        for _ in range(100):
            _ = vectors @ q[0]  # compare against everything, by hand

    with Timer("faiss", quiet=True) as t2:
        for _ in range(100):
            index.search(q, 3)

    print(f"  100 searches by hand : {t1.seconds * 1000:6.1f} ms")
    print(f"  100 searches in FAISS: {t2.seconds * 1000:6.1f} ms")
    print(f"\n  At {len(chunks)} chunks the difference is small - FAISS may even")
    print("  look slower, because there is real overhead per call.")
    print("  The gap appears at 100k+ vectors, which is what it is built for.")

    print("\nDone. Next: 04_rag_pipeline.py")


if __name__ == "__main__":
    main()
