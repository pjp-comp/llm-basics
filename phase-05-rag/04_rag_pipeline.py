"""
Phase 5 - Example 4: The full RAG pipeline

Everything joined up: chunk -> embed -> store -> retrieve -> generate.

This shows the model answering questions about documents it has never
seen, with no training at all.

Needs Ollama:  ollama pull llama3.2:3b

Run:
    uv run python phase-05-rag/04_rag_pipeline.py

Reference:
    https://huggingface.co/docs/datasets/faiss_es
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import ollama  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from llmkit import section  # noqa: E402

DOCS = Path(__file__).parent / "docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama3.2:3b"

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know based on the
documents I have." Do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


class TinyRAG:
    """A complete RAG system. Small enough to read in one sitting."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks = []
        self.index = None

    # -- indexing ----------------------------------------------------------

    def add_documents(self, folder):
        for path in sorted(Path(folder).glob("*.md")):
            self.chunks.extend(self._split(path.read_text(), path.name))

        vectors = np.asarray(
            self.model.encode([c["text"] for c in self.chunks], show_progress_bar=False),
            dtype="float32",
        )
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return len(self.chunks)

    @staticmethod
    def _split(text, source):
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

    # -- retrieval ---------------------------------------------------------

    def retrieve(self, question, k=3, min_score=0.0):
        q = np.asarray(self.model.encode([question]), dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, k)

        return [
            (float(s), self.chunks[i])
            for s, i in zip(scores[0], ids[0])
            if s >= min_score
        ]

    # -- generation --------------------------------------------------------

    def answer(self, question, k=3, show_context=False):
        hits = self.retrieve(question, k)

        if not hits:
            return "No relevant documents found.", []

        context = "\n\n---\n\n".join(
            f"[from {c['source']}]\n{c['text']}" for _, c in hits
        )

        if show_context:
            print(f"\n  Context sent to the model ({len(context)} chars):")
            for score, chunk in hits:
                first = chunk["text"].split("\n")[0][:50]
                print(f"    [{score:.3f}] {chunk['source']:<24} {first}")

        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(context=context, question=question),
                }
            ],
            options={"temperature": 0.0, "num_predict": 200},
        )

        return (response["message"].get("content") or "").strip(), hits


def main():
    try:
        ollama.list()
    except Exception:
        print(f"Cannot reach Ollama. Start it:  ollama serve")
        print(f"Then:  ollama pull {CHAT_MODEL}")
        sys.exit(1)

    section("Building the index")

    rag = TinyRAG()
    n = rag.add_documents(DOCS)
    print(f"  indexed {n} chunks from {len(list(DOCS.glob('*.md')))} documents")

    section("1. Without RAG: the model has never seen these documents")

    question = "How many holiday days do Nimbus employees get?"

    plain = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": question}],
        options={"temperature": 0.0, "num_predict": 100},
    )
    print(f"  Q: {question}")
    print(f"\n  Plain model:")
    print(f"    {(plain['message'].get('content') or '').strip()[:280]}")
    print("\n  It cannot know - Nimbus is invented. Notice whether it")
    print("  admits that, or invents a confident number anyway.")

    section("2. With RAG: same question, retrieved context")

    answer, _ = rag.answer(question, show_context=True)
    print(f"\n  Answer:")
    print(f"    {answer[:300]}")

    section("3. More questions across different documents")

    for q in (
        "What does the Team plan cost and what do I get?",
        "Why did the cache stampede happen and how was it fixed?",
        "Can I work from Spain?",
    ):
        answer, hits = rag.answer(q)
        sources = ", ".join(sorted({c["source"] for _, c in hits}))
        print(f"\n  Q: {q}")
        print(f"  A: {answer[:260]}")
        print(f"     (sources: {sources})")

    section("4. Questions the documents cannot answer")

    # A good RAG system says "I don't know" instead of inventing.
    for q in (
        "What is the CEO's home address?",
        "How do I train a neural network?",
    ):
        answer, hits = rag.answer(q)
        best = max((s for s, _ in hits), default=0.0)
        print(f"\n  Q: {q}")
        print(f"  A: {answer[:200]}")
        print(f"     (best match scored {best:.3f})")

    print("\n  The prompt told the model to refuse when the context")
    print("  lacks the answer. Without that instruction it would guess.")

    section("5. How many chunks should you retrieve?")

    question = "What are the deployment and build limits on each plan?"

    for k in (1, 3, 5):
        answer, hits = rag.answer(question, k=k)
        chars = sum(len(c["text"]) for _, c in hits)
        print(f"\n  k={k}  ({chars} chars of context)")
        print(f"    {answer[:200]}")

    print("\n  Too few: the answer may be missing.")
    print("  Too many: the real answer gets buried in noise, and you pay")
    print("  for more tokens. k=3 to 5 is a reasonable default.")

    section("6. What RAG did and did not do")

    print("  Did:")
    print("    - answered questions about documents the model never saw")
    print("    - cited which file each answer came from")
    print("    - refused when the answer was not in the documents")
    print("    - updates instantly if you edit a file and re-index")
    print("\n  Did NOT:")
    print("    - change the model at all - no training happened")
    print("    - teach the model a new skill or writing style")
    print("\n  That second list is what fine-tuning is for. RAG supplies")
    print("  knowledge; fine-tuning changes behaviour. See Phase 6.")

    print("\nDone. Next: 05_rag_quality.py")


if __name__ == "__main__":
    main()
