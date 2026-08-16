# Phase 5: RAG (Retrieval-Augmented Generation)

**Goal:** Let a model answer questions about your own documents — without
training it at all.

This is the phase most people skip on the way to fine-tuning, and it's the one
that solves the problem they actually had.

---

## The idea in four steps

```
1. Split your documents into chunks
2. Turn each chunk into a vector (embedding) and store it
3. When a question arrives, find the chunks closest to it
4. Paste those chunks into the prompt and ask the model to answer from them
```

No training. No GPU. Update a document and the answers update immediately.

---

## Concepts

### 1. Embeddings

An embedding turns text into a list of numbers positioned so that similar
meanings land near each other. `all-MiniLM-L6-v2` uses 384 numbers per text.

The key property: `"I love cats"` and `"Felines are wonderful"` score **0.65**
similar despite sharing no words. Keyword search would score zero. Embeddings
match *meaning*, not spelling.

### 2. Cosine similarity

How we compare two vectors — the angle between them, not the distance:

| Score | Meaning |
|---|---|
| 1.0 | Same direction (same meaning) |
| ~0.6+ | Related |
| ~0.0 | Unrelated |

We use the angle because vector *length* tends to reflect text length rather
than meaning.

### 3. Chunking — the part that decides everything

You can't embed a whole document usefully. One vector for a 20-page handbook is
a blurry average of every topic in it; it matches everything and nothing.

So you cut it up. **How** you cut it is the single biggest quality factor:

| Strategy | How | Quality |
|---|---|---|
| Fixed size | Every N characters | Poor — cuts mid-sentence |
| Fixed + overlap | N characters, repeating the last M | Better |
| Paragraphs | Split on blank lines | Good |
| Headings | Split on `##`, keep heading attached | Best for structured docs |

In `02_chunking.py` you'll see heading-split retrieve the correct answer while
fixed-size retrieval misses it — on the same document, same question.

### 4. Vector stores

Comparing a question against every chunk is fine for hundreds and hopeless for
millions. A vector store indexes them for fast search. We use **FAISS** — local,
free, no server.

One trick worth knowing: FAISS has no cosine index. Normalise your vectors to
unit length and use `IndexFlatIP` (inner product) — that *is* cosine similarity.

### 5. The retrieval → generation loop

```python
chunks = retrieve(question, k=3)       # find relevant text
prompt = f"Context:\n{chunks}\n\nQuestion: {question}"
answer = model(prompt)                 # answer from that text
```

The instruction *"answer using ONLY the context, otherwise say you don't know"*
matters a lot. Without it, the model falls back on invented knowledge.

### 6. Measuring it: recall@k

**Most bad RAG systems are bad at retrieval, not generation.** If the right chunk
never reaches the model, no model can rescue the answer.

**Recall@k** = the fraction of test questions where the answer appears in the top
k retrieved chunks. Measured in `05_rag_quality.py`:

| Strategy | recall@1 | recall@3 | recall@5 |
|---|---|---|---|
| Heading split | 80% | 90% | 90% |
| Fixed 200 chars | 40% | 60% | 70% |

Same documents, same questions, same embedding model. The only difference is
how the text was cut.

### 7. RAG vs fine-tuning

> **RAG supplies knowledge. Fine-tuning changes behaviour.**

| You want | Use |
|---|---|
| The model to know your documents | RAG |
| Facts that change often | RAG |
| To cite sources | RAG |
| A specific output format or tone | Fine-tuning |
| A task the base model does poorly | Fine-tuning |

Trying to teach facts by fine-tuning is the most expensive beginner mistake.
The model learns what your answers *look like*, then invents details confidently.

---

## Examples

```bash
uv run python phase-05-rag/01_embeddings.py
uv run python phase-05-rag/02_chunking.py
uv run python phase-05-rag/03_vector_store.py
uv run python phase-05-rag/04_rag_pipeline.py
uv run python phase-05-rag/05_rag_quality.py
```

| File | What it shows | Needs |
|---|---|---|
| `01_embeddings.py` | Embeddings, cosine similarity, ranking | downloads ~90 MB |
| `02_chunking.py` | Four strategies, and which finds the answer | cached model |
| `03_vector_store.py` | FAISS index, search, save/load | cached model |
| `04_rag_pipeline.py` | The complete system, end to end | Ollama |
| `05_rag_quality.py` | recall@k, failure analysis | cached model |

The [`docs/`](docs/) folder has three invented documents — a company handbook, a
product FAQ, and engineering notes. They're fictional on purpose: the model
cannot possibly know them from pretraining, so any correct answer must have come
from retrieval.

### What to look for

**`01_embeddings.py`** — `"The bank raised interest rates"` scores 0.298 against
`"I sat on the river bank"` but 0.618 against `"The central bank hiked borrowing
costs"`. Same word, different meaning, and the embedding knows.

**`02_chunking.py`** — the important result: heading chunks score **lower** (0.571
vs 0.659) yet are the only strategy that retrieves the answer. A high similarity
score on a chunk that got cut before the key sentence is confidently useless.

**`04_rag_pipeline.py`** — asks the same question with and without RAG. Without,
the model admits it doesn't know. With, it answers correctly and names the source
file. It also refuses questions the documents can't answer.

**`05_rag_quality.py`** — the recall table above, plus which specific question
failed and what came back instead. That loop is how you improve a real system.

---

## Try it yourself

1. **Add your own documents.** Drop any `.md` file into `docs/` and re-run
   `04_rag_pipeline.py`. It picks up new files automatically.
2. In `02_chunking.py`, change `chunk_fixed(text, size=200)` to `size=600`.
   Does bigger help?
3. In `04_rag_pipeline.py`, delete the *"say I don't know"* line from `PROMPT`
   and ask an unanswerable question. Watch it start inventing.
4. In `05_rag_quality.py`, add 5 of your own questions to `TEST_SET`.
5. Run the embedding comparison:
   `COMPARE_EMBEDDERS=1 uv run python phase-05-rag/05_rag_quality.py`

---

## Done when

You can:

- explain the four RAG steps without notes
- say why chunking strategy matters more than model choice
- explain what recall@k measures and why to measure it before anything else
- decide correctly between RAG and fine-tuning for a given problem
- build a small Q&A system over a folder of your own documents

---

## Common problems

| Problem | Cause | Fix |
|---|---|---|
| Answers are wrong but confident | Retrieval missed; model guessed | Measure recall@k; add the "don't know" instruction |
| Retrieved chunks look irrelevant | Bad chunking | Split on headings, not fixed size |
| Answer is cut off mid-fact | Chunk boundary split the fact | Add overlap, or chunk on structure |
| Everything scores ~0.9 | Chunks too similar or too short | Longer, more distinct chunks |
| Slow with many documents | Re-embedding every run | Save the FAISS index and reload it |
| Model ignores the context | Weak prompt | Say "use ONLY the context" explicitly |
| Works for some questions only | Vocabulary mismatch | Try a stronger embedding model |

---

## Where to go next in RAG

This phase teaches the fundamentals with the simplest possible tools. Real
systems add:

- **Hybrid search** — combine embeddings with keyword search (BM25). Embeddings
  miss exact identifiers like error codes and product SKUs; keywords catch those.
- **Reranking** — retrieve 20 chunks with a fast model, then reorder with a
  slower, more accurate cross-encoder.
- **Query rewriting** — expand or clarify the question before searching.
- **Frameworks** — LangChain and LlamaIndex package all of this. Learn the
  mechanics first (this phase), then use them if you want the convenience.

---

## References

**Embeddings**
- [Sentence Transformers docs](https://www.sbert.net/) — the library used here
- [Pretrained models](https://www.sbert.net/docs/pretrained_models.html) — how to pick one
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — embedding models ranked
- [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — where embeddings came from

**Chunking and retrieval**
- [Pinecone — Chunking strategies](https://www.pinecone.io/learn/chunking-strategies/)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) — index types explained
- [FAISS with `datasets`](https://huggingface.co/docs/datasets/faiss_es)
- [BM25 explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — the keyword half of hybrid search

**RAG**
- [Lewis et al. — Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) — the original 2020 paper
- [Ragas](https://docs.ragas.io/en/stable/concepts/metrics/) — proper RAG evaluation metrics
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — a strong modern improvement
- [LlamaIndex](https://docs.llamaindex.ai/) / [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/) — frameworks

**Next:** [Phase 6 — Fine-tuning Basics](../phase-06-finetuning-basics/) — when RAG isn't the answer.
