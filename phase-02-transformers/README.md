# Phase 2: NLP and Transformer Basics

**Goal:** Understand how text becomes numbers, and what happens inside the model.

This phase is about the *architecture*. Phase 3 is about actually running models
and generating text — here you open the box and look.

---

## Concepts

### 1. Tokenization

Models can't read text. A **tokenizer** splits text into pieces and maps each
to an ID number.

```
"Hello world" → ["Hello", " world"] → [19556, 1002]
```

Things that surprise people:

- A token is roughly ¾ of a word in English, not one word
- The leading space is part of the token — `"cat"` and `" cat"` are different IDs
- Rare words split into several pieces
- Other languages often need far more tokens for the same meaning
- One emoji can be several tokens, none of which is valid text alone

This matters because you pay — in memory, time, and money — per token.

### 2. Embeddings

Each token ID becomes a list of numbers (a **vector**) that captures meaning.
Similar words land near each other in that space.

A 135M model uses 576 numbers per token. Large ones use 4096+.

The lookup table has shape `(vocab_size, hidden_size)` — one row per possible
token. For SmolLM2 that's `(49152, 576)`, about 28 million numbers just to
represent the vocabulary.

### 3. Positional encoding

Attention alone has no sense of order — `"dog bites man"` and `"man bites dog"`
would look identical. Positional information is added so the model knows where
each token sits.

### 4. Attention

The core transformer idea: when processing a word, look at every other word and
decide which ones matter.

In *"The animal didn't cross the street because **it** was too tired"* — attention
is how the model connects "it" back to "animal".

Three vectors per token, which is where Q, K, V come from:

| Name | Meaning |
|---|---|
| **Q** (query) | "what am I looking for?" |
| **K** (key) | "what do I offer?" |
| **V** (value) | "what do I actually contribute?" |

Compare every query against every key to get attention scores, then use those
scores to blend the values. **Multi-head** attention runs this several times in
parallel, so different heads can track different relationships.

### 5. Stacking layers

One attention block isn't enough. Models stack many — SmolLM2-135M has 30 —
and each one refines the same-shaped vectors:

```
(1, 5, 576) → layer 1 → (1, 5, 576) → layer 2 → ... → (1, 5, 576)
```

The shape never changes. Only the *meaning* held in those numbers does.

### 6. Logits

The final layer produces one score per vocabulary token — the **logits**.
Softmax turns those scores into probabilities that sum to 1.

For SmolLM2 that's 49,152 numbers, at every position, saying how likely each
possible next token is.

### 7. Model families

| Type | Reads | Good at | Examples |
|---|---|---|---|
| Decoder | Left to right | Generating text | GPT, Llama, Qwen, SmolLM |
| Encoder | Both directions | Classifying, embeddings | BERT |
| Encoder-decoder | Both, then generates | Translating, summarizing | T5 |

This repo uses **decoder** models — that's what "LLM" usually means.

---

## Examples

```bash
uv run python phase-02-transformers/01_tokenizer.py
uv run python phase-02-transformers/02_inside_model.py
```

| File | What it shows | Needs |
|---|---|---|
| `01_tokenizer.py` | Text → tokens → IDs → back again | downloads ~2 MB |
| `02_inside_model.py` | Embeddings, layers, logits, temperature | downloads ~270 MB |

First run downloads [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
and caches it in `~/.cache/huggingface/`. Later runs are instant.

### What to look for

**`01_tokenizer.py`** — compare token counts for `"cat"`, `" cat"`, and
`"antidisestablishmentarianism"`. Watch how Devanagari text costs 6 tokens for
one short word. See what padding and attention masks actually contain.

**`02_inside_model.py`** — follows `"The capital of France is"` through every
stage, printing real shapes. It ends with the top 10 predictions: `" Paris"` at
around 44%. That number is the model's actual belief, not an illustration.

---

## Using the tools

Instead of editing the examples, inspect any model directly:

```bash
# how does this text tokenize?
uv run python tools/inspect_model.py --tokens "Your text here"

# what does the model predict next, with probabilities?
uv run python tools/inspect_model.py --predict "The capital of France is"

# model size, layers, vocab, chat template
uv run python tools/inspect_model.py --info

# try a different model
uv run python tools/inspect_model.py --info --model gpt2
```

Comparing `--info` across `gpt2` and SmolLM2 is a fast way to see how
architectures differ. See [`tools/README.md`](../tools/README.md).

---

## Try it yourself

1. In `01_tokenizer.py`, add your own name to `WORDS`. Does it split?
2. Add a long sentence in a non-English language and compare token counts.
3. In `02_inside_model.py`, change the prompt to `"The capital of Japan is"`.
   Does the model know?
4. Try `"2 + 2 ="` and look at the top 10. Small models are bad at maths — you'll
   see the probability spread across many wrong answers.

---

## Done when

You can:

- explain why `"cat"` and `" cat"` are different tokens
- say what Q, K, and V are for
- read a shape like `(1, 7, 576)` and say what each number means
- explain what logits are and what softmax does to them
- describe why the shape doesn't change as data moves through layers

---

## Common problems

| Problem | Fix |
|---|---|
| Download is slow | Normal on first run only; it's cached afterwards |
| Warning about `HF_TOKEN` | Harmless — a rate-limit notice for anonymous downloads |
| `OSError: model not found` | Check the model id spelling against huggingface.co |
| Emoji prints as `�` | Correct — one token is only part of the character |

---

## References

**Tokenization**
- [HF — Tokenizers summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Tiktokenizer](https://tiktokenizer.vercel.app/) — paste text, watch it tokenize live
- [Karpathy — Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — 2 hours, worth it
- [Byte-Pair Encoding explained](https://huggingface.co/learn/nlp-course/chapter6/5)

**Transformers**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the clearest visual explanation
- [Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — you'll implement this in Phase 15
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the 2017 paper that started it
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — the paper, line by line, in code
- [3Blue1Brown — Attention in transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) — excellent animation

**Embeddings**
- [HF NLP Course, ch. 1–2](https://huggingface.co/learn/nlp-course)
- [Word2Vec explained](https://jalammar.github.io/illustrated-word2vec/) — where embeddings came from

**Reference**
- [`AutoTokenizer`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer)
- [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)
- [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

**Next:** [Phase 3 — Hugging Face Basics](../phase-03-huggingface-basics/) — run models and generate text.
