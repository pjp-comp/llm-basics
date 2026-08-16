# Phase 3: Hugging Face Basics

**Goal:** Load real models and generate text — and control *how* they generate.

Phase 2 opened the model up. This phase is about using it: loading from the Hub,
the generation settings that matter, and running models locally through Ollama.

---

## Concepts

### 1. The three things you load

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
```

| Object | Job |
|---|---|
| **Tokenizer** | Text ↔ numbers |
| **Model** | Numbers → predictions |
| **Config** | The model's shape: layers, hidden size, vocab |

**The tokenizer and model must match.** They're trained together; mixing them
gives nonsense with no error message.

The `Auto` classes read the model's config and pick the right implementation,
so the same two lines work for Llama, Qwen, Mistral, and the rest.

### 2. Where models come from

`from_pretrained()` downloads from the [Hugging Face Hub](https://huggingface.co/models)
and caches in `~/.cache/huggingface/`. Downloaded once, reused forever.

Model ids look like `organization/model-name`. Look for:

- **Size** — `135M`, `1.7B`, `8B`. Rough memory need: 2 bytes × parameters.
- **`-Instruct` / `-Chat`** — trained to follow instructions. Without it you get
  a base model that only continues text.
- **License** — some need approval before download.

### 3. Generation is a loop

The model predicts *one* token. To get a sentence you feed the output back in
and predict again:

```
"The cat" → predict " sat" → "The cat sat" → predict " on" → ...
```

`model.generate()` does this loop for you. `02_generate.py` in Phase 2's folder
history showed it manually — `01_load_and_generate.py` here shows both.

### 4. Generation settings

These are the knobs you'll actually turn:

| Setting | What it does | Typical |
|---|---|---|
| `max_new_tokens` | How many tokens to generate | 50–500 |
| `do_sample` | `False` = always pick the top token | `True` for chat |
| `temperature` | Randomness. Low = focused, high = wild | 0.7 |
| `top_p` | Only consider tokens in the top X% of probability | 0.9 |
| `repetition_penalty` | Discourage repeating | 1.1 |

**Greedy** (`do_sample=False`) is deterministic — same input, same output every
time. Good for tests, but it can fall into loops.

**Sampling** (`do_sample=True`) picks randomly, weighted by probability. More
natural, different every run.

### 5. Temperature, concretely

Temperature divides the logits before softmax:

- `temperature=0.1` → the top token dominates → predictable, repetitive
- `temperature=1.0` → the model's natural distribution
- `temperature=2.0` → probabilities flatten → creative, then incoherent

### 6. Two ways to run a model

| | HF `transformers` | Ollama |
|---|---|---|
| Start-up | Slower, downloads weights | Fast, already local |
| Control | Full — every internal tensor | Just the API |
| Fine-tuning | Yes (Phase 8+) | No |
| Best for | Learning internals, training | Running and testing quickly |

**Rule of thumb:** Ollama to *use* a model, `transformers` to *understand or
train* one. You'll use both throughout this repo.

### 7. Reasoning models return an empty answer

Some models — `qwen3`, DeepSeek-R1 and similar — think first into a hidden
`thinking` field, then write the real answer into `content`.

With a small token budget, the model can spend all of it thinking and return an
**empty** `content`. Not a bug: raise the limit, or use a non-reasoning model.
`02_ollama_basics.py` detects this and tells you.

---

## Examples

```bash
uv run python phase-03-huggingface-basics/01_load_and_generate.py
uv run python phase-03-huggingface-basics/02_ollama_basics.py
```

| File | What it shows | Needs |
|---|---|---|
| `01_load_and_generate.py` | Loading, greedy vs sampling, temperature, the manual loop | cached model |
| `02_ollama_basics.py` | Chat, system prompts, streaming, history, embeddings | Ollama running |

### What to look for

**`01_load_and_generate.py`** runs greedy twice to prove it's identical, then
sampling twice to prove it isn't. The temperature ladder (0.1 / 0.7 / 1.5) shows
coherence degrading as you turn it up. The last section generates 5 tokens by
hand so the loop is visible.

**`02_ollama_basics.py`** shows the same system prompt turning a model into a
pirate or an academic, streaming output arriving token by token, and cosine
similarity between embeddings — `"I love cats"` vs `"I adore kittens"` scores
about 0.86, versus 0.31 for an unrelated sentence. That gap is what makes search
and RAG work.

---

## Using the tools

```bash
# tokens/sec on your hardware
uv run python tools/inspect_model.py --speed

# the same prompt across five temperatures
uv run python tools/inspect_model.py --temps "Once upon a time"

# small local model vs a bigger Ollama one
uv run python tools/inspect_model.py --compare "Explain gravity" --ollama llama3.2:3b

# two Ollama models on a set of questions
uv run python tools/evaluate.py --a llama3.2:3b --b qwen3:1.7b
```

`evaluate.py` is worth getting used to now. It's the same side-by-side comparison
you'll need in Phase 12 to prove a fine-tuned model actually beat the original.
See [`tools/README.md`](../tools/README.md).

---

## Try it yourself

1. In `01_load_and_generate.py`, set `TEMPERATURE = 2.0` and watch it fall apart.
2. Change `MODEL` to `"gpt2"`. It's a base model — notice it rambles instead of answering.
3. In `02_ollama_basics.py`, change `MODEL` to `llama3.1:8b` and compare quality and speed.
4. Switch it to `qwen3:1.7b` to see the reasoning-model behaviour first-hand.
5. Run `tools/evaluate.py` with your own questions file.

---

## Done when

You can:

- load a model and tokenizer and generate a response
- explain the difference between the model and the tokenizer
- say what `temperature` and `top_p` do
- explain why greedy decoding gives identical output each run
- decide when to reach for Ollama vs `transformers`

---

## Common problems

| Problem | Cause | Fix |
|---|---|---|
| Download is slow | First run only | It's cached afterwards |
| `ConnectionError` from Ollama | Not running | `ollama serve` |
| `model not found` | Not pulled | `ollama pull llama3.2:3b` |
| Empty Ollama response | Reasoning model | Raise `num_predict`, or use `llama3.2:3b` |
| Output repeats forever | Greedy loop | Use `do_sample=True` or `repetition_penalty=1.1` |
| Model rambles, won't answer | Base model, not instruct | Use the `-Instruct` version |
| `expected all tensors on same device` | Model and inputs on different devices | `.to(device)` on both |
| Out of memory | Model too big | Use a smaller model, or run on Colab |

---

## References

**Generation**
- [Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies) — the main reference
- [How to generate text](https://huggingface.co/blog/how-to-generate) — greedy, beam, top-k, top-p compared
- [`GenerationConfig` API](https://huggingface.co/docs/transformers/main_classes/text_generation) — every parameter
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — the paper that introduced top-p

**Hugging Face**
- [HF LLM Course](https://huggingface.co/learn/llm-course) — official, free
- [Loading models](https://huggingface.co/docs/transformers/main/en/models)
- [Model Hub](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) — browse by size and task
- [Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) — the one-liner API

**Ollama**
- [Ollama Python library](https://github.com/ollama/ollama-python)
- [Ollama API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Model library](https://ollama.com/library) — what you can pull
- [Valid options](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values) — `num_predict`, `temperature`, etc.

**Embeddings**
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text) — the model used in the example
- [Sentence Transformers](https://www.sbert.net/) — the standard library, used in Phase 5

**Next:** [Phase 4 — Chat Templates](../phase-04-chat-templates/) — format prompts the way models expect.
