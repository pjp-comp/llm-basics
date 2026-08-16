# Phase 6: Fine-Tuning Basics

**Goal:** Understand the difference between *using* a model and *changing* it —
and be honest about whether you need to.

No large training runs here. This phase is about the concepts and the decision.
The real training starts in Phase 8.

---

## Concepts

### 1. What fine-tuning actually is

A model is a pile of numbers — 134 million of them in SmolLM2-135M. Those numbers
*are* the model. **Fine-tuning means changing some of them.**

The loop is identical to Phase 1:

```
forward pass → loss → backward pass → optimizer step → zero gradients
```

Nothing new. Only the model got bigger.

### 2. Loss, for text

For a language model, loss measures **how surprised the model was by the token
that actually came next.** Lower = it expected that text.

From `02_what_training_does.py`, on the base model:

| Text | Loss |
|---|---|
| `The capital of France is Paris` | 3.24 |
| `The capital of France is Berlin` | 4.91 |
| `The capital of France is banana` | 6.40 |

Training pushes loss down on *your* examples — which means making your text the
thing the model expects to produce.

### 3. Pretraining vs fine-tuning

| | Pretraining | Fine-tuning |
|---|---|---|
| Data | Trillions of tokens | Hundreds to thousands of examples |
| Cost | Millions of dollars | Free to a few dollars |
| Who does it | Meta, Google, Alibaba | You |
| Teaches | Language itself | Your task or style |

You will never pretrain. Someone already spent the millions; you adapt the result.

### 4. Behaviour vs knowledge — the decision that matters

> **Fine-tuning teaches behaviour. RAG supplies knowledge.**

| You want | Use |
|---|---|
| A consistent output format | Fine-tune |
| A specific tone or persona | Fine-tune |
| A task the base model does badly | Fine-tune |
| The model to know your documents | **RAG** (Phase 5) |
| Facts that change weekly | **RAG** |

Fine-tuning to inject facts is the most expensive beginner mistake. The model
learns what your answers *look like*, then confidently invents the details.

### 5. Try the cheap options first

Each step below costs roughly 10× the one above it:

1. **Write a clearer prompt** — minutes, free
2. **Add 2–5 few-shot examples** — minutes, costs tokens on every call
3. **Use RAG** — hours, no training
4. **Fine-tune** — days, needs data and a GPU

Most projects stop at step 2.

But there's a real limit to prompting, and `01_when_to_finetune.py` demonstrates
it: with few-shot examples, the model produced **3/3 valid JSON but only 1/3
matching the required schema** — inventing a `"security"` category and a
`"critical"` urgency that were never allowed.

**Prompts request a schema. They cannot enforce one.** That gap is a legitimate
reason to fine-tune.

*(A third option for strict schemas: constrained decoding, where the runtime only
allows tokens that keep the output valid. Ollama supports it via `format`. Often
cheaper than training.)*

### 6. The three-way split

| Split | Purpose | Rule |
|---|---|---|
| **Train** | The model learns from these | Most of your data |
| **Validation** | Catch overfitting during training | Check every epoch |
| **Test** | Final honest measurement | Touch **once**, at the end |

If you tune your settings against the test set, it stops being a fair measure.
Set it aside and forget about it until you're done.

### 7. Overfitting

The model stops learning the *pattern* and starts memorising your *examples*.

From `03_split_and_overfit.py` — a real run on a real model:

```
epoch  1  | train 4.590 | val 4.955
epoch 12  | train 0.746 | val 2.260   <- best
epoch 40  | train 0.150 | val 2.425   <- val rising
```

Train loss fell 97%. Validation bottomed out at epoch 12 and got worse after.
**The best model existed at epoch 12; everything after was wasted compute making
it worse.**

Training loss alone always looks great. That's exactly why you need the split.

### 8. Settings to start with

| Setting | Typical | Note |
|---|---|---|
| Epochs | 1–3 | More than 3 usually memorises |
| Learning rate (full) | 1e-5 to 5e-5 | Small — you're nudging, not teaching |
| Learning rate (LoRA) | 1e-4 to 2e-4 | Higher, only adapters move |
| Batch size | 1–8 on small GPUs | Raise until you run out of memory |

Change one thing at a time, and only when validation loss tells you to.

---

## Examples

```bash
uv run python phase-06-finetuning-basics/01_when_to_finetune.py
uv run python phase-06-finetuning-basics/02_what_training_does.py
uv run python phase-06-finetuning-basics/03_split_and_overfit.py
```

| File | What it shows | Needs |
|---|---|---|
| `01_when_to_finetune.py` | Prompt vs few-shot vs fine-tune, with schema checking | Ollama |
| `02_what_training_does.py` | Loss on real text, one real gradient step | cached model |
| `03_split_and_overfit.py` | A real fine-tune overfitting, with a loss curve | cached model, ~1 min |

### What to look for

**`01_when_to_finetune.py`** — the schema-validation section is the point. Valid
JSON is not correct JSON, and no amount of prompt wording fully closes the gap.

**`02_what_training_does.py`** — runs six real gradient steps and shows the loss
drop from 5.996 to 4.551. That *is* fine-tuning, at the smallest possible scale.

**`03_split_and_overfit.py`** — trains for 40 epochs on 8 examples and draws an
ASCII loss curve where train and validation visibly separate. The `TrainingMonitor`
flags `<- val rising` live as it happens.

---

## Try it yourself

1. In `03_split_and_overfit.py`, raise `EPOCHS` to 100. The gap gets worse.
2. Set `LEARNING_RATE = 5e-4`. Watch it overfit much faster, or go unstable.
3. Add 8 more examples to `EXAMPLES`. Overfitting should reduce — more data is
   the best fix.
4. In `01_when_to_finetune.py`, add a fourth few-shot example. Does schema
   compliance improve?
5. Write down, in one sentence, what you'd fine-tune a model to do. If you can't,
   you're not ready for Phase 7.

---

## Done when

You can:

- state your fine-tuning goal in one sentence, and how you'd measure it
- explain what loss means for a language model
- decide correctly between prompting, RAG, and fine-tuning
- explain why validation data must be separate from training data
- read a train/validation loss table and spot overfitting
- name a reasonable starting learning rate and epoch count

---

## Common problems

| Problem | Cause | Fix |
|---|---|---|
| Loss goes to 0 fast | Memorising, or dataset too small | More data, fewer epochs |
| Loss becomes `nan` | Learning rate too high | Lower it 10× |
| Model got worse at everything | Catastrophic forgetting | Lower LR, fewer epochs, use LoRA |
| Validation loss rises | Overfitting | Stop at the best epoch |
| No improvement at all | LR too low, or too few steps | Raise LR, train longer |
| Fine-tuned model invents facts | Wrong tool for the job | Use RAG instead |

---

## References

**Concepts**
- [HF — Fine-tuning guide](https://huggingface.co/docs/transformers/training)
- [HF LLM Course — Fine-tuning chapter](https://huggingface.co/learn/llm-course/chapter3/1)
- [Catastrophic forgetting explained](https://arxiv.org/abs/1612.00796)
- [Google — Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting)

**Decision-making**
- [OpenAI — When to fine-tune](https://platform.openai.com/docs/guides/fine-tuning#when-to-use-fine-tuning) — vendor-neutral advice
- [Anthropic — Prompt engineering first](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Ollama structured outputs](https://ollama.com/blog/structured-outputs) — constrained decoding

**Training mechanics**
- [Hyperparameter guidance](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- [AdamW paper](https://arxiv.org/abs/1711.05101) — the standard optimizer
- [Learning rate schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)

**Next:** [Phase 7 — Building Your Dataset](../phase-07-datasets/) — where most of the real work is.
