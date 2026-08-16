# Phase 1: Foundations

**Goal:** Understand how a model *learns*, before touching any LLM.

Everything in this repo — including fine-tuning a chat model later — is the same
idea repeated at a bigger scale: make a guess, measure how wrong it is, nudge the
numbers, repeat. If you understand that loop, LLMs stop being magic.

---

## Why these examples start with tensors

The roadmap lists Python basics first, then tensors. The examples here skip
straight to tensors, on purpose:

- **Python basics are assumed.** If you can read the code in these files, you
  already know enough Python. Go learn it elsewhere if not — a repo about LLMs
  is a poor place to teach `for` loops.
- **Tensors are where LLM-specific knowledge actually starts.** Every error you
  will hit later is a shape error. `(1, 7, 576)` will mean something to you by
  the end of Phase 2.
- **Everything after depends on the training loop.** Fine-tuning in Phase 8 is
  this same loop with a bigger model. Learn it here, where it fits on one screen.

If you want the ML theory alongside this, the Karpathy course in the resources
below is the best free option.

---

## Concepts

### 1. Tensors

A **tensor** is just an array of numbers with a shape.

| Shape | Name | Example |
|---|---|---|
| `()` | scalar | `5.0` |
| `(3,)` | vector | `[1, 2, 3]` |
| `(2, 3)` | matrix | 2 rows, 3 columns |
| `(2, 3, 4)` | 3D tensor | a batch of 2 matrices |

In LLMs you'll constantly see shapes like `(batch, sequence_length, hidden_size)`.
When code breaks, it's usually a shape mismatch — so get comfortable printing
`.shape` early and often.

### 2. The training loop

Every model, from a 2-parameter line to a 70B LLM, trains like this:

```
1. forward pass  → make a prediction
2. loss          → measure how wrong it was
3. backward pass → compute which direction each number should move
4. optimizer step→ move the numbers a little
5. zero gradients→ clear the slate
   repeat
```

Miss step 5 and gradients pile up across iterations, which quietly breaks training.
It's the most common beginner bug.

### 3. Loss

A single number saying "how wrong was this guess?" Lower is better.

- **MSE** (mean squared error) — for predicting numbers
- **Cross-entropy** — for picking a category, and this is what LLMs use.
  Predicting the next token *is* a classification problem over the whole vocabulary.

### 4. Gradient descent

The gradient tells you which way to move each number to make the loss smaller.
The **learning rate** decides how big a step you take.

- Too small → training crawls
- Too large → loss bounces around or becomes `nan`

### 5. Overfitting

Your model memorizes the training data instead of learning the pattern.
You catch it by holding out data the model never trains on:

- **train** — the model learns from this
- **validation** — you check progress on this while tuning
- **test** — touched once, at the very end

Warning sign: training loss keeps dropping while validation loss starts rising.

### 6. From numbers to text

LLMs can't read text — only numbers. The pipeline is:

```
"hello world" → [15339, 1917] → embeddings → model → scores → next token
                 tokenize        lookup table          over vocabulary
```

**Tokens** are word pieces (roughly ¾ of a word each in English).
**Embeddings** turn each token ID into a list of numbers capturing meaning.
You'll do this hands-on in Phase 2.

---

## Examples

Run them in order:

```bash
uv run python phase-01-foundations/01_tensors.py
uv run python phase-01-foundations/02_training_loop.py
uv run python phase-01-foundations/03_overfitting.py
```

| File | What it shows |
|---|---|
| `01_tensors.py` | Tensor shapes, indexing, matrix multiply, broadcasting, MPS |
| `02_training_loop.py` | The same model twice — by hand, then in PyTorch |
| `03_overfitting.py` | Train/val split, and watching a model memorize noise |

### What to look for

**`02_training_loop.py`** trains `y = 2x + 1` from scratch with manual gradients,
then again with `nn.Linear` + `loss.backward()`. Both reach the same answer. This
shows you exactly what PyTorch does for you.

**`03_overfitting.py`** prints a table where validation loss bottoms out and then
climbs while training loss keeps falling. That gap *is* overfitting. You'll watch
for this same pattern when fine-tuning in Phase 4.

---

## Try it yourself

1. In `02_training_loop.py`, set `LEARNING_RATE = 1.0`. Watch the loss explode to `nan`.
2. Then set it to `0.0001`. Watch it barely move in 200 steps.
3. In `03_overfitting.py`, raise `HIDDEN_SIZE` from 64 to 256. Overfitting gets worse.
4. In the same file, raise `N_TRAIN` from 12 to 200. Overfitting mostly disappears —
   **more data is the best fix.** Remember this for Phase 4.

---

## Done when

You can explain, without looking:

- what the 5 steps of a training loop do
- why we call `optimizer.zero_grad()`
- what a learning rate controls, and what happens if it's wrong
- how you'd detect overfitting from a train/validation loss table

---

## Using the tools

`03_overfitting.py` prints its own table, but the shared
[`TrainingMonitor`](../tools/README.md) does the same job in any script you write:

```python
import sys; sys.path.insert(0, "tools")
from llmkit import TrainingMonitor

mon = TrainingMonitor("my-run")
for step in range(steps):
    ...
    mon.log(step, train_loss, val_loss)   # warns when val starts rising

mon.plot()      # ASCII loss curve
mon.summary()   # best epoch + overfitting verdict
```

This is the same signal TensorBoard and Weights & Biases give you later, just
without the setup. See [`tools/README.md`](../tools/README.md).

---

## References

**Courses**
- [Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — start with the micrograd video; it builds backprop from nothing
- [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) — official quick start
- [fast.ai — Practical Deep Learning](https://course.fast.ai/) — top-down alternative if theory-first doesn't suit you

**Specific topics**
- [PyTorch tensor tutorial](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Autograd explained](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) — what `loss.backward()` actually does
- [Broadcasting rules](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [Google — Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting)
- [Distill — Why momentum really works](https://distill.pub/2017/momentum/) — optional, beautifully explained optimizers

**Reference**
- [PyTorch docs](https://pytorch.org/docs/stable/index.html)
- [`torch.nn` layers](https://pytorch.org/docs/stable/nn.html)

**Next:** [Phase 2 — Transformers](../phase-02-transformers/) — load a real model and see tokenization for yourself.
