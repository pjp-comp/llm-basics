# Tools

Small utilities for looking at what a model is doing. Used from Phase 1 onward.

Nothing here is required to learn — but seeing numbers beats guessing.

---

## `inspect_model.py`

Examine any model from the command line, no code needed.

```bash
# size, layers, config, whether it has a chat template
uv run python tools/inspect_model.py --info

# how text splits into tokens
uv run python tools/inspect_model.py --tokens "Hello world"

# what the model thinks comes next, with probabilities
uv run python tools/inspect_model.py --predict "The capital of France is"

# the same prompt at 5 temperatures
uv run python tools/inspect_model.py --temps "Once upon a time"

# tokens per second on your hardware
uv run python tools/inspect_model.py --speed

# local HF model vs an Ollama model, same prompt
uv run python tools/inspect_model.py --compare "Explain gravity" --ollama llama3.2:3b
```

Use `--model <hf-id>` to point at a different model.

**Useful in:** Phase 2 (tokenization, predictions), Phase 3 (speed, comparison).

---

## `evaluate.py`

Run two models on the same questions and compare. This is the honest
evaluation loop — the one that tells you whether a change actually helped.

```bash
# side by side, you judge
uv run python tools/evaluate.py --a llama3.2:3b --b qwen3:1.7b

# let a third model score them
uv run python tools/evaluate.py --a llama3.2:3b --b llama3.1:8b --judge

# your own questions, saved results
uv run python tools/evaluate.py --a llama3.2:3b --b llama3.1:8b \
    --file questions.txt --save results.json
```

**On LLM-as-judge:** convenient, not truth. Judges favour longer and more
confident answers regardless of correctness. Use it to narrow things down,
then read a sample yourself.

**Useful in:** Phase 3 (comparing prompts and models). Essential later when
you need to prove a fine-tuned model beats the original.

---

## `llmkit.py`

Helpers you can import into your own scripts:

```python
import sys; sys.path.insert(0, "tools")
from llmkit import section, pick_device, Timer, TrainingMonitor, print_model_size
```

| Function | Does |
|---|---|
| `section(title)` | Prints a labelled divider |
| `table(rows, headers)` | Aligned text table |
| `bar(value, max)` | Text bar for charts |
| `pick_device()` | Returns `mps` / `cuda` / `cpu` |
| `print_model_size(model)` | Parameter counts and memory |
| `Timer("label")` | Context manager that times a block |
| `TrainingMonitor` | Records loss, detects overfitting, ASCII chart |
| `print_top_predictions(logits, tok)` | Top-k next tokens with bars |
| `compare_outputs(...)` | Two outputs side by side |

### `TrainingMonitor`

Tracks training and validation loss, and warns when they diverge:

```python
mon = TrainingMonitor("my-run")
for step in range(steps):
    ...
    mon.log(step, train_loss, val_loss)

mon.plot()      # ASCII loss curve
mon.summary()   # best epoch, overfitting warning
mon.save("run.json")
```

It's deliberately dependency-free so it runs anywhere.

---

## Real monitoring tools

`TrainingMonitor` is for learning. For actual training runs, use one of these —
both are one config line in `SFTTrainer` later:

| Tool | Setup | Good for |
|---|---|---|
| **TensorBoard** | `pip install tensorboard`, no account | Local, private, quick |
| **Weights & Biases** | `pip install wandb`, free account | Comparing many runs, sharing |

```python
# TensorBoard
SFTConfig(report_to="tensorboard", logging_dir="./logs")
# then:  tensorboard --logdir ./logs

# Weights & Biases
SFTConfig(report_to="wandb", run_name="my-experiment")
```

You'll want one of these from Phase 8 onward, when runs take long enough that
watching numbers scroll past stops working.

---

## References

- [TensorBoard guide](https://www.tensorflow.org/tensorboard/get_started)
- [Weights & Biases quickstart](https://docs.wandb.ai/quickstart)
- [HF Trainer logging](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.report_to)
- [Judging LLM-as-a-Judge (MT-Bench paper)](https://arxiv.org/abs/2306.05685) — how reliable judges actually are
