# LLM Learn

Learning how to work with LLMs (Large Language Models) using Hugging Face — starting from the basics.

## Where to start

- [LLM-HuggingFace-Roadmap.md](LLM-HuggingFace-Roadmap.md) — the full roadmap, from Python basics to training your own small model
- [phase-01-foundations/](phase-01-foundations/) — start running code here

```bash
uv sync
uv run python phase-01-foundations/01_tensors.py
```

## Phases

Each folder matches a phase in the roadmap, with a README explaining the concepts
and runnable examples.

| Phase | Folder | Status |
|---|---|---|
| 1. Foundations | [phase-01-foundations](phase-01-foundations/) | ready — 3 examples |
| 2. NLP & Transformers | [phase-02-transformers](phase-02-transformers/) | ready — 2 examples |
| 3. Hugging Face Basics | [phase-03-huggingface-basics](phase-03-huggingface-basics/) | ready — 2 examples |
| 4. Chat Templates | [phase-04-chat-templates](phase-04-chat-templates/) | ready — 4 examples |
| 5. RAG | [phase-05-rag](phase-05-rag/) | ready — 5 examples |
| 6. Fine-tuning Basics | [phase-06-finetuning-basics](phase-06-finetuning-basics/) | ready — 3 examples |
| 7. Datasets | [phase-07-datasets](phase-07-datasets/) | planned |
| 8. Training Workflow | [phase-08-training-workflow](phase-08-training-workflow/) | planned |
| 9. Loss Masking | [phase-09-loss-masking](phase-09-loss-masking/) | planned |
| 10. LoRA & PEFT | [phase-10-lora-peft](phase-10-lora-peft/) | planned |
| 11. Fine-tune a Model | [phase-11-finetune-small-llm](phase-11-finetune-small-llm/) | planned |
| 12. Evaluation | [phase-12-evaluation](phase-12-evaluation/) | planned |
| 13. Export & Merge | [phase-13-export-merge](phase-13-export-merge/) | planned |
| 14. Run Locally | [phase-14-local-inference](phase-14-local-inference/) | planned |
| 15. Build Your Own LLM | [phase-15-build-your-own-llm](phase-15-build-your-own-llm/) | planned |

## Tools

[`tools/`](tools/) has utilities for seeing what a model is doing — usable from
Phase 1 onward:

```bash
# model size, tokenization, predictions, speed
uv run python tools/inspect_model.py --predict "The capital of France is"

# compare two models on the same questions
uv run python tools/evaluate.py --a llama3.2:3b --b qwen3:1.7b
```

See [tools/README.md](tools/README.md).

## Setup

This repo uses [uv](https://docs.astral.sh/uv/). One command installs everything:

```bash
uv sync
```

Then run any example with `uv run python <path>`.

<details>
<summary>Using pip instead</summary>

```bash
pip install transformers datasets accelerate peft trl sentencepiece \
            sentence-transformers faiss-cpu ollama torch
```
</details>

**Only if you have an NVIDIA GPU** — `bitsandbytes` allows 4-bit training (QLoRA):

```bash
pip install bitsandbytes
```

### If you are on a Mac

`bitsandbytes` only works with NVIDIA graphics cards, so 4-bit training (QLoRA) will not run on a Mac. You have two options:

- **Use Google Colab** for the training phases. The free version is enough. This is what the roadmap assumes.
- Or use **MLX**, Apple's own tool (`pip install mlx-lm`). It works well on Mac, but it is a different set of tools than this roadmap teaches.

Everything up through RAG runs fine on a normal computer. You only need Colab starting at the fine-tuning phases.

## Notes

This repo will grow over time with example scripts, notebooks, and small training experiments.

## License

For learning and experimentation.
