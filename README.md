# LLM Learn

Learning how to work with LLMs (Large Language Models) using Hugging Face — starting from the basics.

## Where to start

- [LLM-HuggingFace-Roadmap.md](LLM-HuggingFace-Roadmap.md) — the full roadmap, from Python basics to training your own small model

## The path

1. Read the roadmap first
2. Learn Python, PyTorch, and NLP basics
3. Load and run models with Hugging Face
4. Learn chat templates (small topic, saves a lot of pain)
5. Build a RAG demo — and learn when you *don't* need to train a model
6. Learn LoRA and fine-tune a small model
7. Test your model properly and improve your data
8. Export it and run it on your own computer

## Setup

**Base install** — works on Mac, Linux, and Windows:

```bash
pip install transformers datasets accelerate peft trl sentencepiece
```

**For the RAG phase:**

```bash
pip install sentence-transformers faiss-cpu
```

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
