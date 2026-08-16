# Phase 8: Merge, Export, and Local Deployment

## Goal
Take a fine-tuned model and prepare it for real use.

## Concepts

### Adapter weights
Fine-tuned LoRA adapters are often separate from the base model weights.

### Merge and unload
This combines the adapter weights with the base model for inference.

### Exporting models
You may save the final merged model or convert it for local runtimes.

### Local inference runtimes
Examples:
- Ollama
- llama.cpp
- vLLM
- GGUF-based runtimes

## Why this matters
A fine-tuned model is only useful if it can be run and tested in a practical environment.

## Example
See `example_deploy.py`.

## Suggested next step
Create your own project using a small dataset and a small model.
