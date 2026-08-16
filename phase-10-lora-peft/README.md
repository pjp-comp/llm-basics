# Phase 5: LoRA and PEFT

## Goal
Learn how to fine-tune a model without updating all of its parameters.

## Concepts

### Full fine-tuning
Updates the entire model, which is expensive and memory heavy.

### LoRA
Low-Rank Adaptation adds small trainable matrices instead of retraining everything.

### PEFT
Parameter-Efficient Fine-Tuning is a family of methods that reduce the cost of training.

### QLoRA
A common version of LoRA that uses quantized weights to save memory.

### Why this matters
This is the standard path for fine-tuning small open models in realistic settings.

## Common terms
- rank
- alpha
- target modules
- adapters
- merge and unload

## Example
See `example_lora.py`.

## Suggested next step
Move to Phase 6 and learn the modern fine-tuning trainer flow in TRL.
