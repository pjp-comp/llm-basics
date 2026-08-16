# Phase 6: TRL and SFTTrainer

## Goal
Learn the modern supervised fine-tuning workflow used in Hugging Face for LLMs.

## Concepts

### TRL
The Transformers Reinforcement Learning library provides modern LLM training helpers.

### SFTTrainer
A trainer for supervised fine-tuning that works well with chat datasets and PEFT workflows.

### SFTConfig
Configuration object for supervised fine-tuning training settings.

### Why this matters
It is easier than building a custom training loop from scratch for most LLM workflows.

## Typical workflow
- prepare dataset
- format as instruction or chat examples
- load tokenizer
- configure SFT trainer
- train with LoRA
- save model or adapters

## Example
See `example_sfttrainer.py`.

## Suggested next step
Move to Phase 7 and learn evaluation and quality checks.
