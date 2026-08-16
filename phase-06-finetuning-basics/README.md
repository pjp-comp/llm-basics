# Phase 4: Fine-Tuning Basics

## Goal
Learn what fine-tuning means and why it is different from prompt engineering or inference.

## Concepts

### Pretrained model
A model that has already learned language patterns from large-scale data.

### Fine-tuning
The process of adapting a pretrained model to your target task or domain.

### Supervised fine-tuning
Training on labeled examples such as:
- prompt + expected answer
- instruction + output
- question + answer

### Dataset quality
High-quality data matters more than model size in many beginner cases.

### Train vs validation
- train set: used for optimization
- validation set: used to check if the model generalizes

## Why this matters
A base model can generate text, but it may not follow your task format or your domain rules. Fine-tuning helps align the model with your desired behavior.

## Example
See `example_fine_tuning_basics.py`.

## Suggested next step
Move to Phase 5 and learn LoRA / PEFT, which is the most practical way to fine-tune small models.
