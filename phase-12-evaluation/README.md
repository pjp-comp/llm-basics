# Phase 7: Evaluation and Quality Checks

## Goal
Learn how to assess whether a fine-tuned model actually performs well.

## Concepts

### Training loss is not enough
A model can achieve low training loss but still produce poor outputs.

### Validation set
Use a set of examples that were not used during training.

### Human evaluation
Look at candidate answers manually and judge quality, truthfulness, and format.

### LLM-as-a-judge
Use a stronger model to evaluate generated responses.

### Benchmark suites
Tools like `lm-evaluation-harness` and `lighteval` help run systematic evaluations.

## Important questions
- Did the model follow the prompt?
- Did it answer correctly?
- Did it hallucinate?
- Is the format stable and useful?

## Example
See `example_evaluation.py`.

## Suggested next step
Move to Phase 8 and learn export, merge, and local deployment.
