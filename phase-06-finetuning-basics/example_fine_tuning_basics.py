# This is a simple conceptual example showing the fine-tuning workflow.
# It does not train a real large model in this file.

from dataclasses import dataclass


@dataclass
class TrainingExample:
    prompt: str
    response: str


examples = [
    TrainingExample(
        prompt="Summarize this sentence: The system is fast and stable.",
        response="A fast and stable system.",
    ),
    TrainingExample(
        prompt="Summarize this sentence: Customer support resolved the issue quickly.",
        response="Customer support fixed the issue quickly.",
    ),
]


def show_training_flow():
    print("Fine-tuning workflow:")
    print("1. Load a pretrained model")
    print("2. Prepare training examples")
    print("3. Format prompt + answer pairs")
    print("4. Train on the dataset")
    print("5. Validate on a held-out set")
    print("6. Save the tuned model")

    for item in examples:
        print(f"Prompt: {item.prompt}")
        print(f"Expected response: {item.response}")
        print("---")


if __name__ == "__main__":
    show_training_flow()
