from dataclasses import dataclass


@dataclass
class ExampleDatasetItem:
    prompt: str
    response: str


train_data = [
    ExampleDatasetItem(
        prompt="What is a neural network?",
        response="A neural network is a machine learning model made of interconnected layers that learn patterns from data.",
    ),
    ExampleDatasetItem(
        prompt="Explain overfitting in one sentence.",
        response="Overfitting happens when a model learns training data too closely and performs poorly on new data.",
    ),
]

print("Example training data prepared for SFTTrainer:")
for i, item in enumerate(train_data, start=1):
    print(f"Example {i}:")
    print(f"Prompt: {item.prompt}")
    print(f"Response: {item.response}")
    print("---")

print("Typical steps:")
print("1. Tokenize prompt + response")
print("2. Use chat template or instruction format")
print("3. Train with SFTTrainer")
print("4. Save adapters or merged model")
