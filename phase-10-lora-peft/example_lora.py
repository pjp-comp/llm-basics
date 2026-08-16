# Simple conceptual example for LoRA setup
# This shows the usual idea, not a full training run.

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] | None = None
    bias: str = "none"


config = LoRAConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

print("LoRA config created:")
print(f"rank: {config.r}")
print(f"alpha: {config.lora_alpha}")
print(f"target modules: {config.target_modules}")
print("This is the small adapter style used in PEFT-based fine-tuning.")
