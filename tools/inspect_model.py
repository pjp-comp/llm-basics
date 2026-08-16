"""
inspect_model.py - look at any model without writing code.

Examples:
    uv run python tools/inspect_model.py --info
    uv run python tools/inspect_model.py --tokens "Hello world"
    uv run python tools/inspect_model.py --predict "The capital of France is"
    uv run python tools/inspect_model.py --temps "Once upon a time"
    uv run python tools/inspect_model.py --speed
    uv run python tools/inspect_model.py --compare "Explain gravity" --ollama llama3.2:3b
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llmkit import (  # noqa: E402
    Timer,
    pick_device,
    print_model_size,
    print_top_predictions,
    section,
    tokens_per_second,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with Timer(f"loading {model_name}"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def cmd_info(tokenizer, model, args):
    section("Model information")
    cfg = model.config

    print(f"  name        : {args.model}")
    print(f"  type        : {cfg.model_type}")
    print(f"  layers      : {getattr(cfg, 'num_hidden_layers', '?')}")
    print(f"  hidden size : {getattr(cfg, 'hidden_size', '?')}")
    print(f"  attn heads  : {getattr(cfg, 'num_attention_heads', '?')}")
    print(f"  vocab size  : {cfg.vocab_size:,}")
    print(f"  max context : {getattr(cfg, 'max_position_embeddings', '?')}")
    print()
    print_model_size(model)

    print(f"\n  chat template: {'yes' if tokenizer.chat_template else 'NO (base model)'}")
    print(f"  eos token    : {tokenizer.eos_token!r}")
    print(f"  pad token    : {tokenizer.pad_token!r}")


def cmd_tokens(tokenizer, model, args):
    section(f"Tokenizing: {args.tokens!r}")

    ids = tokenizer.encode(args.tokens)
    print(f"  {len(args.tokens)} characters -> {len(ids)} tokens")
    print(f"  ~{len(args.tokens) / max(len(ids), 1):.1f} characters per token\n")

    print(f"  {'#':>3}  {'id':>7}  token")
    print(f"  {'-' * 3}  {'-' * 7}  {'-' * 26}")
    for i, tid in enumerate(ids):
        print(f"  {i:>3}  {tid:>7}  {tokenizer.decode([tid])!r}")


def cmd_predict(tokenizer, model, args):
    import torch

    section(f"Next-token predictions for: {args.predict!r}")

    inputs = tokenizer(args.predict, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    print()
    print_top_predictions(logits[0, -1], tokenizer, k=args.top)


def cmd_temps(tokenizer, model, args):
    import torch

    section(f"Same prompt at different temperatures: {args.temps!r}")

    inputs = tokenizer(args.temps, return_tensors="pt").to(model.device)

    for temp in (0.1, 0.5, 1.0, 1.5, 2.0):
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.tokens_out,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        new = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new, skip_special_tokens=True).strip().replace("\n", " ")
        print(f"\n  temp {temp}:")
        print(f"    {text[:180]}")

    print("\n  Low = predictable. High = creative, then incoherent.")


def cmd_speed(tokenizer, model, args):
    import torch

    section("Generation speed")

    prompt = "Write a short story about a robot."

    # Use the chat template so an instruct model actually keeps talking.
    # Without it the model can emit EOS immediately and generate 1 token,
    # which makes the measurement meaningless.
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # First run includes warm-up cost, so discard it.
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=8, pad_token_id=tokenizer.pad_token_id)

    for n in (20, 60):
        with Timer(f"{n} tokens", quiet=True) as t, torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=n,
                do_sample=False,
                min_new_tokens=n,  # don't stop early, or we time nothing
                pad_token_id=tokenizer.pad_token_id,
            )
        made = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"  {made:>3} tokens in {t.seconds:5.2f}s  "
              f"= {tokens_per_second(made, t.seconds):5.1f} tokens/sec")

    print(f"\n  device: {model.device}")
    print("  Bigger models are slower. Quantized models are faster but less accurate.")


def cmd_compare(tokenizer, model, args):
    """Same prompt through the local HF model and an Ollama model."""
    import torch

    section(f"Comparing models on: {args.compare!r}")

    messages = [{"role": "user", "content": args.compare}]

    # --- Hugging Face model ---
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = args.compare

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with Timer("hf", quiet=True) as t, torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.tokens_out,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    hf_text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    print(f"\n  --- {args.model}  ({t.seconds:.1f}s) ---")
    print(f"  {hf_text[:400]}")

    # --- Ollama model ---
    try:
        import ollama

        with Timer("ollama", quiet=True) as t2:
            response = ollama.chat(
                model=args.ollama,
                messages=messages,
                options={"num_predict": args.tokens_out},
            )
        text = (response["message"].get("content") or "").strip()
        if not text and (response["message"].get("thinking") or "").strip():
            text = "[reasoning model still thinking - raise --tokens-out]"

        print(f"\n  --- {args.ollama}  ({t2.seconds:.1f}s) ---")
        print(f"  {text[:400]}")

        print("\n  Bigger models are usually better and slower.")
        print("  This is the comparison you'll run in Phase 12 to judge fine-tuning.")
    except Exception as e:
        print(f"\n  Ollama unavailable ({type(e).__name__}). Start it with: ollama serve")


def main():
    p = argparse.ArgumentParser(description="Inspect a language model.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument("--ollama", default="llama3.2:3b", help="ollama model for --compare")
    p.add_argument("--top", type=int, default=10, help="how many predictions to show")
    p.add_argument("--tokens-out", type=int, default=60, help="tokens to generate")

    p.add_argument("--info", action="store_true", help="size, layers, config")
    p.add_argument("--tokens", metavar="TEXT", help="show tokenization")
    p.add_argument("--predict", metavar="TEXT", help="top next-token predictions")
    p.add_argument("--temps", metavar="TEXT", help="compare temperatures")
    p.add_argument("--speed", action="store_true", help="measure tokens/sec")
    p.add_argument("--compare", metavar="TEXT", help="HF model vs ollama model")

    args = p.parse_args()

    actions = [
        (args.info, cmd_info),
        (args.tokens, cmd_tokens),
        (args.predict, cmd_predict),
        (args.temps, cmd_temps),
        (args.speed, cmd_speed),
        (args.compare, cmd_compare),
    ]
    chosen = [fn for flag, fn in actions if flag]

    if not chosen:
        p.print_help()
        print("\nTip: start with  --info  or  --predict 'The capital of France is'")
        return

    device = pick_device(verbose=True)
    tokenizer, model = load(args.model, device)

    for fn in chosen:
        fn(tokenizer, model, args)


if __name__ == "__main__":
    main()
