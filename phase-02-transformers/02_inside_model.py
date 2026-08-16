"""
Phase 2 - Example 2: Inside the model

Follows one prompt through every stage and prints the real tensor
shapes. This is where Phase 1's shape practice pays off.

Uses the model cached on first download (~270 MB).

Run:
    uv run python phase-02-transformers/02_inside_model.py

Reference:
    https://jalammar.github.io/illustrated-transformer/
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT = "The capital of France is"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    cfg = model.config
    print(f"Model: {MODEL}")
    print(f"  layers      : {cfg.num_hidden_layers}")
    print(f"  hidden size : {cfg.hidden_size}   (numbers per token)")
    print(f"  attn heads  : {cfg.num_attention_heads}")
    print(f"  vocab size  : {cfg.vocab_size:,}")

    section("Stage 1: text -> token IDs")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    ids = inputs["input_ids"]

    print(f"  prompt: {PROMPT!r}")
    print(f"  shape : {tuple(ids.shape)}  = (batch=1, {ids.shape[1]} tokens)\n")
    for pos, tid in enumerate(ids[0].tolist()):
        print(f"    pos {pos}: {tid:>6} = {tokenizer.decode([tid])!r}")

    section("Stage 2: token IDs -> embeddings")

    embed_layer = model.get_input_embeddings()
    embeddings = embed_layer(ids)

    print(f"  embedding table: {tuple(embed_layer.weight.shape)}")
    print(f"                   = (vocab {cfg.vocab_size:,}, {cfg.hidden_size} numbers each)")
    print(f"\n  after lookup:    {tuple(embeddings.shape)}")
    print(f"                   = (batch, tokens, hidden)")
    print(f"\n  first 6 numbers for token 0: {embeddings[0, 0, :6].tolist()}")
    print("  (meaningless alone - meaning lives in the pattern)")

    section("Stage 3: through the layers")

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    states = out.hidden_states
    print(f"  {len(states)} snapshots = 1 input + {cfg.num_hidden_layers} layers\n")
    print(f"  {'stage':<22} {'shape':<18} {'avg magnitude':>14}")
    print(f"  {'-' * 22} {'-' * 18} {'-' * 14}")

    for i in (0, 1, len(states) // 2, len(states) - 1):
        name = "input embeddings" if i == 0 else f"after layer {i}"
        norm = states[i].norm(dim=-1).mean().item()
        print(f"  {name:<22} {str(tuple(states[i].shape)):<18} {norm:>14.2f}")

    print("\n  Shape never changes - each layer refines the same-sized vectors.")

    section("Stage 4: -> scores for the next token")

    logits = out.logits
    print(f"  logits shape: {tuple(logits.shape)}")
    print(f"                = (batch, tokens, vocab {cfg.vocab_size:,})")
    print("\n  One score per vocabulary word, at EVERY position.")
    print("  For generating, we only care about the last position.")

    last = logits[0, -1]
    probs = torch.softmax(last, dim=-1)
    top = torch.topk(probs, 10)

    print(f"\n  Top 10 predictions after {PROMPT!r}:\n")
    print(f"    {'token':<14} {'probability':>12}  bar")
    print(f"    {'-' * 14} {'-' * 12}  {'-' * 22}")

    for prob, tid in zip(top.values.tolist(), top.indices.tolist()):
        piece = tokenizer.decode([tid])
        bar = "#" * max(1, int(prob * 40))
        print(f"    {piece!r:<14} {prob:>11.1%}  {bar}")

    section("Stage 5: temperature reshapes those probabilities")

    print(f"  Same scores, different temperature. Watch the top token's share:\n")
    print(f"    {'temp':<8} {'top token':<14} {'its probability':>16}")
    print(f"    {'-' * 8} {'-' * 14} {'-' * 16}")

    for temp in (0.1, 0.5, 1.0, 2.0):
        scaled = torch.softmax(last / temp, dim=-1)
        best = scaled.argmax().item()
        print(f"    {temp:<8} {tokenizer.decode([best])!r:<14} {scaled[best].item():>15.1%}")

    print("\n  Low temp  -> one token dominates -> predictable")
    print("  High temp -> probabilities flatten -> more variety")
    print("\n  Dividing by temperature BEFORE softmax is the whole trick.")

    print("\nDone. Phase 2 complete -> phase-03-huggingface-basics/")


if __name__ == "__main__":
    main()
