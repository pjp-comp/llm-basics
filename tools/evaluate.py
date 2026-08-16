"""
evaluate.py - compare two models on the same questions.

This is the "minimum viable evaluation" from Phase 12 of the roadmap:
run both models on held-out questions, then judge the answers.

    # side by side, you score them yourself
    uv run python tools/evaluate.py --a llama3.2:3b --b qwen3:1.7b

    # automatic scoring with a third model as judge
    uv run python tools/evaluate.py --a llama3.2:3b --b llama3.1:8b --judge

    # your own questions
    uv run python tools/evaluate.py --a llama3.2:3b --b llama3.1:8b --file q.txt

A judge model is convenient but not truth. Read some answers yourself.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llmkit import Timer, section  # noqa: E402

DEFAULT_QUESTIONS = [
    "What is the capital of Japan?",
    "Explain what a variable is, in one sentence.",
    "Write a Python function that adds two numbers.",
    "What is 15% of 200?",
    "Name three primary colours.",
]

JUDGE_PROMPT = """You are grading two answers to the same question.

Question: {question}

Answer A: {answer_a}

Answer B: {answer_b}

Which answer is better? Consider accuracy, clarity and whether it actually
answers the question. Reply with exactly one word: A, B, or TIE."""


def ask_ollama(model, question, num_predict=150):
    import ollama

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": question}],
            options={"num_predict": num_predict, "temperature": 0.0},
        )
    except Exception as e:
        return f"[error: {type(e).__name__}]"

    message = response["message"]
    text = (message.get("content") or "").strip()

    if not text and (message.get("thinking") or "").strip():
        return "[reasoning model ran out of tokens - raise --tokens]"
    return text or "[empty]"


def judge(judge_model, question, answer_a, answer_b):
    """Ask a third model which answer is better. Returns 'A', 'B' or 'TIE'."""
    verdict = ask_ollama(
        judge_model,
        JUDGE_PROMPT.format(question=question, answer_a=answer_a, answer_b=answer_b),
        num_predict=10,
    ).upper()

    # The model may write a sentence; take the first verdict word it contains.
    for token in verdict.replace("*", " ").split():
        if token.startswith("TIE"):
            return "TIE"
        if token in ("A", "A.", "A,"):
            return "A"
        if token in ("B", "B.", "B,"):
            return "B"
    return "TIE"


def load_questions(path):
    lines = Path(path).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def main():
    p = argparse.ArgumentParser(description="Compare two models.")
    p.add_argument("--a", required=True, help="first ollama model")
    p.add_argument("--b", required=True, help="second ollama model")
    p.add_argument("--judge", nargs="?", const="llama3.1:8b", default=None,
                   help="auto-score with a judge model (default llama3.1:8b)")
    p.add_argument("--file", help="text file of questions, one per line")
    p.add_argument("--tokens", type=int, default=150, help="max tokens per answer")
    p.add_argument("--save", help="write results to a JSON file")
    args = p.parse_args()

    questions = load_questions(args.file) if args.file else DEFAULT_QUESTIONS

    print(f"model A : {args.a}")
    print(f"model B : {args.b}")
    print(f"questions: {len(questions)}")
    if args.judge:
        print(f"judge   : {args.judge}")

    results = []
    tally = {"A": 0, "B": 0, "TIE": 0}

    for i, question in enumerate(questions, 1):
        section(f"Q{i}: {question}")

        with Timer("A", quiet=True) as ta:
            answer_a = ask_ollama(args.a, question, args.tokens)
        with Timer("B", quiet=True) as tb:
            answer_b = ask_ollama(args.b, question, args.tokens)

        print(f"\n  --- A: {args.a}  ({ta.seconds:.1f}s) ---")
        print(f"  {answer_a[:300]}")
        print(f"\n  --- B: {args.b}  ({tb.seconds:.1f}s) ---")
        print(f"  {answer_b[:300]}")

        row = {
            "question": question,
            "a": answer_a,
            "b": answer_b,
            "seconds_a": round(ta.seconds, 2),
            "seconds_b": round(tb.seconds, 2),
        }

        if args.judge:
            verdict = judge(args.judge, question, answer_a, answer_b)
            tally[verdict] += 1
            row["winner"] = verdict
            label = {"A": args.a, "B": args.b}.get(verdict, "tie")
            print(f"\n  judge: {verdict}  ({label})")
        else:
            print("\n  Your call: which is better?")

        results.append(row)

    section("Summary")

    avg_a = sum(r["seconds_a"] for r in results) / len(results)
    avg_b = sum(r["seconds_b"] for r in results) / len(results)
    print(f"  average time A: {avg_a:.1f}s")
    print(f"  average time B: {avg_b:.1f}s")

    if args.judge:
        total = sum(tally.values())
        print(f"\n  {args.a}: {tally['A']}/{total} wins")
        print(f"  {args.b}: {tally['B']}/{total} wins")
        print(f"  ties     : {tally['TIE']}/{total}")

        if tally["A"] > tally["B"]:
            print(f"\n  Judge preferred: {args.a}")
        elif tally["B"] > tally["A"]:
            print(f"\n  Judge preferred: {args.b}")
        else:
            print("\n  Too close to call.")

        print("\n  Remember: an LLM judge is a rough signal, not truth.")
        print("  It favours longer, more confident answers. Read some yourself.")
    else:
        print("\n  No judge used. Read the answers and score them 1-5 each.")
        print("  Doing this by hand for 30-50 questions is the honest way.")

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"\n  saved to {args.save}")


if __name__ == "__main__":
    main()
