# Conceptual evaluation example

responses = [
    "The model correctly answered the question.",
    "The answer is mostly correct but a bit vague.",
    "The response is incorrect or unsupported by the prompt.",
]


def score_response(response: str):
    if "correct" in response.lower():
        return 1
    if "mostly correct" in response.lower():
        return 0.5
    return 0


scores = [score_response(r) for r in responses]
print("Evaluation scores:")
for r, s in zip(responses, scores):
    print(f"Response: {r} | Score: {s}")

print(f"Average score: {sum(scores)/len(scores):.2f}")
print("This is the basic idea behind evals: measure quality, not just loss.")
