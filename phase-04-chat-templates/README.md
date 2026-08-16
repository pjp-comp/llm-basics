# Phase 4: Chat Templates and Prompting

**Goal:** Format prompts the way each model expects — and write prompts that work.

This is a short phase with an unusually high payoff. Getting the format wrong is
one of the most common reasons a model "doesn't work", and it's invisible until
you print the formatted string.

---

## Concepts

### 1. Base models vs instruct models

| | Base model | Instruct / Chat model |
|---|---|---|
| Trained to | Continue text | Follow instructions |
| Ask it a question | It may write *more questions* | It answers |
| Needs chat template | No | **Yes** |
| Name usually contains | *(nothing)* | `-Instruct`, `-Chat`, `-it` |

`SmolLM2-135M` continues text. `SmolLM2-135M-Instruct` answers you.
Using a base model and wondering why it won't answer is a classic beginner hour lost.

### 2. Why chat templates exist

Instruct models are trained with special marker tokens around each turn:

```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

Those markers are how the model knows where your question ends and its answer
should begin. **Every model family uses different markers.**

| Model | Marker style |
|---|---|
| SmolLM2, Qwen | `<|im_start|>user ... <|im_end|>` |
| Llama 3 | `<|start_header_id|>user<|end_header_id|> ...` |
| Mistral | `[INST] ... [/INST]` |

If you write `"User: hi\nAssistant:"` by hand, you're using a format the model was
never trained on. It usually still responds — just worse, and you won't know why.

### 3. `apply_chat_template()`

You don't need to memorize any of those formats. The tokenizer knows:

```python
messages = [{"role": "user", "content": "Hello"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

**`add_generation_prompt=True` matters.** It appends the opening marker for the
assistant's turn, so the model knows it's *its* turn to write. Without it, the model
may keep writing as the user instead.

Use `tokenize=False` when you want to *see* the string; drop it when feeding a model.

### 4. The three roles

| Role | Purpose |
|---|---|
| `system` | Standing instructions — persona, rules, output format |
| `user` | What the person said |
| `assistant` | What the model said (used to give conversation history) |

### 5. Models are stateless

The model remembers nothing between calls. A "conversation" is just you resending
the whole history every time:

```python
messages = [
    {"role": "user",      "content": "My name is Sam"},
    {"role": "assistant", "content": "Nice to meet you, Sam!"},
    {"role": "user",      "content": "What's my name?"},   # works only because
]                                                          # the history is here
```

This is also why long conversations cost more — you resend everything each turn.

### 6. Prompting techniques that actually help

| Technique | What it means |
|---|---|
| **Be specific** | "3 bullet points, under 10 words each" beats "be brief" |
| **Few-shot** | Show 2–3 examples of the input/output you want |
| **Assign a role** | "You are a SQL expert" shifts vocabulary and depth |
| **Give an out** | "If unsure, say 'I don't know'" reduces made-up answers |
| **Delimit input** | Put user text in ``` ``` ``` so it isn't read as instructions |

**Few-shot examples are the highest-value trick here.** Often they solve a formatting
problem you were about to fine-tune for — which is exactly the Phase 4.5 decision
in the main roadmap.

---

## Examples

```bash
uv run python phase-04-chat-templates/01_chat_template.py
uv run python phase-04-chat-templates/02_conversation.py
uv run python phase-04-chat-templates/03_prompt_techniques.py
uv run python phase-04-chat-templates/04_chatbot.py
```

| File | What it shows | Needs |
|---|---|---|
| `01_chat_template.py` | The raw formatted string, across models | cached model |
| `02_conversation.py` | Multi-turn memory, and what breaks without it | Ollama |
| `03_prompt_techniques.py` | Vague vs specific, zero-shot vs few-shot | Ollama |
| `04_chatbot.py` | A working chatbot in ~60 lines | Ollama |

### What to look for

**`01_chat_template.py`** prints the actual string sent to the model, with markers
visible. It also shows the same messages formatted for two different models so you
can see the formats differ — and demonstrates what `add_generation_prompt` changes.

**`02_conversation.py`** asks "what's my name?" with and without history. Without it,
the model has no idea. This makes statelessness concrete.

**`03_prompt_techniques.py`** runs vague vs specific prompts side by side, then shows
few-shot teaching a format that a zero-shot prompt gets wrong.

**`04_chatbot.py`** is interactive. Type `/history` to see the message list grow —
that's the whole trick.

---

## Try it yourself

1. In `01_chat_template.py`, add a `system` message and see where it lands in the string.
2. Set `add_generation_prompt=False` and compare the ending.
3. In `03_prompt_techniques.py`, add a 4th few-shot example and see if output gets more consistent.
4. In `04_chatbot.py`, change `SYSTEM_PROMPT` to something odd ("answer only in questions").

---

## Done when

You can:

- explain why `"User: ...\nAssistant:"` is worse than `apply_chat_template()`
- say what `add_generation_prompt=True` adds and why
- explain how multi-turn memory works when the model is stateless
- write a few-shot prompt to lock an output format

---

## Common problems

| Problem | Cause | Fix |
|---|---|---|
| Model replies to itself / writes your next line | Missing `add_generation_prompt=True` | Add it |
| `<\|im_start\|>` visible in output | Template applied twice | Apply once — in Phase 8, `trl` does it for you |
| Model ignores the system prompt | Some small models barely use it | Put key rules in the user message too |
| Model forgets earlier turns | History not resent | Append every turn to `messages` |
| `TemplateError` | Base model with no chat template | Use the `-Instruct` version |

---

## References

**Chat templates**
- [Hugging Face — Chat templating guide](https://huggingface.co/docs/transformers/chat_templating) — the main reference
- [`apply_chat_template` API](https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template)
- [ChatML format](https://github.com/openai/openai-python/blob/release-v0.28.0/chatml.md) — the `<|im_start|>` convention

**Prompting**
- [Anthropic — Prompt engineering guide](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) — clear and practical
- [OpenAI — Prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)
- [Brown et al. — Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — the GPT-3 paper that made few-shot famous
- [Wei et al. — Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — "think step by step"

**Ollama**
- [Ollama Python library](https://github.com/ollama/ollama-python)
- [Ollama model library](https://ollama.com/library) — browse available models

**Next:** [Phase 5 — RAG](../phase-05-rag/)
