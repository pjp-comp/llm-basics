# LLM Engineering Roadmap with Hugging Face

This roadmap is for complete beginners who want to learn how to work with LLMs (Large Language Models) using Hugging Face, and eventually train a small one on their own data.

You will go from:
- basic Python and machine learning
- to understanding how transformers work
- to running models and writing good prompts
- to RAG (giving a model your own documents)
- to fine-tuning (training a model on your own examples)
- to LoRA / PEFT (cheap training that works on small computers)
- to testing your model and running it locally

This is a practical roadmap, not just theory.

## How to use this document

Go through the 14 phases in order. Each phase has:

- **Goal** — what you are trying to learn
- **Topics** — what to study
- **Done when** — how you know you can move on
- **Rough time** — a loose estimate, not a deadline

Two things to read before you start training anything:

- **Section 4: What computer do you need?** — decides what you can actually run
- **Phase 4.5: Do you even need fine-tuning?** — many projects don't

**Total time:** around 3–5 months if you study part-time. Most of that is the early fundamentals.

Do not worry if some words here are new. Each one is explained when you reach it.

---

## 1. What you should learn first

Before training models, get comfortable with these basics:

- **Python basics**
  - variables, loops, functions, classes
  - reading and writing files
  - working with JSON and CSV files
- **Machine learning basics**
  - training, validation, and test data
  - what a loss function is
  - gradient descent (how a model improves)
  - overfitting (memorizing instead of learning)
- **Deep learning basics**
  - neural networks
  - tensors (arrays of numbers the model works with)
  - backpropagation (how the model learns from mistakes)
  - optimizers
- **NLP basics** (Natural Language Processing)
  - tokenization (splitting text into pieces)
  - embeddings (turning words into numbers)
  - text generation vs classification

If you understand these, everything after gets much easier.

---

## 2. Core concepts you must understand

### 2.1 Tokenization
A tokenizer turns text into small pieces called tokens, then into numbers the model can read.

Learn:
- token IDs
- attention masks
- padding (making all inputs the same length)
- truncation (cutting text that is too long)
- special tokens

### 2.2 Embeddings
Tokens get turned into lists of numbers (vectors) that capture meaning.

Learn:
- word embeddings
- positional embeddings (where a word sits in the sentence)
- contextual embeddings (meaning changes with context)

### 2.3 Attention
This is the main idea behind transformer models. It lets the model look at every other word when deciding what a word means.

Learn:
- self-attention
- multi-head attention
- why attention helps with long text

### 2.4 Model families
Know the difference:
- **GPT-style** — writes text (this is what you will mostly use)
- **BERT-style** — classifies text, makes embeddings
- **T5 / encoder-decoder** — translates and transforms text

### 2.5 Pretraining vs Fine-tuning
- **Pretraining** — the model learns general language from huge amounts of text. Very expensive. Someone else already did this for you.
- **Fine-tuning** — you take that finished model and teach it your specific task. This is what you will do.

---

## 3. Hugging Face libraries to learn

These are the tools you will use:

| Library | What it does |
|---|---|
| `transformers` | Load models and tokenizers, run them |
| `datasets` | Load and prepare training data |
| `accelerate` | Makes training work on your GPU |
| `peft` | LoRA and other cheap training methods |
| `trl` | Modern training tools, especially `SFTTrainer` |
| `bitsandbytes` | Shrinks models so they fit in less memory |
| `lighteval` | Tests how good your model is |

You don't need to master all of these now. Just know what each one is for.

**Two small notes:**
- You may see a library called `tokenizers`. You almost never use it directly — `transformers` uses it for you behind the scenes.
- Older tutorials suggest `evaluate` for scoring models. It is mostly replaced by `lighteval` now.

**One thing to add early: experiment tracking.** Use TensorBoard (free, no account) or Weights & Biases (`wandb`). This shows you a graph of whether training is actually working, instead of staring at numbers scrolling by. `SFTTrainer` connects to both with one line of config.

---

## 4. What computer do you need?

Read this before Phase 6. It decides which parts of this roadmap you can run on your own machine.

### The problem that surprises people

`bitsandbytes` is the library that makes QLoRA (4-bit training) work. It only runs on NVIDIA graphics cards. **On a Mac, it will not work.** It is better to know this now than to find out halfway through Phase 6.

### What you can do on each setup

| Your setup | Phases 1–5 (learning, RAG) | Phases 6–9 (fine-tuning) | QLoRA / 4-bit |
|---|---|---|---|
| **Mac (Apple Silicon)**, 16GB+ | Yes, works well | Yes, small models only, slow | No |
| **Google Colab free** (T4) | Yes | Yes, 1–3B models | Yes |
| **Colab Pro / rented GPU** | Yes | Yes, 3–8B models | Yes |
| **CPU only, no GPU** | Small models only | No | No |

### What to do

Do Phases 1–5 on your own computer — it is mostly reading, tokenizing, and running small models. Starting at Phase 6, **use Google Colab.** The free version is enough to finish this roadmap.

Keep your dataset saved in your project folder, not only inside a notebook. Notebooks are easy to lose.

**If you are on a Mac** and want to train locally anyway, there is a separate Apple tool called MLX (`pip install mlx-lm`). It works well on Mac, but it is a different set of tools than this roadmap teaches, so you would be learning two things at once.

---

## 5. Roadmap phases

### Phase 1: Foundations

#### Goal
Understand how neural networks learn.

#### Topics
- Python basics
- NumPy and PyTorch basics
- tensors and how to work with them
- training a simple neural network
- loss and optimization
- supervised vs unsupervised learning

#### Done when
You can write a small training loop yourself and explain what each line does: forward pass, loss, `backward()`, optimizer step, and zeroing gradients.

#### Resources
- [Karpathy, "Neural Networks: Zero to Hero"](https://karpathy.ai/zero-to-hero.html) — start with the micrograd video
- [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**Rough time:** 2–4 weeks if you are new to machine learning.

---

### Phase 2: NLP and Transformer Basics

#### Goal
Learn the pieces that modern LLMs are built from.

#### Topics
- text preprocessing
- tokenizers
- embeddings
- positional encoding
- attention
- transformer architecture
- encoder vs decoder vs encoder-decoder

#### Done when
You can explain how a sentence travels through a model: text → tokens → embeddings → attention layers → output scores → the next word. You can say what Q, K, and V mean in attention.

#### Resources
- [Karpathy, "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the most useful thing in this phase
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — pictures, very beginner friendly
- [HF NLP Course, chapters 1–2](https://huggingface.co/learn/nlp-course)

**Rough time:** 1–2 weeks.

---

### Phase 3: Hugging Face Basics

#### Goal
Actually load and run a real model.

#### Topics
- loading a pretrained model
- loading a tokenizer
- using `AutoTokenizer` and `AutoModelForCausalLM`
- running a small model
- using `pipeline()` (the easiest way to start)
- generation settings:
  - `max_new_tokens` — how long the answer can be
  - `temperature` — higher means more random and creative
  - `top_p` — another way to control randomness
  - `do_sample` — whether to be random at all

#### Done when
You can download a model from the Hugging Face Hub, give it a prompt, get an answer back, and explain the difference between a model and a tokenizer.

Try this: set `temperature` to 0.1, then to 1.5, and see how the answers change.

#### Resources
- [HF LLM Course](https://huggingface.co/learn/llm-course)
- [Text generation guide](https://huggingface.co/docs/transformers/generation_strategies)

**Rough time:** 3–5 days.

---

### Phase 4: Chat Templates

#### Goal
Learn how to format prompts the way each model expects.

#### Why this matters
Every chat model was trained with a specific format — special markers around the user's message and the assistant's reply. If you write your prompt as plain text and skip that format, the model may ignore your instructions or give strange answers.

**This is one of the most common beginner problems, and it is easy to fix.**

#### The one thing to learn
`tokenizer.apply_chat_template()`

It formats your messages correctly for whichever model you loaded. You do not have to know the format yourself.

#### Topics
- messages with roles (system, user, assistant)
- how the format differs between models
- writing messages as a list:
  ```python
  messages = [
      {"role": "user", "content": "Explain transformers in simple terms."}
  ]
  ```

#### Done when
You use chat templates instead of writing raw prompt text by hand.

Try this to see what it actually does:
```python
print(tokenizer.apply_chat_template(messages, tokenize=False))
```
Run it with two different models and compare. The special tokens will be different.

**Rough time:** 1–2 days. Short phase, big payoff.

---

### Phase 4.5: Do you even need fine-tuning?

Before spending a week training, check the two easier options first. Most beginner projects that sound like they need fine-tuning actually don't.

#### Try in this order

**1. Write a better prompt.** A clear system prompt plus 3–5 examples inside the prompt solves a lot of problems. It costs nothing and takes minutes. Always try this first.

**2. Use RAG.** If the problem is that the model *doesn't know your information* — your documents, your product list, your notes — then you need RAG, not fine-tuning. That is Phase 5.

**3. Fine-tune.** This is the right choice when you need to change how the model *behaves*: always answering in a certain format, using a certain tone, or doing a task it currently does badly.

#### The rule that saves the most time

> **Fine-tuning teaches the model how to behave. RAG gives the model facts to use.**

Trying to teach a model new facts by fine-tuning is the most common beginner mistake. The model learns what your answers *look like* and then makes up details that sound right but are wrong.

So if your goal is "the model should know what's in our documents," the answer is RAG.

---

### Phase 5: RAG (Retrieval-Augmented Generation)

#### Goal
Let a model answer questions about your own documents, without training it.

#### How it works, simply
1. Split your documents into small chunks
2. Turn each chunk into numbers (an embedding) and store them
3. When a question comes in, find the chunks most similar to it
4. Paste those chunks into the prompt and ask the model to answer using them

That's it. No training involved.

#### Why this comes before fine-tuning
It is faster, cheaper, and you can update your documents anytime without retraining. For most "answer questions about my stuff" projects, this is the correct answer.

#### Topics
- embedding models (`sentence-transformers`, try `all-MiniLM-L6-v2` first)
- similarity between vectors
- chunking — how big to make the pieces, and why it matters
- vector stores — FAISS is the simplest to start with
- the retrieve → add to prompt → generate loop

#### Done when
You have built a small Q&A system over a folder of your own documents, and you understand why chopping documents up badly leads to bad answers even with a good model.

#### Resources
- [Sentence Transformers docs](https://www.sbert.net/)
- [FAISS with `datasets`](https://huggingface.co/docs/datasets/faiss_es)

**Rough time:** 1 week.

---

### Phase 6: Fine-Tuning Basics

#### Goal
Understand the difference between *using* a model and *changing* it.

#### Topics
- what a pretrained model is
- what a task dataset is
- supervised fine-tuning (SFT)
- train / validation / test split
- overfitting
- learning rate and epochs

#### The question to answer first
**What exactly am I teaching the model to do?**

Some examples:
- decide if a review is positive or negative
- answer questions about one specific topic
- summarize documents
- always reply in a certain style
- follow instructions more reliably

#### Done when
You can say your goal in one sentence, and explain how you would check whether it worked.

**Rough time:** 2–3 days.

---

### Phase 7: Building Your Dataset

#### Goal
Build the examples you will train on.

#### Why this phase matters most
Your data matters far more than your model choice or your settings. A clean set of 500 examples usually beats a messy set of 50,000.

When a fine-tuned model turns out bad, the cause is almost always the data — not the training settings. Beginners usually spend too long adjusting settings and not enough time fixing examples.

#### The two data formats

**Chat format** — best for chat models, because `trl` handles the template for you:
```python
{"messages": [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What does attention do?"},
    {"role": "assistant", "content": "It lets each word look at every other word."}
]}
```

**Prompt/completion format** — simpler, fine for non-chat models:
```python
{"prompt": "What does attention do?", "completion": "It lets each word look at every other word."}
```

Pick one and use it for your whole dataset. Don't mix them.

#### Topics
- loading data from JSONL, CSV, or the Hub with `datasets`
- writing your own examples from scratch
- cleaning: removing duplicates, dropping bad examples, fixing formatting
- splitting into train / validation / test
- **reading your own data by hand** — this is not optional
- making examples varied, so the model doesn't learn one repeated pattern

#### How many examples do you need?

| Your goal | Roughly how many |
|---|---|
| Consistent format or style | 200–1,000 |
| One narrow task done well | 1,000–5,000 |
| General instruction following | 10,000+ |
| Teaching new facts | Wrong tool — use RAG (Phase 5) |

Start small. Add more only if testing shows you need it.

#### Done when
You have a cleaned, split dataset saved as a JSONL file — **and you have personally read at least 50 of your own examples.**

#### Resources
- [`datasets` docs](https://huggingface.co/docs/datasets)
- [TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)

**Rough time:** 1–2 weeks. Expect this to take longer than the training itself.

---

### Phase 8: The Training Workflow

#### Goal
Learn the tools people actually use to fine-tune models today.

#### Topics
- the `trl` library
- `SFTTrainer` — runs the training for you
- `SFTConfig` — your training settings
- LoRA / PEFT
- QLoRA

#### The key idea
You do **not** retrain the whole model. Modern training works like this:
- freeze almost all of the original model
- add small new pieces (adapters) and train only those
- save the adapters, and combine them with the model later

This is why fine-tuning is possible on a free Colab GPU.

#### Your goal for this phase
Train a small model with LoRA on a small dataset of your own.

#### Done when
Your training script runs start to finish without errors, and your loss graph goes down. Check it in TensorBoard or W&B.

#### Resources
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)

**Rough time:** 1 week.

---

### Phase 9: Prompt Loss Masking (important)

#### Goal
Make sure the model learns to write *answers*, not to repeat *questions*.

#### The problem, in plain terms
During training, the model sees your whole example — the question and the answer together. By default it tries to learn to predict all of it, including the question part.

But you don't want a model that's good at writing questions. You want one that's good at writing answers.

**Masking** means telling the trainer: "ignore the question part when scoring, only grade the answer."

#### What to learn
- `DataCollatorForCompletionOnlyLM`

#### Done when
You understand why we mask the prompt, and you can check that it worked. When masking is on, the question tokens get a label of `-100`, which means "skip this."

**Rough time:** 2–3 days.

---

### Phase 10: LoRA and PEFT

#### Goal
Learn the cheap training method that makes all of this possible on small hardware.

#### The core idea
- **Full fine-tuning** changes every number in the model. Needs a lot of memory.
- **LoRA** freezes the model and adds small extra pieces to train instead. Needs far less.
- **PEFT** is the Hugging Face library that does this for you.
- **QLoRA** is LoRA plus compression, for even less memory (NVIDIA GPUs only).

LoRA usually trains **less than 1%** of the model's numbers and still changes its behavior a lot.

#### Words you will see
- **adapter** — the small trainable piece LoRA adds
- **rank (`r`)** — how big that piece is
- **alpha** — how strongly the adapter affects the model
- **target modules** — which parts of the model get adapters
- **merge and unload** — combining the adapter back into the model

#### Settings to start with
Don't tune these at first. Use these and change them only if you have a reason:

- `r`: 16
- `lora_alpha`: 32 (a common rule is alpha = 2 × r)
- `lora_dropout`: 0.05
- `target_modules`: `"all-linear"`
- learning rate: `2e-4` (higher than normal fine-tuning — this is expected)

#### Done when
You can explain what rank controls, and why training under 1% of a model still changes how it behaves.

#### Resources
- [PEFT docs](https://huggingface.co/docs/peft)

**Rough time:** 4–6 days.

---

### Phase 11: Fine-tune a Small Model

#### Goal
Put it all together and actually train something.

#### Good first tasks
- sentiment classification
- answering questions about one narrow topic
- summarizing
- turning text into structured JSON
- a chat assistant for one specific job

#### Which model to pick
Start between **0.5B and 3B parameters**. Good choices right now:

- **Qwen3 0.6B / 1.7B** — strong for its size, good chat support
- **Llama 3.2 1B / 3B** — lots of tutorials available
- **Gemma 3 1B** — solid and well supported
- **SmolLM2 360M / 1.7B** — made by Hugging Face for learning

**Avoid GPT-2** for this. It is older than chat templates and has no chat version, so you would learn habits you'd have to unlearn.

> This list will go out of date — new models come out constantly. Check [trending models on Hugging Face](https://huggingface.co/models?sort=trending) and filter by size.

#### The steps
1. Pick a small base model
2. Prepare your dataset
3. Format examples as question/answer pairs
4. Apply the chat template
5. Train with LoRA and `SFTTrainer`
6. Save the adapter
7. Test the output
8. Merge and export

#### Done when
Your trained model behaves differently from the original on your task — and you have checked both on the same prompts to prove it.

**Rough time:** 1–2 weeks including retries. Expect a few failed runs. That's normal.

#### What comes next (just so you know it exists)

After SFT, real production models get another round of training on *preference* data — pairs showing a better and a worse answer to the same question. The main method is called **DPO**, and `trl` supports it with `DPOTrainer`.

**You do not need this to finish this roadmap.** Just know the word. Come back to it once SFT works and your only remaining complaints are about polish.

---

### Phase 12: Testing Your Model Properly

#### Goal
Find out if your model is actually better, not just different.

#### Why loss is not enough
A lower training loss does not mean a more useful model. A model can score well during training and still give bad answers.

#### The simplest test that works

Do this before anything fancy. It takes about an hour:

1. Set aside 30–50 examples the model never saw in training
2. Get answers from **both** the original model and your trained one
3. Put them side by side
4. Score each yourself from 1 to 5

**If your trained model doesn't clearly win, more training won't fix it.** Go back and improve your data.

#### Things to check
- Are the answers correct?
- Does it follow the format you wanted?
- Does it make things up less often?
- Would this actually be useful to someone?

#### Later, if you want scores on standard benchmarks
- [lighteval](https://github.com/huggingface/lighteval)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

#### Done when
You have a held-out test set, a side-by-side comparison against the original model, and you can say specifically what got better.

**Rough time:** 4–6 days.

---

### Phase 13: Export and Merge

#### Goal
Turn your trained adapter into a normal model file you can use anywhere.

#### Steps
- save the adapter weights
- merge them into the base model with `model.merge_and_unload()`
- save the finished model to disk
- optionally push it to the Hugging Face Hub

#### Making it smaller (quantization)

These two sound similar but are different:

- **QLoRA** (Phase 10) shrinks the model **while training**, so training fits in memory.
- **Quantization here** shrinks the **finished** model, so it runs on a normal computer.

Merge your adapter first, then shrink. Common sizes in GGUF format:

| Level | Size | Quality | When to use |
|---|---|---|---|
| `Q8_0` | ~50% | Almost perfect | You have plenty of memory |
| `Q5_K_M` | ~35% | Very good | If Q4 feels worse |
| `Q4_K_M` | ~28% | Good | **Start here** — best balance |
| `Q3_K_M` | ~22% | Noticeably worse | Only if memory is tight |
| `Q2_K` | ~16% | Poor | Usually not worth it |

Use `Q4_K_M` unless you have a reason not to. Note that small models suffer more from shrinking than big ones — a 1B model at `Q2_K` is often too damaged to use.

#### Done when
You have a merged model saved on disk, and either uploaded it to the Hub or converted it to GGUF.

**Rough time:** 2–3 days.

---

### Phase 14: Running It Locally

#### Goal
Use your model outside a notebook.

#### Which tool to pick
- **Ollama** — easiest by far. Give it your GGUF file and you get a working local chat. **Start here.**
- **llama.cpp** — what Ollama runs on. Use it directly if you want more control.
- **vLLM** — for serving many users at once. Overkill for personal use.
- **HF `pipeline`** — fine inside scripts, not meant for serving.

#### Done when
Your model answers a question from somewhere other than a notebook — a terminal, a script, or a small local app.

**Rough time:** 3–5 days.

---

## 6. When something breaks

Most beginner problems are not deep — they are small technical errors. Check here first.

| What you see | What it usually means | How to fix it |
|---|---|---|
| `CUDA out of memory` | Batch too big for your GPU | Set `per_device_train_batch_size=1`, raise `gradient_accumulation_steps`, turn on `gradient_checkpointing`, lower `max_seq_length` |
| `Asking to pad but the tokenizer does not have a padding token` | Model has no pad token | `tokenizer.pad_token = tokenizer.eos_token` |
| Loss drops to 0 almost instantly | Model is copying the prompt | Check your masking (Phase 9) — prompt tokens should have label `-100` |
| Loss shows `nan` | Learning rate too high, or number format issue | Lower the learning rate 10×; use `bf16=True` instead of `fp16=True` |
| Model repeats itself forever | No end token in training data | Make sure examples end with the EOS token; try `repetition_penalty=1.1` |
| You see `<\|im_start\|>` in the output | Chat template applied twice | `trl` applies it for you — don't also do it yourself |
| Trained model acts exactly like the original | Adapter not loaded, or LoRA hit nothing | Check `print_trainable_parameters()` is not zero; check `target_modules` |
| Shape mismatch when loading adapter | Adapter belongs to a different base model | Adapters only work with the exact model they were trained on |
| Training works but answers are bad | Almost always the data | Re-read 50 of your examples before touching any setting |
| `bitsandbytes` fails on Mac | It needs NVIDIA | Expected — see Section 4. Use Colab |

---

## 7. Common beginner mistakes

- picking too large a model too early
- **fine-tuning to teach facts, when RAG was the right tool** — the most expensive mistake here
- writing prompts by hand instead of using chat templates
- training on messy or inconsistent data
- never actually reading your own dataset
- forgetting to mask the prompt during training
- judging your model only by training loss
- skipping the validation split
- doing full fine-tuning when LoRA would work
- changing training settings over and over when the real problem is the data
- never testing the original model first, so you can't tell what changed

---

## 8. Study order and strategy

The 14 phases group into four stages:

**Stage A — Understand (Phases 1–2).** Python, PyTorch, tensors, training loops, tokenization, attention. Mostly reading and small experiments.

**Stage B — Use (Phases 3–5).** Load real models, generate text, use chat templates, build a RAG demo. This is where you find out whether you need fine-tuning at all.

**Stage C — Train (Phases 6–11).** Fine-tuning concepts, building your dataset, `SFTTrainer`, LoRA, masking, then a real trained model. Most of your time here goes to data, not training.

**Stage D — Check and ship (Phases 12–14).** Test properly, merge and export, run locally.

**Don't jump straight to Stage C.** The usual failure is training a big model on data nobody looked at, getting a bad result, and having no idea why.

---

## 9. Project ideas

Pick **one** project and carry it through the whole roadmap. One finished project teaches more than five abandoned ones.

| Project | Right approach | Why |
|---|---|---|
| Sentiment classifier for reviews | Fine-tune | Narrow task, easy to score |
| Summarizer for your notes | Fine-tune | Teaching a consistent style |
| Assistant with a specific tone | Fine-tune | This is behavior |
| Text → JSON extraction | Fine-tune | Format following, ideal first project |
| Q&A over your documents | **RAG** | This is facts, not behavior |
| Chat over a knowledge base | **RAG** (plus light fine-tune for tone) | Facts need to stay updatable |

Keep it small and realistic.

---

## 10. Milestone checklist

- [ ] explain what a transformer is and what attention does
- [ ] load a model and tokenizer from the Hub
- [ ] generate text and explain the settings
- [ ] format messages with `apply_chat_template()`
- [ ] build a working RAG demo over your own files
- [ ] explain when to use RAG vs fine-tuning vs a better prompt
- [ ] build, clean, and split a dataset — and read it yourself
- [ ] fine-tune a small model with LoRA
- [ ] use `SFTTrainer`
- [ ] apply prompt loss masking and verify it
- [ ] watch a training run in TensorBoard or W&B
- [ ] compare your model against the original on a test set
- [ ] merge and export your adapter
- [ ] convert to GGUF and run it locally

---

## 11. Coming later to this repo

Runnable examples will be added:

- Example 1: loading and running a model
- Example 2: using chat templates
- Example 3: a simple RAG demo
- Example 4: preparing and cleaning a dataset
- Example 5: fine-tuning with LoRA and `SFTTrainer`
- Example 6: testing against a held-out set
- Example 7: merge, shrink, and run locally

---

## 12. The short version

The beginner path is **not** "download a huge model and hope it works." It is:

1. Understand transformers and the Hugging Face tools
2. Check whether a better prompt or RAG already solves your problem
3. If not, build a small clean dataset — this is most of the work
4. Fine-tune a small model with LoRA
5. Compare against the original model, not against the loss number
6. Merge, shrink, and run it locally

Start with a small model, a small dataset, a clear task, and honest testing. That is how you get good at this.
