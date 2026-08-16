"""
llmkit - small helpers used across every phase.

Import from any example:

    import sys; sys.path.insert(0, "tools")
    from llmkit import section, pick_device, Timer, TrainingMonitor

Nothing here is magic. Read it - it's all short.
"""

import json
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------------


def section(title, width=60):
    """A labelled divider, so long output stays readable."""
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")


def bar(value, maximum, width=30, char="#"):
    """Text bar chart. Useful for probabilities and scores."""
    if maximum <= 0:
        return ""
    filled = int((value / maximum) * width)
    return char * max(1, filled) if value > 0 else ""


def table(rows, headers):
    """Print aligned rows. rows = list of tuples, all converted to str."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]

    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


# ----------------------------------------------------------------------------
# hardware
# ----------------------------------------------------------------------------


def pick_device(verbose=False):
    """
    Return the best available torch device name.

    mps  = Apple Silicon GPU
    cuda = NVIDIA GPU
    cpu  = always works, slowest
    """
    import torch

    if torch.backends.mps.is_available():
        device, note = "mps", "Apple Silicon GPU"
    elif torch.cuda.is_available():
        device, note = "cuda", torch.cuda.get_device_name(0)
    else:
        device, note = "cpu", "no GPU found"

    if verbose:
        print(f"device: {device}  ({note})")
    return device


def memory_used_mb():
    """Rough process memory in MB. Works without extra packages."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def model_size(model):
    """Parameter counts and approximate memory for a torch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_used = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total else 0.0,
        "mb": bytes_used / (1024**2),
    }


def print_model_size(model, label="model"):
    info = model_size(model)
    print(f"  {label}:")
    print(f"    total parameters : {info['total']:,}")
    print(f"    trainable        : {info['trainable']:,} ({info['trainable_pct']:.2f}%)")
    print(f"    frozen           : {info['frozen']:,}")
    print(f"    weights in memory: {info['mb']:.1f} MB")
    return info


# ----------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------


class Timer:
    """
    with Timer("loading model"):
        ...
    """

    def __init__(self, label="elapsed", quiet=False):
        self.label = label
        self.quiet = quiet
        self.seconds = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        if not self.quiet:
            print(f"  [{self.label}: {self.seconds:.2f}s]")
        return False


def tokens_per_second(n_tokens, seconds):
    return n_tokens / seconds if seconds > 0 else 0.0


# ----------------------------------------------------------------------------
# training monitor
# ----------------------------------------------------------------------------


class TrainingMonitor:
    """
    Records train/validation loss and prints a live text chart.

    Deliberately dependency-free so it runs anywhere. TensorBoard and
    Weights & Biases do the same job with nicer graphs - see tools/README.md.

        mon = TrainingMonitor()
        mon.log(step, train_loss, val_loss)
        mon.summary()
        mon.plot()
    """

    def __init__(self, name="run"):
        self.name = name
        self.steps = []
        self.train = []
        self.val = []
        self.started = time.perf_counter()

    def log(self, step, train_loss, val_loss=None, quiet=False):
        self.steps.append(step)
        self.train.append(float(train_loss))
        self.val.append(float(val_loss) if val_loss is not None else None)

        if not quiet:
            line = f"  step {step:>5} | train {train_loss:8.4f}"
            if val_loss is not None:
                line += f" | val {val_loss:8.4f}"
                if self.is_overfitting():
                    line += "  <- val rising"
            print(line)

    @property
    def best_val(self):
        vals = [(v, s) for v, s in zip(self.val, self.steps) if v is not None]
        return min(vals) if vals else (None, None)

    def is_overfitting(self, patience=3, tolerance=1.02):
        """True if val loss has been above its best for `patience` logs."""
        vals = [v for v in self.val if v is not None]
        if len(vals) < patience + 1:
            return False

        best = min(vals[:-patience])
        return all(v > best * tolerance for v in vals[-patience:])

    def plot(self, width=54, height=12):
        """ASCII loss curve. Rough, but instant and needs nothing installed."""
        if len(self.train) < 2:
            print("  (need at least 2 points to plot)")
            return

        series = [("train", self.train, "*")]
        if any(v is not None for v in self.val):
            series.append(("val", [v for v in self.val if v is not None], "o"))

        everything = [v for _, s, _ in series for v in s]
        lo, hi = min(everything), max(everything)
        if hi == lo:
            hi = lo + 1e-9

        grid = [[" "] * width for _ in range(height)]

        for _, values, mark in series:
            for x in range(width):
                idx = int(x * (len(values) - 1) / max(width - 1, 1))
                norm = (values[idx] - lo) / (hi - lo)
                y = height - 1 - int(norm * (height - 1))
                grid[y][x] = mark

        print(f"\n  loss  {hi:.4f} +" + "-" * width + "+")
        for row in grid:
            print("        " + " " * 8 + "|" + "".join(row) + "|")
        print(f"        {lo:.4f} +" + "-" * width + "+")
        print(f"        {'':>8}  step {self.steps[0]}" + " " * (width - 16) + f"step {self.steps[-1]}")

        legend = "  * = train"
        if len(series) > 1:
            legend += "   o = validation"
        print(legend)

    def summary(self):
        elapsed = time.perf_counter() - self.started
        print(f"\n  run       : {self.name}")
        print(f"  logged    : {len(self.steps)} points over {elapsed:.1f}s")
        print(f"  train loss: {self.train[0]:.4f} -> {self.train[-1]:.4f}")

        best, best_step = self.best_val
        if best is not None:
            print(f"  best val  : {best:.4f} at step {best_step}")
            final = [v for v in self.val if v is not None][-1]
            if final > best * 1.05:
                print(f"  final val : {final:.4f}  <- WORSE than best")
                print(f"\n  Overfitting. The model peaked at step {best_step}.")
                print("  Fix: more data, stop earlier, or a smaller model.")
            else:
                print(f"  final val : {final:.4f}")

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "name": self.name,
                    "steps": self.steps,
                    "train": self.train,
                    "val": self.val,
                },
                indent=2,
            )
        )
        print(f"  saved to {path}")


# ----------------------------------------------------------------------------
# generation analysis
# ----------------------------------------------------------------------------


def top_predictions(logits, tokenizer, k=10):
    """
    Turn raw model scores into a readable top-k list.
    Returns [(token_text, probability), ...]
    """
    import torch

    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    return [
        (tokenizer.decode([tid]), prob)
        for prob, tid in zip(top.values.tolist(), top.indices.tolist())
    ]


def print_top_predictions(logits, tokenizer, k=10):
    rows = top_predictions(logits, tokenizer, k)
    biggest = rows[0][1] if rows else 1.0

    print(f"    {'token':<16} {'probability':>11}  ")
    print(f"    {'-' * 16} {'-' * 11}  {'-' * 24}")
    for text, prob in rows:
        print(f"    {text!r:<16} {prob:>10.1%}  {bar(prob, biggest, 24)}")
    return rows


def compare_outputs(label_a, text_a, label_b, text_b, width=70):
    """Print two model outputs side by side for eyeball comparison."""
    print(f"\n  --- {label_a} ---")
    for line in _wrap(text_a, width):
        print(f"  {line}")
    print(f"\n  --- {label_b} ---")
    for line in _wrap(text_b, width):
        print(f"  {line}")


def _wrap(text, width):
    import textwrap

    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out
