"""Load and format the medical Q&A dataset for QLoRA fine-tuning.

Dataset: ``lavita/ChatDoctor-HealthCareMagic-100k`` — instruction-formatted
patient questions with doctor answers. We map each row to a single Mistral
training string and hold out a fixed slice for evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset

from prompt import build_training_text

DATASET_ID = "lavita/ChatDoctor-HealthCareMagic-100k"
SEED = 42


@dataclass
class SplitConfig:
    train_size: int = 8000  # subsample so a free T4 finishes in ~2-4h
    eval_size: int = 500
    min_answer_chars: int = 20  # drop near-empty answers


def _row_to_text(row: dict) -> dict:
    # ChatDoctor fields: "input" (patient question), "output" (doctor answer).
    # "instruction" is a generic system-ish line we ignore in favour of our own.
    question = (row.get("input") or "").strip()
    answer = (row.get("output") or "").strip()
    return {"text": build_training_text(question, answer), "question": question, "answer": answer}


def load_splits(cfg: SplitConfig | None = None) -> DatasetDict:
    """Return a DatasetDict with ``train`` and ``eval`` splits, formatted + deduped."""
    cfg = cfg or SplitConfig()
    raw = load_dataset(DATASET_ID, split="train")

    mapped = raw.map(_row_to_text, remove_columns=raw.column_names)
    mapped = mapped.filter(lambda r: len(r["answer"]) >= cfg.min_answer_chars and len(r["question"]) > 0)

    # Dedup on the question text.
    seen: set[str] = set()
    keep_idx: list[int] = []
    for i, q in enumerate(mapped["question"]):
        key = q.lower()
        if key not in seen:
            seen.add(key)
            keep_idx.append(i)
    mapped = mapped.select(keep_idx)

    shuffled = mapped.shuffle(seed=SEED)
    total_needed = cfg.train_size + cfg.eval_size
    shuffled = shuffled.select(range(min(total_needed, len(shuffled))))

    eval_ds = shuffled.select(range(cfg.eval_size))
    train_ds = shuffled.select(range(cfg.eval_size, len(shuffled)))
    return DatasetDict(train=train_ds, eval=eval_ds)


def smoke_dataset(n: int = 8) -> Dataset:
    """Tiny in-memory dataset for CI / local wiring tests (no network)."""
    rows = [
        {
            "question": f"What are common symptoms of condition {i}?",
            "answer": f"Common symptoms may include fatigue and discomfort (example {i}). "
            "Please consult a clinician.",
        }
        for i in range(n)
    ]
    for r in rows:
        r["text"] = build_training_text(r["question"], r["answer"])
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect the formatted dataset.")
    ap.add_argument("--train-size", type=int, default=SplitConfig.train_size)
    ap.add_argument("--eval-size", type=int, default=SplitConfig.eval_size)
    args = ap.parse_args()

    ds = load_splits(SplitConfig(train_size=args.train_size, eval_size=args.eval_size))
    print(f"train={len(ds['train'])}  eval={len(ds['eval'])}")
    print("\n--- example training text ---\n")
    print(ds["train"][0]["text"][:600])
