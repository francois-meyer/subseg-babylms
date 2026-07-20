"""
Shared utilities for preparing SubSegGPT/SubSegDeBERTa submissions to the 2026 BabyLM challenge.
"""

import os
from pathlib import Path

from transformers import TrainerCallback

WORDS_PER_TRACK = {"strict-small": 10_000_000, "strict": 100_000_000, "multilingual": 100_000_000}


def load_multilingual_dataset(data_dir):
    # Read multilingual mixture dir into an HF Dataset with a "text" column.
    # Mirrors babylm-baseline's load_texts_from_dir(): English *.train.txt -> one row per
    # non-empty line; nld/zho *.train.parquet -> one row per "text" cell. Row order is
    # irrelevant since the HF Trainer shuffles the train set.
    import pandas as pd
    from datasets import Dataset

    rows = []
    for f in sorted(Path(data_dir).glob("*.train.txt")):
        rows.extend(ln for ln in f.read_text().splitlines() if ln.strip())
    for f in sorted(Path(data_dir).glob("*.train.parquet")):
        rows.extend(t for t in pd.read_parquet(f)["text"].tolist() if t and t.strip())
    return Dataset.from_dict({"text": rows})


def word_milestone_label(words):
    # Label for a word-count milestone, e.g. 1_000_000 -> "1M", 1_000_000_000 -> "1B".
    if words < 1_000_000_000:
        return f"{words // 1_000_000}M"
    return f"{words // 1_000_000_000}B"


def compute_word_checkpoint_steps(max_steps, total_words):
    # Map word-count milestones to training steps. Milestones: every 1M to 10M, every 10M to 
    # 100M, every 100M to 1B.
    milestones = []
    milestones += range(1_000_000, min(10_000_000, total_words) + 1, 1_000_000)
    milestones += range(10_000_000, min(100_000_000, total_words) + 1, 10_000_000)
    milestones += range(100_000_000, min(1_000_000_000, total_words) + 1, 100_000_000)
    words_per_step = total_words / max_steps
    steps = {}
    for w in sorted(set(milestones)):
        step = min(round(w / words_per_step), max_steps)
        if step > 0:
            steps[step] = word_milestone_label(w)
    return steps


class WordMilestoneCheckpointCallback(TrainerCallback):

    def __init__(self, total_words, tokenizer):
        self.total_words = total_words
        self.tokenizer = tokenizer
        self.checkpoint_steps = {}

    def on_train_begin(self, args, state, control, **kwargs):
        # Collect training steps corresponding to word-count milestones, now that Trainer has set 
        # up the schedule and max_steps is known.
        self.checkpoint_steps = compute_word_checkpoint_steps(state.max_steps, self.total_words)
        if state.is_world_process_zero:
            print(f"Word milestone checkpoints at steps: {self.checkpoint_steps}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Save checkpoint when a word-count milestone is reached.
        label = self.checkpoint_steps.get(state.global_step)
        if label is None or not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(args.output_dir, f"epoch_{label}")  # match what upload_checkpoints.py expects
        model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        print(f"  Saved word-count milestone checkpoint {label} (step {state.global_step}) -> {ckpt_dir}")
