"""
Train SubSegGPT on BabyLM corpus. 
Adapted from github.com/babylm-org/multilingual-training/train.py for SubSegGPT.
"""

import glob
import os
import argparse

import torch
from datasets import load_dataset, DatasetDict

TRACK_DATASETS = {
    "strict-small": "BabyLM-community/BabyLM-2026-Strict-Small",
    "strict": "BabyLM-community/BabyLM-2026-Strict",
}
DEV_DATASET = "BabyLM-community/BabyLM-dev"

from subseggpt import (
    SubSegGPTConfig,
    SubSegGPTForCausalLM,
    SubSegGPTTokenizer,
    DataCollatorForSubSegGPT,
)
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from challenge_utils import WordMilestoneCheckpointCallback, WORDS_PER_TRACK, load_multilingual_dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def train_tokenizer(dataset, args):
    print("Training tokenizer...")
    def text_iterator():
        for example in dataset:
            text = example["text"]
            if text and text.strip():
                yield text

    tokenizer = SubSegGPTTokenizer.train_from_iterator(
        text_iterator(),
        max_seg_len=args.max_seg_len,
        lex_min_count=args.lex_min_count,
        lex_vocab_size=args.lex_vocab_size,
    )

    print(f"  Character vocab size: {tokenizer.vocab_size}")
    print(f"  Lexicon size: {tokenizer.lexicon_size}")
    return tokenizer


def tokenize_and_chunk(examples, tokenizer, block_size):
    # Tokenize all texts in this batch to character IDs
    all_ids = []
    for text in examples["text"]:
        if text and text.strip():
            tokens = list(text)
            ids = [tokenizer._convert_token_to_id(tok) for tok in tokens]
            ids.append(tokenizer.eos_token_id)
            all_ids.extend(ids)

    # Split into chunks, each prepended with BOS
    # Each block is [BOS, c_1, c_2, ..., c_{block_size-1}].
    # Original text boundaries are marked with EOS tokens.
    bos = tokenizer.bos_token_id
    content_len = block_size - 1  # Reserve position 0 for BOS
    chunks = []
    for i in range(0, len(all_ids) - content_len + 1, content_len):
        chunk = [bos] + all_ids[i : i + content_len]
        chunks.append(chunk)

    return {"input_ids": chunks}


def load_datasets(args):
    # Load train/validation datasets from HuggingFace BabyLM 2026.
    if args.track == "multilingual":
        if args.multilingual_data_dir is None:
            raise ValueError("--multilingual_data_dir is required for the multilingual track.")
        print(f"Loading built multilingual mixture from {args.multilingual_data_dir}")
        full = load_multilingual_dataset(args.multilingual_data_dir)
        # Unlike babylm-baseline, we hold out a small validation set for eval_loss monitoring.
        split = full.train_test_split(test_size=0.01, seed=args.seed)
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    train_repo = TRACK_DATASETS[args.track]
    print(f"Loading HuggingFace training dataset ({args.track}): {train_repo}")
    train_hf = load_dataset(train_repo, trust_remote_code=True)
    train_split = train_hf["train"] if "train" in train_hf else train_hf[next(iter(train_hf))]

    print(f"Loading HuggingFace dev dataset: {DEV_DATASET}")
    dev_hf = load_dataset(
        "text",
        data_files={"validation": f"hf://datasets/{DEV_DATASET}/*.dev"},
    )
    dev_split = dev_hf["validation"]

    return DatasetDict({"train": train_split, "validation": dev_split})


def build_run_name(args):
    # Create descriptive W&B run name, e.g. ``SubSegGPT-strict_lr=5e-5,bs=32,seg=5,...``."""
    data = args.track
    hp = [
        f"lr={args.learning_rate}", f"bs={args.batch_size}",
        f"ga={args.gradient_accumulation_steps}", f"block={args.block_size}",
        f"seg={args.max_seg_len}", f"lex={args.lex_vocab_size}",
        f"sched={args.lr_scheduler_type}", f"wu={args.warmup_ratio}",
        f"wd={args.weight_decay}", f"drop={args.dropout}", f"ep={args.epochs}",
    ]
    return f"SubSegGPT-{data}_" + ",".join(hp)


def main():
    parser = argparse.ArgumentParser(description="Train SubSegGPT on BabyLM")
    parser.add_argument("--track", type=str, default="strict-small",
                        choices=["strict-small", "strict", "multilingual"])
    parser.add_argument("--multilingual_data_dir", type=str, default=None,
                        help="Path to the built multilingual mixture dir (required for --track multilingual)")
    parser.add_argument("--output_dir", type=str)

    # SubSegGPT architecture
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--char_decoder_hidden_size", type=int, default=256)
    parser.add_argument("--char_decoder_num_layers", type=int, default=1)
    parser.add_argument("--char_embed_dim", type=int, default=128)
    parser.add_argument("--max_seg_len", type=int, default=5)
    parser.add_argument("--lex_vocab_size", type=int, default=10000)
    parser.add_argument("--lex_min_count", type=int, default=5)

    # Training
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    parser.add_argument("--hub_model_id", type=str, default=None)

    # Experiment tracking
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)

    args = parser.parse_args()
    print(args)

    # W&B
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = args.run_name or build_run_name(args)
    print(f"Run name: {run_name}")

    # Load dataset
    dataset = load_datasets(args)
    print(f"  Train examples: {len(dataset['train'])}", end="")
    print(f", Validation examples: {len(dataset['validation'])}")

    # Check for existing checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint is not None:
        print(f"Found checkpoint, will resume from: {last_checkpoint}")

    # Train or load tokenizer
    vocab_file = os.path.join(args.output_dir, "vocab.json")
    if last_checkpoint is not None and os.path.exists(vocab_file):
        print(f"Loading existing tokenizer from {args.output_dir}...")
        tokenizer = SubSegGPTTokenizer.from_pretrained(args.output_dir)
        print(f"  Character vocab size: {tokenizer.vocab_size}")
        print(f"  Lexicon size: {tokenizer.lexicon_size}")
    else:
        tokenizer = train_tokenizer(dataset["train"], args)
        tokenizer.save_pretrained(args.output_dir)
    tokenizer.model_max_length = args.block_size

    # Tokenize and chunk
    print(f"Tokenizing and chunking (block_size={args.block_size})...")
    tokenize_fn = lambda examples: tokenize_and_chunk(examples, tokenizer, args.block_size)
    train_dataset = dataset["train"].map(
        tokenize_fn, batched=True, batch_size=1000,
        remove_columns=dataset["train"].column_names, desc="Tokenizing train",
    )     
    eval_dataset = dataset["validation"].map(
        tokenize_fn, batched=True, batch_size=1000,
        remove_columns=dataset["validation"].column_names, desc="Tokenizing validation",
    )
    print(f"  Train chunks: {len(train_dataset)}", end="")
    print(f", Eval chunks: {len(eval_dataset)}")

    # Create model 
    print("Creating SubSegGPT model...")
    config = SubSegGPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_alpha=tokenizer.n_alpha,
        lex_vocab_size=tokenizer.lexicon_size,
        max_seg_len=args.max_seg_len,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        char_decoder_hidden_size=args.char_decoder_hidden_size,
        char_decoder_num_layers=args.char_decoder_num_layers,
        char_embed_dim=args.char_embed_dim,
        dropout=args.dropout,
    )
    model = SubSegGPTForCausalLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Data collator
    data_collator = DataCollatorForSubSegGPT(tokenizer=tokenizer)

    # Training arguments
    has_eval = True
    # Word-milestone checkpoints are the only saves needed for challenge.
    save_best = has_eval and args.save_strategy != "no"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=args.bf16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=1,  # keep one rolling checkpoint just for job resume (load_best keeps best too)
        load_best_model_at_end=save_best,
        metric_for_best_model="eval_loss" if save_best else None,
        eval_strategy="epoch" if has_eval else "no",
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        seed=args.seed,
        run_name=run_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    # Train
    words_per_epoch = WORDS_PER_TRACK[args.track]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[WordMilestoneCheckpointCallback(total_words=words_per_epoch * args.epochs, tokenizer=tokenizer)],
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        print(f"Pushing to hub: {args.hub_model_id}")
        trainer.push_to_hub()

    print("Done.")


if __name__ == "__main__":
    main()
