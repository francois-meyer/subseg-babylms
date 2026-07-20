"""
Train SubSegDeBERTa on BabyLM corpus.
Adapted from github.com/babylm-org/multilingual-training/train.py for SubSegDeBERTa.
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

from subsegdeberta import (
    SubSegDeBERTaConfig,
    SubSegDeBERTaForMaskedLM,
    SubSegDeBERTaTokenizer,
    DataCollatorForSubSegDeBERTa,
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

    tokenizer = SubSegDeBERTaTokenizer.train_from_iterator(
        text_iterator(),
        max_seg_len=args.max_seg_len,
        lex_min_count=args.lex_min_count,
        lex_vocab_size=args.lex_vocab_size,
    )

    print(f"  Character vocab size: {tokenizer.vocab_size}")
    print(f"  Lexicon size: {tokenizer.lexicon_size}")
    return tokenizer


def pack_words_into_features(examples, tokenizer, block_size):
    """
    Concatenates words across examples into blocks of at most ``block_size` characters, 
    breaking only at whitespace so words aren't split.
    """
    all_input_ids, all_word_ids = [], []
    buf, buf_len = [], 0  # words in the current block

    def flush(buf):
        if buf:
            enc = tokenizer.encode_for_mlm(" ".join(buf))
            all_input_ids.append(enc["input_ids"])
            all_word_ids.append(enc["word_ids"])

    for text in examples["text"]:
        if not (text and text.strip()):
            continue
        for w in text.split():
            if len(w) > block_size:  # truncate word longer than block_size
                w = w[:block_size]
            add = len(w) + (1 if buf else 0)  # +1 for the joining space if not first word
            if buf and buf_len + add > block_size:
                flush(buf)
                buf, buf_len = [w], len(w)
            else:
                buf.append(w)
                buf_len += add
    flush(buf)
    return {"input_ids": all_input_ids, "word_ids": all_word_ids}


def _pack_split(split, tokenizer, block_size, desc):
    """
    Converts text to blocks of ids used for MLM.

    Args:
        split: HF Dataset with a "text" column (one row per document).
        tokenizer: SubSegDeBERTaTokenizer instance.
        block_size: Max chars per packed block (sequence).
        desc: Progress-bar.

    Returns:
        HF Dataset with two columns and each item is two parallel lists of equal length:
            - input_ids is char ids wrapped with [CLS] ... [SEP]
            - word_ids tracks word position in the sequence (-1 for [CLS]/[SEP] and whitespace)          
    """
    return split.map(
        lambda ex: pack_words_into_features(ex, tokenizer, block_size),
        batched=True, batch_size=1000,
        remove_columns=split.column_names, desc=desc,
    )


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
    # Create descriptive W&B run name, e.g. ``SubSegDeBERTa-strict-small_lr=5e-5,bs=16,mlm=0.15,...``.
    data = args.track
    hp = [
        f"lr={args.learning_rate}", f"bs={args.batch_size}",
        f"ga={args.gradient_accumulation_steps}", f"block={args.block_size}",
        f"seg={args.max_seg_len}", f"lex={args.lex_vocab_size}",
        f"mlm={args.mlm_probability}", f"sched={args.lr_scheduler_type}",
        f"wu={args.warmup_ratio}", f"wd={args.weight_decay}",
        f"drop={args.dropout}", f"norm={args.loss_normalization}", f"ep={args.epochs}",
    ]
    return f"SubSegDeBERTa-{data}_" + ",".join(hp)


def main():
    parser = argparse.ArgumentParser(description="Train SubSegDeBERTa on BabyLM")
    parser.add_argument("--track", type=str, default="strict-small",
                   choices=["strict-small", "strict", "multilingual"])
    parser.add_argument("--multilingual_data_dir", type=str, default=None,
                   help="Path to the built multilingual mixture dir (required for --track multilingual)")
    parser.add_argument("--output_dir", type=str, default="./output/subsegdeberta-babylm-10m")

    # SubSegDeBERTa architectire
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--word_context_hidden_size", type=int, default=768)
    parser.add_argument("--char_decoder_hidden_size", type=int, default=256)
    parser.add_argument("--char_embed_dim", type=int, default=128)
    parser.add_argument("--max_position_embeddings", type=int, default=1024)
    parser.add_argument("--max_seg_len", type=int, default=5)
    parser.add_argument("--lex_vocab_size", type=int, default=10000)
    parser.add_argument("--lex_min_count", type=int, default=5)

    # Training    
    parser.add_argument("--loss_normalization", choices=["word", "char"], default="word")
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--max_word_len", type=int, default=20, help="Words longer than this (chars) never masked")
    parser.add_argument("--block_size", type=int, default=1022,)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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

    if args.block_size > args.max_position_embeddings - 2:
        raise ValueError(
            f"block_size ({args.block_size}) must be <= max_position_embeddings - 2 "
            f"({args.max_position_embeddings - 2}) to leave room for [CLS]/[SEP]."
        )

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
        tokenizer = SubSegDeBERTaTokenizer.from_pretrained(args.output_dir)
        print(f"  Character vocab size: {tokenizer.vocab_size}")
        print(f"  Lexicon size: {tokenizer.lexicon_size}")
    else:
        tokenizer = train_tokenizer(dataset["train"], args)
        tokenizer.save_pretrained(args.output_dir)
    tokenizer.model_max_length = args.max_position_embeddings

    # Create model
    print("Creating SubSegDeBERTa model...")
    config = SubSegDeBERTaConfig(
        vocab_size=tokenizer.vocab_size,
        n_alpha=tokenizer.n_alpha,
        lex_vocab_size=tokenizer.lexicon_size,
        max_seg_len=args.max_seg_len,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        word_context_hidden_size=args.word_context_hidden_size,
        char_decoder_hidden_size=args.char_decoder_hidden_size,
        char_embed_dim=args.char_embed_dim,
        max_position_embeddings=args.max_position_embeddings,
        dropout=args.dropout,
        loss_normalization=args.loss_normalization,
        mlm_probability=args.mlm_probability,
    )
    model = SubSegDeBERTaForMaskedLM(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Data collator
    collator = DataCollatorForSubSegDeBERTa(
        tokenizer, mlm_probability=args.mlm_probability, max_word_len=args.max_word_len
    )

    # Pack corpus into character blocks across examples.
    has_eval = True
    print(f"Packing character blocks across examples (block_size={args.block_size})...")
    train_dataset = _pack_split(dataset["train"], tokenizer, args.block_size, "Packing train")
    eval_dataset = _pack_split(dataset["validation"], tokenizer, args.block_size, "Packing val") if has_eval else None
    print(f"  Train blocks: {len(train_dataset)}"
          + (f", Eval blocks: {len(eval_dataset)}" if eval_dataset is not None else ""))

    # Training arguments
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
        run_name=run_name,
        bf16=args.bf16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=1,  # keep one rolling checkpoint just for job resume (load_best keeps best too)
        load_best_model_at_end=save_best,
        metric_for_best_model="eval_loss" if save_best else None,
        eval_strategy="epoch" if has_eval else "no",
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,        
        remove_unused_columns=False,  # Don't drop `word_ids` (collator uses it).
        label_names=["word_lens"],  # Loss is actually computed in model without labels.
                                    # Pass something as labels so eval loss is reported.
        prediction_loss_only=True,  # for eval efficiency
    )

    # Train
    words_per_epoch = WORDS_PER_TRACK[args.track]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
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
