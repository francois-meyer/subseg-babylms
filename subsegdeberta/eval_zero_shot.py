"""
Zero-shot minimal pair evaluator for SubSegDeBERTa. 

Unlike for SubSegGPT, we cannot use the BabyLM eval pipeline directly for SubSegDeBERTa 
zero-shot eval, because it doesn't fit any of the pipeline's standard backends, all of 
which read vocab logits. This class implements the equivalent of pseudo log-likelihood 
for SubSegDeBERTa. It masks each word in a sentence, in turn, and sums the marginal 
word probabilities.
    
This script adapts the BabyLM eval pipeline code. It is based on
github.com/babylm-org/babylm-eval/blob/main/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py
and either directly calls or adapts methods in babylm-eval/blob/main/strict/evaluation_pipeline.

Usage: Invoked via run_eval_subsegdeberta.py zero_shot ... with the same args as the BabyLM pipeline. 
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM

# Reuse the official pipeline's data layer + reporting (imported, not copied).
from evaluation_pipeline.sentence_zero_shot.read_files import read_files
from evaluation_pipeline.sentence_zero_shot.compute_results import (
    update_subset_to_stats,
    rank_and_evaluate,
)
from evaluation_pipeline.sentence_zero_shot.run import (
    _parse_arguments,
    process_results,
    create_evaluation_report,
    save_predictions,
)

from subsegdeberta import SubSegDeBERTaTokenizer

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# For the hidden task, Global PIQA, candidates differ in length so the pipeline length-normalizes scores.
# We divide by the number of characters, since we marginalise over character seequences of words.
LENGTH_NORMALIZED_TASKS = {"global_piqa_parallel", "global_piqa_nonparallel"}

def build_word_masked_variants(tokenizer, sentence, completion, max_encoder_len=None):
    """
    Create masked variants of a sentence, masking each word in turn.
    Following BabyLM eval pipeline, only words overlapping the completion span are scored.

    Args:
        tokenizer: SubSegDeBERTaTokenizer instance.
        sentence: String of full candidate sentence (prefix + completion) to score.
        completion: String of suffix to score (only overlapping words overlapping are scored).
        max_encoder_len: Max length for encoder sequences (= encoder max_position_embeddings).

    Returns:
        variants: list of dicts, one per scored word, each with:
            - input_ids: char ids of the masked sentence ([CLS] ... [MASK] ... [SEP]).
            - slot: position of the [MASK] in input_ids (where the encoder context is read).
            - target: the masked word's char ids + <eow> (what the segmental DP scores).
            - word_len: len(target), the index at which to read the per-word marginal.
    """
    enc = tokenizer.encode_for_mlm(sentence)
    input_ids, word_ids = enc["input_ids"], enc["word_ids"]
    start_char = len(sentence) - len(completion)

    # Create dict of {word_id: [first_pos, last_pos]} from word_ids
    words = {}  
    for pos, wid in enumerate(word_ids):
        if wid < 0:
            continue
        if wid not in words:
            words[wid] = [pos, pos]
        else:
            words[wid][1] = pos

    eow_id = tokenizer.eow_token_id
    mask_id = tokenizer.mask_token_id
    variants = []
    for wid, (first, last) in words.items():
        # Get char index (in sentence) of word's last char (offset for [CLS] in words_ids/input_ids).
        last_char = last - 1
        # Score this word if is in the completion span.
        if last_char + 1 <= start_char:
            continue
        word_char_ids = input_ids[first:last + 1]
        masked_input = input_ids[:first] + [mask_id] + input_ids[last + 1:]
        slot = first  # [MASK] index

        # If masked sequence is longer than the encoder cap, keep a max_encoder_len window
        # centred on the [MASK], clamped to stay within the sequence.
        if max_encoder_len is not None and len(masked_input) > max_encoder_len:
            half_window = max_encoder_len // 2
            start = max(0, min(slot - half_window, len(masked_input) - max_encoder_len))
            masked_input = masked_input[start:start + max_encoder_len]
            slot = slot - start

        variants.append({
            "input_ids": masked_input,
            "slot": slot,
            "target": word_char_ids + [eow_id],
            "word_len": len(word_char_ids) + 1,  # +1 for EOW
        })
    return variants


def _collate_variants(variants, tokenizer):
    """
    Convert list of variants into a padded tensor batch (one masked word per row).
    """
    pad_id = tokenizer.pad_token_id
    num_masked_words = len(variants)
    max_seq_len = max(len(v["input_ids"]) for v in variants)
    max_masked_len = max(v["word_len"] for v in variants)

    input_ids = torch.full((num_masked_words, max_seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((num_masked_words, max_seq_len), dtype=torch.long)
    target_chars = torch.full((num_masked_words, max_masked_len), pad_id, dtype=torch.long)
    masked_pos = torch.zeros((num_masked_words, 2), dtype=torch.long)
    word_lens = torch.zeros((num_masked_words,), dtype=torch.long)

    for variant_num, variant in enumerate(variants):
        variant_len = len(variant["input_ids"])
        input_ids[variant_num, :variant_len] = torch.tensor(variant["input_ids"], dtype=torch.long)
        attention_mask[variant_num, :variant_len] = 1
        target_chars[variant_num, :variant["word_len"]] = torch.tensor(variant["target"], dtype=torch.long)
        masked_pos[variant_num, 0] = variant_num
        masked_pos[variant_num, 1] = variant["slot"]
        word_lens[variant_num] = variant["word_len"]

    # Lexicon windows are pure CPU string lookups (same as the training collator).
    lex_ids = tokenizer.compute_lex_ids(target_chars)
    batch = {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "masked_pos": masked_pos, "target_chars": target_chars,
        "word_lens": word_lens, "lex_ids": lex_ids,
    }
    return {k: t.to(DEVICE) for k, t in batch.items()}


@torch.no_grad()
def score_sentence(model, tokenizer, sentence, completion, args):
    """
    Computes pseudo log-likelihood of sentence under SubSegDeBERTa model.
    Returns sum log p(word | masked context) over all words in sentence. 
    """

    # Extract variants of sentences with each word masked in turn.
    tokenizer.max_seg_len = model.config.max_seg_len
    variants = build_word_masked_variants(
        tokenizer, sentence, completion, max_encoder_len=model.config.max_position_embeddings
    )
    if not variants:
        return 0.0

    # Compute per-variant log-probs in batches.
    batch_size = args.non_causal_batch_size
    total = 0.0
    for i in range(0, len(variants), batch_size):
        batch_variants = _collate_variants(variants[i:i + batch_size], tokenizer)
        out = model(**batch_variants)
        total += out.word_log_probs.sum().item()
    if args.task in LENGTH_NORMALIZED_TASKS:
        # Length-normalise for hidden task to match BabyLM eval pipeline.
        num_chars = sum(v["word_len"] for v in variants)
        total = total / max(num_chars, 1)
    return total


def compute_subseg_results(args, model, tokenizer, data, temperatures):
    """
    Equivalent of compute_mlm_results in github.com/babylm-org/babylm-eval/blob/main/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py
    but adapted for subword segmental masked language modelling.   
    """
    subset_to_stats = {t: {} for t in temperatures}
    predictions = {t: defaultdict(list) for t in temperatures}
    final_predictions = {t: {} for t in temperatures}

    for d in tqdm(data):
        raw = {"sentences": d["sentences"], "prefixes": d.get("prefixes"),
               "completions": d["completions"]}
        label, uid = d["label"], d["UID"]
        metadata = {k: v for k, v in d.items()
                    if k not in ("sentences", "completions", "prefixes", "label", "image")}
        update_subset_to_stats(subset_to_stats, [metadata])

        # Compute score for each candidate sentence
        scores = []
        for sentence, completion in  zip(d["sentences"], d["completions"]):
            scores.append(score_sentence(model, tokenizer, sentence, completion, args))
        
        # Temperature is undefined for subword segmental marginal, so has no effect.
        all_log_probs = {t: [torch.tensor([s]) for s in scores] for t in temperatures}

        # Process datapoint
        rank_and_evaluate(args, subset_to_stats, all_log_probs, [raw], [label],
                          [metadata], [uid], predictions)

    if args.save_predictions:
        for t in temperatures:
            final_predictions[t] = {uid: {"predictions": preds}
                                    for uid, preds in predictions[t].items()}
    return subset_to_stats, final_predictions


def main():
    args = _parse_arguments()
    dataset = args.data_path.stem
    args.model_name = getattr(args, "model_name", None) or pathlib.Path(args.model_path_or_name).name
    output_dir = getattr(args, "output_dir", None) or pathlib.Path("results")
    if args.revision_name is None:
        revision_name = "main"
    else:
        revision_name = args.revision_name
    args.output_path = output_dir / args.model_name / revision_name / "zero_shot" / args.backend / args.task / dataset
    args.output_path.mkdir(parents=True, exist_ok=True)

    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path_or_name, trust_remote_code=True, revision=args.revision_name
    ).to(DEVICE).eval()
    tokenizer = SubSegDeBERTaTokenizer.from_pretrained(args.model_path_or_name)

    # Get results
    data = read_files(args)
    temperatures = [1.0]
    results, predictions = compute_subseg_results(args, model, tokenizer, data, temperatures)

    # Process results
    accuracies, average_accuracies = process_results(args, results)
    best_temp = 1.0
    print(f"{best_temp}\t{average_accuracies[best_temp]:.2f}\n")

    # Report and save
    create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp], task=args.task)
    with (args.output_path / "best_temperature_report.txt").open("w") as f:
        create_evaluation_report(best_temp, average_accuracies[best_temp], accuracies[best_temp], task=args.task, file=f)

    # Save predictions
    if args.save_predictions:
        save_predictions(args, predictions, best_temp)


if __name__ == "__main__":
    main()