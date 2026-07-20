"""
Zero-shot GlobalPIQA evaluator for SubSegGPT.

We score GlobalPIQA with the model's conditional marginal:
score(sentence) = log p(completion | prefix) / len(completion)

This script adapts the BabyLM eval pipeline code. It is based on
github.com/babylm-org/babylm-eval/blob/main/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py
and either directly calls or adapts methods in babylm-eval/blob/main/strict/evaluation_pipeline.

Usage: Invoked via run_eval_subseggpt.py global_piqa ... with the same args as the BabyLM pipeline.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

import torch
from tqdm import tqdm

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

from subseggpt.word_surprisal import word_logprob, load_model_and_tokenizer

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def score_sentence(model, tokenizer, sentence, completion, args):
    """
    Compute log p(completion | prefix) / len(completion) from the model marginals.
    """

    prefix = sentence[:len(sentence) - len(completion)]
    logprob = word_logprob(model, tokenizer, prefix, completion)

    # Length-normalise to match BabyLM eval pipeline. 
    num_chars = len(completion.strip())
    return logprob / max(num_chars, 1)


def compute_subseg_results(args, model, tokenizer, data, temperatures):
    """
    Equivalent of compute_causal_results in github.com/babylm-org/babylm-eval/blob/main/strict/evaluation_pipeline/sentence_zero_shot/compute_results.py
    but adapted for SubSegGPT.
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

    model, tokenizer = load_model_and_tokenizer(args.model_path_or_name, revision=args.revision_name)

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
