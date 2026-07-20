"""
Age-of-Acquisition (AoA) evaluator for SubSegDeBERTa.

The BabyLM eval pipeline relies on vocab logits of masked tokens to compute word-level surprisal. 
SubSegDeBERTa does not compute logits - it computes marginal probabilities for masked words.
We adapt the BabyLM AoA pipeline to use word-level surprisal for SubSegDeBERTa.

This script reuses and adapts the BabyLM AoA pipeline
github.com/babylm-org/babylm-eval/blob/main/strict/evaluation_pipeline/AoA

Usage: Invoked via run_eval_subsegdeberta.py aoa ... with the same args as the BabyLM pipeline.
"""

from __future__ import annotations

import pathlib

import torch

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

# Reuse BabyLM AoA pipeline code directly
from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, StepConfig, load_eval
from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor
from evaluation_pipeline.AoA_word.run import parse_args, config_paths, save_results
from evaluation_pipeline.utils import AoAEvaluator

from challenge_utils import word_milestone_label
from subsegdeberta.word_surprisal import word_logprob, load_model_and_tokenizer


def _hub_revisions_exist(repo_id, revisions):
    # Check if all given revisions exist on Hub repo.
    try:
        refs = HfApi().list_repo_refs(repo_id)
    except (RepositoryNotFoundError, HfHubHTTPError):
        return False
    available = {b.name for b in refs.branches}
    return all(r in available for r in revisions)


def get_checkpoint_steps(model_dir, expected_word_counts):
    """
    Find the word-milestone checkpoints for the AoA trajectory eval, regardles of whether this is run on
    local checkpoints during development or on Hub checkpoints submitted for the challenge.
        1. First check if epoch_<label> dirs are available locally (as saved by SubSegDeBERTa training script).
        2. Otherwise, check if chck_<N>M revisions are available on the Hub repo model_dir (meaning the checkpoints have been uploaded).
        3. Otherwise raise an error.
    Steps are recorded as canonical chck_<N>M names (what collate_preds expects); load_model_for_step
    maps each back to its local epoch_<label> dir (or Hub revision).
    Returns: (list of chck_<N>M checkpoint names, list of corresponding checkpoint word counts)
    """
    model_path = pathlib.Path(model_dir)
    labels = [f"chck_{w // 1_000_000}M" for w in expected_word_counts]
    local_names = [f"epoch_{word_milestone_label(w)}" for w in expected_word_counts]
    if all((model_path / n).is_dir() for n in local_names):
        return labels, list(expected_word_counts)

    if _hub_revisions_exist(str(model_dir), labels):
        return labels, list(expected_word_counts)

    raise FileNotFoundError(
        f"AoA found no complete checkpoint trajectory for '{model_dir}': neither the local dirs "
        f"{local_names} nor the Hub revisions {labels} are all present."
    )


class SubSegSurprisalExtractor(StepSurprisalExtractor):
    """StepSurprisalExtractor that loads our model, tokeniser, and uses SubSegDeBERTa surprisal function."""

    def load_model_for_step(self, step):
        # step is a canonical chck_<N>M name. Load the matching local epoch_<label> dir if present,
        # else the Hub revision of self.model_name.
        word_count = int(step[len("chck_"):-1]) * 1_000_000
        local_dir = pathlib.Path(self.model_name) / f"epoch_{word_milestone_label(word_count)}"
        if local_dir.is_dir():
            model, tokenizer = load_model_and_tokenizer(str(local_dir))
        else:
            model, tokenizer = load_model_and_tokenizer(self.model_name, revision=step)
        self._tokenizer = tokenizer
        return model

    def load_tokenizer_for_step(self, step):
        return self._tokenizer, self._tokenizer

    def compute_surprisal(self, model, processor, tokenizer, context, target_word, use_bos_only=True):
        return -word_logprob(model, tokenizer, context, target_word, num_mask_tokens=1)


def main():  # Mirrors main function of babylm-eval/strict/evaluation_pipeline/AoA_word/run.py
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_words, contexts = load_eval(args.word_path, args.min_context, args.debug)
    result_file, resume_file = config_paths(args)

    steps_config = StepConfig(
        resume=args.resume,
        track=args.track_name,
        file_path=resume_file,
        debug=args.debug
    )
    # CHANGED: locate the word-milestone checkpoints saved during training (epoch_<label> dirs).
    _, expected_word_counts = steps_config.generate_checkpoint_steps(args.track_name)
    steps_config.steps, steps_config.word_counts = get_checkpoint_steps(
        args.model_name, expected_word_counts
    )
    if args.debug:
        steps_config.steps = steps_config.steps[:5]
        steps_config.word_counts = steps_config.word_counts[:5]

    extractor = SubSegSurprisalExtractor(
        config=steps_config,
        model_name=args.model_name,
        backend=args.backend,
        device=device
    )

    results_data = extractor.analyze_steps(
        contexts=contexts,
        target_words=target_words,
        resume_path=resume_file
    )
    save_results(results_data, result_file)

    # Compute AOA score to mirror eval pipeline update
    cdi_human = args.word_path.parent / "cdi_human.csv"
    if results_data.get("results") and cdi_human.is_file():
        score = AoAEvaluator(cdi_human).compute_curve_fitness(results_data, extractor._tokenizer)["curve_fitness"]
        score_file = result_file.parent / "aoa_score.json"
        JsonProcessor.save_json({"aoa": score}, score_file)
        print(f"AoA score {score:.4f} saved to {score_file}")
    else:
        print(f"Skipping AoA scoring (no results, or cdi_human.csv missing at {cdi_human})")


if __name__ == "__main__":
    main()
