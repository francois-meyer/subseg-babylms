"""
Methods to compute conditional word probabilities / surprisal for SubSegDeBERTa.
Used by reading and AoA evaluators.
"""


from __future__ import annotations

import math

import torch
from transformers import AutoModelForMaskedLM

from subsegdeberta import SubSegDeBERTaTokenizer
from subsegdeberta.eval_zero_shot import _collate_variants

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_and_tokenizer(model_path, revision=None):
    # revision passed if Hub model
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True, revision=revision
    ).to(DEVICE).eval()
    tokenizer = SubSegDeBERTaTokenizer.from_pretrained(model_path, revision=revision)
    return model, tokenizer


def build_masked_word_input(tokenizer, context, word, num_mask_tokens, max_encoder_len=None):
    """
    Build the single masked variant for scoring word after context for SubSegDeBERTa.
    (This is similar to build_word_masked_variants in subsegdeberta.eval_zero_shot.py but only
    creates one variant in the format required for surprisal-based eval.)    

    Args:
        tokenizer: SubSegDeBERTaTokenizer instance.
        context: left context string e.g. "The cat is in the " (no BOS)
        word: target word string e.g. "hat"
        num_mask_tokens: number of [MASK]s to append to context
        max_encoder_len: max length for encoder sequences (= encoder max_position_embeddings).

    Returns:      
        variants dict with:
            - input_ids: [CLS] + context chars + " " + [MASK]*num_mask_tokens + [SEP].
            - slot: position of the first [MASK] in input_ids.
            - target: the word's char ids + <eow> (what the segmental DP scores).
            - word_len: len(target), the index at which to read the per-word marginal.
    """
    context = context.strip()
    word = word.strip()
    context_ids = [tokenizer._convert_token_to_id(ch) for ch in context]
    space_id = tokenizer._convert_token_to_id(" ")
    word_char_ids = [tokenizer._convert_token_to_id(ch) for ch in word]

    eow_id = tokenizer.eow_token_id
    mask_id = tokenizer.mask_token_id
    input_ids = (
        [tokenizer.cls_token_id] + context_ids + [space_id]
        + [mask_id] * num_mask_tokens + [tokenizer.sep_token_id]
    )
    slot = len(context_ids) + 2  # [CLS] + context + " "

    # If masked sequence is longer than the encoder cap, keep a max_encoder_len window
    # centred on the [MASK], clamped to stay within the sequence.
    if max_encoder_len is not None and len(input_ids) > max_encoder_len:
        half_window = max_encoder_len // 2
        start = max(0, min(slot - half_window, len(input_ids) - max_encoder_len))
        input_ids = input_ids[start:start + max_encoder_len]
        slot = slot - start

    return {
        "input_ids": input_ids,
        "slot": slot,
        "target": word_char_ids + [eow_id],
        "word_len": len(word_char_ids) + 1,  # +1 for EOW
    }


@torch.no_grad()
def word_logprob(model, tokenizer, context, word, num_mask_tokens=3):
    """
    Compute log P(word | context) for SubSegDeBERTa from masked word marginals.

    We mirror how the babylm-eval pipeline handles MLMs for word surprisal-based evaluation.
    The word is replaced by num_mask_tokens [MASK]s and [SEP]. The first [MASK] target is scored.
    Reading-time eval uses num_mask_tokens=3 (padding with extra masks before the end-of-sequence
    [SEP]), while AoA eval uses num_mask_tokens=1.

    Args:
        model: SubSegDeBERTaForMaskedLM instance.
        tokenizer: SubSegDeBERTaTokenizer instance.
        context: left context string e.g. "The cat is in the " (no BOS).
        word: target word string e.g. "hat".
        num_mask_tokens: number of [MASK]s to append to context.
    """
    tokenizer.max_seg_len = model.config.max_seg_len
    variant = build_masked_word_input(
        tokenizer, context, word, num_mask_tokens,
        max_encoder_len=model.config.max_position_embeddings,
    )
    batch = _collate_variants([variant], tokenizer)
    out = model(**batch)
    return out.word_log_probs[0].item()


@torch.no_grad()
def word_prob(context, word, model, tokenizer, num_mask_tokens=3):
    """
    Compute p(word | context) for SubSegDeBERTa from masked per-word marginal.
    Argument order and return type matches what BabyLM eval pipeline expects.
    """
    return (math.exp(word_logprob(model, tokenizer, context, word, num_mask_tokens=num_mask_tokens)), 0)
