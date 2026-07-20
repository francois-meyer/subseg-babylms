"""
Methods to compute conditional word probabilities / surprisal for SubSegGPT.
Used by reading and AoA evaluators.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM
import math

from subseggpt import SubSegGPTTokenizer

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_model_and_tokenizer(model_path, revision=None):
    # revision passed if Hub model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, revision=revision
    ).to(DEVICE).eval()
    tokenizer = SubSegGPTTokenizer.from_pretrained(model_path, revision=revision)
    return model, tokenizer


@torch.no_grad()
def word_logprob(model, tokenizer, context, word):
    """
    Compute log P(word | context) for SubSegGPT from conditional marginals. 

    Args:
        model: SubSegGPTForCausalLM with tokenizer attached, so lex_ids computed in forward.
        tokenizer: SubSegGPTTokenizer instance.
        context: left context string e.g. "The cat is in the " (no BOS).
        word: target word string e.g. "hat".
    """
    context = context.strip()
    word = word.strip()
    full = context + " " + word

    # Left-truncate if the sequence is too long for the model.
    max_len = model.config.max_position_embeddings
    if len(full) + 1 > max_len:
        full = full[len(full) + 1 - max_len:]

    # Pass full sequence to model
    input_ids = [tokenizer.bos_token_id] + [tokenizer._convert_token_to_id(ch) for ch in full]
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
    out = model(input_ids=input_ids)
    log_alpha = out.log_alpha[0]  # (len(full) + 1,)

    # Compute conditional marginal
    len_full = len(full)
    len_context = len_full - len(word)
    return (log_alpha[len_full] - log_alpha[len_context]).item()


@torch.no_grad()
def word_prob(context, word, model, tokenizer):
    """
    Compute p(word | context) for SubSegGPT from conditional marginals. 
    Argument order and return type matches what BabyLM eval pipeline expects.
    """
    return (math.exp(word_logprob(model, tokenizer, context, word)), 0)
