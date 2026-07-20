import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorForSubSegDeBERTa:
    """
    Data collator for masked language modelling with SubSegDeBERTa.
    Collects MLM inputs (character ids of surrouding context with [MASK] token replacing
    sampled words) and MLM targets (character ids of masked words). Pads both sets of sequences
    to common lengths. Precomputes lex_ids.
    """

    tokenizer: Any
    mlm_probability: float = 0.15
    min_masks: int = 1  # Miminum number of words masked per sequence.
    max_word_len: int = 20  # Words longer than this (in characters) are never masked.

    def _select_masked_words(self, eligible):
        if not eligible:
            return set()
        masked = {w for w in eligible if random.random() < self.mlm_probability}
        if len(masked) < self.min_masks:
            need = min(self.min_masks, len(eligible))
            masked = set(random.sample(eligible, need))
        return masked  # set of word indices for words to be masked

    def __call__(self, features):
        """
        Word masking data collator for SubSegDeBERTa. The characters for each masked word are
        replaced by a single [MASK] token (used as encoder input) and the masked word characters 
        are collected (used as decoder target output).

        Args:
            features: list of batch_size dicts {"input_ids": list[int], "word_ids": list[int]}
                Each item is two parallel lists of equal length:
                - input_ids is char ids wrapped with [CLS] ... [SEP]
                - word_ids tracks word position in the sequence (-1 for [CLS]/[SEP] and whitespace)
            
        Returns:
            Dictionary with:
                - input_ids (batch_size, max_seq_len): char ids with each masked word set to one [MASK] char
                - attention_mask (batch_size, max_seq_len): encoder padding mask
                - masked_pos (num_masked_words, 2): (batch_idx, position) of each [MASK] char in input_ids
                - target_chars (num_masked_words, max_masked_len): padded chars + <eow> of masked words
                - word_lens (num_masked_words, ): actual lengths (including <eow>) of masked words
                - lex_ids (num_masked_words, max_masked_len, max_seg_len): lexicon index per (start, segment_len)
        """

        pad_id = self.tokenizer.pad_token_id
        mask_id = self.tokenizer.mask_token_id
        eow_id = self.tokenizer.eow_token_id

        masked_inputs = []
        masked_pos = []
        target_words = []
        word_lens = []

        for b, feature in enumerate(features):
            input_ids = list(feature["input_ids"])
            word_ids = list(feature["word_ids"])
            
            # Words longer than max_word_len chars are ineligible for masking
            char_counts = Counter(w for w in word_ids if w >= 0)
            eligible = [w for w, n in char_counts.items() if n <= self.max_word_len]
            masked_set = self._select_masked_words(eligible)

            new_ids = []
            i, n = 0, len(word_ids)
            while i < n:
                w = word_ids[i]
                if w < 0:  # space or special token
                    new_ids.append(input_ids[i])
                    i += 1
                    continue
                
                # Collect contiguous characters of this word.
                j = i
                while j < n and word_ids[j] == w:
                    j += 1

                # Set up word as masked/unmasked
                if w in masked_set:
                    masked_pos.append([b, len(new_ids)])
                    new_ids.append(mask_id)
                    chars = input_ids[i:j]
                    target_words.append(chars + [eow_id])
                    word_lens.append(len(chars) + 1)
                else:
                    new_ids.extend(input_ids[i:j])
                i = j

            masked_inputs.append(new_ids)

        # Create and pad input_ids (batch_size, max_seq_len) for encoder
        batch_size = len(masked_inputs)
        max_seq_len = max((len(ids) for ids in masked_inputs), default=0)
        input_ids = torch.full((batch_size, max_seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        for batch_num, ids in enumerate(masked_inputs):
            input_ids[batch_num, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[batch_num, : len(ids)] = 1

        # Create and pad ids of masked target words (num_masked_words, max_masked_len) for decoder
        num_masked_words = len(target_words)
        max_masked_len = max((len(t) for t in target_words), default=1)
        target_chars = torch.full((num_masked_words, max_masked_len), pad_id, dtype=torch.long)
        for target_word_num, ids in enumerate(target_words):
            target_chars[target_word_num, : len(ids)] = torch.tensor(ids, dtype=torch.long)

        if num_masked_words > 0:
            masked_pos = torch.tensor(masked_pos, dtype=torch.long)
        else:  # No masked words in this batch, so pass empty tensor.
            masked_pos = torch.zeros((0, 2), dtype=torch.long)
        
        lex_ids = self.tokenizer.compute_lex_ids(target_chars)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "masked_pos": masked_pos,
            "target_chars": target_chars,
            "word_lens": torch.tensor(word_lens, dtype=torch.long),
            "lex_ids": lex_ids,
        }
