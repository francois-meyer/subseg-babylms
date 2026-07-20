import json
import os
from collections import Counter
from typing import Optional, Tuple

import torch
from transformers import PreTrainedTokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

SPECIAL_TOKENS = ["<pad>", "<cls>", "<sep>", "<mask>", "<unk>", "<eow>", "<seg>"]


def sort_chars(chars):
    # Sort characters: alphabetical first, then numerical, then others
    # (for more readable vocab files).   
    alpha_chars = []
    numeric_chars = []
    other_chars = []

    for char in chars:
        if char.isalpha():
            alpha_chars.append(char)
        elif char.isdigit():
            numeric_chars.append(char)
        else:
            other_chars.append(char)

    # Sort each group
    alpha_chars.sort()
    numeric_chars.sort()
    other_chars.sort()

    return alpha_chars + numeric_chars + other_chars


def extract_subwords(word, max_seg_len):
    # Extract list of all subwords from a word.
    subwords = []
    for seg_len in range(1, min(max_seg_len + 1, len(word) + 1)):
        for start in range(len(word) - seg_len + 1):
            segment = word[start:start + seg_len]
            # For length > 1 only include alphabetical segments
            if seg_len > 1:
                if segment.isalpha():
                    subwords.append(segment)
            else:
                # Single characters always included
                subwords.append(segment)
    return subwords


class SubSegDeBERTaTokenizer(PreTrainedTokenizer):
    """
    Tokenisation functionality for SubSegDeBERTa:
    - Character-level tokenizer for bi-directional context encoder, char-level word context 
        encoder, and char-level decoder.
    - Subword lexicon for lexicon decoder (code for creating lexicon, mapping character 
        segments to lexicon ids, and computing lexicon indices present in input sequences).
    - Text encoding for masked language modelling.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab=None,
        lexicon=None,
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        unk_token="<unk>",
        eow_token="<eow>",
        seg_end_token="<seg>",
        max_seg_len=5,
        **kwargs,
    ):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        self.lexicon = lexicon or {}
        self.ids_to_lexicon = {v: k for k, v in self.lexicon.items()}
        self.max_seg_len = max_seg_len

        self._eow_token = eow_token
        self._seg_end_token = seg_end_token

        super().__init__(
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

        # Register <eow> (end-of-word character appended to masked target words)  
        # and <seg> (end-of-segment character appended to char decoder subword segment)
        self.add_special_tokens(
            {"additional_special_tokens": [eow_token, seg_end_token]}
        )

        # Persist max_seg_len so a reloaded tokenizer keeps it.
        self.init_kwargs["max_seg_len"] = max_seg_len

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def lexicon_size(self):
        return len(self.lexicon)

    @property
    def n_alpha(self):
        return sum(1 for c in self.vocab if c.isalpha())

    @property
    def eow_token(self):
        return self._eow_token

    @property
    def eow_token_id(self):
        return self.vocab.get(self._eow_token, 5)

    @property
    def seg_end_token(self):
        return self._seg_end_token

    @property
    def seg_end_token_id(self):
        return self.vocab.get(self._seg_end_token, 6)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def encode_for_mlm(self, text):
        """
        Encode text for word masking. Output is used by the data collator to mask
        whole words with one mask token.

        Args:
            text: Raw input string.

        Returns:
            dict with input_ids and word_ids (parallel lists, equal length).
            - input_ids is char ids wrapped with [CLS] ... [SEP]
            - word_ids tracks word position in the sequence (-1 for [CLS]/[SEP] and whitespace)
        """
        char_ids = [self.cls_token_id]
        word_ids = [-1]

        wid = -1
        prev_space = True
        for ch in text:
            if ch.isspace():
                char_ids.append(self._convert_token_to_id(ch))
                word_ids.append(-1)
                prev_space = True
            else:
                if prev_space:
                    wid += 1
                char_ids.append(self._convert_token_to_id(ch))
                word_ids.append(wid)
                prev_space = False

        char_ids.append(self.sep_token_id)
        word_ids.append(-1)

        return {"input_ids": char_ids, "word_ids": word_ids}

    def get_lexicon_index(self, segment):
        return self.lexicon.get(segment, -1)

    def compute_lex_ids(self, input_ids):
        """
        Compute lexicon indices for batch input_ids. For each position t and segment
        length l, look up segment input_ids[t:t+l] in the lexicon and store its index
        (or -1 if not found).

        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len)

        Returns:
            lex_ids: torch.Tensor of shape (batch_size, seq_len, self.max_seg_len).
                lex_ids[b, t, k] is the lexicon id of the length-(k+1) segment starting 
                at position t in batch element b, or -1 if that segment is not in the lexicon 
                (or when the segment would run past the end of the sequence).                
        """
        device = input_ids.device

        all_lex_ids = []
        for seq_ids in input_ids.tolist():
            seq_len = len(seq_ids)
            seq_lex_ids = []
            for t in range(seq_len):
                position_indices = []
                for seg_len in range(1, self.max_seg_len + 1):
                    if t + seg_len <= seq_len:
                        segment_ids = seq_ids[t:t + seg_len]
                        segment_str = "".join(
                            self._convert_id_to_token(idx) for idx in segment_ids
                        )
                        lex_idx = self.get_lexicon_index(segment_str)
                    else:
                        lex_idx = -1
                    position_indices.append(lex_idx)
                seq_lex_ids.append(position_indices)
            all_lex_ids.append(seq_lex_ids)

        return torch.tensor(all_lex_ids, dtype=torch.long, device=device)

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str, ...]:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        prefix = (filename_prefix + "-") if filename_prefix else ""

        vocab_file = os.path.join(save_directory, prefix + "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        files = [vocab_file]
        if self.lexicon:
            lexicon_file = os.path.join(save_directory, prefix + "lexicon.json")
            with open(lexicon_file, "w", encoding="utf-8") as f:
                json.dump(self.lexicon, f, ensure_ascii=False, indent=2)
            files.append(lexicon_file)
        return tuple(files)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "SubSegDeBERTaTokenizer":

        # Resolve files from local dir if available, otherwise download from the Hub.
        path = str(pretrained_model_name_or_path)
        dl_kwargs = {k: kwargs.pop(k) for k in
                     ("revision", "cache_dir", "token", "local_files_only", "force_download")
                     if k in kwargs}

        def resolve(filename, required):
            local = os.path.join(path, filename)
            if os.path.exists(local):
                return local
            if not os.path.isdir(path):  # treat path as a Hub repo id
                try:
                    return hf_hub_download(repo_id=path, filename=filename, **dl_kwargs)  #  pulls from revision is passed, otherwise main
                except (EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError, HfHubHTTPError):
                    if required:
                        raise
                    return None
            if required:
                raise FileNotFoundError(f"{filename} not found in {path}")
            return None

        # Load vocab
        with open(resolve("vocab.json", required=True), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Load lexicon if present
        lexicon = None
        lexicon_file = resolve("lexicon.json", required=False)
        if lexicon_file:
            with open(lexicon_file, "r", encoding="utf-8") as f:
                lexicon = json.load(f)

        # Load tokenizer config if present
        config_file = resolve("tokenizer_config.json", required=False)
        if config_file:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                # Remove keys saved in HF tokenizer config file that would disrupt the
                # SubSegDeBERTaTokenizer initialization
                config.pop("added_tokens_decoder", None)
                config.pop("tokenizer_class", None)
                kwargs.update(config)

        return cls(vocab=vocab, lexicon=lexicon, **kwargs)

    @classmethod
    def _build_vocab_and_lexicon(
        cls,
        char_counter,
        segment_counter,
        max_seg_len,
        lex_min_count,
        lex_vocab_size,
    ):
        # Get unique characters from corpus
        corpus_chars = [char for char in char_counter.keys() if char not in SPECIAL_TOKENS]
        sorted_chars = sort_chars(corpus_chars)

        # Build character vocab
        vocab = {}
        for idx, token in enumerate(SPECIAL_TOKENS):
            vocab[token] = idx

        idx = len(SPECIAL_TOKENS)
        for char in sorted_chars:
            vocab[char] = idx
            idx += 1

        # Build lexicon from frequent segments
        lexicon = {"<unk>": 0}
        lex_idx = 1

        # Add most frequent segments
        for segment, count in segment_counter.most_common():
            if count < lex_min_count:
                continue
            if lex_idx >= lex_vocab_size:
                break
            if segment not in lexicon:
                lexicon[segment] = lex_idx
                lex_idx += 1

        # Add space to lexicon
        if " " not in lexicon:
            lexicon[" "] = lex_idx

        return vocab, lexicon

    @classmethod
    def train_from_iterator(
        cls,
        iterator,
        max_seg_len=5,
        lex_min_count=5,
        lex_vocab_size=10000,
        **kwargs,
    ):
        # Count characters and subword segments from corpus
        char_counter = Counter()
        segment_counter = Counter()

        for text in iterator:
            # Count characters
            for char in text:
                char_counter[char] += 1

            # Extract subword segments from each word
            words = text.split()
            for word in words:
                subwords = extract_subwords(word, max_seg_len)
                for subword in subwords:
                    segment_counter[subword] += 1

        vocab, lexicon = cls._build_vocab_and_lexicon(
            char_counter, segment_counter, max_seg_len, lex_min_count, lex_vocab_size
        )
        return cls(vocab=vocab, lexicon=lexicon, max_seg_len=max_seg_len, **kwargs)

