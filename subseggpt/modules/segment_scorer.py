import torch
import torch.nn as nn

from .char_decoder import LSTMCharDecoder
from .lex_decoder import LexiconDecoder, MixtureGate


class SegmentScorer(nn.Module):
    """
    Computes log-probabilites of subword segments given preceding context representations.
    """

    def __init__(
        self,
        vocab_size,
        lex_vocab_size,
        context_size,
        char_decoder_hidden_size=512,
        char_decoder_num_layers=1,
        char_embed_dim=128,
        max_seg_len=5,
        dropout=0.1,
        seg_end_token_id=3,
        pad_token_id=0,
        n_special=5,
        n_alpha=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.lex_vocab_size = lex_vocab_size
        self.context_size = context_size
        self.max_seg_len = max_seg_len
        self.seg_end_token_id = seg_end_token_id
        self.pad_token_id = pad_token_id
        self.n_special = n_special
        self.n_alpha = n_alpha

        self.char_decoder = LSTMCharDecoder(
            vocab_size=vocab_size,
            hidden_size=char_decoder_hidden_size,
            context_size=context_size,
            char_embed_dim=char_embed_dim,
            num_layers=char_decoder_num_layers,
            dropout=dropout,
            seg_end_token_id=seg_end_token_id
        )

        self.lex_decoder = LexiconDecoder(
            lex_vocab_size=lex_vocab_size,
            context_size=context_size,
            dropout=dropout
        )

        self.mixture_gate = MixtureGate(context_size=context_size)

    def extract_windows(self, target_chars):
        """
        Extract sliding windows of target characters, each of which is max_seg_len long.
        Each [a, b, c, d, e] -> [[a, b, c], [b, c, d], [c, d, e], [d, e, <pad>], [e, <pad>, <pad>]]
        (max_seg_len = 3 in this example)

        Args:
            target_chars: (num_masked_words, max_masked_len)

        Returns:
            windows: (num_masked_words, max_masked_len, max_seg_len)
            windows[n, j] = target_chars[n, j:j+max_seg_len] (padded so all windows are max_seg_len).
        """
        num_masked_words, max_masked_len = target_chars.shape
        pad = torch.full((num_masked_words, self.max_seg_len - 1), self.pad_token_id, device=target_chars.device, dtype=target_chars.dtype)
        # Add max_seg_len - 1 pad tokens to each word, to pad max_seg_len-length window starting with final character
        padded = torch.cat([target_chars, pad], dim=1)  # (num_masked_words, max_masked_len + max_seg_len - 1)
        windows = padded.unfold(dimension=1, size=self.max_seg_len, step=1)  # (num_masked_words, max_masked_len, max_seg_len)
        return windows

    def forward(self, hidden_states, input_ids, lex_ids, return_gate=False, content_lens=None):
        """
        Computes subword logprobs for all sequence positions and segment lengths.

        Args:
            hidden_states: (batch_size, seq_len, context_size)
                Context representations from transformer backbone.
            input_ids: (batch_size, seq_len)
                Input character indices, starting with BOS.
            lex_ids: (batch_size, seq_len, max_seg_len)
                Lexicon index for each segment. Includes BOS position.
            return_gate: Whether to return gate values.
            content_lens: Optional (batch_size,) Lengths of content (excluding BOS).

        Returns:
            Dictionary with:
                - 'seg_logp': (batch_size, content_len, max_seg_len)
                    Log-probabilities for each segment at each content position.
                    seg_logp[b, t, l-1] = log P(content[t:t+l] | context[t]).
                - 'gate': Optional (batch_size, content_len)
                    Per-position mixture gate values if return_gate=True.
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Content to model is everything after BOS
        content = input_ids[:, 1:]  # (batch_size, content_len)
        content_len = content.size(1)

        # Skip BOS, so lex_ids[:, t, l-1] is for segment starting at content position t
        lex_ids = lex_ids[:, 1:, :]  # (batch_size, content_len, max_seg_len)

        # Contexts for content position t come from hidden_states[:, t, :]
        # We use hidden_states[:, :-1, :] since last position state has no content to predict
        all_contexts = hidden_states[:, :-1, :]  # (batch_size, content_len, context_size)

        # Extract sliding windows of subword segments
        windows = self.extract_windows(content)  # (batch_size, content_len, max_seg_len)

        # Get input chars for each position
        prev_chars = input_ids[:, :content_len]  # (batch_size, content_len)

        # Compute char decoder logprobs
        logp_char = self.char_decoder.score_all_lengths(
            all_contexts.reshape(batch_size * content_len, self.context_size),
            windows.reshape(batch_size * content_len, self.max_seg_len),
            prev_chars.reshape(batch_size * content_len),
        )  # (batch_size * content_len, max_seg_len)
        logp_char = logp_char.view(batch_size, content_len, self.max_seg_len)

        # Compute lex decoder logprobs
        logp_lex_all = self.lex_decoder(all_contexts)  # (batch_size, content_len, lex_vocab_size)
        in_lexicon = lex_ids >= 0

        # To handle out-of-lexicon segments without crashing the lex decoder, we clamp lex_idx 
        # to valid range and then mask out out-of-lexicon log-probs after decoding. 
        safe_lex_idx = lex_ids.clamp(min=0)
        
        # Extract target subword lex logprobs
        logp_lex = logp_lex_all.gather(dim=-1, index=safe_lex_idx)  # (batch_size, content_len, max_seg_len)
        
        # Set log probs of segments not in lexicon to -inf 
        logp_lex = torch.where(in_lexicon, logp_lex, torch.full_like(logp_lex, float("-inf")))

        # Expand context tensor, so MixtureGate can broadcast across segment lengths.
        # (batch_size, content_len, context_size) -> (batch_size, content_len, 1, context_size)
        expanded_contexts = all_contexts.unsqueeze(2)

        # Combine via mixture gate.
        seg_logp, gate_values = self.mixture_gate(expanded_contexts, logp_char, logp_lex)  # seg_logp = (batch_size, content_len, max_seg_len)
        gate_values = gate_values.squeeze(-1)  # (batch_size, content_len)

        # To ensure all segments are valid subwords units, no segments may cross whitespaces.
        # More generally, subword segments cannot cross non-alphabetical characters.
        # To enforce this, we keep only multi-char segments that are fully alphabetical.
        # Single-char segments can be any character - whitespaces, special tokens, punctuation,
        # numbers, etc. are all treated as 1-char subwords.
        is_alpha = (windows >= self.n_special) & (windows < self.n_special + self.n_alpha)
        # torch.cumprod calculates the cumulative product of elements in a tensor along a specified dimension.
        all_alpha = is_alpha.long().cumprod(dim=-1).bool()  # (batch_size, content_len, max_seg_len)
        seg_lens = torch.arange(1, self.max_seg_len + 1, device=device).view(1, 1, self.max_seg_len)
        # Only valid subword segments if fully alphabetical OR 1 character long.
        alpha_valid = (seg_lens == 1) | all_alpha  # (batch_size, content_len, max_seg_len)
        seg_logp = seg_logp.masked_fill(~alpha_valid, float("-inf"))

        # Mask out windows that go beyond the actual content length. Use per-sequence
        # content_lens if given, else the padded content length.
        positions = torch.arange(content_len, device=device).view(1, content_len, 1)
        lengths = (
            content_lens if content_lens is not None
            else torch.full((batch_size,), content_len, device=device, dtype=torch.long)
        )
        valid = (positions + seg_lens) <= lengths.view(-1, 1, 1)
        seg_logp = seg_logp.masked_fill(~valid, float("-inf"))

        result = {'seg_logp': seg_logp}
        if return_gate:
            result['gate'] = gate_values
        return result