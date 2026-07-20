import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCharDecoder(nn.Module):
    """
    Character-level LSTM decoder that computes p_char for subword segments,
    conditioned on outputs from the word context encoder.

    Computes log P(segment) = log(P(c1|h)) + log(P(c2|h,c1)) + ... + log(P(seg_end|h,c1..cn))
    where h is computed by the word context encoder LSTM, based on the [MASK] encoder representation.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,  # LSTM hidden size
        context_size,  # Context produced by word context encoder
        char_embed_dim=128,
        num_layers=1,
        dropout=0.1,
        seg_end_token_id=6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.char_embed_dim = char_embed_dim
        self.num_layers = num_layers
        self.seg_end_token_id = seg_end_token_id

        # Character embeddings
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim)

        # Project context to LSTM initial states
        self.context_to_h = nn.Linear(context_size, hidden_size * num_layers)
        self.context_to_c = nn.Linear(context_size, hidden_size * num_layers)

        # LSTM for autoregressive character generation
        self.lstm = nn.LSTM(
            input_size=char_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection to character vocabulary logits
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, context):
        """
        Initialize LSTM states from context vector with learned linear projections.

        Args:
            context: (batch_size, context_size) context vectors from word context encoder.

        Returns:
            (h_0, c_0), each of shape (num_layers, batch_size, hidden_size)
        """
        batch_size = context.size(0)

        h_0 = self.context_to_h(context)  # (batch_size, hidden_size * num_layers)
        c_0 = self.context_to_c(context)  # (batch_size, hidden_size * num_layers)

        # Reshape to (num_layers, batch_size, hidden_size), because LSTM expects that format
        h_0 = h_0.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c_0 = c_0.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()

        return h_0, c_0

    def forward(self, context, input_ids, target_ids):
        """
        Compute log-probabilities for target character segments.
        batch_size here refers to the number of segments being scored in parallel,
        not the original batch size of the SubSegDeBERTa model training.

        Args:
            context: (batch_size, context_size)
                Context vectors from word context encoder.
            input_ids: (batch_size, seg_len+1)
                Input character indices for LSTM - [prev_char, c1, c2, ..., c_n].
            target_ids: (batch_size, seg_len+1)
                Target character indices to predict - [c1, c2, ..., cn, seg_end].

        Returns:
            log_probs: (batch_size,)
                Summed character log-probabilities of each subword segment.
        """
        # Initialize hidden state from context
        h_0, c_0 = self.init_hidden(context)

        # Get character embeddings for input
        input_embeds = self.char_embedding(input_ids)  # (batch_size, seg_len, char_embed_dim)

        # Run LSTM
        lstm_out, _ = self.lstm(input_embeds, (h_0, c_0))  # (batch_size, seg_len, hidden_size)
        lstm_out = self.dropout(lstm_out)

        # Compute logits for each position
        logits = self.output_proj(lstm_out)  # (batch_size, seg_len, vocab_size)

        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=-1) # numerically stable way to compute log-probs from logits

        # Gather log-probs for target characters
        target_log_probs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, seg_len)

        # Sum log-probs across sequence (chain rule)
        seg_logp = target_log_probs.sum(dim=-1)  # (batch_size,)

        return seg_logp

    def score_all_lengths(self, contexts, windows, prev_chars):
        """
        Compute log-probabilities for all subword segments in a batch of character sequences.
        Each sequence is max_seg_len characters long. To compute char decoder log-probs efficiently,
        we use prefix-sharing: we pass a full sequence through our character-level LSTM and then
        extract the log-probs of subword segments of lengths 1...max_seg_len (+ seg_end character)
        via cumulative summing.

        batch_size here refers to the number of character sequences being scored in parallel,
        not the original batch size of the SubSegDeBERTa model training.

        Args:
            contexts: (batch_size, context_size)
                Context vectors from word context encoder.
            windows: (batch_size, max_seg_len)
                Windows of character sequences, each of which is max_seg_len long.
            prev_chars: (batch_size,)
                Starting character input for each character sequence.

        Returns:
            log_probs: (batch_size, max_seg_len)
                log_probs [:, l-1] = log p_char of the length-l subword segment, including seg_end.
        """
        batch_size, max_seg_len = windows.shape

        # Prepend starting chars to target chars
        input_ids = torch.cat([prev_chars.unsqueeze(1), windows], dim=1)  # (batch_size, max_seg_len + 1)

        # Initialize hidden state from context
        h_0, c_0 = self.init_hidden(contexts)

        # Get character embeddings for input
        input_embeds = self.char_embedding(input_ids)  # (batch_size, max_seg_len + 1, char_embed_dim)

        # Run LSTM
        lstm_out, _ = self.lstm(input_embeds, (h_0, c_0))  # (batch_size, max_seg_len + 1, hidden_size)
        lstm_out = self.dropout(lstm_out)

        # Compute logits for each position
        logits = self.output_proj(lstm_out)  # (batch_size, max_seg_len + 1, vocab_size)

        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=-1) # numerically stable way to compute log-probs from logits

        # Gather logprobs of target chars (excluding <seg>)
        target_char_logprobs = log_probs[:, :max_seg_len, :].gather(  # (batch_size, max_seg_len, vocab_size)
            dim=-1, index=windows.unsqueeze(-1)  # (batch_size, max_seg_len, 1)
        ).squeeze(-1)  # (batch_size, max_seg_len)

        # torch.cumsum calculates the cumulative sum of elements along a specified dimension in a tensor
        target_segment_logprobs = target_char_logprobs.cumsum(dim=-1)  # (batch_size, max_seg_len)
                                                                       # [:, l-1] = sum of logprobs of first l chars

        # Add logprobs of <seg> to each segment
        seg_end_logprobs = log_probs[:, 1: max_seg_len + 1, self.seg_end_token_id]  # (batch_size, max_seg_len)
        target_segment_logprobs = target_segment_logprobs + seg_end_logprobs

        return target_segment_logprobs