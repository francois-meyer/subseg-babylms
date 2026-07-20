import torch
import torch.nn as nn


class LSTMWordContextEncoder(nn.Module):
    """
    Character-level LSTM that produces a context representation for each position
    in a masked word, providing context to the subword segmental decoder.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,  # size of context representations produced by this encoder
        context_size,  # size of encoder/backbone representations of masked words 
        char_embed_dim=128,  # size of input character embeddings fed to LSTM word context encoder
        lstm_hidden_size=768,  # hidden size of LSTM word context encoder 
        num_layers=1,
        dropout=0.1,
        start_token_id=1, # token fed at position 0 (default: cls)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.start_token_id = start_token_id

        # Character embeddings
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim)

        # Project masked word context to LSTM initial states
        self.context_to_h = nn.Linear(context_size, lstm_hidden_size * num_layers)
        self.context_to_c = nn.Linear(context_size, lstm_hidden_size * num_layers)

        # LSTM to encode context for each position in a masked word
        self.lstm = nn.LSTM(
            input_size=char_embed_dim + context_size,  # input is [char_embedding ; mask_encoder_output]
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection to word context size
        self.out_proj = nn.Linear(lstm_hidden_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _init_hidden(self, context):
        """
        Initialize LSTM states from mask context vector with learned linear projections.

        Args:
            context: (num_masked_words, context_size) output embedding from backbone transformer

        Returns:
            (h_0, c_0), each of shape (num_layers, num_masked_words, lstm_hidden_size)
        """ 
        num_masked_words = context.size(0)

        h_0 = self.context_to_h(context)  # (num_masked_words, lstm_hidden_size * num_layers)
        c_0 = self.context_to_c(context)  # (num_masked_words, lstm_hidden_size * num_layers)  

        # Reshape to (num_layers, num_masked_words, lstm_hidden_size), because LSTM expects that format
        h_0 = h_0.view(num_masked_words, self.num_layers, self.lstm_hidden_size).permute(1, 0, 2).contiguous()
        c_0 = c_0.view(num_masked_words, self.num_layers, self.lstm_hidden_size).permute(1, 0, 2).contiguous()

        return h_0, c_0

    def forward(self, context, target_chars):
        """
        Compute representations that encode word history up to each character in each target masked 
        word. These represenations are computed with this LSTM from the [MASK] encoder representation 
        and character embeddings in the target word. 

        Args:
            context: (num_masked_words, context_size)
                [MASK] representations from bidirectional encoder.
            target_chars: (num_masked_words, max_masked_len): padded chars + <eow> of masked words
                Target masked word characters + <eow> (padded)

        Returns:
            word_context_embeddings: (num_masked_words, max_masked_len, hidden_size)
                Per-position context: word_context_embeddings[:, j] is context for target_char[:, j].
        """
        num_masked_words, max_masked_len = target_chars.shape
        device = target_chars.device

        # Append start-of-word token to input characters
        start = torch.full((num_masked_words, 1), self.start_token_id, device=device, dtype=target_chars.dtype)
        input_ids = torch.cat([start, target_chars[:, :-1]], dim=1)  # (num_masked_words, max_masked_len)

        # Concatenate mask output representation to every character embedding
        char_input_embeddings = self.char_embedding(input_ids)  # (num_masked_words, max_masked_len, char_embed_dim)
        context_unsqueezed = context.unsqueeze(1)  # (num_masked_words, 1, context_size)
        context_expanded = context_unsqueezed.expand(-1, max_masked_len, -1)  # (num_masked_words, max_masked_len, context_size)
        concatenated_embeddings = torch.cat([char_input_embeddings, context_expanded], dim=-1)  # (num_masked_words, max_masked_len, char_embed_dim + context_size)

        # Initialize hidden state from context
        h_0, c_0 = self._init_hidden(context)

        # Run LSTM and project to the boundary-state size
        lstm_out, _ = self.lstm(concatenated_embeddings, (h_0, c_0))  
        lstm_out = self.dropout(lstm_out)  # (num_masked_words, max_masked_len, lstm_hidden_size)
        word_context_embeddings = self.out_proj(lstm_out)  # (num_masked_words, max_masked_len, hidden_size)
        return word_context_embeddings  
