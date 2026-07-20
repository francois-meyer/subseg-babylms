from transformers import PretrainedConfig


class SubSegDeBERTaConfig(PretrainedConfig):  
    model_type = "subsegdeberta"

    def __init__(
        self,
        vocab_size=320,  # Character vocabulary size
        n_alpha=0,  # Number of alphabetical chars in char vocab (subword segments cannot cross non-alphabetical chars)
        lex_vocab_size=10000,  # Lexicon size (subwords)
        max_seg_len=5,  # Maximum subword segment length
        # DeBERTa-v2 encoder backbone
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        max_position_embeddings=512,
        relative_attention=True,
        position_buckets=256,
        max_relative_positions=-1,
        pos_att_type="p2c|c2p",
        norm_rel_ebd="layer_norm",
        # Word-context LSTM
        word_context_hidden_size=768, # Hidden size of word-context LSTM
        word_context_num_layers=1, # Number of layers of word-context LSTM
        # Char decoder
        char_decoder_hidden_size=512, # LSTM hidden size for char decoder
        char_decoder_num_layers=1, # Number of LSTM layers for char decoder
        char_embed_dim=128, # Character input embedding size for char decoder
                            # For word-context LSTM, input dim is char_embed_dim + hidden_size
                            # (concatenated char input embeddings and backbone output embeddings.)
                            # Word-context LSTM produces hidden_size-dim context for each step.
        # Regularisation
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        dropout=0.1,
        layer_norm_eps=1e-7,
        initializer_range=0.02,
        # Special token IDs
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        unk_token_id=4,
        eow_token_id=5,  # end-of-word
        seg_end_token_id=6,  # end-of-segment
        # Training options
        mlm_probability=0.15,
        loss_normalization="word",  # "word" (mean -log alpha) or "char" (bits-per-char)
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_alpha = n_alpha
        self.lex_vocab_size = lex_vocab_size
        self.max_seg_len = max_seg_len

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.relative_attention = relative_attention
        self.position_buckets = position_buckets
        self.max_relative_positions = max_relative_positions
        self.pos_att_type = pos_att_type
        self.norm_rel_ebd = norm_rel_ebd

        self.word_context_hidden_size = word_context_hidden_size
        self.word_context_num_layers = word_context_num_layers

        self.char_decoder_hidden_size = char_decoder_hidden_size
        self.char_decoder_num_layers = char_decoder_num_layers
        self.char_embed_dim = char_embed_dim

        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.unk_token_id = unk_token_id
        self.eow_token_id = eow_token_id
        self.seg_end_token_id = seg_end_token_id

        self.mlm_probability = mlm_probability
        self.loss_normalization = loss_normalization

        # Drop special tokens we won't use, which save_pretrained writes to config.json.
        kwargs.pop("bos_token_id", None)
        kwargs.pop("eos_token_id", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=cls_token_id,
            eos_token_id=sep_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )