from transformers import PretrainedConfig


class SubSegGPTConfig(PretrainedConfig):
    model_type = "subseggpt"

    def __init__(
        self,
        vocab_size=256,  # Character vocabulary size
        n_alpha=0,  # Number of alphabetical chars in char vocab (subword segments cannot cross non-alphabetical chars)
        lex_vocab_size=10000,  # Lexicon size (subwords)
        max_seg_len=5,  # Maximum subword segment length
        hidden_size=768,  # Transformer hidden size (and transformer backbone character input embedding size)
        num_hidden_layers=12,  # Number of transformer layers
        num_attention_heads=12,  # Number of attention heads
        intermediate_size=3072,  # Feedforward intermediate size
        char_decoder_hidden_size=512,  # LSTM hidden size for char decoder
        char_decoder_num_layers=1,  # Number of LSTM layers for char decoder
        char_embed_dim=128,  # Character input embedding size for char decoder
        dropout=0.1,
        attention_dropout=0.1,
        activation_function="gelu_new",
        # Special token IDs
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        seg_end_token_id=3,  # End-of-subword unit (subword segment boundary)
        # Position encoding
        max_position_embeddings=1024,
        # Initialization
        initializer_range=0.02,
        layer_norm_eps=1e-5,
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
        self.char_decoder_hidden_size = char_decoder_hidden_size
        self.char_decoder_num_layers = char_decoder_num_layers
        self.char_embed_dim = char_embed_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.seg_end_token_id = seg_end_token_id
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
