class Config:
    def __init__(
        self,
        vocab_size=10000,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=2048,
        max_position_embeddings=128,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        initializer_range=0.02,
        is_causal=False,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.is_causal = is_causal
