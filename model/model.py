# model.py
import torch
from torch import nn
import torch.nn.functional as F
from .config import Config

class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor):
        seq_len = input_ids.size(1)
        position_ids = (
            torch.arange(seq_len, dtype=torch.long).unsqueeze(0).to(input_ids.device)
        )
        position_embeddings = self.position_embeddings(position_ids)
        word_embeddings = self.word_embeddings(input_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embeddings = Embedding(config)
        layers = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=F.gelu,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=layers, num_layers=config.num_hidden_layers
        )

    def forward(
        self,
        input_ids,
        attn_mask = None,
        padding_mask = None,
    ):
        if self.config.is_causal and attn_mask is None:
            size = input_ids.shape[1]
            device = input_ids.device
            attn_mask = torch.triu(
                torch.ones(size, size) * float("-inf"), diagonal=1
            ).to(device)

        x = self.embeddings(input_ids)
        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        return x

class MLMHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class MaskedLanguageModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder: Encoder = Encoder(config)
        self.mlm_head = MLMHead(config)
        self.apply(self._init_weights)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L748
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, padding_mask=None):
        x = self.encoder(input_ids, padding_mask=padding_mask)
        x = self.mlm_head(x)
        return x

    # 평가 메소드 추가
    def evaluate(self, dataloader, criterion, device):
        self.eval()
        total_loss = 0.0
        total_iter = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                padding_mask = batch["padding_mask"].to(device)

                logits = self(input_ids, padding_mask=padding_mask)
                loss = criterion(logits.view(-1, self.config.vocab_size), labels.view(-1))

                total_loss += loss.item()
                total_iter += 1

        mean_loss = total_loss / total_iter
        return mean_loss

    # 모델 저장 메소드 추가
    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)