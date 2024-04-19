# main.py
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm  
from model import Config, MaskedLanguageModel
from model.dataset import MyDataset, MLMCollator

vocab_size = 36000
batch_size = 1024
seq_length = 256

device = torch.device("cuda")
path = "C:/Users/USER/Desktop/Project_LLM/data/wikitext-2-raw"
train_dataset = MyDataset(path + "/wiki.train.raw", "tokenizer.json", seq_length, encoding='utf-8')
collate_fn = MLMCollator(train_dataset.tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)

val_dataset = MyDataset(path + "/wiki.valid.raw", "tokenizer.json", seq_length, encoding='utf-8')
val_dataloader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)

config = Config(
    vocab_size=vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=seq_length,
)
model = MaskedLanguageModel(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# 학습
for epoch in range(50):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)  # tqdm 진행 막대 생성
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        logits = model(input_ids, padding_mask=padding_mask)
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.set_postfix(loss=loss.item())

    val_loss = model.evaluate(val_dataloader, criterion, device)
    print(f"\nValidation Loss after epoch {epoch + 1}: {val_loss:1.4f}")


model.save_model("C:/Users/USER/Desktop/Project_LLM/model/model.pt")
