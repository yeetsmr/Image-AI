# ---------- improved_train.py ----------

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTModel
import pandas as pd
from tqdm import tqdm
import re
import nltk
nltk.download('punkt')
#from nltk.tokenize import word_tokenize


# ------------------------------ Config ------------------------------
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size = 256
    num_heads = 4
    num_layers = 4
    dropout = 0.3
    lr = 1e-4
    batch_size = 16
    num_epochs = 30
    max_len = 20

config = Config()

# --------------------------- Tokenizer ------------------------------
def tokenize(text):
    import re
    return re.findall(r"\w+|[^\w\s]", text.lower())


# --------------------------- Vocabulary -----------------------------
def build_vocab(captions, threshold=1):
    word_counts = {}
    for caption in captions:
        for word in tokenize(caption):
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    idx = 4
    for word, count in word_counts.items():
        if count >= threshold:
            vocab[word] = idx
            idx += 1

    return vocab

# ---------------------------- Dataset -------------------------------
class CaptionDataset(Dataset):
    def __init__(self, csv_file, img_folder, vocab, transform=None, max_len=20):
        self.df = pd.read_csv(csv_file, quotechar='"', escapechar='\\')
        self.df = self.df[self.df.iloc[:, 1].str.split().apply(lambda x: len(x) > 3)]  # Filter short captions
        self.img_folder = img_folder
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = str(row[0]) + ".jpg"
        image_path = os.path.join(self.img_folder, image_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption_words = tokenize(str(row[1]))
        tokens = [self.vocab.get(word, self.vocab['<unk>']) for word in caption_words]
        tokens = [self.vocab['<start>']] + tokens + [self.vocab['<end>']]
        tokens = tokens[:self.max_len]
        tokens += [self.vocab['<pad>']] * (self.max_len - len(tokens))

        return image, torch.tensor(tokens)

# ---------------------------- Encoder -------------------------------
class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze encoder initially
        self.fc = nn.Linear(self.model.config.hidden_size, config.embed_size)

    def forward(self, x):
        out = self.model(pixel_values=x).last_hidden_state
        out = self.fc(out)
        return out.permute(1, 0, 2)

# ---------------------------- Decoder -------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_size)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(config.embed_size, config.num_heads, dim_feedforward=2048, dropout=config.dropout),
            num_layers=config.num_layers
        )
        self.fc = nn.Linear(config.embed_size, vocab_size)

    def forward(self, tgt, memory):
        embed = self.embedding(tgt).permute(1, 0, 2)
        tgt_len = embed.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(embed.device)
        out = self.transformer(embed, memory, tgt_mask=tgt_mask)
        return self.fc(out)

# ------------------------- Full Model -------------------------------
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder = TransformerDecoder(len(vocab))
        self.vocab = vocab

    def forward(self, images, captions):
        memory = self.encoder(images)
        output = self.decoder(captions[:, :-1], memory)
        return output.permute(1, 0, 2)

# ----------------------- Label Smoothing ----------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        true_dist = pred.data.clone()
        true_dist.fill_(self.smoothing / (self.cls - 1))
        ignore = target == self.ignore_index
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist.masked_fill_(ignore.unsqueeze(1), 0)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# --------------------------- Training -------------------------------
def train():
    print(f"\n🚀 Training on device: {config.device}")
    df = pd.read_csv("train.csv", quotechar='"', escapechar='\\')
    df.to_csv("train.csv", index=False)
    vocab = build_vocab(df.iloc[:, 1].tolist(), threshold=1)  # Threshold 1 yap


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = CaptionDataset("train.csv", "train", vocab, transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=12, pin_memory=True)

    model = ImageCaptioningModel(vocab).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = LabelSmoothingLoss(len(vocab), smoothing=0.1, ignore_index=vocab['<pad>'])

    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        if epoch == 5:  # Fine-tune encoder after few epochs
            for param in model.encoder.model.parameters():
                param.requires_grad = True

        for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images, captions = images.to(config.device), captions.to(config.device)
            outputs = model(images, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

if __name__ == '__main__':
    train()
