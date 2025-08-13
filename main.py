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

# ------------------------------ Configuration ------------------------------

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size = 512
    num_heads = 16
    num_layers = 8
    dropout = 0.1
    lr = 1e-4
    batch_size = 16
    num_epochs = 15
    max_len = 20
    beam_size = 7

config = Config()

# ------------------------------ Vocabulary ------------------------------

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

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

    idx2word = {i: w for w, i in vocab.items()}
    return vocab, idx2word

# ------------------------------ Dataset ------------------------------

class CaptionDataset(Dataset):
    def __init__(self, csv_file, img_folder, vocab, transform=None, max_len=20):
        self.df = pd.read_csv(csv_file, quotechar='"', escapechar='\\')
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

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption_words = tokenize(str(row[1]))
        tokens = [self.vocab.get(word, self.vocab['<unk>']) for word in caption_words]
        tokens = [self.vocab['<start>']] + tokens + [self.vocab['<end>']]
        tokens = tokens[:self.max_len]
        tokens += [self.vocab['<pad>']] * (self.max_len - len(tokens))

        return image, torch.tensor(tokens)

# ------------------------------ Model ------------------------------

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.fc = nn.Linear(self.model.config.hidden_size, config.embed_size)

    def forward(self, x):
        out = self.model(pixel_values=x).last_hidden_state
        out = self.fc(out) 
        return out.permute(1, 0, 2) 


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
        tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float('-inf')), 1).to(embed.device)
        out = self.transformer(embed, memory, tgt_mask=tgt_mask)
        return self.fc(out)


class ImageCaptioningModel(nn.Module):
     def __init__(self, vocab):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder = TransformerDecoder(len(vocab))
        self.vocab = vocab
        self.idx2word = {i: w for w, i in vocab.items()}

     def forward(self, images, captions):
        # captions: (B, T)
        memory = self.encoder(images)
        output = self.decoder(captions[:, :-1], memory)
        return output.permute(1, 0, 2)

     def caption_image(self, image, max_len=20, beam_size=7):
        self.eval()
        if self.vocab is None or self.idx2word is None:
            raise ValueError("Model's vocab or idx2word not set.")
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0).to(config.device))
            sequences = [[[self.vocab['<start>']], 0.0]]

            for _ in range(max_len):
                all_candidates = []
                for seq, score in sequences:
                    input_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(config.device)
                    embed = self.decoder.embedding(input_seq).permute(1, 0, 2)
                    tgt_len = embed.size(0)
                    tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float('-inf')), 1).to(config.device)
                    memory = features
                    output = self.decoder.transformer(embed, memory, tgt_mask=tgt_mask)
                    output = self.decoder.fc(output)
                    probs = F.softmax(output[-1, 0], dim=0)
                    top_probs, top_idx = torch.topk(probs, beam_size)

                    for i in range(beam_size):
                        candidate = [seq + [top_idx[i].item()], score + torch.log(top_probs[i] + 1e-9).item()]
                        all_candidates.append(candidate)

                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                sequences = ordered[:beam_size]

                print("Current best:", [self.idx2word.get(tok, "<unk>") for tok in sequences[0][0]])

                if all(seq[-1] == self.vocab['<end>'] for seq, _ in sequences):
                    break

            best_seq = sequences[0][0]
            caption = []
            for token in best_seq[1:]:
                if token == self.vocab['<end>']:
                    break
                caption.append(self.idx2word.get(token, '<unk>'))
            return ' '.join(caption)

# ------------------------------ Training ------------------------------

def train():
    print(f"🚀 Training on device: {config.device}")

    df = pd.read_csv("train.csv", quotechar='"', escapechar='\\')
    df.to_csv("train.csv", index=False)
    vocab, idx2word = build_vocab(df.iloc[:, 1].tolist())

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = CaptionDataset("train.csv", "train", vocab, transform)

    dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=os.cpu_count() or 2,
    pin_memory=True
    )


    model = ImageCaptioningModel(vocab).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images, captions = images.to(config.device), captions.to(config.device)
            outputs = model(images, captions)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
           
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

if __name__ == '__main__':
    train()
    
def generate_submission(model_path="model.pth", vocab_path="vocab.json", image_folder="test", output_file="submission.csv"):
    from glob import glob


    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
    image_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    model = ImageCaptioningModel(vocab).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()


    submissions = []

    for img_id, path in tqdm(zip(image_ids, image_paths), total=len(image_paths), desc="Captioning"):
        image = Image.open(path).convert("RGB")
        image = transform(image).to(config.device)
        caption = model.caption_image(image, max_len=config.max_len, beam_size=config.beam_size)
        submissions.append({"image_id": img_id, "caption": caption})


    pd.DataFrame(submissions).to_csv(output_file, index=False)
    print(f"\n✅ Submission dosyası oluşturuldu: {output_file}")

if __name__ == '__main__':

   generate_submission(
    model_path="model.pth",
    vocab_path="vocab.json",
    image_folder="test",
    output_file="submission.csv"
)




