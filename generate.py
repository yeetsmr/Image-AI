import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from glob import glob
from train import Config, ImageCaptioningModel
from tqdm import tqdm


# ------------------------------ Configuration ------------------------------
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed_size = 512
    num_heads = 16
    num_layers = 12
    dropout = 0.3
    lr = 1e-5
    batch_size = 16
    num_epochs = 15
    max_len = 20
    beam_size = 5

config = Config()


def caption_image(model, image, vocab, idx2word, max_len=20, beam_size=5):
    model.eval()
    with torch.no_grad():
        features = model.encoder(image.unsqueeze(0).to(config.device))
        sequences = [([vocab['<start>']], 0.0)]

        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == vocab['<end>']:
                    all_candidates.append((seq, score))
                    continue

                input_seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(config.device)
                embed = model.decoder.embedding(input_seq).permute(1, 0, 2)
                tgt_len = embed.size(0)
                tgt_mask = torch.triu(torch.full((tgt_len, tgt_len), float('-inf')), 1).to(config.device)
                output = model.decoder.transformer(embed, features, tgt_mask=tgt_mask)
                output = model.decoder.fc(output)
                probs = F.log_softmax(output[-1, 0], dim=0)
                top_probs, top_idx = torch.topk(probs, beam_size)

                for i in range(beam_size):
                    candidate = (seq + [top_idx[i].item()], score + top_probs[i].item())
                    all_candidates.append(candidate)

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        best_seq = sequences[0][0]
        caption = []
        for token in best_seq[1:]:
            word = idx2word.get(token, '')
            if word == '<end>':
                break
            if word not in ['<pad>', '<unk>']:
                caption.append(word)
        return ' '.join(caption)


def generate_submission(model_path="model.pth", vocab_path="vocab.json", image_folder="test", output_file="submission.csv"):
    image_paths = sorted(glob(os.path.join(image_folder, "*.jpg")))
    image_ids = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    idx2word = {i: w for w, i in vocab.items()}

    model = ImageCaptioningModel(vocab).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    submissions = []
    for img_id, path in tqdm(zip(image_ids, image_paths), total=len(image_paths), desc="🔍 Captioning"):
        image = Image.open(path).convert("RGB")
        image = transform(image).to(config.device)
        caption = caption_image(model, image, vocab, idx2word, max_len=config.max_len, beam_size=config.beam_size)
        submissions.append({"image_id": img_id, "caption": caption})

    pd.DataFrame(submissions).to_csv(output_file, index=False)
    print(f"\n✅ Caption dosyası oluşturuldu: {output_file}")


if __name__ == '__main__':
    generate_submission()
