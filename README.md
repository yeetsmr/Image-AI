This project implements an image captioning system using a Vision Transformer (ViT) encoder and a Transformer decoder in PyTorch.
It takes an image as input and generates a descriptive caption in natural language.

🔹 Features
ViT Encoder: Extracts high-level visual features from images.

Transformer Decoder: Generates captions word-by-word using attention.

Beam Search: Improves caption quality by exploring multiple decoding paths.

Label Smoothing: Stabilizes training and improves generalization.

Configurable Parameters: Easily adjust embedding size, layers, dropout, beam size, etc.

🔹 Workflow
Training (improved_train.py)

Builds vocabulary from captions.

Loads images & captions, applies augmentations.

Trains the encoder–decoder model.

Saves the trained model and vocabulary.

Inference (generate_submission.py)

Loads the trained model & vocabulary.

Generates captions for test images using beam search.

Saves results to a CSV file.

🔹 Tech Stack
Language: Python 3.x

Framework: PyTorch

Pretrained Models: HuggingFace ViT

Data Processing: Pandas, PIL, torchvision

The "main.py" file can do all the work on its own, but I achieved more consistent results when I split it into two parts: "generate.py" and "train.py".

# 🖼️ Image Captioning with Vision Transformer & Transformer Decoder

This project implements an **image captioning system** using a **Vision Transformer (ViT) encoder** and a **Transformer decoder** in PyTorch.  
It takes an image as input and generates a descriptive caption in natural language.

---

Features
- **ViT Encoder** – Extracts high-level visual features from images.
- **Transformer Decoder** – Generates captions word-by-word using attention.
- **Beam Search** – Improves caption quality by exploring multiple decoding paths.
- **Label Smoothing** – Stabilizes training and improves generalization.
- **Configurable Parameters** – Easily adjust embedding size, layers, dropout, beam size, etc.

---

 Project Structure


 
---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-captioning-vit-transformer.git
   cd image-captioning-vit-transformer
pip install torch torchvision transformers pandas pillow tqdm nltk

Example Output:
Image ID	Caption
0001	a group of people standing in a park
0002	a dog running on the grass

 Tech Stack
Python 3.x

PyTorch

HuggingFace Transformers

Pandas, PIL, torchvision, tqdm
