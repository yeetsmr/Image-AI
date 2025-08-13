from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import torch
from tqdm import tqdm
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_caption(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=20)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

test_df = pd.read_csv("test.csv")
results = []

for image_id in tqdm(test_df["image_id"]):
    img_path = os.path.join("test", str(image_id) + ".jpg")
    caption = generate_caption(img_path)
    results.append([image_id, caption]) 

  

submission_df = pd.DataFrame(results, columns=["image_id", "caption"])
submission_df.to_csv("submission.csv", index=False)

