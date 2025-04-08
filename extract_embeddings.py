import os
import torch
import clip
from PIL import Image
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Your image folder
IMAGE_FOLDER = "images"

# Your text descriptions (customize as needed)
TEXTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a sketch of a boy",
    "a ball",
    "a bag",
    "a car",
    "a house",
    "a tree",
    "a bike",
    "a book",
]

output = []

for img_file in os.listdir(IMAGE_FOLDER):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(IMAGE_FOLDER, img_file)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(TEXTS).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy().tolist()[0]
        text_features = model.encode_text(text).cpu().numpy().tolist()

    # Save each pair
    output.append({
        "image_name": img_file,
        "image_embedding": image_features,
        "text_candidates": TEXTS,
        "text_embeddings": text_features
    })


os.makedirs("data", exist_ok=True)
with open("data/embeddings.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Embeddings extracted and saved to data/embeddings.json")
