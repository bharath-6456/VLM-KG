import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_clip_predictions(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_labels = ["a person", "a dog", "a laptop", "a cat", "a car", "a phone"]
    text = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    return list(zip(text_labels, map(lambda p: round(float(p), 4), probs)))
