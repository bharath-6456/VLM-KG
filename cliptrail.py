import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load the model
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image = preprocess(Image.open("image.png")).unsqueeze(0).to(device)

# Prepare the text inputs
text = clip.tokenize(["a bag", "a ball", "a dog"]).to(device)

# Run the model
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # Example: [[0.98, 0.01, 0.01]]
