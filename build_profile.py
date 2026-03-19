import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import pickle


model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
model.eval()
print("Model Loaded!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_embedding(path):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(tensor)

    return embedding.squeeze().numpy()

def build_profile():
    print("Processing Photos...")
    embeddings = []

    for filename in os.listdir("data/cropped_images"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join("data/cropped_images", filename)
            print(f"Processing {filename}...")
            embedding = get_embedding(path)
            embeddings.append(embedding)

    print(f"Processed {len(embeddings)} photos")

    pet_profile = np.mean(embeddings, axis=0)
    print(f"Fiona's profile shape: {pet_profile.shape}")

    os.makedirs("data/profiles", exist_ok=True)

    with open("data/profiles/fiona_profile.pkl", "wb") as f:
        pickle.dump(pet_profile, f)

    print("Done! Fiona's profile saved to fiona_profile.pkl")
