import torch
import numpy as np
import os
import pickle
from PIL import Image
from torchvision import transforms
from train import PetEmbedder

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PetEmbedder().to(device)
model.load_state_dict(torch.load("data/models/pet_embedder.pth"))
model.eval()
print("Trained model loaded!")

def get_embedding(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.squeeze().cpu().numpy()

def build_profile():
    print("Building Fiona's profile...")
    embeddings = []

    for filename in os.listdir("data/cropped_fiona"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join("data/cropped_fiona", filename)
            print(f"Processing {filename}...")
            image = Image.open(path).convert("RGB")
            embedding = get_embedding(image)
            embeddings.append(embedding)

    print(f"Processed {len(embeddings)} photos!")

    pet_profile = np.mean(embeddings, axis=0)
    pet_profile = pet_profile / np.linalg.norm(pet_profile)

    os.makedirs("data/profiles", exist_ok=True)

    with open("data/profiles/fiona_profile.pkl", "wb") as f:
        pickle.dump(pet_profile, f)

    print("Done! Fiona's profile saved!")

if __name__ == "__main__":
    build_profile()