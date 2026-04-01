import torch
import numpy as np
import os
import pickle
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load CLIP 
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP model loaded!")

def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        embedding = outputs.pooler_output
    
    # Normalize
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.squeeze().numpy()

def build_profile():
    print("Processing photos...")
    embeddings = []

    for filename in os.listdir("data/cropped_images"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join("data/cropped_images", filename)
            print(f"Processing {filename}...")
            image = Image.open(path).convert("RGB")
            embedding = get_embedding(image)
            embeddings.append(embedding)

    print(f"Processed {len(embeddings)} photos!")

    pet_profile = np.mean(embeddings, axis=0)

    # Normalize the averaged profile too
    pet_profile = pet_profile / np.linalg.norm(pet_profile)

    print(f"Profile shape: {pet_profile.shape}")

    os.makedirs("data/profiles", exist_ok=True)

    with open("data/profiles/fiona_profile.pkl", "wb") as f:
        pickle.dump(pet_profile, f)

    print("Done! Fiona's profile saved!")

if __name__ == "__main__":
    build_profile()