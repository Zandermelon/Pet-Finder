import torch
import numpy as np
import os
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

print("Loading DINO model...")
extractor = AutoImageProcessor.from_pretrained("facebook/dino-vitb8")
model = AutoModel.from_pretrained("facebook/dino-vitb8")
model.eval()
print("DINO loaded!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_embedding(image):
    inputs = extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Use the CLS token — DINO's summary of the whole image
        embedding = outputs.last_hidden_state[:, 0, :]

    # Normalize
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy()

def build_profile():
    print("Building profile from target photos...")
    embeddings = []

    for filename in os.listdir("data/target/cropped"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join("data/target/cropped", filename)
            print(f"Processing {filename}...")
            image = Image.open(path).convert("RGB")
            embedding = get_embedding(image)
            embeddings.append(embedding)

    print(f"Processed {len(embeddings)} photos!")

    # Save ALL embeddings instead of averaging
    # This keeps all the information about how the pet looks
    profile = np.array(embeddings)
    print(f"Profile shape: {profile.shape}")  # should be (num_photos, 768)

    os.makedirs("data/profiles", exist_ok=True)

    with open("data/profiles/pet_profile.pkl", "wb") as f:
        pickle.dump(profile, f)

    print(f"Done! Profile saved with {len(embeddings)} reference fingerprints")

if __name__ == "__main__":
    build_profile()