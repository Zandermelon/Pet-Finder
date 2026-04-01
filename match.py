import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from crop_animal import crop_image
from build_profile import get_embedding

# Load Fiona's saved profile
print("Loading Fiona's profile...")
with open("data/profiles/fiona_profile.pkl", "rb") as f:
    fiona_profile = pickle.load(f)
print("Fiona's profile loaded!")

def match(image_path, threshold=0.7):
    # Step 1 - crop all animals out of the image
    crops = crop_image(image_path)

    if len(crops) == 0:
        print("No animals detected in this image")
        return False

    # Step 2 - check each animal
    for i, crop in enumerate(crops):
        embedding = get_embedding(crop)

        score = cosine_similarity(
            embedding.reshape(1, -1),
            fiona_profile.reshape(1, -1)
        )[0][0]

        print(f"Animal {i+1}: similarity score = {score:.4f}")

        if score >= threshold:
            print(f"Animal {i+1} is Fiona! (score: {score:.4f})")
            return True

    print("No match found — not Fiona")
    return False

if __name__ == "__main__":
    print("\n=== Testing with Fiona's photo ===")
    match("test_fiona.jpg")

    print("\n=== Testing with a random cat ===")
    match("test_other_cat.jpg")