import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from crop_animal import crop_image
from build_profile import get_embedding

print("Loading pet profile...")
with open("data/profiles/pet_profile.pkl", "rb") as f:
    pet_profile = pickle.load(f)
print("Profile loaded!")

def score_to_percentage(score):
    clamped = max(-0.5, min(0.5, score))
    percentage = (clamped + 0.5) * 100
    return round(percentage, 1)

def match(image_path, threshold=70):
    crops = crop_image(image_path)

    if len(crops) == 0:
        print("No animals detected")
        return False

    for i, crop in enumerate(crops):
        embedding = get_embedding(crop)

        raw_score = cosine_similarity(
            embedding.reshape(1, -1),
            pet_profile.reshape(1, -1)
        )[0][0]

        confidence = score_to_percentage(raw_score)
        print(f"Animal {i+1}: {confidence}% confidence")

        if confidence >= threshold:
            print(f"Target pet found! ({confidence}% confident)")
            return True

    print("✗ Not a match")
    return False

if __name__ == "__main__":
    print("\n=== Testing ===")
    match("test_target.jpg")