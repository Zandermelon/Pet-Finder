import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from crop_animal import crop_image
from build_profile import get_embedding

print("Loading Fiona's profile...")
with open("data/profiles/fiona_profile.pkl", "rb") as f:
    fiona_profile = pickle.load(f)
print("Fiona's profile loaded!")

def score_to_percentage(score):
    # Our scores range from about -0.5 to +0.5
    # We map this to 0% to 100%
    # Clamp between -0.5 and 0.5 first to avoid going outside 0-100
    clamped = max(-0.5, min(0.5, score))
    percentage = (clamped + 0.5) * 100
    return round(percentage, 1)

def match(image_path, threshold=70):  # threshold is now a percentage
    crops = crop_image(image_path)

    if len(crops) == 0:
        print("No animals detected in this image")
        return False

    image_opened = __import__('PIL').Image.open(image_path)

    for i, crop in enumerate(crops):
        embedding = get_embedding(crop)

        raw_score = cosine_similarity(
            embedding.reshape(1, -1),
            fiona_profile.reshape(1, -1)
        )[0][0]

        confidence = score_to_percentage(raw_score)

        print(f"Animal {i+1}: {confidence}% confidence it's Fiona")

        if confidence >= threshold:
            print(f"Fiona found! ({confidence}% confident)")
            return True

    print("Not Fiona")
    return False

if __name__ == "__main__":
    print("\n=== Testing with Fiona's photo ===")
    match("test_fiona.jpg")

    print("\n=== Testing with a random cat ===")
    match("test_other_cat.jpg")