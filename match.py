import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from crop_animal import crop_image
from build_profile import get_embedding

print("Loading pet profile...")
with open("data/profiles/pet_profile.pkl", "rb") as f:
    pet_profile = pickle.load(f)
print(f"Profile loaded! ({len(pet_profile)} reference photos)")

def score_to_percentage(score):
    # DINO scores range roughly from 0.4 to 0.7
    # Map this range to 0-100%
    clamped = max(0.4, min(0.7, score))
    percentage = ((clamped - 0.4) / 0.3) * 100
    return round(percentage, 1)

def match(image_path, threshold=60):
    crops = crop_image(image_path)

    if len(crops) == 0:
        print("No animals detected")
        return False

    for i, crop in enumerate(crops):
        embedding = get_embedding(crop)

        # Compare against ALL reference photos
        scores = cosine_similarity(
            embedding.reshape(1, -1),
            pet_profile
        )[0]

        # Take the best match
        best_score = float(np.max(scores))
        avg_score = float(np.mean(scores))
        top5_score = float(np.mean(np.sort(scores)[-5:]))

        confidence = score_to_percentage(top5_score)

        print(f"Animal {i+1}:")
        print(f"  Best match:    {best_score:.4f}")
        print(f"  Top 5 avg:     {top5_score:.4f}")
        print(f"  Overall avg:   {avg_score:.4f}")
        print(f"  Confidence:    {confidence}%")

        if confidence >= threshold:
            print(f"Target pet found! ({confidence}% confident)")
            return True

    print("Not a match")
    return False

if __name__ == "__main__":
    print("\n=== Testing ===")
    match("test_target.jpg")