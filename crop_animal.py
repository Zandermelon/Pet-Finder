import os
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print("YOLO loaded!")

os.makedirs("data/cropped_images", exist_ok=True)


def crop_image(path):
    results = model(path, classes=[15,16], conf=0.4, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return []
    image = Image.open(path).convert("RGB")
    crops = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image.crop((x1, y1, x2, y2))
        crops.append(cropped)

    return crops


def crop():
    print("Cropping photos...")
    saved = 0
    skipped = 0

    for filename in os.listdir("data/fiona"):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join("data/fiona", filename)

            crops = crop_image(path)

            if len(crops) == 0:
                print(f"Skipping {filename} - No cat detected!")
                skipped += 1
                continue

            name = os.path.splitext(filename)[0]  # filename without extension
            ext = os.path.splitext(filename)[1]   # just the extension like .jpg

            for i, crop in enumerate(crops):
                save_path = os.path.join("data/cropped_images", f"{name}_cat{i}{ext}")
                crop.save(save_path)
                print(f"Saved crop {i+1} of {len(crops)} from {filename}")

            saved += len(crops)

    print(f"Done! Saved {saved} total crops, skipped {skipped} photos")
    print("Cropped photos are in data/cropped_images")

