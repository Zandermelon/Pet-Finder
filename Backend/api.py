import os
import sys
import uuid
import shutil
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/spotted", exist_ok=True)
app.mount("/spotted", StaticFiles(directory="data/spotted"), name="spotted")

@app.get("/")
def root():
    return {"status": "Pet Finder API is running"}

# ─────────────────────────────────────────
# Upload pet photos — returns a session_id
# ─────────────────────────────────────────
@app.post("/api/upload-photos")
async def upload_photos(files: list[UploadFile] = File(...)):
    # Generate a unique session ID for this user
    session_id = str(uuid.uuid4())

    photos_folder = f"data/sessions/{session_id}/photos"
    os.makedirs(photos_folder, exist_ok=True)

    saved = 0
    for file in files:
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        save_path = os.path.join(photos_folder, file.filename)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        saved += 1

    return {
        "status": "success",
        "message": f"Uploaded {saved} photos",
        "count": saved,
        "session_id": session_id  # send this back to the frontend
    }

# ─────────────────────────────────────────
# Crop photos — needs session_id
# ─────────────────────────────────────────
@app.post("/api/crop-photos/{session_id}")
def crop_photos(session_id: str):
    photos_folder = f"data/sessions/{session_id}/photos"
    cropped_folder = f"data/sessions/{session_id}/cropped"

    if not os.path.exists(photos_folder):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        from crop_animal import crop
        crop(photos_folder, cropped_folder)

        cropped_count = len([
            f for f in os.listdir(cropped_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        return {
            "status": "success",
            "message": f"Cropped {cropped_count} images",
            "count": cropped_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# Build profile — needs session_id
# ─────────────────────────────────────────
@app.post("/api/build-profile/{session_id}")
def build_profile_endpoint(session_id: str):
    cropped_folder = f"data/sessions/{session_id}/cropped"
    profile_path = f"data/sessions/{session_id}/pet_profile.pkl"

    if not os.path.exists(cropped_folder):
        raise HTTPException(status_code=404, detail="Session not found")

    cropped_count = len([
        f for f in os.listdir(cropped_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if cropped_count == 0:
        raise HTTPException(status_code=400, detail="No cropped photos found")

    try:
        import torch
        import numpy as np
        import pickle
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb8")
        model = AutoModel.from_pretrained("facebook/dino-vitb8")
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        def get_embedding(image):
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.squeeze().cpu().numpy()

        embeddings = []
        for filename in os.listdir(cropped_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(cropped_folder, filename)
                image = Image.open(path).convert("RGB")
                embedding = get_embedding(image)
                embeddings.append(embedding)

        pet_profile = np.mean(embeddings, axis=0)
        pet_profile = pet_profile / np.linalg.norm(pet_profile)

        # Save profile to session folder
        with open(profile_path, "wb") as f:
            pickle.dump(pet_profile, f)

        # Clean up photos and crops after profile is built
        shutil.rmtree(f"data/sessions/{session_id}/photos")
        shutil.rmtree(f"data/sessions/{session_id}/cropped")

        return {
            "status": "success",
            "message": f"Profile built from {cropped_count} photos",
            "photo_count": cropped_count,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# Profile status — needs session_id
# ─────────────────────────────────────────
@app.get("/api/profile-status/{session_id}")
def profile_status(session_id: str):
    profile_exists = os.path.exists(f"data/sessions/{session_id}/pet_profile.pkl")
    return {"profile_exists": profile_exists, "session_id": session_id}

# ─────────────────────────────────────────
# Scan video — needs session_id
# ─────────────────────────────────────────
@app.post("/api/scan-video/{session_id}")
async def scan_video_endpoint(
    session_id: str,
    file: UploadFile = File(...),
    threshold: int = 60
):
    profile_path = f"data/sessions/{session_id}/pet_profile.pkl"

    if not os.path.exists(profile_path):
        raise HTTPException(status_code=400, detail="No pet profile found for this session")

    # Save uploaded video to session folder
    video_folder = f"data/sessions/{session_id}/videos"
    os.makedirs(video_folder, exist_ok=True)
    video_path = f"{video_folder}/upload_{int(time.time())}.mp4"

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Session specific spotted folder
    spotted_folder = f"data/sessions/{session_id}/spotted"
    os.makedirs(spotted_folder, exist_ok=True)

    try:
        import cv2
        import pickle
        import numpy as np
        from PIL import Image
        from sklearn.metrics.pairwise import cosine_similarity
        from crop_animal import crop_image
        import torch
        from transformers import AutoImageProcessor, AutoModel

        # Load this user's profile
        with open(profile_path, "rb") as f:
            pet_profile = pickle.load(f)

        # Load DINO
        processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb8")
        model = AutoModel.from_pretrained("facebook/dino-vitb8")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        def get_embedding(image):
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.squeeze().cpu().numpy()

        def score_to_percentage(score):
            clamped = max(0.4, min(0.7, score))
            return round(((clamped - 0.4) / 0.3) * 100, 1)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise HTTPException(status_code=400, detail="Could not read video FPS")

        frames_to_skip = max(1, int(fps * 3))
        frame_count = 0
        matches_found = 0
        spotted_urls = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frames_to_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save("temp_scan_frame.jpg")

                crops = crop_image("temp_scan_frame.jpg")

                for crop in crops:
                    embedding = get_embedding(crop)
                    scores = cosine_similarity(
                        embedding.reshape(1, -1),
                        pet_profile
                    )[0]
                    top5_score = float(np.mean(np.sort(scores)[-5:]))
                    confidence = score_to_percentage(top5_score)

                    if confidence >= threshold:
                        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                        save_path = f"{spotted_folder}/spotted_{timestamp_str}_{matches_found}.jpg"
                        cv2.imwrite(save_path, frame)
                        spotted_urls.append(
                            f"/spotted/{session_id}/spotted_{timestamp_str}_{matches_found}.jpg"
                        )
                        matches_found += 1
                        break

            frame_count += 1

        cap.release()
        if os.path.exists("temp_scan_frame.jpg"):
            os.remove("temp_scan_frame.jpg")
        if os.path.exists(video_path):
            os.remove(video_path)

        return {
            "status": "success",
            "message": f"Scan complete! Found {matches_found} sightings",
            "sightings": matches_found,
            "spotted_urls": spotted_urls
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# Match single image — needs session_id
# ─────────────────────────────────────────
@app.post("/api/match-image/{session_id}")
async def match_image(
    session_id: str,
    file: UploadFile = File(...),
    threshold: int = 60
):
    profile_path = f"data/sessions/{session_id}/pet_profile.pkl"

    if not os.path.exists(profile_path):
        raise HTTPException(status_code=400, detail="No pet profile found for this session")

    temp_path = f"data/sessions/{session_id}/temp_match_{int(time.time())}.jpg"

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        import pickle
        import numpy as np
        import torch
        from PIL import Image
        from sklearn.metrics.pairwise import cosine_similarity
        from transformers import AutoImageProcessor, AutoModel
        from crop_animal import crop_image

        with open(profile_path, "rb") as f:
            pet_profile = pickle.load(f)

        processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb8")
        model = AutoModel.from_pretrained("facebook/dino-vitb8")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        def get_embedding(image):
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :]
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            return embedding.squeeze().cpu().numpy()

        def score_to_percentage(score):
            clamped = max(0.4, min(0.7, score))
            return round(((clamped - 0.4) / 0.3) * 100, 1)

        crops = crop_image(temp_path)
        found = False

        for crop in crops:
            embedding = get_embedding(crop)
            scores = cosine_similarity(
                embedding.reshape(1, -1),
                pet_profile
            )[0]
            top5_score = float(np.mean(np.sort(scores)[-5:]))
            confidence = score_to_percentage(top5_score)

            if confidence >= threshold:
                found = True
                break

        return {
            "status": "success",
            "found": found,
            "message": "Pet found!" if found else "Pet not found"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─────────────────────────────────────────
# Get spotted frames — needs session_id
# ─────────────────────────────────────────
@app.get("/api/spotted-frames/{session_id}")
def get_spotted_frames(session_id: str):
    spotted_folder = f"data/sessions/{session_id}/spotted"

    if not os.path.exists(spotted_folder):
        return {"frames": []}

    files = sorted([
        f for f in os.listdir(spotted_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ], reverse=True)

    frames = [
        {
            "filename": f,
            "url": f"/sessions/{session_id}/spotted/{f}"
        }
        for f in files
    ]

    return {"frames": frames}

# Serve session spotted images
from fastapi.staticfiles import StaticFiles
os.makedirs("data/sessions", exist_ok=True)
app.mount("/sessions", StaticFiles(directory="data/sessions"), name="sessions")