import os
import re
import uuid
import shutil
import time
import threading
import asyncio
from functools import partial
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModel

print("Loading DINO model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb8")
dino_model = AutoModel.from_pretrained("facebook/dino-vitb8")
dino_model.eval()
dino_model = dino_model.to(device)
dino_lock = threading.Lock()
print("DINO loaded!")

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
PET_NAME_RE = re.compile(r"^[a-zA-Z0-9 \-]{1,50}$")

SESSION_TTL_SECONDS = 30 * 60  # delete sessions idle for 30+ minutes
HEARTBEAT_INTERVAL = 60        # check for expired sessions every 60 seconds

_session_last_seen: dict[str, float] = {}
_session_lock = threading.Lock()


def _touch_session(session_id: str):
    with _session_lock:
        _session_last_seen[session_id] = time.time()


def _cleanup_expired_sessions():
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        now = time.time()
        with _session_lock:
            expired = [
                sid for sid, last in _session_last_seen.items()
                if now - last > SESSION_TTL_SECONDS
            ]
            for sid in expired:
                del _session_last_seen[sid]
        for sid in expired:
            shutil.rmtree(f"data/profiles/{sid}", ignore_errors=True)
            shutil.rmtree(f"data/sessions/{sid}", ignore_errors=True)


threading.Thread(target=_cleanup_expired_sessions, daemon=True).start()

def validate_session_id(session_id: str):
    if not UUID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")

def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", name.strip().lower())[:50]

def validate_pet_name(pet_name: str):
    if not PET_NAME_RE.match(pet_name.strip()):
        raise HTTPException(status_code=400, detail="Pet name must be 1–50 characters (letters, numbers, spaces, hyphens)")
    return slugify(pet_name)

def get_embedding(image):
    inputs = dino_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with dino_lock:
        with torch.no_grad():
            outputs = dino_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze().cpu().numpy()

def score_to_percentage(score):
    # DINO cosine similarity ranges (ViT-B/8, no fine-tuning):
    #   different individuals of same species: ~0.50–0.70
    #   same individual:                       ~0.65–0.85
    # Map [0.55, 0.85] → [0%, 100%]
    clamped = max(0.55, min(0.85, score))
    return round(((clamped - 0.55) / 0.30) * 100, 1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/sessions", exist_ok=True)
os.makedirs("data/profiles", exist_ok=True)
app.mount("/sessions", StaticFiles(directory="data/sessions"), name="sessions")

@app.get("/")
def root():
    return {"status": "Pet Finder API is running"}

# ─────────────────────────────────────────
# Upload profile video — extracts frames, returns session_id
# ─────────────────────────────────────────
def _extract_frames(video_path: str, photos_folder: str) -> int:
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise ValueError("Could not read video FPS")
        frames_to_skip = max(1, int(fps))
        frame_count = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frames_to_skip == 0:
                cv2.imwrite(f"{photos_folder}/frame_{saved:05d}.jpg", frame)
                saved += 1
            frame_count += 1
        cap.release()
        return saved
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.post("/api/upload-profile-video")
async def upload_profile_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    session_id = str(uuid.uuid4())
    photos_folder = f"data/sessions/{session_id}/photos"
    os.makedirs(photos_folder, exist_ok=True)

    video_path = f"data/sessions/{session_id}/profile_video_{int(time.time())}.mp4"
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    loop = asyncio.get_event_loop()
    try:
        saved = await loop.run_in_executor(None, partial(_extract_frames, video_path, photos_folder))
    except ValueError as e:
        shutil.rmtree(f"data/sessions/{session_id}", ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))

    if saved == 0:
        shutil.rmtree(f"data/sessions/{session_id}", ignore_errors=True)
        raise HTTPException(status_code=400, detail="No frames could be extracted from video")

    return {
        "status": "success",
        "message": f"Extracted {saved} frames from video",
        "count": saved,
        "session_id": session_id
    }

# ─────────────────────────────────────────
# Upload pet photos — returns a session_id
# ─────────────────────────────────────────
@app.post("/api/upload-photos")
async def upload_photos(files: list[UploadFile] = File(...)):
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
        "session_id": session_id
    }

# ─────────────────────────────────────────
# Crop photos — needs session_id
# ─────────────────────────────────────────
@app.post("/api/crop-photos/{session_id}")
def crop_photos(session_id: str):
    validate_session_id(session_id)
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
# Build profile — needs session_id + pet_name
# ─────────────────────────────────────────
@app.post("/api/build-profile/{session_id}")
def build_profile_endpoint(session_id: str, pet_name: str = Query(...)):
    validate_session_id(session_id)
    pet_slug = validate_pet_name(pet_name)

    cropped_folder = f"data/sessions/{session_id}/cropped"
    pet_profile_dir = f"data/profiles/{session_id}/{pet_slug}"
    profile_path = f"{pet_profile_dir}/pet_profile.pkl"

    if not os.path.exists(cropped_folder):
        raise HTTPException(status_code=404, detail="Session not found")

    cropped_count = len([
        f for f in os.listdir(cropped_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if cropped_count == 0:
        raise HTTPException(status_code=400, detail="No cropped photos found")

    try:
        import pickle
        from PIL import Image

        embeddings = []
        for filename in os.listdir(cropped_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(cropped_folder, filename)
                image = Image.open(path).convert("RGB")
                embeddings.append(get_embedding(image))

        pet_profile = np.mean(embeddings, axis=0)
        pet_profile = pet_profile / np.linalg.norm(pet_profile)

        os.makedirs(pet_profile_dir, exist_ok=True)
        with open(profile_path, "wb") as f:
            pickle.dump(pet_profile, f)

        # Save display name alongside the profile
        with open(f"{pet_profile_dir}/name.txt", "w") as f:
            f.write(pet_name.strip())

        # Clean up all upload/crop folders for this session
        session_dir = f"data/sessions/{session_id}"
        for entry in os.scandir(session_dir):
            if entry.is_dir():
                shutil.rmtree(entry.path)

        return {
            "status": "success",
            "message": f"Profile built from {cropped_count} photos",
            "photo_count": cropped_count,
            "session_id": session_id,
            "pet_name": pet_name.strip(),
            "pet_slug": pet_slug
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# List pets for a session
# ─────────────────────────────────────────
@app.get("/api/pets/{session_id}")
def list_pets(session_id: str):
    validate_session_id(session_id)
    session_profiles = f"data/profiles/{session_id}"

    if not os.path.exists(session_profiles):
        return {"pets": []}

    pets = []
    for entry in os.scandir(session_profiles):
        if entry.is_dir():
            name_file = f"{entry.path}/name.txt"
            display_name = entry.name  # fallback
            if os.path.exists(name_file):
                with open(name_file) as f:
                    display_name = f.read().strip()
            pets.append({"slug": entry.name, "name": display_name})

    return {"pets": sorted(pets, key=lambda p: p["name"])}

# ─────────────────────────────────────────
# Profile status — checks if a specific pet profile exists
# ─────────────────────────────────────────
@app.get("/api/profile-status/{session_id}")
def profile_status(session_id: str, pet_slug: str = Query(None)):
    validate_session_id(session_id)
    if pet_slug:
        profile_exists = os.path.exists(f"data/profiles/{session_id}/{pet_slug}/pet_profile.pkl")
    else:
        # True if any pet profile exists for this session
        session_profiles = f"data/profiles/{session_id}"
        profile_exists = os.path.exists(session_profiles) and any(
            e.is_dir() for e in os.scandir(session_profiles)
        ) if os.path.exists(session_profiles) else False
    return {"profile_exists": profile_exists, "session_id": session_id}

# ─────────────────────────────────────────
# Delete a specific pet profile
# ─────────────────────────────────────────
@app.delete("/api/pets/{session_id}/{pet_slug}")
def delete_pet(session_id: str, pet_slug: str):
    validate_session_id(session_id)
    pet_dir = f"data/profiles/{session_id}/{pet_slug}"
    if not os.path.exists(pet_dir):
        raise HTTPException(status_code=404, detail="Pet profile not found")
    shutil.rmtree(pet_dir)
    return {"status": "success", "message": f"Deleted profile for {pet_slug}"}

# ─────────────────────────────────────────
# Heartbeat — keeps session alive
# ─────────────────────────────────────────
@app.post("/api/session/{session_id}/heartbeat")
def heartbeat(session_id: str):
    validate_session_id(session_id)
    _touch_session(session_id)
    return {"status": "ok"}

# ─────────────────────────────────────────
# End session — deletes all profiles and session data
# ─────────────────────────────────────────
@app.delete("/api/session/{session_id}")
def end_session(session_id: str):
    validate_session_id(session_id)
    with _session_lock:
        _session_last_seen.pop(session_id, None)
    shutil.rmtree(f"data/profiles/{session_id}", ignore_errors=True)
    shutil.rmtree(f"data/sessions/{session_id}", ignore_errors=True)
    return {"status": "success", "message": "Session ended and all data deleted"}

# ─────────────────────────────────────────
# Scan video — needs session_id + pet_slug
# ─────────────────────────────────────────
def _run_scan(video_path: str, profile_path: str, spotted_folder: str, session_id: str, threshold: int):
    import cv2
    import pickle
    from PIL import Image
    from sklearn.metrics.pairwise import cosine_similarity
    from crop_animal import crop_image

    with open(profile_path, "rb") as f:
        pet_profile = pickle.load(f)
    pet_profile_2d = pet_profile.reshape(1, -1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Could not read video FPS")

    frames_to_skip = max(1, int(fps * 3))
    frame_count = 0
    matches_found = 0
    spotted_urls = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frames_to_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                temp_frame_path = f"data/sessions/{session_id}/temp_frame_{frame_count}.jpg"
                pil_image.save(temp_frame_path)
                crops = crop_image(temp_frame_path)
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)

                for crop in crops:
                    embedding = get_embedding(crop)
                    score = float(cosine_similarity(
                        embedding.reshape(1, -1), pet_profile_2d
                    )[0][0])
                    if score_to_percentage(score) >= threshold:
                        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                        filename = f"spotted_{timestamp_str}_{matches_found}.jpg"
                        cv2.imwrite(f"{spotted_folder}/{filename}", frame)
                        spotted_urls.append(f"/sessions/{session_id}/spotted/{filename}")
                        matches_found += 1
                        break

            frame_count += 1
    finally:
        cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

    return matches_found, spotted_urls

@app.post("/api/scan-video/{session_id}")
async def scan_video_endpoint(
    session_id: str,
    file: UploadFile = File(...),
    pet_slug: str = Query(...),
    threshold: int = Query(65)
):
    validate_session_id(session_id)
    profile_path = f"data/profiles/{session_id}/{pet_slug}/pet_profile.pkl"

    if not os.path.exists(profile_path):
        raise HTTPException(status_code=400, detail="No pet profile found for this session")

    video_folder = f"data/sessions/{session_id}/videos"
    os.makedirs(video_folder, exist_ok=True)
    video_path = f"{video_folder}/upload_{int(time.time())}.mp4"
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    spotted_folder = f"data/sessions/{session_id}/spotted"
    os.makedirs(spotted_folder, exist_ok=True)

    loop = asyncio.get_event_loop()
    try:
        matches_found, spotted_urls = await loop.run_in_executor(
            None, partial(_run_scan, video_path, profile_path, spotted_folder, session_id, threshold)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "message": f"Scan complete! Found {matches_found} sightings",
        "sightings": matches_found,
        "spotted_urls": spotted_urls
    }

# ─────────────────────────────────────────
# Match single image — needs session_id + pet_slug
# ─────────────────────────────────────────
def _run_match(temp_path: str, profile_path: str, threshold: int) -> bool:
    import pickle
    from sklearn.metrics.pairwise import cosine_similarity
    from crop_animal import crop_image

    with open(profile_path, "rb") as f:
        pet_profile = pickle.load(f)
    pet_profile_2d = pet_profile.reshape(1, -1)

    for crop in crop_image(temp_path):
        embedding = get_embedding(crop)
        score = float(cosine_similarity(embedding.reshape(1, -1), pet_profile_2d)[0][0])
        if score_to_percentage(score) >= threshold:
            return True
    return False

@app.post("/api/match-image/{session_id}")
async def match_image(
    session_id: str,
    file: UploadFile = File(...),
    pet_slug: str = Query(...),
    threshold: int = Query(65)
):
    validate_session_id(session_id)
    profile_path = f"data/profiles/{session_id}/{pet_slug}/pet_profile.pkl"

    if not os.path.exists(profile_path):
        raise HTTPException(status_code=400, detail="No pet profile found for this session")

    temp_path = f"data/sessions/{session_id}/temp_match_{int(time.time())}.jpg"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    loop = asyncio.get_event_loop()
    try:
        found = await loop.run_in_executor(
            None, partial(_run_match, temp_path, profile_path, threshold)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "status": "success",
        "found": found,
        "message": "Pet found!" if found else "Pet not found"
    }

# ─────────────────────────────────────────
# Get spotted frames — needs session_id
# ─────────────────────────────────────────
@app.get("/api/spotted-frames/{session_id}")
def get_spotted_frames(session_id: str):
    validate_session_id(session_id)
    spotted_folder = f"data/sessions/{session_id}/spotted"

    if not os.path.exists(spotted_folder):
        return {"frames": []}

    files = sorted([
        f for f in os.listdir(spotted_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ], reverse=True)

    return {"frames": [
        {"filename": f, "url": f"/sessions/{session_id}/spotted/{f}"}
        for f in files
    ]}

# ─────────────────────────────────────────
# Clear spotted frames — needs session_id
# ─────────────────────────────────────────
@app.delete("/api/clear-spotted/{session_id}")
def clear_spotted(session_id: str):
    validate_session_id(session_id)
    spotted_folder = f"data/sessions/{session_id}/spotted"
    if os.path.exists(spotted_folder):
        shutil.rmtree(spotted_folder)
        os.makedirs(spotted_folder)
    return {"status": "success", "message": "Cleared all spotted frames"}
