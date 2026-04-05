import os
import sys
import shutil
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS — allows the Next.js frontend to talk to this backend
# Without this the browser blocks requests between localhost:3000 and localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve spotted images as static files so frontend can display them
os.makedirs("data/spotted", exist_ok=True)
app.mount("/spotted", StaticFiles(directory="data/spotted"), name="spotted")

# ─────────────────────────────────────────
# Health check — just to confirm API is running
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Pet Finder API is running"}

# ─────────────────────────────────────────
# Upload pet photos
# ─────────────────────────────────────────
@app.post("/api/upload-photos")
async def upload_photos(files: list[UploadFile] = File(...)):
    os.makedirs("data/target/photos", exist_ok=True)

    saved = 0
    for file in files:
        # Only accept image files
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        save_path = os.path.join("data/target/photos", file.filename)

        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        saved += 1

    return {
        "status": "success",
        "message": f"Uploaded {saved} photos",
        "count": saved
    }

# ─────────────────────────────────────────
# Crop uploaded photos
# ─────────────────────────────────────────
@app.post("/api/crop-photos")
def crop_photos():
    try:
        from crop_animal import crop
        crop("data/target/photos", "data/target/cropped")
        
        # Count how many crops were created
        cropped_count = len([
            f for f in os.listdir("data/target/cropped")
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
# Build pet profile
# ─────────────────────────────────────────
@app.post("/api/build-profile")
def build_profile():
    # Check that cropped photos exist first
    cropped_folder = "data/target/cropped"
    if not os.path.exists(cropped_folder):
        raise HTTPException(status_code=400, detail="No cropped photos found. Run crop first.")

    cropped_count = len([
        f for f in os.listdir(cropped_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if cropped_count == 0:
        raise HTTPException(status_code=400, detail="No cropped photos found. Run crop first.")

    try:
        from build_profile import build_profile
        build_profile()

        # Clean up photos after profile is built
        shutil.rmtree("data/target/photos")
        shutil.rmtree("data/target/cropped")
        os.makedirs("data/target/photos")
        os.makedirs("data/target/cropped")

        return {
            "status": "success",
            "message": f"Profile built from {cropped_count} photos",
            "photo_count": cropped_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────
# Check if profile exists
# ─────────────────────────────────────────
@app.get("/api/profile-status")
def profile_status():
    profile_exists = os.path.exists("data/profiles/pet_profile.pkl")
    photo_count = 0

    if os.path.exists("data/target/photos"):
        photo_count = len([
            f for f in os.listdir("data/target/photos")
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    return {
        "profile_exists": profile_exists,
        "photo_count": photo_count
    }

# ─────────────────────────────────────────
# Upload and scan a video file
# ─────────────────────────────────────────
@app.post("/api/scan-video")
async def scan_video(
    file: UploadFile = File(...),
    threshold: int = 60
):
    # Check profile exists first
    if not os.path.exists("data/profiles/pet_profile.pkl"):
        raise HTTPException(status_code=400, detail="No pet profile found. Register your pet first.")

    # Save the uploaded video
    os.makedirs("data/uploads", exist_ok=True)
    video_path = f"data/uploads/uploaded_video_{int(time.time())}.mp4"

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        from scan import scan_video
        scan_video(video_path, threshold=threshold)

        # Collect all spotted frames
        spotted_files = sorted([
            f for f in os.listdir("data/spotted")
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Return URLs the frontend can use to display the images
        spotted_urls = [f"/spotted/{f}" for f in spotted_files]

        return {
            "status": "success",
            "message": f"Scan complete! Found {len(spotted_urls)} sightings",
            "sightings": len(spotted_urls),
            "spotted_urls": spotted_urls
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded video after scanning
        if os.path.exists(video_path):
            os.remove(video_path)

# ─────────────────────────────────────────
# Get all spotted frames
# ─────────────────────────────────────────
@app.get("/api/spotted-frames")
def get_spotted_frames():
    if not os.path.exists("data/spotted"):
        return {"frames": []}

    files = sorted([
        f for f in os.listdir("data/spotted")
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ], reverse=True)  # newest first

    frames = []
    for filename in files:
        frames.append({
            "filename": filename,
            "url": f"/spotted/{filename}",
            "timestamp": filename  # filename contains timestamp
        })

    return {"frames": frames}

# ─────────────────────────────────────────
# Clear all spotted frames
# ─────────────────────────────────────────
@app.delete("/api/clear-spotted")
def clear_spotted():
    if os.path.exists("data/spotted"):
        shutil.rmtree("data/spotted")
        os.makedirs("data/spotted")
    return {"status": "success", "message": "Cleared all spotted frames"}

# ─────────────────────────────────────────
# Match a single uploaded image
# ─────────────────────────────────────────
@app.post("/api/match-image")
async def match_image(
    file: UploadFile = File(...),
    threshold: int = 60
):
    if not os.path.exists("data/profiles/pet_profile.pkl"):
        raise HTTPException(status_code=400, detail="No pet profile found. Register your pet first.")

    # Save temp image
    temp_path = f"data/uploads/temp_match_{int(time.time())}.jpg"
    os.makedirs("data/uploads", exist_ok=True)

    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        from match import match
        found = match(temp_path, threshold=threshold)
        return {
            "status": "success",
            "found": found,
            "message": "Pet found!" if found else "Pet not found in this image"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)