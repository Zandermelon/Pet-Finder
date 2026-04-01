import os
from crop_animal import crop
from build_profile import build_profile

def main():
    os.makedirs("data", exist_ok=True)
    
    print("=== Pet Finder ===")
    print("1. Build Fiona's profile (first time setup)")
    print("2. Scan webcam")
    print("3. Scan video file")

    choice = input("Enter 1, 2, 3 or 4: ")

    if choice == "1":
        # Crop Fiona's photos
        crop("data/fiona", "data/cropped_fiona")
        # Crop other cats
        crop("data/other_cats", "data/cropped_other_cats")
        build_profile()
    elif choice == "2":
        from scan import scan_webcam
        scan_webcam(check_every_seconds=3)
    elif choice == "3":
        from scan import scan_video
        video_path = input("Enter path to video file: ")
        scan_video(video_path, check_every_seconds=3)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()