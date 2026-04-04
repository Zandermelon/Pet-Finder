import os

def main():
    os.makedirs("data", exist_ok=True)

    while True:
        print("\n=== Pet Finder ===")
        print("1. Crop target pet photos")
        print("2. Build profile")
        print("3. Scan webcam")
        print("4. Scan video file")
        print("5. Quit")

        choice = input("Enter 1-5: ")

        if choice == "1":
            from crop_animal import crop
            crop("data/target/photos", "data/target/cropped")

        elif choice == "2":
            from build_profile import build_profile
            build_profile()

        elif choice == "3":
            from scan import scan_webcam
            scan_webcam(check_every_seconds=3)

        elif choice == "4":
            from scan import scan_video
            video_path = input("Enter path to video file: ")
            scan_video(video_path, check_every_seconds=3)

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()