import os

def main():
    os.makedirs("data", exist_ok=True)

    while True:
        print("\n=== Pet Finder ===")
        print("1. Crop photos")
        print("2. Train model")
        print("3. Build profile")
        print("4. Scan webcam")
        print("5. Scan video file")
        print("6. Quit")

        choice = input("Enter 1, 2, 3, 4, 5 or 6: ")

        if choice == "1":
            from crop_animal import crop
            crop("data/fiona", "data/cropped_fiona")
            crop("data/other_cats", "data/cropped_other_cats")

        elif choice == "2":
            from train import train
            train()

        elif choice == "3":
            from build_profile import build_profile
            build_profile()

        elif choice == "4":
            from scan import scan_webcam
            scan_webcam(check_every_seconds=3)

        elif choice == "5":
            from scan import scan_video
            video_path = input("Enter path to video file: ")
            scan_video(video_path, check_every_seconds=3)

        elif choice == "6":
            print("Goodbye!")
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()