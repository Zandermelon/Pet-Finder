import os

def main():
    os.makedirs("data", exist_ok=True)

    while True:
        print("\n=== Pet Finder ===")
        print("1. Crop training photos")
        print("2. Train model")
        print("3. Build profile")
        print("4. Scan webcam")
        print("5. Scan video file")
        print("6. Quit")

        choice = input("Enter 1-7: ")

        if choice == "1":
            from crop_animal import crop
            crop("data/training/cats/same_cat",       "data/training/cats/same_cat_cropped")
            crop("data/training/cats/different_cats", "data/training/cats/different_cats_cropped")
            crop("data/training/dogs/same_dog",       "data/training/dogs/same_dog_cropped")
            crop("data/training/dogs/different_dogs", "data/training/dogs/different_dogs_cropped")
            crop("data/target/photos", "data/target/cropped")

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