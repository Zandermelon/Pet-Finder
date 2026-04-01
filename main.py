import os
from scan import scan_webcam, scan_video

def main():
    os.makedirs("data", exist_ok=True) # create data folder (other files dependant on it)
    
    choice = input("Enter 1 to scan webcam or 2 to scan video file\n")

    # picks what to scan from
    if choice == "1":
        scan_webcam(check_every_seconds=3)
    elif choice == "2":
        video_path = input("Enter path to video file: ")
        scan_video(video_path, check_every_seconds=3)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()