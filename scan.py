import cv2
import time
import os
from PIL import Image
from match import match

os.makedirs("data/spotted", exist_ok=True)

def convert_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def scan_webcam(check_every_seconds=3, threshold=60):
    print("Starting webcam scan...")
    print("Press Q to quit")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam!")
        return

    last_check = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame!")
                break

            cv2.imshow("Pet Finder — Press Q to quit", frame)

            current_time = time.time()
            if current_time - last_check >= check_every_seconds:
                print(f"\nChecking frame at {time.strftime('%H:%M:%S')}...")

                image = convert_frame(frame)
                image.save("temp_frame.jpg")

                found = match("temp_frame.jpg", threshold=threshold)

                if found:
                    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                    save_path = f"data/spotted/webcam_{timestamp_str}.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"TARGET SPOTTED ON WEBCAM!")
                    print(f"Frame saved to {save_path}")

                last_check = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopping scan...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
            print("Cleaned up temp files")


def scan_video(video_path, check_every_seconds=3, threshold=60):
    print(f"Scanning video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        print("Could not read video FPS!")
        cap.release()
        return

    duration = total_frames / fps
    print(f"Video is {duration:.1f} seconds long at {fps:.1f} fps")

    frames_to_skip = max(1, int(fps * check_every_seconds))
    frame_count = 0
    matches_found = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Finished scanning video!")
                break

            if frame_count % frames_to_skip == 0:
                timestamp = frame_count / fps
                print(f"\nChecking frame at {timestamp:.1f}s...")

                image = convert_frame(frame)
                image.save("temp_frame.jpg")

                found = match("temp_frame.jpg", threshold=threshold)

                if found:
                    timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                    save_path = f"data/spotted/video_{timestamp_str}_at_{timestamp:.0f}s.jpg"
                    cv2.imwrite(save_path, frame)
                    print(f"TARGET SPOTTED at {timestamp:.1f}s!")
                    print(f"Frame saved to {save_path}")
                    matches_found += 1

            frame_count += 1

    finally:
        cap.release()
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
            print("Cleaned up temp files")

    print(f"\nScan complete! Target spotted in {matches_found} frames")

    cap.release()

    # Clean up temp file
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")

    print(f"\nScan complete! Target spotted in {matches_found} frames")