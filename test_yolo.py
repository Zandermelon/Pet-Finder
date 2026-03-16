from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
# webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
webcam = cv2.VideoCapture("catVideo.mp4")

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, classes=[15,16])

    annotated_frame = results[0].plot()

    cv2.imshow("Webcam Footage", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()