import cv2
from ultralytics import YOLOv10
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 150)

video_source = "IP_Address"  # Change with your phone's IP addrss. Can be found out using IP Webcam
cap = cv2.VideoCapture(video_source)

model = YOLOv10.from_pretrained('jameslahm/yolov10x')

last_announcement_time = 0
announcement_interval = 3 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run pred
    results = model.predict(source=frame)

    frame_with_results = results[0].plot()
    cv2.imshow("YOLOv10 Detection", frame_with_results)

    current_time = time.time()
    if current_time - last_announcement_time >= announcement_interval:
        detected_labels = set()

        if hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                class_index = int(box.cls)
                label = results[0].names[class_index]
                detected_labels.add(label)
        else:
            print("No boxes attribute found in results")

        if detected_labels:
            announcement = "Detected: " + ", ".join(detected_labels)
            print(announcement)
            engine.say(announcement)
            engine.runAndWait()

        last_announcement_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
