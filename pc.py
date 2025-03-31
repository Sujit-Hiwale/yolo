from ultralytics import YOLOv10
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

model = YOLOv10.from_pretrained('jameslahm/yolov10x')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model.predict(source=frame)

    frame_with_results = results[0].plot()

    cv2.imshow("YOLOv10 Detection", frame_with_results)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit program on pressing 'q'
        break

cap.release()
cv2.destroyAllWindows()