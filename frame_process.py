from vidgear.gears import CamGear
from ultralytics import YOLO
import time
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')
options = {"STREAM_RESOLUTION": "144p"}
stream = CamGear(source='https://www.youtube.com/watch?v=Gd9d4q6WvUY', stream_mode=True, logging=True, **options).start()
while True:
    frame = stream.read()
    result = model(frame)[0]
    # save class label names
    names = result.names  # same as model.names

    # store number of objects detected per class label
    class_detections_values = []
    for k, v in names.items():
        class_detections_values.append(result.boxes.cls.tolist().count(k))
    # create dictionary of objects detected per class
    classes_detected = dict(zip(names.values(), class_detections_values))
    final_dict = {k: v for k, v in classes_detected.items() if v > 0}
    if bool(final_dict):
        final_dict['recorded_at'] = time.time_ns() // 1_000_000
        print(final_dict)
    time.sleep(0.2)
stream.stop()