from ultralytics import YOLO
from PIL import Image

print("Loading object detection model...")

model = YOLO("yolo11n.pt")

image_path = "images/test.png"

results = model(image_path)


for result in results:
    boxes = result.boxes

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]

        print(f"{label} -> {confidence:.2%}")

    # Save image with bounding boxes
    result.save(filename="detected.jpg")

print("\nDetection image saved!")