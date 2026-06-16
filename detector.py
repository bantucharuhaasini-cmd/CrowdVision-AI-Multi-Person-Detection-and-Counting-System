from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  


def detect_persons(image):

    results = model(image, conf=0.5,iou=0.4)  

    person_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # class 0 = person (COCO)
            if cls == 0 and box.conf[0] > 0.6:

                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, "Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    return image, person_count