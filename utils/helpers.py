import cv2
import numpy as np
def read_image(file):
    """
    Convert uploaded file to OpenCV image
    """
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def draw_box(image, x1, y1, x2, y2, label="Person"):
    """
    Draw bounding box with label
    """
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        image,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return image

def format_result(count):
    """
    Return user-friendly text output
    """
    return f"Total Persons Detected: {count}"