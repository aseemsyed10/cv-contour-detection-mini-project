"""Reproducible OpenCV edge and contour demo."""
from pathlib import Path
import cv2
import numpy as np

def make_example_image(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((420, 640, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (55, 80), (250, 310), (40, 95, 210), -1)
    cv2.circle(image, (425, 205), 105, (45, 170, 65), -1)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save image at {path}")
    return image

def detect_contours(image, canny_low=30, canny_high=100, min_area=800, top_k=8):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) >= min_area][:top_k]
    overlay = image.copy()
    cv2.drawContours(overlay, selected, -1, (0, 255, 0), 2)
    return {"gray": gray, "edges": edges, "closed": closed, "overlay": overlay, "contours": selected}
