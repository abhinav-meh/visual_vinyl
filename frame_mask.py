import numpy as np
import cv2

def apply_circle_mask(frame):
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    radius = min(center[0], center[1]) // 2

    cv2.circle(mask, center, radius, 255, -1)

    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_frame, mask