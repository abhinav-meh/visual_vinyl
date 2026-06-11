import numpy as np
import cv2

def apply_circle_mask(frame):
    h, w = frame.shape[:2]

    # true center of the frame
    cx = w // 2
    cy = h // 2

    # radius as a fraction of the smaller dimension
    # so it works regardless of resolution
    radius = int(min(w, h) * 0.45)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    masked = cv2.bitwise_and(frame, frame, mask=mask)

    return masked, mask