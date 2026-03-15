import numpy as np
import cv2

def apply_circle_mask(frame, radius_scale=0.7):

    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    radius = int(min(center[0], center[1]) * radius_scale)

    cv2.circle(mask, center, radius, 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return masked_frame, mask