import cv2
import numpy as np
from frame_mask import apply_circle_mask

BRIGHTNESS_THRESHOLD = 40  # tweak this — lower = more sensitive to dark

def read_color_stream():
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise RuntimeError("Could not open webcam")

    last_color = None

    try:
        while True:
            ret, frame = vid.read()
            if not ret:
                continue

            masked_frame, mask = apply_circle_mask(frame)

            pixels = frame[mask == 255]
            b_mean, g_mean, r_mean = pixels.mean(axis=0)

            brightness = (r_mean + g_mean + b_mean) / 3

            # --- BLACK DETECTION ---
            if brightness < BRIGHTNESS_THRESHOLD:
                color = None
            elif b_mean > g_mean and b_mean > r_mean:
                color = "Blue"
            elif g_mean > r_mean and g_mean > b_mean:
                color = "Green"
            else:
                color = "Red"

            # debug display
            label = color if color else "Black (silent)"
            debug = masked_frame.copy()
            cv2.putText(debug,
                        f"{label}  (B:{b_mean:.1f} G:{g_mean:.1f} R:{r_mean:.1f} Br:{brightness:.1f})",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("frame", debug)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if color != last_color:
                last_color = color
                if color is not None:  # only yield when not black
                    yield color

    finally:
        vid.release()
        cv2.destroyAllWindows()ç