import cv2
import numpy as np
import time
from pythonosc.udp_client import SimpleUDPClient
from frame_mask import apply_circle_mask

# ---- OSC client to SuperCollider ----
sc = SimpleUDPClient("127.0.0.1", 57120)

def sc_note_on(note: int, vel: float = 0.7):
    sc.send_message("/noteOn", [int(note), float(vel)])

def sc_note_off(note: int):
    sc.send_message("/noteOff", [int(note)])

def sc_all_off():
    sc.send_message("/allOff", [])

# ---- HSV color buckets ----
COLOR_BUCKETS = {
    "Red": [(0, 10), (170, 179)],
    "Orange": [(11, 20)],
    "Yellow": [(21, 35)],
    "Green": [(36, 85)],
    "Cyan": [(86, 100)],
    "Blue": [(101, 140)],
    "Purple": [(141, 169)],
}

def hue_in_ranges(h, ranges):
    m = np.zeros_like(h, dtype=bool)
    for lo, hi in ranges:
        m |= (h >= lo) & (h <= hi)
    return m

def top_colors(frame, mask, min_sv=(50, 50), top_k=2):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    pixels = hsv[mask == 255]
    if pixels.size == 0:
        return []

    H, S, V = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    s_min, v_min = min_sv
    valid = (S >= s_min) & (V >= v_min)

    H = H[valid]
    if H.size == 0:
        return []

    total = H.size
    scores = []

    for name, ranges in COLOR_BUCKETS.items():
        count = hue_in_ranges(H, ranges).sum()

        if count:
            ratio = count / total

            # ignore small noisy colors
            if ratio > 0.08:
                scores.append((name, ratio))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def classify(top, chord_min=0.25, dominance_gap=0.18):
    if not top:
        return None, None

    c1, p1 = top[0]

    if len(top) == 1:
        return "SINGLE", c1

    c2, p2 = top[1]

    if p2 >= chord_min and (p1 - p2) <= (p1 * 0.6):
        return "CHORD", (c1, c2)

    if p1 >= chord_min and (p1 - p2) >= dominance_gap:
        return "SINGLE", c1

    return None, None

# ---- Map colors to notes ----
COLOR_TO_NOTE = {
    "Red": 60,
    "Orange": 62,
    "Yellow": 64,
    "Green": 65,
    "Cyan": 67,
    "Blue": 69,
    "Purple": 71,
}

PAIR_TO_CHORD = {
    ("Red", "Blue"): [60, 64, 67],
    ("Blue", "Green"): [57, 60, 64],
    ("Red", "Green"): [62, 65, 69],
}

def notes_for(event_type, value):
    if event_type == "SINGLE":
        return [COLOR_TO_NOTE.get(value, 60)]

    if event_type == "CHORD":
        a, b = value

        if (a, b) in PAIR_TO_CHORD:
            return PAIR_TO_CHORD[(a, b)]

        if (b, a) in PAIR_TO_CHORD:
            return PAIR_TO_CHORD[(b, a)]

        return [COLOR_TO_NOTE.get(a, 60), COLOR_TO_NOTE.get(b, 67)]

    return []

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # reduce CPU load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    last_sig = None
    last_notes = []

    stable_sig = None
    stable_count = 0
    STABLE_FRAMES = 6

    last_trigger_time = 0
    NOTE_COOLDOWN = 0.3

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            frame = cv2.resize(frame, (320, 240))

            masked_frame, mask = apply_circle_mask(frame)

            top = top_colors(frame, mask, min_sv=(50, 50), top_k=2)

            event_type, value = classify(top)

            sig = (event_type, value)

            # temporal smoothing
            if sig == stable_sig:
                stable_count += 1
            else:
                stable_sig = sig
                stable_count = 0

            now = time.time()

            if (
                stable_count >= STABLE_FRAMES
                and sig != last_sig
                and event_type is not None
                and (now - last_trigger_time) > NOTE_COOLDOWN
            ):

                print("Detected:", sig)

                for n in last_notes:
                    sc_note_off(n)

                new_notes = notes_for(event_type, value)

                for n in new_notes:
                    sc_note_on(n, vel=0.7)

                last_notes = new_notes
                last_sig = sig
                last_trigger_time = now

            time.sleep(0.01)

    finally:

        sc_all_off()
        cap.release()

if __name__ == "__main__":
    main()