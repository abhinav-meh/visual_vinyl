import cv2
import numpy as np
import time
from pythonosc.udp_client import SimpleUDPClient
from frame_mask import apply_circle_mask  # your version that returns (masked_frame, mask)

# ---- OSC client to SuperCollider ----
sc = SimpleUDPClient("127.0.0.1", 57120)

def sc_note_on(note: int, vel: float = 0.7):
    sc.send_message("/noteOn", [int(note), float(vel)])

def sc_note_off(note: int):
    sc.send_message("/noteOff", [int(note)])

def sc_all_off():
    sc.send_message("/allOff", [])

# ---- HSV color buckets (H: 0..179 in OpenCV) ----
COLOR_BUCKETS = {
    "Red":    [(0, 10), (170, 179)],
    "Orange": [(11, 20)],
    "Yellow": [(21, 35)],
    "Green":  [(36, 85)],
    "Cyan":   [(86, 100)],
    "Blue":   [(101, 140)],
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
            scores.append((name, count / total))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def classify(top, chord_min=0.25, dominance_gap=0.18):
    if not top:
        return None, None

    c1, p1 = top[0]
    if len(top) == 1:
        return "SINGLE", c1

    c2, p2 = top[1]

    # chord if second color is strong
    if p2 >= chord_min and (p1 - p2) <= (p1 * 0.6):
        return "CHORD", (c1, c2)

    # single if dominant enough
    if p1 >= chord_min and (p1 - p2) >= dominance_gap:
        return "SINGLE", c1

    return None, None

# ---- Map colors to notes / chords ----
COLOR_TO_NOTE = {
    "Red": 60,     # C4
    "Orange": 62,  # D4
    "Yellow": 64,  # E4
    "Green": 65,   # F4
    "Cyan": 67,    # G4
    "Blue": 69,    # A4
    "Purple": 71,  # B4
}

PAIR_TO_CHORD = {
    ("Red", "Blue"):   [60, 64, 67],  # C major
    ("Blue", "Green"): [57, 60, 64],  # A minor
    ("Red", "Green"):  [62, 65, 69],  # D minor-ish
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

    last_sig = None
    last_notes = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            masked_frame, mask = apply_circle_mask(frame)

            top = top_colors(frame, mask, min_sv=(50, 50), top_k=2)
            event_type, value = classify(top)

            debug = masked_frame.copy()
            txt = " | ".join([f"{c}:{p:.2f}" for c, p in top]) if top else "No color"
            cv2.putText(debug, txt, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if event_type == "SINGLE":
                cv2.putText(debug, f"SINGLE: {value}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            elif event_type == "CHORD":
                cv2.putText(debug, f"CHORD: {value[0]} + {value[1]}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                cv2.putText(debug, "Uncertain", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Visual Vinyl", debug)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            sig = (event_type, value)
            if event_type is not None and sig != last_sig:
                # turn off previous notes
                for n in last_notes:
                    sc_note_off(n)

                # play new notes
                new_notes = notes_for(event_type, value)
                for n in new_notes:
                    sc_note_on(n, vel=0.7)

                last_notes = new_notes
                last_sig = sig

            time.sleep(0.01)

    finally:
        # safety stop
        sc_all_off()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()