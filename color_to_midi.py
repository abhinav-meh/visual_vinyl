import cv2
import numpy as np
import mido
import time
from frame_mask import apply_circle_mask  # still used for nice circular preview/debug

COLOR_BUCKETS = {
    "Red":    [(0, 10), (170, 179)],
    "Orange": [(11, 20)],
    "Yellow": [(21, 35)],
    "Green":  [(36, 85)],
    "Cyan":   [(86, 100)],
    "Blue":   [(101, 140)],
    "Purple": [(141, 169)],
}

def _hue_in_ranges(h, ranges):
    m = np.zeros_like(h, dtype=bool)
    for lo, hi in ranges:
        m |= (h >= lo) & (h <= hi)
    return m

def detect_top_colors(frame, mask, min_sv=(50, 50), top_k=2):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]
    if pixels.size == 0:
        return []

    H = pixels[:, 0]
    S = pixels[:, 1]
    V = pixels[:, 2]

    s_min, v_min = min_sv
    valid = (S >= s_min) & (V >= v_min)
    H = H[valid]
    if H.size == 0:
        return []

    total = H.size
    scores = []
    for name, ranges in COLOR_BUCKETS.items():
        count = _hue_in_ranges(H, ranges).sum()
        if count:
            scores.append((name, count / total))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def classify_color_or_chord(top, chord_min=0.25, dominance_gap=0.18):
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

def get_notes(event_type, value):
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

def make_ring_masks(h, w, center=None, ring_specs=None):

    if center is None:
        center = (w // 2, h // 2)

    if ring_specs is None:
        R = min(w, h) // 2
        ring_specs = [
            (int(R * 0.10), int(R * 0.25)),  # inner ring
            (int(R * 0.28), int(R * 0.45)),  # middle ring
            (int(R * 0.48), int(R * 0.68)),  # outer ring
        ]

    masks = []
    for r_in, r_out in ring_specs:
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(outer, center, r_out, 255, -1)
        cv2.circle(inner, center, r_in, 255, -1)
        ring = cv2.bitwise_and(outer, cv2.bitwise_not(inner))
        masks.append(ring)

    return masks, ring_specs

def note_on(out, notes, vel=90, channel=0):
    for n in notes:
        out.send(mido.Message("note_on", note=int(n), velocity=int(vel), channel=int(channel)))

def note_off(out, notes, channel=0):
    for n in notes:
        out.send(mido.Message("note_off", note=int(n), velocity=0, channel=int(channel)))

def pick_output_port(prefer="IAC"):
    outs = mido.get_output_names()
    if not outs:
        raise RuntimeError("No MIDI outputs found. Enable IAC Driver or install a synth.")
    for name in outs:
        if prefer in name:
            return name
    return outs[0]

def transpose(notes, semitones):
    return [int(n + semitones) for n in notes]

def main():
    port = pick_output_port("IAC")
    print("Using MIDI output:", port)
    out = mido.open_output(port)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    last_signature = [None, None, None]
    last_notes = [[], [], []]

    ring_channels = [0, 1, 2]        # inner=bass, middle=synth, outer=beat
    ring_transpose = [-12, 0, 0]     # inner ring one octave down

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            center = (w // 2, h // 2)

            masked_frame, _circle_mask = apply_circle_mask(frame)

            ring_masks, ring_specs = make_ring_masks(h, w, center=center)

            debug = masked_frame.copy()

            for i in range(3):
                top = detect_top_colors(frame, ring_masks[i], min_sv=(50, 50), top_k=2)
                event_type, value = classify_color_or_chord(top, chord_min=0.25, dominance_gap=0.18)

                r_in, r_out = ring_specs[i]
                cv2.circle(debug, center, r_in, (255, 255, 255), 1)
                cv2.circle(debug, center, r_out, (255, 255, 255), 1)

                label = ["Inner", "Middle", "Outer"][i]
                txt = " | ".join([f"{c}:{p:.2f}" for c, p in top]) if top else "No color"
                cv2.putText(debug, f"{label}: {txt}", (20, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                signature = (event_type, value)

                if event_type is not None and signature != last_signature[i]:
                    if last_notes[i]:
                        note_off(out, last_notes[i], channel=ring_channels[i])

                    notes = get_notes(event_type, value)
                    notes = transpose(notes, ring_transpose[i])

                    note_on(out, notes, vel=90, channel=ring_channels[i])

                    last_notes[i] = notes
                    last_signature[i] = signature

            cv2.imshow("Visual Vinyl - 3 Rings (MIDI)", debug)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.01)

    finally:
        for i in range(3):
            if last_notes[i]:
                note_off(out, last_notes[i], channel=ring_channels[i])
        cap.release()
        cv2.destroyAllWindows()
        out.close()

if __name__ == "__main__":
    main()