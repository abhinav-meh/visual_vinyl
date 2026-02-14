import mido

COLOR_TO_NOTE = {
    "Red": 60,
    "Green": 64,
    "Blue": 67,
}

def main():
    print("MIDI outputs:", mido.get_output_names())
    out = mido.open_output("IAC Driver Bus 1")

    last_note = None
    try:
        from read_color import read_color_stream

        for color in read_color_stream():
            if last_note is not None:
                out.send(mido.Message("note_off", note=last_note, velocity=0))

            note = COLOR_TO_NOTE[color]
            out.send(mido.Message("note_on", note=note, velocity=90))

            last_note = note

    finally:
        if last_note is not None:
            out.send(mido.Message("note_off", note=last_note, velocity=0))
        out.close()

if __name__ == "__main__":
    main()
