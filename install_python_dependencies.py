
from __future__ import annotations

import subprocess
import sys
from typing import Iterable


REQUIREMENTS: list[str] = [
    # Core
    "numpy",
    "opencv-python",
    # MIDI
    "mido",
]


def _run(cmd: Iterable[str]) -> None:
    subprocess.check_call(list(cmd))


def main() -> None:
    # Ensure pip itself is available/updated enough
    _run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install dependencies
    _run([sys.executable, "-m", "pip", "install", "--upgrade", *REQUIREMENTS])

    print("Done. Installed:", ", ".join(REQUIREMENTS))


if __name__ == "__main__":
    main()
