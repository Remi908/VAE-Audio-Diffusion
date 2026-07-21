import argparse
import json
import sys
from pathlib import Path

# Add repo root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from dataset.tempo_features import extract_scalar_bpm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dir",
        required=True,
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)

    files = []

    for ext in ["wav", "flac", "ogg", "aiff", "aif", "mp3"]:
        files.extend(audio_dir.rglob(f"*.{ext}"))

    files = sorted(files)

    for file in files:
        bpm = extract_scalar_bpm(
            str(file),
            target_sr=args.sr,
            hop_length=args.hop_length,
        )

        print(f"{file}: {bpm:.2f} BPM")


if __name__ == "__main__":
    main()
