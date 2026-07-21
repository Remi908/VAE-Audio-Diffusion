import argparse
import json
from pathlib import Path

from dataset.tempo_features import extract_scalar_bpm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Path to folder containing training audio files",
    )

    parser.add_argument(
        "--out",
        default="tempo_cache.json",
        help="Output JSON file path",
    )

    parser.add_argument(
        "--sr",
        type=int,
        default=8000,
        help="Target sample rate used for BPM extraction",
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

    print(f"Found {len(files)} audio files")

    if len(files) == 0:
        raise RuntimeError(f"No audio files found in {audio_dir}")

    cache = {}

    for file in files:
        try:
            bpm = extract_scalar_bpm(
                str(file),
                target_sr=args.sr,
                hop_length=args.hop_length,
            )

            cache[str(file)] = bpm
            print(f"{file}: {bpm:.2f} BPM")

        except Exception as e:
            print(f"FAILED: {file} | {type(e).__name__}: {e}")

    with open(args.out, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Saved tempo cache to {args.out}")


if __name__ == "__main__":
    main()
