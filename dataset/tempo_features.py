import numpy as np
import librosa


def extract_scalar_bpm(
    filename,
    target_sr=8000,
    hop_length=512,
):
    """
    Extract one global BPM scalar from an audio file using librosa's
    dynamic-programming beat tracker.

    Returns:
        bpm: float
    """
    y, sr = librosa.load(
        filename,
        sr=target_sr,
        mono=True,
    )

    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop_length,
    )

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
    )

    # librosa may return a scalar or array depending on version
    tempo = float(np.asarray(tempo).reshape(-1)[0])

    if np.isnan(tempo) or tempo <= 0:
        tempo = 0.0

    return tempo


def normalize_bpm(
    bpm,
    min_bpm=60.0,
    max_bpm=180.0,
):
    """
    Normalize BPM to [-1, 1].

    60 BPM  -> -1
    120 BPM ->  0
    180 BPM -> +1
    """
    bpm = np.clip(
        bpm,
        min_bpm,
        max_bpm,
    )

    bpm_norm = (
        2.0
        * ((bpm - min_bpm) / (max_bpm - min_bpm))
        - 1.0
    )

    return float(bpm_norm)


def make_tempo_condition(
    bpm,
    min_bpm=60.0,
    max_bpm=180.0,
):
    """
    Return a 1D tempo condition vector.

    Shape before batching:
        [1]

    Example:
        90 BPM -> [-0.5]
    """
    bpm_norm = normalize_bpm(
        bpm,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    return np.array(
        [bpm_norm],
        dtype=np.float32,
    )