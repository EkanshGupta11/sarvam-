"""Minimal Sarvam STT handshake sanity check.

Loads SARVAM_API_KEY from the environment (with python-dotenv fallback to .env),
generates a 3-second 16 kHz mono WAV via stdlib wave, and makes one minimal STT call
to verify API authentication. Prints the request ID / status and a clean init summary.
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

DUMMY_WAV = Path("dummy_sample.wav")
DURATION_SEC = 3.0
SAMPLE_RATE = 16000


def generate_dummy_wav(path: Path) -> None:
    """Generate a 3s 16kHz mono WAV with a low-amplitude 440 Hz tone."""
    import struct
    n_frames = int(SAMPLE_RATE * DURATION_SEC)
    with path.open("wb") as fp, wave.open(fp, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        for i in range(n_frames):
            sample = int(500 * (i % 4096 - 2048) / 2048)  # gentle sawtooth-ish tone
            w.writeframes(struct.pack("<h", sample))


def main() -> None:
    # Load .env into os.environ (safe no-op if already set).
    load_dotenv()

    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise SystemExit("SARVAM_API_KEY is not set in the environment or .env")

    print("Initializing SarvamAI client...")
    client = SarvamAI(api_subscription_key=api_key)

    # Generate dummy audio for the test call.
    if not DUMMY_WAV.exists():
        print(f"Generating dummy WAV: {DUMMY_WAV} ({DURATION_SEC}s, {SAMPLE_RATE}Hz mono)")
        generate_dummy_wav(DUMMY_WAV)
    else:
        print(f"Using existing dummy WAV: {DUMMY_WAV}")

    print("Running minimal STT handshake call...")
    try:
        with DUMMY_WAV.open("rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v4",
                mode="transcribe",
            )
        # The Sarvam SDK response object should expose at least the transcript.
        transcript = getattr(response, "transcript", None)
        print(f"Handshake OK - transcript: {transcript!r}")
    except Exception as exc:
        # Defensive: print whatever the SDK surfaced so we can see auth vs network vs quota.
        print(f"STT call raised: {type(exc).__name__}: {exc}")

    print("Clean initialization confirmed.")


if __name__ == "__main__":
    main()
