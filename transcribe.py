import os
from sarvamai import SarvamAI

api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    raise SystemExit("SARVAM_API_KEY not set in environment. Add it to .env and re-run.")

client = SarvamAI(api_subscription_key=api_key)

# Default sample text; overridable via CLI arg for convenience.
AUDIO_PATH = os.environ.get("SARVAM_AUDIO_PATH", "sample.wav")

try:
    with open(AUDIO_PATH, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v4",
            mode="transcribe",
        )
    print("Transcript:", response.transcript)
except FileNotFoundError:
    print(f"Error: audio file not found at {AUDIO_PATH}")
    print("Place an audio file (e.g. sample.wav) in this directory, or set SARVAM_AUDIO_PATH.")
except Exception as e:
    print(f"Error: {e}")
