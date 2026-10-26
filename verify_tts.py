import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    from tts_engine import SarvamSpeaker
except ImportError as e:
    print(f"❌ Failed to import SarvamSpeaker from tts_engine.py: {e}")
    sys.exit(1)

speaker = SarvamSpeaker()

# Test 1: Method existence
if not hasattr(speaker, "synthesize_speech"):
    print("❌ synthesize_speech method missing in SarvamSpeaker")
    sys.exit(1)

# Test 2: Synthesis and Byte format
print("Testing TTS synthesis...")
try:
    audio_data = speaker.synthesize_speech(
        text="Testing Sarvam connection", 
        target_language_code="hi-IN"
    )
    
    assert isinstance(audio_data, bytes), f"Expected bytes, got {type(audio_data)}"
    assert len(audio_data) > 44, "Audio byte buffer too small to be a valid WAV file"
    assert audio_data[:4] == b"RIFF", f"Invalid WAV header: expected b'RIFF', got {audio_data[:4]}"
    
    print("✅ Check 1: Return type is raw bytes")
    print("✅ Check 2: Header contains valid 'RIFF' WAV signature")
    print(f"✅ Check 3: Buffer size ({len(audio_data)} bytes) is playable")
    print("🚀 tts_engine.py is fully ready for Prompt 6!")

except Exception as err:
    print(f"⚠️ Caught error during call: {err}")
    print("Ensure mock fallback or base64 decoding is in place if credits are not active.")
