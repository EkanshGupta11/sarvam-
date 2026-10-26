"""
test_tts_check.py — Standalone QA contract and sanity check for tts_engine.py.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

# 1. Import check
try:
    from tts_engine import SarvamSpeaker, generate_dummy_wav_bytes
except ImportError as err:
    print(f"[ERROR] Failed to import from tts_engine.py: {err}")
    sys.exit(1)


def test_signature_and_contract() -> None:
    """Verify class existence, method signature, and default parameter values."""
    print("Checking method signature and contract...")
    sig = inspect.signature(SarvamSpeaker.synthesize_speech)
    params = sig.parameters

    assert "text" in params, "Parameter 'text' missing in synthesize_speech"
    assert "target_language_code" in params, "Parameter 'target_language_code' missing"
    assert params["target_language_code"].default == "hi-IN", (
        f"Expected default 'hi-IN', got {params['target_language_code'].default}"
    )
    assert "speaker" in params, "Parameter 'speaker' missing"
    assert params["speaker"].default == "shubh", (
        f"Expected default 'shubh', got {params['speaker'].default}"
    )
    print("  [OK] Method contract & default parameters verified.")


def test_normalization_and_dummy() -> None:
    """Test speaker lowercasing and ISO language code normalization."""
    print("Checking language and speaker normalization...")
    assert SarvamSpeaker._resolve_language("hi") == "hi-IN"
    assert SarvamSpeaker._resolve_language("ta") == "ta-IN"
    assert SarvamSpeaker._resolve_language("en") == "en-IN"
    assert SarvamSpeaker._resolve_language("HI-IN") == "hi-IN"
    assert SarvamSpeaker._resolve_language("unknown-xyz") == "en-IN"

    # Speaker lowercasing
    assert SarvamSpeaker._resolve_speaker("SHUBH", "hi-IN") == "shubh"
    assert SarvamSpeaker._resolve_speaker("Aditya", "hi-IN") == "aditya"
    assert SarvamSpeaker._resolve_speaker("meera", "hi-IN") == "shubh"  # aliased
    print("  [OK] Speaker & ISO language normalization verified.")

    # Dummy wave generation check
    dummy_wav = generate_dummy_wav_bytes(1.0, 16000)
    assert isinstance(dummy_wav, bytes), f"Expected bytes, got {type(dummy_wav)}"
    assert len(dummy_wav) > 44, "Dummy WAV buffer too small"
    assert dummy_wav[:4] == b"RIFF", f"Expected b'RIFF', got {dummy_wav[:4]}"
    print("  [OK] Minimal dummy WAV generator verified.")


def test_zero_credit_fallback() -> None:
    """Test zero-credit / offline mode when no API key or invalid API key is provided."""
    print("Checking zero-credit / offline fallback...")
    mock_speaker = SarvamSpeaker(api_key="")
    assert mock_speaker._client is None, "Client should be None when API key is empty"

    mock_result = mock_speaker.synthesize_speech("Fallback test text", target_language_code="hi")
    assert isinstance(mock_result, bytes), f"Expected bytes, got {type(mock_result)}"
    assert len(mock_result) > 44, "Fallback buffer too small"
    assert mock_result[:4] == b"RIFF", f"Invalid header: {mock_result[:4]}"
    print("  [OK] Zero-credit mock mode safely returns valid RIFF WAV bytes without raising exceptions.")


def test_synthesis_live_or_fallback() -> None:
    """Test default speaker instance with actual synthesis or automatic fallback."""
    print("Testing SarvamSpeaker.synthesize_speech execution...")
    speaker = SarvamSpeaker()
    result = speaker.synthesize_speech(
        text="Namaste, this is an integration sanity test.",
        target_language_code="hi-IN",
        speaker="shubh",
    )

    assert isinstance(result, bytes), f"Expected bytes, got {type(result)}"
    assert len(result) > 44, f"Result buffer ({len(result)} bytes) too small to be valid audio"
    assert result[:4] == b"RIFF", f"Invalid WAV header: expected b'RIFF', got {result[:4]}"
    print(f"  [OK] Returned audio byte buffer length: {len(result):,} bytes")
    print("  [OK] Header verified: starts with b'RIFF'")


def main() -> None:
    test_signature_and_contract()
    test_normalization_and_dummy()
    test_zero_credit_fallback()
    test_synthesis_live_or_fallback()

    print("\n" + "=" * 55)
    print("[PASS] SarvamSpeaker contract and audio payload verified.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
