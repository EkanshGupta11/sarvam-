"""
tts_engine.py — Sarvam TTS wrapper for voice-to-voice assistant.

Exposes SarvamSpeaker, a thin class over the Sarvam `text_to_speech.convert`
endpoint.  It handles:
  • Method contract: synthesize_speech(text, target_language_code, speaker) -> bytes
  • Base64 decoding of the Sarvam response audios payload.
  • Language-code normalisation / alias mapping (ISO codes -> BCP-47).
  • Lowercasing and validation of speaker names for bulbul:v3.
  • Zero-credit mock / exception fallback returning a valid minimal WAV stream.
  • Optional WAV save-to-disk and temporary file creation.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import struct
import tempfile
import wave
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

try:
    from sarvamai import SarvamAI
except ImportError:
    SarvamAI = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported language codes (as accepted by the Sarvam TTS API)
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGE_CODES: frozenset[str] = frozenset(
    {
        "bn-IN",  # Bengali
        "en-IN",  # English (Indian)
        "gu-IN",  # Gujarati
        "hi-IN",  # Hindi
        "kn-IN",  # Kannada
        "ml-IN",  # Malayalam
        "mr-IN",  # Marathi
        "od-IN",  # Odia
        "pa-IN",  # Punjabi
        "ta-IN",  # Tamil
        "te-IN",  # Telugu
    }
)

# ---------------------------------------------------------------------------
# Alias map: ISO codes & common names -> canonical Sarvam BCP-47 code
# ---------------------------------------------------------------------------
LANGUAGE_ALIASES: dict[str, str] = {
    # Hindi
    "hi":        "hi-IN",
    "hindi":     "hi-IN",
    "hi-in":     "hi-IN",
    # Tamil
    "ta":        "ta-IN",
    "tamil":     "ta-IN",
    "ta-in":     "ta-IN",
    # Bengali
    "bn":        "bn-IN",
    "bengali":   "bn-IN",
    "bangla":    "bn-IN",
    "bn-in":     "bn-IN",
    # English (Indian)
    "en":        "en-IN",
    "english":   "en-IN",
    "en-in":     "en-IN",
    "en-us":     "en-IN",
    "en-gb":     "en-IN",
    # Gujarati
    "gu":        "gu-IN",
    "gujarati":  "gu-IN",
    "gu-in":     "gu-IN",
    # Kannada
    "kn":        "kn-IN",
    "kannada":   "kn-IN",
    "kn-in":     "kn-IN",
    # Malayalam
    "ml":        "ml-IN",
    "malayalam": "ml-IN",
    "ml-in":     "ml-IN",
    # Marathi
    "mr":        "mr-IN",
    "marathi":   "mr-IN",
    "mr-in":     "mr-IN",
    # Odia
    "od":        "od-IN",
    "odia":      "od-IN",
    "oriya":     "od-IN",
    "od-in":     "od-IN",
    # Punjabi
    "pa":        "pa-IN",
    "punjabi":   "pa-IN",
    "pa-in":     "pa-IN",
    # Telugu
    "te":        "te-IN",
    "telugu":    "te-IN",
    "te-in":     "te-IN",
}

# Per-language default speakers — valid bulbul:v3 generic names
DEFAULT_SPEAKER_BY_LANGUAGE: dict[str, str] = {
    "hi-IN": "shubh",
    "ta-IN": "gokul",
    "bn-IN": "rohan",
    "en-IN": "simran",
    "gu-IN": "pooja",
    "kn-IN": "kavitha",
    "ml-IN": "priya",
    "mr-IN": "ishita",
    "od-IN": "neha",
    "pa-IN": "anand",
    "te-IN": "kavitha",
}

# Generic speakers valid for bulbul:v3 (lowercase short names only)
_GENERIC_SPEAKERS: frozenset[str] = frozenset(
    {
        "aditya", "ritu", "ashutosh", "priya", "neha", "rahul",
        "pooja", "rohan", "simran", "kavya", "amit", "dev", "ishita",
        "shreya", "ratan", "varun", "manan", "sumit", "roopa", "kabir",
        "aayan", "shubh", "advait", "anand", "tanya", "tarun", "sunny",
        "mani", "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha",
        "rehan", "soham", "rupali",
    }
)

# Known legacy aliases mapped to valid bulbul:v3 voices
SPEAKER_ALIASES: dict[str, str] = {
    "meera": "shubh",
    "anushka": "ritu",
}

# Fallback constants
_FALLBACK_LANGUAGE = "en-IN"
_FALLBACK_SPEAKER  = "shubh"


# ---------------------------------------------------------------------------
# Dummy audio generator (for zero-credit / offline / failure mode)
# ---------------------------------------------------------------------------
def generate_dummy_wav_bytes(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Generate a valid minimal 16kHz mono 16-bit PCM WAV byte stream.

    Used when SARVAM_API_KEY is missing or the API returns quota/auth errors,
    ensuring downstream tests and UI components do not crash.
    """
    n_frames = max(1, int(sample_rate * duration_sec))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n_frames):
            sample = int(400 * (i % 36 - 18) / 18)  # Gentle 440Hz test tone
            w.writeframes(struct.pack("<h", sample))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helper: decode the audio payload returned by the Sarvam SDK
# ---------------------------------------------------------------------------
def _extract_audio_bytes(response) -> bytes:
    """
    Extract and base64-decode the audio bytes from the Sarvam TTS API response.

    Sarvam returns an `audios` array containing base64-encoded audio chunks.
    Per API spec:
        audio_b64 = response.audios[0]
        return base64.b64decode(audio_b64)
    """
    if hasattr(response, "audios") and response.audios:
        if len(response.audios) == 1:
            return base64.b64decode(response.audios[0])
        decoded = [base64.b64decode(c) for c in response.audios]
        return _merge_wav_chunks(decoded)

    if hasattr(response, "audio") and response.audio:
        return base64.b64decode(response.audio)

    raise ValueError(
        f"Unexpected TTS response shape -- no 'audios' or 'audio' field found. "
        f"Response type: {type(response)}"
    )


def _merge_wav_chunks(chunks: list[bytes]) -> bytes:
    """Merge several WAV byte strings into one by concatenating their PCM frames."""
    frames_list: list[bytes] = []
    params = None

    for raw in chunks:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            if params is None:
                params = wf.getparams()
            frames_list.append(wf.readframes(wf.getnframes()))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        for frames in frames_list:
            out.writeframes(frames)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class SarvamSpeaker:
    """
    High-level wrapper around Sarvam's Text-to-Speech API.

    Parameters
    ----------
    api_key : str, optional
        Sarvam API key. If omitted, read from ``SARVAM_API_KEY`` env var.
    model : str
        Sarvam TTS model name. Defaults to ``"bulbul:v3"``.
    """

    DEFAULT_MODEL = "bulbul:v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        load_dotenv()
        if api_key is not None:
            resolved_key = api_key.strip()
        else:
            resolved_key = (os.environ.get("SARVAM_API_KEY") or "").strip()

        self.model = model
        self._client: Optional[SarvamAI] = None

        if not resolved_key:
            logger.warning(
                "SARVAM_API_KEY is not set. Operating in zero-credit mock fallback mode."
            )
        elif SarvamAI is None:
            logger.warning("sarvamai SDK not installed. Operating in mock fallback mode.")
        else:
            try:
                self._client = SarvamAI(api_subscription_key=resolved_key)
                logger.info("SarvamSpeaker initialised with live client (model=%s)", self.model)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize SarvamAI client (%s). Operating in mock fallback mode.",
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_language(raw: str) -> str:
        """Normalise an arbitrary language string or ISO code to a Sarvam BCP-47 code."""
        if not raw:
            return _FALLBACK_LANGUAGE
        normalised = raw.strip().lower()
        if normalised in LANGUAGE_ALIASES:
            return LANGUAGE_ALIASES[normalised]
        # Check canonical match (e.g. "hi-in" -> "hi-IN")
        upper_match = normalised.replace("_", "-")
        for supported in SUPPORTED_LANGUAGE_CODES:
            if supported.lower() == upper_match:
                return supported

        logger.warning(
            "Unsupported language code %r -- falling back to %s",
            raw,
            _FALLBACK_LANGUAGE,
        )
        return _FALLBACK_LANGUAGE

    @staticmethod
    def _resolve_speaker(speaker: str, language_code: str) -> str:
        """
        Return a valid, lowercased speaker for (speaker, language_code).
        Ensures lowercase formatting required by bulbul:v3.
        """
        spkr = (speaker or "").strip().lower()

        # Handle known aliases (e.g. meera -> shubh)
        if spkr in SPEAKER_ALIASES:
            spkr = SPEAKER_ALIASES[spkr]

        # Valid generic bulbul:v3 speaker?
        if spkr in _GENERIC_SPEAKERS:
            return spkr

        # Fall back to per-language default
        default = DEFAULT_SPEAKER_BY_LANGUAGE.get(language_code, _FALLBACK_SPEAKER)
        logger.warning(
            "Speaker %r may not be recognized by bulbul:v3; substituting default %r.",
            speaker,
            default,
        )
        return default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_speech(
        self,
        text: str,
        target_language_code: str = "hi-IN",
        speaker: str = "shubh",
        *,
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.5,
        speech_sample_rate: int = 22050,
        enable_preprocessing: bool = True,
        save_to: Optional[str | Path] = None,
    ) -> bytes:
        """
        Synthesise *text* to speech and return raw WAV audio bytes.

        Parameters
        ----------
        text : str
            The text to synthesise.
        target_language_code : str
            BCP-47-style language tag (e.g. ``"hi-IN"``, ``"ta-IN"``) or standard
            ISO code (e.g. ``"hi"``, ``"ta"``, ``"en"``).
        speaker : str
            A Sarvam speaker name (lowercased, e.g. ``"shubh"``, ``"aditya"``, ``"ritu"``).
            Defaults to ``"shubh"``.
        pitch : float
            Pitch adjustment (semitones). 0.0 = neutral.
        pace : float
            Speaking rate multiplier. 1.0 = normal speed.
        loudness : float
            Loudness multiplier. 1.5 = slightly amplified.
        speech_sample_rate : int
            Output sample rate in Hz (8000, 16000, 22050, etc.).
        enable_preprocessing : bool
            Allow Sarvam to normalise numbers, abbreviations, etc.
        save_to : str | Path | None
            Optional file path to write the WAV bytes to disk.

        Returns
        -------
        bytes
            Raw WAV audio bytes (always starts with b'RIFF').
        """
        # Empty text check -> produce short dummy WAV
        if not text or not text.strip():
            logger.warning("Empty text passed to synthesize_speech. Returning dummy audio.")
            audio_bytes = generate_dummy_wav_bytes(duration_sec=1.0, sample_rate=speech_sample_rate)
            if save_to is not None:
                out_path = Path(save_to)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(audio_bytes)
            return audio_bytes

        lang = self._resolve_language(target_language_code)
        spkr = self._resolve_speaker(speaker, lang)

        # Fallback mode when client is not initialized (no key / offline)
        if self._client is None:
            logger.info("Zero-credit/Mock mode: returning dummy 16kHz WAV bytes.")
            audio_bytes = generate_dummy_wav_bytes(duration_sec=1.0, sample_rate=speech_sample_rate)
            if save_to is not None:
                out_path = Path(save_to)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(audio_bytes)
            return audio_bytes

        logger.info(
            "TTS request | lang=%s  speaker=%s  model=%s  chars=%d",
            lang, spkr, self.model, len(text),
        )

        try:
            response = self._client.text_to_speech.convert(
                text=text,
                language_code=lang,
                speaker=spkr,
                pitch=pitch,
                pace=pace,
                loudness=loudness,
                speech_sample_rate=speech_sample_rate,
                enable_preprocessing=enable_preprocessing,
                model=self.model,
                output_audio_codec="wav",
            )
            audio_bytes = _extract_audio_bytes(response)
        except Exception as exc:
            # Check if we should retry with fallback language first
            if lang != _FALLBACK_LANGUAGE:
                logger.warning(
                    "TTS call failed for lang=%s (%s: %s). Retrying with fallback %s / speaker %s.",
                    lang, type(exc).__name__, exc, _FALLBACK_LANGUAGE, _FALLBACK_SPEAKER,
                )
                try:
                    response = self._client.text_to_speech.convert(
                        text=text,
                        language_code=_FALLBACK_LANGUAGE,
                        speaker=_FALLBACK_SPEAKER,
                        pitch=pitch,
                        pace=pace,
                        loudness=loudness,
                        speech_sample_rate=speech_sample_rate,
                        enable_preprocessing=enable_preprocessing,
                        model=self.model,
                        output_audio_codec="wav",
                    )
                    audio_bytes = _extract_audio_bytes(response)
                except Exception as fallback_exc:
                    logger.error(
                        "Fallback TTS request also failed (%s: %s). Engaging zero-credit dummy audio stream.",
                        type(fallback_exc).__name__, fallback_exc,
                    )
                    audio_bytes = generate_dummy_wav_bytes(duration_sec=1.0, sample_rate=speech_sample_rate)
            else:
                logger.error(
                    "Sarvam TTS call failed (%s: %s). Returning zero-credit dummy audio stream.",
                    type(exc).__name__, exc,
                )
                audio_bytes = generate_dummy_wav_bytes(duration_sec=1.0, sample_rate=speech_sample_rate)

        if save_to is not None:
            out_path = Path(save_to)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_bytes)
            logger.info("Audio saved -> %s  (%d bytes)", out_path, len(audio_bytes))

        return audio_bytes

    def synthesize_to_temp_file(
        self,
        text: str,
        target_language_code: str = "hi-IN",
        speaker: str = "shubh",
        **kwargs,
    ) -> Path:
        """
        Convenience wrapper: synthesise and save to a unique temporary file.

        Returns the ``Path`` to the written ``.wav`` file.
        """
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, prefix="sarvam_tts_"
        ) as tmp:
            tmp_path = Path(tmp.name)

        self.synthesize_speech(
            text=text,
            target_language_code=target_language_code,
            speaker=speaker,
            save_to=tmp_path,
            **kwargs,
        )
        return tmp_path


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    OUTPUT_WAV = Path("./test_output.wav")

    GREETINGS: list[tuple[str, str, str, str]] = [
        (
            "Hindi greeting",
            "Namaste! Main Sarvam AI hoon. Aapase milkar bahut khushi hui.",
            "hi-IN",
            "shubh",
        ),
        (
            "Tamil greeting",
            "Vanakkam! Naan Sarvam AI. Ungalai sandippathil makizhchi.",
            "ta-IN",
            "gokul",
        ),
    ]

    speaker_engine = SarvamSpeaker()
    combined_frames: list[bytes] = []
    wav_params = None

    sep = "-" * 55
    print(f"\n{sep}")
    print(" Sarvam TTS smoke test  --  bulbul:v3")
    print(f"{sep}\n")

    for description, text, lang, spkr in GREETINGS:
        print(f"  [{description}]")
        print(f"   lang    : {lang}")
        print(f"   speaker : {spkr}")
        print(f"   text    : {text[:70]}...")

        raw = speaker_engine.synthesize_speech(
            text=text,
            target_language_code=lang,
            speaker=spkr,
        )
        print(f"   bytes   : {len(raw):,}\n")

        with wave.open(io.BytesIO(raw), "rb") as wf:
            if wav_params is None:
                wav_params = wf.getparams()
            combined_frames.append(wf.readframes(wf.getnframes()))

    if wav_params and combined_frames:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out_wf:
            out_wf.setparams(wav_params)
            for frames in combined_frames:
                out_wf.writeframes(frames)
        OUTPUT_WAV.write_bytes(buf.getvalue())
        print(f"  Combined greeting saved -> {OUTPUT_WAV.resolve()}")
        print(f"  Total size             : {OUTPUT_WAV.stat().st_size:,} bytes\n")
    else:
        print("  No audio produced -- check errors above.", file=sys.stderr)
        sys.exit(1)
