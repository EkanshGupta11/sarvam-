"""
generate_pitch_assets.py
========================
Pre-generates all demo assets for the Sarvam ConvoMatrix pitch presentation.

Run once before going on stage:
    python generate_pitch_assets.py

Outputs (under ./demo_assets/):
    scenario_a.wav               — Bilingual Tech Standup audio
    scenario_a_transcript.json   — Hardcoded diarized transcript (offline fallback)
    scenario_a_minutes.json      — Pre-computed meeting minutes
    scenario_b.wav               — Customer Support Escalation audio
    scenario_b_transcript.json   — Hardcoded diarized transcript (offline fallback)
    scenario_b_minutes.json      — Pre-computed meeting minutes

Both JSON files are loaded directly by app.py in Demo Mode so the presentation
works even if the internet drops completely on stage.
"""
from __future__ import annotations

import io
import json
import struct
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Each line: (speaker_label, text, language_code, sarvam_speaker, duration_hint)
# duration_hint is used to synthesise realistic timestamps.

SCENARIO_A_LINES: list[tuple[str, str, str, str]] = [
    ("Speaker 0", "Okay team, let's do a quick standup. Aaj kya plan hai?",        "hi-IN", "ritu"),
    ("Speaker 1", "Main kal wali feature branch merge kar raha hoon — PR is ready.","hi-IN", "shubh"),
    ("Speaker 0", "Nice! Any blockers on the auth module?",                          "en-IN", "ritu"),
    ("Speaker 1", "Ek issue hai — the JWT expiry logic needs review. Thoda time lagega.", "hi-IN", "shubh"),
    ("Speaker 0", "Okay, let's sync at 3 PM. I'll loop in the backend team.",        "en-IN", "ritu"),
    ("Speaker 1", "Sure. Aur deployment pipeline ka kya? Still failing on staging?", "hi-IN", "shubh"),
    ("Speaker 0", "Haan, DevOps ne fix push kiya hai — should be green by EOD.",     "hi-IN", "ritu"),
    ("Speaker 1", "Perfect. Main testing start kar deta hoon post-lunch.",            "hi-IN", "shubh"),
]

SCENARIO_B_LINES: list[tuple[str, str, str, str]] = [
    ("Speaker 0", "Hello sir, aapka complaint number kya hai?",                      "hi-IN", "priya"),
    ("Speaker 1", "Naan ticket number TK-4821 solluven. Three days aachu, no response.", "ta-IN", "gokul"),
    ("Speaker 0", "Ji, TK-4821 dekh raha hoon. Yeh escalation queue mein hai.",     "hi-IN", "priya"),
    ("Speaker 1", "Enna da idu! Moonu naal aayiduchu. Manager kitta pesanum.",       "ta-IN", "gokul"),
    ("Speaker 0", "Bilkul, main abhi manager ko transfer karti hoon. Hold kijiye.", "hi-IN", "priya"),
    ("Speaker 1", "Okay, but please make sure they call me back within the hour.",   "en-IN", "gokul"),
    ("Speaker 0", "Confirmed. Ek ghante mein callback guarantee hai. Sorry for the inconvenience.", "hi-IN", "priya"),
]

SCENARIO_A_TRANSCRIPT: list[dict] = []
SCENARIO_B_TRANSCRIPT: list[dict] = []

SCENARIO_A_MINUTES: dict = {
    "executive_summary": (
        "The team held a quick standup covering the status of the feature branch PR, "
        "a JWT expiry blocker in the auth module, a 3 PM sync meeting, and a "
        "staging pipeline fix expected to go green by EOD."
    ),
    "speaker_key_points": {
        "Speaker 0": [
            "Requested standup and asked about the day's plan",
            "Enquired about blockers on the auth module",
            "Scheduled a 3 PM sync with the backend team",
            "Confirmed DevOps has pushed a fix for the staging pipeline (green by EOD)",
        ],
        "Speaker 1": [
            "PR for the feature branch is ready to merge",
            "JWT expiry logic in auth module needs review — will take more time",
            "Will start testing post-lunch once pipeline is green",
        ],
    },
    "decisions": [
        {
            "decision": "3 PM sync meeting to review JWT expiry logic",
            "by": "Speaker 0",
            "context": "Backend team to be looped in",
        },
        {
            "decision": "Deploy pipeline fix expected to be green by EOD",
            "by": "Speaker 0",
            "context": "DevOps has already pushed the fix",
        },
    ],
    "action_items": [
        {"action": "Merge feature branch PR",           "owner": "Speaker 1", "deadline": "Today"},
        {"action": "Review JWT expiry logic",           "owner": "Speaker 0", "deadline": "3 PM today"},
        {"action": "Confirm staging pipeline is green", "owner": "Speaker 0", "deadline": "EOD"},
        {"action": "Start post-pipeline testing",       "owner": "Speaker 1", "deadline": "Post-lunch"},
    ],
}

SCENARIO_B_MINUTES: dict = {
    "executive_summary": (
        "A customer (Speaker 1) contacted support regarding an unresolved complaint "
        "ticket TK-4821, open for three days. The support agent (Speaker 0) acknowledged "
        "the delay, confirmed the ticket is in the escalation queue, and committed to a "
        "manager callback within one hour."
    ),
    "speaker_key_points": {
        "Speaker 0": [
            "Identified ticket TK-4821 in the escalation queue",
            "Apologised for the 3-day delay",
            "Committed to transferring the call to a manager",
            "Guaranteed a callback within one hour",
        ],
        "Speaker 1": [
            "Raised complaint TK-4821 with no response for three days",
            "Demanded to speak with a manager",
            "Requested a callback within the hour",
        ],
    },
    "decisions": [
        {
            "decision": "Escalate ticket TK-4821 to manager",
            "by": "Speaker 0",
            "context": "Customer has been waiting three days without a response",
        },
    ],
    "action_items": [
        {"action": "Manager callback to customer",          "owner": "Speaker 0", "deadline": "Within 1 hour"},
        {"action": "Resolve ticket TK-4821",                "owner": "Speaker 0", "deadline": "TBD"},
        {"action": "Review escalation queue SLA adherence", "owner": "Speaker 0", "deadline": "TBD"},
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_timestamps(lines: list[tuple]) -> list[dict]:
    """Assign plausible start/end timestamps based on text word count."""
    turns = []
    cursor = 0.0
    for i, (speaker, text, _lang, _spkr) in enumerate(lines):
        # Approximate speaking rate: ~2.5 words/second
        words    = len(text.split())
        duration = max(2.0, words / 2.5)
        end      = cursor + duration
        turns.append({
            "speaker": speaker,
            "start":   round(cursor, 2),
            "end":     round(end,    2),
            "text":    text,
        })
        cursor = end + 0.4  # 400 ms pause between turns
    return turns


def _merge_wav_bytes(chunks: list[bytes]) -> bytes:
    """Concatenate a list of WAV byte buffers into one."""
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


def _silent_wav(duration_sec: float = 0.3, sample_rate: int = 22050) -> bytes:
    """Generate a short silent WAV segment (for speaker gaps)."""
    n_frames = int(sample_rate * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<h", 0) * n_frames)
    return buf.getvalue()


def _synthesise_scenario(
    lines: list[tuple[str, str, str, str]],
    speaker: "SarvamSpeaker",
    label: str,
) -> bytes:
    """Synthesise every line and concatenate into a single WAV."""
    chunks: list[bytes] = []
    total = len(lines)
    for idx, (spkr_label, text, lang, voice) in enumerate(lines, 1):
        print(f"  [{label}] line {idx}/{total} — {spkr_label} ({lang}): {text[:55]}...")
        try:
            wav = speaker.synthesize_speech(
                text=text,
                target_language_code=lang,
                speaker=voice,
            )
            chunks.append(wav)
            chunks.append(_silent_wav(0.4))   # inter-turn pause
        except Exception as exc:
            print(f"    WARNING: TTS failed for line {idx}: {exc}. Using silence.")
            # Fallback: 2-second silence so the WAV file is still valid
            chunks.append(_silent_wav(2.0))
    return _merge_wav_bytes([c for c in chunks if c])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Sarvam ConvoMatrix — Pitch Asset Generator ===\n")

    assets_dir = Path("demo_assets")
    assets_dir.mkdir(exist_ok=True)
    print(f"Output directory: {assets_dir.resolve()}\n")

    # Build hardcoded transcripts (timestamps only, no network needed)
    global SCENARIO_A_TRANSCRIPT, SCENARIO_B_TRANSCRIPT
    SCENARIO_A_TRANSCRIPT = _build_timestamps(SCENARIO_A_LINES)
    SCENARIO_B_TRANSCRIPT = _build_timestamps(SCENARIO_B_LINES)

    # Always write JSON fallbacks — these work 100% offline
    print("Writing transcript JSON files (offline-safe fallbacks)...")
    (assets_dir / "scenario_a_transcript.json").write_text(
        json.dumps(SCENARIO_A_TRANSCRIPT, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (assets_dir / "scenario_b_transcript.json").write_text(
        json.dumps(SCENARIO_B_TRANSCRIPT, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (assets_dir / "scenario_a_minutes.json").write_text(
        json.dumps(SCENARIO_A_MINUTES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (assets_dir / "scenario_b_minutes.json").write_text(
        json.dumps(SCENARIO_B_MINUTES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("  scenario_a_transcript.json  OK")
    print("  scenario_b_transcript.json  OK")
    print("  scenario_a_minutes.json     OK")
    print("  scenario_b_minutes.json     OK\n")

    # Attempt TTS audio generation (requires API key)
    try:
        from tts_engine import SarvamSpeaker
        spkr = SarvamSpeaker()

        if spkr._client is None:
            print(
                "WARNING: SARVAM_API_KEY not set or SarvamAI client unavailable.\n"
                "Audio WAV files will be generated as silence-only placeholders.\n"
            )

        print("Generating Scenario A WAV (Bilingual Tech Standup)...")
        wav_a = _synthesise_scenario(SCENARIO_A_LINES, spkr, "A")
        (assets_dir / "scenario_a.wav").write_bytes(wav_a)
        print(f"  scenario_a.wav  OK ({len(wav_a):,} bytes)\n")

        print("Generating Scenario B WAV (Customer Support Escalation)...")
        wav_b = _synthesise_scenario(SCENARIO_B_LINES, spkr, "B")
        (assets_dir / "scenario_b.wav").write_bytes(wav_b)
        print(f"  scenario_b.wav  OK ({len(wav_b):,} bytes)\n")

    except ImportError as exc:
        print(f"WARNING: Could not import tts_engine ({exc}). Skipping WAV generation.")
        print("The JSON fallbacks are still fully usable for offline demo mode.\n")

    print("=== Asset generation complete! ===")
    print(f"All files written to: {assets_dir.resolve()}\n")

    print("Files generated:")
    for f in sorted(assets_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:<40} {size:>10,} bytes")
    print()


if __name__ == "__main__":
    main()
