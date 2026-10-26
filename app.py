"""
Sarvam ConvoMatrix — Indic Multi-Speaker & Code-Switched Voice Intelligence
Streamlit demo interface.

Layout:
- Header + global API-key health banner
- Sidebar: audio input (upload / mic), output mode, speaker count, reset
            + Pitch Demo Mode toggle & scenario selector
- Main (left):  diarized transcript timeline with speaker badges/avatars
- Main (right): Q&A chat tab (with TTS playback) + auto minutes tab
"""
from __future__ import annotations

import html
import io
import json
import logging
import os
import struct
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — the UI must stay importable even if a dependency is missing.
# ---------------------------------------------------------------------------

_IMPORT_ERRORS: List[str] = []

try:
    from diarized_stt import SarvamDiarizedSTT
except Exception as _e:
    SarvamDiarizedSTT = None  # type: ignore[assignment, misc]
    _IMPORT_ERRORS.append(f"diarized_stt: {_e}")

try:
    from convo_brain import ConversationBrain
except Exception as _e:
    ConversationBrain = None  # type: ignore[assignment, misc]
    _IMPORT_ERRORS.append(f"convo_brain: {_e}")

try:
    from tts_engine import SarvamSpeaker
except Exception as _e:
    SarvamSpeaker = None  # type: ignore[assignment, misc]
    _IMPORT_ERRORS.append(f"tts_engine: {_e}")

# ---------------------------------------------------------------------------
# Page config  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sarvam ConvoMatrix",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global API-key health check
# ---------------------------------------------------------------------------

_SARVAM_KEY: str = os.environ.get("SARVAM_API_KEY", "").strip()

if not _SARVAM_KEY:
    st.error(
        "**SARVAM_API_KEY is not set.** "
        "Add it to your `.env` file or export it in the shell before running.\n\n"
        "STT / TTS / Chat features will run in **mock / degraded mode** until the key is provided.",
        icon="🔑",
    )

if _IMPORT_ERRORS:
    with st.expander("Module import warnings (expand for details)"):
        for _msg in _IMPORT_ERRORS:
            st.warning(_msg)

# ---------------------------------------------------------------------------
# Speaker theming
# ---------------------------------------------------------------------------

SPEAKER_THEME: Dict[str, Dict[str, str]] = {
    "Speaker 0": {
        "badge": "cyan",
        "avatar": "https://api.dicebear.com/9.x/initials/svg?seed=S0&backgroundColor=cyan&textColor=000",
        "color": "#00BCD4",
        "bg":    "#E0F7FA",
    },
    "Speaker 1": {
        "badge": "emerald",
        "avatar": "https://api.dicebear.com/9.x/initials/svg?seed=S1&backgroundColor=green&textColor=fff",
        "color": "#10B981",
        "bg":    "#D1FAE5",
    },
    "Speaker 2": {
        "badge": "amber",
        "avatar": "https://api.dicebear.com/9.x/initials/svg?seed=S2&backgroundColor=orange&textColor=000",
        "color": "#F59E0B",
        "bg":    "#FEF3C7",
    },
    "Speaker 3": {
        "badge": "violet",
        "avatar": "https://api.dicebear.com/9.x/initials/svg?seed=S3&backgroundColor=purple&textColor=fff",
        "color": "#8B5CF6",
        "bg":    "#EDE9FE",
    },
}

CODE_SWITCH_PATTERN = (
    r"\b(?:haan|nahi|thoda|zyada|karunga|karungi|kal|abhi|please|okay|fine|thanks|"
    r"sure|meeting|deck|slides|numbers|confirm|update|EOD|FY|projections|finalize|"
    r"agar|tabhi|lekin|kyunki|aur|toh|matlab|yaar|bhai|dost|bro|sis)\b"
)

# ---------------------------------------------------------------------------
# Demo-mode scenario registry
# ---------------------------------------------------------------------------

DEMO_ASSETS_DIR = Path(__file__).parent / "demo_assets"

DEMO_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Scenario A: Bilingual Tech Standup (Hinglish + English, 2 Speakers)": {
        "transcript_json": DEMO_ASSETS_DIR / "scenario_a_transcript.json",
        "minutes_json":    DEMO_ASSETS_DIR / "scenario_a_minutes.json",
        "wav":             DEMO_ASSETS_DIR / "scenario_a.wav",
        "description":     "Daily engineering standup mixing Hindi, Hinglish & English — feature PRs, JWT blockers, deployment pipeline.",
        "badge_color":     "#00BCD4",
        "icon":            "💻",
    },
    "Scenario B: Customer Support Escalation (Hindi + Tamil code-switching)": {
        "transcript_json": DEMO_ASSETS_DIR / "scenario_b_transcript.json",
        "minutes_json":    DEMO_ASSETS_DIR / "scenario_b_minutes.json",
        "wav":             DEMO_ASSETS_DIR / "scenario_b.wav",
        "description":     "Support agent (Hindi) handles escalation from Tamil-speaking customer — ticket TK-4821 unresolved for 3 days.",
        "badge_color":     "#8B5CF6",
        "icon":            "🎧",
    },
}

_SCENARIO_KEYS = list(DEMO_SCENARIOS.keys())

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

_MIN_WAV_BYTES = 45
_AUDIO_TYPES   = ["wav", "mp3", "m4a", "ogg", "webm", "flac", "aac"]


def save_temp_audio(audio_buffer: Any) -> str:
    """
    Accept any audio buffer — ``st.file_uploader`` result, ``st.audio_input``
    result, or raw ``bytes`` — validate it, and write it to a uniquely-named
    temporary ``.wav`` file on disk.

    Returns the path string of the saved file.
    """
    if isinstance(audio_buffer, (bytes, bytearray)):
        raw = bytes(audio_buffer)
    elif hasattr(audio_buffer, "read"):
        audio_buffer.seek(0)
        raw = audio_buffer.read()
    elif hasattr(audio_buffer, "getvalue"):
        raw = audio_buffer.getvalue()
    else:
        raise ValueError(f"Unsupported audio buffer type: {type(audio_buffer)}")

    if len(raw) < _MIN_WAV_BYTES:
        raise ValueError(
            f"Audio buffer is too small ({len(raw)} bytes). "
            "The file may be empty, truncated, or corrupt."
        )

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="convo_", delete=False,
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
    except OSError as exc:
        raise RuntimeError(f"Failed to write temporary audio file: {exc}") from exc

    logger.info("Saved audio buffer to temp file: %s (%d bytes)", tmp_path, len(raw))
    return tmp_path


# ---------------------------------------------------------------------------
# Demo-mode helpers
# ---------------------------------------------------------------------------

def _load_demo_scenario(key: str) -> Dict[str, Any]:
    """
    Load transcript + minutes from pre-generated JSON files.
    Returns ``{"turns": [...], "minutes": {...}}`` — works 100% offline.
    """
    meta = DEMO_SCENARIOS[key]

    turns: List[Dict[str, Any]] = []
    t_path: Path = meta["transcript_json"]
    if t_path.exists():
        try:
            turns = json.loads(t_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read %s: %s", t_path, exc)
    else:
        logger.warning("Transcript JSON not found: %s — run generate_pitch_assets.py", t_path)

    minutes: Dict[str, Any] = {}
    m_path: Path = meta["minutes_json"]
    if m_path.exists():
        try:
            minutes = json.loads(m_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read %s: %s", m_path, exc)

    return {"turns": turns, "minutes": minutes}


def _demo_wav_bytes(key: str) -> Optional[bytes]:
    """Return the pre-generated WAV bytes for the scenario, or None if missing."""
    wav_path: Path = DEMO_SCENARIOS[key]["wav"]
    if wav_path.exists():
        return wav_path.read_bytes()
    return None


def _load_demo_into_state(scenario_key: str) -> None:
    """Populate session_state from pre-generated assets — instant, no network."""
    data = _load_demo_scenario(scenario_key)
    turns = data["turns"]
    minutes = data["minutes"]

    st.session_state["turns"]                = turns
    st.session_state["current_transcript"]   = turns
    st.session_state["conversation_history"] = []
    st.session_state["demo_minutes"]         = minutes
    st.session_state["last_uploaded_name"]   = scenario_key
    st.session_state["chat_history"] = [
        {
            "role": "ai",
            "content": (
                f"**Demo scenario loaded:** {scenario_key.split(':')[0].strip()}\n\n"
                "Ask me anything about the conversation — e.g. *'What did Speaker 1 commit to?'* "
                "or *'What are the action items?'*"
            ),
        }
    ]
    if ConversationBrain is not None:
        client = _make_sarvam_client()
        st.session_state["brain"] = ConversationBrain(
            turns=turns,
            llm_client=client,
            llm_model="sarvam-105b-conversations",
        )
    else:
        st.session_state["brain"] = None


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _reset_session() -> None:
    for key in (
        "turns", "brain", "chat_history", "last_error", "audio_mode_radio",
        "output_mode", "num_speakers", "processing", "conversation_history",
        "current_transcript", "cached_audio", "last_uploaded_name",
        "minutes_demo", "tts_speaker", "demo_minutes", "demo_mode_active",
        "demo_scenario_key",
    ):
        st.session_state.pop(key, None)


def _ensure_turns() -> List[Dict[str, Any]]:
    if "turns" not in st.session_state:
        st.session_state["turns"] = []
    return st.session_state["turns"]


def _ensure_brain() -> Optional["ConversationBrain"]:
    if ConversationBrain is None:
        return None
    if "brain" not in st.session_state or st.session_state["brain"] is None:
        turns = _ensure_turns()
        try:
            client = _make_sarvam_client()
            st.session_state["brain"] = ConversationBrain(
                turns=turns, llm_client=client, llm_model="sarvam-105b-conversations",
            )
        except Exception as exc:
            logger.warning("ConversationBrain init failed: %s", exc)
            st.session_state["brain"] = ConversationBrain(turns=turns)
    return st.session_state["brain"]


def _ensure_chat() -> List[Dict[str, str]]:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "ai", "content": "Upload or record audio, then ask me anything about the conversation."},
        ]
    return st.session_state["chat_history"]


def _ensure_tts() -> Optional["SarvamSpeaker"]:
    if SarvamSpeaker is None:
        return None
    if "tts_engine" not in st.session_state:
        try:
            st.session_state["tts_engine"] = SarvamSpeaker()
        except Exception as exc:
            logger.warning("SarvamSpeaker init failed: %s", exc)
            st.session_state["tts_engine"] = None
    return st.session_state["tts_engine"]


def _make_sarvam_client() -> Any:
    if not _SARVAM_KEY:
        return None
    try:
        from sarvamai import SarvamAI
        return SarvamAI(api_subscription_key=_SARVAM_KEY)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _speaker_style(speaker: str) -> Dict[str, str]:
    return SPEAKER_THEME.get(speaker, SPEAKER_THEME["Speaker 0"])


def _code_switch_highlight(text: str) -> str:
    import re
    if not text:
        return html.escape(text or "")

    def _repl(match: Any) -> str:
        token = match.group(0)
        return f'<span class="code-switch">{html.escape(token)}</span>'

    pattern = re.compile(CODE_SWITCH_PATTERN, flags=re.IGNORECASE)
    return pattern.sub(_repl, html.escape(text))


def _render_turn_card(turn: Dict[str, Any]) -> None:
    speaker = turn.get("speaker", "Speaker ?")
    start   = float(turn.get("start", 0.0))
    end     = float(turn.get("end",   0.0))
    text    = turn.get("text", "") or ""

    theme = _speaker_style(speaker)
    ts = (
        f"{int(start//60):02d}:{int(round(start%60)):02d}"
        " - "
        f"{int(end//60):02d}:{int(round(end%60)):02d}"
    )

    st.markdown(
        f"""
        <div style="
            background: {theme['bg']};
            border-left: 6px solid {theme['color']};
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 10px;
        ">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <img src="{theme['avatar']}" width="34" height="34"
                     style="border-radius:50%;background:{theme['color']};padding:2px;" />
                <span style="
                    background: {theme['color']};
                    color: #fff;
                    padding: 2px 10px;
                    border-radius: 20px;
                    font-size: 0.8rem;
                    font-weight: 600;
                ">{html.escape(speaker)}</span>
                <span style="margin-left:auto;font-size:0.78rem;color:#666;">{ts}</span>
            </div>
            <div style="font-size:0.98rem;line-height:1.45;">{_code_switch_highlight(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# TTS helper
# ---------------------------------------------------------------------------

def _speak_response(text: str, language_code: str = "hi-IN", speaker: str = "shubh") -> None:
    tts = _ensure_tts()
    if tts is None:
        st.caption("_(TTS unavailable — sarvamai not installed)_")
        return
    if not text or not text.strip():
        return
    try:
        audio_bytes = tts.synthesize_speech(
            text=text, target_language_code=language_code, speaker=speaker,
        )
        if audio_bytes and len(audio_bytes) > 44:
            st.audio(audio_bytes, format="audio/wav", autoplay=False)
        else:
            st.caption("_(TTS returned an empty audio buffer)_")
    except Exception as exc:
        err_lower = str(exc).lower()
        if any(kw in err_lower for kw in ("401", "403", "unauthorized", "authentication", "quota", "402", "credit")):
            st.warning(
                "**TTS unavailable**: API key is invalid or credits are exhausted. "
                "Check your Sarvam dashboard.", icon="💳",
            )
        else:
            st.warning(f"TTS synthesis failed: {exc}", icon="🔊")


# ---------------------------------------------------------------------------
# Transcription trigger (live path)
# ---------------------------------------------------------------------------

def _run_transcription(audio_buffer: Any, label: str) -> None:
    if SarvamDiarizedSTT is None:
        st.error("**diarized_stt module failed to load.** Check the import warnings above.", icon="🚫")
        return
    if not _SARVAM_KEY:
        st.error("**Cannot transcribe — SARVAM_API_KEY is missing.**", icon="🔑")
        return

    with st.spinner(f"Saving audio buffer for '{label}'..."):
        try:
            tmp_path = save_temp_audio(audio_buffer)
        except (ValueError, RuntimeError) as exc:
            st.error(f"**Audio buffer error:** {exc}", icon="🎵")
            st.session_state["last_error"] = str(exc)
            return

    mode = str(st.session_state.get("output_mode", "codemix"))
    ns   = int(st.session_state.get("num_speakers", 2))

    with st.spinner("Transcribing with speaker diarization (Saaras v4)..."):
        try:
            stt       = SarvamDiarizedSTT()
            new_turns = stt.transcribe_multispeaker(
                audio_path=tmp_path, mode=mode, num_speakers=ns,
            )
        except RuntimeError as exc:
            err = str(exc)
            if any(kw in err.lower() for kw in ("401", "403", "unauthorized", "authentication")):
                st.error("**Authentication failed (STT).** Your SARVAM_API_KEY is invalid.", icon="🔑")
            elif any(kw in err.lower() for kw in ("quota", "402", "credit", "rate")):
                st.error("**Quota / credits exhausted (STT).** Upgrade your Sarvam plan.", icon="💳")
            else:
                st.error(f"**Transcription failed:** {exc}", icon="❌")
            st.session_state["last_error"] = str(exc)
            return
        except Exception as exc:
            st.error(f"**Unexpected transcription error:** {exc}", icon="❌")
            st.session_state["last_error"] = str(exc)
            return
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    st.session_state["turns"]                = new_turns
    st.session_state["current_transcript"]   = new_turns
    st.session_state["conversation_history"] = []
    st.session_state["chat_history"] = [
        {"role": "ai", "content": "Transcript loaded! Ask me anything about this conversation."}
    ]
    if ConversationBrain is not None:
        client = _make_sarvam_client()
        st.session_state["brain"] = ConversationBrain(
            turns=new_turns, llm_client=client, llm_model="sarvam-105b-conversations",
        )
    st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🎙️ Sarvam ConvoMatrix")
st.caption("Indic Multi-Speaker & Code-Switched Voice Intelligence")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Input & Config")

    # ── DEMO MODE TOGGLE ────────────────────────────────────────────────────
    st.markdown("---")
    demo_mode = st.toggle(
        "Pitch Demo Mode (Fast Pre-loaded Audio)",
        value=st.session_state.get("demo_mode_active", False),
        key="demo_mode_toggle",
        help=(
            "Instantly loads a pre-generated scenario with hardcoded transcripts "
            "and minutes. Works 100% offline — perfect for live presentations."
        ),
    )
    st.session_state["demo_mode_active"] = demo_mode

    if demo_mode:
        st.markdown(
            """
            <div style="background:#1a1a2e;border:1.5px solid #8B5CF6;border-radius:10px;
                        padding:10px 14px;margin:8px 0;">
                <span style="color:#8B5CF6;font-weight:700;font-size:0.9rem;">
                    DEMO MODE ACTIVE
                </span><br>
                <span style="color:#ccc;font-size:0.78rem;">
                    Pre-loaded scenario — no network calls for STT.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        scenario_key = st.selectbox(
            "Demo Scenario",
            options=_SCENARIO_KEYS,
            index=_SCENARIO_KEYS.index(
                st.session_state.get("demo_scenario_key", _SCENARIO_KEYS[0])
            ),
            key="demo_scenario_select",
        )

        # Show scenario description card
        s_meta = DEMO_SCENARIOS[scenario_key]
        st.markdown(
            f"""
            <div style="background:#111;border-left:4px solid {s_meta['badge_color']};
                        border-radius:8px;padding:9px 12px;margin:6px 0 10px 0;">
                <span style="font-size:1.1rem;">{s_meta['icon']}</span>
                <span style="color:#ddd;font-size:0.8rem;margin-left:6px;">
                    {html.escape(s_meta['description'])}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Asset availability indicators
        wav_ok  = s_meta["wav"].exists()
        json_ok = s_meta["transcript_json"].exists()
        st.caption(
            ("WAV audio: ready" if wav_ok else "WAV audio: not found — run generate_pitch_assets.py")
            + " · "
            + ("Transcript: ready" if json_ok else "Transcript JSON: missing")
        )

        if st.button(
            f"Run Demo Scenario",
            type="primary",
            use_container_width=True,
            key="run_demo_btn",
        ):
            if not json_ok:
                st.error(
                    "Transcript JSON not found. "
                    "Run `python generate_pitch_assets.py` first.",
                    icon="🚫",
                )
            else:
                st.session_state["demo_scenario_key"] = scenario_key
                _load_demo_into_state(scenario_key)
                st.rerun()

        st.markdown("---")

    else:
        # ── LIVE MODE INPUTS ─────────────────────────────────────────────────
        audio_mode = st.radio(
            "Audio Input",
            options=["Upload Audio File (WAV/MP3)", "Record via Microphone"],
            index=0,
            key="audio_mode_radio",
            help="Live mic uses st.audio_input (Streamlit >= 1.33).",
        )

        uploaded_file = None
        mic_audio     = None

        if audio_mode == "Upload Audio File (WAV/MP3)":
            uploaded_file = st.file_uploader(
                "Upload audio",
                type=_AUDIO_TYPES,
                help="WAV/MP3 recommended. Sarvam STT accepts a wide range of codecs.",
            )
        else:
            if hasattr(st, "audio_input"):
                mic_audio = st.audio_input(
                    "Record your message",
                    help="Click the mic icon to start/stop recording.",
                )
            else:
                st.info("Live mic requires Streamlit >= 1.33. Run: `pip install --upgrade streamlit`")

        st.divider()

        output_mode = st.selectbox(
            "Output Mode",
            options=["codemix", "transcribe", "translate"],
            index=0,
            key="output_mode",
            help=(
                "**codemix** — preserves native script + English code-switching\n"
                "**transcribe** — same-language transcript\n"
                "**translate** — direct English translation"
            ),
        )

        num_speakers = st.slider(
            "Expected speaker count (optional)",
            min_value=1, max_value=6, value=2, step=1,
            key="num_speakers",
            help="Hint for diarization; Sarvam can also auto-detect speakers.",
        )

    # ── TTS SETTINGS (always visible) ───────────────────────────────────────
    st.divider()
    st.markdown("**TTS Voice Settings**")
    tts_lang = st.selectbox(
        "Response language",
        options=["hi-IN", "en-IN", "ta-IN", "bn-IN", "mr-IN", "gu-IN", "kn-IN", "te-IN", "pa-IN"],
        index=0,
        key="tts_lang",
        help="Language for the spoken Q&A response.",
    )
    tts_speaker_opt = st.selectbox(
        "Voice / Speaker",
        options=["shubh", "ritu", "simran", "aditya", "priya", "gokul", "kavitha", "anand", "neha", "pooja"],
        index=0,
        key="tts_speaker",
        help="Sarvam bulbul:v3 speaker voice.",
    )

    st.divider()

    if st.button("Reset Session", use_container_width=True):
        _reset_session()
        st.success("Session reset.")
        st.rerun()

    st.divider()
    st.caption("Powered by **Sarvam AI** — Saaras v4 (STT) · Sarvam-105B (NLU) · Bulbul v3 (TTS)")


# ---------------------------------------------------------------------------
# Transcription trigger — live mode only
# ---------------------------------------------------------------------------

if not demo_mode:
    _audio_source  = uploaded_file or mic_audio
    _uploaded_name = getattr(uploaded_file, "name", None) or ("mic_recording" if mic_audio else None)
    _last_name     = st.session_state.get("last_uploaded_name")

    if _audio_source is not None:
        turns = _ensure_turns()
        if not turns or _uploaded_name != _last_name:
            st.session_state["last_uploaded_name"] = _uploaded_name
            _run_transcription(_audio_source, _uploaded_name or "audio")
        else:
            with st.sidebar:
                st.success("Transcript cached — no re-transcription needed.")
else:
    _audio_source  = None
    _uploaded_name = None


# ---------------------------------------------------------------------------
# Demo mode — pre-loaded WAV audio player banner
# ---------------------------------------------------------------------------

if demo_mode and st.session_state.get("demo_scenario_key"):
    active_key = st.session_state["demo_scenario_key"]
    wav_bytes  = _demo_wav_bytes(active_key)
    if wav_bytes:
        meta = DEMO_SCENARIOS[active_key]
        with st.expander(
            f"{meta['icon']} Pre-recorded demo audio — {active_key.split(':')[0].strip()}",
            expanded=False,
        ):
            st.audio(wav_bytes, format="audio/wav", autoplay=False)
            st.caption(
                "This is the pre-synthesised multi-speaker audio generated by "
                "`generate_pitch_assets.py`. The transcript below was loaded from "
                "the offline JSON fallback — no internet required."
            )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

left_col, right_col = st.columns([1.15, 1.0], gap="large")

# ── Left: diarized timeline ─────────────────────────────────────────────────
with left_col:
    st.subheader("Diarized Transcript Timeline")

    if demo_mode and st.session_state.get("demo_scenario_key"):
        active_key = st.session_state["demo_scenario_key"]
        s_meta = DEMO_SCENARIOS[active_key]
        st.markdown(
            f"""
            <div style="background:linear-gradient(90deg,{s_meta['badge_color']}22,transparent);
                        border-left:4px solid {s_meta['badge_color']};
                        border-radius:8px;padding:8px 14px;margin-bottom:12px;">
                <span style="font-size:0.85rem;color:{s_meta['badge_color']};font-weight:600;">
                    {s_meta['icon']} DEMO: {html.escape(active_key.split(':')[0].strip())}
                </span>
                <span style="font-size:0.78rem;color:#888;margin-left:8px;">
                    (offline JSON fallback)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    turns = _ensure_turns()

    if not turns:
        if demo_mode:
            st.info(
                "Select a scenario in the sidebar and click **Run Demo Scenario** to load the timeline instantly."
            )
        else:
            st.info("Upload an audio file (or record via the mic) in the sidebar to populate the timeline.")
    else:
        unique_speakers = {t.get("speaker") for t in turns}
        st.caption(f"{len(turns)} turn(s) · {len(unique_speakers)} speaker(s) detected")
        for turn in turns:
            _render_turn_card(turn)

# ── Right: Q&A + Minutes ─────────────────────────────────────────────────────
with right_col:
    brain = _ensure_brain()
    chat  = _ensure_chat()

    tab_qa, tab_minutes = st.tabs(["Ask the Conversation", "Auto Minutes & Action Items"])

    # ── Q&A tab ────────────────────────────────────────────────────────────
    with tab_qa:
        st.caption("Ask in Hindi, Hinglish, or English. Grounded in the transcript only.")

        for message in chat:
            with st.chat_message("user" if message["role"] == "user" else "assistant"):
                st.markdown(message["content"])
                if message["role"] == "ai" and message.get("audio_bytes"):
                    st.audio(message["audio_bytes"], format="audio/wav", autoplay=False)

        user_input = st.chat_input("e.g. Speaker 1 ne deadline kya boli?")

        if user_input:
            chat.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response       = ""

                if brain is None:
                    full_response = "ConversationBrain is unavailable. Check import errors at the top of the page."
                    message_placeholder.markdown(full_response)
                elif not _ensure_turns():
                    full_response = (
                        "No transcript loaded yet. "
                        + ("Click **Run Demo Scenario** in the sidebar." if demo_mode
                           else "Please upload an audio file and wait for transcription to complete.")
                    )
                    message_placeholder.markdown(full_response)
                else:
                    try:
                        with st.spinner("Thinking across the conversation..."):
                            answer = brain.query_conversation(user_input)

                        for chunk in answer.split():
                            full_response += chunk + " "
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.018)
                        message_placeholder.markdown(full_response.strip())

                    except Exception as exc:
                        err = str(exc)
                        if any(kw in err.lower() for kw in ("401", "403", "unauthorized", "authentication")):
                            full_response = "**Authentication error**: SARVAM_API_KEY is invalid or expired."
                            st.error(full_response, icon="🔑")
                        elif any(kw in err.lower() for kw in ("quota", "402", "credit", "rate")):
                            full_response = "**Credits exhausted**: Your Sarvam quota has been used up."
                            st.error(full_response, icon="💳")
                        else:
                            full_response = f"Could not answer: {exc}"
                            st.error(full_response, icon="⚠️")
                        message_placeholder.empty()

                # TTS playback
                tts_audio_bytes: Optional[bytes] = None
                if full_response.strip() and not full_response.startswith("**"):
                    _speak_response(
                        text=full_response.strip(),
                        language_code=st.session_state.get("tts_lang", "hi-IN"),
                        speaker=st.session_state.get("tts_speaker", "shubh"),
                    )
                    tts = _ensure_tts()
                    if tts is not None:
                        try:
                            tts_audio_bytes = tts.synthesize_speech(
                                text=full_response.strip(),
                                target_language_code=st.session_state.get("tts_lang", "hi-IN"),
                                speaker=st.session_state.get("tts_speaker", "shubh"),
                            )
                        except Exception:
                            tts_audio_bytes = None

            chat.append({
                "role": "ai",
                "content": full_response.strip(),
                "audio_bytes": tts_audio_bytes,
            })

    # ── Minutes tab ────────────────────────────────────────────────────────
    with tab_minutes:
        st.caption("Structured minutes with speaker-wise action items.")

        # In demo mode, serve pre-computed minutes instantly
        if demo_mode and st.session_state.get("demo_minutes"):
            st.success(
                "Pre-computed minutes loaded from offline JSON — instant, no network.",
                icon="⚡",
            )
            minutes: Optional[Dict[str, Any]] = st.session_state["demo_minutes"]
        else:
            col_run, col_note = st.columns([1, 2])
            with col_run:
                run_minutes = st.button("Generate Minutes", use_container_width=True)
            with col_note:
                st.caption("If no transcript exists yet, a demo dialogue will be used.")

            minutes = None

            if run_minutes or (not turns and "minutes_demo" not in st.session_state):
                with st.spinner("Generating executive summary and action items..."):
                    try:
                        _brain_for_minutes = brain
                        if not turns:
                            if ConversationBrain is not None:
                                _brain_for_minutes = ConversationBrain(
                                    turns=[
                                        {"speaker": "Speaker 0", "start": 0.0,  "end": 4.2,  "text": "Haan, main kal meeting join karunga, lekin mujhe thoda late hoga,"},
                                        {"speaker": "Speaker 0", "start": 4.2,  "end": 9.0,  "text": "kyunki train schedule change ho gaya."},
                                        {"speaker": "Speaker 1", "start": 9.0,  "end": 14.5, "text": "Okay, no problem. Please make sure the deck is ready by 9 AM."},
                                        {"speaker": "Speaker 1", "start": 14.5, "end": 18.3, "text": "Aur FY26 projections slip kar diye hain, hence we need those updated slides."},
                                        {"speaker": "Speaker 0", "start": 18.3, "end": 23.0, "text": "Haan, slides main update kar dunga, but FY26 numbers verify karne hain."},
                                        {"speaker": "Speaker 0", "start": 23.0, "end": 28.4, "text": "Mujhe finance se confirmation chahiye, tabhi finalize kar paunga."},
                                        {"speaker": "Speaker 1", "start": 28.4, "end": 33.0, "text": "Fine, then action items: you confirm numbers, I share the final agenda by EOD."},
                                    ],
                                    llm_client=_make_sarvam_client(),
                                    llm_model="sarvam-105b-conversations",
                                )
                                st.session_state["minutes_demo"] = True

                        if _brain_for_minutes is not None:
                            minutes = _brain_for_minutes.generate_action_minutes()
                        else:
                            st.error("ConversationBrain unavailable.", icon="🚫")
                    except Exception as exc:
                        err = str(exc)
                        if any(kw in err.lower() for kw in ("401", "403", "unauthorized", "authentication")):
                            st.error("**Authentication error**: Check your API key.", icon="🔑")
                        elif any(kw in err.lower() for kw in ("quota", "402", "credit")):
                            st.error("**Credits exhausted**: Upgrade your Sarvam plan.", icon="💳")
                        else:
                            st.error(f"Minutes generation failed: {exc}", icon="❌")
            elif not run_minutes:
                st.info("Press **Generate Minutes** above.")

        # Render minutes (shared renderer for both demo & live)
        if minutes:
            st.markdown("### Executive Summary")
            st.markdown(minutes.get("executive_summary") or "_(no summary returned)_")

            st.markdown("### Speaker-wise Key Points")
            speaker_points = minutes.get("speaker_key_points") or {}
            if speaker_points:
                for spkr, points in speaker_points.items():
                    theme = _speaker_style(spkr)
                    with st.container(border=True):
                        st.markdown(
                            f"<span style='color:{theme['color']};font-weight:600;'>"
                            f"{html.escape(spkr)}</span>",
                            unsafe_allow_html=True,
                        )
                        for point in points:
                            st.markdown(f"- {html.escape(str(point))}")
            else:
                st.info("No speaker-wise key points returned.")

            st.markdown("### Decisions")
            decisions = minutes.get("decisions") or []
            if decisions:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Decision": d.get("decision", ""),
                            "By":       d.get("by", d.get("owner", "") or ""),
                            "Context":  d.get("context", ""),
                        }
                        for d in decisions
                    ]),
                    hide_index=True, use_container_width=True,
                )
            else:
                st.info("No explicit decisions detected in the transcript.")

            st.markdown("### Action Items")
            actions = minutes.get("action_items") or []
            if actions:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Action":   a.get("action", ""),
                            "Owner":    a.get("owner", a.get("by", "") or ""),
                            "Deadline": a.get("deadline", "TBD"),
                        }
                        for a in actions
                    ]),
                    hide_index=True, use_container_width=True,
                )
            else:
                st.info("No action items detected in the transcript.")
