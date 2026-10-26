"""
Sarvam Diarized Speech-to-Text module.

Provides:
- SarvamDiarizedSTT: multi-speaker diarization via the Sarvam Batch STT Job API
  (saaras:v4 + with_diarization), with a fast single-utterance REST fallback.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.types import (
    DiarizedEntry,
    DiarizedTranscript,
    JobStatusResponse,
    SpeechToTextTranscriptionData,
)
from sarvamai.speech_to_text_job.job import SpeechToTextJob


def _load_api_key() -> str:
    load_dotenv()
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set in the environment or .env")
    return key


class SarvamDiarizedSTT:
    """Sarvam AI multi-speaker diarization wrapper.

    Uses the Batch Speech-to-Text Job API with ``saaras:v4`` and
    ``with_diarization=True`` for multi-speaker transcripts, plus a synchronous
    REST fallback for quick single-utterance testing.
    """

    SUPPORTED_MODES = {"codemix", "transcribe", "translate"}

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._client = SarvamAI(api_subscription_key=api_key or _load_api_key())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_multispeaker(
        self,
        audio_path: str,
        mode: str = "codemix",
        num_speakers: Optional[int] = None,
        *,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> List[Dict[str, Any]]:
        """Transcribe an audio file with speaker diarization.

        Parameters
        ----------
        audio_path:
            Path to the input audio file (on disk).
        mode:
            One of ``codemix``, ``transcribe``, or ``translate``.
        num_speakers:
            Optional hint for the expected number of speakers. Passed through to
            the Sarvam batch STT job API when the SDK supports it; otherwise
            ignored.
        poll_interval:
            Seconds between status polls while waiting for the job to complete.
        timeout:
            Maximum time to wait for the job to reach a terminal state.

        Returns
        -------
        List of turns, each with ``speaker``, ``start``, ``end``, and ``text``.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; supported: {self.SUPPORTED_MODES}")

        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        job = self._create_diarization_job(audio, mode, num_speakers)
        self._wait_until_complete(job, poll_interval=poll_interval, timeout=timeout)
        return self._extract_diarized_turns(job)

    def transcribe_quick(self, audio_path: str, mode: str = "codemix") -> str:
        """Synchronous single-utterance STT via the REST fallback endpoint.

        Best for short utterances (under ~30 seconds). For multi-speaker
        diarization, prefer :meth:`transcribe_multispeaker`.
        """
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; supported: {self.SUPPORTED_MODES}")

        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with audio.open("rb") as f:
            response = self._client.speech_to_text.transcribe(
                file=f,
                model="saaras:v4",
                mode=mode,
            )
        transcript = getattr(response, "transcript", "")
        if isinstance(transcript, str):
            return transcript
        # Defensive: some SDK versions may return a structured object.
        if hasattr(transcript, "get"):
            return transcript.get("transcript") or transcript.get("text") or str(transcript)
        return str(transcript)

    # ------------------------------------------------------------------
    # Batch job helpers
    # ------------------------------------------------------------------

    def _create_diarization_job(
        self,
        audio: Path,
        mode: str,
        num_speakers: Optional[int],
    ) -> SpeechToTextJob:
        """Create a Sarvam batch STT job with diarization enabled.

        Uses ``client.speech_to_text_job.create_job(...)`` which provisions the
        batch job and returns a ``SpeechToTextJob`` handle. The returned handle
        is then used to upload the audio file and start the job.
        """
        job = self._client.speech_to_text_job.create_job(
            model="saaras:v4",
            mode=mode,
            with_diarization=True,
            num_speakers=num_speakers,
        )
        if not job or not getattr(job, "job_id", None):
            raise RuntimeError("Failed to create Sarvam batch STT job (no job_id returned)")

        # Upload the local audio file into the job's storage, then start it.
        uploaded = job.upload_files(file_paths=[str(audio.resolve())])
        if not uploaded:
            raise RuntimeError("Failed to upload audio file to the Sarvam batch STT job")
        started = job.start()
        if not getattr(started, "job_id", None):
            raise RuntimeError("Failed to start the Sarvam batch STT job")
        return job

    def _wait_until_complete(
        self,
        job: SpeechToTextJob,
        *,
        poll_interval: float = 5.0,
        timeout: float = 600.0,
    ) -> None:
        """Poll ``SpeechToTextJob.wait_until_complete(...)`` until the job is done."""
        try:
            job.wait_until_complete(
                poll_interval=int(poll_interval),
                timeout=int(timeout),
            )
        except Exception as exc:
            # If the SDK raises on failure/timeout, re-raise with a clearer message.
            status = self._unwrap_status(job)
            if self._is_terminal_failed(status):
                msg = self._terminal_failure_message(status) or str(exc)
                raise RuntimeError(f"Sarvam batch STT job failed: {msg}") from exc
            raise

    def _unwrap_status(self, job: SpeechToTextJob) -> Any:
        """Best-effort fetch of the job's current status."""
        try:
            return job.get_status()
        except Exception:
            return None

    @staticmethod
    def _is_terminal_failed(status: Any) -> bool:
        if status is None:
            return False
        state = getattr(status, "job_state", None) or getattr(status, "state", None) or getattr(status, "status", None)
        return str(state).lower() in {"failed", "error"}

    @staticmethod
    def _terminal_failure_message(status: Any) -> Optional[str]:
        if status is None:
            return None
        msg = getattr(status, "error_message", None)
        if msg:
            return str(msg)
        details = getattr(status, "job_details", None)
        if details:
            try:
                return str(details)
            except Exception:
                return None
        return None

    def _extract_diarized_turns(self, job: SpeechToTextJob) -> List[Dict[str, Any]]:
        """Parse the completed job into a list of speaker turns.

        The Sarvam batch STT job exposes its results through
        ``SpeechToTextJob.get_file_results()``, which returns a mapping of file
        name -> list of ``SpeechToTextTranscriptionData``-like objects. Each
        transcription data object may carry a ``diarized_transcript`` made up of
        ``DiarizedEntry`` items with ``speaker_id``, ``start_time_seconds``,
        ``end_time_seconds``, and ``transcript``.
        """
        raw = job.get_file_results()
        if not raw:
            return []

        # ``get_file_results`` returns Dict[str, List[Dict[str, Any]]] in some
        # SDK versions, and Dict[str, List[SpeechToTextTranscriptionData]] in
        # others. Normalize to a flat list of the first available file's results.
        entries_flat: List[Any] = []
        if isinstance(raw, dict):
            for file_results in raw.values():
                if isinstance(file_results, list):
                    entries_flat.extend(file_results)
        elif isinstance(raw, list):
            entries_flat.extend(raw)

        turns: List[Dict[str, Any]] = []
        for item in entries_flat:
            item = self._to_dict(item) or item
            # Transcription data may be wrapped; drill into the diarized payload.
            diarized = (
                item.get("diarized_transcript")
                if isinstance(item, dict)
                else getattr(item, "diarized_transcript", None)
            )
            if diarized is None:
                # Fallback: some shapes put entries directly on the item.
                diarized = (
                    item.get("entries")
                    if isinstance(item, dict)
                    else getattr(item, "entries", None)
                )
            if not diarized:
                continue
            # ``diarized`` may itself be a ``DiarizedTranscript`` (with an
            # ``entries`` list) or a bare list of entries.
            entry_list = (
                diarized.get("entries")
                if isinstance(diarized, dict)
                else getattr(diarized, "entries", None)
                if hasattr(diarized, "entries")
                else diarized
            )
            if not entry_list:
                continue
            if not isinstance(entry_list, list):
                entry_list = [entry_list]
            for entry in entry_list:
                entry = self._to_dict(entry) or entry
                speaker_id = entry.get("speaker_id")
                if speaker_id is None:
                    speaker_id = entry.get("speaker") or entry.get("id")
                start = entry.get("start_time_seconds", entry.get("start", 0.0))
                end = entry.get("end_time_seconds", entry.get("end", 0.0))
                text = entry.get("transcript", entry.get("text", ""))
                if not text and not speaker_id:
                    continue
                turns.append(
                    {
                        "speaker": f"Speaker {speaker_id}" if speaker_id is not None else "Speaker ?",
                        "start": round(float(start), 2),
                        "end": round(float(end), 2),
                        "text": str(text or "").strip(),
                    }
                )
        return turns

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return dict(obj)
        try:
            # Pydantic-style models / dataclasses.
            return dict(obj)
        except (TypeError, ValueError):
            return None


# ----------------------------------------------------------------------
# CLI test runner
# ----------------------------------------------------------------------

def _format_timestamp(seconds: float) -> str:
    """Render seconds as [MM:SS - MM:SS]."""
    def _mmss(s: float) -> str:
        s = max(0.0, s)
        m = int(s // 60)
        sec = int(round(s % 60))
        return f"{m:02d}:{sec:02d}"
    return f"[{_mmss(seconds)} - {_mmss(seconds)}]"


def main(argv: Optional[List[str]] = None) -> None:
    argv = argv or sys.argv[1:]
    if len(argv) < 1:
        print("Usage: python diarized_stt.py <audio_file> [mode] [num_speakers]")
        print("  mode: codemix (default), transcribe, translate")
        sys.exit(1)

    audio_path = argv[0]
    mode = argv[1] if len(argv) > 1 else "codemix"
    num_speakers: Optional[int] = None
    if len(argv) > 2 and argv[2] not in ("codemix", "transcribe", "translate"):
        try:
            num_speakers = int(argv[2])
        except ValueError:
            num_speakers = None

    print(f"Audio: {audio_path}")
    print(f"Mode: {mode!r}" + (f", num_speakers={num_speakers}" if num_speakers else ""))

    service = SarvamDiarizedSTT()
    start = time.monotonic()
    try:
        turns = service.transcribe_multispeaker(audio_path, mode=mode, num_speakers=num_speakers)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(2)

    elapsed = time.monotonic() - start
    print(f"\nDiarized transcript ({len(turns)} turn(s), {elapsed:.1f}s):\n")
    if not turns:
        print("(no transcribed turns returned)")
        return
    for turn in turns:
        label = turn.get("speaker", "Speaker ?")
        text = turn.get("text", "")
        ts = _format_timestamp(turn.get("start", 0.0))
        print(f"{ts} {label}: {text}")


if __name__ == "__main__":
    main()
