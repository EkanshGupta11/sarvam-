"""
Conversation intelligence layer for multi-speaker, code-switched Indic dialogues.

Provides ``ConversationBrain``: ingest a diarized turn list, then either answer
natural-language questions about the conversation (in the language of the query)
or produce structured meeting minutes. The LLM layer is pluggable so you can point
it at Sarvam Chat Completion or any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "You are an Indic meeting & conversation intelligence agent. You are provided "
    "with a multi-speaker transcript where speakers switch between Indian languages "
    "(Hindi, Tamil, Telugu, Marathi, etc.) and English.\n"
    "- Ground all answers strictly in what each speaker explicitly stated.\n"
    "- If asked what a specific speaker (e.g., Speaker 0 or Speaker 1) said or "
    "committed to, isolate their dialogue.\n"
    "- Respond clearly in the language of the user's query.\n"
    "- If a question cannot be answered from the transcript, say so explicitly "
    "instead of guessing.\n"
    "- Preserve names, numbers, dates, deadlines, and commitments exactly as stated.\n"
)


class ConversationBrain:
    """Intelligence layer over a structured, diarized multi-speaker transcript."""

    def __init__(
        self,
        turns: Optional[List[Dict[str, Any]]] = None,
        *,
        llm_client: Optional[Any] = None,
        llm_model: str = "sarvam-105b",
        fallback_endpoint: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the conversation brain.

        Parameters
        ----------
        turns:
            Structured diarized turns from ``SarvamDiarizedSTT`` or similar, each
            with at least ``speaker``, ``start``, ``end``, ``text``.
        llm_client:
            Optional already-initialized Sarvam AI client. If provided, it must
            expose ``chat.completions(model=..., messages=...)``.
        llm_model:
            Model id to use for chat completion when using the Sarvam client.
        fallback_endpoint:
            Optional OpenAI-compatible HTTP endpoint (for Sarvam, a local proxy,
            or any compatible LLM host). If set, ``fallback_api_key`` must also be
            provided.
        fallback_api_key:
            API key to pass as ``Authorization: Bearer <key>`` for the fallback
            endpoint.
        """
        self._turns: List[Dict[str, Any]] = list(turns or [])
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._fallback_endpoint = fallback_endpoint
        self._fallback_api_key = fallback_api_key

    # ------------------------------------------------------------------
    # Turn management
    # ------------------------------------------------------------------

    def set_turns(self, turns: List[Dict[str, Any]]) -> None:
        self._turns = list(turns or [])

    def add_turn(self, speaker: str, start: float, end: float, text: str) -> None:
        entry: Dict[str, Any] = {"speaker": speaker, "start": start, "end": end, "text": text}
        self._turns.append(entry)

    @property
    def turns(self) -> List[Dict[str, Any]]:
        return list(self._turns)

    # ------------------------------------------------------------------
    # Transcript formatting
    # ------------------------------------------------------------------

    def transcript_text(self) -> str:
        """Render the full transcript with speaker tags, in order."""
        if not self._turns:
            return "(no transcript provided)"
        lines = [
            f"[{self._format_ts(turn.get('start', 0.0))} - {self._format_ts(turn.get('end', 0.0))}] "
            f"{turn.get('speaker', 'Speaker ?')}: {turn.get('text', '') or ''}"
            for turn in self._turns
        ]
        return "\n".join(lines)

    def speaker_lines(self, speaker: str) -> List[str]:
        """Return transcript lines spoken by a specific speaker label."""
        out: List[str] = []
        for turn in self._turns:
            if turn.get("speaker", "").lower() == speaker.lower():
                text = turn.get("text") or ""
                if text.strip():
                    out.append(f"{turn.get('speaker')}: {text}")
        return out

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_conversation(self, user_question: str) -> str:
        """Answer a natural-language question about the conversation.

        The answer is grounded strictly in the transcript, and is returned in the
        language of the user's query (which may be an Indian language, Hinglish,
        or English).
        """
        if not user_question.strip():
            return ""

        transcript = self.transcript_text()
        user_prompt = (
            f"Transcript:\n{transcript}\n\n"
            f"Question: {user_question}"
        )
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self._chat_complete(message)

    # ------------------------------------------------------------------
    # Structured minutes
    # ------------------------------------------------------------------

    def generate_action_minutes(self) -> Dict[str, Any]:
        """Generate structured meeting minutes from the transcript.

        Returns
        -------
        Dict with keys:
        - executive_summary
        - speaker_key_points: Dict[str, List[str]]
        - decisions: List[Dict[str, Any]]
        - action_items: List[Dict[str, Any]]
        """
        transcript = self.transcript_text()
        instruction = (
            "You are an Indic meeting & conversation intelligence agent.\n"
            "From the transcript below, produce a concise, structured report in "
            "the same language used in the transcript (or English if the transcript "
            "is in English).\n\n"
            "Return ONLY a valid JSON object with these exact keys:\n"
            "{\n"
            "  \"executive_summary\": \"one-paragraph pan-conversation summary\",\n"
            "  \"speaker_key_points\": { \"Speaker 0\": [\"point 1\", ...], ... },\n"
            "  \"decisions\": [{\"decision\": \"...\", \"by\": \"Speaker N\", \"context\": \"...\"}], \n"
            "  \"action_items\": [{\"action\": \"...\", \"owner\": \"Speaker N\", \"deadline\": \"...\" or \"TBD\"}]\n"
            "}\n\n"
            "Ground everything strictly in the transcript. If something is unknown, "
            "use \"TBD\" or omit it. Do NOT invent facts. Do NOT add markdown fences.\n"
            "Transcript:\n"
            f"{transcript}"
        )
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        raw = self._chat_complete(message).strip()
        return self._parse_minutes_json(raw)

    # ------------------------------------------------------------------
    # LLM gateway
    # ------------------------------------------------------------------

    def _chat_complete(self, messages: List[Dict[str, str]]) -> str:
        """Delegate to the configured LLM backend.

        Priority:
        1. Sarvam client (``llm_client.chat.completions``)
        2. OpenAI-compatible fallback endpoint (HTTP POST)
        """
        if self._llm_client is not None:
            return self._chat_via_sarvam(messages)

        if self._fallback_endpoint and self._fallback_api_key:
            return self._chat_via_fallback(messages)

        raise RuntimeError(
            "No LLM backend configured. Pass an ``llm_client`` to ConversationBrain, "
            "or set ``fallback_endpoint`` and ``fallback_api_key``."
        )

    def _chat_via_sarvam(self, messages: List[Dict[str, str]]) -> str:
        client = self._llm_client
        response = client.chat.completions(model=self._llm_model, messages=messages)
        # Sarvam response shape: response.choices[0].message.content
        choices = getattr(response, "choices", None)
        if choices:
            first = choices[0] if isinstance(choices, list) else choices
            msg = getattr(first, "message", None)
            if msg:
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    return content
        # Defensive fallback: dump the object to string.
        return str(response)

    def _chat_via_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Minimal OpenAI-compatible chat completions over HTTP (stdlib only)."""
        import urllib.request
        import urllib.error

        payload = json.dumps({"model": self._llm_model, "messages": messages}).encode("utf-8")
        req = urllib.request.Request(
            self._fallback_endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._fallback_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM fallback HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM fallback request failed: {exc}") from exc

        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            return json.dumps(data)
        first = choices[0]
        msg = first.get("message", {})
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        return json.dumps(data)

    # ------------------------------------------------------------------
    # Minutes parsing
    # ------------------------------------------------------------------

    def _parse_minutes_json(self, raw: str) -> Dict[str, Any]:
        """Extract a JSON object from a possibly-marked LLM response."""
        text = raw.strip()
        # Strip common markdown fences.
        for fence in ["```json", "```", "```JSON"]:
            if text.startswith(fence):
                text = text[len(fence):].lstrip()
            if text.endswith("```"):
                text = text[:-3].rstrip()
        # Sometimes the model returns ``{"key": ...}`` inline; try to find the
        # outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                "executive_summary": text,
                "speaker_key_points": {},
                "decisions": [],
                "action_items": [],
                "_raw": raw,
                "_parse_error": True,
            }
        if not isinstance(parsed, dict):
            return {
                "executive_summary": str(parsed),
                "speaker_key_points": {},
                "decisions": [],
                "action_items": [],
                "_raw": raw,
                "_parse_error": True,
            }
        return {
            "executive_summary": parsed.get("executive_summary") or "",
            "speaker_key_points": self._ensure_dict_of_lists(parsed.get("speaker_key_points")),
            "decisions": self._ensure_list_of_dicts(parsed.get("decisions")),
            "action_items": self._ensure_list_of_dicts(parsed.get("action_items")),
        }

    @staticmethod
    def _ensure_dict_of_lists(value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, list):
                out[str(k)] = [str(x) for x in v if x is not None]
            elif v is not None:
                out[str(k)] = [str(v)]
            else:
                out[str(k)] = []
        return out

    @staticmethod
    def _ensure_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        out: List[Dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict):
                out.append({str(k): v for k, v in item.items()})
            elif item is not None:
                out.append({"item": str(item)})
        return out

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_ts(seconds: float) -> str:
        s = max(0.0, float(seconds))
        m = int(s // 60)
        sec = int(round(s % 60))
        return f"{m:02d}:{sec:02d}"


# ----------------------------------------------------------------------
# Standalone mock tests
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    SAMPLE_TURNS: List[Dict[str, Any]] = [
        {
            "speaker": "Speaker 0",
            "start": 0.0,
            "end": 4.2,
            "text": "Haan, main kal meeting join karunga, lekin mujhe thoda late hoga,",
        },
        {
            "speaker": "Speaker 0",
            "start": 4.2,
            "end": 9.0,
            "text": "kyunki train schedule change ho gaya.",
        },
        {
            "speaker": "Speaker 1",
            "start": 9.0,
            "end": 14.5,
            "text": "Okay, no problem. Please make sure the deck is ready by 9 AM.",
        },
        {
            "speaker": "Speaker 1",
            "start": 14.5,
            "end": 18.3,
            "text": "Aur FY26 projections slip kar diye hain, hence we need those updated slides.",
        },
        {
            "speaker": "Speaker 0",
            "start": 18.3,
            "end": 23.0,
            "text": "Haan, slides main update kar dunga, but FY26 numbers verify karne hain.",
        },
        {
            "speaker": "Speaker 0",
            "start": 23.0,
            "end": 28.4,
            "text": "Mujhe finance se confirmation chahiye, tabhi finalize kar paunga.",
        },
        {
            "speaker": "Speaker 1",
            "start": 28.4,
            "end": 33.0,
            "text": "Fine, then action items: you confirm numbers, I share the final agenda by EOD.",
        },
    ]

    brain = ConversationBrain(turns=SAMPLE_TURNS)

    print("=" * 72)
    print("TRANSCRIPT")
    print("=" * 72)
    print(brain.transcript_text())
    print()

    print("=" * 72)
    print("QUERY 1: In English — what did Speaker 0 commit to?")
    print("=" * 72)
    # In mock mode we have no LLM backend, so this will raise a clear error
    # explaining how to wire one in. That is exactly the verification we want
    # before connecting a real model.
    try:
        print(brain.query_conversation("What did Speaker 0 commit to?"))
    except RuntimeError as exc:
        print(f"[mock run — no LLM backend configured: {exc}]")
    print()

    print("=" * 72)
    print("QUERY 2: In Hinglish — Speaker 1 ne kya maanga ya confirm kiya?")
    print("=" * 72)
    try:
        print(brain.query_conversation("Speaker 1 ne kya maanga ya confirm kiya?"))
    except RuntimeError as exc:
        print(f"[mock run — no LLM backend configured: {exc}]")
    print()

    print("=" * 72)
    print("ACTION MINUTES (mock — no LLM backend)")
    print("=" * 72)
    try:
        minutes = brain.generate_action_minutes()
        print(json.dumps(minutes, indent=2, ensure_ascii=False))
    except RuntimeError as exc:
        print(f"[mock run — no LLM backend configured: {exc}]")
    print()

    # To actually exercise the LLM path, set e.g.:
    #   SARVAM_CHAT_MODE=sarvam-105b
    # and one of:
    #   - SARVAM_API_KEY (so we can build a SarvamAI client here)
    #   - or SARVAM_FALLBACK_ENDPOINT + SARVAM_FALLBACK_API_KEY
    print("=" * 72)
    print("LIVE TEST HARNESS (if credentials are available)")
    print("=" * 72)

    api_key = os.environ.get("SARVAM_API_KEY")
    fallback_endpoint = os.environ.get("SARVAM_FALLBACK_ENDPOINT")
    fallback_api_key = os.environ.get("SARVAM_FALLBACK_API_KEY") or api_key
    live_model = os.environ.get("SARVAM_CHAT_MODEL", "sarvam-105b")

    if api_key or fallback_endpoint:
        try:
            # Lazy import so the module stays usable even if sarvamai isn't installed.
            from sarvamai import SarvamAI  # type: ignore[import-not-found]
        except Exception:
            SarvamAI = None  # type: ignore[assignment]

        client = None
        if api_key and SarvamAI is not None:
            client = SarvamAI(api_subscription_key=api_key)

        brain_live = ConversationBrain(
            turns=SAMPLE_TURNS,
            llm_client=client,
            llm_model=live_model,
            fallback_endpoint=fallback_endpoint,
            fallback_api_key=fallback_api_key,
        )

        print("— query_conversation (English) —")
        start = time.monotonic()
        answer = brain_live.query_conversation("What did Speaker 0 commit to?")
        print(answer)
        print(f"(elapsed: {time.monotonic() - start:.2f}s)\n")

        print("— query_conversation (Hinglish) —")
        start = time.monotonic()
        answer = brain_live.query_conversation("Speaker 1 ne kya maanga ya confirm kiya?")
        print(answer)
        print(f"(elapsed: {time.monotonic() - start:.2f}s)\n")

        print("— generate_action_minutes —")
        start = time.monotonic()
        minutes = brain_live.generate_action_minutes()
        print(json.dumps(minutes, indent=2, ensure_ascii=False))
        print(f"(elapsed: {time.monotonic() - start:.2f}s)\n")
    else:
        print(
            "No LLM backend available for a live run. Set one of:\n"
            "  - SARVAM_API_KEY (Sarvam Chat Completion), or\n"
            "  - SARVAM_FALLBACK_ENDPOINT + SARVAM_FALLBACK_API_KEY\n"
            "and re-run this file."
        )
