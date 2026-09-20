"""Gemini runner (google-genai SDK): temperature 0, the thinking level of the system's params, safety filters off."""

from __future__ import annotations

import os
import time

from .. import paths
from ..schema import Page, Prediction
from .base import Refusal, Runner, RunnerError, image_bytes

_SAFETY_CATEGORIES = ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                      "HARM_CATEGORY_DANGEROUS_CONTENT")


def safety_off() -> list:
    """Court and council records speak of violence in the language of their time; a content filter has no business here."""
    from google.genai import types

    return [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in _SAFETY_CATEGORIES]


def make_client(timeout_s: float = 300.0):
    """The google-genai client of the runner and the judge. The per-request timeout matters: without it a hung
    connection blocks a worker for good."""
    from google import genai
    from google.genai import types

    paths.load_env()
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_API_KEY missing (.env)")
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))


class GeminiRunner(Runner):
    def setup(self) -> None:
        self.client = make_client(float(self.params.get("timeout_s", 300)))

    def _config(self):
        from google.genai import types

        p = self.params
        return types.GenerateContentConfig(
            temperature=float(p.get("temperature", 0.0)),
            max_output_tokens=int(p.get("max_output_tokens", 8192)),
            thinking_config=types.ThinkingConfig(thinking_level=p["thinking_level"]) if "thinking_level" in p else None,
            safety_settings=safety_off(),
            response_mime_type="text/plain",
        )

    def predict_one(self, page: Page, prompt: str) -> Prediction:
        from google.genai import types

        data, mime = image_bytes(page)
        t0 = time.perf_counter()
        try:
            resp = self.client.models.generate_content(
                model=self.spec.model_id, contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt], config=self._config(),
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "PROHIBITED_CONTENT" in msg or "SAFETY" in msg or "blocked" in msg.lower():
                raise Refusal(msg[:300]) from e
            raise RunnerError(msg[:300]) from e
        latency = time.perf_counter() - t0

        try:
            text = resp.text
        except Exception:  # noqa: BLE001 - the SDK raises when the response has no text part
            text = None
        if text is None:
            finish = str(resp.candidates[0].finish_reason) if resp.candidates else None
            feedback = getattr(resp, "prompt_feedback", None)
            if (finish and any(k in finish for k in ("SAFETY", "PROHIBITED", "BLOCKLIST"))) or getattr(feedback, "block_reason", None):
                raise Refusal(f"blocked: finish_reason={finish} prompt_feedback={feedback}")
            raise RunnerError(f"empty response (finish_reason={finish})")
        usage = resp.usage_metadata
        return Prediction(
            page_id=page.page_id, text=text, text_raw=text, latency_s=latency,
            tokens_in=getattr(usage, "prompt_token_count", None), tokens_out=getattr(usage, "candidates_token_count", None),
            tokens_thinking=getattr(usage, "thoughts_token_count", None),
        )
