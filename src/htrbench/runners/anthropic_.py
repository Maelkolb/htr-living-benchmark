"""Anthropic Messages API runner (Claude Sonnet 5, Claude Opus 5).

* These models take no sampling parameters, so there is no temperature to set. Decoding is steered by ``thinking``
  (default: adaptive) and ``output_config.effort`` (default: low, the closest counterpart of the Gemini runner's
  low thinking level).
* No server-side fallback for refusals is requested: a refusal is an outcome the benchmark records.
* The API rejects images above 8000 px on a side or 5 MB; such pages are downscaled and re-encoded as JPEG.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time

from ..schema import Page, Prediction
from .base import Refusal, Runner, RunnerError, image_bytes

log = logging.getLogger("htrbench.run.anthropic")

MAX_SIDE_PX = 8000
MAX_BYTES = 5 * 1024 * 1024
JPEG_QUALITY = 90

_REFUSAL_MARKERS = ("refus", "safety", "harmful", "content policy", "cannot process", "blocked")


def prepare_image(
    data: bytes, mime: str, max_side: int = MAX_SIDE_PX, max_bytes: int = MAX_BYTES
) -> tuple[bytes, str]:
    """Return (bytes, mime) acceptable to the Messages API; re-encode as JPEG q90 only when needed."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        w, h = im.size
        if len(data) <= max_bytes and max(w, h) <= max_side:
            return data, mime
        rgb = im.convert("RGB")
    scale = min(1.0, max_side / max(w, h))
    while True:
        if scale >= 1.0:
            out = rgb
        else:
            out = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        b = buf.getvalue()
        if len(b) <= max_bytes or scale < 0.1:
            log.info(
                "image re-encoded for Anthropic: %dx%d -> %dx%d, %d -> %d bytes",
                w, h, out.size[0], out.size[1], len(data), len(b),
            )
            return b, "image/jpeg"
        scale *= 0.8


class AnthropicRunner(Runner):
    def setup(self) -> None:
        import anthropic

        from .. import paths

        paths.load_env()
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                f"ANTHROPIC_API_KEY missing (.env) - required to run system {self.spec.system_id!r} "
                f"(model {self.spec.model_id})"
            )
        # The base class retries; the SDK's own retries stay off so that the latency of one attempt is what is measured.
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": ws} if ws else None  # keys tied to a workspace need its id
        self.client = anthropic.Anthropic(
            api_key=key,
            max_retries=int(self.params.get("sdk_max_retries", 0)),
            timeout=float(self.params.get("timeout_s", 600.0)),
            default_headers=headers,
        )

    def _request_kwargs(self, prompt: str, b64: str, mime: str) -> dict:
        p = self.params
        kw: dict = dict(
            model=self.spec.model_id,
            max_tokens=int(p.get("max_tokens", 8192)),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        thinking = p.get("thinking", "adaptive")
        if thinking:
            kw["thinking"] = {"type": str(thinking)}
        effort = p.get("effort", "low")
        if effort:
            kw["output_config"] = {"effort": str(effort)}
        return kw

    def predict_one(self, page: Page, prompt: str) -> Prediction:
        import anthropic

        data, mime = image_bytes(page)
        data, mime = prepare_image(data, mime)
        b64 = base64.standard_b64encode(data).decode("ascii")
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(**self._request_kwargs(prompt, b64, mime))
        except anthropic.BadRequestError as e:
            msg = str(e)
            if any(m in msg.lower() for m in _REFUSAL_MARKERS):
                raise Refusal(msg[:300]) from e
            raise RunnerError(msg[:300]) from e
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            # 429 / 529 overloaded / 5xx / network: retryable in the base class
            raise RunnerError(f"{type(e).__name__}: {e}"[:300]) from e
        except Exception as e:  # noqa: BLE001 - any other SDK error is retried too
            raise RunnerError(f"{type(e).__name__}: {e}"[:300]) from e
        latency = time.perf_counter() - t0

        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "input_tokens", None) if usage else None
        tout = getattr(usage, "output_tokens", None) if usage else None
        details = getattr(usage, "output_tokens_details", None) if usage else None
        tth = getattr(details, "reasoning_tokens", None) if details is not None else None

        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            sd = getattr(resp, "stop_details", None)
            cat = getattr(sd, "category", None) if sd is not None else None
            expl = getattr(sd, "explanation", None) if sd is not None else None
            raise Refusal(f"stop_reason=refusal category={cat} explanation={expl}")
        blocks = getattr(resp, "content", None) or []
        text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text")
        if not blocks or not text.strip():
            raise Refusal(f"empty content (stop_reason={stop})")
        if stop == "max_tokens":
            log.warning("%s: output truncated at max_tokens=%s", page.page_id, self.params.get("max_tokens"))
        return Prediction(
            page_id=page.page_id,
            text=text,
            text_raw=text,
            latency_s=latency,
            tokens_in=tin,
            tokens_out=tout,
            tokens_thinking=tth,
        )
