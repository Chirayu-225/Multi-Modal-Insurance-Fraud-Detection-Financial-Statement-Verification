"""
Vision extraction: turning a claim photo into a (damage_location,
damage_severity) pair. Two backends behind one interface, selected by
app/config.py's `vision_backend` setting -- this is the seam the whole
architecture is built around, so swapping mock -> Gemini -> your own
fine-tuned classifier never touches the API layer or the consistency
engine.

  - MockVisionClient: no network call, no model weights. Deterministic
    per file (seeded by a hash of the file bytes), so the same uploaded
    photo always gets the same mock classification -- useful for demos
    and for not lying to yourself about reproducibility.
  - GeminiVisionClient: a real, working implementation against the
    Gemini API. Untested in this environment (no API key here) but
    correct against the documented generateContent REST shape as of
    writing -- verify against current docs before your viva, since API
    shapes do change.

Neither backend is where your "own ML work" claim should rest -- that's
the fine-tuned classifier from the build plan (ResNet-18/EfficientNet-B0
on the Kaggle damage set), which slots in here as a third implementation
of VisionClient once you've trained it.
"""
from __future__ import annotations

import base64
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import httpx

from app.ml.domain import DamageLocation, Severity
from app.services.image_utils import extract_exif_capture_date


class SchemaViolationError(RuntimeError):
    """Raised when the vision backend's response doesn't match the
    strict schema the prompt demanded. Deliberately its own exception
    type (not a bare ValueError) so pipeline.py can catch it
    specifically and treat it as a security-relevant event -- a
    well-behaved model asked for exactly ["front_bumper", ..., "windshield"]
    doesn't return something else by accident nearly as often as it does
    because injected text talked it into it."""


@dataclass
class DamageExtraction:
    location: DamageLocation
    severity: Severity
    capture_date: date | None
    raw_model_output: str | None = None  # kept for auditability / explanation phrasing


class VisionClient(ABC):
    @abstractmethod
    def extract_damage(self, photo_path: str) -> DamageExtraction: ...


class MockVisionClient(VisionClient):
    """Deterministic per-file-content mock. Swap for GeminiVisionClient
    or your fine-tuned classifier by changing VISION_BACKEND in .env --
    nothing else in the codebase needs to change."""

    def extract_damage(self, photo_path: str) -> DamageExtraction:
        with open(photo_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        seed = int(digest[:8], 16)
        locations = list(DamageLocation)
        severities = list(Severity)
        location = locations[seed % len(locations)]
        severity = severities[(seed // len(locations)) % len(severities)]
        capture_date = extract_exif_capture_date(photo_path)  # real, not mocked
        return DamageExtraction(
            location=location,
            severity=severity,
            capture_date=capture_date,
            raw_model_output="mock: no vision model called, classification derived from file hash",
        )


class GeminiVisionClient(VisionClient):
    """Real implementation against the Gemini generateContent REST
    endpoint. Requires GEMINI_API_KEY. The prompt asks the model to
    return strict JSON so this stays a structured-extraction step, not
    a free-form narrative the consistency engine would have to parse
    unreliably."""

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def extract_damage(self, photo_path: str) -> DamageExtraction:
        with open(photo_path, "rb") as f:
            image_bytes = f.read()
        b64 = base64.b64encode(image_bytes).decode()

        valid_locations = [l.value for l in DamageLocation]
        valid_severities = [s.value for s in Severity]
        prompt = (
            "You are assessing vehicle damage from a photo for an insurance claim. "
            f"Respond with ONLY a JSON object, no markdown fences, with exactly these keys: "
            f'"location" (one of {valid_locations}), '
            f'"severity" (one of {valid_severities}). '
            "Pick the single most prominent damaged panel visible in the photo."
        )

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            }],
            "generationConfig": {"temperature": 0.0},
        }
        url = self.ENDPOINT.format(model=self.model)
        resp = httpx.post(url, params={"key": self.api_key}, json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Schema-constrained output as an injection defense: this is not
        # just error handling. A response that doesn't parse as JSON, is
        # missing a key, or names a location/severity outside the
        # enumerated set is treated as a schema violation and surfaced to
        # pipeline.py as a suspected-injection signal, not swallowed as
        # an ordinary bug -- see SchemaViolationError's docstring.
        try:
            parsed = json.loads(text)
            location = DamageLocation(parsed["location"])
            severity = Severity(parsed["severity"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise SchemaViolationError(
                f"Vision model response did not match the required schema (expected JSON with "
                f"location in {valid_locations} and severity in {valid_severities}): {exc}. Raw response: {text!r}"
            ) from exc

        capture_date = extract_exif_capture_date(photo_path)  # real, independent of the API call
        return DamageExtraction(
            location=location,
            severity=severity,
            capture_date=capture_date,
            raw_model_output=text,
        )


def get_vision_client(backend: str, api_key: str | None, model: str) -> VisionClient:
    if backend == "gemini":
        if not api_key:
            raise RuntimeError("VISION_BACKEND=gemini requires GEMINI_API_KEY to be set")
        return GeminiVisionClient(api_key=api_key, model=model)
    return MockVisionClient()
