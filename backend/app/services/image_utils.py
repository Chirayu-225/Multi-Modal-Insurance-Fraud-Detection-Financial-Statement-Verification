"""
Real, non-mocked utilities that work on the actual uploaded file bytes
regardless of which vision backend (mock or Gemini) is selected:
perceptual hashing (for reused-photo detection) and EXIF capture-date
extraction (for the timestamp check). Unlike damage classification or
bill OCR, these don't need an ML model or an API call -- they're just
correct code, so there's no reason for them to be mocked.
"""
from __future__ import annotations

import io
from datetime import date, datetime

import imagehash
import numpy as np
from PIL import Image, ImageChops


def compute_phash(image_path: str) -> str:
    """Perceptual hash as a hex string, comparable via Hamming distance
    (see app/services/consistency.py::_hamming_distance). This is the
    pHash baseline the review recommended over CLIP for duplicate-photo
    detection -- robust to recompression and minor edits, cheap to run,
    no model download required."""
    with Image.open(image_path) as img:
        return str(imagehash.phash(img))


def extract_exif_capture_date(image_path: str) -> date | None:
    """Best-effort EXIF DateTimeOriginal / DateTime extraction. Returns
    None for images with no EXIF (screenshots, edited/re-saved photos,
    most PNGs) -- which is itself a mild signal worth surfacing in the
    UI ("no capture metadata available"), even though this function
    only returns the date when it's actually present."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if not exif:
                return None
            # 0x9003 = DateTimeOriginal, 0x0132 = DateTime
            raw = exif.get(0x9003) or exif.get(0x0132)
            if not raw:
                return None
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").date()
    except Exception:
        return None


def compute_ela_score(image_path: str, quality: int = 90) -> float:
    """Error Level Analysis: re-compress the image at a fixed JPEG
    quality and measure how much it changes. Genuine, untouched photo
    regions settle into a stable, low, fairly uniform compression error;
    a region that was pasted in or edited after the fact was compressed
    a different number of times / at different settings, so it lights up
    with a distinctly higher error under re-compression than the rest of
    the image.

    This is a real, cheap, well-known forensic baseline -- and a naive
    one. State-of-the-art inpainting can defeat it, and it produces false
    positives on images that were legitimately re-saved or lightly edited
    for innocent reasons (e.g. a messaging app's own recompression).
    Treat the returned score (roughly 0-1, higher = more suspicious) as
    one weak signal, never a verdict -- see the caller in consistency.py
    for how it's weighted, and domain.py's `ela_score` field docstring
    for why it's never populated in synthetic/evaluation data.
    """
    with Image.open(image_path) as original:
        original = original.convert("RGB")
        buffer = io.BytesIO()
        original.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            diff = ImageChops.difference(original, recompressed)
            arr = np.asarray(diff, dtype=np.float32)

    if arr.size == 0:
        return 0.0

    # Normalize: mean absolute error as a fraction of the max possible
    # (255), then rescale so "typical" recompression noise (~a few
    # units) sits well below the 0.6 flag threshold in consistency.py,
    # and a large localized discrepancy pushes above it.
    mean_error = float(arr.mean())
    return min(1.0, mean_error / 12.0)
