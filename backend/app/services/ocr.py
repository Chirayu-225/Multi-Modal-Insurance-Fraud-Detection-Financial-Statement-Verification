"""
Bill line-item extraction. Same two-backend pattern as vision.py.

  - MockOCR: deterministic per-file-content fake line items.
  - TesseractOCR: a real (not mocked) OCR pass using pytesseract, plus a
    deliberately simple regex parser that looks for "description ...
    amount" lines. It's a genuine baseline, not a placeholder -- but be
    upfront in your report that it's naive: real repair bills vary a lot
    in layout, and a production version would likely replace the regex
    parser with a Gemini prompt asking for structured line items the
    same way vision.py does for damage. Swapping that in later doesn't
    change this module's interface.
"""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import pytesseract
from PIL import Image

from app.ml.domain import BillLineItem

_LINE_ITEM_PATTERN = re.compile(r"([A-Za-z][A-Za-z \-_/]{2,40}?)\s+[₹Rr][sS]?\.?\s?([\d,]{3,7})")


class OCRClient(ABC):
    @abstractmethod
    def extract_bill_items(self, bill_image_path: str) -> list[BillLineItem]: ...

    @abstractmethod
    def extract_raw_text(self, bill_image_path: str) -> str:
        """The unparsed text OCR pulled off the image -- fed to
        injection_detection.detect() and compare_bill_extractions(),
        which need the full text, not just the line items a regex
        managed to parse out of it (a hidden injection payload might sit
        in a part of the bill this parser doesn't recognize as a line
        item at all)."""
        ...


class MockOCR(OCRClient):
    _CATEGORIES = ["front_bumper_repair", "rear_bumper_repair", "left_door_repair", "paint_work"]

    def extract_bill_items(self, bill_image_path: str) -> list[BillLineItem]:
        with open(bill_image_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        seed = int(digest[:8], 16)
        category = self._CATEGORIES[seed % len(self._CATEGORIES)]
        cost = 5000 + (seed % 15000)
        return [BillLineItem(category=category, cost_inr=cost)]

    def extract_raw_text(self, bill_image_path: str) -> str:
        items = self.extract_bill_items(bill_image_path)
        return "\n".join(f"{i.category.replace('_', ' ').title()} ... Rs. {i.cost_inr:,}" for i in items)


class TesseractOCR(OCRClient):
    def extract_raw_text(self, bill_image_path: str) -> str:
        with Image.open(bill_image_path) as img:
            return pytesseract.image_to_string(img)

    def extract_bill_items(self, bill_image_path: str) -> list[BillLineItem]:
        text = self.extract_raw_text(bill_image_path)
        items: list[BillLineItem] = []
        for line in text.splitlines():
            match = _LINE_ITEM_PATTERN.search(line)
            if not match:
                continue
            description, amount_str = match.groups()
            amount = int(amount_str.replace(",", ""))
            category = description.strip().lower().replace(" ", "_")
            items.append(BillLineItem(category=category, cost_inr=amount))
        return items


def get_ocr_client(backend: str) -> OCRClient:
    if backend == "gemini":
        # Bill extraction reuses the same Gemini call as damage
        # extraction in a real deployment (one multimodal prompt can
        # return both); left as TesseractOCR here to keep this file
        # focused. See vision.py's GeminiVisionClient for the pattern
        # to extend.
        return TesseractOCR()
    return MockOCR()
