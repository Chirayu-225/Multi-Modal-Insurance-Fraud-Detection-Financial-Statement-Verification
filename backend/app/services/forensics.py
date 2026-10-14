"""
Document forensics shared by repair bills (claims) and submitted
financial statements (underwriting).

Status:
  - ELABaselineDetector: real, wraps the existing image_utils.compute_ela_score.
    Naive by design -- the original README's caveat applies unchanged:
    expect it to catch crude edits and miss careful ones.
  - CNNForgeryDetector and LayoutLMv3FieldExtractor: scaffolds for
    Phase 3. Interfaces are fixed now so the underwriting pipeline can be
    written against them; implementations need torch/transformers/timm,
    which live in requirements-ml.txt rather than the base install.

Intended Phase 3 implementation:
  - CNNForgeryDetector: EfficientNet-B0 (timm) trained on DocTamper, input
    = RGB crop + ELA map + noise residual (5 channels), output = image-level
    tamper probability + coarse heatmap. fp16 + batch 4 fits a 4 GB GPU.
  - LayoutLMv3FieldExtractor: layoutlmv3-base fine-tuned for token
    classification on SROIE/CORD/FUNSD; emits Assertions with page and
    bounding box in SourceRef.locator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.case import Assertion
from app.services.image_utils import compute_ela_score


@dataclass
class ForgeryResult:
    probability: float          # 0-1, higher = more likely edited
    method: str
    heatmap_path: str | None = None


class ForgeryDetector(ABC):
    @abstractmethod
    def predict(self, image_path: str) -> ForgeryResult: ...


class ELABaselineDetector(ForgeryDetector):
    """Maps the existing ELA score straight to a 'probability'. It is not
    calibrated -- calibrating it needs labelled tampered/clean documents,
    which is exactly what DocTamper provides in Phase 3."""

    def predict(self, image_path: str) -> ForgeryResult:
        score = compute_ela_score(image_path)
        return ForgeryResult(probability=max(0.0, min(1.0, score)), method="ela_baseline_uncalibrated")


class CNNForgeryDetector(ForgeryDetector):
    def __init__(self, weights_path: str):
        raise NotImplementedError("Phase 3: train on DocTamper first (see module docstring).")

    def predict(self, image_path: str) -> ForgeryResult:  # pragma: no cover
        raise NotImplementedError


class FieldExtractor(ABC):
    @abstractmethod
    def extract(self, case_id: str, document_path: str) -> list[Assertion]: ...


class LayoutLMv3FieldExtractor(FieldExtractor):
    def __init__(self, model_path: str):
        raise NotImplementedError("Phase 3: fine-tune layoutlmv3-base first (see module docstring).")

    def extract(self, case_id: str, document_path: str) -> list[Assertion]:  # pragma: no cover
        raise NotImplementedError
