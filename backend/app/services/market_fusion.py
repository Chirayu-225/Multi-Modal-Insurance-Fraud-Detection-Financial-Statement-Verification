"""
Market-signal fusion for underwriting of listed companies (stretch goal).

Status: interface only. NullMarketSignal returns None, which the
underwriting rule pack treats as "not run" -- so the whole pipeline works
without this module, and it can be dropped first if time runs short.

Intended implementation: frozen FinBERT (headlines), ResNet-18 (candlestick
images rendered with mplfinance) and a small Transformer (price windows);
cached embeddings; a gated-attention fusion head predicting next-period
realised volatility; walk-forward validation to avoid look-ahead bias.
The output feeds UnderwritingEvidence.market_stress_score, a weak,
low-severity signal only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date


class MarketSignalProvider(ABC):
    @abstractmethod
    def stress_score(self, ticker: str, as_of: date) -> float | None: ...


class NullMarketSignal(MarketSignalProvider):
    def stress_score(self, ticker: str, as_of: date) -> float | None:
        return None
