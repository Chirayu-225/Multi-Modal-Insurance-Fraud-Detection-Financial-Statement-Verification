"""
One gateway for every generative-model call.

Responsibilities, in the order a request meets them:

  1. Privacy gate + redaction   services/privacy_guard.py; restricted
                                content never leaves the machine.
  2. Task routing               text -> e.g. Groq, then OpenRouter;
                                vision -> e.g. Gemini. Order comes from
                                config (llm_text_providers, ...).
  3. Response cache             SQLite, keyed on provider + model + task
                                + prompt + image hash. Repeat runs cost no
                                quota and evaluation becomes reproducible.
  4. Rate limiting              per-provider requests/minute and /day
                                budgets, so the app queues or fails over
                                instead of hammering a free tier into 429s.
  5. Retry and failover         backoff on transient errors, then the next
                                provider, then an optional local model.
  6. Call log                   provider, model, latency, cache hit for
                                every call, for the evaluation tables.

Status:
  - Gateway logic, cache, limiter, failover, mock provider: real, tested.
  - OpenAI-compatible provider (Groq, OpenRouter, local Ollama) and the
    Gemini provider: real code against the documented request shapes,
    NOT tested against live APIs in this build (no keys were used).
    Verify against each provider's current docs before relying on them.
  - The published free limits below are defaults to edit, not facts:
    providers change them without notice.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import httpx

from app.config import settings
from app.core.case import Sensitivity
from app.services.privacy_guard import RestrictedContentError, prepare_for_hosted


class Task(str, Enum):
    TEXT = "text"
    VISION = "vision"


@dataclass
class LLMRequest:
    task: Task
    prompt: str
    sensitivity: Sensitivity
    system: str | None = None
    image_bytes: bytes | None = None
    image_mime: str = "image/jpeg"
    json_output: bool = True
    temperature: float = 0.0
    max_tokens: int = 1024


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    cached: bool
    latency_ms: float
    redactions: dict[str, int] = field(default_factory=dict)


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class AllProvidersFailedError(RuntimeError):
    pass


# ---------------------------------------------------------------- providers

class Provider(ABC):
    name: str
    model: str
    tasks: frozenset[Task]
    hosted: bool = True

    def supports(self, task: Task) -> bool:
        return task in self.tasks

    @abstractmethod
    def complete(self, req: LLMRequest, prompt: str) -> str: ...


class MockProvider(Provider):
    """Offline, deterministic stand-in so the whole stack runs with no
    keys. Returns a fixed JSON envelope derived from the prompt hash; it
    has no reasoning surface, so it can't be used to measure accuracy or
    injection resistance -- only to exercise the plumbing."""
    name = "mock"
    model = "mock-1"
    tasks = frozenset({Task.TEXT, Task.VISION})
    hosted = False

    def complete(self, req: LLMRequest, prompt: str) -> str:
        digest = hashlib.sha256((prompt + (req.task.value)).encode()).hexdigest()[:12]
        return json.dumps({"mock": True, "digest": digest, "answer": None})


class OpenAICompatibleProvider(Provider):
    """Groq, OpenRouter and Ollama all expose an OpenAI-style
    /chat/completions endpoint, so one class covers all three."""

    def __init__(self, name: str, base_url: str, api_key: str | None, model: str, hosted: bool = True, timeout: float = 60.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.hosted = hosted
        self.timeout = timeout
        self.tasks = frozenset({Task.TEXT})

    def complete(self, req: LLMRequest, prompt: str) -> str:
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.model, "messages": messages, "temperature": req.temperature, "max_tokens": req.max_tokens}
        if req.json_output:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}: network error {exc}") from exc
        if r.status_code == 429 or r.status_code >= 500:
            raise ProviderError(f"{self.name}: HTTP {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {r.status_code} {r.text[:200]}", retryable=False)
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"{self.name}: unexpected response shape", retryable=False) from exc


class GeminiProvider(Provider):
    """Gemini generateContent REST endpoint, text or image+text."""
    name = "gemini"
    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.tasks = frozenset({Task.TEXT, Task.VISION})

    def complete(self, req: LLMRequest, prompt: str) -> str:
        parts: list[dict] = [{"text": prompt}]
        if req.image_bytes is not None:
            parts.insert(0, {"inline_data": {"mime_type": req.image_mime, "data": base64.b64encode(req.image_bytes).decode()}})
        body: dict = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": req.temperature, "maxOutputTokens": req.max_tokens},
        }
        if req.json_output:
            body["generationConfig"]["responseMimeType"] = "application/json"
        if req.system:
            body["systemInstruction"] = {"parts": [{"text": req.system}]}
        try:
            r = httpx.post(self.ENDPOINT.format(model=self.model), params={"key": self.api_key}, json=body, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise ProviderError(f"gemini: network error {exc}") from exc
        if r.status_code == 429 or r.status_code >= 500:
            raise ProviderError(f"gemini: HTTP {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise ProviderError(f"gemini: HTTP {r.status_code} {r.text[:200]}", retryable=False)
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError("gemini: unexpected response shape", retryable=False) from exc


# ------------------------------------------------------------ rate limiting

class RateLimiter:
    """Sliding-window requests-per-minute plus a daily cap. `clock` is
    injectable so tests don't sleep."""

    def __init__(self, rpm: int, rpd: int, clock: Callable[[], float] = time.monotonic):
        self.rpm, self.rpd, self.clock = rpm, rpd, clock
        self._minute: deque[float] = deque()
        self._day: deque[float] = deque()

    def try_acquire(self) -> bool:
        now = self.clock()
        while self._minute and now - self._minute[0] >= 60:
            self._minute.popleft()
        while self._day and now - self._day[0] >= 86_400:
            self._day.popleft()
        if len(self._minute) >= self.rpm or len(self._day) >= self.rpd:
            return False
        self._minute.append(now)
        self._day.append(now)
        return True


# Edit to match what each provider currently publishes for your account.
DEFAULT_LIMITS = {
    "groq": (30, 1000),
    "openrouter": (20, 50),
    "gemini": (10, 250),
    "local": (10_000, 10_000_000),
    "mock": (10_000, 10_000_000),
}


# -------------------------------------------------------------------- cache

class ResponseCache:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, text TEXT, created REAL)")

    @staticmethod
    def key(provider: str, model: str, req: LLMRequest, prompt: str) -> str:
        image_digest = hashlib.sha256(req.image_bytes).hexdigest() if req.image_bytes else ""
        material = json.dumps(
            [provider, model, req.task.value, req.system, prompt, image_digest, req.temperature, req.json_output],
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT text FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, text: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, text, time.time()))
        self._conn.commit()


# ------------------------------------------------------------------ gateway

@dataclass
class CallRecord:
    provider: str
    model: str
    task: str
    cached: bool
    ok: bool
    latency_ms: float
    error: str | None = None


class LLMGateway:
    def __init__(
        self,
        routes: dict[Task, list[Provider]],
        cache: ResponseCache | None = None,
        limiters: dict[str, RateLimiter] | None = None,
        local_fallback: Provider | None = None,
        max_retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.routes = routes
        self.cache = cache
        self.limiters = limiters or {}
        self.local_fallback = local_fallback
        self.max_retries = max_retries
        self.sleep = sleep
        self.call_log: list[CallRecord] = []

    def _attempt(self, provider: Provider, req: LLMRequest, prompt: str) -> str:
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                return provider.complete(req, prompt)
            except ProviderError as exc:
                if not exc.retryable or attempt == self.max_retries:
                    raise
                self.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def _call(self, provider: Provider, req: LLMRequest, prompt: str) -> LLMResponse:
        key = ResponseCache.key(provider.name, provider.model, req, prompt) if self.cache else None
        if key and (hit := self.cache.get(key)) is not None:
            self.call_log.append(CallRecord(provider.name, provider.model, req.task.value, True, True, 0.0))
            return LLMResponse(hit, provider.name, provider.model, True, 0.0)
        limiter = self.limiters.get(provider.name)
        if limiter and not limiter.try_acquire():
            raise ProviderError(f"{provider.name}: local rate budget exhausted", retryable=False)
        start = time.perf_counter()
        text = self._attempt(provider, req, prompt)
        latency = (time.perf_counter() - start) * 1000
        if key:
            self.cache.put(key, text)
        self.call_log.append(CallRecord(provider.name, provider.model, req.task.value, False, True, latency))
        return LLMResponse(text, provider.name, provider.model, False, latency)

    def complete(self, req: LLMRequest) -> LLMResponse:
        # Gate + redact once, before any provider sees anything.
        # Restricted content goes to the local model if one is configured
        # (unredacted -- it never leaves the machine); otherwise the
        # RestrictedContentError propagates. Silently trying another
        # hosted provider would defeat the point.
        try:
            prepared = prepare_for_hosted(req.prompt, req.sensitivity)
        except RestrictedContentError:
            if self.local_fallback and not self.local_fallback.hosted and self.local_fallback.supports(req.task):
                return self._call(self.local_fallback, req, req.prompt)
            raise
        errors: list[str] = []
        for provider in self.routes.get(req.task, []):
            if not provider.supports(req.task):
                continue
            try:
                resp = self._call(provider, req, prepared.text)
                resp.redactions = prepared.counts
                return resp
            except ProviderError as exc:
                errors.append(str(exc))
                self.call_log.append(CallRecord(provider.name, provider.model, req.task.value, False, False, 0.0, str(exc)))
        if self.local_fallback and self.local_fallback.supports(req.task):
            try:
                resp = self._call(self.local_fallback, req, prepared.text)
                resp.redactions = prepared.counts
                return resp
            except ProviderError as exc:
                errors.append(str(exc))
        raise AllProvidersFailedError("; ".join(errors) or f"no provider configured for task {req.task.value}")


def build_gateway_from_settings() -> LLMGateway:
    """Assemble providers from config, skipping any without a key."""
    available: dict[str, Provider] = {"mock": MockProvider()}
    if settings.groq_api_key:
        available["groq"] = OpenAICompatibleProvider("groq", "https://api.groq.com/openai/v1", settings.groq_api_key, settings.groq_model)
    if settings.openrouter_api_key and settings.openrouter_model:
        available["openrouter"] = OpenAICompatibleProvider(
            "openrouter", "https://openrouter.ai/api/v1", settings.openrouter_api_key, settings.openrouter_model
        )
    if settings.gemini_api_key:
        available["gemini"] = GeminiProvider(settings.gemini_api_key, settings.gemini_model)

    def chain(spec: str) -> list[Provider]:
        return [available[n.strip()] for n in spec.split(",") if n.strip() in available]

    local = None
    if settings.local_llm_base_url:
        local = OpenAICompatibleProvider("local", settings.local_llm_base_url, None, settings.local_llm_model, hosted=False)

    limiters = {name: RateLimiter(*DEFAULT_LIMITS[name]) for name in list(available) + (["local"] if local else []) if name in DEFAULT_LIMITS}
    return LLMGateway(
        routes={Task.TEXT: chain(settings.llm_text_providers), Task.VISION: chain(settings.llm_vision_providers)},
        cache=ResponseCache(settings.llm_cache_path),
        limiters=limiters,
        local_fallback=local,
        max_retries=settings.llm_max_retries,
    )
