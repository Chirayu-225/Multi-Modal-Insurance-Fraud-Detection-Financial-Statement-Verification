"""Gateway plumbing, entirely offline: fake providers, injected clock and sleep."""
import pytest

from app.core.case import Sensitivity
from app.services.llm_gateway import (
    AllProvidersFailedError,
    LLMGateway,
    LLMRequest,
    MockProvider,
    Provider,
    ProviderError,
    RateLimiter,
    ResponseCache,
    Task,
)
from app.services.privacy_guard import RestrictedContentError


class FakeProvider(Provider):
    def __init__(self, name, fail_times=0, retryable=True, hosted=True):
        self.name, self.model, self.hosted = name, f"{name}-m", hosted
        self.tasks = frozenset({Task.TEXT})
        self.fail_times, self.retryable, self.calls, self.prompts = fail_times, retryable, 0, []

    def complete(self, req, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls <= self.fail_times:
            raise ProviderError(f"{self.name} down", retryable=self.retryable)
        return f'{{"from": "{self.name}"}}'


def req(text="What was FY2025 revenue?", sensitivity=Sensitivity.PUBLIC):
    return LLMRequest(task=Task.TEXT, prompt=text, sensitivity=sensitivity)


def test_cache_hit_skips_provider():
    p = FakeProvider("a")
    gw = LLMGateway({Task.TEXT: [p]}, cache=ResponseCache(":memory:"), sleep=lambda s: None)
    first, second = gw.complete(req()), gw.complete(req())
    assert p.calls == 1 and not first.cached and second.cached and second.text == first.text


def test_retry_then_success():
    p = FakeProvider("a", fail_times=1)
    gw = LLMGateway({Task.TEXT: [p]}, max_retries=2, sleep=lambda s: None)
    assert gw.complete(req()).provider == "a" and p.calls == 2


def test_failover_to_next_provider_on_non_retryable_error():
    a, b = FakeProvider("a", fail_times=99, retryable=False), FakeProvider("b")
    gw = LLMGateway({Task.TEXT: [a, b]}, sleep=lambda s: None)
    assert gw.complete(req()).provider == "b"


def test_rate_budget_exhaustion_fails_over():
    clock = [0.0]
    a, b = FakeProvider("a"), FakeProvider("b")
    gw = LLMGateway({Task.TEXT: [a, b]}, limiters={"a": RateLimiter(rpm=1, rpd=100, clock=lambda: clock[0])},
                    sleep=lambda s: None)
    assert gw.complete(req("q1")).provider == "a"
    assert gw.complete(req("q2")).provider == "b"   # a's per-minute budget used up
    clock[0] = 61.0
    assert gw.complete(req("q3")).provider == "a"   # window has rolled over


def test_all_failed_raises():
    gw = LLMGateway({Task.TEXT: [FakeProvider("a", fail_times=99, retryable=False)]}, sleep=lambda s: None)
    with pytest.raises(AllProvidersFailedError):
        gw.complete(req())


def test_restricted_content_never_reaches_hosted_provider():
    hosted = FakeProvider("hosted")
    gw = LLMGateway({Task.TEXT: [hosted]}, sleep=lambda s: None)
    with pytest.raises(RestrictedContentError):
        gw.complete(req(sensitivity=Sensitivity.RESTRICTED))
    assert hosted.calls == 0


def test_restricted_content_goes_to_local_model_when_configured():
    hosted, local = FakeProvider("hosted"), FakeProvider("local", hosted=False)
    gw = LLMGateway({Task.TEXT: [hosted]}, local_fallback=local, sleep=lambda s: None)
    assert gw.complete(req(sensitivity=Sensitivity.RESTRICTED)).provider == "local"
    assert hosted.calls == 0


def test_prompt_is_redacted_before_hosted_call():
    p = FakeProvider("a")
    gw = LLMGateway({Task.TEXT: [p]}, sleep=lambda s: None)
    resp = gw.complete(req("Contact ABCDE1234F about revenue"))
    assert "ABCDE1234F" not in p.prompts[0] and resp.redactions.get("PAN") == 1


def test_mock_provider_runs_offline():
    gw = LLMGateway({Task.TEXT: [MockProvider()]})
    assert '"mock": true' in gw.complete(req()).text
