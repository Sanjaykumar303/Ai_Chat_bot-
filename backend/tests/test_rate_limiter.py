# services/rate_limiter.py's enforce_rate_limit() only ever reads
# request.headers and request.client.host, so a minimal stub is enough
# to exercise it without spinning up a real FastAPI app/TestClient.

import pytest
from fastapi import HTTPException

from services import rate_limiter


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host="1.2.3.4", forwarded_for=None):
        self.client = _FakeClient(host)
        self.headers = {"x-forwarded-for": forwarded_for} if forwarded_for else {}


@pytest.fixture(autouse=True)
def _reset_rate_limiter_state():
    # Each test gets a clean slate - this module-level dict is shared
    # global state (see rate_limiter.py's own docstring for why), so
    # tests would otherwise interfere with each other's request counts.
    rate_limiter._requests_by_client.clear()
    yield
    rate_limiter._requests_by_client.clear()


def test_requests_under_the_limit_all_pass(monkeypatch):
    monkeypatch.setattr(rate_limiter, "RATE_LIMIT_MAX_REQUESTS", 3)
    request = _FakeRequest()

    for _ in range(3):
        rate_limiter.enforce_rate_limit(request)  # should not raise


def test_request_over_the_limit_raises_429(monkeypatch):
    monkeypatch.setattr(rate_limiter, "RATE_LIMIT_MAX_REQUESTS", 2)
    request = _FakeRequest()

    rate_limiter.enforce_rate_limit(request)
    rate_limiter.enforce_rate_limit(request)

    with pytest.raises(HTTPException) as excinfo:
        rate_limiter.enforce_rate_limit(request)

    assert excinfo.value.status_code == 429
    assert "Retry-After" in excinfo.value.headers


def test_different_clients_are_tracked_independently(monkeypatch):
    monkeypatch.setattr(rate_limiter, "RATE_LIMIT_MAX_REQUESTS", 1)
    request_a = _FakeRequest(host="1.1.1.1")
    request_b = _FakeRequest(host="2.2.2.2")

    rate_limiter.enforce_rate_limit(request_a)
    rate_limiter.enforce_rate_limit(request_b)  # different client - should not raise

    with pytest.raises(HTTPException):
        rate_limiter.enforce_rate_limit(request_a)


def test_x_forwarded_for_takes_priority_over_direct_client_host():
    request_a = _FakeRequest(host="1.1.1.1", forwarded_for="9.9.9.9")
    request_b = _FakeRequest(host="2.2.2.2", forwarded_for="9.9.9.9")

    assert rate_limiter._client_key(request_a) == rate_limiter._client_key(request_b) == "9.9.9.9"


def test_expired_entries_are_evicted_not_just_ignored(monkeypatch):
    # A fake, explicitly-advanced clock instead of a real elapsed-time
    # sleep - two real time.monotonic() calls this close together can
    # land on the same tick (Windows' clock resolution isn't
    # sub-millisecond), which would make this test flaky rather than
    # actually prove eviction happened.
    fake_now = [1000.0]
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: fake_now[0])
    monkeypatch.setattr(rate_limiter, "RATE_LIMIT_WINDOW_SECONDS", 10)
    request = _FakeRequest()

    rate_limiter.enforce_rate_limit(request)
    assert len(rate_limiter._requests_by_client["1.2.3.4"]) == 1

    fake_now[0] += 20  # well past the window
    rate_limiter.enforce_rate_limit(request)
    # The old (now-expired) timestamp was swept, not just ignored -
    # only the new one remains.
    assert len(rate_limiter._requests_by_client["1.2.3.4"]) == 1
