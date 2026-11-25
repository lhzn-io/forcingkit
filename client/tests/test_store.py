"""CachingHTTPStore: chunk reads that fail in transport are retried on a new connection."""

import httpx
import pytest

from forcingkit_client.store import CachingHTTPStore


def _store(tmp_path, handler, attempts=3):
    store = CachingHTTPStore(
        tmp_path / "cache",
        "http://svc/api/v1/stores/s.zarr",
        attempts=attempts,
        retry_wait_s=0.0,
    )
    store._http = httpx.Client(transport=httpx.MockTransport(handler))
    return store


def test_read_that_times_out_is_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.ReadTimeout("no answer", request=request)
        return httpx.Response(200, content=b"chunk")

    store = _store(tmp_path, handler)
    assert store._fetch("u/0.0.0.0")
    assert (tmp_path / "cache" / "u" / "0.0.0.0").read_bytes() == b"chunk"
    assert calls == ["/api/v1/stores/s.zarr/u/0.0.0.0"] * 2


def test_read_gives_up_after_its_attempts(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    store = _store(tmp_path, handler, attempts=3)
    with pytest.raises(httpx.ConnectError):
        store._fetch("u/0.0.0.0")
    assert len(calls) == 3
    assert not (tmp_path / "cache" / "u" / "0.0.0.0").exists()


def test_missing_chunk_is_not_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404)

    store = _store(tmp_path, handler)
    assert not store._fetch("u/0.0.0.0")
    assert len(calls) == 1


def test_connections_are_not_kept_between_reads(tmp_path):
    store = CachingHTTPStore(tmp_path / "cache", "http://svc/s.zarr")
    pool = store._client()._transport._pool
    assert pool._max_keepalive_connections == 0
