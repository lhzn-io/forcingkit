import re
import sys

import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, "service")
from forcingkit_serve.main import app  # noqa: E402
from forcingkit_serve.routers.removed import REMOVED_ROUTES  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", sorted(REMOVED_ROUTES))
async def test_removed_route_answers_gone_with_replacement(method, path):
    url = re.sub(r"\{[^}]+\}", "x", path)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.request(method, url, json={})
    assert response.status_code == 410
    body = response.json()
    assert body["status"] == "gone"
    assert body["replacement"] == REMOVED_ROUTES[(method, path)]
    assert response.headers["Deprecation"] == "@1791158400"
    assert response.headers["Sunset"] == "Mon, 05 Oct 2026 00:00:00 GMT"
    assert 'rel="sunset"' in response.headers["Link"]


def test_removed_routes_are_marked_deprecated_in_openapi():
    paths = app.openapi()["paths"]
    for method, path in REMOVED_ROUTES:
        assert paths[path][method.lower()]["deprecated"] is True
