import pytest
from httpx import AsyncClient, ASGITransport
import sys

# To enable importing src directly
sys.path.insert(0, "service")
from ecodata_serve.main import app


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "ecodata-cache"}
