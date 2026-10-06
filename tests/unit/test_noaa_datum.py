"""CO-OPS water level: NAVD88 where a station has it, MSL where it does not."""

from unittest import mock

from forcingkit.fetchers import noaa


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.content = b"x"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fetch(tmp_path, responses):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["datum"])
        return responses[params["datum"]]

    with mock.patch.object(noaa.requests, "get", side_effect=fake_get):
        data = noaa.fetch_noaa_tide_data(
            "8465705",
            "2026-04-02T00:00:00",
            "2026-04-02T04:00:00",
            cache_dir=str(tmp_path),
        )
    return data, calls


def test_falls_back_to_msl_when_navd_is_unsupported(tmp_path):
    rows = {
        "metadata": {"id": "8465705"},
        "data": [{"t": "2026-04-02 00:00", "v": "0.1"}],
    }
    data, calls = _fetch(
        tmp_path,
        {
            "NAVD": FakeResponse(
                {
                    "error": {
                        "message": " The supported Datum values are: MHHW, MHW, MTL, MSL"
                    }
                },
                status=400,
            ),
            "MSL": FakeResponse(rows),
        },
    )
    assert calls == ["NAVD", "MSL"]
    assert data["metadata"]["datum"] == "MSL"


def test_uses_navd_when_available(tmp_path):
    rows = {"metadata": {"id": "8516945"}, "data": []}
    data, calls = _fetch(tmp_path, {"NAVD": FakeResponse(rows)})
    assert calls == ["NAVD"]
    assert data["metadata"]["datum"] == "NAVD"
