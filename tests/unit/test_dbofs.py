"""Unit tests for DBOFS fetcher."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch

from ecodata_cache.fetchers import dbofs


class TestDBOFSMetadata:
    """Test DBOFS metadata and domain support."""

    def test_get_metadata_has_required_keys(self):
        meta = dbofs.get_metadata()
        assert meta["id"] == "dbofs"
        assert meta["name"] == "NOAA DBOFS (Delaware Bay / Offshore NJ)"
        assert meta["resolution_approx_m"] == 100.0
        assert "domain_bbox" in meta
        assert len(meta["domain_bbox"]) == 4

    def test_domain_bbox_values(self):
        """Domain must cover offshore NJ (39.7-39.9°N, ~73.8-74.0°W)."""
        meta = dbofs.get_metadata()
        min_lon, min_lat, max_lon, max_lat = meta["domain_bbox"]
        assert min_lon <= -73.96
        assert max_lon >= -73.78
        assert min_lat <= 39.72
        assert max_lat >= 39.88

    def test_rejects_bbox_mid_atlantic_bight(self):
        """Mid-Atlantic Bight offshore NJ bbox MUST be rejected because it misses the hydro mask."""
        bbox = [-73.96, 39.72, -73.78, 39.88]
        assert not dbofs.supports_bbox(bbox)

    def test_supports_bbox_delaware_bay(self):
        """Delaware Bay bbox must be supported."""
        bbox = [-75.5, 38.5, -74.5, 39.5]
        assert dbofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_north(self):
        """Bbox north of 40°N (NY Harbor) must be rejected."""
        bbox = [-74.0, 40.2, -73.5, 40.8]
        assert not dbofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_east(self):
        """Bbox east of -73.0°W must be rejected."""
        bbox = [-73.1, 39.0, -72.5, 39.5]
        assert not dbofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_gulf_of_maine(self):
        """Gulf of Maine bbox must be rejected."""
        bbox = [-70.0, 43.0, -69.0, 44.0]
        assert not dbofs.supports_bbox(bbox)

    def test_domain_area_smaller_than_necofs(self):
        """DBOFS domain area must be much smaller than NECOFS (132°²)."""
        meta = dbofs.get_metadata()
        min_lon, min_lat, max_lon, max_lat = meta["domain_bbox"]
        area = (max_lon - min_lon) * (max_lat - min_lat)
        assert area < 20.0, f"DBOFS domain area {area:.2f}°² should be < 20°²"


class TestURLResolution:
    """Test URL resolution logic for FMRC vs NCEI."""

    def test_get_url_recent_returns_fmrc(self):
        target_dt = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=5)
        mode, url = dbofs._get_dbofs_url(target_dt)

        assert mode == "fmrc"
        assert "Aggregated_7_day_DBOFS_Fields_Forecast_best" in url
        assert "opendap.co-ops.nos.noaa.gov" in url

    def test_get_url_historical_returns_aws(self):
        target_dt = pd.Timestamp("2024-01-15", tz="UTC").tz_localize(None)
        mode, url = dbofs._get_dbofs_url(target_dt)

        assert mode == "aws_s3"

    def test_get_url_post_sept_2024_naming(self):
        target_dt = pd.Timestamp("2024-10-01", tz="UTC").tz_localize(None)
        mode, url = dbofs._get_dbofs_url(target_dt)

        assert mode == "aws_s3"

    def test_get_url_pre_sept_2024_naming(self):
        target_dt = pd.Timestamp("2024-08-01", tz="UTC").tz_localize(None)
        mode, url = dbofs._get_dbofs_url(target_dt)

        assert mode == "aws_s3"


class TestEnumerateNCEIFiles:
    """Test NCEI file enumeration."""

    def test_enumerate_single_hour(self):
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files/"
            "{yyyy}/{mm}/{dd}/dbofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        start_dt = pd.Timestamp("2024-10-15 12:30:00")
        end_dt = start_dt + pd.Timedelta(hours=1)

        files = dbofs._enumerate_ncei_dbofs_files(pattern, start_dt, end_dt)

        assert len(files) >= 1
        assert "2024" in files[0]

    def test_enumerate_multiple_hours(self):
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files/"
            "{yyyy}/{mm}/{dd}/dbofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        start_dt = pd.Timestamp("2024-10-15 06:00:00")
        end_dt = start_dt + pd.Timedelta(hours=6)

        files = dbofs._enumerate_ncei_dbofs_files(pattern, start_dt, end_dt)

        assert len(files) >= 6


class TestCGridInterpolation:
    """Test shared C-grid interpolation (same logic as NYOFS)."""

    def test_c_grid_to_rho_shape_preserved(self):
        nk, neta, nxi = 3, 8, 10
        u = np.ones((nk, neta, nxi))
        v = np.ones((nk, neta, nxi))

        u_rho, v_rho = dbofs._c_grid_to_rho(u, v)

        assert u_rho.shape == (nk, neta, nxi)
        assert v_rho.shape == (nk, neta, nxi)

    def test_c_grid_to_rho_uniform_field_unchanged(self):
        u = np.full((2, 5, 7), 0.8)
        v = np.full((2, 5, 7), -0.4)

        u_rho, v_rho = dbofs._c_grid_to_rho(u, v)

        np.testing.assert_allclose(u_rho, u)
        np.testing.assert_allclose(v_rho, v)

    def test_c_grid_to_rho_obc_4d_shape(self):
        nt, nk, neta, nxi = 4, 3, 6, 8
        u = np.random.rand(nt, nk, neta, nxi)
        v = np.random.rand(nt, nk, neta, nxi)

        u_rho, v_rho = dbofs._c_grid_to_rho(u, v)

        assert u_rho.shape == (nt, nk, neta, nxi)
        assert v_rho.shape == (nt, nk, neta, nxi)


class TestResolveVar:
    """Test variable name resolution."""

    def test_resolves_canonical_names(self):
        ds = xr.Dataset(
            {
                "u": (("eta",), [1.0]),
                "temp": (("eta",), [15.0]),
                "salt": (("eta",), [32.0]),
            }
        )
        assert dbofs._resolve_var(ds, "u") == "u"
        assert dbofs._resolve_var(ds, "temp") == "temp"
        assert dbofs._resolve_var(ds, "salt") == "salt"

    def test_resolves_roms_alternate_names(self):
        ds = xr.Dataset(
            {
                "water_u": (("eta",), [1.0]),
                "water_temp": (("eta",), [15.0]),
                "salinity": (("eta",), [32.0]),
                "sea_surface_height": (("eta",), [0.1]),
            }
        )
        assert dbofs._resolve_var(ds, "u") == "water_u"
        assert dbofs._resolve_var(ds, "temp") == "water_temp"
        assert dbofs._resolve_var(ds, "salt") == "salinity"
        assert dbofs._resolve_var(ds, "zeta") == "sea_surface_height"

    def test_raises_on_missing_var(self):
        ds = xr.Dataset({"unknown": (("eta",), [1.0])})
        with pytest.raises(KeyError, match="No variable found for role 'u'"):
            dbofs._resolve_var(ds, "u")


class TestBoundaryConditions:
    """Test OBC fetcher logic (mocked OPeNDAP)."""

    @patch("ecodata_cache.fetchers.dbofs._open_dbofs_dataset")
    def test_fetch_dbofs_obc_success(self, mock_open):
        """Successful OBC fetch returns dataset with correct dims."""
        nt, nk, neta, nxi = 6, 10, 20, 25
        times = pd.date_range("2026-03-02T05:00:00", periods=nt, freq="1h")

        mock_ds = xr.Dataset(
            data_vars={
                "u": (
                    ("time", "s_rho", "eta_rho", "xi_rho"),
                    np.random.rand(nt, nk, neta, nxi).astype(np.float32),
                ),
                "v": (
                    ("time", "s_rho", "eta_rho", "xi_rho"),
                    np.random.rand(nt, nk, neta, nxi).astype(np.float32),
                ),
                "lon_rho": (
                    ("eta_rho", "xi_rho"),
                    np.linspace(-75.0, -74.0, neta * nxi).reshape(neta, nxi),
                ),
                "lat_rho": (
                    ("eta_rho", "xi_rho"),
                    np.linspace(38.5, 39.5, neta * nxi).reshape(neta, nxi),
                ),
                "mask_rho": (("eta_rho", "xi_rho"), np.ones((neta, nxi))),
            },
            coords={"time": times, "s_rho": np.linspace(-1, 0, nk)},
        )
        mock_open.return_value = mock_ds

        result = dbofs.fetch_dbofs_boundary_conditions(
            "2026-03-02T05:00:00Z", 6, [-75.0, 38.5, -74.0, 39.5]
        )

        assert result is not None
        assert "u" in result.data_vars
        assert "v" in result.data_vars
        assert set(result.dims) == {"time", "depth", "eta", "xi"}
        assert result.u.dtype == np.float32
        assert result.sizes["time"] == nt

    def test_fetch_dbofs_obc_outside_bbox(self):
        """OBC fetch returns None for out-of-domain bbox."""
        result = dbofs.fetch_dbofs_boundary_conditions(
            "2026-03-02T05:00:00Z", 6, [-70.0, 43.0, -69.0, 44.0]
        )
        assert result is None

    @patch("ecodata_cache.fetchers.dbofs._open_dbofs_dataset")
    def test_fetch_dbofs_obc_pydap_error(self, mock_open):
        """OBC fetch returns None when dataset open fails."""
        mock_open.return_value = None

        result = dbofs.fetch_dbofs_boundary_conditions(
            "2026-03-02T05:00:00Z", 6, [-75.0, 38.5, -74.0, 39.5]
        )

        assert result is None


class TestDispatcherRegistration:
    """Test DBOFS registration in the dispatcher."""

    def test_dbofs_in_obc_fetchers(self):
        """DBOFS must appear in OBC fetchers."""
        from ecodata_cache.dispatcher import get_obc_fetchers

        ids = [m.get_metadata()["id"] for m, _ in get_obc_fetchers()]
        assert "dbofs" in ids

    def test_dbofs_ranks_above_necofs_for_obc_delaware_bay(self):
        """DBOFS must outrank NECOFS for OBC on the delaware bay bbox."""
        from ecodata_cache.dispatcher import _rank_obc_candidates

        bbox = [-75.5, 38.5, -74.5, 39.5]
        ranked = _rank_obc_candidates(bbox)

        assert len(ranked) > 0
        ids = [r[2]["id"] for r in ranked]
        assert ids.index("dbofs") < ids.index("necofs"), (
            f"DBOFS should rank before NECOFS in OBC. Order: {ids}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
