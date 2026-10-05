"""Unit tests for NYOFS fetcher."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import patch

from ecodata_cache.fetchers import nyofs


class TestNYOFSMetadata:
    """Test NYOFS metadata and domain support."""

    def test_get_metadata_has_required_keys(self):
        """Test that metadata contains all required keys."""
        meta = nyofs.get_metadata()
        assert meta["id"] == "nyofs"
        assert meta["name"] == "NOAA NYOFS (NY/NJ Harbor)"
        assert meta["resolution_approx_m"] == 100.0
        assert meta["type_desc"] == "Structured curvilinear POM grid"
        assert "domain_bbox" in meta
        assert len(meta["domain_bbox"]) == 4

    def test_supports_bbox_within_domain(self):
        """Test bbox validation for NY Harbor."""
        # Throgs Neck Bridge area — well within NYOFS domain
        bbox = [-73.815, 40.785, -73.775, 40.815]
        assert nyofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_west(self):
        """Test bbox outside domain (too far west)."""
        bbox = [-75.0, 40.6, -74.5, 40.8]
        assert not nyofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_north(self):
        """Test bbox outside domain (too far north)."""
        bbox = [-73.8, 41.2, -73.5, 41.5]
        assert not nyofs.supports_bbox(bbox)

    def test_supports_bbox_outside_domain_gulf_of_maine(self):
        """Test bbox in Gulf of Maine (completely outside NYOFS)."""
        bbox = [-70.0, 43.0, -69.0, 44.0]
        assert not nyofs.supports_bbox(bbox)


class TestURLResolution:
    """Test URL resolution logic for FMRC vs NCEI vs AWS S3."""

    def test_get_url_recent_returns_aws_s3(self):
        """Test that data returns AWS S3 bucket keys instead of FMRC/NCEI."""
        target_dt = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=7)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "aws_s3"
        assert "noaa-nos-ofs-pds" in url

    def test_get_url_historical_returns_aws_s3(self):
        """Test that historical data (> 31 days) returns AWS S3 bucket keys instead of NCEI."""
        target_dt = pd.Timestamp("2024-01-15", tz="UTC").tz_localize(None)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "aws_s3"
        assert "noaa-nos-ofs-pds" in url

    def test_get_url_post_sept_2024_naming(self):
        """Test AWS naming convention after Sept 9, 2024."""
        target_dt = pd.Timestamp("2024-10-01", tz="UTC").tz_localize(None)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "aws_s3"

    def test_get_url_pre_sept_2024_naming(self):
        """Test AWS naming convention before Sept 9, 2024."""
        target_dt = pd.Timestamp("2024-08-01", tz="UTC").tz_localize(None)
        mode, url = nyofs._get_nyofs_url(target_dt)

        assert mode == "aws_s3"


class TestEnumerateNCEIFiles:
    """Test NCEI file enumeration."""

    def test_enumerate_ncei_single_hour(self):
        """Test file enumeration for single hour request."""
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/"
            "{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        start_dt = pd.Timestamp("2024-10-15 12:30:00")
        end_dt = start_dt + pd.Timedelta(hours=1)

        files = nyofs._enumerate_ncei_nyofs_files(pattern, start_dt, end_dt)

        assert len(files) >= 1
        assert "2024" in files[0]
        assert "10" in files[0]
        assert "15" in files[0]

    def test_enumerate_ncei_multiple_hours(self):
        """Test file enumeration across multiple hours."""
        pattern = (
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/"
            "{yyyy}/{mm}/{dd}/nyofs.t{cc}z.{yyyymmdd}.fields.{type}{hhh:03d}.nc"
        )
        start_dt = pd.Timestamp("2024-10-15 12:30:00")
        end_dt = start_dt + pd.Timedelta(hours=6)

        files = nyofs._enumerate_ncei_nyofs_files(pattern, start_dt, end_dt)

        assert len(files) >= 6


class TestCGridInterpolation:
    """Test C-grid velocity interpolation."""

    def test_c_grid_to_rho_ic_shape(self):
        """Output shape matches input for IC (sigma, eta, xi) arrays."""
        nk, neta, nxi = 2, 6, 8
        u = np.ones((nk, neta, nxi))
        v = np.ones((nk, neta, nxi))

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        assert u_rho.shape == (nk, neta, nxi)
        assert v_rho.shape == (nk, neta, nxi)

    def test_c_grid_to_rho_obc_shape(self):
        """Output shape matches input for OBC (time, sigma, eta, xi) arrays."""
        nt, nk, neta, nxi = 3, 2, 6, 8
        u = np.ones((nt, nk, neta, nxi))
        v = np.ones((nt, nk, neta, nxi))

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        assert u_rho.shape == (nt, nk, neta, nxi)
        assert v_rho.shape == (nt, nk, neta, nxi)

    def test_c_grid_to_rho_averages_adjacent_cells(self):
        """Interior cells are averages of adjacent face values."""
        nk, neta, nxi = 1, 4, 4
        # u[..., i] = float(i), so average of i and i+1
        u = np.tile(np.arange(nxi, dtype=float), (nk, neta, 1))
        v = np.tile(np.arange(neta, dtype=float)[:, None], (nk, 1, nxi))

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        # Interior: 0.5*(i + i+1)
        np.testing.assert_allclose(u_rho[..., 0], 0.5)  # 0.5*(0+1)
        np.testing.assert_allclose(u_rho[..., 1], 1.5)  # 0.5*(1+2)
        # Boundary: copy of last edge value
        np.testing.assert_allclose(u_rho[..., -1], u[..., -1])
        np.testing.assert_allclose(v_rho[..., -1, :], v[..., -1, :])

    def test_c_grid_to_rho_uniform_field_unchanged(self):
        """Uniform velocity field should be unchanged after averaging."""
        u = np.full((2, 5, 7), 1.5)
        v = np.full((2, 5, 7), -0.3)

        u_rho, v_rho = nyofs._c_grid_to_rho(u, v)

        np.testing.assert_allclose(u_rho, u)
        np.testing.assert_allclose(v_rho, v)


class TestResolveVar:
    """Test variable name resolution for alternate FMRC naming conventions."""

    def test_resolves_canonical_names(self):
        ds = xr.Dataset({"u": (("eta",), [1.0]), "temp": (("eta",), [15.0])})
        assert nyofs._resolve_var(ds, "u") == "u"
        assert nyofs._resolve_var(ds, "temp") == "temp"

    def test_resolves_water_prefixed_names(self):
        ds = xr.Dataset(
            {
                "water_u": (("eta",), [1.0]),
                "water_v": (("eta",), [0.5]),
                "water_temp": (("eta",), [15.0]),
                "salinity": (("eta",), [32.0]),
                "zeta": (("eta",), [0.1]),
            }
        )
        assert nyofs._resolve_var(ds, "u") == "water_u"
        assert nyofs._resolve_var(ds, "v") == "water_v"
        assert nyofs._resolve_var(ds, "temp") == "water_temp"
        assert nyofs._resolve_var(ds, "salt") == "salinity"

    def test_raises_on_missing_var(self):
        ds = xr.Dataset({"unknown_var": (("eta",), [1.0])})
        with pytest.raises(KeyError, match="No variable found for role 'u'"):
            nyofs._resolve_var(ds, "u")


class TestOpenNYOFSDataset:
    """Test _open_nyofs_dataset handles real-world FMRC quirks."""

    @patch("ecodata_cache.fetchers.nyofs.xr.open_dataset")
    def test_fmrc_non_monotonic_time_index(self, mock_xr_open):
        """FMRC dataset with non-monotonic time should still slice correctly."""
        # Simulate a shuffled FMRC time axis (observed in production)
        times_shuffled = pd.to_datetime(
            ["2026-03-31", "2026-03-29", "2026-03-30", "2026-04-01"]
        )
        mock_ds = xr.Dataset(
            data_vars={"u": (("time", "eta", "xi"), np.ones((4, 3, 3)))},
            coords={"time": times_shuffled},
        )
        mock_xr_open.return_value = mock_ds

        target_dt = pd.Timestamp("2026-03-30")
        end_dt = pd.Timestamp("2026-03-31")

        result = nyofs._open_nyofs_dataset("fmrc", "dap2://fake", target_dt, end_dt)

        assert result is not None
        assert result.sizes["time"] == 2  # 2026-03-30 and 2026-03-31


class TestBoundaryConditions:
    """Test OBC fetcher logic (mocked OPeNDAP)."""

    @patch("ecodata_cache.fetchers.nyofs._open_nyofs_dataset")
    def test_fetch_nyofs_obc_success(self, mock_open):
        """Test successful OBC fetch with mocked pydap."""
        # Create mock time-series dataset
        times = pd.date_range("2024-10-15", periods=3, freq="1h")
        mock_ds = xr.Dataset(
            data_vars={
                "u": (
                    ("time", "sigma", "eta", "xi"),
                    np.random.rand(3, 2, 3, 4) * 0.5,
                ),
                "v": (
                    ("time", "sigma", "eta", "xi"),
                    np.random.rand(3, 2, 3, 4) * 0.5,
                ),
                "lon": (("eta", "xi"), np.linspace(-74.0, -73.8, 12).reshape(3, 4)),
                "lat": (("eta", "xi"), np.linspace(40.6, 40.8, 12).reshape(3, 4)),
                "mask": (("eta", "xi"), np.ones((3, 4))),
            },
            coords={"time": times, "sigma": np.linspace(0, -1, 2)},
        )
        mock_open.return_value = mock_ds

        bbox = [-74.0, 40.6, -73.8, 40.8]
        start_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_boundary_conditions(start_date, 3, bbox)

        assert result is not None
        assert "u" in result.data_vars
        assert "v" in result.data_vars
        assert set(result.dims) == {"time", "depth", "eta", "xi"}
        assert result.u.dtype == np.float32

    def test_fetch_nyofs_obc_outside_bbox(self):
        """Test that OBC fetch returns None for out-of-bbox request."""
        bbox = [-70.0, 43.0, -69.0, 44.0]  # Gulf of Maine
        start_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_boundary_conditions(start_date, 3, bbox)

        assert result is None

    @patch("ecodata_cache.fetchers.nyofs.xr.open_dataset")
    def test_fetch_nyofs_obc_pydap_error(self, mock_xr_open):
        """Test OBC fetch gracefully handles pydap errors."""
        mock_xr_open.side_effect = Exception("OPeNDAP connection failed")

        bbox = [-74.0, 40.6, -73.8, 40.8]
        start_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_boundary_conditions(start_date, 6, bbox)

        # Should return None when pydap fails
        assert result is None


class TestDispatcherIntegration:
    """Test dispatcher registration."""

    def test_nyofs_in_obc_fetchers(self):
        """Test that NYOFS is registered in OBC fetchers."""
        from ecodata_cache.dispatcher import get_obc_fetchers

        obc_fetchers = get_obc_fetchers()
        fetcher_ids = [m.get_metadata()["id"] for m, _ in obc_fetchers]

        assert "nyofs" in fetcher_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
