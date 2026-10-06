"""Unit tests for DBOFS fetcher."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, patch

from forcingkit.fetchers import dbofs


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
    """Access mode: FMRC for the past 6 days, the nowcast archive before that."""

    def test_get_url_recent_returns_fmrc(self):
        target_dt = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=5)
        mode, url = dbofs._get_dbofs_url(target_dt)

        assert mode == "fmrc"
        assert "Aggregated_7_day_DBOFS_Fields_Forecast_best" in url
        assert "opendap.co-ops.nos.noaa.gov" in url

    @pytest.mark.parametrize("date", ["2026-09-15", "2024-06-15", "2023-12-15"])
    def test_older_returns_archive(self, date):
        mode, url = dbofs._get_dbofs_url(pd.Timestamp(date))
        assert mode == "archive"
        assert url is None


class TestNowcastFiles:
    """Hour-to-file mapping: file n001 to n006 of cycle tCCz hold hours CC-5 to CC."""

    @pytest.mark.parametrize(
        "hour, cycle, n",
        [
            ("2026-09-15 00:00", "2026-09-15 00:00", 6),
            ("2026-09-15 01:00", "2026-09-15 06:00", 1),
            ("2026-09-15 06:00", "2026-09-15 06:00", 6),
            ("2026-09-15 07:30", "2026-09-15 12:00", 1),
            ("2026-09-15 19:00", "2026-09-16 00:00", 1),
            ("2026-09-15 23:00", "2026-09-16 00:00", 5),
        ],
    )
    def test_nowcast_file(self, hour, cycle, n):
        assert dbofs._nowcast_file(pd.Timestamp(hour)) == (pd.Timestamp(cycle), n)

    def test_aws_per_day_then_ncei(self):
        urls = dbofs._nowcast_file_urls(pd.Timestamp("2026-09-15 04:00"))
        assert urls == [
            "s3://noaa-nos-ofs-pds/dbofs/netcdf/2026/09/15/"
            "dbofs.t06z.20260915.fields.n004.nc",
            "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files/2026/09/"
            "dbofs.t06z.20260915.fields.n004.nc",
        ]

    def test_hour_in_next_days_cycle_uses_that_days_directory(self):
        urls = dbofs._nowcast_file_urls(pd.Timestamp("2026-09-30 21:00"))
        assert urls[0] == (
            "s3://noaa-nos-ofs-pds/dbofs/netcdf/2026/10/01/"
            "dbofs.t00z.20261001.fields.n003.nc"
        )

    def test_before_aws_per_day_uses_ncei_only(self):
        urls = dbofs._nowcast_file_urls(pd.Timestamp("2024-10-15 12:00"))
        assert urls == [
            "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files/2024/10/"
            "dbofs.t12z.20241015.fields.n006.nc"
        ]

    def test_ncei_new_names_from_2024_09_09(self):
        urls = dbofs._nowcast_file_urls(pd.Timestamp("2024-09-09 01:00"))
        assert urls == [
            "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files/2024/09/"
            "dbofs.t06z.20240909.fields.n001.nc"
        ]

    def test_ncei_old_names_before_2024_09_09(self):
        urls = dbofs._nowcast_file_urls(pd.Timestamp("2023-12-15 06:00"))
        assert urls == [
            "https://www.ncei.noaa.gov/thredds/dodsC/model-dbofs-files/2023/12/"
            "nos.dbofs.fields.n006.20231215.t06z.nc"
        ]


def _hourly_file(hour: pd.Timestamp) -> xr.Dataset:
    """A synthetic ROMS fields file holding one record at `hour`."""
    return xr.Dataset(
        data_vars={
            "u": (("ocean_time", "s_rho", "eta_u", "xi_u"), np.ones((1, 2, 3, 3))),
            "lon_rho": (("eta_rho", "xi_rho"), np.zeros((3, 4))),
        },
        coords={"ocean_time": [hour]},
    )


class TestOpenArchive:
    """The archive path opens one file per hour and joins them along ocean_time."""

    @patch("forcingkit.fetchers.dbofs._nowcast_file_urls", lambda h: [str(h)])
    @patch("forcingkit.fetchers.dbofs._open_first_available")
    def test_concatenates_along_ocean_time(self, mock_open):
        mock_open.side_effect = lambda urls: _hourly_file(pd.Timestamp(urls[0]))
        start, end = pd.Timestamp("2026-09-15 04:00"), pd.Timestamp("2026-09-15 07:00")

        result = dbofs._open_dbofs_dataset("archive", None, start, end)

        assert result is not None
        assert result.u.dims == ("ocean_time", "s_rho", "eta_u", "xi_u")
        assert list(pd.DatetimeIndex(result.ocean_time.values)) == list(
            pd.date_range(start, end, freq="h")
        )
        assert result.lon_rho.dims == ("eta_rho", "xi_rho")

    @patch("forcingkit.fetchers.dbofs._open_first_available")
    def test_missing_hour_returns_none(self, mock_open):
        mock_open.side_effect = [_hourly_file(pd.Timestamp("2026-09-15 04:00")), None]
        result = dbofs._open_dbofs_dataset(
            "archive",
            None,
            pd.Timestamp("2026-09-15 04:00"),
            pd.Timestamp("2026-09-15 05:00"),
        )
        assert result is None

    @patch("forcingkit.fetchers.dbofs.xr.open_dataset")
    @patch("fsspec.open")
    def test_s3_uses_h5netcdf(self, mock_fsspec_open, mock_xr_open):
        mock_fsspec_open.return_value = MagicMock()
        dbofs._open_archive_file("s3://noaa-nos-ofs-pds/dbofs/netcdf/x.nc")
        assert mock_xr_open.call_args.kwargs["engine"] == "h5netcdf"

    @patch("forcingkit.fetchers.dbofs._open_archive_file")
    def test_falls_back_to_ncei(self, mock_open):
        ds = xr.Dataset()
        mock_open.side_effect = [FileNotFoundError("not on AWS"), ds]
        assert dbofs._open_first_available(["s3://a", "https://b"]) is ds


class TestFillValues:
    def test_roms_fill_becomes_nan(self):
        out = dbofs._fill_to_nan(np.array([0.5, 1e37, -1e37, -0.2]))
        assert out.dtype == np.float32
        np.testing.assert_array_equal(np.isnan(out), [False, True, True, False])
        assert out[0] == np.float32(0.5)


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

    @patch("forcingkit.fetchers.dbofs._open_dbofs_dataset")
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

    @patch("forcingkit.fetchers.dbofs._open_dbofs_dataset")
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
        from forcingkit.dispatcher import get_obc_fetchers

        ids = [m.get_metadata()["id"] for m, _ in get_obc_fetchers()]
        assert "dbofs" in ids

    def test_dbofs_ranks_above_necofs_for_obc_delaware_bay(self):
        """DBOFS must outrank NECOFS for OBC on the delaware bay bbox."""
        from forcingkit.dispatcher import _rank_obc_candidates

        bbox = [-75.5, 38.5, -74.5, 39.5]
        ranked = _rank_obc_candidates(bbox)

        assert len(ranked) > 0
        ids = [r[2]["id"] for r in ranked]
        assert ids.index("dbofs") < ids.index("necofs"), (
            f"DBOFS should rank before NECOFS in OBC. Order: {ids}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
