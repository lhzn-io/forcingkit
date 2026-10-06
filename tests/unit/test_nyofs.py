"""Unit tests for NYOFS fetcher."""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, patch

from forcingkit.fetchers import nyofs


def _nowcast_ds(cycle: pd.Timestamp, jitter_s: float = 14.06) -> xr.Dataset:
    """A synthetic nowcast file: 6 hourly records up to `cycle`, times jittered as NYOFS does."""
    hours = pd.date_range(cycle - pd.Timedelta(hours=5), cycle, freq="h")
    times = hours + pd.to_timedelta(
        [0, -jitter_s, jitter_s, 0, -jitter_s, jitter_s], unit="s"
    )
    ny, nx = 3, 4
    return xr.Dataset(
        data_vars={
            "u": (("time", "sigma", "ny", "nx"), np.ones((6, 2, ny, nx))),
            "v": (("time", "sigma", "ny", "nx"), np.ones((6, 2, ny, nx))),
            "lon": (("ny", "nx"), np.linspace(-74.0, -73.8, ny * nx).reshape(ny, nx)),
            "lat": (("ny", "nx"), np.linspace(40.6, 40.8, ny * nx).reshape(ny, nx)),
            "mask": (("ny", "nx"), np.ones((ny, nx))),
        },
        coords={"time": times, "sigma": np.linspace(0, 1, 2)},
    )


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

    def test_domain_bbox_matches_coarse_grid(self):
        """The declared domain is the coarse grid extent, not a wider margin."""
        assert nyofs.get_metadata()["domain_bbox"] == [-74.475, 40.389, -73.743, 40.940]

    def test_supports_bbox_within_domain(self):
        """Test bbox validation for NY Harbor."""
        # Throgs Neck Bridge area, inside the NYOFS grid
        bbox = [-73.815, 40.785, -73.775, 40.815]
        assert nyofs.supports_bbox(bbox)

    def test_supports_bbox_upper_bay(self):
        assert nyofs.supports_bbox([-74.10, 40.60, -74.00, 40.70])

    def test_rejects_western_long_island_sound(self):
        """East of the grid's edge (73.74 W) NYOFS has no cells, so it must not claim the box."""
        bbox = [-73.70, 40.80, -73.50, 40.90]
        assert not nyofs.supports_bbox(bbox)

    def test_rejects_south_of_grid(self):
        """The grid stops at 40.39 N; a box in the outer NY Bight is not NYOFS's."""
        bbox = [-74.0, 40.25, -73.9, 40.35]
        assert not nyofs.supports_bbox(bbox)

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
    """Access mode: FMRC for the past 6 days, the nowcast archive before that."""

    def test_recent_returns_fmrc(self):
        target_dt = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=2)
        mode, url = nyofs._get_nyofs_url(target_dt)
        assert mode == "fmrc"
        assert "Aggregated_7_day_NYOFS_Fields_Forecast_best" in url

    @pytest.mark.parametrize("date", ["2026-09-15", "2024-06-15", "2023-12-15"])
    def test_older_returns_archive(self, date):
        mode, url = nyofs._get_nyofs_url(pd.Timestamp(date))
        assert mode == "archive"
        assert url is None


class TestNowcastCycles:
    """Hour-to-file mapping: cycle tCCz (05/11/17/23 UTC) holds hours CC-5 to CC."""

    @pytest.mark.parametrize(
        "hour, cycle_hour",
        [(0, 5), (5, 5), (6, 11), (11, 11), (12, 17), (17, 17), (18, 23), (23, 23)],
    )
    def test_nowcast_cycle(self, hour, cycle_hour):
        hour_dt = pd.Timestamp("2026-09-15") + pd.Timedelta(hours=hour)
        assert nyofs._nowcast_cycle(hour_dt) == pd.Timestamp(
            "2026-09-15"
        ) + pd.Timedelta(hours=cycle_hour)

    def test_single_cycle_window(self):
        cycles = nyofs._nowcast_cycles(
            pd.Timestamp("2026-09-15 12:00"), pd.Timestamp("2026-09-15 15:00")
        )
        assert cycles == [pd.Timestamp("2026-09-15 17:00")]

    def test_window_across_midnight(self):
        cycles = nyofs._nowcast_cycles(
            pd.Timestamp("2026-09-15 22:00"), pd.Timestamp("2026-09-16 07:00")
        )
        assert cycles == [
            pd.Timestamp("2026-09-15 23:00"),
            pd.Timestamp("2026-09-16 05:00"),
            pd.Timestamp("2026-09-16 11:00"),
        ]

    def test_full_day_needs_four_files_plus_end(self):
        cycles = nyofs._nowcast_cycles(
            pd.Timestamp("2026-09-15 00:00"), pd.Timestamp("2026-09-15 23:00")
        )
        assert [c.hour for c in cycles] == [5, 11, 17, 23]


class TestNowcastFileUrls:
    """File names and locations for each archive era."""

    def test_aws_per_day_then_ncei(self):
        urls = nyofs._nowcast_file_urls(pd.Timestamp("2026-09-15 11:00"))
        assert urls == [
            "s3://noaa-nos-ofs-pds/nyofs/netcdf/2026/09/15/"
            "nyofs.t11z.20260915.fields.nowcast.nc",
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/2026/09/"
            "nyofs.t11z.20260915.fields.nowcast.nc",
        ]

    def test_first_aws_per_day_date(self):
        urls = nyofs._nowcast_file_urls(pd.Timestamp("2024-11-19 23:00"))
        assert urls[0].startswith("s3://noaa-nos-ofs-pds/nyofs/netcdf/2024/11/19/")
        assert len(urls) == 2

    def test_before_aws_per_day_uses_ncei_only(self):
        urls = nyofs._nowcast_file_urls(pd.Timestamp("2024-11-18 23:00"))
        assert urls == [
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/2024/11/"
            "nyofs.t23z.20241118.fields.nowcast.nc"
        ]

    def test_ncei_new_names_from_2024_09_09(self):
        urls = nyofs._nowcast_file_urls(pd.Timestamp("2024-09-09 05:00"))
        assert urls == [
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/2024/09/"
            "nyofs.t05z.20240909.fields.nowcast.nc"
        ]

    def test_ncei_old_names_before_2024_09_09(self):
        urls = nyofs._nowcast_file_urls(pd.Timestamp("2024-09-08 23:00"))
        assert urls == [
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/2024/09/"
            "nos.nyofs.fields.nowcast.20240908.t23z.nc"
        ]

    def test_ncei_old_names_2023(self):
        urls = nyofs._nowcast_file_urls(pd.Timestamp("2023-12-15 05:00"))
        assert urls == [
            "https://www.ncei.noaa.gov/thredds/dodsC/model-nyofs-files/2023/12/"
            "nos.nyofs.fields.nowcast.20231215.t05z.nc"
        ]


class TestOpenArchiveFile:
    """Engine choice: AWS copies are netCDF3, NCEI is read over OPeNDAP."""

    @patch("forcingkit.fetchers.nyofs.xr.open_dataset")
    @patch("fsspec.open")
    def test_s3_uses_scipy_engine(self, mock_fsspec_open, mock_xr_open):
        mock_fsspec_open.return_value = MagicMock()
        nyofs._open_archive_file("s3://noaa-nos-ofs-pds/nyofs/netcdf/x.nc")
        assert mock_xr_open.call_args.kwargs["engine"] == "scipy"
        assert mock_fsspec_open.call_args.kwargs["anon"] is True

    @patch("forcingkit.fetchers.nyofs.xr.open_dataset")
    def test_ncei_uses_pydap(self, mock_xr_open):
        nyofs._open_archive_file("https://www.ncei.noaa.gov/thredds/dodsC/x.nc")
        args, kwargs = mock_xr_open.call_args
        assert args[0] == "dap2://www.ncei.noaa.gov/thredds/dodsC/x.nc"
        assert kwargs["engine"] == "pydap"

    @patch("forcingkit.fetchers.nyofs._open_archive_file")
    def test_falls_back_to_next_candidate(self, mock_open):
        ds = xr.Dataset()
        mock_open.side_effect = [FileNotFoundError("not on AWS"), ds]
        assert nyofs._open_first_available(["s3://a", "https://b"]) is ds

    @patch("forcingkit.fetchers.nyofs._open_archive_file")
    def test_all_candidates_missing(self, mock_open):
        mock_open.side_effect = FileNotFoundError("missing")
        assert nyofs._open_first_available(["s3://a", "https://b"]) is None


class TestOpenNYOFSDataset:
    """Test _open_nyofs_dataset handles real-world FMRC quirks and the archive chain."""

    @patch("forcingkit.fetchers.nyofs.xr.open_dataset")
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

    @patch("forcingkit.fetchers.nyofs._nowcast_file_urls", lambda c: [str(c)])
    @patch("forcingkit.fetchers.nyofs._open_first_available")
    def test_archive_concatenates_and_rounds_times(self, mock_open):
        mock_open.side_effect = lambda urls: _nowcast_ds(pd.Timestamp(urls[0]))
        start, end = pd.Timestamp("2026-09-15 04:00"), pd.Timestamp("2026-09-15 07:00")

        result = nyofs._open_nyofs_dataset("archive", None, start, end)

        assert result is not None
        assert list(pd.DatetimeIndex(result.time.values)) == list(
            pd.date_range(start, end, freq="h")
        )
        assert mock_open.call_count == 2  # t05z and t11z

    @patch("forcingkit.fetchers.nyofs._open_first_available")
    def test_archive_missing_cycle_returns_none(self, mock_open):
        """A gap in the archive falls back to the next donor rather than a short store."""
        mock_open.side_effect = [_nowcast_ds(pd.Timestamp("2026-09-15 05:00")), None]
        result = nyofs._open_nyofs_dataset(
            "archive",
            None,
            pd.Timestamp("2026-09-15 04:00"),
            pd.Timestamp("2026-09-15 07:00"),
        )
        assert result is None


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


class TestBoundaryConditions:
    """Test OBC fetcher logic (mocked data access)."""

    @patch("forcingkit.fetchers.nyofs._open_nyofs_dataset")
    def test_fetch_nyofs_obc_success(self, mock_open):
        """Test successful OBC fetch with mocked data."""
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

    @patch("forcingkit.fetchers.nyofs._open_nyofs_dataset")
    def test_velocities_pass_through_unaveraged(self, mock_open):
        """u, v are co-located with lon/lat; the fetcher must not average neighbours."""
        ds = _nowcast_ds(pd.Timestamp("2026-09-15 05:00"), jitter_s=0)
        ds["u"].values[:] = np.arange(12, dtype=float).reshape(3, 4)
        ds["v"].values[:] = np.arange(12, dtype=float).reshape(3, 4)[:, ::-1]
        mock_open.return_value = ds

        result = nyofs.fetch_nyofs_boundary_conditions(
            "2026-09-15T00:00:00Z", 5, [-74.0, 40.6, -73.8, 40.8]
        )

        assert result is not None
        np.testing.assert_array_equal(result.u.values[0, 0], ds.u.values[0, 0])
        np.testing.assert_array_equal(result.v.values[0, 0], ds.v.values[0, 0])

    @patch("forcingkit.fetchers.nyofs._open_nyofs_dataset")
    def test_land_and_out_of_box_cells_are_nan(self, mock_open):
        ds = _nowcast_ds(pd.Timestamp("2026-09-15 05:00"), jitter_s=0)
        ds["mask"].values[0, 0] = 0
        mock_open.return_value = ds

        result = nyofs.fetch_nyofs_boundary_conditions(
            "2026-09-15T00:00:00Z", 5, [-74.0, 40.6, -73.8, 40.8]
        )

        assert np.isnan(result.u.values[:, :, 0, 0]).all()
        assert np.isfinite(result.u.values[:, :, 1, 1]).all()

    @patch("forcingkit.fetchers.nyofs._open_nyofs_dataset")
    def test_fmrc_duplicate_hours_keep_the_record_with_data(self, mock_open):
        """FMRC repeats some hours; one copy can be all fill and must not win."""
        ds = _nowcast_ds(pd.Timestamp("2026-10-04 05:00"), jitter_s=0)
        dup = ds.isel(time=[1])
        dup = dup.assign_coords(time=dup.time + np.timedelta64(14, "s"))
        dup["u"].values[:] = np.nan
        dup["v"].values[:] = np.nan
        combined = xr.concat(
            [ds.isel(time=[0, 1]), dup, ds.isel(time=slice(2, None))],
            "time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
        mock_open.return_value = nyofs._round_to_hour(combined, "time")

        result = nyofs.fetch_nyofs_boundary_conditions(
            "2026-10-04T00:00:00Z", 5, [-74.0, 40.6, -73.8, 40.8]
        )

        assert result.sizes["time"] == 6
        assert np.isfinite(result.u.values).all()

    def test_fetch_nyofs_obc_outside_bbox(self):
        """Test that OBC fetch returns None for out-of-bbox request."""
        bbox = [-70.0, 43.0, -69.0, 44.0]  # Gulf of Maine
        start_date = "2024-10-15T12:00:00Z"

        result = nyofs.fetch_nyofs_boundary_conditions(start_date, 3, bbox)

        assert result is None

    @patch("forcingkit.fetchers.nyofs.xr.open_dataset")
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
        from forcingkit.dispatcher import get_obc_fetchers

        obc_fetchers = get_obc_fetchers()
        fetcher_ids = [m.get_metadata()["id"] for m, _ in obc_fetchers]

        assert "nyofs" in fetcher_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
