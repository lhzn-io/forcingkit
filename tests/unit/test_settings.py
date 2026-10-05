import os

import pytest

from forcingkit import settings

ALL_NAMES = [
    "FORCINGKIT_CACHE_DIR",
    "ECODATA_CACHE_CACHE_DIR",
    "COASTAL_SIM_DATA_CACHE_DIR",
    "FORCINGKIT_MAX_WORKERS",
    "ECODATA_CACHE_MAX_WORKERS",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No forcingkit variables set, HOME in a temporary directory, warnings re-armed."""
    for name in ALL_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(settings, "_warned", set())
    return tmp_path


def test_default_cache_dir_is_forcingkit(clean_env):
    assert settings.cache_dir() == os.path.join(str(clean_env), ".cache", "forcingkit")
    assert settings.cache_dir("noaa") == os.path.join(
        str(clean_env), ".cache", "forcingkit", "noaa"
    )


def test_new_name_wins_over_legacy_names(clean_env, monkeypatch):
    monkeypatch.setenv("FORCINGKIT_CACHE_DIR", "/new")
    monkeypatch.setenv("ECODATA_CACHE_CACHE_DIR", "/old")
    assert settings.cache_dir() == "/new"


@pytest.mark.parametrize(
    "old", ["ECODATA_CACHE_CACHE_DIR", "COASTAL_SIM_DATA_CACHE_DIR"]
)
def test_legacy_cache_names_still_read_with_a_warning(clean_env, monkeypatch, old):
    monkeypatch.setenv(old, "/old")
    with pytest.warns(
        FutureWarning, match=f"{old} is deprecated; use FORCINGKIT_CACHE_DIR"
    ):
        assert settings.cache_dir("erddap") == os.path.join("/old", "erddap")


def test_legacy_cache_directory_used_until_moved(clean_env):
    legacy = clean_env / ".cache" / "ecodata-cache"
    legacy.mkdir(parents=True)
    with pytest.warns(FutureWarning, match="move the directory"):
        assert settings.cache_dir() == str(legacy)
    (clean_env / ".cache" / "forcingkit").mkdir()
    assert settings.cache_dir() == str(clean_env / ".cache" / "forcingkit")


def test_max_workers(clean_env, monkeypatch):
    assert settings.max_workers() == 4
    monkeypatch.setenv("ECODATA_CACHE_MAX_WORKERS", "2")
    with pytest.warns(FutureWarning):
        assert settings.max_workers() == 2
    monkeypatch.setenv("FORCINGKIT_MAX_WORKERS", "8")
    assert settings.max_workers() == 8
