"""Unit tests for HRRR .idx parsing and selective byte-range fetching."""

from forcingkit.fetchers.hrrr import _parse_idx

# The 10 m wind messages, as a sample of what the HRRR atmosphere fetcher selects.
WIND_IDX_PATTERNS = ("UGRD:10 m above ground", "VGRD:10 m above ground")


# Representative .idx content from a real HRRR wrfsfcf00.grib2.idx
SAMPLE_IDX = """\
1:0:d=2026032500:REFC:entire atmosphere:anl:
2:156487:d=2026032500:RETOP:cloud top:anl:
3:289977:d=2026032500:VIL:entire atmosphere:anl:
4:415682:d=2026032500:VIS:surface:anl:
5:606403:d=2026032500:REFD:1000 m above ground:anl:
6:731024:d=2026032500:GUST:surface:anl:
7:903870:d=2026032500:UGRD:10 m above ground:anl:
8:1078492:d=2026032500:VGRD:10 m above ground:anl:
9:1253100:d=2026032500:WIND:10 m above ground:anl:
10:1371812:d=2026032500:TMP:2 m above ground:anl:
11:1544833:d=2026032500:SPFH:2 m above ground:anl:
12:1675220:d=2026032500:DPT:2 m above ground:anl:
13:1849503:d=2026032500:PRES:surface:anl:
"""


def test_parse_idx_finds_wind_variables():
    ranges = _parse_idx(SAMPLE_IDX, WIND_IDX_PATTERNS)
    assert len(ranges) == 2

    # UGRD starts at 903870, ends at 1078492 (start of VGRD)
    assert ranges[0] == (903870, 1078492)

    # VGRD starts at 1078492, ends at 1253100 (start of WIND)
    assert ranges[1] == (1078492, 1253100)


def test_parse_idx_no_matches():
    ranges = _parse_idx(SAMPLE_IDX, ("NONEXISTENT:foo",))
    assert ranges == []


def test_parse_idx_last_message():
    """If the matched variable is the last line, end_byte should be None (EOF)."""
    idx_text = """\
1:0:d=2026032500:PRES:surface:anl:
2:100000:d=2026032500:UGRD:10 m above ground:anl:
"""
    ranges = _parse_idx(idx_text, ("UGRD:10 m above ground",))
    assert len(ranges) == 1
    assert ranges[0] == (100000, None)


def test_parse_idx_empty_input():
    ranges = _parse_idx("", WIND_IDX_PATTERNS)
    assert ranges == []


def test_parse_idx_malformed_lines():
    """Malformed lines should be skipped without error."""
    idx_text = """\
bad line
1:0:d=2026032500:PRES:surface:anl:
not:a:valid:int:offset
2:500:d=2026032500:UGRD:10 m above ground:anl:
3:1000:d=2026032500:VGRD:10 m above ground:anl:
4:1500:d=2026032500:TMP:2 m above ground:anl:
"""
    ranges = _parse_idx(idx_text, WIND_IDX_PATTERNS)
    assert len(ranges) == 2
    assert ranges[0] == (500, 1000)
    assert ranges[1] == (1000, 1500)
