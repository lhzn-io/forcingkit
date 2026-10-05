import s3fs
import logging


logger = logging.getLogger(__name__)

# NOAA makes HRRR freely available on AWS S3 without authentication.
# The same bucket (noaa-hrrr-bdp-pds) holds both the rolling operational
# window and the full historical archive back to 2014-07-30.
fs = s3fs.S3FileSystem(anon=True)


# First date for which HRRR data exists in the noaa-hrrr-bdp-pds archive.
# GRIB2 .idx variable patterns for 10-meter wind components.
# Each HRRR .grib2 on S3 has a sidecar .idx listing byte offsets per message.
def _parse_idx(
    idx_text: str, patterns: tuple[str, ...]
) -> list[tuple[int, int | None]]:
    """
    Parse a GRIB2 .idx sidecar file and return byte ranges for matching messages.

    Each .idx line has the format:
        message_number:byte_offset:date:variable:level:forecast_type:

    Returns a list of (start_byte, end_byte) tuples. end_byte is None for the
    last message in the file (meaning read-to-EOF).
    """
    lines = [line.strip() for line in idx_text.strip().splitlines() if line.strip()]

    # Build list of (message_number, byte_offset, raw_line)
    entries = []
    for line in lines:
        parts = line.split(":")
        if len(parts) < 3:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        entries.append((offset, line))

    # Find matching messages and compute byte ranges
    ranges = []
    for i, (offset, line) in enumerate(entries):
        if any(pattern in line for pattern in patterns):
            # End byte is the start of the next message, or None for EOF
            end = entries[i + 1][0] if i + 1 < len(entries) else None
            ranges.append((offset, end))

    return ranges


def _fetch_s3_byte_ranges(
    s3_path: str, ranges: list[tuple[int, int | None]], output_path: str
) -> None:
    """
    Download specific byte ranges from an S3 object and concatenate into a
    single valid GRIB2 file. Each GRIB2 message is self-contained, so
    concatenating selected messages produces a valid file.
    """
    with open(output_path, "wb") as f:
        for start, end in ranges:
            if end is not None:
                length = end - start
                with fs.open(s3_path, "rb") as s3f:
                    s3f.seek(start)
                    f.write(s3f.read(length))
            else:
                # Read from start to EOF
                with fs.open(s3_path, "rb") as s3f:
                    s3f.seek(start)
                    f.write(s3f.read())
