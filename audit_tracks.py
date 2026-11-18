"""Audit GPS coordinates and consecutive sampling intervals per development trip."""
import csv
import math
import sys
from pathlib import Path


def audit(path):
    invalid_rows = invalid_coordinates = duplicates = backwards = long_gaps = 0
    max_gap = 0.0
    previous = None
    seen = set()
    with path.open() as source:
        for row in source:
            try:
                lon, lat, timestamp = map(float, row.split())
            except ValueError:
                invalid_rows += 1
                previous = None
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                invalid_coordinates += 1
            if not math.isfinite(timestamp):
                invalid_rows += 1
                previous = None
                continue
            duplicates += timestamp in seen
            seen.add(timestamp)
            if previous is not None:
                gap = timestamp - previous
                backwards += gap < 0
                long_gaps += gap > 5
                max_gap = max(max_gap, gap)
            previous = timestamp
    return invalid_rows, invalid_coordinates, duplicates, backwards, long_gaps, max_gap


if __name__ == "__main__":
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("partition", "trip_id", "malformed_or_invalid_time",
                     "invalid_coordinates", "repeated_times", "backwards_steps",
                     "gaps_over_5s", "max_gap_s"))
    root = Path(__file__).resolve().parent / "Student folder"
    for partition in ("train", "student_test"):
        for path in sorted((root / partition).glob("*/*.track")):
            writer.writerow((partition, path.stem, *audit(path)))
