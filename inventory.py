"""Summarize the supplied development trips without opening evaluator data."""
import csv
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent / "Student folder"
    writer = csv.writer(sys.stdout, lineterminator="\n")
    extensions = ("track", "nodes", "arcs", "route")
    writer.writerow(("partition", "trip_id", *extensions))
    for partition in ("train", "student_test"):
        for trip in sorted((root / partition).iterdir()):
            if trip.is_dir():
                counts = []
                for extension in extensions:
                    with (trip / f"{trip.name}.{extension}").open("rb") as rows:
                        counts.append(sum(1 for row in rows if row.strip()))
                writer.writerow((partition, trip.name, *counts))


if __name__ == "__main__":
    main()
