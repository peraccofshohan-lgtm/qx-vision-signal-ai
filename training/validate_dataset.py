#!/usr/bin/env python3
"""Validate and version a user-supplied CSV or Parquet OHLC dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qxvision.dataset import DatasetValidationError, validate_dataset, write_dataset_metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Strict QX Vision OHLC dataset validator")
    parser.add_argument("dataset")
    parser.add_argument("--dataset-version", default="1.0.0")
    parser.add_argument("--source-description", default="user-supplied OHLC dataset")
    parser.add_argument("--metadata", default=None, help="write immutable validation metadata JSON")
    parser.add_argument("--force", action="store_true", help="explicitly replace an existing metadata path")
    args = parser.parse_args(argv)
    try:
        report = validate_dataset(args.dataset, dataset_version=args.dataset_version, source_description=args.source_description)
    except (OSError, DatasetValidationError) as exc:
        report = getattr(exc, "report", None)
        if report is None:
            print(str(exc), file=sys.stderr); return 2
    output = Path(args.metadata) if args.metadata else Path(args.dataset).with_suffix(Path(args.dataset).suffix + ".metadata.json")
    try:
        write_dataset_metadata(report, output, overwrite=args.force)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr); return 2
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(f"metadata: {output}")
    return 0 if report.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
