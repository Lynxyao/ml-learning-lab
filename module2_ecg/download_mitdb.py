# -*- coding: utf-8 -*-
"""Download MIT-BIH Arrhythmia Database records with wfdb."""

import argparse
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia Database from PhysioNet.")
    parser.add_argument("--out_dir", default=project_path("data", "ecg", "mitdb"))
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Optional record ids, e.g. 100 101 102. Omit to download all MIT-BIH records.",
    )
    args = parser.parse_args()
    args.out_dir = resolve_project_path(args.out_dir)

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("Install wfdb first: pip install -r module2_ecg/requirements_ecg.txt") from exc

    os.makedirs(args.out_dir, exist_ok=True)
    records = "all" if args.records is None else args.records
    wfdb.dl_database("mitdb", dl_dir=args.out_dir, records=records)
    print(f"MIT-BIH download complete: {args.out_dir}")


if __name__ == "__main__":
    main()
