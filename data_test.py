import argparse
from pathlib import Path

import pandas as pd


def find_parquet_files(root: Path):
    """Find all parquet files under the given root directory."""
    return sorted(root.rglob("*.parquet"))


def load_parquet_file(path: Path, nrows: int = 5):
    """Load a parquet file and return a small preview plus basic info."""
    df = pd.read_parquet(path)
    return df, df.head(nrows)


def print_file_summary(path: Path):
    """Print a short summary for one parquet file."""
    print(f"\nFile: {path}")
    print(f"Size: {path.stat().st_size:,} bytes")
    print(f"Modified: {path.stat().st_mtime}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore parquet files under the data folder."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Root folder to search for parquet files (default: data)",
    )
    parser.add_argument(
        "--file",
        help="Specific parquet file to inspect. If omitted, the script lists all found parquet files.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=5,
        help="Number of rows to display from the parquet file preview.",
    )

    args = parser.parse_args()
    root_path = Path(args.root)

    if not root_path.exists():
        raise SystemExit(f"Root path does not exist: {root_path}")

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise SystemExit(f"Parquet file not found: {file_path}")
        print(f"Loading parquet file: {file_path}")
        df, preview = load_parquet_file(file_path, nrows=args.nrows)
        print_file_summary(file_path)
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print("\nPreview:")
        print(preview)
        return

    parquet_files = find_parquet_files(root_path)
    print(f"Searching for parquet files under: {root_path}")
    print(f"Found {len(parquet_files)} parquet file(s).\n")

    if not parquet_files:
        print("No parquet files found.")
        return

    for path in parquet_files:
        print(path)

    print("\nTo inspect one of these files, run:")
    print("  python data_test.py --file <path-to-file>")


if __name__ == "__main__":
    main()
