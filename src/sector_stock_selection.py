from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT_STOCKS_DIR = PROJECT_ROOT / "data" / "raw" / "Stocks"
DEFAULT_KAGGLE_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "kagglehub"
    / "datasets"
    / "borismarjanovic"
    / "price-volume-data-for-all-us-stocks-etfs"
    / "versions"
    / "3"
    / "Stocks"
)
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "data" / "raw" / "stocks_sector_clean_01.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sector_stocks"
STOCKS_DIR_CANDIDATES = [
    DEFAULT_PROJECT_STOCKS_DIR,
    DEFAULT_KAGGLE_CACHE_DIR,
    PROJECT_ROOT / "data" / "raw" / "stocks",
    PROJECT_ROOT / "data" / "raw" / "kaggle_raw_stocks",
    PROJECT_ROOT / "data" / "raw" / "all_stocks",
]

EXPECTED_COUNTS = {
    "Banks": 23,
    "Energy": 22,
    "Retail": 24,
    "Shipping": 24,
    "Tech": 25,
}

EXPECTED_MISSING = [
    ("SHEL", "Energy"),
    ("TTE", "Energy"),
    ("EQNR", "Energy"),
    ("MUFG", "Banks"),
    ("TFC", "Banks"),
    ("GAP", "Retail"),
    ("TRMD", "Shipping"),
]


@dataclass(frozen=True)
class MappingRow:
    ticker: str
    sector: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate sector-organized stock files from the cleaned mapping."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Path to the cleaned sector mapping CSV.",
    )
    parser.add_argument(
        "--stocks-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to the Kaggle raw stock files directory. "
            "If omitted, the script will try common local locations automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sector-organized stock files will be written.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing managed sector folders and the missing log before rebuilding.",
    )
    return parser.parse_args()


def resolve_stocks_dir(cli_value: Path | None) -> Path:
    """Find the raw stock folder from CLI input or common local locations."""
    candidate_paths: list[Path] = []
    if cli_value is not None:
        candidate_paths.append(cli_value.expanduser().resolve())

    candidate_paths.extend(path.expanduser().resolve() for path in STOCKS_DIR_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and (any(candidate.glob("*.txt")) or any(candidate.glob("*.csv"))):
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in seen)
    raise FileNotFoundError(
        "Could not find the raw stock directory. Checked:\n"
        f"{searched}\n\n"
        "Pass --stocks-dir explicitly, or place the full Kaggle Stocks folder in one of "
        "the common project data/raw locations."
    )


def load_mapping(mapping_path: Path) -> list[MappingRow]:
    rows: list[MappingRow] = []
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                MappingRow(
                    ticker=row["Ticker"].strip().upper(),
                    sector=row["Sector"].strip(),
                )
            )
    return rows


def clean_output_dir(output_dir: Path, sectors: list[str]) -> None:
    for sector in sectors:
        sector_dir = output_dir / sector
        if sector_dir.exists():
            shutil.rmtree(sector_dir)

    missing_log = output_dir / "missing_tickers.log"
    if missing_log.exists():
        missing_log.unlink()


def build_stock_index(stocks_dir: Path) -> dict[str, Path]:
    stock_index: dict[str, Path] = {}
    for pattern in ("*.txt", "*.csv"):
        for path in stocks_dir.glob(pattern):
            stock_index[path.stem.split(".")[0].upper()] = path
    return stock_index


def write_missing_log(missing_path: Path, missing_rows: list[tuple[str, str]]) -> None:
    lines = ["Missing tickers from local Kaggle stock dataset:"]
    lines.extend(f"{ticker},{sector}" for ticker, sector in missing_rows)
    missing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_results(copied_counts: dict[str, int], missing_rows: list[tuple[str, str]]) -> None:
    if copied_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Sector counts do not match the validated local result. "
            f"Expected {EXPECTED_COUNTS}, got {copied_counts}."
        )

    if missing_rows != EXPECTED_MISSING:
        raise RuntimeError(
            f"Missing tickers do not match the validated local result. "
            f"Expected {EXPECTED_MISSING}, got {missing_rows}."
        )


def main() -> None:
    args = parse_args()
    mapping_path = args.mapping.expanduser().resolve()
    stocks_dir = resolve_stocks_dir(args.stocks_dir)
    output_dir = args.output_dir.expanduser().resolve()

    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    mapping_rows = load_mapping(mapping_path)
    sectors = list(dict.fromkeys(row.sector for row in mapping_rows))

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_output_dir(output_dir, sectors)

    for sector in sectors:
        (output_dir / sector).mkdir(parents=True, exist_ok=True)

    stock_index = build_stock_index(stocks_dir)
    copied_counts = {sector: 0 for sector in sectors}
    missing_rows: list[tuple[str, str]] = []

    for row in mapping_rows:
        source = stock_index.get(row.ticker)
        if source is None:
            missing_rows.append((row.ticker, row.sector))
            continue

        destination = output_dir / row.sector / f"{row.ticker}.csv"
        shutil.copy2(source, destination)
        copied_counts[row.sector] += 1

    write_missing_log(output_dir / "missing_tickers.log", missing_rows)
    validate_results(copied_counts, missing_rows)

    print(f"mapping: {mapping_path}")
    print(f"stocks_dir: {stocks_dir}")
    print(f"output_dir: {output_dir}")
    print("copied_counts:")
    for sector, count in copied_counts.items():
        print(f"  {sector}: {count}")
    print(f"missing_count: {len(missing_rows)}")
    print("missing_tickers:")
    for ticker, sector in missing_rows:
        print(f"  {ticker},{sector}")


if __name__ == "__main__":
    main()
