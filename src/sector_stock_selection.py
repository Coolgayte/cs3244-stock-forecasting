from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STOCKS_DIR = (
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
        default=DEFAULT_STOCKS_DIR,
        help="Path to the Kaggle raw stock files directory.",
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
    return {path.stem.split(".")[0].upper(): path for path in stocks_dir.glob("*.txt")}


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
    stocks_dir = args.stocks_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    if not stocks_dir.exists():
        raise FileNotFoundError(f"Stocks directory not found: {stocks_dir}")

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
