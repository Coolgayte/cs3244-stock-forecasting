from __future__ import annotations

from pathlib import Path

import pandas as pd

from features import (
    END_DATE,
    MIN_RETAINED_FRACTION,
    START_DATE,
    drop_duplicate_dates,
    engineer_features,
    missing_required_columns,
    normalize_ticker,
    read_stock_file,
)

# Reuse the same indicator logic as the top-volume pipeline so sector analysis stays comparable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SECTOR_RAW_ROOT = PROCESSED_DIR / "sector_stocks"
SECTOR_OUT_ROOT = PROCESSED_DIR / "sector_stocks_eng"


def process_sector(sector_dir: Path, out_root: Path) -> list[dict]:
    """Engineer each stock in one sector and save both per-stock and combined outputs."""
    sector_name = sector_dir.name
    sector_out_dir = out_root / sector_name
    sector_out_dir.mkdir(parents=True, exist_ok=True)

    raw_frames: list[pd.DataFrame] = []
    eng_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    stock_files = sorted([p for p in sector_dir.iterdir() if p.suffix.lower() in {".csv", ".txt"}])

    for file_path in stock_files:
        ticker = normalize_ticker(file_path)
        stock_df = read_stock_file(file_path)
        raw_rows = len(stock_df)
        missing_cols = missing_required_columns(stock_df)

        if missing_cols:
            print(f"Skipping {sector_name}/{file_path.name}: missing {missing_cols}")
            summary_rows.append(
                {
                    "sector": sector_name,
                    "ticker": ticker,
                    "raw_rows": raw_rows,
                    "rows_after_dedup": 0,
                    "rows_after_cleaning": 0,
                    "duplicate_count": 0,
                    "output_file": "",
                    "missing_required_cols": ",".join(missing_cols),
                }
            )
            continue

        # Apply the same deterministic cleaning and research window used for top-volume stocks.
        stock_df["ticker"] = ticker
        stock_df = drop_duplicate_dates(stock_df)
        rows_after_dedup = len(stock_df)
        duplicate_count = raw_rows - rows_after_dedup

        stock_df = stock_df[
            (stock_df["Date"] >= START_DATE) & (stock_df["Date"] <= END_DATE)
        ].reset_index(drop=True)
        retained_fraction = (len(stock_df) / raw_rows) if raw_rows else 0.0
        if len(stock_df) == 0 or retained_fraction < MIN_RETAINED_FRACTION:
            print(
                f"Skipping {sector_name}/{ticker}: retained only {len(stock_df)} rows "
                f"({retained_fraction:.2%} of {raw_rows}) in the 2009-2017 window."
            )
            summary_rows.append(
                {
                    "sector": sector_name,
                    "ticker": ticker,
                    "raw_rows": raw_rows,
                    "rows_after_dedup": rows_after_dedup,
                    "rows_after_cleaning": 0,
                    "duplicate_count": duplicate_count,
                    "output_file": "",
                    "missing_required_cols": "window_filtered",
                }
            )
            continue

        eng_df = engineer_features(stock_df)

        raw_frames.append(stock_df.copy())
        eng_frames.append(eng_df.copy())

        out_path = sector_out_dir / f"{ticker}_eng.csv"
        eng_df.to_csv(out_path, index=False)

        summary_rows.append(
            {
                "sector": sector_name,
                "ticker": ticker,
                "raw_rows": raw_rows,
                "rows_after_dedup": rows_after_dedup,
                "rows_after_cleaning": len(eng_df),
                "duplicate_count": duplicate_count,
                "output_file": out_path.name,
                "missing_required_cols": "",
            }
        )

    if eng_frames:
        pd.concat(raw_frames, axis=0, ignore_index=True).sort_values(
            ["ticker", "Date"], kind="mergesort"
        ).to_csv(out_root / f"{sector_name.lower()}_raw_2009_2017.csv", index=False)

        pd.concat(eng_frames, axis=0, ignore_index=True).sort_values(
            ["ticker", "Date"], kind="mergesort"
        ).to_csv(out_root / f"{sector_name.lower()}_eng_2009_2017.csv", index=False)

    return summary_rows


def main() -> None:
    if not SECTOR_RAW_ROOT.exists():
        raise FileNotFoundError(f"Sector root not found: {SECTOR_RAW_ROOT}")

    SECTOR_OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for sector_dir in sorted([p for p in SECTOR_RAW_ROOT.iterdir() if p.is_dir()]):
        print(f"Processing sector: {sector_dir.name}")
        all_rows.extend(process_sector(sector_dir, SECTOR_OUT_ROOT))

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(SECTOR_OUT_ROOT / "sector_run_summary.csv", index=False)

    if not summary_df.empty:
        print("\n=== Sector FE summary ===")
        print(
            summary_df.groupby("sector")[["ticker", "rows_after_cleaning"]]
            .agg({"ticker": "count", "rows_after_cleaning": "sum"})
            .rename(columns={"ticker": "processed_tickers"})
            .to_string()
        )
    print("\nSaved sector outputs to:")
    print(SECTOR_OUT_ROOT)


if __name__ == "__main__":
    main()
