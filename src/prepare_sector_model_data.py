from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prepare_model_data import (
    FEATURE_COLS,
    ID_COLS,
    RAW_COLS,
    TARGET_COL,
    add_next_day_target,
    apply_scaler,
    build_manifest,
    fit_scaler,
    split_by_target_date,
)

# Package each sector into the same split/target format as top-volume data.
FEATURE_DIR = Path(__file__).resolve().parent
SECTOR_ENG_ROOT = FEATURE_DIR / "sector_stocks_eng"
OUT_ROOT = FEATURE_DIR / "model_ready_sectors"


def load_sector_bucket(path: Path) -> pd.DataFrame:
    """Read one combined sector engineered file and restore dates."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values(["ticker", "Date"], kind="mergesort").reset_index(drop=True)


def process_sector_bucket(sector_file: Path, out_root: Path) -> None:
    """Create train/val/test splits and scaling metadata for one sector."""
    sector_name = sector_file.stem.replace("_eng_2009_2017", "")
    sector_out_dir = out_root / sector_name
    sector_out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the same target and split logic so comparisons across datasets stay valid.
    engineered_df = load_sector_bucket(sector_file)
    model_df = add_next_day_target(engineered_df)
    split_frames = split_by_target_date(model_df)

    scaler_stats = fit_scaler(split_frames["train"])
    scaler_stats.to_csv(sector_out_dir / "scaler_stats.csv", index=False)

    split_summary_rows = []
    ordered_columns = ID_COLS + RAW_COLS + FEATURE_COLS + [TARGET_COL]

    for split_name, split_df in split_frames.items():
        split_df = split_df.sort_values(["ticker", "Date"], kind="mergesort").reset_index(drop=True)
        split_df = split_df[ordered_columns]
        split_df.to_csv(sector_out_dir / f"{split_name}.csv", index=False)

        scaled_df = apply_scaler(split_df, scaler_stats)
        scaled_df.to_csv(sector_out_dir / f"{split_name}_scaled.csv", index=False)

        split_summary_rows.append(
            {
                "split": split_name,
                "rows": len(split_df),
                "tickers": split_df["ticker"].nunique(),
                "min_date": split_df["Date"].min().date().isoformat(),
                "max_date": split_df["Date"].max().date().isoformat(),
                "min_target_date": split_df["target_date_1d"].min().date().isoformat(),
                "max_target_date": split_df["target_date_1d"].max().date().isoformat(),
            }
        )

    pd.DataFrame(split_summary_rows).to_csv(sector_out_dir / "split_summary.csv", index=False)

    manifest = build_manifest(split_frames)
    with (sector_out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with (sector_out_dir / "feature_columns.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(FEATURE_COLS) + "\n")

    print(f"\n=== {sector_name} model-ready summary ===")
    print(pd.DataFrame(split_summary_rows).to_string(index=False))


def main() -> None:
    if not SECTOR_ENG_ROOT.exists():
        raise FileNotFoundError(f"Sector engineered root not found: {SECTOR_ENG_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sector_files = sorted(SECTOR_ENG_ROOT.glob("*_eng_2009_2017.csv"))
    if not sector_files:
        raise FileNotFoundError("No combined sector engineered files were found.")

    for sector_file in sector_files:
        process_sector_bucket(sector_file, OUT_ROOT)

    print("\nSaved sector model-ready outputs to:")
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
