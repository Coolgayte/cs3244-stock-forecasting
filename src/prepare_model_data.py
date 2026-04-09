from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import ENGINEERED_COLS

# Define paths relative to this script so teammates can run it from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ENGINEERED_BUCKET_PATH = PROCESSED_DIR / "bucket_top_vol_eng_2009_2017.csv"
OUT_DIR = PROCESSED_DIR / "model_ready_top_volume"

# Keep the training split strictly chronological to avoid leakage.
TRAIN_END = pd.Timestamp("2014-12-31")
VAL_END = pd.Timestamp("2015-12-31")
TEST_END = pd.Timestamp("2017-12-31")

ID_COLS = ["Date", "ticker", "target_date_1d"]
RAW_COLS = ["Open", "High", "Low", "Close", "Volume", "OpenInt"]
TARGET_COL = "target_log_return_1d"
FEATURE_COLS = ENGINEERED_COLS.copy()


def load_engineered_bucket(path: Path = ENGINEERED_BUCKET_PATH) -> pd.DataFrame:
    """Read the engineered bucket file and restore dates."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.sort_values(["ticker", "Date"], kind="mergesort").reset_index(drop=True)


def add_next_day_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the one-step-ahead log-return target inside each ticker."""
    out = df.copy()

    # The model should predict the next trading day's log return, not the current row's return.
    grouped = out.groupby("ticker", sort=False)
    out[TARGET_COL] = grouped["log_return"].shift(-1)
    out["target_date_1d"] = grouped["Date"].shift(-1)

    # Drop only rows that do not have a next-day target.
    out = out.dropna(subset=[TARGET_COL, "target_date_1d"]).reset_index(drop=True)
    return out


def split_by_target_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split rows according to the date of the target to avoid cross-period leakage."""
    train = df[df["target_date_1d"] <= TRAIN_END].copy()
    val = df[(df["target_date_1d"] > TRAIN_END) & (df["target_date_1d"] <= VAL_END)].copy()
    test = df[(df["target_date_1d"] > VAL_END) & (df["target_date_1d"] <= TEST_END)].copy()
    return {"train": train, "val": val, "test": test}


def fit_scaler(train_df: pd.DataFrame) -> pd.DataFrame:
    """Fit simple z-score statistics on the training split only."""
    stats = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "mean": train_df[FEATURE_COLS].mean(),
            "std": train_df[FEATURE_COLS].std(ddof=0).replace(0, 1.0),
        }
    ).reset_index(drop=True)
    return stats


def apply_scaler(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Apply train-only normalization while preserving identifiers and target columns."""
    out = df.copy()
    means = stats.set_index("feature")["mean"]
    stds = stats.set_index("feature")["std"]
    out[FEATURE_COLS] = (out[FEATURE_COLS] - means) / stds
    return out


def build_manifest(split_frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Store the agreed data contract so model teammates know what to read."""
    summary = {}
    for split_name, split_df in split_frames.items():
        summary[split_name] = {
            "rows": int(len(split_df)),
            "tickers": int(split_df["ticker"].nunique()),
            "min_date": split_df["Date"].min().date().isoformat(),
            "max_date": split_df["Date"].max().date().isoformat(),
            "min_target_date": split_df["target_date_1d"].min().date().isoformat(),
            "max_target_date": split_df["target_date_1d"].max().date().isoformat(),
        }

    return {
        "target_column": TARGET_COL,
        "feature_columns": FEATURE_COLS,
        "raw_columns": RAW_COLS,
        "id_columns": ID_COLS,
        "split_rule": {
            "train": "target_date_1d <= 2014-12-31",
            "val": "2015-01-01 <= target_date_1d <= 2015-12-31",
            "test": "2016-01-01 <= target_date_1d <= 2017-12-31",
        },
        "splits": summary,
    }


def main() -> None:
    if not ENGINEERED_BUCKET_PATH.exists():
        raise FileNotFoundError(f"Engineered bucket not found: {ENGINEERED_BUCKET_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start from the existing FE output so we do not duplicate feature calculations.
    engineered_df = load_engineered_bucket(ENGINEERED_BUCKET_PATH)
    model_df = add_next_day_target(engineered_df)
    split_frames = split_by_target_date(model_df)

    # Fit normalization only on the training split, then reuse it for validation and test.
    scaler_stats = fit_scaler(split_frames["train"])
    scaler_stats.to_csv(OUT_DIR / "scaler_stats.csv", index=False)

    split_summary_rows = []
    ordered_columns = ID_COLS + RAW_COLS + FEATURE_COLS + [TARGET_COL]

    for split_name, split_df in split_frames.items():
        split_df = split_df.sort_values(["ticker", "Date"], kind="mergesort").reset_index(drop=True)
        split_df = split_df[ordered_columns]

        split_df.to_csv(OUT_DIR / f"{split_name}.csv", index=False)

        scaled_df = apply_scaler(split_df, scaler_stats)
        scaled_df.to_csv(OUT_DIR / f"{split_name}_scaled.csv", index=False)

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

    pd.DataFrame(split_summary_rows).to_csv(OUT_DIR / "split_summary.csv", index=False)

    manifest = build_manifest(split_frames)
    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with (OUT_DIR / "feature_columns.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(FEATURE_COLS) + "\n")

    print("\n=== Model-ready split summary ===")
    print(pd.DataFrame(split_summary_rows).to_string(index=False))
    print("\nSaved model-ready outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
