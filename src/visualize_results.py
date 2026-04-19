"""
Reproduce evaluation plots from trained LSTM-XGBoost checkpoints.

Matches the training pipeline exactly:
  - Reads UNSCALED test.csv (same as training)
  - Applies per-window self-normalization before extracting LSTM latents
  - No inverse-scaling: predictions are raw log returns
  - Price reconstruction: pred_price = base_close * exp(y_pred)

Outputs to results/lstm_xgboost/figures/report/.
"""
from __future__ import annotations
import pickle, logging
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt

# ─── Config (must match training) ────────────────────────────────────────
LSTM_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
XGB_EXTRA_FEATURES = ["log_return", "rsi_14", "macd_line", "macd_hist",
                      "ema_50_200_ratio", "bb_width"]
EXCLUDE = {"atr_14", "obv", "vwap"}
HIDDEN_SIZE, NUM_LAYERS, DROPOUT = 64, 2, 0.2
SEQ_LEN, BATCH_SIZE = 30, 256
TARGET_COL, PRICE_COL, TICKER_COL = "target_log_return_1d", "Close", "ticker"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data" / "processed"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results" / "lstm_xgboost"
OUTDIR  = RESULTS / "figures" / "report"
DATASETS = {
    "banks":      DATA / "model_ready_sectors" / "banks",
    "energy":     DATA / "model_ready_sectors" / "energy",
    "retail":     DATA / "model_ready_sectors" / "retail",
    "shipping":   DATA / "model_ready_sectors" / "shipping",
    "tech":       DATA / "model_ready_sectors" / "tech",
    "top_volume": DATA / "model_ready_top_volume",
}

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("viz")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# ─── Model ───────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
    def encode(self, x):
        _, (h, _) = self.lstm(x); return h[-1]
    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)

# ─── Sequence builder (must match train_lstm_xgboost.py exactly) ─────────
def build_sequences(df, lstm_feats, extra_feats, target_col, seq_len):
    """
    Per-window self-normalization:
      - Price cols (OHLC): (price / day-0 price) - 1  [per column]
      - Volume col:        (vol / day-0 vol) - 1, clipped to [-5, 5]
      - Indicators (extra_feats): taken at prediction day, used as-is
    """
    Xs, Xe, y, idx, tks = [], [], [], [], []
    has = TICKER_COL in df.columns
    groups = df.groupby(TICKER_COL, sort=False) if has else [("_", df)]
    price_cols = [c for c in lstm_feats if c != "Volume"]
    has_vol = "Volume" in lstm_feats
    eps = 1e-8

    for name, g in groups:
        g = g.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
        price_arr = g[price_cols].to_numpy(np.float32) if price_cols else np.zeros((len(g), 0), np.float32)
        vol_arr   = g["Volume"].to_numpy(np.float32) if has_vol else None
        extra_arr = g[extra_feats].to_numpy(np.float32) if extra_feats else np.zeros((len(g), 0), np.float32)
        tgt       = g[target_col].to_numpy(np.float32)
        orig      = g["_orig_idx"].to_numpy()

        for i in range(seq_len, len(g)):
            if np.isnan(tgt[i]): continue

            wp     = price_arr[i-seq_len:i].copy()
            base_p = wp[0] + eps
            norm_p = (wp / base_p) - 1

            if has_vol:
                wv     = vol_arr[i-seq_len:i].copy()
                base_v = wv[0] + eps
                norm_v = np.clip((wv / base_v) - 1, -5.0, 5.0)
                window = np.column_stack((norm_p, norm_v))
            else:
                window = norm_p

            if not np.all(np.isfinite(window)): continue

            Xs.append(window); Xe.append(extra_arr[i])
            y.append(tgt[i]); idx.append(orig[i]); tks.append(name)

    return (np.asarray(Xs), np.asarray(Xe), np.asarray(y, np.float32),
            np.asarray(idx), np.asarray(tks))

def extract_latents(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).to(DEVICE)
            out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out, 0)

# ─── Reproduce predictions & compute report metrics ──────────────────────
def evaluate(name, folder):
    log.info(f"[{name}] evaluating")

    # Read UNSCALED data — normalization happens inside build_sequences
    test_df = pd.read_csv(folder/"test.csv")
    sample_cols = set(test_df.columns)
    lstm_feats  = [c for c in LSTM_FEATURES if c in sample_cols]
    extra_feats = [c for c in XGB_EXTRA_FEATURES if c not in EXCLUDE and c in sample_cols]

    Xte_s, Xte_e, yte, idx_te, tk_te = build_sequences(
        test_df, lstm_feats, extra_feats, TARGET_COL, SEQ_LEN)

    # Load checkpoint — read architecture from saved config
    ckpt = torch.load(MODELS/name/"lstm.pt", map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    model = LSTMForecaster(
        input_size=cfg["input_size"],
        hidden_size=cfg.get("hidden_size", HIDDEN_SIZE),
        num_layers=cfg.get("num_layers", NUM_LAYERS),
        dropout=cfg.get("dropout", DROPOUT),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    with open(MODELS/name/"xgb.pkl","rb") as f: xgb = pickle.load(f)

    Zte = extract_latents(model, Xte_s)
    lat_cols = [f"h{i}" for i in range(Zte.shape[1])]
    Dte = pd.DataFrame(np.hstack([Zte, Xte_e]) if Xte_e.size else Zte,
                       columns=lat_cols + extra_feats)
    Dte["ticker"] = pd.Categorical(tk_te)

    # Predictions are raw log returns — no inverse scaling
    y_pred = xgb.predict(Dte)
    y_true = yte

    base_close = test_df[PRICE_COL].to_numpy()[idx_te]
    if "target_close_1d" in test_df.columns:
        true_price = test_df["target_close_1d"].to_numpy()[idx_te]
    else:
        true_price = test_df.groupby(TICKER_COL)[PRICE_COL].shift(-1).to_numpy()[idx_te]

    mask = ~np.isnan(true_price) & ~np.isnan(base_close)
    y_true, y_pred = y_true[mask], y_pred[mask]
    base_close, true_price = base_close[mask], true_price[mask]
    pred_price = base_close * np.exp(y_pred)

    mse  = float(np.mean((y_true - y_pred) ** 2))
    da   = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2   = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs((true_price - pred_price) / true_price)) * 100)

    # Feature importance: indicator + ticker columns only (exclude latent h*)
    booster = xgb.get_booster()
    score   = booster.get_score(importance_type="gain")
    fnames  = booster.feature_names or list(Dte.columns)
    keep    = [f for f in fnames if not f.startswith("h")]
    imp     = pd.Series({f: score.get(f, 0.0) for f in keep})
    if imp.sum() > 0:
        imp = imp / imp.sum()
    imp = imp.sort_values(ascending=True)

    return dict(name=name, mse=mse, da=da, r2=r2, mape=mape, importance=imp)

# ─── Plots ───────────────────────────────────────────────────────────────
def plot_cross_sector(results, outdir):
    sectors = [r["name"] for r in results]
    mse  = [r["mse"]  for r in results]
    da   = [r["da"]   for r in results]
    r2   = [r["r2"]   for r in results]
    mape = [r["mape"] for r in results]
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(sectors)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Cross-Sector Model Performance Comparison",
                 fontsize=14, fontweight="bold", y=0.995)

    ax = axes[0,0]
    ax.bar(sectors, mse, color=palette)
    ax.set_title("Test Mean Squared Error (Lower is Better)")
    ax.set_xlabel("sector"); ax.set_ylabel("MSE")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0,1]
    ax.bar(sectors, da, color=palette)
    ax.axhline(50, color="red", ls="--", lw=1, label="Random Guess (50%)")
    ax.set_title("Directional Accuracy (Higher is Better)")
    ax.set_xlabel("sector"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100); ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1,0]
    ax.bar(sectors, r2, color=palette)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("R-Squared on Log Returns (Higher is Better)")
    ax.set_xlabel("sector"); ax.set_ylabel("R²")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1,1]
    ax.bar(sectors, mape, color=palette)
    ax.set_title("Mean Absolute Percentage Error (Lower is Better)")
    ax.set_xlabel("sector"); ax.set_ylabel("MAPE (%)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "cross_sector_comparison.png"); plt.close(fig)

def plot_feature_importance(name, imp, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(imp)))
    ax.barh(imp.index, imp.values, color=palette)
    ax.set_xlabel("Gain (Relative Importance)")
    ax.set_ylabel("Technical Indicator")
    ax.set_title(f"Feature Importance: {name.upper()} Sector")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"feature_importance_{name}.png"); plt.close(fig)

# ─── Driver ──────────────────────────────────────────────────────────────
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, path in DATASETS.items():
        try:
            r = evaluate(name, path)
            results.append(r)
            plot_feature_importance(name, r["importance"], OUTDIR)
        except FileNotFoundError as e:
            log.warning(f"[{name}] skipped (missing file): {e}")
        except Exception as e:
            log.error(f"[{name}] failed: {e}", exc_info=True)

    if results:
        plot_cross_sector(results, OUTDIR)
        df = pd.DataFrame([{"Sector": r["name"].capitalize(),
                            "Test MSE": r["mse"], "DA (%)": r["da"],
                            "R2 Score": r["r2"], "MAPE (%)": r["mape"]}
                           for r in results])
        df.to_csv(OUTDIR / "consolidated_metrics.csv", index=False)
        log.info(f"\n{df.to_string(index=False)}")
    log.info(f"Done. Figures in {OUTDIR}")

if __name__ == "__main__":
    main()