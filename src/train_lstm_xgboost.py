"""
LSTM-XGBoost hybrid training & evaluation pipeline.

LSTM encodes raw OHLCV sequences → latent h_T.
h_T ⊕ indicator features ⊕ categorical ticker → XGBoost (Optuna-tuned).
Target: next-day log return (target_log_return_1d).
"""
from __future__ import annotations
import json, pickle, logging
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
import optuna
import matplotlib.pyplot as plt

# ─── Config ──────────────────────────────────────────────────────────────
LSTM_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
XGB_EXTRA_FEATURES = ["log_return", "rsi_14", "macd_line", "macd_hist",
                      "ema_50_200_ratio", "bb_width"]
EXCLUDE = {"atr_14", "obv", "vwap"}

HIDDEN_SIZE, NUM_LAYERS, DROPOUT = 16, 2, 0.0
SEQ_LEN, BATCH_SIZE = 30, 256
LSTM_EPOCHS, LSTM_LR, ES_PATIENCE = 200, 1e-3, 50
OPTUNA_TRIALS = 200
TARGET_COL, PRICE_COL, TICKER_COL = "target_log_return_1d", "Close", "ticker"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data" / "processed"
MODELS  = ROOT / "models"
RESULTS = ROOT / "results" / "lstm_xgboost"
FIGS    = RESULTS / "figures"
DATASETS = {
    "top_volume": DATA / "model_ready_top_volume",
    "banks":      DATA / "model_ready_sectors" / "banks",
    "energy":     DATA / "model_ready_sectors" / "energy",
    "retail":     DATA / "model_ready_sectors" / "retail",
    "shipping":   DATA / "model_ready_sectors" / "shipping",
    "tech":       DATA / "model_ready_sectors" / "tech",
}

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lstm_xgb")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─── Model ───────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE, 1)
    def encode(self, x):
        _, (h, _) = self.lstm(x); return h[-1]
    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)

# ─── Sequence builder ────────────────────────────────────────────────────
def build_sequences(df, lstm_feats, extra_feats, target_col, seq_len):
    Xs, Xe, y, idx, tks = [], [], [], [], []
    has = TICKER_COL in df.columns
    groups = df.groupby(TICKER_COL, sort=False) if has else [("_", df)]
    for name, g in groups:
        g = g.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
        seq_arr   = g[lstm_feats].to_numpy(np.float32)
        extra_arr = g[extra_feats].to_numpy(np.float32) if extra_feats else np.zeros((len(g),0),np.float32)
        tgt       = g[target_col].to_numpy(np.float32)
        orig      = g["_orig_idx"].to_numpy()
        for i in range(seq_len, len(g)):
            if np.isnan(tgt[i]): continue
            Xs.append(seq_arr[i-seq_len:i]); Xe.append(extra_arr[i])
            y.append(tgt[i]); idx.append(orig[i]); tks.append(name)
    return (np.asarray(Xs), np.asarray(Xe), np.asarray(y, np.float32),
            np.asarray(idx), np.asarray(tks))

# ─── LSTM training ───────────────────────────────────────────────────────
def train_lstm(model, tr_loader, va_loader, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train(); tr_losses = []
        for xb, yb in tr_loader:
          xb, yb = xb.to(DEVICE), yb.to(DEVICE)
          opt.zero_grad()
          l = loss_fn(model(xb), yb)
          l.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← add this
          opt.step()
          tr_losses.append(l.item())
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
                                for xb, yb in va_loader]))
        tr = float(np.mean(tr_losses)); mark = ""
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            mark = "  *"
        else:
            bad += 1
        log.info(f"[{tag}] epoch {epoch:02d}/{LSTM_EPOCHS}  train_mse={tr:.6f}  val_mse={vl:.6f}{mark}")
        if bad >= ES_PATIENCE:
            log.info(f"[{tag}] early stop @ epoch {epoch} (best val_mse={best:.6f})"); break
    if best_state: model.load_state_dict(best_state)
    return model

def extract_latents(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).to(DEVICE)
            out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out, 0)

# ─── Optuna ──────────────────────────────────────────────────────────────
def tune_xgb(Xtr, ytr, Xva, yva, tag):
    def objective(trial):
        p = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1500),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            tree_method="hist", enable_categorical=True,
            objective="reg:squarederror", n_jobs=-1, verbosity=0,
        )
        m = XGBRegressor(**p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return float(np.mean((m.predict(Xva) - yva) ** 2))
    def cb(study, trial):
        log.info(f"[{tag}] optuna {trial.number+1}/{OPTUNA_TRIALS}  "
                 f"val_mse={trial.value:.6f}  best={study.best_value:.6f}")
    study = optuna.create_study(direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED))
    log.info(f"[{tag}] starting Optuna ({OPTUNA_TRIALS} trials)")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, callbacks=[cb])
    log.info(f"[{tag}] best_val_mse={study.best_value:.6f}  params={study.best_params}")
    return study.best_params

# ─── Metrics & plots ─────────────────────────────────────────────────────
def compute_metrics(y_true_lr, y_pred_lr, base_close, true_price):
    pred_price = base_close * np.exp(y_pred_lr)
    ss_res = np.sum((true_price - pred_price) ** 2)
    ss_tot = np.sum((true_price - np.mean(true_price)) ** 2)
    return {
        "mse_log_return": float(np.mean((y_true_lr - y_pred_lr) ** 2)),
        "r2_price": float(1 - ss_res/ss_tot) if ss_tot > 0 else float("nan"),
        "mape_price": float(np.mean(np.abs((true_price-pred_price)/true_price))*100),
        "directional_accuracy": float(np.mean(np.sign(y_true_lr)==np.sign(y_pred_lr))*100),
    }, pred_price

def plot_reconstruction(name, tickers, true_price, pred_price, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    pick = pd.unique(tickers)[:6]; n = len(pick)
    rows = int(np.ceil(n/2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2*rows), squeeze=False)
    for ax, t in zip(axes.flat, pick):
        m = tickers == t
        ax.plot(true_price[m], label="actual", lw=1.2)
        ax.plot(pred_price[m], label="predicted", lw=1.0, alpha=0.8)
        ax.set_title(f"{name} — {t}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes.flat[n:]: ax.axis("off")
    fig.suptitle(f"{name}: reconstructed test prices", y=1.02); fig.tight_layout()
    fig.savefig(outdir/f"{name}_reconstruction.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(true_price, pred_price, s=4, alpha=0.4)
    lo = float(min(true_price.min(), pred_price.min()))
    hi = float(max(true_price.max(), pred_price.max()))
    ax.plot([lo,hi],[lo,hi],"r--",lw=1)
    ax.set_xlabel("actual"); ax.set_ylabel("predicted")
    ax.set_title(f"{name}: predicted vs actual"); ax.grid(alpha=0.3)
    fig.savefig(outdir/f"{name}_scatter.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ─── Helpers ─────────────────────────────────────────────────────────────
def stats_index_col(folder):
    cols = pd.read_csv(folder/"scaler_stats.csv", nrows=0).columns.tolist()
    for c in ("feature","column","name","Unnamed: 0"):
        if c in cols: return c
    return cols[0]

# ─── Per-dataset ─────────────────────────────────────────────────────────
def run_dataset(name, folder):
    log.info(f"===== {name} =====")
    torch.manual_seed(SEED); np.random.seed(SEED)

    sample_cols = set(pd.read_csv(folder/"train_scaled.csv", nrows=0).columns)
    lstm_feats  = [c for c in LSTM_FEATURES if c in sample_cols]
    if not lstm_feats:
        raise ValueError(f"[{name}] no OHLCV columns in train_scaled.csv. Found: {sorted(sample_cols)}")
    extra_feats = [c for c in XGB_EXTRA_FEATURES if c not in EXCLUDE and c in sample_cols]
    log.info(f"[{name}] LSTM feats={lstm_feats}")
    log.info(f"[{name}] XGB extras={extra_feats}")

    stats = pd.read_csv(folder/"scaler_stats.csv").set_index(stats_index_col(folder))
    if TARGET_COL in stats.index:
        tgt_mean, tgt_std = float(stats.loc[TARGET_COL,"mean"]), float(stats.loc[TARGET_COL,"std"])
    else:
        log.info(f"[{name}] {TARGET_COL} not in scaler_stats; assuming unscaled target")
        tgt_mean, tgt_std = 0.0, 1.0

    train_s = pd.read_csv(folder/"train_scaled.csv")
    val_s   = pd.read_csv(folder/"val_scaled.csv")
    test_s  = pd.read_csv(folder/"test_scaled.csv")
    test_r  = pd.read_csv(folder/"test.csv")

    Xtr_s, Xtr_e, ytr, _,      tk_tr = build_sequences(train_s, lstm_feats, extra_feats, TARGET_COL, SEQ_LEN)
    Xva_s, Xva_e, yva, _,      tk_va = build_sequences(val_s,   lstm_feats, extra_feats, TARGET_COL, SEQ_LEN)
    Xte_s, Xte_e, yte, idx_te, tk_te = build_sequences(test_s,  lstm_feats, extra_feats, TARGET_COL, SEQ_LEN)
    log.info(f"[{name}] seqs: train={len(Xtr_s)} val={len(Xva_s)} test={len(Xte_s)}")

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(ytr)),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva_s), torch.from_numpy(yva)),
                           batch_size=BATCH_SIZE)

    model = LSTMForecaster(input_size=len(lstm_feats)).to(DEVICE)
    log.info(f"[{name}] training LSTM on {DEVICE}")
    model = train_lstm(model, tr_loader, va_loader, tag=name)

    Ztr, Zva, Zte = extract_latents(model,Xtr_s), extract_latents(model,Xva_s), extract_latents(model,Xte_s)
    cats = pd.Categorical(np.concatenate([tk_tr,tk_va,tk_te])).categories
    def make_df(Z, extra, tks):
        lat_cols = [f"h{i}" for i in range(Z.shape[1])]
        df = pd.DataFrame(np.hstack([Z, extra]) if extra.size else Z,
                          columns=lat_cols + extra_feats)
        df["ticker"] = pd.Categorical(tks, categories=cats)
        return df
    Dtr, Dva, Dte = make_df(Ztr,Xtr_e,tk_tr), make_df(Zva,Xva_e,tk_va), make_df(Zte,Xte_e,tk_te)

    best_params = tune_xgb(Dtr, ytr, Dva, yva, tag=name)
    xgb = XGBRegressor(**best_params, tree_method="hist", enable_categorical=True,
                       objective="reg:squarederror", n_jobs=-1, verbosity=0)
    xgb.fit(pd.concat([Dtr,Dva],ignore_index=True), np.concatenate([ytr,yva]))

    y_pred = xgb.predict(Dte) * tgt_std + tgt_mean
    y_true = yte * tgt_std + tgt_mean

    base_close = test_r[PRICE_COL].to_numpy()[idx_te]
    if "target_close_1d" in test_r.columns:
        true_price = test_r["target_close_1d"].to_numpy()[idx_te]
    else:
        true_price = test_r.groupby(TICKER_COL)[PRICE_COL].shift(-1).to_numpy()[idx_te]

    mask = ~np.isnan(true_price) & ~np.isnan(base_close)
    y_true, y_pred = y_true[mask], y_pred[mask]
    base_close, true_price, tk_plot = base_close[mask], true_price[mask], tk_te[mask]

    metrics, pred_price = compute_metrics(y_true, y_pred, base_close, true_price)
    log.info(f"[{name}] metrics: {metrics}")

    plot_reconstruction(name, tk_plot, true_price, pred_price, FIGS)

    mdir = MODELS/name; mdir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict":model.state_dict(),
                "config":dict(input_size=len(lstm_feats), hidden_size=HIDDEN_SIZE,
                              num_layers=NUM_LAYERS, dropout=DROPOUT, seq_len=SEQ_LEN,
                              lstm_features=lstm_feats, extra_features=extra_feats,
                              target_col=TARGET_COL)},
               mdir/"lstm.pt")
    with open(mdir/"xgb.pkl","wb") as f: pickle.dump(xgb,f)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS/f"{name}.json").write_text(json.dumps(
        {"metrics":metrics,"best_xgb_params":best_params}, indent=2))
    return metrics

def main():
    summary = {name: run_dataset(name, path) for name, path in DATASETS.items()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).T.to_csv(RESULTS/"summary.csv")
    log.info(f"Done. Results in {RESULTS}")

if __name__ == "__main__":
    main()
