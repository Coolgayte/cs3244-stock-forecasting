"""
Generate evaluation visualizations from saved LSTM-XGBoost models.

Loads models from models/{name}/{lstm.pt, xgb.pkl}, regenerates predictions on
the test set, and writes individual PNGs to results/lstm_xgboost/figures/{name}/.

Visualizations per dataset:
  01_per_ticker_da.png         per-ticker directional accuracy
  02_price_overlay_*.png       predicted vs actual price (per ticker, up to 8)
  03_scatter_returns.png       predicted vs actual log-returns (with y=x)
  04_scatter_prices.png        predicted vs actual reconstructed prices
  05_residuals_time.png        residuals over test sequence index
  06_residuals_hist.png        residual histogram + normal overlay
  07_residuals_qq.png          Q-Q plot of residuals
  08_equity_curve.png          strategy PnL vs buy-and-hold (per ticker, mean)
  09_confusion_direction.png   predicted up/down vs actual up/down
  10_calibration.png           binned predicted vs realized return magnitude
  11_baselines.png             model vs naive baselines (zero, momentum)
  12_shap_summary.png          SHAP beeswarm of XGBoost features
  13_latent_tsne_ticker.png    t-SNE of LSTM latents colored by ticker
  14_latent_tsne_direction.png t-SNE colored by future return sign

Cross-sector summary in results/lstm_xgboost/figures/_summary/:
  per_sector_baseline_comparison.png
  per_sector_da_with_ci.png
"""
from __future__ import annotations
import pickle, logging, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from scipy import stats as sstats

# Optional: SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─── Config (must match training) ────────────────────────────────────────
LSTM_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
XGB_EXTRA_FEATURES = ["log_return", "rsi_14", "macd_line", "macd_hist",
                      "ema_50_200_ratio", "bb_width"]
EXCLUDE = {"atr_14", "obv", "vwap"}
HIDDEN_SIZE, NUM_LAYERS, DROPOUT = 16, 2, 0.0
SEQ_LEN, BATCH_SIZE = 30, 256
TARGET_COL, PRICE_COL, TICKER_COL = "target_log_return_1d", "Close", "ticker"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
log = logging.getLogger("viz")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

# ─── Model (must match training) ─────────────────────────────────────────
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

# ─── Sequence builder (must match training) ──────────────────────────────
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

def extract_latents(model, X):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i+BATCH_SIZE]).to(DEVICE)
            out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out, 0)

def stats_index_col(folder):
    cols = pd.read_csv(folder/"scaler_stats.csv", nrows=0).columns.tolist()
    for c in ("feature","column","name","Unnamed: 0"):
        if c in cols: return c
    return cols[0]

# ─── Per-dataset prediction reproduction ─────────────────────────────────
def reproduce(name, folder):
    log.info(f"[{name}] reproducing predictions")
    sample_cols = set(pd.read_csv(folder/"train_scaled.csv", nrows=0).columns)
    lstm_feats  = [c for c in LSTM_FEATURES if c in sample_cols]
    extra_feats = [c for c in XGB_EXTRA_FEATURES if c not in EXCLUDE and c in sample_cols]

    stats = pd.read_csv(folder/"scaler_stats.csv").set_index(stats_index_col(folder))
    if TARGET_COL in stats.index:
        tgt_mean = float(stats.loc[TARGET_COL,"mean"])
        tgt_std  = float(stats.loc[TARGET_COL,"std"])
    else:
        tgt_mean, tgt_std = 0.0, 1.0

    test_s = pd.read_csv(folder/"test_scaled.csv")
    test_r = pd.read_csv(folder/"test.csv")

    Xte_s, Xte_e, yte, idx_te, tk_te = build_sequences(
        test_s, lstm_feats, extra_feats, TARGET_COL, SEQ_LEN)

    # Load LSTM
    ckpt = torch.load(MODELS/name/"lstm.pt", map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = LSTMForecaster(input_size=cfg["input_size"],
        hidden_size=cfg.get("hidden_size", HIDDEN_SIZE),
        num_layers=cfg.get("num_layers", NUM_LAYERS),
        dropout=cfg.get("dropout", DROPOUT)).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    # Load XGB & build feature DF
    with open(MODELS/name/"xgb.pkl","rb") as f: xgb = pickle.load(f)
    Zte = extract_latents(model, Xte_s)
    lat_cols = [f"h{i}" for i in range(Zte.shape[1])]
    Dte = pd.DataFrame(np.hstack([Zte, Xte_e]) if Xte_e.size else Zte,
                       columns=lat_cols + extra_feats)
    # Use whatever categories the booster saw at fit time
    booster_cats = list(xgb.get_booster().feature_names or [])
    if "ticker" in booster_cats:
        # XGBoost stored its own category set; align via Categorical
        Dte["ticker"] = pd.Categorical(tk_te)
    else:
        Dte["ticker"] = pd.Categorical(tk_te)

    y_pred_scaled = xgb.predict(Dte)
    y_pred = y_pred_scaled * tgt_std + tgt_mean
    y_true = yte * tgt_std + tgt_mean

    base_close = test_r[PRICE_COL].to_numpy()[idx_te]
    if "target_close_1d" in test_r.columns:
        true_price = test_r["target_close_1d"].to_numpy()[idx_te]
    else:
        true_price = test_r.groupby(TICKER_COL)[PRICE_COL].shift(-1).to_numpy()[idx_te]

    mask = ~np.isnan(true_price) & ~np.isnan(base_close)
    pred_price = base_close * np.exp(y_pred)
    return dict(
        name=name, model=model, xgb=xgb, Dte=Dte[mask], Zte=Zte[mask],
        y_true=y_true[mask], y_pred=y_pred[mask],
        base_close=base_close[mask], true_price=true_price[mask],
        pred_price=pred_price[mask], tickers=tk_te[mask],
        extra_feats=extra_feats, lat_cols=lat_cols,
    )

# ─── Plotting helpers ────────────────────────────────────────────────────
def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)

def plot_per_ticker_da(ctx, outdir):
    df = pd.DataFrame({"t": ctx["tickers"], "yt": ctx["y_true"], "yp": ctx["y_pred"]})
    da = df.groupby("t").apply(
        lambda g: float(np.mean(np.sign(g.yt) == np.sign(g.yp)) * 100),
        include_groups=False).sort_values()
    n = df.groupby("t").size().reindex(da.index)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3*len(da))))
    colors = ["#c0392b" if v < 50 else "#27ae60" for v in da.values]
    ax.barh(da.index, da.values, color=colors, alpha=0.85)
    ax.axvline(50, color="k", ls="--", lw=1, label="random (50%)")
    for i,(v,c) in enumerate(zip(da.values, n.values)):
        ax.text(v+0.3, i, f"{v:.1f}% (n={c})", va="center", fontsize=8)
    ax.set_xlabel("Directional Accuracy (%)")
    ax.set_title(f"{ctx['name']}: per-ticker DA")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
    savefig(fig, outdir / "01_per_ticker_da.png")

def plot_price_overlays(ctx, outdir):
    uniq = pd.unique(ctx["tickers"])[:8]
    for t in uniq:
        m = ctx["tickers"] == t
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(ctx["true_price"][m], label="actual", lw=1.3)
        ax.plot(ctx["pred_price"][m], label="predicted", lw=1.0, alpha=0.85)
        ax.set_title(f"{ctx['name']} — {t}: reconstructed test prices")
        ax.set_xlabel("test step"); ax.set_ylabel("price")
        ax.grid(alpha=0.3); ax.legend()
        safe_t = str(t).replace("/","_").replace(" ","_")
        savefig(fig, outdir / f"02_price_overlay_{safe_t}.png")

def plot_scatter(ctx, outdir):
    # Returns
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(ctx["y_true"], ctx["y_pred"], s=4, alpha=0.35)
    lo = float(min(ctx["y_true"].min(), ctx["y_pred"].min()))
    hi = float(max(ctx["y_true"].max(), ctx["y_pred"].max()))
    ax.plot([lo,hi],[lo,hi],"r--",lw=1,label="y=x")
    # OLS fit line
    slope, intercept, r, *_ = sstats.linregress(ctx["y_true"], ctx["y_pred"])
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope*xs + intercept, "g-", lw=1, alpha=0.7,
            label=f"fit: slope={slope:.3f}, r={r:.3f}")
    ax.set_xlabel("actual log return"); ax.set_ylabel("predicted log return")
    ax.set_title(f"{ctx['name']}: returns scatter")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    savefig(fig, outdir / "03_scatter_returns.png")
    # Prices
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(ctx["true_price"], ctx["pred_price"], s=4, alpha=0.35)
    lo = float(min(ctx["true_price"].min(), ctx["pred_price"].min()))
    hi = float(max(ctx["true_price"].max(), ctx["pred_price"].max()))
    ax.plot([lo,hi],[lo,hi],"r--",lw=1,label="y=x")
    ax.set_xlabel("actual price"); ax.set_ylabel("predicted price")
    ax.set_title(f"{ctx['name']}: prices scatter")
    ax.grid(alpha=0.3); ax.legend()
    savefig(fig, outdir / "04_scatter_prices.png")

def plot_residuals(ctx, outdir):
    res = ctx["y_true"] - ctx["y_pred"]
    # over time
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(res, lw=0.5, alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{ctx['name']}: residuals over test sequence")
    ax.set_xlabel("test index"); ax.set_ylabel("residual (y_true - y_pred)")
    ax.grid(alpha=0.3)
    savefig(fig, outdir / "05_residuals_time.png")
    # histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(res, bins=60, density=True, alpha=0.7, color="#3498db")
    mu, sd = float(res.mean()), float(res.std())
    xs = np.linspace(res.min(), res.max(), 200)
    ax.plot(xs, sstats.norm.pdf(xs, mu, sd), "r-", lw=1.5,
            label=f"N({mu:.4f}, {sd:.4f})")
    ax.set_title(f"{ctx['name']}: residual distribution")
    ax.set_xlabel("residual"); ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, outdir / "06_residuals_hist.png")
    # Q-Q
    fig, ax = plt.subplots(figsize=(5,5))
    sstats.probplot(res, dist="norm", plot=ax)
    ax.set_title(f"{ctx['name']}: residual Q-Q plot")
    ax.grid(alpha=0.3)
    savefig(fig, outdir / "07_residuals_qq.png")

def plot_equity_curve(ctx, outdir):
    df = pd.DataFrame({"t": ctx["tickers"], "yt": ctx["y_true"], "yp": ctx["y_pred"]})
    fig, ax = plt.subplots(figsize=(10, 4))
    strat_curves, bh_curves = [], []
    for t, g in df.groupby("t"):
        sig = np.sign(g.yp.to_numpy())  # long/short signal
        strat = np.cumsum(sig * g.yt.to_numpy())
        bh    = np.cumsum(g.yt.to_numpy())
        strat_curves.append(strat); bh_curves.append(bh)
    # Pad to common length and average
    L = max(len(c) for c in strat_curves)
    def pad(c): return np.concatenate([c, np.full(L-len(c), np.nan)])
    strat_mean = np.nanmean([pad(c) for c in strat_curves], axis=0)
    bh_mean    = np.nanmean([pad(c) for c in bh_curves], axis=0)
    ax.plot(strat_mean, label="model signal (long/short)", lw=1.5)
    ax.plot(bh_mean,    label="buy & hold",                lw=1.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{ctx['name']}: cumulative log return (mean across tickers)")
    ax.set_xlabel("test step"); ax.set_ylabel("cumulative log return")
    ax.grid(alpha=0.3); ax.legend()
    savefig(fig, outdir / "08_equity_curve.png")

def plot_confusion(ctx, outdir):
    yt_dir = (ctx["y_true"] > 0).astype(int)
    yp_dir = (ctx["y_pred"] > 0).astype(int)
    cm = np.zeros((2,2), int)
    for a,p in zip(yt_dir, yp_dir): cm[a,p] += 1
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i,j), v in np.ndenumerate(cm):
        pct = v / cm.sum() * 100
        ax.text(j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                color="white" if v > cm.max()/2 else "black", fontsize=11)
    ax.set_xticks([0,1], ["pred down","pred up"])
    ax.set_yticks([0,1], ["actual down","actual up"])
    ax.set_title(f"{ctx['name']}: direction confusion")
    fig.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, outdir / "09_confusion_direction.png")

def plot_calibration(ctx, outdir):
    df = pd.DataFrame({"yp": ctx["y_pred"], "yt": ctx["y_true"]})
    df["bin"] = pd.qcut(df.yp, q=10, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        pred=("yp","mean"), actual=("yt","mean"),
        actual_std=("yt","std"), n=("yt","size"))
    fig, ax = plt.subplots(figsize=(6,5))
    ax.errorbar(g.pred, g.actual, yerr=g.actual_std/np.sqrt(g.n),
                fmt="o-", capsize=3, label="binned mean ± SE")
    lo = float(min(g.pred.min(), g.actual.min()))
    hi = float(max(g.pred.max(), g.actual.max()))
    ax.plot([lo,hi],[lo,hi],"r--",lw=1, label="perfect calibration")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("predicted log return (decile mean)")
    ax.set_ylabel("actual log return (decile mean)")
    ax.set_title(f"{ctx['name']}: calibration (decile binning)")
    ax.grid(alpha=0.3); ax.legend()
    savefig(fig, outdir / "10_calibration.png")

def plot_baselines(ctx, outdir):
    yt, yp = ctx["y_true"], ctx["y_pred"]
    # naive: predict zero, predict yesterday's actual return
    df = pd.DataFrame({"t": ctx["tickers"], "yt": yt, "yp": yp})
    df["yt_prev"] = df.groupby("t")["yt"].shift(1)
    df = df.dropna()
    da_model = float(np.mean(np.sign(df.yt) == np.sign(df.yp)) * 100)
    da_zero  = 50.0  # sign(0) = 0; treat as random
    da_mom   = float(np.mean(np.sign(df.yt) == np.sign(df.yt_prev)) * 100)
    mse_model = float(np.mean((df.yt - df.yp)**2))
    mse_zero  = float(np.mean(df.yt**2))
    mse_mom   = float(np.mean((df.yt - df.yt_prev)**2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ["model", "predict 0", "momentum"]
    axes[0].bar(names, [da_model, da_zero, da_mom],
                color=["#27ae60","#95a5a6","#95a5a6"])
    axes[0].axhline(50, color="k", ls="--", lw=1)
    axes[0].set_ylabel("DA (%)"); axes[0].set_title("Directional Accuracy")
    for i,v in enumerate([da_model, da_zero, da_mom]):
        axes[0].text(i, v+0.3, f"{v:.1f}%", ha="center")
    axes[1].bar(names, [mse_model, mse_zero, mse_mom],
                color=["#27ae60","#95a5a6","#95a5a6"])
    axes[1].set_ylabel("MSE"); axes[1].set_title("Log-return MSE (lower is better)")
    for i,v in enumerate([mse_model, mse_zero, mse_mom]):
        axes[1].text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{ctx['name']}: model vs naive baselines")
    fig.tight_layout()
    savefig(fig, outdir / "11_baselines.png")
    return dict(da_model=da_model, da_mom=da_mom,
                mse_model=mse_model, mse_mom=mse_mom)

def plot_shap(ctx, outdir):
    if not HAS_SHAP:
        log.info(f"[{ctx['name']}] shap not installed; skipping"); return
    try:
        # Sample for speed
        D = ctx["Dte"]
        sample = D.sample(n=min(500, len(D)), random_state=0)
        explainer = shap.TreeExplainer(ctx["xgb"])
        sv = explainer.shap_values(sample)
        plt.figure()
        shap.summary_plot(sv, sample, show=False, max_display=15)
        fig = plt.gcf(); fig.suptitle(f"{ctx['name']}: SHAP feature importance")
        savefig(fig, outdir / "12_shap_summary.png")
    except Exception as e:
        log.warning(f"[{ctx['name']}] SHAP failed: {e}")

def plot_latent_tsne(ctx, outdir):
    Z = ctx["Zte"]
    if len(Z) > 3000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(Z), 3000, replace=False)
    else:
        sel = np.arange(len(Z))
    Zs = Z[sel]
    log.info(f"[{ctx['name']}] running t-SNE on {len(Zs)} latents")
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=0, n_iter=1000).fit_transform(Zs)
    # by ticker
    tk = ctx["tickers"][sel]
    uniq = pd.unique(tk); cmap = plt.cm.tab20(np.linspace(0,1,len(uniq)))
    fig, ax = plt.subplots(figsize=(7,6))
    for c, t in zip(cmap, uniq):
        m = tk == t
        ax.scatter(emb[m,0], emb[m,1], s=8, alpha=0.6, color=c, label=str(t))
    ax.set_title(f"{ctx['name']}: latent t-SNE by ticker")
    if len(uniq) <= 20:
        ax.legend(fontsize=6, markerscale=1.5, loc="best", ncol=2)
    ax.set_xticks([]); ax.set_yticks([])
    savefig(fig, outdir / "13_latent_tsne_ticker.png")
    # by future direction
    sgn = np.sign(ctx["y_true"][sel])
    fig, ax = plt.subplots(figsize=(7,6))
    for v, color, lab in [(1,"#27ae60","up"),(-1,"#c0392b","down"),(0,"#95a5a6","flat")]:
        m = sgn == v
        if m.any():
            ax.scatter(emb[m,0], emb[m,1], s=8, alpha=0.6, color=color, label=lab)
    ax.set_title(f"{ctx['name']}: latent t-SNE by future return sign")
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    savefig(fig, outdir / "14_latent_tsne_direction.png")

# ─── Cross-sector summary ────────────────────────────────────────────────
def plot_cross_sector(baseline_results, da_ci_results, outdir):
    sectors = list(baseline_results.keys())
    da_m  = [baseline_results[s]["da_model"] for s in sectors]
    da_mo = [baseline_results[s]["da_mom"]   for s in sectors]
    x = np.arange(len(sectors)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x-w/2, da_m,  w, label="model",     color="#2980b9")
    ax.bar(x+w/2, da_mo, w, label="momentum",  color="#95a5a6")
    ax.axhline(50, color="k", ls="--", lw=1, label="random")
    ax.set_xticks(x, sectors); ax.set_ylabel("DA (%)")
    ax.set_title("Directional accuracy: model vs momentum baseline")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i,v in enumerate(da_m):  ax.text(i-w/2, v+0.3, f"{v:.1f}", ha="center", fontsize=8)
    for i,v in enumerate(da_mo): ax.text(i+w/2, v+0.3, f"{v:.1f}", ha="center", fontsize=8)
    savefig(fig, outdir / "per_sector_baseline_comparison.png")

    # DA with binomial 95% CI
    fig, ax = plt.subplots(figsize=(10, 4.5))
    das, los, his, ns = [], [], [], []
    for s in sectors:
        d = da_ci_results[s]
        das.append(d["da"]); los.append(d["lo"]); his.append(d["hi"]); ns.append(d["n"])
    das, los, his = np.array(das), np.array(los), np.array(his)
    ax.errorbar(x, das, yerr=[das-los, his-das], fmt="o", capsize=4,
                color="#2980b9", ms=7)
    ax.axhline(50, color="k", ls="--", lw=1, label="random (50%)")
    ax.set_xticks(x, sectors); ax.set_ylabel("DA (%)")
    ax.set_title("Directional accuracy with 95% binomial CI")
    for i,(d,n) in enumerate(zip(das, ns)):
        ax.text(i, d+0.4, f"n={n}", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.3); ax.legend()
    savefig(fig, outdir / "per_sector_da_with_ci.png")

def binomial_ci(k, n, alpha=0.05):
    if n == 0: return 0, 0
    lo, hi = sstats.beta.ppf([alpha/2, 1-alpha/2], [k, k+1], [n-k+1, n-k])
    return float(np.nan_to_num(lo)*100), float(np.nan_to_num(hi, nan=100)*100)

# ─── Driver ──────────────────────────────────────────────────────────────
def run_dataset(name, folder):
    outdir = FIGS / name
    outdir.mkdir(parents=True, exist_ok=True)
    ctx = reproduce(name, folder)
    log.info(f"[{name}] generating plots → {outdir}")
    plot_per_ticker_da(ctx, outdir)
    plot_price_overlays(ctx, outdir)
    plot_scatter(ctx, outdir)
    plot_residuals(ctx, outdir)
    plot_equity_curve(ctx, outdir)
    plot_confusion(ctx, outdir)
    plot_calibration(ctx, outdir)
    base = plot_baselines(ctx, outdir)
    plot_shap(ctx, outdir)
    plot_latent_tsne(ctx, outdir)
    # for cross-sector summary
    k = int(np.sum(np.sign(ctx["y_true"]) == np.sign(ctx["y_pred"])))
    n = len(ctx["y_true"])
    lo, hi = binomial_ci(k, n)
    return base, dict(da=k/n*100, lo=lo, hi=hi, n=n)

def main():
    baseline_results, da_ci = {}, {}
    for name, path in DATASETS.items():
        try:
            base, ci = run_dataset(name, path)
            baseline_results[name] = base
            da_ci[name] = ci
        except FileNotFoundError as e:
            log.warning(f"[{name}] skipped (missing file): {e}")
        except Exception as e:
            log.error(f"[{name}] failed: {e}", exc_info=True)
    if baseline_results:
        summary_dir = FIGS / "_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        plot_cross_sector(baseline_results, da_ci, summary_dir)
    log.info(f"Done. Figures in {FIGS}")

if __name__ == "__main__":
    main()