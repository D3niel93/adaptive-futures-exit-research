"""Causal adaptive TP/SL test for the frozen MNQ continuation signal in 2025.

The entry model is never changed: impulse40 continuation, causal ADX14 30--35,
causal session range 239--407.  At each executable signal, ATR14 classifies the
current market as LOW/NORMAL/HIGH using expanding trailing-90-day tertiles.
Within a regime-appropriate TP/SL menu, the exit pair is selected from prior
executed setups of the same regime whose counterfactual exits are already known.

Selection score = mean net points - 0.5 standard error.  At least eight prior
observations are required; otherwise TP10/SL30 is used.  Every calculation is
causal.  Tick outcomes include stop slippage and $1 round trip per MNQ.
"""
from pathlib import Path
import json

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange

from analyze_equity2025_volatility_regimes import add_volatility_features
from backtest_impulso_mnq_multi_is import utc_naive


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "tick_backtest_mnq/cache/tick_parquet"
TRADES = ROOT / "outputs/impulso_mnq/impulse40_continuation_adx/trades_with_causal_adx.parquet"
MINUTES = ROOT / "research/derived/MNQ_1m_ohlcv_delta_full.parquet"
OUT = ROOT / "outputs/impulso_mnq/adaptive_tp_sl_continuation_2025"

LOOKBACK_DAYS = 90
MIN_HISTORY = 8
BASE_PAIR = (10, 30)
COST_POINTS = 0.5
POINT_VALUE = 2.0

# Wider volatility permits wider exits. RR is deliberately allowed to change.
REGIME_MENUS = {
    "LOW": [(5, 15), (10, 15), (10, 20), (10, 30), (15, 20), (15, 30)],
    "NORMAL": [(10, 20), (10, 30), (15, 25), (15, 30), (15, 40), (20, 30), (20, 40)],
    "HIGH": [(10, 30), (15, 30), (15, 40), (20, 30), (20, 40), (20, 45), (25, 40), (25, 45)],
}
ALL_PAIRS = sorted(set([BASE_PAIR] + [pair for menu in REGIME_MENUS.values() for pair in menu]))


@njit(parallel=True, cache=True)
def scan_outcomes(prices, starts, entries, directions, target_q, stop_q):
    n = len(entries)
    reasons = np.zeros(n, dtype=np.int8)
    pnl_q = np.zeros(n, dtype=np.int32)
    hit_idx = np.full(n, -1, dtype=np.int64)
    for k in prange(n):
        entry = int(entries[k])
        direction = int(directions[k])
        tp = int(target_q[k])
        sl = int(stop_q[k])
        for j in range(int(starts[k]), len(prices)):
            move = direction * (int(prices[j]) - entry)
            if move >= tp:
                reasons[k] = 1
                pnl_q[k] = tp
                hit_idx[k] = j
                break
            if move <= -sl:
                reasons[k] = -1
                pnl_q[k] = move  # first observed tick: preserves stop slippage
                hit_idx[k] = j
                break
    return reasons, pnl_q, hit_idx


def build_signals():
    d = pd.read_parquet(TRADES)
    d = d[
        d["sample"].eq("VALIDATION_2025")
        & d.target_points.eq(10)
        & d.adx14.ge(30)
        & d.adx14.lt(35)
    ].copy()
    d["entry_time_utc"] = pd.to_datetime(d.entry_time_utc).dt.as_unit("ns")
    d["entry_time_rome"] = pd.to_datetime(d.entry_time_rome, utc=True).dt.tz_convert("Europe/Rome")

    bars = add_volatility_features(pd.read_parquet(MINUTES))
    bars["feature_time_utc"] = pd.to_datetime(bars.bar_end_utc).dt.as_unit("ns")
    bars = bars[[
        "feature_time_utc", "session_date", "session_range_so_far", "atr14_points",
        "rvol20", "avg_range_15m", "realized_std_15m",
    ]].rename(columns={"session_date": "feature_session_date"})
    d = pd.merge_asof(
        d.sort_values("entry_time_utc"), bars.sort_values("feature_time_utc"),
        left_on="entry_time_utc", right_on="feature_time_utc",
        direction="backward", allow_exact_matches=True,
    )
    same = d.date.astype(str).eq(d.feature_session_date.astype(str))
    d = d[same & d.session_range_so_far.between(239.0, 407.0, inclusive="both")].copy()
    d = d.sort_values(["entry_time_utc", "event_id"]).reset_index(drop=True)
    d["signal_id"] = np.arange(len(d), dtype=np.int64)

    # Regime thresholds themselves use only earlier signals from the trailing window.
    regimes, q33s, q67s = [], [], []
    for row in d.itertuples():
        start = row.entry_time_utc - pd.Timedelta(days=LOOKBACK_DAYS)
        past = d[(d.entry_time_utc < row.entry_time_utc) & (d.entry_time_utc >= start)].atr14_points.dropna()
        if len(past) < 20 or pd.isna(row.atr14_points):
            regime, q33, q67 = "NORMAL", np.nan, np.nan
        else:
            q33, q67 = past.quantile([1 / 3, 2 / 3]).to_numpy(float)
            regime = "LOW" if row.atr14_points < q33 else ("HIGH" if row.atr14_points > q67 else "NORMAL")
        regimes.append(regime)
        q33s.append(q33)
        q67s.append(q67)
    d["vol_regime"] = regimes
    d["causal_atr_q33"] = q33s
    d["causal_atr_q67"] = q67s
    return d


def make_grid(signals):
    rows = []
    for signal in signals.itertuples():
        for tp, sl in ALL_PAIRS:
            rows.append({
                "signal_id": signal.signal_id, "date": signal.date,
                "entry_time_utc": signal.entry_time_utc,
                "entry_tick_index_session": signal.entry_tick_index_session,
                "entry_q": signal.entry_q, "trade_direction": signal.trade_direction,
                "tp": tp, "sl": sl,
            })
    grid = pd.DataFrame(rows)
    grid["row_id"] = np.arange(len(grid), dtype=np.int64)
    return grid


def calculate_tick_grid(grid):
    folders = sorted(
        p for p in DATA.glob("calendar_date=*")
        if "2025-01-01" <= p.name.split("=", 1)[1] <= "2026-01-31"
    )
    by_date = {k: v.row_id.to_numpy(np.int64) for k, v in grid.groupby("date")}
    reasons = np.zeros(len(grid), dtype=np.int8)
    pnl_q = np.zeros(len(grid), dtype=np.int32)
    exit_ms = np.full(len(grid), -1, dtype=np.int64)
    pending = np.empty(0, dtype=np.int64)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    for folder in folders:
        date = folder.name.split("=", 1)[1]
        file = str(folder / "*.parquet").replace("\\", "/")
        raw = con.execute(
            "SELECT ts,price_ticks FROM read_parquet(?) WHERE volume>0 ORDER BY ts", [file]
        ).fetchnumpy()
        times = raw["ts"].astype("datetime64[ms]").astype(np.int64)
        prices = raw["price_ticks"].astype(np.int64)
        if not len(prices):
            continue

        if len(pending):
            r, p, h = scan_outcomes(
                prices, np.zeros(len(pending), dtype=np.int64),
                grid.loc[pending, "entry_q"].to_numpy(np.int64),
                grid.loc[pending, "trade_direction"].to_numpy(np.int8),
                (grid.loc[pending, "tp"] * 4).to_numpy(np.int32),
                (grid.loc[pending, "sl"] * 4).to_numpy(np.int32),
            )
            done = r != 0
            ids = pending[done]
            reasons[ids], pnl_q[ids], exit_ms[ids] = r[done], p[done], times[h[done]]
            pending = pending[~done]

        ids = by_date.get(date)
        if ids is not None and len(ids):
            session_start = int(utc_naive(pd.Timestamp(date).date(), 15, 30).value // 1_000_000)
            offset = int(np.searchsorted(times, session_start, side="left"))
            starts = offset + grid.loc[ids, "entry_tick_index_session"].to_numpy(np.int64) + 1
            r, p, h = scan_outcomes(
                prices, starts,
                grid.loc[ids, "entry_q"].to_numpy(np.int64),
                grid.loc[ids, "trade_direction"].to_numpy(np.int8),
                (grid.loc[ids, "tp"] * 4).to_numpy(np.int32),
                (grid.loc[ids, "sl"] * 4).to_numpy(np.int32),
            )
            done = r != 0
            done_ids = ids[done]
            reasons[done_ids], pnl_q[done_ids], exit_ms[done_ids] = r[done], p[done], times[h[done]]
            pending = np.r_[pending, ids[~done]]
    con.close()

    grid["exit_reason"] = np.where(reasons == 1, "tp", np.where(reasons == -1, "sl", "open"))
    grid["gross_points"] = pnl_q / 4.0
    grid["net_points"] = grid.gross_points - COST_POINTS
    grid["pnl_usd"] = grid.net_points * POINT_VALUE
    grid["exit_time_utc"] = pd.to_datetime(
        np.where(exit_ms >= 0, exit_ms, np.datetime64("NaT", "ms").astype(np.int64)), unit="ms"
    )
    return grid


def select_adaptively(signals, grid):
    lookup = grid.set_index(["signal_id", "tp", "sl"]).sort_index()
    executed_ids = []
    selected_rows = []
    last_exit = pd.Timestamp.min

    for signal in signals.itertuples():
        if signal.entry_time_utc <= last_exit:
            continue
        menu = REGIME_MENUS[signal.vol_regime]
        window_start = signal.entry_time_utc - pd.Timedelta(days=LOOKBACK_DAYS)
        scores = []
        for tp, sl in menu:
            hist = []
            for old_id in executed_ids:
                old_signal = signals.loc[old_id]
                if old_signal.vol_regime != signal.vol_regime or old_signal.entry_time_utc < window_start:
                    continue
                old = lookup.loc[(old_id, tp, sl)]
                if old.exit_time_utc < signal.entry_time_utc:
                    hist.append(float(old.net_points))
            if len(hist) >= MIN_HISTORY:
                values = np.asarray(hist, dtype=float)
                se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                score = values.mean() - 0.5 * se
                scores.append((score, len(values), tp, sl, values.mean(), se))

        if scores:
            score, n_hist, tp, sl, hist_mean, hist_se = max(scores, key=lambda x: (x[0], -x[3], x[2]))
            selection_source = "adaptive"
        else:
            tp, sl = BASE_PAIR
            score, n_hist, hist_mean, hist_se = np.nan, 0, np.nan, np.nan
            selection_source = "fallback_base"

        result = lookup.loc[(signal.signal_id, tp, sl)]
        selected_rows.append({
            **signal._asdict(), "selected_tp": tp, "selected_sl": sl,
            "selected_rr": tp / sl, "selection_source": selection_source,
            "history_n": n_hist, "selection_score": score,
            "history_mean_net_points": hist_mean, "history_se": hist_se,
            "exit_time_utc": result.exit_time_utc, "exit_reason": result.exit_reason,
            "gross_points": result.gross_points, "net_points": result.net_points,
            "pnl_usd": result.pnl_usd,
        })
        executed_ids.append(signal.signal_id)
        last_exit = result.exit_time_utc
    return pd.DataFrame(selected_rows)


def fixed_base(signals, grid):
    q = grid[(grid.tp == BASE_PAIR[0]) & (grid.sl == BASE_PAIR[1])].set_index("signal_id")
    rows, last_exit = [], pd.Timestamp.min
    for signal in signals.itertuples():
        if signal.entry_time_utc <= last_exit:
            continue
        result = q.loc[signal.signal_id]
        rows.append({**signal._asdict(), **result[["exit_time_utc", "exit_reason", "gross_points", "net_points", "pnl_usd"]].to_dict()})
        last_exit = result.exit_time_utc
    return pd.DataFrame(rows)


def metrics(q):
    pnl = q.pnl_usd.to_numpy(float)
    pos, neg = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = np.cumsum(pnl)
    full = np.r_[0.0, eq]
    dd = (np.maximum.accumulate(full) - full).max()
    daily = q.groupby(q.entry_time_rome.dt.date).pnl_usd.sum()
    return {
        "trades": int(len(q)), "wins": int((pnl > 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor_net": float(pos / neg) if neg else None,
        "net_usd": float(pnl.sum()), "avg_trade_usd": float(pnl.mean()),
        "max_drawdown_usd": float(dd),
        "daily_sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    signals = build_signals()
    grid = calculate_tick_grid(make_grid(signals))
    adaptive = select_adaptively(signals, grid)
    fixed = fixed_base(signals, grid)

    adaptive["equity_usd"] = adaptive.pnl_usd.cumsum()
    fixed["equity_usd"] = fixed.pnl_usd.cumsum()
    adaptive.to_csv(OUT / "adaptive_trades_2025.csv", index=False)
    fixed.to_csv(OUT / "fixed_tp10_sl30_trades_2025.csv", index=False)
    grid.to_parquet(OUT / "counterfactual_exit_grid_2025.parquet", index=False)

    pair_usage = (
        adaptive.groupby(["vol_regime", "selected_tp", "selected_sl"])
        .agg(trades=("pnl_usd", "size"), net_usd=("pnl_usd", "sum"), win_rate_pct=("pnl_usd", lambda x: (x > 0).mean() * 100))
        .reset_index()
    )
    pair_usage.to_csv(OUT / "pair_usage_by_regime.csv", index=False)
    report = {
        "method": (
            "2025 causal sequential test. Fixed impulse40 continuation entries, ADX14 30-35, "
            "session range 239-407. Expanding trailing-90-day ATR tertile regime. TP/SL selected "
            "from prior executed same-regime setups with exits known before entry; score mean net "
            "points minus 0.5 standard error; minimum 8 observations; TP10/SL30 fallback. Tick exact, "
            "stop slippage, one position, one MNQ, $1 RT."
        ),
        "adaptive": metrics(adaptive), "fixed_tp10_sl30": metrics(fixed),
        "signals": int(len(signals)),
        "adaptive_selections": int(adaptive.selection_source.eq("adaptive").sum()),
        "fallback_selections": int(adaptive.selection_source.eq("fallback_base").sum()),
        "pair_usage": pair_usage.to_dict("records"),
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#111315")
    for ax in axes:
        ax.set_facecolor("#111315")
        ax.grid(True, color="#30363d", linewidth=.6, alpha=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot(fixed.entry_time_rome, fixed.equity_usd, color="#8b949e", linewidth=1.7, label="Fissa TP10 / SL30")
    axes[0].plot(adaptive.entry_time_rome, adaptive.equity_usd, color="#f4c95d", linewidth=1.8, label="TP/SL adattivi")
    axes[0].axhline(0, color="#8a9199", linewidth=.8)
    axes[0].set_title("MNQ 2025 — continuation con TP/SL adattivi", loc="left", fontsize=17, pad=16)
    axes[0].set_ylabel("P&L cumulativo netto ($)")
    axes[0].legend(loc="upper left", frameon=False)

    regime_code = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
    axes[1].scatter(adaptive.entry_time_rome, adaptive.vol_regime.map(regime_code), c=adaptive.selected_rr, cmap="viridis", s=18)
    axes[1].set_yticks([0, 1, 2], ["Vol bassa", "Vol normale", "Vol alta"])
    axes[1].set_ylabel("Regime / colore RR")
    axes[1].set_xlabel("Data")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    fig.savefig(OUT / "adaptive_vs_fixed_equity_2025.png", bbox_inches="tight", facecolor=fig.get_facecolor())

    print(json.dumps({k: report[k] for k in ("adaptive", "fixed_tp10_sl30", "signals", "adaptive_selections", "fallback_selections")}, indent=2))
    print("\nPAIR USAGE")
    print(pair_usage.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
