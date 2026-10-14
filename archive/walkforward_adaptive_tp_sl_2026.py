"""Frozen adaptive TP/SL model applied sequentially to 2026 MNQ signals.

The 2025 adaptive run initializes the 90-day history.  During 2026, each new
signal can use only earlier executed setups and only counterfactual exits that
were already complete at that timestamp.  Model menus, score, ATR regimes and
fallback are unchanged from backtest_adaptive_tp_sl_continuation_2025.py.
"""
from pathlib import Path
import json

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_equity2025_volatility_regimes import add_volatility_features
from backtest_impulso_mnq_multi_is import utc_naive
from backtest_adaptive_tp_sl_continuation_2025 import (
    ALL_PAIRS, BASE_PAIR, COST_POINTS, DATA, LOOKBACK_DAYS, MIN_HISTORY,
    MINUTES, POINT_VALUE, REGIME_MENUS, ROOT, TRADES, metrics, scan_outcomes,
)


OUT = ROOT / "outputs/impulso_mnq/adaptive_tp_sl_continuation_wf_2026"
TRAIN_OUT = ROOT / "outputs/impulso_mnq/adaptive_tp_sl_continuation_2025"


def build_all_signals():
    d = pd.read_parquet(TRADES)
    d = d[
        d["sample"].isin(["VALIDATION_2025", "DIAGNOSTIC_2026"])
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


def make_2026_grid(signals):
    rows = []
    for signal in signals[signals["sample"].eq("DIAGNOSTIC_2026")].itertuples():
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


def calculate_2026_tick_grid(grid):
    folders = sorted(
        p for p in DATA.glob("calendar_date=*") if p.name.split("=", 1)[1] >= "2026-01-01"
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


def walkforward_2026(signals, grid_2025, grid_2026, train_trades):
    grid = pd.concat([grid_2025, grid_2026], ignore_index=True)
    lookup = grid.set_index(["signal_id", "tp", "sl"]).sort_index()
    executed_ids = train_trades.signal_id.astype(int).tolist()
    selected_rows = []
    last_exit = pd.to_datetime(train_trades.exit_time_utc).max()
    test = signals[signals["sample"].eq("DIAGNOSTIC_2026")]

    for signal in test.itertuples():
        if signal.entry_time_utc <= last_exit:
            continue
        scores = []
        window_start = signal.entry_time_utc - pd.Timedelta(days=LOOKBACK_DAYS)
        for tp, sl in REGIME_MENUS[signal.vol_regime]:
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
                scores.append((values.mean() - 0.5 * se, len(values), tp, sl, values.mean(), se))
        if scores:
            score, n_hist, tp, sl, hist_mean, hist_se = max(scores, key=lambda x: (x[0], -x[3], x[2]))
            source = "adaptive"
        else:
            tp, sl = BASE_PAIR
            score, n_hist, hist_mean, hist_se = np.nan, 0, np.nan, np.nan
            source = "fallback_base"
        result = lookup.loc[(signal.signal_id, tp, sl)]
        selected_rows.append({
            **signal._asdict(), "selected_tp": tp, "selected_sl": sl,
            "selected_rr": tp / sl, "selection_source": source,
            "history_n": n_hist, "selection_score": score,
            "history_mean_net_points": hist_mean, "history_se": hist_se,
            "exit_time_utc": result.exit_time_utc, "exit_reason": result.exit_reason,
            "gross_points": result.gross_points, "net_points": result.net_points,
            "pnl_usd": result.pnl_usd,
        })
        executed_ids.append(signal.signal_id)
        last_exit = result.exit_time_utc
    return pd.DataFrame(selected_rows)


def fixed_2026(signals, grid_2026):
    q = grid_2026[(grid_2026.tp == 10) & (grid_2026.sl == 30)].set_index("signal_id")
    rows, last_exit = [], pd.Timestamp.min
    for signal in signals[signals["sample"].eq("DIAGNOSTIC_2026")].itertuples():
        if signal.entry_time_utc <= last_exit:
            continue
        result = q.loc[signal.signal_id]
        rows.append({**signal._asdict(), **result[["exit_time_utc", "exit_reason", "gross_points", "net_points", "pnl_usd"]].to_dict()})
        last_exit = result.exit_time_utc
    return pd.DataFrame(rows)


def monthly_metrics(q):
    x = q.copy()
    x["month"] = x.entry_time_rome.dt.strftime("%Y-%m")
    rows = []
    for month, g in x.groupby("month"):
        rows.append({"month": month, **metrics(g)})
    return pd.DataFrame(rows)


def plot_total(train, test, fixed):
    combined = pd.concat([train, test], ignore_index=True).sort_values("entry_time_utc")
    combined["equity_usd"] = combined.pnl_usd.cumsum()
    fixed_all = pd.concat([
        pd.read_csv(TRAIN_OUT / "fixed_tp10_sl30_trades_2025.csv"), fixed
    ], ignore_index=True)
    fixed_all["entry_time_rome"] = pd.to_datetime(fixed_all.entry_time_rome, utc=True).dt.tz_convert("Europe/Rome")
    fixed_all["equity_usd"] = fixed_all.pnl_usd.cumsum()

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=150)
    fig.patch.set_facecolor("#111315")
    ax.set_facecolor("#111315")
    ax.plot(fixed_all.entry_time_rome, fixed_all.equity_usd, color="#8b949e", linewidth=1.6, label="Fissa TP10/SL30")
    ax.plot(combined.entry_time_rome, combined.equity_usd, color="#f4c95d", linewidth=1.8, label="Adattiva WF")
    boundary = pd.Timestamp("2026-01-01", tz="Europe/Rome")
    ax.axvline(boundary, color="#ff7b72", linestyle="--", linewidth=1.2, label="Inizio WF 2026")
    ax.axhline(0, color="#8a9199", linewidth=.8)
    ax.set_title("MNQ — TP/SL adattivi: 2025 + walk-forward 2026", loc="left", fontsize=17, pad=16)
    ax.set_ylabel("P&L cumulativo netto ($)")
    ax.set_xlabel("Data")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, color="#30363d", linewidth=.6, alpha=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "equity_total_2025_wf2026.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    combined.to_csv(OUT / "adaptive_combined_2025_wf2026.csv", index=False)


def plot_months(test, monthly):
    months = sorted(test.entry_time_rome.dt.strftime("%Y-%m").unique())
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), dpi=150)
    fig.patch.set_facecolor("#111315")
    for ax in axes.flat:
        ax.set_facecolor("#111315")
        ax.grid(True, color="#30363d", linewidth=.5, alpha=.65)
        ax.spines[["top", "right"]].set_visible(False)
    for ax, month in zip(axes.flat, months):
        g = test[test.entry_time_rome.dt.strftime("%Y-%m").eq(month)].copy()
        g["month_equity"] = g.pnl_usd.cumsum()
        stat = monthly[monthly.month.eq(month)].iloc[0]
        ax.plot(g.entry_time_rome, g.month_equity, color="#58d6a5", linewidth=1.7)
        ax.axhline(0, color="#8a9199", linewidth=.7)
        ax.set_title(month, loc="left", fontsize=12)
        ax.text(
            .03, .94,
            f"N {int(stat.trades)} | WR {stat.win_rate_pct:.1f}%\nPF {stat.profit_factor_net:.2f} | ${stat.net_usd:+.0f}\nDD ${stat.max_drawdown_usd:.0f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=.35", facecolor="#1b1f23", edgecolor="#3b4249", alpha=.92),
        )
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
    for ax in axes.flat[len(months):]:
        ax.axis("off")
    fig.suptitle("MNQ 2026 walk-forward — equity mensile TP/SL adattivi", fontsize=17, x=.08, ha="left")
    fig.supxlabel("Giorno del mese")
    fig.supylabel("P&L mensile cumulativo ($)")
    fig.tight_layout(rect=[0.02, .02, 1, .95])
    fig.savefig(OUT / "equity_monthly_wf2026.png", bbox_inches="tight", facecolor=fig.get_facecolor())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    signals = build_all_signals()
    train_signals = signals[signals["sample"].eq("VALIDATION_2025")]
    old_grid = pd.read_parquet(TRAIN_OUT / "counterfactual_exit_grid_2025.parquet")
    # 2025 IDs are expected to be identical to the frozen training run.
    if len(train_signals) != old_grid.signal_id.nunique():
        raise RuntimeError("2025 signal IDs no longer align with the frozen training grid")
    grid_2026 = calculate_2026_tick_grid(make_2026_grid(signals))
    if grid_2026.exit_reason.eq("open").any():
        raise RuntimeError("Some 2026 counterfactual exits are unresolved")

    train = pd.read_csv(TRAIN_OUT / "adaptive_trades_2025.csv")
    train["entry_time_utc"] = pd.to_datetime(train.entry_time_utc)
    train["exit_time_utc"] = pd.to_datetime(train.exit_time_utc)
    train["entry_time_rome"] = pd.to_datetime(train.entry_time_rome, utc=True).dt.tz_convert("Europe/Rome")
    test = walkforward_2026(signals, old_grid, grid_2026, train)
    fixed = fixed_2026(signals, grid_2026)
    test["equity_usd"] = test.pnl_usd.cumsum()
    fixed["equity_usd"] = fixed.pnl_usd.cumsum()

    monthly = monthly_metrics(test)
    test.to_csv(OUT / "adaptive_wf_trades_2026.csv", index=False)
    fixed.to_csv(OUT / "fixed_tp10_sl30_trades_2026.csv", index=False)
    monthly.to_csv(OUT / "monthly_stats_2026.csv", index=False)
    grid_2026.to_parquet(OUT / "counterfactual_exit_grid_2026.parquet", index=False)
    report = {
        "method": (
            "Frozen 2025 adaptive exit model applied sequentially in 2026. Prior 90-day executed "
            "same-regime setups only; counterfactual exits must be complete before entry. No 2026 "
            "parameter change. Tick exact, stop slippage, one position, one MNQ, $1 RT."
        ),
        "adaptive_wf_2026": metrics(test),
        "fixed_tp10_sl30_2026": metrics(fixed),
        "adaptive_2025_plus_wf2026": metrics(pd.concat([train, test], ignore_index=True)),
        "monthly_2026": monthly.replace({np.nan: None}).to_dict("records"),
        "adaptive_selections": int(test.selection_source.eq("adaptive").sum()),
        "fallback_selections": int(test.selection_source.eq("fallback_base").sum()),
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_total(train, test, fixed)
    plot_months(test, monthly)
    print(json.dumps({k: report[k] for k in ("adaptive_wf_2026", "fixed_tp10_sl30_2026", "adaptive_2025_plus_wf2026")}, indent=2))
    print("\nMONTHLY")
    print(monthly.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
