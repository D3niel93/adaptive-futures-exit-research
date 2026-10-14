"""Closed-trade analytics. PnL is already net; never subtract costs twice."""
import math
from pathlib import Path
import numpy as np
import pandas as pd

def metrics(pnl):
    a=np.asarray(pnl,dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Require nonempty finite net PnL")
    equity=np.r_[0.0,a.cumsum()]
    gains=a[a>0].sum(); losses=-a[a<0].sum()
    return dict(trades=len(a),win_rate_pct=float((a>0).mean()*100),
        profit_factor=float(gains/losses) if losses else None,
        expectancy_usd=float(a.mean()),net_usd=float(a.sum()),
        max_closed_trade_drawdown_usd=float((np.maximum.accumulate(equity)-equity).max()))

def daily_sharpe(frame):
    # Zero-filled weekdays; exchange holiday calendar is not available here.
    daily=frame.set_index("time").pnl.resample("B").sum()
    sd=daily.std(ddof=1)
    return float(daily.mean()/sd*math.sqrt(252)) if sd>0 else None

def run():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(__file__).resolve().parent
    f=pd.read_csv(root/"data/trades.csv")
    f["time"]=pd.to_datetime(f.time,utc=True,errors="raise")
    if f.time.isna().any() or not np.isfinite(f.pnl).all():
        raise ValueError("Invalid ledger")
    f=f.sort_values("time",kind="stable")
    out=root/"reports";out.mkdir(exist_ok=True)
    records=[]
    groups=[("TOTAL",f)]+list(f.groupby("strategy",sort=True))
    for name,g in groups:
        records.append(dict(strategy=name,start=str(g.time.min()),end=str(g.time.max()),
          daily_sharpe_business_day_proxy=daily_sharpe(g),**metrics(g.pnl)))
    pd.DataFrame(records).to_csv(out/"summary.csv",index=False)
    byyear=[]
    for year,g in f.groupby(f.time.dt.year):
        byyear.append(dict(year=year,**metrics(g.pnl)))
    pd.DataFrame(byyear).to_csv(out/"by_year.csv",index=False)
    f.assign(month=f.time.dt.strftime("%Y-%m")).groupby(["month","strategy"]).pnl.agg(
        ["count","sum","mean"]).to_csv(out/"monthly.csv")
    daily=f.pivot_table(index=f.time.dt.floor("D"),columns="strategy",values="pnl",aggfunc="sum",fill_value=0)
    daily=daily.reindex(pd.bdate_range(daily.index.min(),daily.index.max()),fill_value=0)
    daily.corr().to_csv(out/"daily_correlation.csv")
    fig,ax=plt.subplots(figsize=(11,5))
    for name,g in groups:
        times=[g.time.iloc[0]]+list(g.time)
        equity=np.r_[0.,g.pnl.cumsum().to_numpy()]
        ax.step(times,equity,where="post",label=name,linewidth=2 if name=="TOTAL" else 1)
    ax.set(title="Provisional backtest: closed-trade equity, modeled costs included",
      ylabel="Cumulative USD (not percentage return)")
    ax.legend(fontsize=7);ax.grid(alpha=.25);fig.autofmt_xdate()
    fig.tight_layout();fig.savefig(out/"equity.png",dpi=150);plt.close(fig)
    print(pd.DataFrame(records).to_string(index=False))

if __name__=="__main__":
    run()
