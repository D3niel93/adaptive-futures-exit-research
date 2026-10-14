"""Causal companion selector, independent of entry/backtest implementation."""
from datetime import timedelta
from math import sqrt
from statistics import mean,stdev

def select_exit(history,now,regime,candidates,min_samples=8):
    scores=[]
    for tp,sl in candidates:
        values=[float(x["net_points"]) for x in history
            if now-timedelta(days=90)<=x["entry"]<now and x["exit"]<now
            and x["regime"]==regime and (x["tp"],x["sl"])==(tp,sl)]
        if len(values)>=min_samples:
            score=mean(values)-0.5*stdev(values)/sqrt(len(values))
            scores.append((score,-sl,tp))
    return (max(scores)[2],-max(scores)[1]) if scores else (10,30)
