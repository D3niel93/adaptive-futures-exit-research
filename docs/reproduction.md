# Ledger and execution assumptions

`data/trades.csv` contains actual derived backtest outcomes, no raw market prices. time = exit_time_utc; pnl = original modeled net USD for one MNQ. entry_time_utc and direction, volatility regime, selected TP/SL and historical sample count are retained for audit.

Dates run 2025-01-02 through 2026-08-24. Publication analytics attribute PnL to exit time; old internal summaries sometimes used entry-date attribution. UTC day attribution is explicit; it is not exchange settlement accounting.

TP uses the configured price level, not a favorable tick overshoot. SL uses first observed qualifying tick, permitting worse fills. Modeled RT fee $1/MNQ = 0.5 points. MNQ tick size 0.25; tick value $0.50; point value $2. Queue position, exchange/broker fees beyond the stated model and rejected orders are not reproduced in this companion.

`learner.py` is a portable reimplementation of the score/causality rule. Unit tests prove narrow invariants, not full agreement with every historical decision. Archive scripts need private ticks, event caches, supporting local modules and numba/duckdb/pyarrow dependencies; the public report does not require them.
