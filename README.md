# Indian Market Stage-wise Robo-Advisor (Colab-ready)

This implementation delivers a **4-stage robo-advisor** for Indian-market assets, with:
- **ML-based Stage-1 recommendation filtering**
- **Dynamic SQLite weekly-return store** (rolling 52 weeks)
- **Correlation + low-correlation pair discovery**
- **1000-epoch Monte Carlo 2-asset allocation**
- **Bear/Base/Bull projection vs NIFTY**

## Stage 1: 300+ physical CSV with yfinance tickers
A **physically created** local file is provided at:
- `data/assets_master_300_plus.csv`

It contains **418 assets** with real ticker formats usable in yfinance, including:
- NSE equities (`.NS`) across large/mid/small caps and SME names
- REIT/InvIT tickers
- ETF tickers
- Crypto INR tickers (`-INR`)

Filtering uses an ML suitability model (logistic regression) based on user preferences and asset features.

## Stage 2: Dynamic SQL weekly statistics
Data is persisted in:
- `data/market_returns.db` (SQLite)

`weekly_returns` table stores weekly returns by symbol/date. On refresh:
1. latest (or fallback synthetic) weekly returns are upserted,
2. oldest rows are deleted,
3. each symbol keeps only latest **52 weekly points**.

Mean and standard deviation are computed directly from SQL.

## Stage 3: Monte Carlo allocation
- Uses least-correlated pair + next best 3 pairs.
- Runs **1000 epochs per pair**.
- Recommends allocation with best return/risk score.

## Stage 4: Scenario projection vs NIFTY
For user horizon, simulates and reports:
- Bear case (20th percentile)
- Base case (50th percentile)
- Bull case (80th percentile)

## Run (Colab/local)
```bash
pip install numpy pandas yfinance scikit-learn
python roboadvisor_india.py
```
