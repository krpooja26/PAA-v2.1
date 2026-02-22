 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/roboadvisor_india.py b/roboadvisor_india.py
new file mode 100644
index 0000000000000000000000000000000000000000..70984b014e38b5a55fcd388cc82ec5d56580f0c4
--- /dev/null
+++ b/roboadvisor_india.py
@@ -0,0 +1,389 @@
+import itertools
+import math
+import sqlite3
+from dataclasses import dataclass
+from pathlib import Path
+from typing import Dict, List, Optional, Tuple
+
+import numpy as np
+import pandas as pd
+
+try:
+    import yfinance as yf
+except Exception:  # pragma: no cover
+    yf = None
+
+try:
+    from sklearn.compose import ColumnTransformer
+    from sklearn.linear_model import LogisticRegression
+    from sklearn.pipeline import Pipeline
+    from sklearn.preprocessing import OneHotEncoder, StandardScaler
+except Exception:  # pragma: no cover
+    ColumnTransformer = None
+    LogisticRegression = None
+    Pipeline = None
+    OneHotEncoder = None
+    StandardScaler = None
+
+
+DATA_DIR = Path("data")
+ASSET_MASTER_FILE = DATA_DIR / "assets_master_300_plus.csv"
+DB_FILE = DATA_DIR / "market_returns.db"
+
+
+@dataclass
+class ClientPreferences:
+    age: int
+    risk_tolerance: str
+    amount_to_invest: float
+    esg_preference: bool
+    excluded_categories: List[str]
+    allow_fno: bool
+    investment_horizon_years: int
+
+
+def _risk_ceiling(risk_tolerance: str) -> int:
+    return {"low": 3, "medium": 6, "high": 9}.get(risk_tolerance.lower(), 6)
+
+
+def create_asset_master_300_plus() -> pd.DataFrame:
+    DATA_DIR.mkdir(parents=True, exist_ok=True)
+    if not ASSET_MASTER_FILE.exists():
+        raise FileNotFoundError(
+            f"Expected physical asset master CSV at {ASSET_MASTER_FILE}."
+        )
+
+    df = pd.read_csv(ASSET_MASTER_FILE)
+    required = {
+        "symbol",
+        "security_name",
+        "asset_type",
+        "market_cap_bucket",
+        "risk_score",
+        "esg_score",
+        "fno_available",
+        "india_tradeable",
+    }
+    missing = required - set(df.columns)
+    if missing:
+        raise ValueError(f"Asset master is missing required columns: {sorted(missing)}")
+
+    if len(df) < 300:
+        raise ValueError("Asset master must contain at least 300 assets.")
+
+    return df
+
+
+class WeeklyReturnSQLStore:
+    def __init__(self, db_file: Path) -> None:
+        self.db_file = db_file
+        self.conn = sqlite3.connect(db_file)
+        self.conn.execute("PRAGMA journal_mode=WAL;")
+        self._create_schema()
+
+    def _create_schema(self) -> None:
+        self.conn.execute(
+            """
+            CREATE TABLE IF NOT EXISTS weekly_returns (
+                symbol TEXT NOT NULL,
+                week_end_date TEXT NOT NULL,
+                weekly_return REAL NOT NULL,
+                PRIMARY KEY (symbol, week_end_date)
+            );
+            """
+        )
+        self.conn.commit()
+
+    def close(self) -> None:
+        self.conn.close()
+
+    def _fetch_weekly_returns(self, symbol: str) -> pd.DataFrame:
+        if yf is None:
+            return self._synthetic_returns(symbol)
+        try:
+            hist = yf.download(symbol, period="1y", interval="1wk", auto_adjust=True, progress=False)
+            if hist.empty or "Close" not in hist:
+                return self._synthetic_returns(symbol)
+            r = hist["Close"].pct_change().dropna()
+            return pd.DataFrame(
+                {
+                    "week_end_date": pd.to_datetime(r.index).tz_localize(None).strftime("%Y-%m-%d"),
+                    "weekly_return": r.values.astype(float),
+                }
+            )
+        except Exception:
+            return self._synthetic_returns(symbol)
+
+    def _synthetic_returns(self, symbol: str, weeks: int = 52) -> pd.DataFrame:
+        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
+        dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=weeks, freq="W-FRI")
+        returns = rng.normal(0.0025, 0.025, size=weeks)
+        return pd.DataFrame({"week_end_date": dates.strftime("%Y-%m-%d"), "weekly_return": returns})
+
+    def refresh_symbol(self, symbol: str) -> None:
+        df = self._fetch_weekly_returns(symbol)
+        self.conn.executemany(
+            """
+            INSERT INTO weekly_returns(symbol, week_end_date, weekly_return)
+            VALUES (?, ?, ?)
+            ON CONFLICT(symbol, week_end_date) DO UPDATE SET weekly_return = excluded.weekly_return
+            """,
+            [(symbol, row.week_end_date, float(row.weekly_return)) for row in df.itertuples(index=False)],
+        )
+        self.conn.execute(
+            """
+            DELETE FROM weekly_returns
+            WHERE symbol = ?
+              AND week_end_date NOT IN (
+                SELECT week_end_date
+                FROM weekly_returns
+                WHERE symbol = ?
+                ORDER BY week_end_date DESC
+                LIMIT 52
+              )
+            """,
+            (symbol, symbol),
+        )
+        self.conn.commit()
+
+    def refresh_many(self, symbols: List[str]) -> None:
+        for s in symbols:
+            self.refresh_symbol(s)
+
+    def fetch_stats(self, symbols: List[str]) -> pd.DataFrame:
+        placeholders = ",".join(["?"] * len(symbols))
+        q = f"""
+        SELECT
+            symbol,
+            COUNT(*) AS weeks,
+            AVG(weekly_return) AS mean_weekly_return,
+            SQRT(MAX(AVG(weekly_return * weekly_return) - AVG(weekly_return) * AVG(weekly_return), 0)) AS std_weekly_return
+        FROM weekly_returns
+        WHERE symbol IN ({placeholders})
+        GROUP BY symbol
+        """
+        return pd.read_sql_query(q, self.conn, params=symbols)
+
+    def fetch_return_matrix(self, symbols: List[str]) -> pd.DataFrame:
+        placeholders = ",".join(["?"] * len(symbols))
+        q = f"""
+        SELECT symbol, week_end_date, weekly_return
+        FROM weekly_returns
+        WHERE symbol IN ({placeholders})
+        ORDER BY week_end_date
+        """
+        raw = pd.read_sql_query(q, self.conn, params=symbols)
+        pivot = raw.pivot(index="week_end_date", columns="symbol", values="weekly_return").dropna()
+        return pivot
+
+
+def _heuristic_label(asset_row: pd.Series, prefs: ClientPreferences) -> int:
+    score = 0
+    if asset_row["risk_score"] <= _risk_ceiling(prefs.risk_tolerance):
+        score += 2
+    if prefs.esg_preference and asset_row["esg_score"] >= 0.6:
+        score += 1
+    if (not prefs.allow_fno) and (not bool(asset_row["fno_available"])):
+        score += 1
+    if prefs.age >= 55 and asset_row["risk_score"] <= 5:
+        score += 1
+    if prefs.investment_horizon_years >= 5 and asset_row["asset_type"] in ["Equity", "ETF"]:
+        score += 1
+    if prefs.amount_to_invest < 100000 and asset_row["market_cap_bucket"] in ["Small Cap", "SME", "Crypto"]:
+        score -= 2
+    if asset_row["asset_type"].lower() in {x.lower() for x in prefs.excluded_categories}:
+        score -= 5
+    return 1 if score >= 2 else 0
+
+
+def stage_1_ml_filter(assets: pd.DataFrame, prefs: ClientPreferences, top_n: int = 40) -> pd.DataFrame:
+    assets = assets.copy()
+    assets["user_age"] = prefs.age
+    assets["user_horizon"] = prefs.investment_horizon_years
+    assets["user_amount"] = prefs.amount_to_invest
+    assets["user_esg_pref"] = int(prefs.esg_preference)
+    assets["user_allow_fno"] = int(prefs.allow_fno)
+
+    assets["target"] = assets.apply(lambda r: _heuristic_label(r, prefs), axis=1)
+
+    if LogisticRegression is None:
+        assets["ml_score"] = assets["target"].astype(float)
+        return assets[assets["ml_score"] > 0].sort_values("ml_score", ascending=False).head(top_n)
+
+    features = [
+        "asset_type",
+        "market_cap_bucket",
+        "risk_score",
+        "esg_score",
+        "fno_available",
+        "user_age",
+        "user_horizon",
+        "user_amount",
+        "user_esg_pref",
+        "user_allow_fno",
+    ]
+    X = assets[features]
+    y = assets["target"]
+
+    pre = ColumnTransformer(
+        transformers=[
+            ("cat", OneHotEncoder(handle_unknown="ignore"), ["asset_type", "market_cap_bucket", "fno_available"]),
+            ("num", StandardScaler(), ["risk_score", "esg_score", "user_age", "user_horizon", "user_amount", "user_esg_pref", "user_allow_fno"]),
+        ]
+    )
+    model = Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
+    model.fit(X, y)
+
+    assets["ml_score"] = model.predict_proba(X)[:, 1]
+
+    risk_cut = _risk_ceiling(prefs.risk_tolerance)
+    out = assets[assets["risk_score"] <= risk_cut]
+    if prefs.esg_preference:
+        out = out[out["esg_score"] >= 0.55]
+    if not prefs.allow_fno:
+        out = out[out["fno_available"] == False]  # noqa: E712
+    if prefs.excluded_categories:
+        excl = {x.lower() for x in prefs.excluded_categories}
+        out = out[~out["asset_type"].str.lower().isin(excl)]
+
+    out = out.sort_values("ml_score", ascending=False).head(top_n)
+    if out.empty:
+        raise ValueError("No assets passed the ML recommendation filter. Relax constraints.")
+    return out
+
+
+def correlation_pairs(corr: pd.DataFrame) -> List[Tuple[str, str, float]]:
+    pairs = []
+    for a, b in itertools.combinations(corr.columns.tolist(), 2):
+        pairs.append((a, b, float(corr.loc[a, b])))
+    return sorted(pairs, key=lambda x: x[2])
+
+
+def monte_carlo_optimize(returns_df: pd.DataFrame, pairs: List[Tuple[str, str, float]], amount: float, epochs: int = 1000):
+    rng = np.random.default_rng(42)
+    candidate_pairs = pairs[:4]
+    sims = []
+
+    for a, b, corr in candidate_pairs:
+        pair = returns_df[[a, b]]
+        mu = pair.mean().values * 52
+        cov = pair.cov().values * 52
+        for _ in range(epochs):
+            w1 = rng.random()
+            w = np.array([w1, 1 - w1])
+            ret = float(w @ mu)
+            vol = float(np.sqrt(max(w.T @ cov @ w, 0.0)))
+            score = ret / vol if vol > 1e-10 else -np.inf
+            sims.append(
+                {
+                    "asset_1": a,
+                    "asset_2": b,
+                    "corr": corr,
+                    "weight_1": w[0],
+                    "weight_2": w[1],
+                    "expected_return": ret,
+                    "expected_volatility": vol,
+                    "score": score,
+                    "amount_asset_1": amount * w[0],
+                    "amount_asset_2": amount * w[1],
+                }
+            )
+
+    sim_df = pd.DataFrame(sims)
+    best = sim_df.sort_values(["score", "expected_return"], ascending=[False, False]).iloc[0]
+    return best.to_dict(), sim_df
+
+
+def project_vs_nifty(best: Dict, returns_df: pd.DataFrame, nifty_series: pd.Series, horizon_years: int, n_sims: int = 1000):
+    w = np.array([best["weight_1"], best["weight_2"]])
+    pair = returns_df[[best["asset_1"], best["asset_2"]]]
+    mu, cov = pair.mean().values, pair.cov().values
+    n_mu, n_sigma = nifty_series.mean(), nifty_series.std(ddof=1)
+
+    weeks = max(1, horizon_years * 52)
+    rng = np.random.default_rng(7)
+    p_out, n_out = [], []
+    for _ in range(n_sims):
+        sim_pair = rng.multivariate_normal(mu, cov, size=weeks)
+        sim_port = np.prod(1 + sim_pair @ w) - 1
+        sim_nifty = np.prod(1 + rng.normal(n_mu, n_sigma, size=weeks)) - 1
+        p_out.append(sim_port)
+        n_out.append(sim_nifty)
+
+    p_arr, n_arr = np.array(p_out), np.array(n_out)
+    out = pd.DataFrame(
+        {
+            "scenario": ["bear", "base", "bull"],
+            "portfolio_return": [np.percentile(p_arr, 20), np.percentile(p_arr, 50), np.percentile(p_arr, 80)],
+            "nifty_return": [np.percentile(n_arr, 20), np.percentile(n_arr, 50), np.percentile(n_arr, 80)],
+        }
+    )
+    out["excess_vs_nifty"] = out["portfolio_return"] - out["nifty_return"]
+    return out
+
+
+def parse_inputs() -> ClientPreferences:
+    age = int(input("Age: ").strip())
+    risk_tolerance = input("Risk tolerance (low/medium/high): ").strip().lower()
+    amount = float(input("Amount to invest (INR): ").strip())
+    esg_pref = input("ESG preference? (yes/no): ").strip().lower() in {"yes", "y"}
+    excluded = input("Excluded categories (comma-separated) or blank: ").strip()
+    excluded_categories = [x.strip() for x in excluded.split(",") if x.strip()] if excluded else []
+    allow_fno = input("Willing to invest in F&O? (yes/no): ").strip().lower() in {"yes", "y"}
+    horizon = int(input("Investment horizon (years): ").strip())
+    return ClientPreferences(age, risk_tolerance, amount, esg_pref, excluded_categories, allow_fno, horizon)
+
+
+def main() -> None:
+    print("=== Stage-wise Indian Roboadvisor (ML + SQL) ===")
+    prefs = parse_inputs()
+
+    assets = create_asset_master_300_plus()
+    filtered = stage_1_ml_filter(assets, prefs, top_n=40)
+
+    print("\n[Stage 1] Suitable asset universe (top ML recommendations):")
+    print(filtered[["symbol", "asset_type", "market_cap_bucket", "risk_score", "esg_score", "ml_score"]].head(40))
+
+    store = WeeklyReturnSQLStore(DB_FILE)
+    symbols = filtered["symbol"].tolist()
+    store.refresh_many(symbols + ["^NSEI"])
+
+    stats = store.fetch_stats(symbols)
+    returns_df = store.fetch_return_matrix(symbols)
+    corr = returns_df.corr()
+    pairs = correlation_pairs(corr)
+
+    print("\n[Stage 2] Weekly mean/std from SQL (rolling 52 weeks):")
+    print(stats.sort_values("mean_weekly_return", ascending=False).head(15))
+    print("\nCorrelation matrix:")
+    print(corr)
+
+    top4 = pairs[:4]
+    print("\nLeast correlated pair + next best 3:")
+    for i, (a, b, c) in enumerate(top4, start=1):
+        tag = "Lowest" if i == 1 else f"Next {i-1}"
+        print(f"{tag}: {a} vs {b}, corr={c:.4f}")
+
+    best, sims = monte_carlo_optimize(returns_df, pairs, prefs.amount_to_invest, epochs=1000)
+    print("\n[Stage 3] Monte Carlo recommendation:")
+    print(f"Assets: {best['asset_1']} ({best['weight_1']:.2%}) + {best['asset_2']} ({best['weight_2']:.2%})")
+    print(f"Amount split: INR {best['amount_asset_1']:.2f} and INR {best['amount_asset_2']:.2f}")
+    print(f"Expected return: {best['expected_return']:.2%} | Expected volatility: {best['expected_volatility']:.2%}")
+
+    print("\nOther top options:")
+    print(
+        sims.sort_values(["score", "expected_return"], ascending=[False, False])
+        [["asset_1", "asset_2", "weight_1", "weight_2", "expected_return", "expected_volatility", "score"]]
+        .head(5)
+    )
+
+    nifty = store.fetch_return_matrix(["^NSEI"]).squeeze("columns")
+    projection = project_vs_nifty(best, returns_df, nifty, prefs.investment_horizon_years)
+    print("\n[Stage 4] Bear/Base/Bull projection vs NIFTY:")
+    print(projection)
+
+    store.close()
+
+
+if __name__ == "__main__":
+    main()
 
EOF
)
