EPS = 1e-9

RISER_MAX_DIST_TO_HIGH   = 0.10
RISER_MAX_VARIABILITY    = 0.05
DOWN_MAX_VARIABILITY     = 0.05
TOP_MAX_DIST_TO_HIGH     = 0.03
TOP_MAX_VARIABILITY      = 0.05
MIN_PERIODS              = 20
REGRESSION_WINDOW        = 20
ERA_SIZE                 = 5

MIN_RISER_SCORE          = 0.45
MA_WEEKS                 = 30
MA_SLOPE_LOOKBACK        = 4
RS_WEEKS                 = 26
RS_BENCHMARK             = "SPY"
MIN_RS                   = 0.0
VOL_BASE_WEEKS           = 10
VOL_BREAKOUT_MULT        = 1.5
REQUIRE_VOLUME           = False
SOFTMAX_TEMP             = 0.1

BACKTEST_YEARS = 5
BACKTEST_FORWARD_DAYS = 20
BACKTEST_MIN_SIGNALS = 10


import math
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


def _clean_columns(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        # If this is already a single-ticker slice, reduce it.
        if df.columns.nlevels == 2 and len(set(df.columns.get_level_values(0))) == 1:
            df = df.copy()
            df.columns = df.columns.get_level_values(1)
    return df


def _extract_ticker(batch, ticker):
    """Extract one ticker robustly from either yfinance MultiIndex layout."""
    if batch is None or batch.empty:
        return None

    t = str(ticker).upper().strip()

    try:
        if not isinstance(batch.columns, pd.MultiIndex):
            return _clean_columns(batch.copy())

        for level in range(batch.columns.nlevels):
            values = batch.columns.get_level_values(level)
            matches = [i for i, v in enumerate(values)
                       if str(v).upper().strip() == t]
            if not matches:
                continue

            out = batch.iloc[:, matches].copy()

            # Drop the ticker level and leave OHLCV field names.
            other_levels = [i for i in range(batch.columns.nlevels) if i != level]
            if len(other_levels) == 1:
                out.columns = [
                    str(batch.columns[i][other_levels[0]]).strip().title()
                    for i in matches
                ]
            return _clean_columns(out)

    except Exception:
        return None

    return None

@st.cache_data(ttl=900, max_entries=30, show_spinner=False)
def download_batch(tickers, period, interval):
    syms = tuple(dict.fromkeys(str(x).upper().strip() for x in tickers if str(x).strip()))
    if not syms:
        return pd.DataFrame()
    return yf.download(
        list(syms), period=period, interval=interval,
        progress=False, auto_adjust=True, group_by="column", threads=False,
    )


@st.cache_data(ttl=900, max_entries=30, show_spinner=False)
def download_reliability_batch(tickers):
    syms = tuple(dict.fromkeys(str(x).upper().strip() for x in tickers if str(x).strip()))
    if not syms:
        return pd.DataFrame()
    return yf.download(
        list(syms), period=f"{BACKTEST_YEARS}y", interval="1d",
        progress=False, auto_adjust=True, group_by="column", threads=False,
    )


def _weekly_from_batch(df):
    # This is only a fallback/helper; main Stage 2 uses Yahoo's weekly endpoint
    # to preserve the validated Stage 2 results.
    return df


class MarketData:
    def __init__(self, tickers):
        self.tickers = tuple(dict.fromkeys([t.upper() for t in tickers] + [RS_BENCHMARK]))
        self.daily6 = download_batch(self.tickers, "6mo", "1d")
        self.weekly2 = download_batch(self.tickers, "2y", "1wk")
        self.daily5 = download_batch(self.tickers, "5d", "1d")
        self.reliability = None

    def daily(self, ticker): return _extract_ticker(self.daily6, ticker)
    def weekly(self, ticker):
        df = _extract_ticker(self.weekly2, ticker)
        if df is None or df.empty:
            return None

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Close"]).copy()

        # Ignore the current, potentially incomplete week.
        if len(df) > 1:
            df = df.iloc[:-1].copy()

        return df
    def price_daily(self, ticker): return _extract_ticker(self.daily5, ticker)

    def load_reliability(self, tickers):
        syms = tuple(dict.fromkeys([t.upper() for t in tickers] + [RS_BENCHMARK]))
        self.reliability = download_reliability_batch(syms)

    def reliability_daily(self, ticker):
        if self.reliability is None: return None
        return _extract_ticker(self.reliability, ticker)


def fetch_features_from_df(ticker, df):
    try:
        if df is None or df.empty: return None
        # Yahoo Finance batch downloads can occasionally contain trailing
        # NaN rows for one or more tickers.  Remove incomplete OHLC rows
        # before calculating the score; otherwise NaN values can make
        # _clamp() produce a misleading score of 1.0 and a blank price.
        df = df.copy()
        for col in ("Close", "High", "Low"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close", "High", "Low"])
        if len(df) < MIN_PERIODS: return None
        closes = df["Close"].values.flatten().astype(float)
        highs = df["High"].values.flatten().astype(float)
        lows = df["Low"].values.flatten().astype(float)
        n = len(closes)
        current_price = closes[-1]
        period_high = highs.max()
        dist_to_high = (period_high-current_price)/max(EPS,period_high)
        recent = closes[-REGRESSION_WINDOW:]
        slope = _linear_regression_slope(recent)
        start = n % ERA_SIZE
        eras=[closes[i:i+ERA_SIZE] for i in range(start,n-ERA_SIZE+1,ERA_SIZE)]
        era_returns=[(e[-1]-e[0])/max(EPS,e[0]) for e in eras if len(e)==ERA_SIZE]
        up_eras=sum(r>0 for r in era_returns); down_eras=sum(r<0 for r in era_returns); total=len(era_returns)
        story_score_up=(up_eras/max(1,total))*6.0
        story_score_down=(down_eras/max(1,total))*6.0
        story_variability=float(np.std(era_returns)) if era_returns else 0.0
        recent_returns=np.diff(recent)/recent[:-1]
        up_ratio=float(np.mean(recent_returns>0)); down_ratio=float(np.mean(recent_returns<0))
        rolling_max=np.maximum.accumulate(closes)
        drawdowns=(rolling_max-closes)/np.maximum(EPS,rolling_max)
        max_drawdown=float(drawdowns.max())
        recent_low=lows[-REGRESSION_WINDOW:].min()
        recovery=(current_price-recent_low)/max(EPS,recent_low)*6.0
        segments_up=streak=0
        for r in recent_returns:
            if r>0: streak+=1; segments_up=max(segments_up,streak)
            else: streak=0
        return {"ticker":ticker,"current_price":round(current_price,4),"dist_to_high":round(dist_to_high,4),
          "eras_recent_regression_slope":round(slope,6),"swing_recovery":round(min(recovery,6.0),4),
          "swing_drawdown_n":round(_clamp(1.0-max_drawdown),4),"swing_segments_up_count":float(segments_up),
          "riser_story_score":round(story_score_up,4),"riser_up_ratio":round(up_ratio,4),"riser_story_variability":round(story_variability,6),
          "down_story_score":round(story_score_down,4),"down_recent_down_ratio":round(down_ratio,4),"down_story_variability":round(story_variability,6),
          "top_story_score":round(story_score_up,4),"top_up_ratio":round(up_ratio,4),"top_story_variability":round(story_variability,6)}
    except Exception:
        return None


def fetch_ma30w_from_data(ticker, weekly, daily5, benchmark_weekly):
    try:
        if weekly is None or len(weekly)<MA_WEEKS+MA_SLOPE_LOOKBACK: return None
        close=weekly["Close"].astype(float); ma=close.rolling(MA_WEEKS).mean().values
        ma_now=ma[-1]; ma_prev=ma[-1-MA_SLOPE_LOOKBACK]
        if daily5 is None or daily5.empty: return None
        daily5 = daily5.copy()
        daily5["Close"] = pd.to_numeric(daily5["Close"], errors="coerce")
        daily5 = daily5.dropna(subset=["Close"])
        if daily5.empty: return None
        price=float(daily5["Close"].iloc[-1])
        rs=None
        if benchmark_weekly is not None and len(close)>RS_WEEKS:
            # Accept either a Close Series or a weekly DataFrame.
            if isinstance(benchmark_weekly, pd.DataFrame):
                benchmark_weekly = (
                    benchmark_weekly["Close"]
                    if "Close" in benchmark_weekly.columns
                    else None
                )
            if benchmark_weekly is not None:
                b = pd.to_numeric(
                    benchmark_weekly, errors="coerce"
                ).reindex(close.index, method="ffill")
                if not b.isna().iloc[-RS_WEEKS-1:].any():
                    rs = float(
                        (close.iloc[-1] / close.iloc[-RS_WEEKS-1] - 1)
                        - (b.iloc[-1] / b.iloc[-RS_WEEKS-1] - 1)
                    )
        vol_ratio=None
        if "Volume" in weekly.columns and len(weekly)>VOL_BASE_WEEKS+1:
            vol=weekly["Volume"].astype(float); base=vol.iloc[-VOL_BASE_WEEKS-1:-1].mean()
            vol_ratio=float(vol.iloc[-1]/base) if base>0 else None
        return {"ma30w":round(float(ma_now),4),"price_vs_ma30w":round(float(price/ma_now-1),4),
          "ma30w_slope":round(float(ma_now/ma_prev-1),4),"above_ma30w":bool(price>ma_now),
          "ma30w_rising":bool(ma_now>ma_prev),"rs":round(rs,4) if rs is not None else None,
          "vol_ratio":round(vol_ratio,2) if vol_ratio is not None else None}
    except Exception:
        return None

# --- HELPERS ---

def _clamp(x, lo=0.0, hi=1.0):

    return max(lo, min(hi, x))



def _softmax(scores: dict, temperature=SOFTMAX_TEMP) -> dict:

    exp_vals = {k: math.exp(v / temperature) for k, v in scores.items()}

    total = sum(exp_vals.values()) or 1.0

    return {k: v / total for k, v in exp_vals.items()}



def _normalize(scores: dict) -> dict:

    max_val = max(scores.values()) if scores else 1.0

    if max_val < EPS:

        return {k: 0.0 for k in scores}

    return {k: v / max_val for k, v in scores.items()}



def _linear_regression_slope(series: np.ndarray) -> float:

    n = len(series)

    if n < 2:

        return 0.0

    x = np.arange(n, dtype=float)

    x -= x.mean()

    y = series - series.mean()

    denom = (x * x).sum()

    if abs(denom) < EPS:

        return 0.0

    slope = (x * y).sum() / denom

    return float(slope / max(EPS, series.mean()))



def _weekly(ticker: str) -> Optional[pd.DataFrame]:

    # Weekly data used for MA30w / RS / volume.

    # IMPORTANT: exclude the current, still-open week.

    df = yf.download(ticker, period="2y", interval="1wk",

                     progress=False, auto_adjust=True)

    if df is None or df.empty:

        return None

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)



    # Remove the current incomplete weekly candle.

    # Yahoo's last weekly row can represent the current week while the

    # market is open; Stage 2 must use only completed weeks for its

    # trend calculations.

    if len(df) > 1:

        df = df.iloc[:-1].copy()



    return df






# --- REGIME DETECTION ---

def detect_regime(f: dict) -> dict:

    slope = f["eras_recent_regression_slope"]

    dth   = f["dist_to_high"]

    up_r  = f["riser_up_ratio"]

    dn_r  = f["down_recent_down_ratio"]

    return {

        "swing_valid": 0.3 < up_r < 0.7,

        "riser_valid": slope > 0.002 and up_r > 0.55,

        "down_valid":  slope < -0.002 and dn_r > 0.55,

        "top_valid":   dth < TOP_MAX_DIST_TO_HIGH and up_r > 0.5,

    }






# --- FULL SCORING ---

def score_from_features(f: dict) -> dict:

    regime = detect_regime(f)



    swing_score_raw = 0.0

    if regime["swing_valid"]:

        # weights now sum to 1.0 (was 0.65)

        swing_score_raw = (

            0.55 * _clamp(f["swing_recovery"] / 6.0) +

            0.30 * _clamp(f["swing_drawdown_n"]) +

            0.15 * _clamp(f["swing_segments_up_count"] / 3.0)

        )



    riser_score_raw = 0.0

    if regime["riser_valid"]:

        riser_headroom = _clamp((RISER_MAX_DIST_TO_HIGH - f["dist_to_high"]) /

                                 max(EPS, RISER_MAX_DIST_TO_HIGH))

        riser_clean    = 1.0 - _clamp(f["riser_story_variability"] /

                                       max(EPS, RISER_MAX_VARIABILITY))

        riser_score_raw = (

            0.40 * _clamp(f["riser_story_score"] / 6.0) +

            0.20 * _clamp(f["riser_up_ratio"]) +

            0.20 * _clamp(max(0.0, f["eras_recent_regression_slope"]) / 0.01) +

            0.10 * riser_headroom +

            0.10 * riser_clean

        )



    down_score_raw = 0.0

    if regime["down_valid"]:

        down_tail = 1.0 - _clamp(f["down_story_variability"] /

                                   max(EPS, DOWN_MAX_VARIABILITY))

        down_score_raw = (

            0.45 * _clamp(f["down_story_score"] / 6.0) +

            0.20 * _clamp(f["down_recent_down_ratio"]) +

            0.20 * _clamp(abs(min(0.0, f["eras_recent_regression_slope"])) / 0.01) +

            0.15 * down_tail

        )



    top_score_raw = 0.0

    if regime["top_valid"]:

        top_proximity = _clamp(1.0 - f["dist_to_high"] /

                                max(EPS, TOP_MAX_DIST_TO_HIGH))

        top_flatness  = 1.0 - _clamp(f["top_story_variability"] /

                                      max(EPS, TOP_MAX_VARIABILITY))

        top_score_raw = (

            0.40 * _clamp(f["top_story_score"] / 6.0) +

            0.30 * top_proximity +

            0.20 * _clamp(f["top_up_ratio"]) +

            0.10 * top_flatness

        )



    raw = {"swing": swing_score_raw, "riser": riser_score_raw,

           "down": down_score_raw,   "top":   top_score_raw}



    label = max(raw, key=raw.get)

    score = raw[label]

    if score < EPS:

        label = "none"



    return {

        "ticker":     f["ticker"],

        "price":      f["current_price"],

        "label":      label,

        "score":      round(score, 4),

        "raw":        {k: round(v, 4) for k, v in raw.items()},

        "normalized": {k: round(v, 4) for k, v in _normalize(raw).items()},

        "softmax":    {k: round(v, 4) for k, v in _softmax(raw).items()},

        "regime":     regime,

    }






# --- STAGE 2 FILTER ---

MIN_RISER_SCORE          = 0.45

MA_WEEKS                 = 30

MA_SLOPE_LOOKBACK        = 4

RS_WEEKS                 = 26     # relative-strength window (weeks)

RS_BENCHMARK             = "SPY"

MIN_RS                   = 0.0    # stock return - benchmark return must exceed this

VOL_BASE_WEEKS           = 10     # baseline weeks for volume average

VOL_BREAKOUT_MULT        = 1.5    # last-week volume vs baseline

REQUIRE_VOLUME           = False  # True = volume surge is mandatory for Stage 2

SOFTMAX_TEMP             = 0.1    # low T -> sharper probabilities (scores are in 0-1)





# --- HELPERS ---

def _clamp(x, lo=0.0, hi=1.0):

    return max(lo, min(hi, x))



def _softmax(scores: dict, temperature=SOFTMAX_TEMP) -> dict:

    exp_vals = {k: math.exp(v / temperature) for k, v in scores.items()}

    total = sum(exp_vals.values()) or 1.0

    return {k: v / total for k, v in exp_vals.items()}



def _normalize(scores: dict) -> dict:

    max_val = max(scores.values()) if scores else 1.0

    if max_val < EPS:

        return {k: 0.0 for k in scores}

    return {k: v / max_val for k, v in scores.items()}



def _linear_regression_slope(series: np.ndarray) -> float:

    n = len(series)

    if n < 2:

        return 0.0

    x = np.arange(n, dtype=float)

    x -= x.mean()

    y = series - series.mean()

    denom = (x * x).sum()

    if abs(denom) < EPS:

        return 0.0

    slope = (x * y).sum() / denom

    return float(slope / max(EPS, series.mean()))



def _weekly(ticker: str) -> Optional[pd.DataFrame]:

    # Weekly data used for MA30w / RS / volume.

    # IMPORTANT: exclude the current, still-open week.

    df = yf.download(ticker, period="2y", interval="1wk",

                     progress=False, auto_adjust=True)

    if df is None or df.empty:

        return None

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)



    # Remove the current incomplete weekly candle.

    # Yahoo's last weekly row can represent the current week while the

    # market is open; Stage 2 must use only completed weeks for its

    # trend calculations.

    if len(df) > 1:

        df = df.iloc[:-1].copy()



    return df





# --- BENCHMARK (downloaded once) ---

_BENCH_CACHE: dict = {}



def get_benchmark() -> Optional[pd.Series]:

    if RS_BENCHMARK not in _BENCH_CACHE:

        df = _weekly(RS_BENCHMARK)

        _BENCH_CACHE[RS_BENCHMARK] = df["Close"].astype(float) if df is not None else None

    return _BENCH_CACHE[RS_BENCHMARK]





# --- 30-WEEK MA + RS + VOLUME ---

def fetch_ma30w(ticker: str) -> Optional[dict]:

    try:

        df = _weekly(ticker)

        if df is None or len(df) < MA_WEEKS + MA_SLOPE_LOOKBACK:

            print(f"[{ticker}] Not enough weekly data for {MA_WEEKS}w MA")

            return None



        close = df["Close"].astype(float)

        ma    = close.rolling(MA_WEEKS).mean().values



        # MA30w and its slope come ONLY from completed weeks.

        ma_now = ma[-1]

        ma_prev = ma[-1 - MA_SLOPE_LOOKBACK]



        # Compare the real/latest daily price against the stable MA30w.

        daily = yf.download(ticker, period="5d", interval="1d",

                            progress=False, auto_adjust=True)

        if daily is None or daily.empty:

            return None

        if isinstance(daily.columns, pd.MultiIndex):

            daily.columns = daily.columns.get_level_values(0)

        price = float(daily["Close"].iloc[-1])



        # Relative strength vs benchmark (aligned by date)

        rs = None

        bench = get_benchmark()

        if bench is not None and len(close) > RS_WEEKS:

            b = bench.reindex(close.index, method="ffill")

            if not b.isna().iloc[-RS_WEEKS - 1:].any():

                s_ret = close.iloc[-1] / close.iloc[-RS_WEEKS - 1] - 1.0

                b_ret = b.iloc[-1]     / b.iloc[-RS_WEEKS - 1]     - 1.0

                rs = float(s_ret - b_ret)



        # Volume surge: last week vs average of previous N weeks

        vol_ratio = None

        if "Volume" in df.columns and len(df) > VOL_BASE_WEEKS + 1:

            vol = df["Volume"].astype(float)

            base = vol.iloc[-VOL_BASE_WEEKS - 1:-1].mean()

            vol_ratio = float(vol.iloc[-1] / base) if base > 0 else None



        return {

            "ma30w":          round(float(ma_now), 4),

            "price_vs_ma30w": round(float(price / ma_now - 1.0), 4),

            "ma30w_slope":    round(float(ma_now / ma_prev - 1.0), 4),

            "above_ma30w":    bool(price > ma_now),

            "ma30w_rising":   bool(ma_now > ma_prev),

            "rs":             round(rs, 4) if rs is not None else None,

            "vol_ratio":      round(vol_ratio, 2) if vol_ratio is not None else None,

        }

    except Exception as e:

        print(f"[{ticker}] MA30w error: {e}")

        return None





# --- FEATURE EXTRACTION ---

def fetch_features(ticker: str, period: str = "6mo", interval: str = "1d") -> Optional[dict]:

    try:

        df = yf.download(ticker, period=period, interval=interval,

                         progress=False, auto_adjust=True)

        if df is None or len(df) < MIN_PERIODS:

            print(f"[{ticker}] Insufficient data ({len(df) if df is not None else 0} candles)")

            return None



        closes = df["Close"].values.flatten().astype(float)

        highs  = df["High"].values.flatten().astype(float)

        lows   = df["Low"].values.flatten().astype(float)

        n      = len(closes)



        current_price = closes[-1]

        period_high   = highs.max()

        dist_to_high  = (period_high - current_price) / max(EPS, period_high)



        recent = closes[-REGRESSION_WINDOW:]

        slope  = _linear_regression_slope(recent)



        # Eras aligned to the END (most recent candle always included);

        # the oldest partial block is the one dropped.

        start = n % ERA_SIZE

        eras = [closes[i:i + ERA_SIZE] for i in range(start, n - ERA_SIZE + 1, ERA_SIZE)]

        era_returns = [(e[-1] - e[0]) / max(EPS, e[0]) for e in eras if len(e) == ERA_SIZE]



        up_eras    = sum(1 for r in era_returns if r > 0)

        down_eras  = sum(1 for r in era_returns if r < 0)

        total_eras = len(era_returns)



        story_score_up    = (up_eras / max(1, total_eras)) * 6.0

        story_score_down  = (down_eras / max(1, total_eras)) * 6.0

        story_variability = float(np.std(era_returns)) if era_returns else 0.0



        recent_returns = np.diff(recent) / recent[:-1]

        up_ratio   = float(np.mean(recent_returns > 0))

        down_ratio = float(np.mean(recent_returns < 0))



        rolling_max  = np.maximum.accumulate(closes)

        drawdowns    = (rolling_max - closes) / np.maximum(EPS, rolling_max)

        max_drawdown = float(drawdowns.max())



        recent_low = lows[-REGRESSION_WINDOW:].min()

        recovery   = (current_price - recent_low) / max(EPS, recent_low) * 6.0



        segments_up, streak = 0, 0

        for r in recent_returns:

            if r > 0:

                streak += 1

                segments_up = max(segments_up, streak)

            else:

                streak = 0



        return {

            "ticker":                       ticker,

            "current_price":                round(current_price, 4),

            "dist_to_high":                 round(dist_to_high, 4),

            "eras_recent_regression_slope": round(slope, 6),

            "swing_recovery":               round(min(recovery, 6.0), 4),

            "swing_drawdown_n":             round(_clamp(1.0 - max_drawdown), 4),

            "swing_segments_up_count":      float(segments_up),

            "riser_story_score":            round(story_score_up, 4),

            "riser_up_ratio":               round(up_ratio, 4),

            "riser_story_variability":      round(story_variability, 6),

            "down_story_score":             round(story_score_down, 4),

            "down_recent_down_ratio":       round(down_ratio, 4),

            "down_story_variability":       round(story_variability, 6),

            "top_story_score":              round(story_score_up, 4),

            "top_up_ratio":                 round(up_ratio, 4),

            "top_story_variability":        round(story_variability, 6),

        }



    except Exception as e:

        print(f"[{ticker}] Error: {e}")

        return None





# --- REGIME DETECTION ---

def detect_regime(f: dict) -> dict:

    slope = f["eras_recent_regression_slope"]

    dth   = f["dist_to_high"]

    up_r  = f["riser_up_ratio"]

    dn_r  = f["down_recent_down_ratio"]

    return {

        "swing_valid": 0.3 < up_r < 0.7,

        "riser_valid": slope > 0.002 and up_r > 0.55,

        "down_valid":  slope < -0.002 and dn_r > 0.55,

        "top_valid":   dth < TOP_MAX_DIST_TO_HIGH and up_r > 0.5,

    }





# --- FULL SCORING ---

def score_from_features(f: dict) -> dict:

    regime = detect_regime(f)



    swing_score_raw = 0.0

    if regime["swing_valid"]:

        # weights now sum to 1.0 (was 0.65)

        swing_score_raw = (

            0.55 * _clamp(f["swing_recovery"] / 6.0) +

            0.30 * _clamp(f["swing_drawdown_n"]) +

            0.15 * _clamp(f["swing_segments_up_count"] / 3.0)

        )



    riser_score_raw = 0.0

    if regime["riser_valid"]:

        riser_headroom = _clamp((RISER_MAX_DIST_TO_HIGH - f["dist_to_high"]) /

                                 max(EPS, RISER_MAX_DIST_TO_HIGH))

        riser_clean    = 1.0 - _clamp(f["riser_story_variability"] /

                                       max(EPS, RISER_MAX_VARIABILITY))

        riser_score_raw = (

            0.40 * _clamp(f["riser_story_score"] / 6.0) +

            0.20 * _clamp(f["riser_up_ratio"]) +

            0.20 * _clamp(max(0.0, f["eras_recent_regression_slope"]) / 0.01) +

            0.10 * riser_headroom +

            0.10 * riser_clean

        )



    down_score_raw = 0.0

    if regime["down_valid"]:

        down_tail = 1.0 - _clamp(f["down_story_variability"] /

                                   max(EPS, DOWN_MAX_VARIABILITY))

        down_score_raw = (

            0.45 * _clamp(f["down_story_score"] / 6.0) +

            0.20 * _clamp(f["down_recent_down_ratio"]) +

            0.20 * _clamp(abs(min(0.0, f["eras_recent_regression_slope"])) / 0.01) +

            0.15 * down_tail

        )



    top_score_raw = 0.0

    if regime["top_valid"]:

        top_proximity = _clamp(1.0 - f["dist_to_high"] /

                                max(EPS, TOP_MAX_DIST_TO_HIGH))

        top_flatness  = 1.0 - _clamp(f["top_story_variability"] /

                                      max(EPS, TOP_MAX_VARIABILITY))

        top_score_raw = (

            0.40 * _clamp(f["top_story_score"] / 6.0) +

            0.30 * top_proximity +

            0.20 * _clamp(f["top_up_ratio"]) +

            0.10 * top_flatness

        )



    raw = {"swing": swing_score_raw, "riser": riser_score_raw,

           "down": down_score_raw,   "top":   top_score_raw}



    label = max(raw, key=raw.get)

    score = raw[label]

    if score < EPS:

        label = "none"



    return {

        "ticker":     f["ticker"],

        "price":      f["current_price"],

        "label":      label,

        "score":      round(score, 4),

        "raw":        {k: round(v, 4) for k, v in raw.items()},

        "normalized": {k: round(v, 4) for k, v in _normalize(raw).items()},

        "softmax":    {k: round(v, 4) for k, v in _softmax(raw).items()},

        "regime":     regime,

    }





# --- STAGE 2 FILTER ---

def apply_stage2(result: dict, ma: Optional[dict]) -> dict:

    if ma is None:

        result.update({"ma30w": None, "price_vs_ma30w": None, "ma30w_slope": None,

                       "above_ma30w": False, "ma30w_rising": False,

                       "rs": None, "vol_ratio": None,

                       "rs_ok": False, "vol_ok": False, "stage2": False})

        return result



    result.update(ma)

    result["rs_ok"]  = ma["rs"] is not None and ma["rs"] > MIN_RS

    result["vol_ok"] = ma["vol_ratio"] is not None and ma["vol_ratio"] >= VOL_BREAKOUT_MULT

    result["stage2"] = bool(

        result["label"] == "riser"

        and result["score"] > MIN_RISER_SCORE

        and ma["above_ma30w"]

        and ma["ma30w_rising"]

        and result["rs_ok"]

        and (result["vol_ok"] or not REQUIRE_VOLUME)

    )

    return result







def _reliability_weekly(df):
    if df is None or df.empty: return pd.DataFrame()
    return df.resample("W-FRI").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna(subset=["Close"])


def calc_at_date(daily, w, bw, date):
    d=daily.loc[:date].tail(126); ww=w.loc[:date]
    if len(d)<MIN_PERIODS or len(ww)<MA_WEEKS+MA_SLOPE_LOOKBACK: return None
    c=d["Close"].astype(float).values; hi=d["High"].astype(float).values; lo=d["Low"].astype(float).values
    price=float(c[-1]); dist=(float(hi.max())-price)/max(EPS,float(hi.max())); recent=c[-REGRESSION_WINDOW:]; sl=_linear_regression_slope(recent)
    start=len(c)%ERA_SIZE; eras=[c[i:i+ERA_SIZE] for i in range(start,len(c)-ERA_SIZE+1,ERA_SIZE)]
    ers=[(e[-1]-e[0])/max(EPS,e[0]) for e in eras if len(e)==ERA_SIZE]
    up=sum(x>0 for x in ers); down=sum(x<0 for x in ers); total=len(ers); story_up=(up/max(1,total))*6; story_down=(down/max(1,total))*6; var=float(np.std(ers)) if ers else 0.0
    rr=np.diff(recent)/recent[:-1]; up_ratio=float(np.mean(rr>0)); down_ratio=float(np.mean(rr<0))
    dd=(np.maximum.accumulate(c)-c)/np.maximum(EPS,np.maximum.accumulate(c)); maxdd=float(dd.max()); recent_low=lo[-REGRESSION_WINDOW:].min(); recovery=(price-recent_low)/max(EPS,recent_low)*6
    streak=best=0
    for x in rr:
        if x>0: streak+=1; best=max(best,streak)
        else: streak=0
    swing=0.0
    if 0.3<up_ratio<0.7: swing=0.55*_clamp(min(recovery,6)/6)+0.30*_clamp(1-maxdd)+0.15*_clamp(best/3)
    riser=0.0
    if sl>0.002 and up_ratio>0.55:
        head=_clamp((RISER_MAX_DIST_TO_HIGH-dist)/RISER_MAX_DIST_TO_HIGH); clean=1-_clamp(var/RISER_MAX_VARIABILITY)
        riser=0.40*_clamp(story_up/6)+0.20*_clamp(up_ratio)+0.20*_clamp(max(0,sl)/0.01)+0.10*head+0.10*clean
    downscore=0.0
    if sl<-0.002 and down_ratio>0.55:
        tail=1-_clamp(var/DOWN_MAX_VARIABILITY); downscore=0.45*_clamp(story_down/6)+0.20*_clamp(down_ratio)+0.20*_clamp(abs(min(0,sl))/0.01)+0.15*tail
    topscore=0.0
    if dist<TOP_MAX_DIST_TO_HIGH and up_ratio>0.5:
        prox=_clamp(1-dist/TOP_MAX_DIST_TO_HIGH); flat=1-_clamp(var/TOP_MAX_VARIABILITY); topscore=0.40*_clamp(story_up/6)+0.30*prox+0.20*_clamp(up_ratio)+0.10*flat
    raw={"swing":swing,"riser":riser,"down":downscore,"top":topscore}; label=max(raw,key=raw.get); score=raw[label]
    if score<EPS: label="none"
    cw=ww["Close"].astype(float); ma=cw.rolling(MA_WEEKS).mean(); ma_now=float(ma.iloc[-1]); ma_prev=float(ma.iloc[-1-MA_SLOPE_LOOKBACK])
    b=bw.reindex(cw.index,method="ffill"); rs=None
    if len(cw)>RS_WEEKS:
        idx=-RS_WEEKS-1
        if not b.iloc[idx:].isna().any(): rs=float((cw.iloc[-1]/cw.iloc[idx]-1)-(b.iloc[-1]/b.iloc[idx]-1))
    volx=None
    if len(ww)>VOL_BASE_WEEKS+1:
        vol=ww["Volume"].astype(float); base=vol.iloc[-VOL_BASE_WEEKS-1:-1].mean()
        if base>0: volx=float(vol.iloc[-1]/base)
    stage2=bool(label=="riser" and score>MIN_RISER_SCORE and price>ma_now and ma_now>ma_prev and rs is not None and rs>0 and ((volx is not None and volx>=VOL_BREAKOUT_MULT) or not REQUIRE_VOLUME))
    return {"price":price,"score":score,"label":label,"stage2":stage2}


def historical_stage2_reliability_from_data(ticker, daily, benchmark_daily):
    if daily is None or benchmark_daily is None or len(daily)<300: return None
    w=_reliability_weekly(daily); bw=_reliability_weekly(benchmark_daily)["Close"]
    out=[]; prev=False; start=max(160,MA_WEEKS*5)
    for i in range(start,len(daily)):
        date=daily.index[i]; f=calc_at_date(daily,w,bw,date)
        if f is None: continue
        new_signal=f["stage2"] and not prev
        if new_signal:
            future=daily["Close"].iloc[i+1:]
            ret=float(future.iloc[BACKTEST_FORWARD_DAYS-1]/f["price"]-1) if len(future)>=BACKTEST_FORWARD_DAYS else np.nan
            out.append(ret)
        prev=f["stage2"]
    vals=pd.Series(out).dropna().astype(float)
    if vals.empty: return {"reliability":None,"reliability_label":"⚪ Sense dades","signals":0,"mean20":None,"median20":None}
    rel=float((vals>0).mean())
    if len(vals)<BACKTEST_MIN_SIGNALS: label="⚪ Mostra insuficient"
    elif rel>=0.70: label="🟢 Alta"
    elif rel>=0.60: label="🟢 Bona"
    elif rel>=0.50: label="🟡 Moderada"
    else: label="🔴 Baixa"
    return {"reliability":rel,"reliability_label":label,"signals":int(len(vals)),"mean20":float(vals.mean()),"median20":float(vals.median())}


def analyze_tickers(tickers, data):
    results=[]
    bench_df = data.weekly(RS_BENCHMARK)
    bench = (
        bench_df["Close"].astype(float)
        if bench_df is not None and "Close" in bench_df.columns
        else None
    )
    for ticker in tickers:
        f=fetch_features_from_df(ticker, data.daily(ticker))
        if not f: continue
        r=score_from_features(f)
        ma=fetch_ma30w_from_data(ticker,data.weekly(ticker),data.price_daily(ticker),bench)
        r=apply_stage2(r,ma)
        r["stage2_reliability"]=None
        results.append(r)
    # One batched 5-year download only for current Stage 2 candidates.
    stage2=[r["ticker"] for r in results if r.get("stage2")]
    if stage2:
        data.load_reliability(stage2)
        bench_daily=data.reliability_daily(RS_BENCHMARK)
        for r in results:
            if r.get("stage2"):
                try: r["stage2_reliability"]=historical_stage2_reliability_from_data(r["ticker"],data.reliability_daily(r["ticker"]),bench_daily)
                except Exception as e: r["stage2_reliability"]={"reliability":None,"reliability_label":"⚪ Error backtest","signals":0,"mean20":None,"median20":None}
    results.sort(key=lambda x:x["score"],reverse=True)
    return results
