import os
import warnings
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# signal at 4:30pm ET -> FRED (H.15 posts ~4:15pm next day) lags 1 Fed business day
FED_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
FRED_LAG = 1

CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
CBOE_INDEXES = ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW", "OVX"]
CBOE_START = {"VVIX": "2007-01-01"}  # 2006 is patchy

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SERIES = ["DTB3", "BAA10Y"]


def download_spy(start="2004-01-01"):
    spy = yf.Ticker("SPY").history(start=start, auto_adjust=False)
    spy.index = spy.index.tz_localize(None).normalize()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    spy.to_csv(RAW_DIR / "spy.csv")
    return spy


def download_move(start="2004-01-01"):
    move = yf.Ticker("^MOVE").history(start=start)
    move.index = move.index.tz_localize(None).normalize()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    move.to_csv(RAW_DIR / "move.csv")
    return move


def download_cboe_index(name):
    df = pd.read_csv(CBOE_URL.format(name=name), index_col="DATE",
                     parse_dates=["DATE"], date_format="%m/%d/%Y")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{name.lower()}.csv")
    return df


def download_all_cboe():
    return {name: download_cboe_index(name) for name in CBOE_INDEXES}


def download_fred_series(series):
    df = pd.read_csv(FRED_URL.format(series=series), index_col="observation_date",
                     parse_dates=["observation_date"], na_values=".")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{series.lower()}.csv")
    return df


def download_all_fred():
    return {series: download_fred_series(series) for series in FRED_SERIES}


def download_cpi_release_dates(start="2004-01-01"):
    r = requests.get(
        "https://api.stlouisfed.org/fred/release/dates",
        params={"release_id": 10, "realtime_start": start, "limit": 10000,
                "api_key": os.environ["FRED_API_KEY"], "file_type": "json"},
        timeout=30,
    )
    r.raise_for_status()
    df = pd.DataFrame({"date": pd.to_datetime([x["date"] for x in r.json()["release_dates"]])})
    # feb also has a seasonal factor revision a few days before the actual CPI report
    df = df.groupby(df["date"].dt.to_period("M")).tail(1)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "cpi_release_dates.csv", index=False)
    return df


def load_cboe_close(name):
    df = pd.read_csv(RAW_DIR / f"{name.lower()}.csv", index_col=0, parse_dates=True)
    col = "CLOSE" if "CLOSE" in df.columns else name
    s = df[col].rename(name.lower())
    if name in CBOE_START:
        s = s[s.index >= CBOE_START[name]]
    return s


def align_spy_and_cboe():
    spy = pd.read_csv(RAW_DIR / "spy.csv", index_col=0, parse_dates=True)
    adj = spy["Adj Close"] / spy["Close"]
    daily = pd.DataFrame({
        "open": spy["Open"] * adj,
        "high": spy["High"] * adj,
        "low": spy["Low"] * adj,
        "close": spy["Adj Close"],
    })

    for name in CBOE_INDEXES:
        daily = daily.join(load_cboe_close(name))

    daily = daily.dropna(subset=["vix"])  # never fill vix
    for col in ["vix9d", "vix3m", "vvix", "skew", "ovx"]:
        n_missing = daily[col].isna().sum()
        daily[col] = daily[col].ffill(limit=1)
        print(f"{col}: filled {n_missing - daily[col].isna().sum()}")

    return daily


def load_fred_lagged(series):
    df = pd.read_csv(RAW_DIR / f"{series.lower()}.csv", index_col=0, parse_dates=True)
    s = df[series].dropna()
    s = s[s.index >= "2003-01-01"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        s.index = s.index + FRED_LAG * FED_BDAY

    # saturday holidays: pandas skips friday but the fed is open, so two values land on monday
    s = s.groupby(level=0).last()
    return s.rename(series.lower())


def build_daily_dataset():
    daily = align_spy_and_cboe()

    for series in FRED_SERIES:
        daily[series.lower()] = load_fred_lagged(series).reindex(daily.index, method="ffill")

    # MOVE publish time unknown -> lag 1 day
    move = pd.read_csv(RAW_DIR / "move.csv", index_col=0, parse_dates=True)["Close"]
    daily["move"] = move.reindex(daily.index).shift(1).ffill(limit=1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED_DIR / "daily.csv")
    return daily
