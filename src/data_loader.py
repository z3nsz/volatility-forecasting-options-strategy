import os
import warnings
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# Reads FRED_API_KEY from .env, so the key never appears in the code
load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Signals are computed at 4:30pm ET, after VIX's 4:15pm close and after the
# Fed posts the previous business day's H.15 rates (~4:15pm). So on day t the
# newest FRED value we may use is from the previous Fed business day.
FED_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
FRED_LAG_DAYS = 1

# CBOE computes these indexes, so its own files are the authoritative history
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
CBOE_INDEXES = ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW", "OVX"]

# VVIX exists from 2006 but has ~60 missing days that year, so start it later
CBOE_START = {"VVIX": "2007-01-01"}

# Saved exactly as FRED publishes them. The 1-business-day publication lag
# is applied later, when the datasets are aligned.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
FRED_SERIES = {
    "DTB3": "3-month T-bill rate (risk-free rate)",
    "BAA10Y": "Moody's Baa corporate yield minus 10-year Treasury",
}


def download_spy(start="2004-01-01"):
    # auto_adjust=False keeps both Close and Adj Close: we need the adjusted
    # close for returns (so dividend drops don't look like volatility)
    spy = yf.Ticker("SPY").history(start=start, auto_adjust=False)

    # Yahoo stamps dates with a New York timezone; plain dates are easier
    # to line up with VIX and FRED later
    spy.index = spy.index.tz_localize(None).normalize()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    spy.to_csv(RAW_DIR / "spy.csv")
    return spy


def download_move(start="2004-01-01"):
    # MOVE (Treasury option volatility) is published by ICE, not CBOE
    move = yf.Ticker("^MOVE").history(start=start)
    move.index = move.index.tz_localize(None).normalize()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    move.to_csv(RAW_DIR / "move.csv")
    return move


def download_cboe_index(name):
    df = pd.read_csv(
        CBOE_URL.format(name=name),
        index_col="DATE",
        parse_dates=["DATE"],
        date_format="%m/%d/%Y",  # CBOE writes US-style dates, e.g. 01/02/1990
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{name.lower()}.csv")
    return df


def download_all_cboe():
    return {name: download_cboe_index(name) for name in CBOE_INDEXES}


def download_fred_series(series):
    df = pd.read_csv(
        FRED_URL.format(series=series),
        index_col="observation_date",
        parse_dates=["observation_date"],
        na_values=".",  # FRED has used "." for days with no value
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / f"{series.lower()}.csv")
    return df


def download_all_fred():
    return {series: download_fred_series(series) for series in FRED_SERIES}


def download_cpi_release_dates(start="2004-01-01"):
    response = requests.get(
        "https://api.stlouisfed.org/fred/release/dates",
        params={
            "release_id": 10,  # FRED's id for the BLS Consumer Price Index release
            "realtime_start": start,
            "api_key": os.environ["FRED_API_KEY"],
            "file_type": "json",
            "limit": 10000,
        },
        timeout=30,
    )
    response.raise_for_status()

    dates = pd.to_datetime([row["date"] for row in response.json()["release_dates"]])
    df = pd.DataFrame({"date": dates})

    # Each February BLS also publishes revised seasonal factors a few days
    # before the January CPI report. Only the CPI report moves markets, and
    # it is always the later of the two, so keep the last date per month.
    df = df.groupby(df["date"].dt.to_period("M")).tail(1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "cpi_release_dates.csv", index=False)
    return df


def load_cboe_close(name):
    df = pd.read_csv(RAW_DIR / f"{name.lower()}.csv", index_col=0, parse_dates=True)
    # VIX, VIX9D and VIX3M come with OHLC columns; the others have one value column
    column = "CLOSE" if "CLOSE" in df.columns else name
    series = df[column].rename(name.lower())
    if name in CBOE_START:
        series = series[series.index >= CBOE_START[name]]
    return series


def align_spy_and_cboe():
    spy = pd.read_csv(RAW_DIR / "spy.csv", index_col=0, parse_dates=True)

    # Scale open/high/low by the same dividend adjustment as the close, so an
    # ex-dividend gap doesn't look like an overnight move
    adj_factor = spy["Adj Close"] / spy["Close"]
    daily = pd.DataFrame({
        "open": spy["Open"] * adj_factor,
        "high": spy["High"] * adj_factor,
        "low": spy["Low"] * adj_factor,
        "close": spy["Adj Close"],
    })

    # Left join on SPY's dates: index values on market holidays are dropped
    for name in CBOE_INDEXES:
        daily = daily.join(load_cboe_close(name))

    # VIX defines implied variance (half of our target), so it is never filled.
    # Rows without it, e.g. today before CBOE posts its file, are dropped.
    daily = daily.dropna(subset=["vix"])

    # Other indexes: carry yesterday's value over at most one missing day.
    # Uses only past data; longer gaps stay missing rather than being invented.
    for column in ["vix9d", "vix3m", "vvix", "skew", "ovx"]:
        before = daily[column].isna().sum()
        daily[column] = daily[column].ffill(limit=1)
        print(f"{column}: forward-filled {before - daily[column].isna().sum()} day(s)")

    return daily


def load_fred_lagged(series):
    df = pd.read_csv(RAW_DIR / f"{series.lower()}.csv", index_col=0, parse_dates=True)
    values = df[series].dropna()  # bond-market holidays have no value
    values = values[values.index >= "2003-01-01"]  # older history isn't needed

    # Re-date each value to the day it became public. (pandas warns that this
    # offset isn't vectorized; harmless at a few thousand rows.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        values.index = values.index + FRED_LAG_DAYS * FED_BUSINESS_DAY

    # When a holiday falls on Saturday, pandas treats Friday as a day off but
    # the Fed stays open, so Thursday's and Friday's values both land on Monday.
    # Keep the newer one: it is the latest information public that day.
    values = values.groupby(level=0).last()
    return values.rename(series.lower())


def build_daily_dataset():
    daily = align_spy_and_cboe()

    for series in FRED_SERIES:
        values = load_fred_lagged(series)
        # For each SPY day, take the latest value already public by then
        daily[series.lower()] = values.reindex(daily.index, method="ffill")

    # ICE's publication time for the MOVE close isn't documented, so assume it
    # is only known the next trading day. Then carry over at most one missing day.
    move = pd.read_csv(RAW_DIR / "move.csv", index_col=0, parse_dates=True)["Close"]
    daily["move"] = move.reindex(daily.index).shift(1).ffill(limit=1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(PROCESSED_DIR / "daily.csv")
    return daily
