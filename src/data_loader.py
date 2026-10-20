import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# Reads FRED_API_KEY from .env, so the key never appears in the code
load_dotenv()

RAW_DIR = Path("data/raw")

# CBOE computes these indexes, so its own files are the authoritative history
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
CBOE_INDEXES = ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW", "OVX"]

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
