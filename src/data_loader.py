from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")

# CBOE computes these indexes, so its own files are the authoritative history
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
CBOE_INDEXES = ["VIX", "VIX9D", "VIX3M", "VVIX", "SKEW", "OVX"]


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
