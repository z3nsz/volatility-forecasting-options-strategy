from pathlib import Path

import yfinance as yf

RAW_DIR = Path("data/raw")


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
