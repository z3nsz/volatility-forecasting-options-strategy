import numpy as np
import pandas as pd

HORIZON = 30  # calendar days


def implied_variance(vix):
    return (vix / 100) ** 2 * HORIZON / 365


def realized_variance_target(close):
    r2 = np.log(close).diff() ** 2
    last_date = close.index[-1]
    rv = pd.Series(np.nan, index=close.index, name="rv_target")

    for t in close.index:
        end = t + pd.Timedelta(days=HORIZON)
        if end > last_date:
            break  # window not finished yet
        rv[t] = r2[(r2.index > t) & (r2.index <= end)].sum()

    return rv
