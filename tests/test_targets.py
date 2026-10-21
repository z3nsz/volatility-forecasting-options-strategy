import numpy as np
import pandas as pd
import pytest

from src.targets import implied_variance, realized_variance_target


def make_prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)


def test_hand_calculated_window():
    dates = pd.to_datetime(["2020-01-01", "2020-01-10", "2020-01-31", "2020-02-01", "2020-03-15"])
    close = pd.Series([100, 110, 99, 105, 120], index=dates)
    rv = realized_variance_target(close)

    # window for jan 1 is (jan 1, jan 31]: the returns on jan 10 and jan 31
    expected = np.log(110 / 100) ** 2 + np.log(99 / 110) ** 2
    assert rv["2020-01-01"] == pytest.approx(expected)


def test_past_prices_dont_change_target():
    close = make_prices()
    t = close.index[50]
    before = realized_variance_target(close)[t]

    # close on t itself is the start of the first return in the window, so only change days before t
    changed = close.copy()
    changed[changed.index < t] *= np.linspace(0.5, 2, (changed.index < t).sum())
    after = realized_variance_target(changed)[t]

    assert after == pytest.approx(before)


def test_future_prices_do_change_target():
    close = make_prices()
    t = close.index[50]
    before = realized_variance_target(close)[t]

    changed = close.copy()
    changed[close.index[55]:] *= 1.05  # 5% jump inside the window
    assert realized_variance_target(changed)[t] != pytest.approx(before)


def test_unfinished_windows_are_empty():
    close = make_prices()
    rv = realized_variance_target(close)
    unfinished = close.index + pd.Timedelta(days=30) > close.index[-1]
    assert rv[unfinished].isna().all()
    assert rv[~unfinished].notna().all()


def test_implied_variance():
    assert implied_variance(20) == pytest.approx(0.04 * 30 / 365)
