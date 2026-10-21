import pandas as pd


def earnings_season_flag(dates):
    # rough rule: 10th of month after quarter end -> 20th of the next month
    first = dates.month.isin([1, 4, 7, 10]) & (dates.day >= 10)
    second = dates.month.isin([2, 5, 8, 11]) & (dates.day <= 20)
    return pd.Series((first | second).astype(int), index=dates, name="earnings_season")
