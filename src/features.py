import pandas as pd


def earnings_season_flag(dates):
    # Approximate rule: big banks open the season around the 10th of the month
    # after quarter end, and ~90% of S&P 500 firms have reported by about the
    # 20th of the next month. A fixed calendar, so it uses no future data.
    first_month = dates.month.isin([1, 4, 7, 10]) & (dates.day >= 10)
    second_month = dates.month.isin([2, 5, 8, 11]) & (dates.day <= 20)
    in_season = first_month | second_month
    return pd.Series(in_season.astype(int), index=dates, name="earnings_season")
