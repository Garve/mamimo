"""Deal with time features in dataframes."""

import numpy as np
import pandas as pd


def add_date_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Enrich a pandas dataframes with a new column indicating if there is a special date.
    This new column will contain a one for each date specified in the `dates` keyword,
    zero otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with a DateTime index.

    kwargs : List[str]*
        As many inputs as you want of the form date_name=[date_1, date_2, ...], i.e.
        christmas=['2020-12-24']. See the example below for more information.

    Returns
    -------
    pd.DataFrame
        A dataframe with date indicator columns.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"A": range(7)},
    ...     index=pd.date_range(start="2019-12-29", periods=7)
    ... )
    >>> add_date_indicators(
    ...     df,
    ...     around_new_year_2020=["2019-12-31", "2020-01-01", "2020-01-02"],
    ...     other_date_1=["2019-12-29"],
    ...     other_date_2=["2018-01-01"]
    ... )
                A  around_new_year_2020  other_date_1  other_date_2
    2019-12-29  0                     0             1             0
    2019-12-30  1                     0             0             0
    2019-12-31  2                     1             0             0
    2020-01-01  3                     1             0             0
    2020-01-02  4                     1             0             0
    2020-01-03  5                     0             0             0
    2020-01-04  6                     0             0             0
    """
    return df.assign(
        **{name: df.index.isin(dates).astype(int) for name, dates in kwargs.items()}
    )


def add_time_features(
    df: pd.DataFrame,
    second: bool = False,
    minute: bool = False,
    hour: bool = False,
    day_of_week: bool = False,
    day_of_month: bool = False,
    day_of_year: bool = False,
    week_of_month: bool = False,
    week_of_year: bool = False,
    month: bool = False,
    year: bool = False,
) -> pd.DataFrame:
    """
    Enrich pandas dataframes with new columns which are easy derivations from its
    DatetimeIndex, such as the day of week or the month.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe with a DateTime index.

    second : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    minute : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    hour : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    day_of_week : bool, default=False
        Whether to extract the day of week from the index and add it as a new column.

    day_of_month : bool, default=False
        Whether to extract the day of month from the index and add it as a new column.

    day_of_year : bool, default=False
        Whether to extract the day of year from the index and add it as a new column.

    week_of_month : bool, default=False
        Whether to extract the week of month from the index and add it as a new column.

    week_of_year : bool, default=False
        Whether to extract the week of year from the index and add it as a new column.

    month : bool, default=False
        Whether to extract the month from the index and add it as a new column.

    year : bool, default=False
        Whether to extract the year from the index and add it as a new column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"A": ["a", "b", "c"]},
    ...     index=[
    ...         pd.Timestamp("1988-08-08"),
    ...         pd.Timestamp("2000-01-01"),
    ...         pd.Timestamp("1950-12-31"),
    ...     ])
    >>> add_time_features(df, day_of_month=True, month=True, year=True)
                A  day_of_month  month  year
    1988-08-08  a             8      8  1988
    2000-01-01  b             1      1  2000
    1950-12-31  c            31     12  1950
    """

    def _add_second(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(second=df.index.second) if second else df

    def _add_minute(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(minute=df.index.minute) if minute else df

    def _add_hour(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(hour=df.index.hour) if hour else df

    def _add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(day_of_week=df.index.weekday + 1) if day_of_week else df

    def _add_day_of_month(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(day_of_month=df.index.day) if day_of_month else df

    def _add_day_of_year(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(day_of_year=df.index.dayofyear) if day_of_year else df

    def _add_week_of_month(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.assign(week_of_month=np.ceil(df.index.day / 7).astype(int))
            if week_of_month
            else df
        )

    def _add_week_of_year(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.assign(week_of_year=df.index.isocalendar().week) if week_of_year else df
        )

    def _add_month(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(month=df.index.month) if month else df

    def _add_year(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(year=df.index.year) if year else df

    return (
        df.pipe(_add_second)
        .pipe(_add_minute)
        .pipe(_add_hour)
        .pipe(_add_day_of_week)
        .pipe(_add_day_of_month)
        .pipe(_add_day_of_year)
        .pipe(_add_week_of_month)
        .pipe(_add_week_of_year)
        .pipe(_add_month)
        .pipe(_add_year)
    )
