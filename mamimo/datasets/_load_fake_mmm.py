"""Just some fake data for marketing mix modeling."""

import pandas as pd


def load_fake_mmm():
    """Load the data."""
    return pd.read_csv("./data/mmm.csv", parse_dates=["Date"], index_col="Date")
