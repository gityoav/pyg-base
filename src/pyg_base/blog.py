# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:21:47 2025

@author: Dell
"""

import polars as pl
import arcticdb as adb
import matplotlib.pyplot as plt
import mplfinance as mpf
from pyg_base import dt, dictable
from functools import lru_cache

# Load the csv data using polars
df = pl.scan_csv("D:/fx-data/DAT_ASCII_EURUSD_T_*.csv", has_header=False)
 
# Extract time, bid, ask and mid (mid is calculated from bid and ask)
df = df.select(
    pl.col("column_1").alias("timestamp").str.to_datetime("%Y%m%d %H%M%S%3f", time_unit="ns"),
    pl.col("column_2").alias("bid"),
    pl.col("column_3").alias("ask"),
    (pl.col("column_2") + ((pl.col("column_3") - pl.col("column_2")) / 2)).alias("mid"),
 ).sort("timestamp", descending=False)
 
# Convert to pandas and ensure index is sorted
dfp = df.collect().to_pandas()
dfp = dfp.set_index("timestamp")
dfp = dfp.sort_index()


Arctic = lru_cache()(adb.Arctic)
# Create the ArcticDB library on our local file system
ac = Arctic("lmdb://D:/fx-data/arcticdb")
lib = ac.get_library("fx-test", create_if_missing=True)
 
# Write the DataFrame created above to the ArcticDB library
lib.write("EURUSD", dfp)

def adb_library(url, library_name):
    ac = Arctic(url)
    return ac[library_name]

def raw_tick_data(lib, symbol, start, end):
    data = lib.read(symbol, date_range=[dt(start), dt(end)])
    return data.data

start = dt(2023, 8, 3, 7, 0, 0)
end = dt(2023, 8, 3, 7, 1, 0)
 
data = raw_tick_data(lib, "EURUSD", start, end)

def resampled_tick_data(lib, symbol, start, end, freq):
    qb = adb.QueryBuilder() 
    qb = qb.resample(freq, closed='right').agg({
        'high': ('mid', 'max'),
        'low': ('mid', 'min'),
        'open': ('mid', 'first'),
        'close': ('mid', 'last')
    })
 
    data = lib.read(symbol,
                    date_range=[dt(start), dt(end)],
                    query_builder=qb)
 
    df = data.data.dropna()
    return df

ohlc = resampled_tick_data(lib, 'EURUSD', dt(2023,12,1), dt(2023,12,10), freq='5min')

def plot_bars(ohlc, mav = [], title = 'data', style="seaborn-v0_8"):
    with plt.style.context(style):
        # Get a matplotlib Figure and Axes object to plot the candlestick chart on
        fig, ax = plt.subplots()
        fig.suptitle(title)
        ax.grid(True)
 
        # Use matplotlibfinance to draw the candlestick chart
        mpf.plot(ohlc,
                 type='candle',
                 mav=[x for x in mav if x],
                 ax=ax)
 
        fig.tight_layout()
        fig.show()
        # Show the chart in Excel
    return fig

fig = plot_bars(ohlc, mav = [4,10], title = 'EURUSD')
