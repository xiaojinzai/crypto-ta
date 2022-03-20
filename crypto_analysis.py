import datetime
import pandas as pd
import pandas_ta as ta
import warnings
import os
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option("display.max_rows", 1000)

save_path = os.path.join(os.environ['HOME'],'Downloads','token_analysis','test.csv')

origin_df = pd.read_csv(save_path)

df = origin_df.copy()

df['Date'] = origin_df['time'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

df = df.rename(columns={'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'vol': 'Volume'})
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
df.head(100)

sma = ta.sma(df["Close"], length=60)
df['sma'] = sma

ema = ta.ema(df["Close"], length=60)
df['ema'] = ema

rsi = ta.rsi(df.Close)
df['RSI'] = rsi

atr = ta.atr(df.High, df.Low, df.Close)
atr.plot(figsize=(16, 3), color=["black"], title='ATR', grid=True)

sma_slope = ta.slope(df['sma'], as_angle=True)
df['sma_slope'] = sma_slope

ema_slope = ta.slope(df['ema'])
df['ema_slope'] = ema_slope

print("done")