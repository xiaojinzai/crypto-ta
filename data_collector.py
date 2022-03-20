import requests
import time
import pandas as pd
import datetime
import os

url = 'https://s2.bitcoinwisdom.io/period'

params = dict(
    step='604800',  # unit:second, use 604800 as week
    symbol='bitfinexbtcusd',
    mode='simple',
    nonce=str(int(time.time()))
)

resp = requests.get(url=url, params=params)
crypto_trade = resp.json()  # Check the JSON Response Content documentation below

name = ['time', 'unknown1', 'unknown2', 'open', 'close', 'high', 'low', 'vol', 'zero1', 'zero2', 'unknown3']

raw = pd.DataFrame(columns=name, data=crypto_trade)

now = datetime.datetime.now()
formatted_now = now.strftime('%y%m%d-%H%M%S')

save_path = os.path.join(os.environ['HOME'],'Downloads','token_analysis','test.csv')

raw.to_csv(save_path)
