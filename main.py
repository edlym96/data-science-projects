#%%
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import quandl
import math
import pickle
from authkey import api_key
import model
from sklearn import preprocessing
# import keras
#from googlefinance.client import get_price_data
import datetime

quandl.ApiConfig.api_key = api_key
quandl_code = [
    'SMA/FBD',
    'SMA/FBP',
    # 'SPA/FBUP',
    'SMA/INSD',
    'SMA/INSP',
    'SMA/TWTD',
    'SMA/TWTT',
    'SMA/YTCD',
    'SMA/YTVD'
]

def load_from_quandl(quandl_code: str, ticker=None) -> pd.DataFrame():
    df = quandl.get_table(quandl_code, ticker=ticker) if ticker else quandl.get_table(quandl_code)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return pd.DataFrame(df)

# Facebook Analytics - Daily Metrics (Quandl code: SMA/FBD)
# Facebook Analytics - Admin Posts (Quandl code: SMA/FBP)
# Facebook Analytics - User Posts (Quandl code: SPA/FBUP)
# Instagram Analytics - Daily Metrics (Quandl code: SMA/INSD)
# Instagram Analytics - Posts (Quandl code: SMA/INSP)
# Twitter Analytics - Daily Metrics (Quandl code: SMA/TWTD)
# Twitter Analytics - Tweets (Quandl code: SMA/TWTT)
# Youtube Analytics - Channel Daily Metrics (Quandl code: SMA/YTCD)
# Youtube Analytics - Video Daily Metrics (Quandl code: SMA/YTVD)

def get_financial_data(ticker: str) -> pd.DataFrame():
    df = load_from_quandl('SHARADAR/SEP', ticker=ticker)
    return pd.DataFrame(df)

# def get_data_from_industry(sector: str) -> pd.DataFrame():
#     df = quandl.get_table('SMA/FBUP',brand_ticker = 'MCD')
#     # df = df.loc[df['sector'] == 'Restaurant & Cafe']
#     return pd.DataFrame(df)

def write_to_csv(df: pd.DataFrame(), name):
    # writer = pd.ExcelWriter(f'{name}.xlsx')
    df.to_csv(f'{name}.csv', columns=df.columns)
    # writer.save()

def init():
    for code in quandl_code:
        df = load_from_quandl(code)
        # write_to_csv(df, code.replace('/', '-'))

def get_table(ticker: str):
    df = load_from_quandl('IFT/NSA', ticker=ticker)
    # df = pd.DataFrame(pd.read_csv('SMA-FBD.csv', parse_dates=['date']))
    # type(df.index)
    # df = df.resample('W').mean()
    df = df[['sentiment', 'news_volume', 'news_buzz']]

    df_stock_price = get_financial_data(ticker)
    # df_stock_price = df_stock_price[['open', 'close', 'volume', 'dividends', 'closeunadj']]
    df_stock_price = df_stock_price[['close']]

    df = df.groupby('date').mean()
    df = df.join(df_stock_price)
    df = df.dropna()

    start = df.index.date[0]
    end = df.index.date[len(df.index)-1]

    # Get the S&P Data
    f = web.DataReader('^GSPC', 'yahoo', start, end)
    f = f[['Close']]
    f.columns = ['s_p_close']
    df = df.join(f)

    df['rolling_close'] = df['close'].rolling(50).mean()
    # df['rolling_close'] = df['rolling_close']
    # df['sentiment'] = np.where(df['sentiment'] > 0, 1.0, df['sentiment'])
    # df['sentiment'] = np.where(df['sentiment'] < 0, -1.0, df['sentiment'])
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = 2 * df['daily_return'].rolling(50).std()
    df['excess_volatility'] = np.where(df['daily_return'] > df['volatility'], 1.0, 0.0)
    df['excess_volatility'] = np.where(-df['daily_return'] > df['volatility'], -1.0, df['excess_volatility'])
    df.daily_return.shift(-1)
    df = df.dropna()
    return df

df = get_table('MCD')

# df = df[['sentiment',  'news_volume', 'excess_volatility']]
# df = (df - df.mean()) / (df.max() - df.min())
# df.plot()
# plt.show()

#split data set
train, test = df[:math.floor(0.8*len(df))], df[math.floor(0.8*len(df)):]
x_train, x_test, y_train, y_test = model.create_dataset(train, test)

# print(x_train, y_train, y_test)
# print(np.shape(x_train))
weights, bias = model.create_model(x_train, y_train, x_test, y_test)

print(weights, bias)

# df.plot()

# plt.show()
# start = pd.to_datetime(df.first_valid_index())
# end = pd.to_datetime(datetime(df.last_valid_index()))
# print(start, end)



# min_max_scaler = preprocessing.MinMaxScaler()
# np_scaled = min_max_scaler.fit_transform(df)
# df = pd.DataFrame(np_scaled)
# df = (df - df.mean()) / (df.max() - df.min())
# df.plot()
# plt.show()
# df = df.resample('W').mean()
# df.plot()
# plt.show()
# df.plot()
# plt.show()


# We have news for industry
# We have stock price for company
# filter_out_economy_contribution
