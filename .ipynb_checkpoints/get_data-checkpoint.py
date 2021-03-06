import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import bs4 as bs
import pickle
import requests
import matplotlib.dates as mdates
import datetime as dt
import os
from matplotlib import style
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


import fix_yahoo_finance as yf


style.use('ggplot')

df = pd.read_csv('tesla.csv',parse_dates=True, index_col=0)
#df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
#print(df_ohlc.head())

#ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
#ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

#plt.show()

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    soup = bs.BeautifulSoup(resp.text)
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)

    with open('sp500tickers.pickle','wb') as f:
        pickle.dump(tickers, f)
    return tickers

tickers = save_sp500_tickers()

for index, ticker in enumerate(tickers):
    if ticker == 'BRK.B':
        tickers[index] = 'BRK'
    if ticker == 'BF.B':
        tickers.pop(index)
print(tickers)
def get_yahoo_data(reload_sp500=False):

    #if reload_sp500:
    #    tickers = save_sp500_tickers()
    #else:
    #    with open('sp500tickers.pickle','rb') as f:
    #        tickers = pickle.load(f)

    if not os.path.exists('stocks_dfs'):
        os.makedirs('stocks_dfs')

    start = dt.datetime(2010,1,1)
    end = dt.datetime(2020,1,1)

    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stocks_dfs/{}.csv'.format(ticker)):
            try:
                df = web.DataReader(str(ticker), 'yahoo', start, end)
                df.to_csv('stocks_dfs/{}.csv'.format(ticker))
            except:
                print('couldnt load data.')

        else:
            print('Already have {}'.format(ticker))

#get_yahoo_data()

def compile_data():
    with open("sp500tickers.pickle",'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        try:
            df = pd.read_csv('stocks_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)
            df.rename(columns= {'Adj Close': ticker}, inplace=True)
            df.drop(['Open','High','Low','Close','Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 ==0:
                print(count)
        except:
            print('no data.')
    print(main_df.head())
    print(main_df.values.shape)
    main_df.to_csv('sp500_joined_closes.csv')

#compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    df_corr = df.corr()
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) +0.5,  minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

#visualize_data()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker])

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.2
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf],0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

#extract_featuresets('AAPL')