import pandas as pd
import numpy as np
import jax.numpy as jnp

def process_data(temp_df, h, gamma, T):
    print('from', temp_df.index[0], 'to', temp_df.index[-1])
    temp_df.reset_index(drop=True, inplace=True)

    temp_buy_index = pd.Series(temp_df[((temp_df['bid_price'].iloc[::-1].rolling(window=h).sum()/h).shift(1).iloc[::-1] - temp_df['ask_price']) / temp_df['ask_price'] > gamma].index)
    temp_sell_index = pd.Series(temp_df[(temp_df['bid_price'] - (temp_df['ask_price'].iloc[::-1].rolling(window=h).sum()/h).shift(1)) / temp_df['bid_price'] > gamma].index)

    temp_df.loc[:, 'spread'] = 2 * (temp_df['ask_price'] - temp_df['bid_price']) / (temp_df['ask_price'] + temp_df['bid_price'])
    temp_df[['bid_amount', 'ask_amount']] = temp_df[['bid_amount', 'ask_amount']].div(temp_df[['bid_amount', 'ask_amount']].sum(axis=1), axis=0)

    temp_buy_points = temp_buy_index[temp_buy_index.diff(-1) != -1]
    temp_sell_points = temp_sell_index[temp_sell_index.diff(-1) != -1]

    temp_buy_windows = np.stack([temp_buy_points - t for t in range(T)]).T[:, ::-1]
    temp_sell_windows = np.stack([temp_sell_points - t for t in range(T)]).T[:, ::-1]

    temp_buys = temp_df.values[:, 2:][temp_buy_windows]
    temp_sells = temp_df.values[:, 2:][temp_sell_windows]

    temp_N = min(len(temp_buys), len(temp_sells))

    temp_hold_index = np.random.choice(temp_df.index.difference(temp_buy_index).difference(temp_sell_index), size=temp_N)
    temp_hold_windows = np.stack([temp_hold_index - t for t in range(T)]).T[:, ::-1]
    temp_holds = temp_df.values[:, 2:][temp_hold_windows]

    temp_buys = temp_buys[np.random.choice(temp_buys.shape[0], temp_N, replace=False)]
    temp_sells = temp_sells[np.random.choice(temp_sells.shape[0], temp_N, replace=False)]

    print(temp_buys.shape[0], 'buys')
    print(temp_holds.shape[0], 'holds')
    print(temp_sells.shape[0], 'sells', end='\n\n')

    temp_books = np.vstack((temp_buys, temp_holds, temp_sells))
    temp_labels = np.array([2]*temp_buys.shape[0]+[1]*temp_holds.shape[0]+[0]*temp_sells.shape[0])

    temp_ds = {'book': jnp.array(temp_books),
                'label': jnp.array(temp_labels)}
    return temp_ds


if __name__ == '__main__':
    file_paths = [
        'datasets/binance_book_ticker_2025-01-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-02-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-03-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-04-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-05-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-06-01_BTCUSDT.csv.gz'
    ]

    h = 50
    gamma = 0
    T = 10
    print("Training data:")
    train_df = pd.concat([pd.read_csv(train_file_path, usecols=['timestamp', 'ask_price', 'bid_price', 'ask_amount', 'bid_amount'], engine='pyarrow') for train_file_path in file_paths[:-1]], axis=0)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='us')
    train_df.set_index('timestamp', inplace=True)
    train_ds = process_data(train_df, h=h, gamma=gamma, T=T)
    print(train_ds['book'].shape)
    print(train_ds['label'].shape)