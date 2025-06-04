import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Any  # 添加必要的类型导入

def process_data(temp_df, h, gamma, T):
    """根据历史数据来制作数据集"""
    print('from', temp_df.index[0], 'to', temp_df.index[-1])

    # ① 重置索引
    temp_df.reset_index(drop=True, inplace=True)

    # ② 买入信号识别
    # 步骤分解：
    # 1. temp_df['bid_price']: 获取所有的买入价格
    # 2. .iloc[::-1]: 把数据反转（最后一个变第一个）
    # 3. .rolling(window=h).sum()/h: 计算移动平均（窗口=50）
    # 4. .shift(1): 向前移动1位
    # 5. .iloc[::-1]: 再次反转回原顺序
    # 6. 计算买入收益率 = (未来平均买价 - 当前卖价) / 当前卖价
    # 7. 如果 r_buy > gamma(0)，标记为买入点
    # 8. 获取买入点的时间索引
    temp_buy_index = pd.Series(temp_df[((temp_df['bid_price'].iloc[::-1].rolling(window=h).sum()/h).shift(1).iloc[::-1] - temp_df['ask_price']) / temp_df['ask_price'] > gamma].index)
    print(f"原数据买入点数量: {len(temp_buy_index)}")
    
    # ③ 卖出信号识别
    # 步骤分解：
    # 1. temp_df['bid_price']: 获取所有的买入价格
    # 2. .iloc[::-1]: 把数据反转（最后一个变第一个）
    # 3. .rolling(window=h).sum()/h: 计算移动平均（窗口=50）
    # 4. .shift(1): 向前移动1位
    # 5. .iloc[::-1]: 再次反转回原顺序
    # 6. 计算卖出收益率 = (当前买价 - 未来平均卖价) / 当前买价
    # 7. 如果 r_sell > gamma(0)，标记为卖出点
    # 8. 获取卖出点的时间索引
    temp_sell_index = pd.Series(temp_df[(temp_df['bid_price'] - (temp_df['ask_price'].iloc[::-1].rolling(window=h).sum()/h).shift(1)) / temp_df['bid_price'] > gamma].index)
    print(f"原数据卖出点数量: {len(temp_sell_index)}")

    # ④ 特征工程
    
    # 价差 = 卖价 - 买价
    # 归一化 = 2 × 价差 / (卖价 + 买价)
    # 这样价差变成了一个相对值（百分比形式）
    temp_df.loc[:, 'spread'] = 2 * (temp_df['ask_price'] - temp_df['bid_price']) / (temp_df['ask_price'] + temp_df['bid_price'])

    # 归一化数量
    # 每一行：买量 + 卖量 = 总量
    # 买量 = 买量 / 总量
    # 卖量 = 卖量 / 总量
    # 结果：买量 + 卖量 = 1
    temp_df[['bid_amount', 'ask_amount']] = temp_df[['bid_amount', 'ask_amount']].div(temp_df[['bid_amount', 'ask_amount']].sum(axis=1), axis=0)

    # ⑤ 去除连续信号
    # 例如买入信号在索引[100,101,102,105,106]
    # 只保留差值不等于-1的，得到[102,106]
    # 这样每段连续信号只保留最后一个
    temp_buy_points = temp_buy_index[temp_buy_index.diff(-1) != -1]
    temp_sell_points = temp_sell_index[temp_sell_index.diff(-1) != -1]
    
    # ⑥ 构建时间窗口
    # 举例假设T=3
    # 如果买入信号在索引100
    # 生成：[100-2, 100-1, 100] = [98, 99, 100]
    # 这就是一个包含3个历史时间点的窗口
    temp_buy_windows = np.stack([temp_buy_points - t for t in range(T)]).T[:, ::-1]
    temp_sell_windows = np.stack([temp_sell_points - t for t in range(T)]).T[:, ::-1]

    # ⑦ 提取原数据
    # 获取从第3列开始的所有数据 因为前两列是exchange和symbol，不需要 
    # 实际获取的是：ask_amount, ask_price, bid_price, bid_amount, spread
    temp_buys = temp_df.values[:, 2:][temp_buy_windows]
    temp_sells = temp_df.values[:, 2:][temp_sell_windows]

    # ⑧ 平衡数据
    temp_N = min(len(temp_buys), len(temp_sells))
    # 找出买入和卖出样本中较少的数量，确保数据平衡。
    # difference(): 找出既不是买入也不是卖出的时间点
    # random.choice(): 随机选择temp_N个作为"持有"样本
    temp_hold_index = np.random.choice(temp_df.index.difference(temp_buy_index).difference(temp_sell_index), size=temp_N)
    temp_hold_windows = np.stack([temp_hold_index - t for t in range(T)]).T[:, ::-1]
    temp_holds = temp_df.values[:, 2:][temp_hold_windows]

    # 为持有信号构建窗口并提取数据。
    temp_buys = temp_buys[np.random.choice(temp_buys.shape[0], temp_N, replace=False)]
    temp_sells = temp_sells[np.random.choice(temp_sells.shape[0], temp_N, replace=False)]

    print(temp_buys.shape[0], 'buys')
    print(temp_holds.shape[0], 'holds')
    print(temp_sells.shape[0], 'sells', end='\n\n')

    # ⑨ 组合数据
    temp_books = np.vstack((temp_buys, temp_holds, temp_sells))
    # 将三种类型的数据垂直堆叠成一个大数组。
    # 2 = 买入
    # 1 = 持有
    # 0 = 卖出
    temp_labels = np.array([2]*temp_buys.shape[0]+[1]*temp_holds.shape[0]+[0]*temp_sells.shape[0])

    temp_ds = {'book': temp_books,
                'label': temp_labels}
    return temp_ds


def create_dataset(file_paths: List[str], h: int, gamma: float, T: int):
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, 
                        usecols=['timestamp', 'ask_price', 'bid_price', 'ask_amount', 'bid_amount'], 
                        engine='pyarrow')
        dfs.append(df)
    train_df = pd.concat(dfs, axis=0)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], unit='us')
    train_df.set_index('timestamp', inplace=True)
    train_ds = process_data(train_df, h=h, gamma=gamma, T=T)
    return train_ds
        


if __name__ == '__main__':
    train_file_paths = [
        'datasets/binance_book_ticker_2025-01-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-02-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-03-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-04-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-05-01_BTCUSDT.csv.gz',
    ]
    val_file_paths = [
        'datasets/binance_book_ticker_2025-06-01_BTCUSDT.csv.gz',
    ]

    h = 50
    gamma = 0
    T = 10
    print("Training data:")
    train_ds = create_dataset(train_file_paths, h, gamma, T)
    print(train_ds['book'])
    print(train_ds['label'])