import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from net import CNN
from data_process import create_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path, device):
    """加载保存的模型"""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    hparams = checkpoint['hparams']
    model = CNN(**hparams)
    
    # 初始化fc1层
    dummy_input = torch.randn(1, 10, 3).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input, training=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully!")
    return model, checkpoint

def get_model_predictions(model, data_loader, device):
    """获取模型预测结果"""
    model.eval()
    
    all_probabilities = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_books, batch_labels in data_loader:
            batch_books = batch_books.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(batch_books, training=False)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_probabilities.append(probabilities.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    # 合并所有批次
    all_probabilities = np.vstack(all_probabilities)
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    return all_probabilities, all_predictions, all_labels

def process_data_with_indices(temp_df, h, gamma, T):
    """处理数据并保留原始索引信息，用于收益率分析"""
    print('from', temp_df.index[0], 'to', temp_df.index[-1])

    # 保存原始时间索引，稍后用于收益率计算
    original_timestamps = temp_df.index.copy()
    
    # ① 重置索引
    temp_df_copy = temp_df.copy()
    temp_df_copy.reset_index(drop=True, inplace=True)

    # ② 买入信号识别
    temp_buy_index = pd.Series(temp_df_copy[((temp_df_copy['bid_price'].iloc[::-1].rolling(window=h).sum()/h).shift(1).iloc[::-1] - temp_df_copy['ask_price']) / temp_df_copy['ask_price'] > gamma].index)
    print(f"原数据买入点数量: {len(temp_buy_index)}")
    
    # ③ 卖出信号识别
    temp_sell_index = pd.Series(temp_df_copy[(temp_df_copy['bid_price'] - (temp_df_copy['ask_price'].iloc[::-1].rolling(window=h).sum()/h).shift(1)) / temp_df_copy['bid_price'] > gamma].index)
    print(f"原数据卖出点数量: {len(temp_sell_index)}")

    # ④ 特征工程
    temp_df_copy.loc[:, 'spread'] = 2 * (temp_df_copy['ask_price'] - temp_df_copy['bid_price']) / (temp_df_copy['ask_price'] + temp_df_copy['bid_price'])
    temp_df_copy[['bid_amount', 'ask_amount']] = temp_df_copy[['bid_amount', 'ask_amount']].div(temp_df_copy[['bid_amount', 'ask_amount']].sum(axis=1), axis=0)

    # ⑤ 去除连续信号
    temp_buy_points = temp_buy_index[temp_buy_index.diff(-1) != -1]
    temp_sell_points = temp_sell_index[temp_sell_index.diff(-1) != -1]
    
    # ⑥ 构建时间窗口
    temp_buy_windows = np.stack([temp_buy_points - t for t in range(T)]).T[:, ::-1]
    temp_sell_windows = np.stack([temp_sell_points - t for t in range(T)]).T[:, ::-1]

    # ⑦ 提取原数据
    temp_buys = temp_df_copy.values[:, 2:][temp_buy_windows]
    temp_sells = temp_df_copy.values[:, 2:][temp_sell_windows]

    # ⑧ 平衡数据
    temp_N = min(len(temp_buys), len(temp_sells))
    temp_hold_index = np.random.choice(temp_df_copy.index.difference(temp_buy_index).difference(temp_sell_index), size=temp_N, replace=False)
    temp_hold_windows = np.stack([temp_hold_index - t for t in range(T)]).T[:, ::-1]
    temp_holds = temp_df_copy.values[:, 2:][temp_hold_windows]

    # 随机抽取平衡数据
    buy_selected_indices = np.random.choice(temp_buys.shape[0], temp_N, replace=False)
    sell_selected_indices = np.random.choice(temp_sells.shape[0], temp_N, replace=False)
    
    temp_buys_selected = temp_buys[buy_selected_indices]
    temp_sells_selected = temp_sells[sell_selected_indices]

    # 保存对应的原始索引信息
    buy_original_indices = temp_buy_points.iloc[buy_selected_indices].values
    sell_original_indices = temp_sell_points.iloc[sell_selected_indices].values
    hold_original_indices = temp_hold_index

    print(temp_buys_selected.shape[0], 'buys')
    print(temp_holds.shape[0], 'holds')
    print(temp_sells_selected.shape[0], 'sells', end='\n\n')

    # ⑨ 组合数据
    temp_books = np.vstack((temp_buys_selected, temp_holds, temp_sells_selected))
    temp_labels = np.array([2]*temp_buys_selected.shape[0]+[1]*temp_holds.shape[0]+[0]*temp_sells_selected.shape[0])
    
    # 保存所有样本的原始索引
    all_original_indices = np.concatenate([buy_original_indices, hold_original_indices, sell_original_indices])

    temp_ds = {
        'book': temp_books,
        'label': temp_labels,
        'original_indices': all_original_indices,  # 新增：保存原始索引
        'original_timestamps': original_timestamps  # 新增：保存原始时间戳
    }
    return temp_ds

def calculate_sample_returns(df, sample_indices, h=50):
    """
    为特定样本计算真实的未来收益率
    
    Args:
        df: 原始价格数据DataFrame
        sample_indices: 样本对应的原始数据索引
        h: 未来窗口大小
    
    Returns:
        returns: 每个样本的真实未来收益率
    """
    mid_price = (df['ask_price'] + df['bid_price']) / 2
    
    # 为每个样本索引计算未来收益率
    returns = []
    for idx in sample_indices:
        if idx + h < len(df):
            # 当前价格
            current_price = mid_price.iloc[idx]
            # 未来h期的平均价格
            future_price = mid_price.iloc[idx+1:idx+h+1].mean()
            # 计算收益率
            if current_price > 0:
                ret = (future_price - current_price) / current_price
                returns.append(ret)
            else:
                returns.append(0.0)
        else:
            # 如果未来数据不足，设为0
            returns.append(0.0)
    
    return np.array(returns)

def process_data_with_returns(file_paths: List[str], h: int, gamma: float, T: int):
    """处理数据并计算真实收益率"""
    # 加载原始数据
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, 
                        usecols=['timestamp', 'ask_price', 'bid_price', 'ask_amount', 'bid_amount'])
        dfs.append(df)
    
    full_df = pd.concat(dfs, axis=0)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='us')
    full_df.set_index('timestamp', inplace=True)
    
    # 使用新的处理函数，保留索引信息
    dataset = process_data_with_indices(full_df, h, gamma, T)
    
    # 重置索引以便使用数值索引
    full_df_reset = full_df.reset_index(drop=True)
    
    # 为每个样本计算真实的未来收益率
    future_returns = calculate_sample_returns(full_df_reset, dataset['original_indices'], h)
    
    return dataset, future_returns, full_df

def analyze_top_predictions(probabilities, labels, future_returns, class_names=['Sell', 'Hold', 'Buy']):
    """分析top预测的真实收益率"""
    results = {}
    
    for class_idx, class_name in enumerate(class_names):
        class_probs = probabilities[:, class_idx]
        
        # 计算top 1%和top 0.1%的阈值
        top_1_threshold = np.percentile(class_probs, 99)
        top_01_threshold = np.percentile(class_probs, 99.9)
        
        # 找到top样本的索引
        top_1_mask = class_probs >= top_1_threshold
        top_01_mask = class_probs >= top_01_threshold
        
        # 计算这些样本的真实收益率
        if len(future_returns) == len(labels):
            top_1_returns = future_returns[top_1_mask]
            top_01_returns = future_returns[top_01_mask]
            
            # 移除NaN值
            top_1_returns = top_1_returns[~np.isnan(top_1_returns)]
            top_01_returns = top_01_returns[~np.isnan(top_01_returns)]
        else:
            print(f"Warning: Length mismatch between returns ({len(future_returns)}) and labels ({len(labels)})")
            top_1_returns = []
            top_01_returns = []
        
        results[class_name] = {
            'top_1_percent': {
                'threshold': top_1_threshold,
                'count': np.sum(top_1_mask),
                'returns': top_1_returns,
                'mean_return': np.mean(top_1_returns) if len(top_1_returns) > 0 else 0,
                'std_return': np.std(top_1_returns) if len(top_1_returns) > 0 else 0,
                'sharpe_ratio': np.mean(top_1_returns) / np.std(top_1_returns) if len(top_1_returns) > 0 and np.std(top_1_returns) > 0 else 0
            },
            'top_01_percent': {
                'threshold': top_01_threshold,
                'count': np.sum(top_01_mask),
                'returns': top_01_returns,
                'mean_return': np.mean(top_01_returns) if len(top_01_returns) > 0 else 0,
                'std_return': np.std(top_01_returns) if len(top_01_returns) > 0 else 0,
                'sharpe_ratio': np.mean(top_01_returns) / np.std(top_01_returns) if len(top_01_returns) > 0 and np.std(top_01_returns) > 0 else 0
            }
        }
    
    return results

def print_return_analysis(results):
    """打印收益率分析结果"""
    print("\n" + "="*80)
    print("TOP PREDICTIONS RETURN ANALYSIS")
    print("="*80)
    
    for class_name, class_results in results.items():
        print(f"\n{class_name.upper()} Predictions:")
        print("-" * 50)
        
        # Top 1%
        metrics_1 = class_results['top_1_percent']
        print(f"\nTop 1%:")
        print(f"  Threshold: {metrics_1['threshold']:.4f}")
        print(f"  Sample Count: {metrics_1['count']}")
        print(f"  Mean Return: {metrics_1['mean_return']*10000:.2f} bps")
        print(f"  Std Return: {metrics_1['std_return']*10000:.2f} bps")
        print(f"  Sharpe Ratio: {metrics_1['sharpe_ratio']:.4f}")
        
        # Top 0.1%
        metrics_01 = class_results['top_01_percent']
        print(f"\nTop 0.1%:")
        print(f"  Threshold: {metrics_01['threshold']:.4f}")
        print(f"  Sample Count: {metrics_01['count']}")
        print(f"  Mean Return: {metrics_01['mean_return']*10000:.2f} bps")
        print(f"  Std Return: {metrics_01['std_return']*10000:.2f} bps")
        print(f"  Sharpe Ratio: {metrics_01['sharpe_ratio']:.4f}")

def plot_return_distributions(results):
    """绘制收益率分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Return Distributions for Top Predictions', fontsize=16)
    
    class_names = list(results.keys())
    
    for i, class_name in enumerate(class_names):
        # Top 1%
        returns_1 = results[class_name]['top_1_percent']['returns']
        if len(returns_1) > 0:
            axes[0, i].hist(returns_1 * 10000, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, i].axvline(np.mean(returns_1) * 10000, color='red', linestyle='--', 
                             label=f'Mean: {np.mean(returns_1)*10000:.1f} bps')
            axes[0, i].set_title(f'{class_name} - Top 1%')
            axes[0, i].set_xlabel('Return (bps)')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].legend()
        
        # Top 0.1%
        returns_01 = results[class_name]['top_01_percent']['returns']
        if len(returns_01) > 0:
            axes[1, i].hist(returns_01 * 10000, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, i].axvline(np.mean(returns_01) * 10000, color='red', linestyle='--',
                             label=f'Mean: {np.mean(returns_01)*10000:.1f} bps')
            axes[1, i].set_title(f'{class_name} - Top 0.1%')
            axes[1, i].set_xlabel('Return (bps)')
            axes[1, i].set_ylabel('Frequency')
            axes[1, i].legend()
    
    plt.tight_layout()
    plt.show()

def create_data_loader(dataset, batch_size=1024):
    """创建数据加载器"""
    books = torch.FloatTensor(dataset['book'])
    labels = torch.LongTensor(dataset['label'])
    
    dataset_torch = TensorDataset(books, labels)
    data_loader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=False)
    
    return data_loader

if __name__ == "__main__":
    # 参数设置
    model_path = 'best_model.pth'
    test_file_paths = ['datasets/binance_book_ticker_2025-06-01_BTCUSDT.csv.gz']
    h, gamma, T = 50, 0, 10
    batch_size = 1024
    
    print("="*50)
    print("Top Predictions Return Analysis")
    print("="*50)
    
    # 加载模型
    model, checkpoint = load_model(model_path, device)
    
    # 使用新的函数获取带索引信息的数据和真实收益率
    print(f"\nLoading test data and calculating real future returns...")
    test_ds, future_returns, full_df = process_data_with_returns(test_file_paths, h, gamma, T)
    print(f"Test data shape: {test_ds['book'].shape}, labels: {test_ds['label'].shape}")
    print(f"Future returns shape: {future_returns.shape}")
    print(f"Sample indices shape: {test_ds['original_indices'].shape}")
    
    # 检查收益率统计
    valid_returns = future_returns[~np.isnan(future_returns)]
    print(f"Valid returns: {len(valid_returns)}/{len(future_returns)}")
    print(f"Returns stats: mean={np.mean(valid_returns)*10000:.2f}bps, std={np.std(valid_returns)*10000:.2f}bps")
    
    # 创建数据加载器
    test_loader = create_data_loader(test_ds, batch_size)
    
    # 获取模型预测
    print("\nGetting model predictions...")
    probabilities, predictions, labels = get_model_predictions(model, test_loader, device)
    
    # 检查数据长度
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Future returns length: {len(future_returns)}")
    
    # 现在数据长度应该完全匹配
    if len(future_returns) != len(labels):
        print(f"Error: Length mismatch! Returns: {len(future_returns)}, Labels: {len(labels)}")
        exit(1)
    
    # 分析top预测的收益率
    print("\nAnalyzing top predictions with real returns...")
    results = analyze_top_predictions(probabilities, labels, future_returns)
    
    # 打印结果
    print_return_analysis(results)
    
    # 绘制分布图
    plot_return_distributions(results)
    
    print("\nAnalysis completed using real future returns calculated from exact sample indices!")