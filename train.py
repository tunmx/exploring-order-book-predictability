import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any
import os

from net import CNN, create_model
from data_process import create_dataset

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_optimizer(optimizer_name: str, model_params, learning_rate: float):
    """根据优化器名称创建优化器"""
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model_params, lr=learning_rate)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_params, lr=learning_rate)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_params, lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'novograd':
        # Novograd 需要单独安装：pip install pytorch-optimizer
        try:
            import pytorch_optimizer as torch_opt
            return torch_opt.NovoGrad(model_params, lr=learning_rate)
        except ImportError:
            print("Warning: NovoGrad not available, falling back to AdamW")
            return optim.AdamW(model_params, lr=learning_rate)  # 修复：optax -> optim
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def apply_model(model, books, labels, criterion, device):
    """计算前向传播、损失和准确率"""
    books = books.to(device)
    labels = labels.to(device)
    
    # 前向传播
    logits = model(books, training=model.training)
    
    # 计算损失
    loss = criterion(logits, labels)
    
    # 计算准确率
    predictions = torch.argmax(logits, dim=-1)
    accuracy = (predictions == labels).float().mean()
    
    return loss, accuracy, logits

def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    epoch_loss = []
    epoch_accuracy = []
    
    for batch_books, batch_labels in tqdm(train_loader, desc="Training"):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        loss, accuracy, _ = apply_model(model, batch_books, batch_labels, criterion, device)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        epoch_loss.append(loss.item())
        epoch_accuracy.append(accuracy.item())
    
    return np.mean(epoch_loss), np.mean(epoch_accuracy)

def evaluate_model(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = []
    test_accuracy = []
    
    with torch.no_grad():
        for batch_books, batch_labels in test_loader:
            loss, accuracy, _ = apply_model(model, batch_books, batch_labels, criterion, device)
            test_loss.append(loss.item())
            test_accuracy.append(accuracy.item())
    
    return np.mean(test_loss), np.mean(test_accuracy)

def create_data_loaders(train_ds, test_ds, batch_size):
    """创建数据加载器"""
    # 转换为PyTorch张量
    train_books = torch.FloatTensor(train_ds['book'])
    train_labels = torch.LongTensor(train_ds['label'])
    
    test_books = torch.FloatTensor(test_ds['book'])
    test_labels = torch.LongTensor(test_ds['label'])
    
    # 创建数据集
    train_dataset = TensorDataset(train_books, train_labels)
    test_dataset = TensorDataset(test_books, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def count_parameters(model):
    """统计模型参数"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_and_evaluate(h, gamma, T, optimizer_name, learning_rate, batch_size, num_epochs, hparams):
    """执行模型训练和评估循环"""
    
    # 数据文件路径
    train_file_paths = [
        'datasets/binance_book_ticker_2025-01-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-02-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-03-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-04-01_BTCUSDT.csv.gz',
        'datasets/binance_book_ticker_2025-05-01_BTCUSDT.csv.gz',
    ]
    test_file_paths = [
        'datasets/binance_book_ticker_2025-06-01_BTCUSDT.csv.gz',
    ]
    
    print("Loading training data...")
    train_ds = create_dataset(train_file_paths, h, gamma, T)
    
    print("Loading test data...")
    test_ds = create_dataset(test_file_paths, h, gamma, T)
    
    print(f"Training data shape: {train_ds['book'].shape}, labels: {train_ds['label'].shape}")
    print(f"Test data shape: {test_ds['book'].shape}, labels: {test_ds['label'].shape}")
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(train_ds, test_ds, batch_size)
    
    # 创建模型
    print("\nCreating model...")
    model = CNN(**hparams)
    model = model.to(device)
    
    # 统计参数
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建优化器和损失函数
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting training with:")
    print(f"  - Optimizer: {optimizer_name}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Device: {device}")
    
    # 训练循环
    best_test_accuracy = 0.0
    training_history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # 训练
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 测试
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        
        # 记录历史
        training_history['train_loss'].append(train_loss)
        training_history['train_accuracy'].append(train_accuracy)
        training_history['test_loss'].append(test_loss)
        training_history['test_accuracy'].append(test_accuracy)
        
        # 打印结果（模仿原始格式）
        print('epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f' % 
              (epoch, train_loss, train_accuracy * 100, test_loss, test_accuracy * 100))
        
        # 保存最佳模型
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_accuracy,
                'hparams': hparams
            }, 'best_model.pth')
            print(f"  -> New best model saved! Test accuracy: {test_accuracy:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_accuracy:.4f}")
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_history': training_history,
        'hparams': hparams
    }, 'final_model.pth')
    
    return model, training_history

def load_model(checkpoint_path, hparams, device):
    """加载保存的模型"""
    model = CNN(**hparams)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model, checkpoint

if __name__ == "__main__":
    # 训练参数
    h = 50
    gamma = 0
    T = 10
    
    num_epochs = 50
    learning_rate = 0.03
    batch_size = 1024
    
    hparams = {
        'kernel_size_1': 1,
        'kernel_size_2': 5,
        'kernel_size_3': 8,
        'padding': 'SAME',
        'features': 6,
        'dense': 9,
        'activation': 'hardswish'  # 注意这里改为hardswish（没有下划线）
    }
    
    optimizer_name = "novograd"
    
    # 开始训练
    print("=" * 50)
    print("Starting Training Process")
    print("=" * 50)
    
    try:
        model, history = train_and_evaluate(
            h, gamma, T, optimizer_name, learning_rate, 
            batch_size, num_epochs, hparams
        )
        
        print("\nTraining successful!")
        print(f"Final model saved to: final_model.pth")
        print(f"Best model saved to: best_model.pth")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e