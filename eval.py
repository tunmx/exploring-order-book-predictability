import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from net import CNN
from data_process import create_dataset

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model(checkpoint_path, device):
    """加载保存的模型"""
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    hparams = checkpoint['hparams']
    model = CNN(**hparams)
    
    # 需要先运行一次forward pass来初始化fc1层
    dummy_input = torch.randn(1, 10, 3).to(device)  # batch_size=1, T=10, features=3
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input, training=False)
    
    # 现在加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully!")
    return model, checkpoint

def evaluate_model(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = []
    test_accuracy = []
    
    with torch.no_grad():
        for batch_books, batch_labels in test_loader:
            batch_books = batch_books.to(device)
            batch_labels = batch_labels.to(device)
            
            # 前向传播
            logits = model(batch_books, training=False)
            loss = criterion(logits, batch_labels)
            
            # 计算准确率
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch_labels).float().mean()
            
            test_loss.append(loss.item())
            test_accuracy.append(accuracy.item())
    
    return np.mean(test_loss), np.mean(test_accuracy)

def create_data_loader(dataset, batch_size=1024):
    """创建数据加载器"""
    books = torch.FloatTensor(dataset['book'])
    labels = torch.LongTensor(dataset['label'])
    
    dataset_torch = TensorDataset(books, labels)
    data_loader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=False)
    
    return data_loader

if __name__ == "__main__":
    # 评估参数
    model_path = 'best_model.pth'
    test_file_paths = ['datasets/binance_book_ticker_2025-06-01_BTCUSDT.csv.gz']
    h, gamma, T = 50, 0, 10
    batch_size = 1024
    
    print("="*50)
    print("Model Evaluation")
    print("="*50)
    
    # 加载模型
    model, checkpoint = load_model(model_path, device)
    
    # 加载测试数据
    print(f"\nLoading test data...")
    test_ds = create_dataset(test_file_paths, h, gamma, T)
    print(f"Test data shape: {test_ds['book'].shape}, labels: {test_ds['label'].shape}")
    
    # 创建数据加载器
    test_loader = create_data_loader(test_ds, batch_size)
    
    # 评估模型
    print("\nEvaluating model...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    
    # 打印结果
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*50)