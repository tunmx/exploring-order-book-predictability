import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

class CNN(nn.Module):
    """CNN Model for trading signals prediction"""
    
    def __init__(self, 
                 kernel_size_1: int,
                 kernel_size_2: int, 
                 kernel_size_3: int,
                 padding: str,
                 features: int,
                 dense: int,
                 activation: str):
        super(CNN, self).__init__()
        
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.kernel_size_3 = kernel_size_3
        self.padding = padding
        self.features = features
        self.dense = dense
        self.activation = activation
        
        # Get activation function
        self.activation_fn = self._get_activation_fn(activation)
        
        # Define layers
        # First conv layer: (1, kernel_size_1) - across time dimension
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2**features, 
                              kernel_size=(1, kernel_size_1))
        self.bn1 = nn.BatchNorm2d(2**features)
        
        # Second conv layer: (kernel_size_2, 1) - across feature dimension
        padding_val = self._get_padding_value(kernel_size_2, padding)
        self.conv2 = nn.Conv2d(in_channels=2**features, out_channels=2**features,
                              kernel_size=(kernel_size_2, 1), padding=(padding_val, 0))
        self.bn2 = nn.BatchNorm2d(2**features)
        
        # Third conv layer: (kernel_size_3, 1) - across feature dimension  
        padding_val = self._get_padding_value(kernel_size_3, padding)
        self.conv3 = nn.Conv2d(in_channels=2**features, out_channels=2**features,
                              kernel_size=(kernel_size_3, 1), padding=(padding_val, 0))
        self.bn3 = nn.BatchNorm2d(2**features)
        
        # Dense layers will be initialized in forward pass after we know the flattened size
        self.fc1 = None
        self.fc2 = nn.Linear(2**dense, 3)  # buy, hold, sell
        
    def _get_activation_fn(self, activation: str):
        """Get activation function by name"""
        if activation.lower() == 'relu':
            return F.relu
        elif activation.lower() == 'gelu':
            return F.gelu
        elif activation.lower() == 'tanh':
            return torch.tanh
        elif activation.lower() == 'sigmoid':
            return torch.sigmoid
        elif activation.lower() == 'hardswish':
            return F.hardswish
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _get_padding_value(self, kernel_size: int, padding: str):
        """Convert padding string to padding value"""
        if padding.upper() == 'SAME':
            return kernel_size // 2
        elif padding.upper() == 'VALID':
            return 0
        else:
            raise ValueError(f"Unsupported padding: {padding}")
    
    def forward(self, x, training=True):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, T, 3) where T is sequence length
            training: Whether in training mode (for batch norm)
        """
        # Convert from (batch_size, T, 3) to (batch_size, 3, T, 1) for Conv2d
        # This treats the 3 features as channels and T as one spatial dimension
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (batch_size, 3, T, 1)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation_fn(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation_fn(x)
        
        # Flatten for dense layers
        x_flat = x.view(x.size(0), -1)
        
        # Initialize first dense layer if not done yet
        if self.fc1 is None:
            self.fc1 = nn.Linear(x_flat.size(1), 2**self.dense).to(x.device)
        
        # Dense layers
        x = self.fc1(x_flat)
        x = self.activation_fn(x)
        
        x = self.fc2(x)
        
        return x

# Example usage and model creation
def create_model(hparams: Dict[str, Any], input_shape: tuple = (100, 3)):
    """
    Create CNN model with given hyperparameters
    
    Args:
        hparams: Dictionary containing model hyperparameters
        input_shape: Shape of input (T, features) where T is sequence length
    
    Returns:
        CNN model instance
    """
    model = CNN(**hparams)
    
    # Print model summary
    print("Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

# Example hyperparameters (matching the original structure)
example_hparams = {
    'kernel_size_1': 1,
    'kernel_size_2': 5,
    'kernel_size_3': 8,
    'padding': 'SAME',
    'features': 6,
    'dense': 9,
    'activation': 'hardswish'
}

if __name__ == "__main__":
    # Test the model
    model = create_model(example_hparams)
    
    # Test forward pass
    batch_size = 32
    T = 10  # sequence length
    test_input = torch.randn(batch_size, T, 3)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input, training=False)
        print(f"\nTest input shape: {test_input.shape}")
        print(f"Test output shape: {output.shape}")
        print(f"Output sample: {output[0]}")