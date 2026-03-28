import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Any, Type, Optional

# ==========================================
# 1. Model Architectures
# ==========================================

class BaseModel(nn.Module):
    """Base class to provide common functionality for all models."""
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def get_grad_norms(self) -> Dict[str, float]:
        """Calculates the norm of gradients for each layer."""
        norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.norm().item()
        return norms

    def get_param_norm(self) -> float:
        """Calculates the total norm of all parameters."""
        total_norm = 0.0
        for param in self.parameters():
            total_norm += param.data.norm()**2
        return torch.sqrt(total_norm).item()

class MLP(BaseModel):
    """Multi-Layer Perceptron for tabular data or flattened images."""
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, dropout_rate: float = 0.2):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class SimpleCNN(BaseModel):
    """Simple CNN for MNIST (1 channel, 28x28)."""
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

class DeepCNN(BaseModel):
    """Deeper CNN for CIFAR-10 (3 channels, 32x32)."""
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super(DeepCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ==========================================
# 2. Model Selector
# ==========================================

def get_model(dataset_name: str, model_type: str = "auto", **kwargs) -> nn.Module:
    """
    Dynamically selects the appropriate model based on dataset and user preference.
    """
    dataset_name = dataset_name.lower()
    
    # Auto-selection logic
    if model_type == "auto":
        if dataset_name in ["mnist", "fashion_mnist"]:
            model_type = "cnn"
        elif dataset_name == "cifar10":
            model_type = "deep_cnn"
        else:
            model_type = "mlp"

    # Instantiate selected model
    if model_type == "mlp":
        input_size = kwargs.get("input_size", 784)
        hidden_size = kwargs.get("hidden_size", 256)
        num_classes = kwargs.get("num_classes", 10)
        return MLP(input_size, hidden_size, num_classes)
    
    elif model_type == "cnn":
        num_classes = kwargs.get("num_classes", 10)
        return SimpleCNN(num_classes)
    
    elif model_type == "deep_cnn":
        num_classes = kwargs.get("num_classes", 10)
        return DeepCNN(num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# ==========================================
# 3. Metric Tracker
# ==========================================

class MetricTracker:
    """Tracks and computes various training metrics."""
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_acc': [],
            'grad_norms': [], # Total norm
            'layer_grad_norms': [], # Dict of layer norms
            'update_efficiency': [],
            'stability': [] # Variance of loss over window
        }

    def update(self, train_loss: float, train_acc: float, test_acc: float, 
               grad_norm: float, layer_norms: Dict[str, float], update_ratio: float):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['test_acc'].append(test_acc)
        self.history['grad_norms'].append(grad_norm)
        self.history['layer_grad_norms'].append(layer_norms)
        self.history['update_efficiency'].append(update_ratio)
        
        # Stability: variance of last 5 losses
        if len(self.history['train_loss']) >= 5:
            stability = np.var(self.history['train_loss'][-5:])
        else:
            stability = 0.0
        self.history['stability'].append(stability)

# ==========================================
# 4. Training Engine
# ==========================================

class Trainer:
    def __init__(self, model: nn.Module, optimizer_type: str, lr: float, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer_type = optimizer_type
        self.lr = lr
        
        # Optimizer selection
        if optimizer_type.upper() == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type.upper() == "ADAM":
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type.upper() == "RMSPROP":
            self.optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer_type.upper() == "ADAGRAD":
            self.optimizer = optim.Adagrad(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        self.criterion = nn.CrossEntropyLoss()
        self.tracker = MetricTracker()

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        epoch_grad_norms = []
        epoch_layer_norms = []
        epoch_update_ratios = []

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Metrics before update
            grad_norms = self.model.get_grad_norms()
            total_grad_norm = np.sqrt(sum(v**2 for v in grad_norms.values()))
            
            # Store parameter norm before update
            param_norm_before = self.model.get_param_norm()
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate update efficiency
            param_norm_after = self.model.get_param_norm()
            update_norm = abs(param_norm_after - param_norm_before)
            update_ratio = update_norm / (param_norm_before + 1e-15)
            
            # Accumulate
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            epoch_grad_norms.append(total_grad_norm)
            epoch_layer_norms.append(grad_norms)
            epoch_update_ratios.append(update_ratio)

        avg_loss = total_loss / len(dataloader)
        avg_acc = correct / total
        avg_grad_norm = np.mean(epoch_grad_norms)
        avg_update_ratio = np.mean(epoch_update_ratios)
        
        # Average layer norms
        avg_layer_norms = {}
        if epoch_layer_norms:
            for key in epoch_layer_norms[0].keys():
                avg_layer_norms[key] = np.mean([step[key] for step in epoch_layer_norms])

        return avg_loss, avg_acc, avg_grad_norm, avg_layer_norms, avg_update_ratio

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        return correct / total

    def run(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int):
        print(f"Starting training with {self.optimizer_type}...")
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss, train_acc, grad_norm, layer_norms, update_ratio = self.train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)
            
            self.tracker.update(train_loss, train_acc, test_acc, grad_norm, layer_norms, update_ratio)
            
            duration = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Time: {duration:.2f}s")
        
        return self.tracker.history

# ==========================================
# 5. Data Loading
# ==========================================

def get_dataloaders(dataset_name: str, batch_size: int = 64, subset_size: Optional[int] = None):
    """Fetches MNIST or CIFAR-10 dataloaders."""
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10('./data', train=False, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in this loader.")

    if subset_size:
        train_set = Subset(train_set, range(min(subset_size, len(train_set))))
        test_set = Subset(test_set, range(min(subset_size // 5, len(test_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# ==========================================
# 6. Main Execution & Visualization
# ==========================================

def plot_results(all_results: Dict[str, Any], metric_name: str, title: str):
    plt.figure(figsize=(10, 6))
    for opt_name, history in all_results.items():
        plt.plot(history[metric_name], label=opt_name, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace('_', ' ').capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{metric_name}_comparison.png")
    plt.close()

def main():
    # Configuration
    DATASET = "mnist" # or "cifar10"
    EPOCHS = 5
    BATCH_SIZE = 128
    SUBSET_SIZE = 5000 # Use smaller subset for quick demonstration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running on {DEVICE}")
    
    # Load Data
    train_loader, test_loader = get_dataloaders(DATASET, BATCH_SIZE, SUBSET_SIZE)
    
    optimizers = ["SGD", "Adam", "RMSProp", "Adagrad"]
    results = {}
    
    for opt in optimizers:
        # Get model (auto-selects CNN for MNIST)
        model = get_model(DATASET, model_type="auto")
        
        # Initialize Trainer
        trainer = Trainer(model, opt, lr=0.001, device=DEVICE)
        
        # Run training
        history = trainer.run(train_loader, test_loader, EPOCHS)
        results[opt] = history

    # Visualization
    plot_results(results, 'train_loss', f'Loss Convergence on {DATASET.upper()}')
    plot_results(results, 'test_acc', f'Test Accuracy on {DATASET.upper()}')
    plot_results(results, 'grad_norms', f'Gradient Norms on {DATASET.upper()}')
    plot_results(results, 'update_efficiency', f'Update Efficiency on {DATASET.upper()}')
    plot_results(results, 'stability', f'Training Stability (Loss Variance) on {DATASET.upper()}')

    # Layer-wise gradient flow example for the last optimizer
    last_opt = optimizers[-1]
    layer_history = results[last_opt]['layer_grad_norms']
    if layer_history:
        plt.figure(figsize=(12, 6))
        layers = list(layer_history[0].keys())
        for layer in layers:
            # Filter for weight layers to keep plot clean
            if 'weight' in layer:
                plt.plot([h[layer] for h in layer_history], label=layer)
        plt.title(f"Layer-wise Gradient Flow ({last_opt})")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("layer_gradient_flow.png")
        plt.close()

    print("\nExperiments complete. Plots saved as PNG files.")

if __name__ == "__main__":
    main()
