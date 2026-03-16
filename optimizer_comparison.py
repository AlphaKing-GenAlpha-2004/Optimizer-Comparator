import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix

# 1. Dataset Handling
def load_mnist_subset(train_path, test_path, train_samples=10000, test_samples=2000):
    # Load CSVs (assuming first column is label, rest are pixels)
    train_df = pd.read_csv(train_path, nrows=train_samples)
    test_df = pd.read_csv(test_path, nrows=test_samples)
    
    y_train_raw = train_df.iloc[:, 0].values
    X_train = train_df.iloc[:, 1:].values / 255.0
    
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values / 255.0
    
    # One-hot encoding for training labels
    num_classes = 10
    y_train = np.eye(num_classes)[y_train_raw]
    
    return X_train, y_train, X_test, y_test

# 2. Neural Network Architecture
class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        # He Initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        # Xavier Initialization for Softmax
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(dz2, self.W2.T) * self.relu_deriv(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2

# 3. Optimizers
class Optimizer:
    def __init__(self, nn, lr=0.01):
        self.nn = nn
        self.lr = lr
        
class SGD(Optimizer):
    def update(self, dW1, db1, dW2, db2):
        self.nn.W1 -= self.lr * dW1
        self.nn.b1 -= self.lr * db1
        self.nn.W2 -= self.lr * dW2
        self.nn.b2 -= self.lr * db2
        return [dW1, db1, dW2, db2], [self.lr*dW1, self.lr*db1, self.lr*dW2, self.lr*db2]

class Adagrad(Optimizer):
    def __init__(self, nn, lr=0.01):
        super().__init__(nn, lr)
        self.vW1, self.vb1 = np.zeros_like(nn.W1), np.zeros_like(nn.b1)
        self.vW2, self.vb2 = np.zeros_like(nn.W2), np.zeros_like(nn.b2)
        self.eps = 1e-8
        
    def update(self, dW1, db1, dW2, db2):
        self.vW1 += dW1**2
        self.vb1 += db1**2
        self.vW2 += dW2**2
        self.vb2 += db2**2
        
        up_w1 = (self.lr / np.sqrt(self.vW1 + self.eps)) * dW1
        up_b1 = (self.lr / np.sqrt(self.vb1 + self.eps)) * db1
        up_w2 = (self.lr / np.sqrt(self.vW2 + self.eps)) * dW2
        up_b2 = (self.lr / np.sqrt(self.vb2 + self.eps)) * db2
        
        self.nn.W1 -= up_w1
        self.nn.b1 -= up_b1
        self.nn.W2 -= up_w2
        self.nn.b2 -= up_b2
        return [dW1, db1, dW2, db2], [up_w1, up_b1, up_w2, up_b2]

class RMSProp(Optimizer):
    def __init__(self, nn, lr=0.001, beta=0.9):
        super().__init__(nn, lr)
        self.vW1, self.vb1 = np.zeros_like(nn.W1), np.zeros_like(nn.b1)
        self.vW2, self.vb2 = np.zeros_like(nn.W2), np.zeros_like(nn.b2)
        self.beta = beta
        self.eps = 1e-8
        
    def update(self, dW1, db1, dW2, db2):
        self.vW1 = self.beta * self.vW1 + (1 - self.beta) * dW1**2
        self.vb1 = self.beta * self.vb1 + (1 - self.beta) * db1**2
        self.vW2 = self.beta * self.vW2 + (1 - self.beta) * dW2**2
        self.vb2 = self.beta * self.vb2 + (1 - self.beta) * db2**2
        
        up_w1 = (self.lr / np.sqrt(self.vW1 + self.eps)) * dW1
        up_b1 = (self.lr / np.sqrt(self.vb1 + self.eps)) * db1
        up_w2 = (self.lr / np.sqrt(self.vW2 + self.eps)) * dW2
        up_b2 = (self.lr / np.sqrt(self.vb2 + self.eps)) * db2
        
        self.nn.W1 -= up_w1
        self.nn.b1 -= up_b1
        self.nn.W2 -= up_w2
        self.nn.b2 -= up_b2
        return [dW1, db1, dW2, db2], [up_w1, up_b1, up_w2, up_b2]

class Adam(Optimizer):
    def __init__(self, nn, lr=0.001, b1=0.9, b2=0.999):
        super().__init__(nn, lr)
        self.mW1, self.mb1 = np.zeros_like(nn.W1), np.zeros_like(nn.b1)
        self.mW2, self.mb2 = np.zeros_like(nn.W2), np.zeros_like(nn.b2)
        self.vW1, self.vb1 = np.zeros_like(nn.W1), np.zeros_like(nn.b1)
        self.vW2, self.vb2 = np.zeros_like(nn.W2), np.zeros_like(nn.b2)
        self.b1, self.b2 = b1, b2
        self.eps = 1e-8
        self.t = 0
        
    def update(self, dW1, db1, dW2, db2):
        self.t += 1
        self.mW1 = self.b1 * self.mW1 + (1 - self.b1) * dW1
        self.mb1 = self.b1 * self.mb1 + (1 - self.b1) * db1
        self.mW2 = self.b1 * self.mW2 + (1 - self.b1) * dW2
        self.mb2 = self.b1 * self.mb2 + (1 - self.b1) * db2
        
        self.vW1 = self.b2 * self.vW1 + (1 - self.b2) * dW1**2
        self.vb1 = self.b2 * self.vb1 + (1 - self.b2) * db1**2
        self.vW2 = self.b2 * self.vW2 + (1 - self.b2) * dW2**2
        self.vb2 = self.b2 * self.vb2 + (1 - self.b2) * db2
        
        m_w1_hat = self.mW1 / (1 - self.b1**self.t)
        m_b1_hat = self.mb1 / (1 - self.b1**self.t)
        m_w2_hat = self.mW2 / (1 - self.b1**self.t)
        m_b2_hat = self.mb2 / (1 - self.b1**self.t)
        
        v_w1_hat = self.vW1 / (1 - self.b2**self.t)
        v_b1_hat = self.vb1 / (1 - self.b2**self.t)
        v_w2_hat = self.vW2 / (1 - self.b2**self.t)
        v_b2_hat = self.vb2 / (1 - self.b2**self.t)
        
        up_w1 = (self.lr / (np.sqrt(v_w1_hat) + self.eps)) * m_w1_hat
        up_b1 = (self.lr / (np.sqrt(v_b1_hat) + self.eps)) * m_b1_hat
        up_w2 = (self.lr / (np.sqrt(v_w2_hat) + self.eps)) * m_w2_hat
        up_b2 = (self.lr / (np.sqrt(v_b2_hat) + self.eps)) * m_b2_hat
        
        self.nn.W1 -= up_w1
        self.nn.b1 -= up_b1
        self.nn.W2 -= up_w2
        self.nn.b2 -= up_b2
        return [dW1, db1, dW2, db2], [up_w1, up_b1, up_w2, up_b2]

# 4. Performance Metrics
def compute_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion Matrix for Precision, Recall, F1
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(10):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    # Log Loss
    m = y_true.shape[0]
    log_loss = -np.mean(np.log(y_pred_probs[range(m), y_true] + 1e-15))
    
    return accuracy, macro_precision, macro_recall, macro_f1, log_loss

# 5. Experiment Runner
def run_experiment(X_train, y_train, X_test, y_test, opt_class, name, epochs=10, batch_size=64):
    nn = NeuralNetwork()
    optimizer = opt_class(nn)
    
    history = {
        'loss': [],
        'accuracy': [],
        'grad_norm': [],
        'update_ratio': []
    }
    
    train_start = time.time()
    
    m = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        epoch_acc = 0
        epoch_grad_norm = 0
        epoch_update_ratio = 0
        
        num_batches = m // batch_size
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            output = nn.forward(X_batch)
            grads = nn.backward(X_batch, y_batch, output)
            
            # Update and track norms
            raw_grads, updates = optimizer.update(*grads)
            
            # Gradient Norm
            grad_norm = np.sqrt(sum(np.sum(g**2) for g in raw_grads))
            epoch_grad_norm += grad_norm
            
            # Update Ratio
            update_norm = np.sqrt(sum(np.sum(u**2) for u in updates))
            param_norm = np.sqrt(np.sum(nn.W1**2) + np.sum(nn.b1**2) + np.sum(nn.W2**2) + np.sum(nn.b2**2))
            epoch_update_ratio += update_norm / (param_norm + 1e-15)
            
            # Batch Loss
            batch_loss = -np.mean(np.sum(y_batch * np.log(output + 1e-15), axis=1))
            epoch_loss += batch_loss
            
            # Batch Acc
            batch_acc = np.mean(np.argmax(y_batch, axis=1) == np.argmax(output, axis=1))
            epoch_acc += batch_acc
            
        history['loss'].append(epoch_loss / num_batches)
        history['accuracy'].append(epoch_acc / num_batches)
        history['grad_norm'].append(epoch_grad_norm / num_batches)
        history['update_ratio'].append(epoch_update_ratio / num_batches)
        
        print(f"{name} - Epoch {epoch+1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Acc: {history['accuracy'][-1]:.4f}")
        
    train_time = time.time() - train_start
    
    # Testing
    test_start = time.time()
    test_output = nn.forward(X_test)
    test_time = time.time() - test_start
    
    acc, prec, rec, f1, loss = compute_metrics(y_test, test_output)
    
    # Convergence Rate
    conv_rate = (history['loss'][0] - history['loss'][-1]) / epochs
    
    # Loss Variance
    loss_var = np.var(history['loss'])
    
    results = {
        'optimizer': name,
        'accuracy': acc,
        'f1_score': f1,
        'precision': prec,
        'recall': rec,
        'log_loss': loss,
        'convergence_rate': conv_rate,
        'loss_variance': loss_var,
        'training_time': train_time,
        'testing_time': test_time,
        'history': history
    }
    
    return results

# Main Execution
if __name__ == "__main__":
    # Note: You need mnist_train.csv and mnist_test.csv in the same directory
    # or update paths accordingly.
    try:
        X_train, y_train, X_test, y_test = load_mnist_subset('mnist_train.csv', 'mnist_test.csv')
    except FileNotFoundError:
        print("MNIST CSV files not found. Please ensure mnist_train.csv and mnist_test.csv are available.")
        # Mocking data for demonstration if files are missing
        X_train = np.random.rand(10000, 784)
        y_train = np.eye(10)[np.random.randint(0, 10, 10000)]
        X_test = np.random.rand(2000, 784)
        y_test = np.random.randint(0, 10, 2000)
        print("Using random mock data for demonstration.")

    opts = [
        (SGD, 'SGD', 'black'),
        (Adagrad, 'Adagrad', 'orange'),
        (RMSProp, 'RMSProp', 'green'),
        (Adam, 'Adam', 'blue')
    ]
    
    all_results = []
    for opt_class, name, color in opts:
        res = run_experiment(X_train, y_train, X_test, y_test, opt_class, name)
        res['color'] = color
        all_results.append(res)
        
    # 6. Graphs Generation
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss vs Epoch
    for res in all_results:
        axs[0, 0].plot(res['history']['loss'], label=res['optimizer'], color=res['color'], linewidth=2)
    axs[0, 0].set_title('Loss vs Epoch')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Accuracy vs Epoch
    for res in all_results:
        axs[0, 1].plot(res['history']['accuracy'], label=res['optimizer'], color=res['color'], linewidth=2)
    axs[0, 1].set_title('Accuracy vs Epoch')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)
    
    # Gradient Norm vs Epoch
    for res in all_results:
        axs[1, 0].plot(res['history']['grad_norm'], label=res['optimizer'], color=res['color'], linewidth=2)
    axs[1, 0].set_title('Gradient Norm vs Epoch')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Norm')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)
    
    # Update Ratio vs Epoch
    for res in all_results:
        axs[1, 1].plot(res['history']['update_ratio'], label=res['optimizer'], color=res['color'], linewidth=2)
    axs[1, 1].set_title('Update Ratio vs Epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Ratio')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison_plots.png')
    plt.show()
    
    # 8. Output Table
    df_results = pd.DataFrame(all_results).drop(columns=['history', 'color'])
    print("\nOptimizer Comparison Table:")
    print(df_results.to_string(index=False))
    df_results.to_csv('optimizer_comparison_results.csv', index=False)
