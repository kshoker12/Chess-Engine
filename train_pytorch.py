"""
Train PyTorch neural network for chess position evaluation
Uses the same dataset as the scikit-learn model for fair comparison
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import os
from pytorch_model import ChessEvalModel

class ChessDataset(Dataset):
    """PyTorch Dataset for chess position features and evaluations"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data(prefix: str):
    """Load training data"""
    X = np.load(f"{prefix}_X.npy")
    y = np.load(f"{prefix}_y.npy").astype(np.float32)
    return X, y

def train_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=1024, 
                learning_rate=6e-3, device='cpu'):
    """Train the PyTorch model"""
    
    # Create datasets
    train_dataset = ChessDataset(X_train, y_train)
    val_dataset = ChessDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ChessEvalModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }, 'chess_eval_pytorch.pt')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.2f}, "
                  f"Val Loss: {val_loss:.2f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training complete. Best validation loss: {best_val_loss:.2f}")
    return model

def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate model performance"""
    model.eval()
    
    test_dataset = ChessDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()
            predictions.extend(outputs)
            targets.extend(y_batch.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - (np.sum((targets - predictions) ** 2) / 
              np.sum((targets - np.mean(targets)) ** 2))
    
    print(f"\nTest Metrics:")
    print(f"  MAE: {mae:.2f} cp")
    print(f"  RMSE: {rmse:.2f} cp")
    print(f"  R²: {r2:.4f}")
    
    return mae, rmse, r2

def main():
    parser = argparse.ArgumentParser(description='Train PyTorch chess evaluation model')
    parser.add_argument('--prefix', default='dataset', help='Dataset prefix')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=6e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    X, y = load_data(args.prefix)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train/val/test split (80/10/10)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1/0.9, random_state=args.seed
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize features (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Normalize targets
    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train model
    model = train_model(X_train, y_train, X_val, y_val, 
                       epochs=args.epochs, 
                       batch_size=args.batch_size,
                       learning_rate=args.lr,
                       device=device)
    
    # Load best model
    checkpoint = torch.load('chess_eval_pytorch.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Save scaler info with model
    checkpoint = torch.load('chess_eval_pytorch.pt')
    checkpoint['scaler_mean'] = scaler.mean_
    checkpoint['scaler_std'] = scaler.scale_
    checkpoint['target_scaler_mean'] = target_scaler.mean_[0]
    checkpoint['target_scaler_std'] = target_scaler.scale_[0]
    torch.save(checkpoint, 'chess_eval_pytorch.pt')
    
    # Evaluate
    evaluate_model(model, X_test, y_test, device=device)
    
    print("\n✓ Model saved to chess_eval_pytorch.pt")

if __name__ == "__main__":
    main()

