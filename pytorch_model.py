"""
PyTorch Neural Network Model for Chess Position Evaluation
Architecture: MLP with [512, 264, 64, 32] hidden layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class ChessEvalModel(nn.Module):
    """
    Multi-layer perceptron for chess position evaluation.
    Architecture: 776 input → [512, 264, 64, 32] → 1 output (centipawns)
    """
    
    def __init__(self, input_dim=776):
        super(ChessEvalModel, self).__init__()
        
        # Hidden layers matching scikit-learn architecture
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 264)
        self.fc3 = nn.Linear(264, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        
        # Batch normalization for better training
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(264)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Input: (batch_size, 776)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        
        x = self.fc5(x)
        return x

class ChessEvalModelInference:
    """
    Wrapper for loading and using the PyTorch model
    Handles normalization and device management
    """
    
    def __init__(self, model_path="chess_eval_pytorch.pt"):
        print("🔧 Initializing PyTorch model wrapper...")
        try:
            import torch
            print(f"   PyTorch version: {torch.__version__}")
            print(f"   CUDA available: {torch.cuda.is_available()}")
        except ImportError:
            raise ImportError("PyTorch is not installed. Install with: pip install torch")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.loaded = False
        
        print(f"   Device: {self.device}")
        print(f"   Model path: {model_path}")
        
        # Load the model if it exists
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"⚠️  WARNING: PyTorch model file {model_path} not found.")
            print("   Train the model with: python train_pytorch.py")
            print("   Falling back to scikit-learn model...")
            self.model = None
            self.loaded = False
    
    def load_model(self):
        """Load the PyTorch model from file"""
        try:
            import torch
            self.model = ChessEvalModel()
            # Load with weights_only=False to allow numpy arrays in checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.loaded = True
            
            # Load scaler if available
            if 'scaler_mean' in checkpoint and 'scaler_std' in checkpoint:
                self.scaler = {
                    'mean': torch.tensor(checkpoint['scaler_mean']).to(self.device),
                    'std': torch.tensor(checkpoint['scaler_std']).to(self.device)
                }
            
            print(f"✓ PyTorch model loaded successfully from {self.model_path}")
            print(f"   Model architecture: [776 → 512 → 264 → 64 → 32 → 1]")
            print(f"   Batch norm: Enabled")
            print(f"   Dropout: 0.2")
        except Exception as e:
            print(f"ERROR loading PyTorch model: {e}")
            self.model = None
            self.loaded = False
    
    def eval_pos(self, features):
        """
        Evaluate a chess position
        
        Args:
            features: numpy array of shape (776,) or (1, 776)
        
        Returns:
            centipawns: float evaluation in centipawns
        """
        if self.model is None or not self.loaded:
            raise RuntimeError("PyTorch model not loaded. Train with: python train_pytorch.py")
        
        try:
            import torch
            
            # Convert to torch tensor and ensure float32
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()  # Ensure float32
            
            # Add batch dimension if needed
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            # Normalize if scaler available
            if self.scaler is not None:
                features = (features - self.scaler['mean']) / self.scaler['std']
            
            # Move to device and ensure float32 dtype
            features = features.to(self.device).float()
            
            # Predict
            with torch.no_grad():
                output = self.model(features)
                centipawns = output.item()
            
            return centipawns
        except Exception as e:
            print(f"Error evaluating with PyTorch model: {e}")
            raise

def load_pytorch_model():
    """Lazy loader for PyTorch model"""
    return ChessEvalModelInference()

def eval_cp(features):
    """
    Evaluate centipawns for given features (compatible interface with scikit-learn)
    
    Args:
        features: numpy array of shape (776,)
    
    Returns:
        centipawns: float
    """
    # Initialize model if not already loaded
    if not hasattr(eval_cp, '_model'):
        try:
            eval_cp._model = load_pytorch_model()
        except Exception as e:
            print(f"Could not load PyTorch model: {e}")
            print("Falling back to scikit-learn model...")
            # Fall back to scikit-learn
            from eval_model import eval_cp as sklearn_eval
            return sklearn_eval(features)
    
    try:
        return eval_cp._model.eval_pos(features)
    except RuntimeError:
        # If PyTorch model failed, fall back to scikit-learn
        print("PyTorch model not available, using scikit-learn")
        from eval_model import eval_cp as sklearn_eval
        return sklearn_eval(features)

