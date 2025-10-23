import joblib
import numpy as np

# Global model cache - loaded only when first needed
_MODEL = None

def _load_model():
    """Lazy load the model only when first needed"""
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = joblib.load('sk_eval.joblib')
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    return _MODEL

def eval_cp(features: np.ndarray) -> float:
    """
    features: np.float32 array of shape (776,)
    returns: centipawn float (clamped to [-1000, 1000])
    """
    model = _load_model()  # Lazy load
    cp = float(model.predict(features.reshape(1, -1))[0])
    if cp > 1000: cp = 1000.0
    if cp < -1000: cp = -1000.0
    return cp