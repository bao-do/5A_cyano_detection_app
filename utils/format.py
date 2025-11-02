import torch
import numpy as np

def make_json_safe(obj):
    """Recursively convert numpy / torch types to Python native types for JSON."""
    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # torch scalar or Tensor
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.cpu().detach().numpy().tolist()
    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # dict/list recursion
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return obj