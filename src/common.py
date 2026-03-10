# Used to store common utilities and functions 
# that can be shared across different modules in the project.
# TODO: Consider to delete this file if it becomes too trivial or unnecessary in the future.
import torch

def get_device():
    """
    Automatically detect the available computing device.

    Priority:
    1. NVIDIA GPU (CUDA)
    2. Apple Silicon GPU (MPS)
    3. CPU

    Returns:
        torch.device
    
    How to use:
    device = get_device()
    model.to(device)
    tensor = tensor.to(device)
    """
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    
    return torch.device("cpu")