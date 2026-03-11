"""

Implementation of LoRA (Low-Rank Adaptation).

LoRA modifies a pretrained linear layer by introducing
two low-rank matrices while keeping the original weights frozen.

Original transformation:
    y = Wx

LoRA transformation:
    y = Wx + (alpha / r) * BAx

Where:
    A ∈ R^(r × d_in)
    B ∈ R^(d_out × r)

r is the rank of the low-rank adaptation.
"""

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA wrapper for a linear layer.

    This module keeps the original Linear layer frozen
    and adds a trainable low-rank update.
    """

    def __init__(self, linear_layer, r=8, alpha=16):

        super().__init__()

        self.linear = linear_layer

        # Freeze pretrained weights
        for param in self.linear.parameters():
            param.requires_grad = False

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        self.r = r
        self.alpha = alpha
        # Ensure the low-rank matrices have the same dtype as the original weights
        weight_dtype = linear_layer.weight.dtype
        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(r, in_features, dtype=weight_dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r, dtype=weight_dtype))

        # Scaling factor used in LoRA
        self.scaling = alpha / r

    def forward(self, x):
        """
        Forward computation.

        output = original_linear(x) + LoRA_update
        """

        original_output = self.linear(x)

        lora_update = (x @ self.A.T) @ self.B.T

        return original_output + self.scaling * lora_update


def inject_lora(model, target_modules=("q_proj", "v_proj"), r=8, alpha=16):
    """
    Traverses a pretrained transformer model and replaces specified 
    linear layers with LoRALinear modules.
    
    Args:
        model: The pretrained transformer model instance.
        target_modules: A tuple of strings identifying the layers to target 
                        (e.g., query and value projections in attention).
        r: The rank for the LoRA matrices.
        alpha: The scaling factor for the LoRA update.
    """
    
    # 1. Iterate through all modules in the model's tree structure.
    # 'name' is the full path (e.g., "model.layers.0.self_attn.q_proj")
    # 'module' is the actual PyTorch layer object (e.g., an nn.Linear instance)
    for name, module in model.named_modules():
        
        # 2. Filter: Check if the current module is a Linear layer AND 
        # if its name contains any of our target keywords.
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            
            # Initialize a pointer to the root of the model
            parent = model
            
            # Split the full path into a list of node names
            # e.g., ['model', 'layers', '0', 'self_attn', 'q_proj']
            parts = name.split(".")
            
            # 3. Navigate down the tree to find the direct parent of the target layer.
            # We iterate up to the second-to-last element (parts[:-1])
            for part in parts[:-1]:
                # Dynamically get the child object by its string name
                parent = getattr(parent, part)
            
            # parts[-1] is the name of the layer itself (e.g., "q_proj")
            # Get the actual original layer object before we replace it
            original_layer = getattr(parent, parts[-1])
            
            # 4. Instantiate the new LoRA layer, wrapping the original frozen layer
            lora_layer = LoRALinear(original_layer, r=r, alpha=alpha)
            
            # 5. Dynamic Replacement: Overwrite the parent's attribute with our new layer
            setattr(parent, parts[-1], lora_layer)
            
    return model