import torch
import torch.nn as nn
from typing import List, Tuple

def static_loss(model: nn.Module, f: torch.Tensor, f_bc: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss between the model's predictions and the true labels.
    
    Args:
        model: The neural network model.
        f: Input tensor for the first branch.
        f_bc: Input tensor for the second branch.
        x: Input tensor for the trunk.
        y: True labels.
    
    Returns:
        torch.Tensor: The computed loss.
    """
    y_out = model.forward(f, f_bc, x)
    loss = ((y_out - y)**2).mean()
    return loss

def static_forward(model: nn.Module, f: torch.Tensor, f_bc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Perform the forward pass through the model.
    
    Args:
        model: The neural network model.
        f: Input tensor for the first branch.
        f_bc: Input tensor for the second branch.
        x: Input tensor for the trunk.
    
    Returns:
        torch.Tensor: The model's predictions.
    """
    y_br1 = model._branch1(f)
    y_br2 = model._branch2(f_bc)
    y_br = y_br1 * y_br2

    y_tr = model._trunk(x)
    y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
    return y_out

def static_init(model: nn.Module, branch1_dim: List[int], branch2_dim: List[int], trunk_dim: List[int]) -> None:
    """
    Initialize the model's parameters and architecture.
    
    Args:
        model: The neural network model.
        branch1_dim: Dimensions for the first branch.
        branch2_dim: Dimensions for the second branch.
        trunk_dim: Dimensions for the trunk.
    """
    model.z_dim = trunk_dim[-1]

    # Build branch net for branch1
    modules = []
    in_channels = branch1_dim[0]
    for h_dim in branch1_dim[1:]:
        modules.append(nn.Sequential(
            nn.Linear(in_channels, h_dim),
            nn.Tanh()
        ))
        in_channels = h_dim
    model._branch1 = nn.Sequential(*modules)

    # Build branch net for branch2
    modules = []
    in_channels = branch2_dim[0]
    for h_dim in branch2_dim[1:]:
        modules.append(nn.Sequential(
            nn.Linear(in_channels, h_dim),
            nn.Tanh()
        ))
        in_channels = h_dim
    model._branch2 = nn.Sequential(*modules)

    # Build trunk net
    modules = []
    in_channels = trunk_dim[0]
    for h_dim in trunk_dim[1:]:
        modules.append(nn.Sequential(
            nn.Linear(in_channels, h_dim),
            nn.Tanh()
        ))
        in_channels = h_dim
    model._trunk = nn.Sequential(*modules)

