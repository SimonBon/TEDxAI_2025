# src/losses.py
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmselfsup.registry import MODELS  # same registry your other modules use

@MODELS.register_module()
class MSELoss(nn.Module):
    """Mean Squared Error with OpenMMLab-style signature."""
    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> torch.Tensor:
        """pred, target: same shape (B, 1) or (B,) floats."""
        

        reduction = reduction_override or self.reduction
        loss = F.mse_loss(pred.squeeze(), target.squeeze(), reduction='none')
        
        if weight is not None:
            loss = loss * weight
        if reduction == 'mean':
            loss = loss.mean() if avg_factor is None else loss.sum() / max(avg_factor, 1e-12)
        elif reduction == 'sum':
            loss = loss.sum()
            
        return loss * self.loss_weight
    
@MODELS.register_module()
class EuclideanLoss(nn.Module):
    """Mean Squared Error with OpenMMLab-style signature."""
    def __init__(self, reduction: str = 'mean', loss_weight: float = 1.0):
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[str] = None,
                **kwargs) -> torch.Tensor:
        """pred, target: same shape (B, 1) or (B,) floats."""
        

        reduction = reduction_override or self.reduction
        loss = (pred.squeeze() - target.squeeze()).abs()
        
        if weight is not None:
            loss = loss * weight
        if reduction == 'mean':
            loss = loss.mean() if avg_factor is None else loss.sum() / max(avg_factor, 1e-12)
        elif reduction == 'sum':
            loss = loss.sum()
            
        return loss * self.loss_weight