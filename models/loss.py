###############################################################
# Loss function and Training utilities #
###############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss, gaussian_nll_loss

class MSELoss(torch.nn.Module):
    def __init__(self, target_property: str = "", forces: bool = False):
        super().__init__()
        self.target_property = target_property or "energy"
        self.forces = forces

    def forward(self, preds: dict, batch) -> torch.Tensor:
        # Basic MSE on the target property
        targets = batch.energy if self.target_property == "energy" else batch.targets
        loss = mse_loss(preds[self.target_property], targets)
        
        # If forces are included, add them to the loss
        if self.forces:
            loss_forces = mse_loss(preds["forces"], batch.forces)
            # Combine equally for simplicity
            loss = 0.5 * loss + 0.5 * loss_forces

        return loss

class GaussianNLLLoss(torch.nn.Module):
    def __init__(self, target_property: str = "", variance: float = 1.0, forces: bool = False):
        super().__init__()
        self.target_property = target_property or "energy"
        self.variance = variance
        self.forces = forces

    def forward(self, preds: dict, batch) -> torch.Tensor:
        # Basic Gaussian NLL on the target property
        targets = batch.energy if self.target_property == "energy" else batch.targets
        var = torch.full_like(preds[self.target_property], self.variance)
        loss = gaussian_nll_loss(preds[self.target_property], targets, var=var, reduction='mean')

        # If forces are included, add them
        if self.forces:
            var_forces = torch.full_like(preds["forces"], self.variance)
            loss_forces = gaussian_nll_loss(preds["forces"], batch.forces, var=var_forces, reduction='mean')
            # Combine equally for simplicity
            loss = 0.5 * loss + 0.5 * loss_forces

        return loss
