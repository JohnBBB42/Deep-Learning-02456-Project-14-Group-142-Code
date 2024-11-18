"""Loss functions."""

import torch
from torch.nn.functional import l1_loss, mse_loss

from atomgnn.data.data import Batch


class MSELoss(torch.nn.Module):
    """Mean squared error (MSE) loss function."""

    def __init__(
        self,
        target_property: str = "",
        forces: bool = False,
        forces_property: str = "forces",
        forces_weight: float = 0.5,
        stress: bool = False,
        stress_weight: float = 0.1,
        nodewise: bool = False,
        **kwargs  # Ignore additional arguments
    ) -> None:
        """Initialize the loss function.

        Args:
            target_property: The target property in the dataset (use energy if falsy).
            forces: Include forces in the loss.
            forces_property: The forces property in the predictions.
            forces_weight: Trade-off between target and forces in the loss function (0.0 to 1.0).
            stress: Include stress in the loss.
            stress_weight: Weight of stress in the loss.
            nodewise: Use nodewise loss instead of global loss.
        """
        super().__init__()
        assert forces_weight >= 0.0 and forces_weight <= 1.0
        self.target_property = target_property or "energy"  # Use energy if falsy
        self.forces = forces
        self.forces_property = forces_property
        self.forces_weight = forces_weight
        self.stress = stress
        self.stress_weight = stress_weight
        self.nodewise = nodewise

    def _global_mse_loss(self, preds: dict, batch: Batch) -> torch.Tensor:
        targets = batch.energy if self.target_property == "energy" else batch.targets
        # The mean squared error loss of the predictions (averaged over batches).
        loss = mse_loss(preds[self.target_property], targets)
        if self.forces:
            # Compute the forces loss as the sum of squared errors over all components and nodes.
            # This puts equal weight on each node independently of the number of nodes.
            # Corresponds to the sum of the squared norms of the force errors.
            forces_loss = mse_loss(preds[self.forces_property], batch.forces, reduction="sum")
            # Make the forces loss independent of the batch size by dividing by num_data.
            forces_loss = forces_loss / batch.num_data
            loss = (1 - self.forces_weight) * loss + self.forces_weight * forces_loss
        return loss

    def _nodewise_mse_loss(self, preds: dict, batch: Batch) -> torch.Tensor:
        targets = batch.energy if self.target_property == "energy" else batch.targets
        num_nodes = batch.num_nodes.unsqueeze(1)
        # The nodewise mean squared error loss of the predictions
        loss = mse_loss(preds[self.target_property] / num_nodes, targets / num_nodes)
        if self.forces:
            # Compute the forces loss as the mean of squared errors over all components and nodes.
            forces_loss = mse_loss(preds[self.forces_property], batch.forces)
            loss = (1 - self.forces_weight) * loss + self.forces_weight * forces_loss
        return loss

    def forward(self, preds: dict, batch: Batch) -> torch.Tensor:
        """Compute the loss.

        Args:
            preds: Dictionary of predictions.
            batch: Batch of data.
        Returns:
            The loss.
        """
        if self.nodewise:
            loss = self._nodewise_mse_loss(preds, batch)
        else:
            loss = self._global_mse_loss(preds, batch)
        if self.stress:
            loss = loss + self.stress_weight * mse_loss(preds["stress"], batch.stress)
        return loss


class NequIPLoss(torch.nn.Module):
    """NequIP loss function.

    Resources:
        NequIP paper: https://www.nature.com/articles/s41467-022-29939-5
    """

    def __init__(
        self,
        target_property: str = "",
        target_weight: float = 1.0,
        forces: bool = False,
        forces_property: str = "forces",
        forces_weight: float = 1.0,
        **kwargs  # Ignore additional arguments
    ) -> None:
        """Initialize the loss function.

        Args:
            target_property: The target property in the dataset (use energy if falsy).
            target_weight: Weight of the target property in the loss.
            forces: Include forces in the loss.
            forces_property: The forces property in the predictions.
            forces_weight: Weight of the forces in the loss. A good defualt value is num_nodes**2.
        """
        super().__init__()
        assert target_weight >= 0.0 and forces_weight >= 0.0
        self.target_property = target_property or "energy"  # Use energy if falsy
        self.target_weight = target_weight
        self.forces = forces
        self.forces_property = forces_property
        self.forces_weight = forces_weight

    def forward(self, preds: dict, batch: Batch) -> torch.Tensor:
        """Compute the loss.

        Args:
            preds: Dictionary of predictions.
            batch: Batch of data.
        Returns:
            The loss.
        """
        targets = batch.energy if self.target_property == "energy" else batch.targets
        # The mean squared error loss of the predictions (averaged over batches).
        loss = self.target_weight * mse_loss(preds[self.target_property], targets)
        if self.forces:
            # Sum of squared force errors over all components and nodes divided by 3 * num_nodes.
            forces_loss = self.forces_weight * mse_loss(preds[self.forces_property], batch.forces)
            # Weighted sum of the target and forces losses.
            loss = loss + forces_loss
        return loss


class CHGNetLoss(torch.nn.Module):
    """CHGNet loss function.

    Weighted sum of losses of energy, forces, stress and magmoms.

    Resources:
        CHGNet paper: https://www.nature.com/articles/s42256-023-00716-3
    """

    def __init__(
        self,
        target_property: str = "",
        target_weight: float = 1.0,
        forces: bool = False,
        forces_property: str = "forces",
        forces_weight: float = 1.0,
        stress: bool = False,
        stress_property: str = "stress",
        stress_weight: float = 0.1,
        magmoms: bool = False,
        magmoms_property: str = "magmoms",
        magmoms_weight: float = 0.1,
        criterion: str = "mse",
        huber_delta: float = 0.1,
        nodewise: bool = False,
        **kwargs  # Ignore additional arguments
    ) -> None:
        """Initialize the loss function.

        Args:
            target_property: The target property in the dataset (use energy if falsy).
            target_weight: Weight of the target property in the loss.
            forces: Include forces in the loss.
            forces_property: The forces property in the predictions.
            forces_weight: Weight of the forces in the loss function.
            stress: Include stress in the loss.
            stress_property: The stress property in the predictions.
            stress_weight: Weight of stress in the loss.
            magmoms: Include magnetic moments in the loss.
            magmoms_weight: Weight of magnetic moments in the loss.
            criterion: Loss function criterion (mse, mae, huber).
            huber_delta: Delta value for the Huber loss.
            nodewise: Use nodewise loss instead of global loss.
        """
        super().__init__()
        assert target_weight >= 0.0 and forces_weight >= 0.0
        assert magmoms_weight >= 0.0 and stress_weight >= 0.0
        self.target_property = target_property or "energy"  # Use energy if falsy
        self.target_weight = target_weight
        self.forces = forces
        self.forces_property = forces_property
        self.forces_weight = forces_weight
        self.stress = stress
        self.stress_property = stress_property
        self.stress_weight = stress_weight
        self.magmoms = magmoms
        self.magmoms_property = magmoms_property
        self.magmoms_weight = magmoms_weight
        if criterion.lower() == "mse":
            self.criterion = mse_loss
        elif criterion.lower() == "mae":
            self.criterion = l1_loss
        elif criterion.lower() == "huber":
            self.criterion = torch.nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        self.nodewise = nodewise

    def forward(self, preds: dict, batch: Batch) -> torch.Tensor:
        """Compute the loss.

        Args:
            preds: Dictionary of predictions.
            batch: Batch of data.
        Returns:
            The loss.
        """
        targets = batch.energy if self.target_property == "energy" else batch.targets
        if self.nodewise:
            num_nodes = batch.num_nodes.unsqueeze(1)
            loss = self.criterion(preds[self.target_property] / num_nodes, targets / num_nodes)
        else:
            loss = self.criterion(preds[self.target_property], targets)
        loss = self.target_weight * loss
        if self.forces:
            forces_loss = self.criterion(preds[self.forces_property], batch.forces)
            loss = loss + self.forces_weight * forces_loss
        if self.stress:
            stress_loss = self.criterion(preds[self.stress_property], batch.stress)
            loss = loss + self.stress_weight * stress_loss
        if self.magmoms:
            # TODO: Exclude data without magmoms
            magmoms_loss = self.criterion(preds[self.magmoms_property], batch.magmoms)
            loss = loss + self.magmoms_weight * magmoms_loss
        return loss


class MACELoss(torch.nn.Module):
    """MACE loss function.

    Weighted sum of losses of energy, forces and stress.

    Note: The loss function described in the original MACE paper (https://arxiv.org/abs/2206.07697)
    corresponds to the NequIP loss function. This implementation is based on the MACE code and is
    closer to the MACE MP paper (https://arxiv.org/abs/2401.00096).

    Resources:
        MACE paper: https://arxiv.org/abs/2401.00096
    """

    def __init__(
        self,
        target_property: str = "",
        target_weight: float = 1.0,
        forces: bool = False,
        forces_property: str = "forces",
        forces_weight: float = 1.0,
        stress: bool = False,
        stress_weight: float = 1.0,
        criterion: str = "huber",
        huber_delta: float = 0.01,
        **kwargs  # Ignore additional arguments
    ) -> None:
        """Initialize the loss function.

        Args:
            target_property: The target property in the dataset (use energy if falsy).
            target_weight: Weight of the target property in the loss.
            forces: Include forces in the loss.
            forces_property: The forces property in the predictions.
            forces_weight: Weight of the forces in the loss. A good defualt value is num_nodes**2.
            stress: Include stress in the loss.
            stress_weight: Weight of stress in the loss.
            criterion: Loss function criterion (mse, huber).
            huber_delta: Delta value for the Huber loss.
        """
        super().__init__()
        self.target_property = target_property or "energy"  # Use energy if falsy
        self.target_weight = target_weight
        self.forces = forces
        self.forces_property = forces_property
        self.forces_weight = forces_weight
        self.stress = stress
        self.stress_weight = stress_weight
        if criterion.lower() == "mse":
            self.criterion = mse_loss
        elif criterion.lower() == "huber":
            self.criterion = torch.nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    def forward(self, preds: dict, batch: Batch) -> torch.Tensor:
        """Compute the loss.

        Args:
            preds: Dictionary of predictions.
            batch: Batch of data.
        Returns:
            The loss.
        """
        targets = batch.energy if self.target_property == "energy" else batch.targets
        num_nodes = batch.num_nodes.unsqueeze(1)
        loss = self.criterion(preds[self.target_property] / num_nodes, targets / num_nodes)
        loss = self.target_weight * loss
        if self.forces:
            # TODO: Make Huber delta adaptive to the force magnitude on each node.
            forces_loss = self.criterion(preds[self.forces_property], batch.forces)
            loss = loss + self.forces_weight * forces_loss
        if self.stress:
            stress_loss = self.criterion(preds["stress"], batch.stress)
            loss = loss + self.stress_weight * stress_loss
        return loss
