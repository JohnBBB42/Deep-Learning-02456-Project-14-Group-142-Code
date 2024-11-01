"""Run PaiNN model with atoms data."""

import lightning as L
import torch

import _atomgnn  # noqa: F401
import atomgnn.models.painn
import atomgnn.models.loss
import atomgnn.models.utils

from run_atoms import configure_cli, run
from torch.optim.swa_utils import AveragedModel, SWALR


class LitPaiNNModel(L.LightningModule):
    """PaiNN model."""

    def __init__(
        self,
        node_size: int = 64,
        edge_size: int = 20,
        num_interaction_blocks: int = 3,
        cutoff: float = 5.0,
        pbc: bool = False,
        readout_reduction: str = "sum",
        target_property: str = "",
        forces: bool = False,
        stress: bool = False,
        # Loss function
        loss_forces_weight: float = 0.5,
        loss_stress_weight: float = 0.1,
        loss_nodewise: bool = False,
        # Optimizer
        init_lr: float = 1e-4,
        # Underscores hide these arguments from the CLI
        _output_scale: float = 1.0,
        _output_offset: float = 0.0,
        _nodewise_offset: bool = True,
        **kwargs,  # Ignore additional arguments
    ):
        """Initialize model.

        Args:
            node_size: Size of the node state embedding.
            edge_size: Size of the edge state expansion.
            num_interaction_blocks: Number of interaction blocks.
            cutoff: Cutoff distance for interactions.
            pbc: Enable periodic boundary conditions (pbc).
            readout_reduction: Readout reduction operation (sum, mean).
            target_property: The target property in the dataset (use energy if falsy).
            forces: Enable calculation of forces.
            stress: Enable calculation of stress.
            loss_forces_weight: Trade-off between energy and forces in the loss function.
            loss_stress_weight: Weight of stress in the loss function.
            loss_nodewise: Use nodewise loss instead of global loss.
            init_lr: Initial learning rate.
            _output_scale: Output scale parameter (set automatically).
            _output_offset: Output offset parameter (set automatically).
            _nodewise_offset: Enable nodewise offset (set automatically).
        """
        super().__init__()
        self.save_hyperparameters()
        self.target_property = target_property or "energy"  # Use energy if falsy property is given
        self.forces_property = "forces"
        self.stress_property = "stress"
        self.forces = forces
        self.stress = stress
        self.init_lr = init_lr
        # Initialize model
        model: torch.nn.Module = atomgnn.models.painn.PaiNN(
            node_size=node_size,
            edge_size=edge_size,
            num_interaction_blocks=num_interaction_blocks,
            cutoff=cutoff,
            pbc=pbc,
            readout_reduction=readout_reduction,
        )
        model = atomgnn.models.utils.DictOutputWrapper(
            model,
            output_keys=[self.target_property],
        )
        model = atomgnn.models.utils.ScaleOutputWrapper(
            model,
            output_property=self.target_property,
            scale=torch.tensor(_output_scale),
            offset=torch.tensor(_output_offset),
            nodewise=_nodewise_offset,
        )
        model = atomgnn.models.utils.GradOutputWrapper(
            model,
            forces=forces,
            stress=stress,
            energy_property=self.target_property,
            forces_property=self.forces_property,
            stress_property=self.stress_property,
        )
        self.model = model
        # Initialize loss function
        self.loss_function = atomgnn.models.loss.MSELoss(
            target_property=self.target_property,
            forces=forces,
            forces_property=self.forces_property,
            forces_weight=loss_forces_weight,
            stress=stress,
            stress_weight=loss_stress_weight,
            nodewise=loss_nodewise,
        )
        self._metrics: dict[str, torch.Tensor] = dict()  # Accumulated evaluation metrics
        # Initialize SWAG weight snapshots
        self.weight_snapshots = []  # This will store weight snapshots for SWAG

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        # Compute loss
        preds = self.forward(batch)
        loss = self.loss_function(preds, batch)
        self.log("train_loss", loss)
        
        # Update learning rate for the base scheduler
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        
        return loss  # Return loss to trigger the backward pass

    def _compute_swag_moments(self):
        """Calculate the mean and squared mean for SWAG."""
        num_snapshots = len(self.weight_snapshots)
        swag_mean = {k: torch.zeros_like(v) for k, v in self.weight_snapshots[0].items()}
        swag_squared_mean = {k: torch.zeros_like(v) for k, v in self.weight_snapshots[0].items()}
    
        # Compute first and second moments
        for snapshot in self.weight_snapshots:
            for k, v in snapshot.items():
                swag_mean[k] += v / num_snapshots
                swag_squared_mean[k] += v ** 2 / num_snapshots
    
        return swag_mean, swag_squared_mean
        
    def sample_swag_weights(self):
        """Sample weights from the SWAG posterior and load into model."""
        sampled_weights = {}
        for k, mean in self.swag_mean.items():
            variance = self.swag_squared_mean[k] - mean ** 2
            std_dev = torch.sqrt(variance.clamp(min=1e-6))  # Add stability to prevent negative variances
            sampled_weights[k] = mean + std_dev * torch.randn_like(mean)
    
        # Load sampled weights into the model
        self.model.load_state_dict(sampled_weights)


    def on_train_epoch_end(self):
        # Apply SWA model updates at the end of each epoch
        self.swa_model.update_parameters(self)
        self.swa_scheduler.step()
    
        # --- SWAG addition ---
        # Take a weight snapshot for SWAG
        state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.weight_snapshots.append(state_dict)
        
        # Limit the number of snapshots to avoid memory issues
        if len(self.weight_snapshots) > 20:
            self.weight_snapshots.pop(0)
        # ----------------------

    def on_before_optimizer_step(self, optimizer):
        # Log the total gradient norm of the model
        if self.global_step % 100 == 0:
            # Calculate norm only if there are gradients
            grad_norms = [p.grad.norm(2) for p in self.model.parameters() if p.grad is not None]
            if grad_norms:  # Check that grad_norms is not empty
                grad_norm = torch.norm(torch.stack(grad_norms))
                self.log("grad_norm", grad_norm)


    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "val")

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        self._on_eval_epoch_end("test")

    def _eval_step(self, batch, batch_idx, prefix):
        # Compute predictions and error
        with torch.enable_grad():  # Enable gradients for computing forces
            preds = self.forward(batch)
        targets = batch.energy if self.target_property == "energy" else batch.targets
        error = targets - preds[self.target_property]
        # Initialize evaluation metrics on first step
        if not self._metrics:
            for k in ["num_data", "num_nodes", "loss", "sse", "sae"]:
                self._metrics[k] = 0.0
            if self.forces:
                for k in ["forces_sse", "forces_sae"]:
                    self._metrics[k] = 0.0
            if self.stress:
                for k in ["stress_sse", "stress_sae"]:
                    self._metrics[k] = 0.0
        # Accumulate evaluation metrics
        self._metrics["num_data"] += batch.num_data
        self._metrics["num_nodes"] += torch.sum(batch.num_nodes)
        self._metrics["loss"] += self.loss_function(preds, batch) * batch.num_data
        self._metrics["sse"] += torch.sum(torch.square(error))
        self._metrics["sae"] += torch.sum(torch.abs(error))
        if self.forces:
            # The force errors are component-wise and averaged over the spatial dimensions and atoms
            forces_error = batch.forces - preds[self.forces_property]
            self._metrics["forces_sse"] += torch.sum(torch.square(forces_error))
            self._metrics["forces_sae"] += torch.sum(torch.abs(forces_error))
        if self.stress:
            # The stress errors are component-wise and averaged over the number of data
            stress_error = batch.stress - preds[self.stress_property]
            self._metrics["stress_sse"] += torch.sum(torch.square(stress_error))
            self._metrics["stress_sae"] += torch.sum(torch.abs(stress_error))

    def _on_eval_epoch_end(self, prefix):
        # Compute and log metrics over the entire evaluation set.
        metrics = {}
        metrics["loss"] = self._metrics["loss"] / self._metrics["num_data"]
        metrics["mse"] = self._metrics["sse"] / self._metrics["num_data"]
        metrics["mae"] = self._metrics["sae"] / self._metrics["num_data"]
        if self.forces:
            # The force errors are component-wise and averaged over the spatial dimensions and atoms
            metrics["forces_mse"] = self._metrics["forces_sse"] / (self._metrics["num_nodes"] * 3)
            metrics["forces_mae"] = self._metrics["forces_sae"] / (self._metrics["num_nodes"] * 3)
        if self.stress:
            # The stress errors are component-wise and averaged over the number of data
            metrics["stress_mse"] = self._metrics["stress_sse"] / self._metrics["num_data"]
            metrics["stress_mae"] = self._metrics["stress_sae"] / self._metrics["num_data"]
        # Log evaluation metrics on epoch end
        self.log_dict({f"{prefix}_{k}": v for k, v in metrics.items()}, sync_dist=True)
        self._metrics.clear()  # Clear accumulated evaluation metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if self.forces or self.stress:
            torch.set_grad_enabled(True)
        
        # --- SWAG addition ---
        # Sample from SWAG if in testing mode
        if self.trainer.state.fn == "test":
            self.sample_swag_weights()
        # ----------------------
    
        preds = self.model(batch)
        return {k: v.detach().cpu() for k, v in preds.items()}


    def configure_optimizers(self):
        # Initialize the base optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        # Define a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999996)
        
        # Initialize the SWA model and SWA scheduler
        self.swa_model = AveragedModel(self)  # Create an SWA wrapper around the model
        self.swa_scheduler = SWALR(optimizer, anneal_strategy="cos", anneal_epochs=10, swa_lr=5e-5)
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


    def on_train_end(self):
        # Finalize SWA
        torch.optim.swa_utils.update_bn(self.train_dataloader(), self.swa_model)
        
        # --- SWAG addition ---
        # Compute SWAG mean and squared mean from snapshots
        self.swag_mean, self.swag_squared_mean = self._compute_swag_moments()
        # Use SWA model as the default model for predictions
        self.model = self.swa_model
        # ----------------------



def main():
    cli = configure_cli("run_painn")
    cli.add_lightning_class_args(LitPaiNNModel, "model")
    cli.link_arguments("data.cutoff", "model.cutoff", apply_on="parse")
    cli.link_arguments("data.pbc", "model.pbc", apply_on="parse")
    cli.link_arguments("data.target_property", "model.target_property", apply_on="parse")
    run(cli, LitPaiNNModel)


if __name__ == '__main__':
    main()
