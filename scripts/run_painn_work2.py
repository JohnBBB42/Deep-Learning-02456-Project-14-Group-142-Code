"""Run PaiNN model with atoms data."""

import lightning as L
import torch

import _atomgnn  # noqa: F401
import atomgnn.models.painn
import atomgnn.models.loss
import atomgnn.models.utils

import sys
sys.path.append(r'/home/energy/s244501/sam')
from sam import SAM
import pandas as pd
from pathlib import Path
from run_atoms import configure_cli, run

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
        heteroscedastic: bool = False, # Add NLL loss
        loss_variance: float = 1.0,
        # Optimizer
        init_lr: float = 1e-4,
        use_sam: bool = False,  # Add SAM
        use_asam: bool = False, # Add ASAM
        sam_rho: float = 0.01,
        # Laplace approximation parameters
        use_laplace: bool = False,
        num_laplace_samples: int = 10,
        prior_precision: float = 1.0,
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
        self.use_sam = use_sam
        self.use_asam = use_asam
        self.sam_rho = sam_rho
        self.heteroscedastic = heteroscedastic
        self.use_laplace = use_laplace
        self.num_laplace_samples = num_laplace_samples,
        self.prior_precision = prior_precision
        if self.use_sam and self.use_asam:
            raise ValueError("Cannot use both SAM and ASAM at the same time. Please select one.")
        if self.use_sam or self.use_asam:
            self.automatic_optimization = False  # Disable automatic optimization when using SAM
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
        if self.heteroscedastic:
            self.loss_function = atomgnn.models.loss.GaussianNLLLoss(
                target_property=self.target_property,
                variance=loss_variance,
                forces=forces,
                forces_property=self.forces_property,
                forces_weight=loss_forces_weight,
                stress=stress,
                stress_weight=loss_stress_weight,
                nodewise=loss_nodewise,
            )
        else:
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
        # Now that all parameters are defined, initialize accumulated_squared_gradients

    def on_train_start(self):
        if self.use_laplace:
            # Now that the model is on the correct device, initialize attributes
            self.param_shapes = [p.shape for p in self.parameters()]
            self.accumulated_squared_gradients = [torch.zeros_like(p) for p in self.parameters()]
            self.total_batches = 0  # Reset batch counter
    
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if self.use_sam or self.use_asam:
            # SAM/ASAM logic from notebook:
            def closure():
                optimizer.zero_grad()
                preds = self.forward(batch)
                loss = self.loss_function(preds, batch)
                self.manual_backward(loss)
                return loss
            # First forward-backward pass
            loss = closure()
            # Step optimizer (this will do SAM step)
            optimizer.step(closure)
    
            # Recompute loss after SAM step if needed
            preds = self.forward(batch)
            loss_after = self.loss_function(preds, batch)
            self.log("train_loss", loss_after)
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()
    
            return loss_after
            
        else:
            # Regular training step
            # Compute loss
            preds = self.forward(batch)
            loss = self.loss_function(preds, batch)
            self.log("train_loss", loss)
            if self.use_laplace:
                with torch.no_grad():
                    for i, p in enumerate(self.parameters()):
                        if p.grad is not None:
                            self.accumulated_squared_gradients[i] += p.grad.data.clone() ** 2
                self.total_batches += 1
            # Update learning rate
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()

            return loss
            
    def _grad_norm(self):
        grad_norms = []
        for p in self.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.detach().view(-1))
        grad_norms = torch.cat(grad_norms)
        total_norm = torch.norm(grad_norms, p=2)
        return total_norm

    def on_before_optimizer_step(self, optimizer):
        # Log the total gradient norm of the model
        # Only every 100 steps to reduce overhead
        if self.global_step % 100 == 0:
            norms = L.pytorch.utilities.grad_norm(self.model, norm_type=2)
            self.log("grad_norm", norms["grad_2.0_norm_total"])

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
        if self.use_laplace and hasattr(self, 'hessian_diagonal'):
            # Sample weights from the approximated posterior
            predictions = []
            for _ in range(self.num_laplace_samples):
                sampled_params = []
                for mean, var in zip(self.param_means, self.hessian_diagonal):
                    # Variance is inverse of Hessian diagonal (Laplace approximation)
                    std = (1.0 / (var + 1e-6)).sqrt()
                    noise = torch.randn_like(std) * std
                    sampled_param = mean + noise
                    sampled_params.append(sampled_param)
                # Temporarily load sampled parameters into the model
                param_backup = [p.detach().clone() for p in self.parameters()]
                with torch.no_grad():
                    for p, sp in zip(self.parameters(), sampled_params):
                        p.copy_(sp)
                # Make prediction
                preds = self.forward(batch)
                predictions.append(preds[self.target_property].detach().cpu())
                # Restore original parameters
                with torch.no_grad():
                    for p, pb in zip(self.parameters(), param_backup):
                        p.copy_(pb)
            # Compute mean and variance over samples
            preds_mean = torch.stack(predictions).mean(dim=0)
            preds_var = torch.stack(predictions).var(dim=0)
            return {
                self.target_property: preds_mean,
                f"{self.target_property}_var": preds_var,
            }
        else:
            if self.forces or self.stress:
                torch.set_grad_enabled(True)  # Enable gradients
            preds = self.forward(batch)
            return {k: v.detach().cpu() for k, v in preds.items()}

    def configure_optimizers(self):
      if self.use_sam or self.use_asam:
          base_optimizer = torch.optim.SGD
          adaptive = True if self.use_asam else False
          optimizer = SAM(
              self.parameters(),
              base_optimizer,
              rho=self.sam_rho,
              lr=5e-4,
              momentum=0.9,
              weight_decay=1e-4,
              adaptive=adaptive
          )
          lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
          return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
      else:
          optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
          lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999996)
          return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_train_end(self):
        if self.use_laplace:
            # Compute the approximate Hessian diagonal
            self.hessian_diagonal = []
            for sq_grad in self.accumulated_squared_gradients:
                h_diag = (sq_grad / self.total_batches) + self.prior_precision
                self.hessian_diagonal.append(h_diag)
            # Store the parameter means (trained weights)
            self.param_means = [p.detach().clone() for p in self.parameters()]
            # Save Laplace results as CSV
            param_means_arr = torch.cat([p.view(-1) for p in self.param_means]).cpu().numpy()
            hessian_diagonal_arr = torch.cat([h.view(-1) for h in self.hessian_diagonal]).cpu().numpy()
            pd.DataFrame(param_means_arr).to_csv(Path(self.trainer.log_dir) / "laplace_param_means.csv", index=False)
            pd.DataFrame(hessian_diagonal_arr).to_csv(Path(self.trainer.log_dir) / "laplace_hessian_diagonal.csv", index=False)
            
        super().on_train_end()  # Ensure parent method behavior is preserved
            

class LitSWAGPaiNNModel(LitPaiNNModel):
    """SWAG model extending LitPaiNNModel."""

    def __init__(self, swa_start=0.8, max_num_models=20, no_cov_mat=True, **kwargs):
        super().__init__(**kwargs)
        self.swa_start = swa_start
        self.max_num_models = max_num_models
        self.no_cov_mat = no_cov_mat
        self.num_parameters = sum(p.numel() for p in self.parameters())
        self.register_buffer('mean', torch.zeros(self.num_parameters))
        self.register_buffer('sq_mean', torch.zeros(self.num_parameters))
        if not self.no_cov_mat:
            self.register_buffer('cov_mat_sqrt', torch.zeros((self.max_num_models, self.num_parameters)))
        self.num_models_collected = 0

    def training_step(self, batch, batch_idx):
        # Call the original training_step from LitPaiNNModel
        loss = super().training_step(batch, batch_idx)
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        if current_epoch >= self.swa_start * max_epochs:
            self.collect_model()
        return loss

    def collect_model(self):
        param_vector = torch.nn.utils.parameters_to_vector(self.parameters())
        if self.num_models_collected == 0:
            self.mean.data.copy_(param_vector)
            self.sq_mean.data.copy_(param_vector ** 2)
        else:
            delta = param_vector - self.mean.data
            self.mean.data += delta / (self.num_models_collected + 1)
            delta2 = param_vector ** 2 - self.sq_mean.data
            self.sq_mean.data += delta2 / (self.num_models_collected + 1)
        if not self.no_cov_mat and self.num_models_collected < self.max_num_models:
            idx = self.num_models_collected % self.max_num_models
            self.cov_mat_sqrt[idx].data.copy_(param_vector - self.mean.data)
        self.num_models_collected += 1

    def sample(self, scale=1.0, cov=False):
        mean = self.mean
        sq_mean = self.sq_mean
        var = sq_mean - mean ** 2
        std = torch.sqrt(var + 1e-30)
        z = torch.randn(self.num_parameters, device=mean.device)
        if cov and not self.no_cov_mat:
            cov_mat_sqrt = self.cov_mat_sqrt[:min(self.num_models_collected, self.max_num_models)]
            z_cov = torch.randn(cov_mat_sqrt.size(0), device=mean.device)
            sample = mean + scale * (z * std + cov_mat_sqrt.t().matmul(z_cov) / (self.num_models_collected - 1) ** 0.5)
        else:
            sample = mean + scale * z * std
        torch.nn.utils.vector_to_parameters(sample, self.parameters())

    def predict_step(self, batch, batch_idx, dataloader_idx=None, num_swag_samples=30):
        # Draw multiple samples from SWAG distribution
        predictions = []
        for _ in range(num_swag_samples):
            # Sample a set of parameters from SWAG distribution
            self.sample(scale=1.0, cov=not self.no_cov_mat)
            preds = super().predict_step(batch, batch_idx, dataloader_idx)
            predictions.append(preds)
    
        # Aggregate predictions
        # Assuming 'predictions' is a list of dictionaries where each dictionary has identical keys
        # For each key, stack predictions and compute mean & variance
        aggregated = {}
        keys = predictions[0].keys()
        for k in keys:
            stacked = torch.stack([p[k] for p in predictions])  # [num_swag_samples, ...]
            aggregated[f"{k}_mean"] = stacked.mean(dim=0)
            aggregated[f"{k}_var"] = stacked.var(dim=0, unbiased=False)
    
        return aggregated


    def on_save_checkpoint(self, checkpoint):
        checkpoint['mean'] = self.mean
        checkpoint['sq_mean'] = self.sq_mean
        checkpoint['num_models_collected'] = self.num_models_collected
        if not self.no_cov_mat:
            checkpoint['cov_mat_sqrt'] = self.cov_mat_sqrt

    def on_load_checkpoint(self, checkpoint):
        self.mean = checkpoint['mean']
        self.sq_mean = checkpoint['sq_mean']
        self.num_models_collected = checkpoint['num_models_collected']
        if not self.no_cov_mat:
            self.cov_mat_sqrt = checkpoint['cov_mat_sqrt']

    def on_train_end(self):
        # Save SWAG buffers as a checkpoint
        swag_state = {
            'mean': self.mean,
            'sq_mean': self.sq_mean,
            'num_models_collected': self.num_models_collected,
        }
        if not self.no_cov_mat:
            swag_state['cov_mat_sqrt'] = self.cov_mat_sqrt
        swag_ckpt_path = Path(self.trainer.log_dir) / "swag.ckpt"
        torch.save(swag_state, swag_ckpt_path)
    
        # Save SWAG mean and variance explicitly
        mean_arr = self.mean.cpu().numpy()
        sq_mean_arr = self.sq_mean.cpu().numpy()
        variance_arr = sq_mean_arr - mean_arr**2  # Compute variance
    
        # Save the results as CSV
        pd.DataFrame(mean_arr).to_csv(Path(self.trainer.log_dir) / "swag_mean.csv", index=False)
        pd.DataFrame(variance_arr).to_csv(Path(self.trainer.log_dir) / "swag_variance.csv", index=False)
    
        # Save the square mean (optional, but keeping it for traceability)
        pd.DataFrame(sq_mean_arr).to_csv(Path(self.trainer.log_dir) / "swag_sq_mean.csv", index=False)
    
        # Save covariance matrix (if enabled)
        if not self.no_cov_mat:
            pd.DataFrame(self.cov_mat_sqrt.cpu().numpy()).to_csv(Path(self.trainer.log_dir) / "swag_cov_mat_sqrt.csv", index=False)
    
        # Call the parent method to preserve existing behavior
        super().on_train_end()



def main():
    cli = configure_cli("run_painn")
    # Add model-specific arguments
    cli.add_lightning_class_args(LitPaiNNModel, "model")
    # Link arguments as needed
    cli.link_arguments("data.cutoff", "model.cutoff", apply_on="parse")
    cli.link_arguments("data.pbc", "model.pbc", apply_on="parse")
    cli.link_arguments("data.target_property", "model.target_property", apply_on="parse")
    # Link SAM arguments
    cli.link_arguments("use_sam", "model.use_sam", apply_on="parse")
    cli.link_arguments("use_asam", "model.use_asam", apply_on="parse")
    cli.link_arguments("sam_rho", "model.sam_rho", apply_on="parse")
    # Link heteroscedastic argument
    cli.link_arguments("heteroscedastic", "model.heteroscedastic", apply_on="parse")
    # Laplace
    cli.link_arguments("use_laplace", "model.use_laplace", apply_on="parse")
    cli.link_arguments("num_laplace_samples", "model.num_laplace_samples", apply_on="parse")

    # Run the script
    cfg = cli.parse_args()
    if cfg.use_swag:
        lit_model_cls = LitSWAGPaiNNModel
    else:
        lit_model_cls = LitPaiNNModel

    run(cli, lit_model_cls=lit_model_cls)


if __name__ == '__main__':
    main()
