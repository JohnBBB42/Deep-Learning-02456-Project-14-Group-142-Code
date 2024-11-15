"""Run PaiNN model with atoms data."""

import lightning as L
import torch
import torch.nn.functional as F
from torch_scatter import scatter

import _atomgnn  # noqa: F401
import atomgnn.models.painn
import atomgnn.models.loss
import atomgnn.models.utils

from pathlib import Path
from run_atoms import configure_cli, run

# Import necessary components from the PaiNN model
from atomgnn.data.data import Batch
from atomgnn.models.utils import (BesselExpansion, compute_edge_vectors_and_norms,
                                  cosine_cutoff, reduce_splits, sum_index)
from atomgnn.models.painn import PaiNNInteractionBlock

class PaiNNWithEmbeddings(torch.nn.Module):
    """PaiNN model that returns node embeddings."""

    def __init__(
        self,
        node_size: int = 64,
        edge_size: int = 20,
        num_interaction_blocks: int = 3,
        cutoff: float = 5.0,
        pbc: bool = False,
        # Readout
        use_readout: bool = True,
        num_readout_layers: int = 2,
        readout_size: int = 1,
        readout_reduction: str | None = "sum",
        **kwargs,
    ) -> None:
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.num_interaction_blocks = num_interaction_blocks
        self.cutoff = cutoff
        self.pbc = pbc
        self.use_readout = use_readout
        self.num_readout_layers = num_readout_layers
        self.readout_size = readout_size
        self.readout_reduction = readout_reduction
        # Setup node embedding
        num_embeddings = 119  # atomic numbers + 1
        self.node_embedding = torch.nn.Embedding(num_embeddings, node_size)
        # Setup edge expansion
        self.edge_expansion = BesselExpansion(edge_size, cutoff)
        # Setup interaction blocks
        self.interaction_blocks = torch.nn.ModuleList(
            PaiNNInteractionBlock(node_size, edge_size, cutoff)
            for _ in range(num_interaction_blocks))
        # Setup readout network
        if self.use_readout:
            self.readout_net = torch.nn.Sequential()
            for _ in range(num_readout_layers - 1):
                self.readout_net.append(torch.nn.Linear(node_size, node_size))
                self.readout_net.append(torch.nn.SiLU())
            self.readout_net.append(torch.nn.Linear(node_size, readout_size))

    def forward(self, input: Batch) -> torch.Tensor:
        # Check for the presence of node features in the input batch
        if not hasattr(input, 'node_features') or input.node_features is None:
            raise AttributeError("Input batch does not have 'node_features' or it is None.")
    
        node_states_scalar = self.node_embedding(input.node_features.squeeze())
        # Init vector node states to zero as there is no initial directional information
        node_states_vector = torch.zeros(
            (node_states_scalar.shape[0], 3, self.node_size),
            dtype=node_states_scalar.dtype,
            device=node_states_scalar.device
        )
        # Compute edge vectors and distances
        edge_vectors, edge_norms = compute_edge_vectors_and_norms(
            input.node_positions, input.edge_index,
            input.edge_shift if self.pbc else None,
            input.cell[input.edge_data_index] if self.pbc else None,
        )
        # Normalize edge vectors (add small number to avoid division by zero)
        edge_vectors = edge_vectors / (edge_norms + 1e-10)
        # Get edge states by expansion of the edge distances
        edge_states = self.edge_expansion(edge_norms)
        # Apply interaction blocks
        for interaction_block in self.interaction_blocks:
            node_states_scalar, node_states_vector = interaction_block(
                node_states_scalar, node_states_vector,
                edge_states, edge_vectors, edge_norms, input.edge_index)
        # Apply node-level readout
        if self.use_readout:
            node_states_scalar_readout = self.readout_net(node_states_scalar)
        else:
            node_states_scalar_readout = node_states_scalar
        # Prepare output
        if self.readout_reduction:
            # Graph-level output
            graph_states_scalar = reduce_splits(
                node_states_scalar_readout, input.num_nodes, reduction=self.readout_reduction
            )
            output_scalar = graph_states_scalar
        else:
            # Node-level output
            output_scalar = node_states_scalar_readout
        # Calculate Laplacian of node states (as an example, for scalar states)
        laplacian_scalar = self._compute_laplacian(node_states_scalar, input.edge_index)
        # Return both the output and node embeddings
        return output_scalar, node_states_scalar, laplacian_scalar  # node_states_scalar are the node embeddings
    
    def _compute_laplacian(self, node_states, edge_index):
        """Compute the graph Laplacian for node states."""
        # Ensure edge_index is of shape (2, num_edges)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Expected edge_index to have shape (2, num_edges), but got {edge_index.shape}")
        
        row, col = edge_index[0], edge_index[1]
        degree = scatter(torch.ones_like(row, dtype=node_states.dtype), row, dim=0, reduce="sum")
        laplacian = degree.unsqueeze(1) * node_states - scatter(node_states[col], row, dim=0, reduce="sum")
        return laplacian


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
        use_sam: bool = False,  # Add SAM
        use_asam: bool = False, # Add ASAM
        sam_rho: float = 0.05,
        # Heteroscedastic
        heteroscedastic: bool = False,
        use_laplace: bool = False,
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
        self.use_sam = use_sam  # Save use_sam as an instance variable
        self.use_asam = use_asam
        self.sam_rho = sam_rho  # Add this line
        self.readout_reduction = readout_reduction
        self.heteroscedastic = heteroscedastic
        self.use_laplace = use_laplace
        self._output_scale = _output_scale
        self._output_offset = _output_offset
        if self.use_sam and self.use_asam:
            raise ValueError("Cannot use both SAM and ASAM at the same time. Please select one.")
        if self.use_sam or self.use_asam:
            self.automatic_optimization = False  # Disable automatic optimization when using SAM
        # Initialize model

        if self.heteroscedastic:
            # Use the custom PaiNN model that returns embeddings
            model: torch.nn.Module = PaiNNWithEmbeddings(
                node_size=node_size,
                edge_size=edge_size,
                num_interaction_blocks=num_interaction_blocks,
                cutoff=cutoff,
                pbc=pbc,
                readout_reduction=readout_reduction,
            )
        else:
            model: torch.nn.Module = atomgnn.models.painn.PaiNN(
                node_size=node_size,
                edge_size=edge_size,
                num_interaction_blocks=num_interaction_blocks,
                cutoff=cutoff,
                pbc=pbc,
                readout_reduction=readout_reduction,
            )
        # Define the readout layer
        if self.heteroscedastic:
            self.readout = HeteroscedasticReadout(input_size=node_size, reduction=readout_reduction)
        else:
            self.readout = torch.nn.Linear(node_size, 1)
        output_keys = [self.target_property]
        if self.heteroscedastic:
            output_keys.append('node_states_scalar')
        if not self.heteroscedastic:
            model = atomgnn.models.utils.DictOutputWrapper(
                model,
                #output_keys=[self.target_property],
                output_keys=output_keys,
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
            self.loss_function = self.heteroscedastic_loss
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
        if self.use_laplace:
            self.num_parameters = sum(p.numel() for p in self.parameters())
            self.posterior_mean = torch.nn.utils.parameters_to_vector(self.parameters()).detach()
            self.hessian_diag = None  # Will be computed later
        self._metrics: dict[str, torch.Tensor] = dict()  # Accumulated evaluation metrics

    def heteroscedastic_loss(self, preds, batch):
        targets = batch.energy if self.target_property == "energy" else batch.targets
        error = targets - preds[self.target_property]
        log_variance = preds["log_variance"]
        variance = torch.exp(log_variance)
        loss_energy = 0.5 * ((error ** 2) / variance + log_variance).mean()
        loss = loss_energy
        if self.forces:
            forces_error = batch.forces - preds[self.forces_property]
            loss_forces = self.hparams.loss_forces_weight * torch.nn.functional.mse_loss(
                preds[self.forces_property], batch.forces)
            loss += loss_forces
        return loss


    def forward(self, batch):
        if self.use_laplace:
            output_scalar, node_states_scalar, laplacian_scalar = self.model(batch)
            mean, log_variance = self.readout(node_states_scalar, batch.node_data_index)
            # Apply scaling to mean
            outputs = {
                self.target_property: mean,
                'log_variance': log_variance.squeeze(-1),
                'laplacian': laplacian_scalar  # Include Laplacian in output
            }
        else:
            outputs = self.model(batch)
        return outputs

    #def forward(self, batch):
        #return self.model(batch)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        if self.use_sam:
            # First forward-backward pass
            optimizer.zero_grad()
            preds = self.forward(batch)
            targets = batch.energy if self.target_property == "energy" else batch.targets
            loss = self.loss_function(preds, batch)
            # Add Laplacian regularization term if desired
            if self.use_laplace and 'laplacian' in preds:
                laplacian_term = torch.mean(torch.norm(preds['laplacian'], dim=1))
                loss += 0.01 * laplacian_term  # Adjust the coefficient as needed
            self.manual_backward(loss)
            # Compute gradient norm
            grad_norm = torch.norm(
                torch.stack([
                    p.grad.detach().norm(2)
                    for p in self.parameters()
                    if p.grad is not None
                ])
            )
            # Log grad_norm
            self.log('grad_norm', grad_norm)
            # Perturb parameters
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is None:
                        continue
                    e_w = p.grad / (grad_norm + 1e-12) * self.sam_rho
                    p.add_(e_w)
            # Second forward-backward pass
            optimizer.zero_grad()
            preds_adv = self.forward(batch)
            loss_adv = self.loss_function(preds_adv, batch)
            self.manual_backward(loss_adv)
            # Restore original parameters
            with torch.no_grad():
                for p in self.parameters():
                    if p.grad is None:
                        continue
                    e_w = p.grad / (grad_norm + 1e-12) * self.sam_rho
                    p.sub_(e_w)
            # Update parameters
            optimizer.step()
            self.log('train_loss', loss_adv)
            # Step the learning rate scheduler
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()
        elif self.use_asam:
            # ASAM optimization
            optimizer.zero_grad()
            preds = self.forward(batch)
            targets = batch.energy if self.target_property == "energy" else batch.targets
            loss = self.loss_function(preds, batch)
            # Add Laplacian regularization term if desired
            if self.use_laplace and 'laplacian' in preds:
                laplacian_term = torch.mean(torch.norm(preds['laplacian'], dim=1))
                loss += 0.01 * laplacian_term  # Adjust the coefficient as needed
            self.manual_backward(loss)
            # Compute parameter norms and scaled gradients
            with torch.no_grad():
                param_norms = []
                for p in self.parameters():
                    if p.grad is None:
                        param_norms.append(None)
                        continue
                    param_norm = torch.norm(p)
                    param_norms.append(param_norm)
                # Compute scaled gradients
                scaled_grads = []
                for p, param_norm in zip(self.parameters(), param_norms):
                    if p.grad is None:
                        continue
                    scaled_grad = p.grad / (param_norm + 1e-12)
                    scaled_grads.append(scaled_grad.view(-1))
                # Compute the overall scaled gradient norm
                scaled_grad_norm = torch.norm(torch.cat(scaled_grads))
                # Compute epsilon
                epsilon = self.sam_rho / (scaled_grad_norm + 1e-12)
                # Perturb parameters
                for p, param_norm in zip(self.parameters(), param_norms):
                    if p.grad is None:
                        continue
                    perturbation = epsilon * p.grad / (param_norm + 1e-12)
                    p.add_(perturbation)
            # Second forward-backward pass
            optimizer.zero_grad()
            preds_adv = self.forward(batch)
            loss_adv = self.loss_function(preds_adv, batch)
            self.manual_backward(loss_adv)
            # Restore original parameters
            with torch.no_grad():
                for p, param_norm in zip(self.parameters(), param_norms):
                    if p.grad is None:
                        continue
                    perturbation = epsilon * p.grad / (param_norm + 1e-12)
                    p.sub_(perturbation)
            # Update parameters
            optimizer.step()
            self.log('train_loss', loss_adv)
            # Step the learning rate scheduler
            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()
        else:
            # Regular training step
            # Compute loss
            preds = self.forward(batch)
            targets = batch.energy if self.target_property == "energy" else batch.targets
            loss = self.loss_function(preds, batch)
            # Add Laplacian regularization term if desired
            if self.use_laplace and 'laplacian' in preds:
                laplacian_term = torch.mean(torch.norm(preds['laplacian'], dim=1))
                loss += 0.01 * laplacian_term  # Adjust the coefficient as needed
            self.log("train_loss", loss)
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
        # Check if node_features are present
        if not hasattr(batch, 'node_features') or batch.node_features is None:
            raise AttributeError(f"{prefix} Batch {batch_idx} does not have 'node_features' or it is None.")
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
            torch.set_grad_enabled(True)  # Enable gradients
        if self.use_laplace:
            self.sample_from_posterior()
        preds = self.forward(batch)
        return {k: v.detach().cpu() for k, v in preds.items()}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999996)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def sample_from_posterior(self):
        """Sample parameters from the approximate posterior."""
        if self.hessian_diag is None:
            raise ValueError("Hessian diagonal not computed. Call laplace_approximation first.")
        std = torch.sqrt(1.0 / (self.hessian_diag + 1e-6))
        sampled_params = self.posterior_mean + std * torch.randn_like(self.posterior_mean)
        torch.nn.utils.vector_to_parameters(sampled_params, self.parameters())
        
    def on_save_checkpoint(self, checkpoint):
        # Call the superclass method to handle default behavior
        super().on_save_checkpoint(checkpoint)
        if self.use_laplace:
            # Save the Laplace approximation parameters
            checkpoint['posterior_mean'] = self.posterior_mean
            checkpoint['hessian_diag'] = self.hessian_diag

    def on_load_checkpoint(self, checkpoint):
        # Call the superclass method to handle default behavior
        super().on_load_checkpoint(checkpoint)
        if self.use_laplace:
            # Load the Laplace approximation parameters
            self.posterior_mean = checkpoint['posterior_mean']
            self.hessian_diag = checkpoint['hessian_diag']
    
    def on_fit_end(self):
        if self.use_laplace:
            self.laplace_approximation()


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

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.sample(scale=1.0, cov=not self.no_cov_mat)
        preds = super().predict_step(batch, batch_idx, dataloader_idx)
        return preds

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
        # Save SWAG buffers
        swag_state = {
            'mean': self.mean,
            'sq_mean': self.sq_mean,
            'num_models_collected': self.num_models_collected,
        }
        if not self.no_cov_mat:
            swag_state['cov_mat_sqrt'] = self.cov_mat_sqrt
        swag_ckpt_path = Path(self.trainer.log_dir) / "swag.ckpt"
        torch.save(swag_state, swag_ckpt_path)

class HeteroscedasticReadout(torch.nn.Module):
    def __init__(self, input_size, reduction='sum'):
        super().__init__()
        self.reduction = reduction
        self.mean_layer = torch.nn.Linear(input_size, 1)
        self.log_variance_layer = torch.nn.Linear(input_size, 1)

    def forward(self, x, batch):
        # x: node embeddings, batch: batch indices for nodes
        # Aggregate node embeddings to graph-level embeddings
        graph_embeddings = scatter(x, batch, dim=0, reduce=self.reduction)
        mean = self.mean_layer(graph_embeddings)
        log_variance = self.log_variance_layer(graph_embeddings)
        return mean, log_variance


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
    # Link Laplace approximation argument
    cli.link_arguments("use_laplace", "model.use_laplace", apply_on="parse")
  
    # Run the script
    cfg = cli.parse_args()
    if cfg.use_swag:
        lit_model_cls = LitSWAGPaiNNModel
    else:
        lit_model_cls = LitPaiNNModel

    run(cli, lit_model_cls=lit_model_cls)


if __name__ == '__main__':
    main()
