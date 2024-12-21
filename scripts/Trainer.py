################################################################
# Trainer with SWA, SWAG, SAM, ASAM and Laplace Implementation #
################################################################

import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import models.painn
import models.loss
import data.utils

class Trainer:
    def __init__(self, model, lr=1e-3, use_sam=False, use_asam=False, sam_rho=0.05,
                 use_laplace=False, num_laplace_samples=10, prior_precision=0.0,
                 use_swa=False, swa_lrs=1e-4, swa_start_percent=0.8, annealing_percent=0.05, 
                 annealing_strategy='cos', 
                 use_swag=False, max_num_models=20, no_cov_mat=True, loss_type="mse", max_steps=10000): 

        self.model = model

        self.use_sam = use_sam
        self.use_asam = use_asam
        self.sam_rho = sam_rho

        self.use_laplace = use_laplace
        self.num_laplace_samples = num_laplace_samples
        self.prior_precision = prior_precision

        self.use_swa = use_swa
        self.use_swag = use_swag
        self.swa_lrs = swa_lrs
        self.swa_start_percent = swa_start_percent
        self.annealing_percent = annealing_percent
        self.annealing_strategy = annealing_strategy
        self.max_num_models = max_num_models
        self.no_cov_mat = no_cov_mat

        if self.use_sam or self.use_asam:
            # Initialize base optimizer (SGD with Momentum)
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            mode = "ASAM" if self.use_asam else "SAM"
            print(f"Optimizer set to SGD with Momentum for {mode} (rho={self.sam_rho}.")
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print("Optimizer set to Adam.")
        
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)

        # Choose loss function
        if loss_type == "mse":
            print("Using MSE Loss.")
            self.loss_function = MSELoss(target_property="energy", forces=False)
        elif loss_type == "nll":
            print("Using Gaussian NLL Loss.")
            self.loss_function = GaussianNLLLoss(target_property="energy", variance=1.0, forces=False)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        if use_swag:
            self.num_parameters = sum(p.numel() for p in self.model.parameters())
            device = next(self.model.parameters()).device  # Get device from model
            self.mean = torch.zeros(self.num_parameters, device=device)
            self.sq_mean = torch.zeros(self.num_parameters, device=device)
            if not self.no_cov_mat:
                self.cov_mat_sqrt = torch.zeros(self.max_num_models, self.num_parameters, device=device)
            self.num_models_collected = 0

        if use_laplace:
            self._init_laplace()

        if use_sam and use_asam:
            raise ValueError("Cannot use both SAM and ASAM simultaneously.")
        
        if self.use_swa and self.use_swag:
            raise ValueError("Cannot use both SWA and SWAG simultaneously.")

        self.current_step = 0          # Initialize step counter
        self.max_steps = max_steps      # Total number of training steps
        self.swa_start_step = None      # Will be set in set_max_steps
        self.annealing_steps = None     # Will be set in set_max_steps
        self.epoch = 0  # Initialize epoch counter

        # Initialize SWA/SWAG if enabled
        if self.use_swa or self.use_swag:
            self.setup_swa()



    def set_max_steps(self, max_steps):
        """Set the maximum number of training steps and compute SWA parameters."""
        self.max_steps = max_steps
        if self.use_swa or self.use_swag:
            self.swa_start_step = int(self.swa_start_percent * self.max_steps)
            self.annealing_steps = int(self.annealing_percent * self.max_steps)
            print(f"SWA will start at step {self.swa_start_step} and anneal over {self.annealing_steps} steps.")

    def setup_swa(self):
        if self.use_swa:
            self.averaged_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lrs)
            print("SWA has been set up.")
        
        if self.use_swag:
            # SWAG setup already handled in __init__
            print("SWAG has been set up.")

    def _init_laplace(self):
        self.accumulated_squared_gradients = [torch.zeros_like(p) for p in self.model.parameters()]
        self.total_batches = 0


    def train_step(self, batch):
        self.model.train()
        # Forward pass
        pred = self.model(batch)
        # Assume batch.energy is the target
        preds_dict = {"energy": pred}  
        loss = self.loss_function(preds_dict, batch)

        if self.use_sam:
            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm
            grad_norm = torch.norm(
                torch.stack([p.grad.detach().norm(2) for p in self.model.parameters() if p.grad is not None])
            )

            # SAM perturbation
            e_ws = []
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is None:
                        e_ws.append(None)
                        continue
                    e_w = p.grad / (grad_norm + 1e-12) * self.sam_rho
                    p.add_(e_w)
                    e_ws.append(e_w)

            # Second forward-backward pass
            self.optimizer.zero_grad()
            pred2 = self.model(batch)
            loss2 = self.loss_function({"energy": pred2}, batch)  # Corrected line
            loss2.backward()

            with torch.no_grad():
                for p, e_w in zip(self.model.parameters(), e_ws):
                    if e_w is not None:
                        p.sub_(e_w)

            self.optimizer.step()

            # Define print frequency (e.g., every 10 steps)
            print_frequency = 10
            if self.current_step % print_frequency == 0:
                # Calculate average perturbation norm
                perturbation_norms = [e_w.norm().item() for e_w in e_ws if e_w is not None]
                avg_perturbation_norm = (
                    sum(perturbation_norms) / len(perturbation_norms) if perturbation_norms else 0.0
                )

                print(f"[SAM] Step {self.current_step}, Gradient Norm: {grad_norm.item():.4f}")
                print(f"[SAM] Step {self.current_step}, Average Perturbation Norm: {avg_perturbation_norm:.6f}")
                print(f"[SAM] Step {self.current_step}, Loss Before SAM: {loss.item():.6f}, Loss After SAM: {loss2.item():.6f}")
                print(f"[SAM] Step {self.current_step} SAM optimization completed.")

            final_loss = loss2

        elif self.use_asam:
            self.optimizer.zero_grad()
            loss.backward()
            param_norms = []
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is None:
                        param_norms.append(None)
                        continue
                    param_norm = torch.norm(p)
                    param_norms.append(param_norm)
                scaled_grads = []
                for p, pn in zip(self.model.parameters(), param_norms):
                    if p.grad is None:
                        continue
                    scaled_grad = p.grad / (pn + 1e-12)
                    scaled_grads.append(scaled_grad.view(-1))
                scaled_grad_norm = torch.norm(torch.cat(scaled_grads))
                epsilon = self.sam_rho / (scaled_grad_norm + 1e-12)

                perturbations = []
                for p, pn in zip(self.model.parameters(), param_norms):
                    if p.grad is None:
                        perturbations.append(None)
                        continue
                    perturbation = epsilon * p.grad / (pn + 1e-12)
                    p.add_(perturbation)
                    perturbations.append(perturbation)

            # Second forward-backward pass
            self.optimizer.zero_grad()
            pred2 = self.model(batch)
            loss2 = self.loss_function({"energy": pred2}, batch)  # Corrected line
            loss2.backward()

            with torch.no_grad():
                for p, perturbation in zip(self.model.parameters(), perturbations):
                    if perturbation is not None:
                        p.sub_(perturbation)

            self.optimizer.step()

            # Define print frequency (e.g., every 10 steps)
            print_frequency = 10
            if self.current_step % print_frequency == 0:
                # Calculate average perturbation norm for ASAM
                perturbation_norms_asam = [perturbation.norm().item() for perturbation in perturbations if perturbation is not None]
                avg_perturbation_norm_asam = (
                    sum(perturbation_norms_asam) / len(perturbation_norms_asam) if perturbation_norms_asam else 0.0
                )

                print(f"[ASAM] Step {self.current_step}, Average Perturbation Norm: {avg_perturbation_norm_asam:.6f}")
                print(f"[ASAM] Step {self.current_step}, Loss Before ASAM: {loss.item():.6f}, Loss After ASAM: {loss2.item():.6f}")
                print(f"[ASAM] Step {self.current_step} ASAM optimization completed.")

            final_loss = loss2
            
        else:
            self.optimizer.zero_grad()
            loss.backward()
            if self.use_laplace:
                # Accumulate grad^2
                with torch.no_grad():
                    for i, p in enumerate(self.model.parameters()):
                        if p.grad is not None:
                            self.accumulated_squared_gradients[i] += p.grad.data.clone() ** 2
                self.total_batches += 1
            self.optimizer.step()
            final_loss = loss

        # Increment step counter
        self.current_step += 1

        # Handle SWA based on steps
        if self.use_swa and self.current_step >= self.swa_start_step:
            # Update SWA parameters
            self.averaged_model.update_parameters(self.model)
            # Annealing the SWA learning rate
            if self.annealing_steps > 0 and self.current_step <= (self.swa_start_step + self.annealing_steps):
                self.swa_scheduler.step()
            elif self.annealing_steps > 0 and self.current_step > (self.swa_start_step + self.annealing_steps):
                # After annealing_steps, set SWA LR to a minimum value or keep it constant
                pass  # You can implement a strategy here if needed

        # Handle SWAG if enabled
        if self.use_swag and self.current_step >= self.swa_start_step:
            self.collect_swag_model()

        return final_loss.item()

    def end_epoch(self):
        # Step the scheduler
        self.scheduler.step()

        # Increment epoch counter
        self.epoch += 1

    def collect_swag_model(self):
        param_vector = torch.nn.utils.parameters_to_vector(self.model.parameters())
        if torch.isnan(param_vector).any():
            print("NaNs detected in parameter vector during SWAG collection!")
        
        n = self.num_models_collected
        if n == 0:
            self.mean.copy_(param_vector)
            self.sq_mean.copy_(param_vector**2)
        else:
            delta = param_vector - self.mean
            self.mean += delta / (n + 1)
            delta2 = param_vector**2 - self.sq_mean
            self.sq_mean += delta2 / (n + 1)
        if not self.no_cov_mat and n < self.max_num_models:
            idx = n % self.max_num_models
            self.cov_mat_sqrt[idx].copy_(param_vector - self.mean)
        self.num_models_collected += 1

    def swag_sample(self, scale=1.0, cov=False):
        mean = self.mean
        sq_mean = self.sq_mean
        var = sq_mean - mean**2
        
        # **Clamp variance to zero to avoid negative values**
        var = torch.clamp(var, min=0.0)
        
        std = torch.sqrt(var + 1e-30)
        
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("NaNs detected in SWAG mean or std!")
        
        z = torch.randn_like(mean)
        if cov and not self.no_cov_mat:
            c = self.cov_mat_sqrt[:min(self.num_models_collected, self.max_num_models)]
            z_cov = torch.randn(c.size(0), device=c.device)
            sample = mean + scale * (z * std + (c.t().matmul(z_cov) / (self.num_models_collected - 1)**0.5))
        else:
            sample = mean + scale * z * std
        
        if torch.isnan(sample).any():
            print("NaNs detected in SWAG sampled parameters!")
        
        torch.nn.utils.vector_to_parameters(sample, self.model.parameters())


    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            if self.use_laplace and hasattr(self, 'hessian_diagonal'):
                predictions = []
                
                # Clamp Hessian diagonals to avoid extreme values
                for i, var_p in enumerate(self.hessian_diagonal):
                    self.hessian_diagonal[i] = torch.clamp(var_p, min=1e-10)

                # Sample parameters multiple times
                for _ in range(self.num_laplace_samples):
                    sampled_params = []
                    for mean_p, var_p, p in zip(self.param_means, self.hessian_diagonal, self.model.parameters()):
                        var_p = torch.clamp(var_p, min=1e-10)
                        std = (1.0 / (var_p + 1e-6))**0.5
                        std = torch.clamp(std, max=0.001)
                        noise = torch.randn_like(std)
                        sampled_p = mean_p + noise * std
                        sampled_params.append(sampled_p)

                    # Backup current params
                    backup_params = [p.detach().clone() for p in self.model.parameters()]
                    for p, sp in zip(self.model.parameters(), sampled_params):
                        p.copy_(sp)

                    pred = self.model(batch)

                    # If pred is valid, store it
                    # (If you want to skip NaN predictions, you could check here and skip them)
                    predictions.append(pred.detach().cpu())

                    # Restore original parameters
                    for p, bp in zip(self.model.parameters(), backup_params):
                        p.copy_(bp)

                # Handle cases with zero or one valid sample
                if len(predictions) == 0:
                    # No valid samples, return NaNs
                    preds_mean = torch.full_like(batch.energy, float('nan'))
                    preds_var = torch.full_like(batch.energy, float('nan'))
                    return preds_mean, preds_var

                if len(predictions) == 1:
                    # Only one sample, variance is zero
                    preds_mean = predictions[0]
                    preds_var = torch.zeros_like(preds_mean)
                    return preds_mean, preds_var

                # Multiple samples: compute mean and var
                predictions_tensor = torch.stack(predictions)
                preds_mean = predictions_tensor.mean(dim=0)
                preds_var = predictions_tensor.var(dim=0, unbiased=False)
                return preds_mean, preds_var

            elif self.use_swag and self.num_models_collected > 0:
                self.swag_sample(scale=1.0, cov=not self.no_cov_mat)
                pred = self.model(batch)
                return pred, None
            else:
                pred = self.model(batch)
                return pred, None

    
    def finalize_laplace(self):
        if self.use_laplace:
            self.hessian_diagonal = []
            print("total batches: ", self.total_batches)
            for sq_grad in self.accumulated_squared_gradients:
                h_diag = (sq_grad / self.total_batches) + self.prior_precision
                self.hessian_diagonal.append(h_diag)
            self.param_means = [p.detach().clone() for p in self.model.parameters()]
            self.accumulated_squared_gradients = None

           
    def finalize(self):
        # Finalize SWA
        if self.use_swa:
            if not hasattr(self, 'train_loader'):
                raise ValueError("Train loader not set. Please assign train_loader to the trainer.")
            update_bn(self.train_loader, self.averaged_model)
            self.model = self.averaged_model.module  # Use the averaged model for evaluation
            print("SWA has been finalized and batch norms updated.")

        # Finalize SWAG
        if self.use_swag:
            # SWAG does not require additional finalization
            print("SWAG has been finalized.")

        # Finalize Laplace
        if self.use_laplace:
            self.finalize_laplace()
            print("Laplace Approximation finalized.")
