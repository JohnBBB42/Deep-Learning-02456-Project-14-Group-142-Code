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
