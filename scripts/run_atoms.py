"""Common script for training and predicting with atoms data."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.cli import LightningArgumentParser
from jsonargparse import ActionConfigFile, ActionYesNo
from lightning.pytorch.callbacks import StochasticWeightAveraging

import _atomgnn  # noqa: F401
import atomgnn.data.ase
import atomgnn.data.transforms
import atomgnn.models.schnet


class LitData(L.LightningDataModule):
    """Atoms data."""

    def __init__(
        self,
        dataset: str,
        target_property: str | None = None,
        energy_property: str = "energy",
        forces_property: str = "forces",
        magmoms_property: str = "magmoms",
        splits: str = "",
        batch_size: int = 10,
        cutoff: float = 5.0,
        pbc: bool = False,
        add_zero_forces: bool = False,
        nodewise_scaling: bool = True,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        """Initialize data module.

        Args:
            dataset: Path to dataset file.
            target_property: The target property in the dataset.
            energy_property: The energy property in the dataset.
            forces_property: The forces property in the dataset.
            magmoms_property: The magnetic moments property in the dataset.
            splits: Path to data splits file. Default is an unshuffled 80/10/10 split.
            batch_size: Batch size for data loaders.
            cutoff: Cutoff distance for edges and interactions.
            pbc: Enable periodic boundary conditions (pbc).
            add_zero_forces: Add zero forces with the same shape as positions.
            nodewise_scaling: Scale target per node instead of per graph.
            num_workers: Number of workers for data loaders.
            persistent_workers: Enable persistent workers in data loaders.
        """
        super().__init__()
        self.dataset_path = dataset
        self.target_property = target_property
        self.energy_property = energy_property
        self.forces_property = forces_property
        self.magmoms_property = magmoms_property
        self.splits_path = splits
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.pbc = pbc
        self.add_zero_forces = add_zero_forces
        self.nodewise_scaling = nodewise_scaling
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage):
        # Dataset
        transform = atomgnn.data.transforms.Compose([
            atomgnn.data.ase.AseDbAtomsRowToAtomsDataTransform(
                target_property=self.target_property, energy_property=self.energy_property,
                forces_property=self.forces_property, magmoms_property=self.magmoms_property,
                pbc=self.pbc),
            atomgnn.data.transforms.AddEdgesWithinCutoffDistanceTransform(
                cutoff=self.cutoff, pbc=self.pbc),
        ])
        if self.add_zero_forces:
            transform.transforms.append(atomgnn.data.transforms.AddZeroForcesTransform())
        self.dataset = atomgnn.data.ase.AseDbDataset(
            asedb_path=self.dataset_path,
            transform=transform,
        )
        # Data splits
        if self.splits_path:
            with open(self.splits_path, "r") as f:
                self.splits = json.load(f)
            if "validation" in self.splits:
                # Rename validation to val to be compatible with old split files
                self.splits["val"] = self.splits.pop("validation")
        else:
            datalen = len(self.dataset)
            tenth = datalen // 10
            index = torch.arange(datalen).tolist()
            self.splits = {
                "train": index[:-2*tenth],
                "val": index[-2*tenth:-tenth],
                "test": index[-tenth:],
            }
        if stage == "train":
            assert all(k in self.splits.keys() for k in ["train", "val"]), \
                "Train stage requires 'train' and 'val' splits."
        elif stage == "predict":
            assert any(k in self.splits.keys() for k in ["val", "test"]), \
                "Predict stage requires 'val' and/or 'test' splits."

    def train_dataloader(self):
        assert "train" in self.splits, "No 'train' split in splits."
        train_data = torch.utils.data.Subset(self.dataset, self.splits["train"])
        return torch.utils.data.DataLoader(
            train_data,
            self.batch_size,
            sampler=torch.utils.data.RandomSampler(train_data),
            collate_fn=atomgnn.data.data.collate_data,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        if "val" not in self.splits:
            return None
        val_data = torch.utils.data.Subset(self.dataset, self.splits["val"])
        return torch.utils.data.DataLoader(
            val_data,
            self.batch_size,
            collate_fn=atomgnn.data.data.collate_data,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        if "test" not in self.splits:
            return None
        test_data = torch.utils.data.Subset(self.dataset, self.splits["test"])
        return torch.utils.data.DataLoader(
            test_data,
            self.batch_size,
            collate_fn=atomgnn.data.data.collate_data,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return {"val": self.val_dataloader(), "test": self.test_dataloader()}

    def get_stats(self, split="train", max_batches=-1):
        """Compute dataset statistics.

        If self.nodewise_scaling, compute statistics about the target per node instead of per graph.

        Args:
            split: Dataset split to compute statistics for (train, val, test).
            max_batches: Maximum number of batches used to compute the statistics.
        Returns:
            A dictionary with dataset statistics.
        """
        if split == "train":
            dataloader = self.train_dataloader()
        elif split == "val":
            dataloader = self.val_dataloader()
        elif split == "test":
            dataloader = self.test_dataloader()
        else:
            raise ValueError(f"Unknown split: {split}")
        target_property = self.target_property or "energy"  # Use energy if target property is falsy
        # Compute stats
        for i, batch in enumerate(dataloader):
            y = batch.energy if target_property == "energy" else batch.targets
            assert y.ndim == 2 and y.shape[0] == batch.num_data, "Unexpected target shape"
            y = y / batch.num_nodes.unsqueeze(1) if self.nodewise_scaling else y  # Nodewise targets
            if i == 0:  # init from first batch
                num_targets = y.shape[1]
                y_bias = torch.mean(y, dim=0)  # Use a bias to avoid overflow
                running_y = torch.zeros(num_targets, dtype=torch.double)
                running_y2 = torch.zeros(num_targets, dtype=torch.double)
                running_num_data = 0
                running_num_nodes = 0
                running_num_edges = 0
            y = y - y_bias
            running_y += torch.sum(y, dim=0)
            running_y2 += torch.sum(torch.square(y), dim=0)
            running_num_nodes += torch.sum(batch.num_nodes).item()
            running_num_edges += torch.sum(batch.num_edges).item()
            running_num_data += batch.num_data
            if i + 1 == max_batches:
                break
        y_mean = running_y / running_num_data
        y_var = running_y2 / running_num_data - torch.square(y_mean)  # Var(Y) = E[Y^2] - E[Y]^2
        y_std = torch.sqrt(y_var)
        y_mean += y_bias  # Add bias back
        return {
            "target_mean": y_mean.type(torch.get_default_dtype()),
            "target_std": y_std.type(torch.get_default_dtype()),
            "avg_num_nodes": running_num_nodes / running_num_data,
            "avg_num_edges": running_num_edges / running_num_data,
            "avg_num_neighbors": running_num_edges / running_num_nodes,
        }

class CsvPredictionWriter(L.pytorch.callbacks.BasePredictionWriter):
    """Write predictions to CSV file with one line per data point.

    The predictions are expected to be a dict of tensors with the 'num_data'
    and 'num_nodes' keys in addition to arbitrary keys, that will be used as
    column labels, and values with either 'num_data' or 'num_nodes' elements.

    CSV file format:
        Columns are separated by a semicolon (comma is reserved for lists).
        Tensors are stored as (nested) lists of comma-separated values.
        Lists can be parsed with 'json.loads' and converted back to tensors.
    """

    def __init__(self, output_dir, name):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.name = name

    def on_predict_start(self, trainer, pl_module):
        # Reset
        self.file = None
        self.keys = None

    def _get_or_create_file(self, keys=None):
        # Create file if it does not exist
        if self.file is None:
            file_path = self.output_dir / f"predictions_{self.name}.csv"
            assert not file_path.exists(), f"File already exists: {file_path}"
            self.file = open(file_path, "w")
            if keys:
                self.keys = keys
                self.file.write(";".join(keys))
                self.file.write("\n")
        # Check that keys are consistent with the current file
        assert (keys is None and self.keys is None) or set(keys) == set(self.keys)
        return self.file

    def write_on_batch_end(
        self, trainer, pl_module, predictions, batch_indices, batch, batch_idx, dataloader_idx
    ):
        assert isinstance(predictions, dict), "Prediction writer expects a dict of tensors"
        num_nodes_total = torch.sum(batch.num_nodes).item()
        slices = [0] + torch.cumsum(batch.num_nodes, dim=0).tolist()
        lengths = {batch.num_data, num_nodes_total}  # Acceptable lengths are graph- and node-wise
        predictions = {k: v.squeeze() for k, v in predictions.items()}
        file = self._get_or_create_file(list(predictions.keys()))
        assert all(len(v) in lengths and k in self.keys for k, v in predictions.items())
        # Write one line per data point
        for i in range(batch.num_data):
            line = []
            for k in self.keys:
                v = predictions[k]
                if len(v) == batch.num_data:  # Graph-wise predictions
                    line.append(v[i].tolist())
                else:  # Node-wise predictions
                    assert len(v) == num_nodes_total
                    line.append(v[slices[i]:slices[i+1]].tolist())
            # Write line without whitespace
            file.write(";".join(str(v).replace(" ", "") for v in line))
            file.write("\n")

    def on_predict_end(self, trainer, pl_module):
        self.file.close()


def train(cfg, lit_data_cls, lit_model_cls, trainer):
    logging.info("Start train.")
    # Setup data
    data = lit_data_cls(**vars(cfg.data))
    data.setup("train")
    # Save splits
    with open(Path(trainer.log_dir) / "data_splits.json", "w") as f:
        json.dump(data.splits, f)
    # Compute scale and offset parameters from training data
    # TODO: max_batches is hardcoded
    # TODO: Could do this in the model as part of the training loop instead?
    stats = data.get_stats(split="train", max_batches=100)
    assert stats["target_std"].ndim == 1 and stats["target_mean"].ndim == 1, \
        "Multidimensional target is not supported"
    if trainer.is_global_zero:
        logging.info(f"data stats: {stats}")
    # Setup model
    model_kwargs = vars(cfg.model)
    model_kwargs.update({
        "_output_scale": stats["target_std"].item(),
        "_output_offset": stats["target_mean"].item(),
        "_nodewise_offset": data.nodewise_scaling,
        "_avg_num_neighbors": stats["avg_num_neighbors"],
    })
    model = lit_model_cls(**model_kwargs)
    if cfg.compile:
        model = torch.compile(model)
    # Option to resume training from model checkpoint
    ckpt_path = Path(cfg.checkpoint) if cfg.checkpoint else None
    assert ckpt_path is None or ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    # Run training loop
    trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)
    # Evaluate
    val_metrics = trainer.validate(ckpt_path="best", datamodule=data)
    if trainer.is_global_zero:
        logging.info(val_metrics)
    if "test" in data.splits:
        test_metrics = trainer.test(ckpt_path="best", datamodule=data)
        if trainer.is_global_zero:
            logging.info(test_metrics)


def predict(cfg, lit_data_cls, lit_model_cls, trainer):
    logging.info("Start predict.")
    # Setup data
    data = lit_data_cls(**vars(cfg.data))
    data.setup("predict")
    dataloaders = data.predict_dataloader()  # Collection of dataloaders
    # Setup model
    if not cfg.train:
        assert cfg.checkpoint, "Missing required argument: --checkpoint"
        ckpt_path = Path(cfg.checkpoint)
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        if cfg.trainer.use_swag:
            model = LitSWAGPaiNNModel.load_from_checkpoint(
                ckpt_path,
                swa_start=cfg.trainer.swag_swa_start,
                max_num_models=cfg.trainer.swag_max_num_models,
                no_cov_mat=cfg.trainer.no_cov_mat,
            )
        else:
            model = lit_model_cls.load_from_checkpoint(ckpt_path)
    else:
        ckpt_path = Path(trainer.log_dir) / "checkpoints/best.ckpt"
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        if cfg.trainer.use_swag:
            model = LitSWAGPaiNNModel.load_from_checkpoint(
                ckpt_path,
                swa_start=cfg.trainer.swag_swa_start,
                max_num_models=cfg.trainer.swag_max_num_models,
                no_cov_mat=cfg.trainer.no_cov_mat,
            )
        else:
            model = lit_model_cls.load_from_checkpoint(ckpt_path)
    if cfg.compile:
        model = torch.compile(model)
    # Predict
    if cfg.trainer.use_swag:
        num_samples = cfg.num_swag_samples if hasattr(cfg, 'num_swag_samples') else 30
        all_predictions = []
        for _ in range(num_samples):
            predictions = []
            for name, dataloader in dataloaders.items():
                if dataloader is None:
                    continue
                prediction_writer = CsvPredictionWriter(trainer.log_dir, name=name)
                predicter = L.Trainer(
                    accelerator=cfg.trainer.accelerator,
                    logger=False,
                    callbacks=[prediction_writer],
                    inference_mode=False,
                )
                preds = predicter.predict(model, dataloader, return_predictions=True)
                predictions.append(preds)
            all_predictions.append(predictions)
        # Aggregate predictions to compute mean and variance
        # Implement code to compute mean and uncertainty
        aggregated_predictions = {}
        for name in dataloaders.keys():
            preds_list = [preds[name] for preds in all_predictions]
            # Concatenate predictions
            concatenated_preds = [torch.cat([batch_pred for batch_pred in preds_batch], dim=0) for preds_batch in zip(*preds_list)]
            # Compute mean and std
            mean_preds = torch.mean(torch.stack(concatenated_preds), dim=0)
            std_preds = torch.std(torch.stack(concatenated_preds), dim=0)
    else:
        for name, dataloader in dataloaders.items():
            if dataloader is None:
                logging.info(f"Skip split: {name}")
                continue
            else:
                logging.info(f"Predict split: {name}")
                prediction_writer = CsvPredictionWriter(trainer.log_dir, name=name)
                limit_predict_batches = 2 if cfg.smoketest else 1.0
                predicter = L.Trainer(
                    accelerator=cfg.trainer.accelerator,
                    logger=False,
                    callbacks=[prediction_writer],
                    limit_predict_batches=limit_predict_batches,
                    inference_mode=False,
                )
                predicter.predict(model, dataloader, return_predictions=False)

def configure_trainer(
    output_root_dir: str = "./output",
    name: str = "run_atoms",
    version: str = "",
    accelerator: str = "auto",
    devices: str | int = -1,
    nodes: int = 1,
    precision: str | None = None,  # Default value is set by Lightning
    matmul_precision: str = "high",  # Default value 'high' to utilize tensor cores
    max_steps: int = 1_000_000,
    log_steps: int = 10_000,
    grad_accum: int = 1,
    grad_clip: float | None = None,
    wandb: bool = False,
    enable_progress_bar: bool = False,
    profiler: str | None = None,
    device_stats_monitor: bool = False,
    learning_rate_monitor: bool = False,
    smoketest: bool = False,
    # Add SWA parameters
    use_swa: bool = False,
    swa_lrs: float | None = None,
    swa_epoch_start: float = 0.8,
    annealing_epochs: int = 10,
    annealing_strategy: str = "cos",
    # Add SWAG parameters
    use_swag: bool = False,
    swag_swa_start: float = 0.8,
    swag_max_num_models: int = 20,
    swag_constant_lr: float | None = None,
    no_cov_mat: bool = True,
) -> L.Trainer:
    """Trainer.

    Args:
        output_root_dir: Output is saved in 'output_root_dir/name/version/'.
        name: Job name.
        version: Job version (default is a timestamp).
        accelerator: Accelerator type: 'auto', 'gpu', 'cpu', etc.
        devices: Number of GPUs or -1 for all available or 'auto' for automatic detection.
        nodes: Number of nodes for distributed training.
        precision: Set floating point precision. See Lightning documentation for details.
        matmul_precision: Set matrix multiplication precision: 'highest', 'high', 'medium'.
        max_steps: Maximum number of training steps.
        log_steps: Logging and validation step frequency.
        grad_accum: Number of batches to accumulate before stepping the optimizer.
        grad_clip: The value at which to clip gradients.
        wandb: Enable wandb logger.
        enable_progress_bar: Show the progress bar.
        profiler: Profiler type: 'simple', 'advanced'.
        device_stats_monitor: Enable device stats monitor.
        learning_rate_monitor: Enable learning rate monitor.
        smoketest: Use a small number of steps for testing.
    """
    # Set matmul precision
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    # Loggers
    tensorboard_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=output_root_dir, name=name, version=version)
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir=output_root_dir, name=name, version=version)
    loggers = [tensorboard_logger, csv_logger]
    if wandb:
        loggers.append(L.pytorch.loggers.WandbLogger(
            save_dir=output_root_dir, name=name, version=version, project=name))
    # Assert all loggers have the same version
    assert len(set(str(logger.version) for logger in loggers)) == 1
    # Callbacks
    callbacks: list[L.pytorch.callbacks.Callback] = []
    callbacks.append(
        L.pytorch.callbacks.ModelCheckpoint(filename="best", monitor="val_loss", mode="min"))
    callbacks.append(
        L.pytorch.callbacks.ModelCheckpoint(filename="{step}", save_last=True))
    if device_stats_monitor:
        callbacks.append(L.pytorch.callbacks.DeviceStatsMonitor())
    if learning_rate_monitor:
        callbacks.append(L.pytorch.callbacks.LearningRateMonitor())
    # Add SWA callback if enabled
    if use_swa:
        from lightning.pytorch.callbacks import StochasticWeightAveraging
        swa_callback = StochasticWeightAveraging(
            swa_lrs=swa_lrs,
            swa_epoch_start=swa_epoch_start,
            annealing_epochs=annealing_epochs,
            annealing_strategy=annealing_strategy,
        )
        callbacks.append(swa_callback)
    if use_swag:
        # SWAG is handled in the model, so no need for a callback here
        pass
    # Trainer settings
    limit_train_batches = 2 if smoketest else 1.0
    limit_val_batches = 2 if smoketest else 1.0
    limit_test_batches = 2 if smoketest else 1.0
    limit_predict_batches = 2 if smoketest else 1.0
    max_steps = 4 if smoketest else max_steps
    log_steps = 2 if smoketest else log_steps
    # Trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        strategy="auto",
        devices=1 if accelerator == "cpu" else devices,
        num_nodes=nodes,
        precision=precision,  # type: ignore
        logger=loggers,
        callbacks=callbacks,
        # fast_dev_run=smoketest,  # Disables loggers and therefore log_dir is not set
        max_steps=max_steps,
        val_check_interval=log_steps,
        log_every_n_steps=log_steps,
        check_val_every_n_epoch=None,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        limit_predict_batches=limit_predict_batches,
        accumulate_grad_batches=grad_accum,
        gradient_clip_val=grad_clip,
        enable_progress_bar=enable_progress_bar,
        profiler=profiler,
        inference_mode=False,  # Allow to enable grad for computing forces
        # default_root_dir=args.output_dir,
    )
    return trainer


def run(cli, lit_model_cls, lit_data_cls=LitData):
    """Run script."""
    # Parse and check configuration
    cfg = cli.parse_args()
    assert cfg.dataset is not None, "Missing required argument: --dataset"
    # Setup loggers and trainer
    trainer = configure_trainer(**vars(cfg.trainer))
    # Create log_dir (normally the log_dir is created when trainer.fit is called).
    # TODO: Maybe create the log_dir before configuring the trainer and later assert they are equal?
    log_dir = Path(trainer.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    # Configure logging
    log_file_handler = logging.FileHandler(log_dir / "script.log", mode="w")
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{trainer.global_rank}] [%(levelname)s] %(message)s",
        handlers=[
            log_file_handler,
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logging.getLogger("pytorch_lightning").addHandler(log_file_handler)
    logging.info("Ready to work!")
    if trainer.is_global_zero:
        logging.info(f"log_dir: {log_dir}")
        logging.info(f"cfg:\n{cli.dump(cfg)}")
        # Save configuration in log_dir
        cli.save(cfg, log_dir / "config.yaml")
    # Train
    if cfg.train:
        logging.info("Start train.")
        # Setup data
        data = lit_data_cls(**vars(cfg.data))
        data.setup("train")
        # Save splits
        with open(Path(trainer.log_dir) / "data_splits.json", "w") as f:
            json.dump(data.splits, f)
        # Compute scale and offset parameters from training data
        stats = data.get_stats(split="train", max_batches=100)
        # ... (assertions and logging)
        # Setup model
        model_kwargs = vars(cfg.model)
        model_kwargs.update({
            "_output_scale": stats["target_std"].item(),
            "_output_offset": stats["target_mean"].item(),
            "_nodewise_offset": data.nodewise_scaling,
            "_avg_num_neighbors": stats["avg_num_neighbors"],
        })
        
        # Instantiate the appropriate model
        if cfg.use_swag:
            model_kwargs.update({
                "swa_start": cfg.swag_swa_start,
                "max_num_models": cfg.swag_max_num_models,
                "no_cov_mat": cfg.no_cov_mat,
                "swag_constant_lr": cfg.swag_constant_lr,  # Pass the new LR argument
            })

        model = lit_model_cls(**model_kwargs)

        if cfg.compile:
            model = torch.compile(model)
        # Option to resume training from model checkpoint
        ckpt_path = Path(cfg.checkpoint) if cfg.checkpoint else None
        assert ckpt_path is None or ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        # Run training loop
        trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)
        # Evaluate
        val_metrics = trainer.validate(ckpt_path="best", datamodule=data)
        if trainer.is_global_zero:
            logging.info(val_metrics)
        if "test" in data.splits:
            test_metrics = trainer.test(ckpt_path="best", datamodule=data)
            if trainer.is_global_zero:
                logging.info(test_metrics)
        # train(cfg, lit_data_cls, lit_model_cls, trainer)
    # Predict
    if cfg.predict:
        predict(cfg, lit_data_cls, lit_model_cls, trainer)
    logging.info("Job's done!")


def configure_cli(default_job_name, add_trainer_args=True, add_data_args=True):
    """Build CLI argument parser with common arguments."""
    cli = LightningArgumentParser()
    cli.add_argument('--config', action=ActionConfigFile)
    # Script tasks
    cli.add_argument("--train", action=ActionYesNo, help="Run training task.")
    cli.add_argument("--predict", action=ActionYesNo, help="Run prediction task.")
    cli.add_argument("--compile", action=ActionYesNo, help="Compile model with torch.compile.")
    cli.add_argument("--smoketest", action=ActionYesNo,
                     help="Quickly test that the script runs without errors.")
    cli.add_argument("--checkpoint",
                     help="Path to checkpoint used to resume training or predict.")
    # Loggers and output
    cli.add_argument("--output_root_dir",
                     default=os.getenv("JOB_OUTPUT_ROOT_DIR", "./output"),
                     help="Output is saved in 'output_root_dir/name/version/'")
    cli.add_argument("--name", default=os.getenv("JOB_NAME", default_job_name), help="Job name.")
    cli.add_argument("--version",
                     default=os.getenv("JOB_VERSION", datetime.now().strftime("%Y%m%d_%H%M%S")),
                     help="Job version (default is a timestamp).")
    # SWAG Arguments
    cli.add_argument("--use_swag", type=bool, default=False, help="Enable SWAG.")
    cli.add_argument("--swag_swa_start", type=float, default=0.8, help="When to start SWAG.")
    cli.add_argument("--swag_max_num_models", type=int, default=20, help="Max number of models to collect.")
    cli.add_argument("--swag_constant_lr", type=float, default=None, help="Constant LR value to set after SWA start (if enabled).")
    cli.add_argument("--no_cov_mat", type=bool, default=True, help="Do not store covariance matrix in SWAG.")
    # SAM Arguments
    cli.add_argument("--use_sam", type=bool, default=False, help="Enable SAM.")  # Add this line
    cli.add_argument("--sam_rho", type=float, default=0.05, help="SAM perturbation parameter.")  # Add this line
    # ASAM Arguments
    cli.add_argument("--use_asam", type=bool, default=False, help="Enable ASAM.")
    # HR arguments
    cli.add_argument("--heteroscedastic", type=bool, default=False, help="Enable heteroscedastic regression.")
    cli.add_argument("--use_laplace", type=bool, default=False, help="Apply Laplace approximation for Bayesian posterior.")
    cli.add_argument('--num_laplace_samples', type=int, default=10,
                     help='Number of weight samples during Laplace prediction.')
    if add_trainer_args:
        # cli = pl.Trainer.add_argparse_args(cli)  # Add all trainer arguments to cli
        cli.add_lightning_class_args(configure_trainer, "trainer")
        # Link SWAG arguments
        cli.link_arguments("use_swag", "trainer.use_swag", apply_on="parse")
        cli.link_arguments("swag_swa_start", "trainer.swag_swa_start", apply_on="parse")
        cli.link_arguments("swag_max_num_models", "trainer.swag_max_num_models", apply_on="parse")
        cli.link_arguments("swag_constant_lr", "trainer.swag_constant_lr", apply_on="parse")
        cli.link_arguments("no_cov_mat", "trainer.no_cov_mat", apply_on="parse")
        # Original
        cli.link_arguments("output_root_dir", "trainer.output_root_dir", apply_on="parse")
        cli.link_arguments("name", "trainer.name", apply_on="parse")
        cli.link_arguments("version", "trainer.version", apply_on="parse")
        cli.link_arguments("smoketest", "trainer.smoketest", apply_on="parse")
    # Data
    if add_data_args:
        cli.add_argument("--dataset", help="Path to dataset file.")  # Required
        cli.add_lightning_class_args(LitData, "data")
        cli.link_arguments("dataset", "data.dataset", apply_on="parse")
    return cli


# Example main function
def main():
    cli = configure_cli(default_job_name="run_atoms")
    # raise NotImplementedError("This script is not fully implemented.")
    run(cli, lit_model_cls=None)


if __name__ == '__main__':
    main()
