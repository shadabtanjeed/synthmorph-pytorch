"""Project configuration for SynthMorph training and validation."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
import torch


is_windows = os.name == "nt"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use true/false.")


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return _default_device()
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[Config] CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested_device


# Reproducibility
seed = 42

# Runtime and optimization
device = _default_device()
batch_size = 4
num_workers = 0
pin_memory = False
prefetch_factor = 2
persistent_workers = num_workers > 0
num_epochs = 250
learning_rate = 1e-4
optimizer_type = "AdamW"
weight_decay = 1e-5
regularization_weight = 1.0
use_amp = device == "cuda"
amp_dtype = "bfloat16"
early_stopping_patience = 15
early_stopping_metric = "val"
early_stopping_min_delta = 1e-3

# Optional debug tracing
debug_training = True
debug_every_n_epochs = 5
debug_batches_per_epoch = 4

# Data and label settings
image_size = (160, 192, 224)
train_dataset_size = 500
train_num_classes = 26
val_num_classes = 35
ignore_label = 0

# Integration / deformation settings
integration_steps = 7
# Optional multiplier applied to the predicted velocity field before integration.
# Use 1.0 for baseline behavior, or try 10.0 as a jumpstart if deformations stay tiny.
flow_scale = 10.0

# Validation schedule and data
validate_every = 5
val_data_dir = "/kaggle/neurite-oasis-split/val"
val_image_filename = "aligned_norm.nii.gz"
val_label_filename = "aligned_seg35.nii.gz"

# Artifact output
output_dir = os.path.join("outputs", "train")
best_model_filename = "best_model.pt"
loss_plot_filename = "loss_curve.png"
dice_plot_filename = "dice_curve.png"


# Synthetic generator config used by training dataset
generator_config = {
    "full_image_size": image_size,
    "num_classes": train_num_classes,
    "device": device,
    "integration_steps": integration_steps,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SynthMorph training configuration overrides",
    )

    # Reproducibility
    parser.add_argument("--seed", type=int)

    # Runtime and optimization
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--pin-memory", type=_parse_bool)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer-type", choices=["Adam", "AdamW", "adam", "adamw"])
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--regularization-weight", type=float)
    parser.add_argument("--use-amp", type=_parse_bool)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16"])
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-metric", choices=["val", "train"])
    parser.add_argument("--early-stopping-min-delta", type=float)

    # Optional debug tracing
    parser.add_argument("--debug-training", type=_parse_bool)
    parser.add_argument("--debug-every-n-epochs", type=int)
    parser.add_argument("--debug-batches-per-epoch", type=int)

    # Data and label settings
    parser.add_argument("--image-size", type=int, nargs=3, metavar=("D", "H", "W"))
    parser.add_argument("--train-dataset-size", type=int)
    parser.add_argument("--train-num-classes", type=int)
    parser.add_argument("--val-num-classes", type=int)
    parser.add_argument("--ignore-label", type=int)

    # Integration / deformation settings
    parser.add_argument("--integration-steps", type=int)
    parser.add_argument("--flow-scale", type=float)

    # Validation schedule and data
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--val-data-dir", type=str)
    parser.add_argument("--val-image-filename", type=str)
    parser.add_argument("--val-label-filename", type=str)

    # Artifact output
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--best-model-filename", type=str)
    parser.add_argument("--loss-plot-filename", type=str)
    parser.add_argument("--dice-plot-filename", type=str)

    return parser


def apply_cli_overrides(args: argparse.Namespace) -> None:
    global seed
    global device, batch_size, num_workers, pin_memory, prefetch_factor
    global persistent_workers, num_epochs, learning_rate, optimizer_type
    global weight_decay, regularization_weight, use_amp, amp_dtype
    global early_stopping_patience, early_stopping_metric, early_stopping_min_delta
    global debug_training, debug_every_n_epochs, debug_batches_per_epoch
    global image_size, train_dataset_size, train_num_classes, val_num_classes
    global ignore_label, integration_steps, flow_scale
    global validate_every, val_data_dir, val_image_filename, val_label_filename
    global output_dir, best_model_filename, loss_plot_filename, dice_plot_filename
    global generator_config

    use_amp_set_by_cli = args.use_amp is not None

    if args.seed is not None:
        seed = args.seed

    if args.device is not None:
        device = _resolve_device(args.device)
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.num_workers is not None:
        num_workers = args.num_workers
    if args.pin_memory is not None:
        pin_memory = args.pin_memory
    if args.prefetch_factor is not None:
        prefetch_factor = args.prefetch_factor
    if args.num_epochs is not None:
        num_epochs = args.num_epochs
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    if args.optimizer_type is not None:
        optimizer_type = args.optimizer_type
    if args.weight_decay is not None:
        weight_decay = args.weight_decay
    if args.regularization_weight is not None:
        regularization_weight = args.regularization_weight
    if args.use_amp is not None:
        use_amp = args.use_amp
    if args.amp_dtype is not None:
        amp_dtype = args.amp_dtype
    if args.early_stopping_patience is not None:
        early_stopping_patience = args.early_stopping_patience
    if args.early_stopping_metric is not None:
        early_stopping_metric = args.early_stopping_metric
    if args.early_stopping_min_delta is not None:
        early_stopping_min_delta = args.early_stopping_min_delta

    if args.debug_training is not None:
        debug_training = args.debug_training
    if args.debug_every_n_epochs is not None:
        debug_every_n_epochs = args.debug_every_n_epochs
    if args.debug_batches_per_epoch is not None:
        debug_batches_per_epoch = args.debug_batches_per_epoch

    if args.image_size is not None:
        image_size = tuple(args.image_size)
    if args.train_dataset_size is not None:
        train_dataset_size = args.train_dataset_size
    if args.train_num_classes is not None:
        train_num_classes = args.train_num_classes
    if args.val_num_classes is not None:
        val_num_classes = args.val_num_classes
    if args.ignore_label is not None:
        ignore_label = args.ignore_label

    if args.integration_steps is not None:
        integration_steps = args.integration_steps
    if args.flow_scale is not None:
        flow_scale = args.flow_scale

    if args.validate_every is not None:
        validate_every = args.validate_every
    if args.val_data_dir is not None:
        val_data_dir = args.val_data_dir
    if args.val_image_filename is not None:
        val_image_filename = args.val_image_filename
    if args.val_label_filename is not None:
        val_label_filename = args.val_label_filename

    if args.output_dir is not None:
        output_dir = args.output_dir
    if args.best_model_filename is not None:
        best_model_filename = args.best_model_filename
    if args.loss_plot_filename is not None:
        loss_plot_filename = args.loss_plot_filename
    if args.dice_plot_filename is not None:
        dice_plot_filename = args.dice_plot_filename

    optimizer_type = str(optimizer_type)
    if optimizer_type.lower() == "adamw":
        optimizer_type = "AdamW"
    elif optimizer_type.lower() == "adam":
        optimizer_type = "Adam"

    early_stopping_metric = str(early_stopping_metric).lower()
    persistent_workers = num_workers > 0
    if not use_amp_set_by_cli:
        use_amp = device == "cuda"

    generator_config = {
        "full_image_size": image_size,
        "num_classes": train_num_classes,
        "device": device,
        "integration_steps": integration_steps,
    }


def configure_from_cli(cli_args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    parsed_args, unknown_args = parser.parse_known_args(
        list(cli_args) if cli_args is not None else None
    )

    if unknown_args:
        print(f"[Config] Ignoring unknown CLI args: {' '.join(unknown_args)}")

    apply_cli_overrides(parsed_args)
    return parsed_args
