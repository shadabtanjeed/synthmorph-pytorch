"""Project configuration for SynthMorph training and validation."""

from __future__ import annotations

import os
import torch


is_windows = os.name == "nt"


# Reproducibility
seed = 42

# Runtime and optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
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
debug_training = False
debug_every_n_epochs = 5
debug_batches_per_epoch = 4

# Data and label settings
image_size = (160, 192, 224)
train_dataset_size = 500
train_num_classes = 26
val_num_classes = 35
ignore_label = 0

# Integration / deformation settings
integration_steps = 5
# Optional multiplier applied to the predicted velocity field before integration.
# Use 1.0 for baseline behavior, or try 10.0 as a jumpstart if deformations stay tiny.
flow_scale = 10.0

# Memory-aware soft Dice settings
dice_chunk_size = 4
dice_epsilon = 1e-5

# Patch-based processing settings
use_patch_based = True
patch_size = 64
patch_overlap = 16  # Overlap for stitching (default 1/4 of patch size)

# Validation schedule and data
validate_every = 5
val_data_dir = "/home/ndag/synthmorph/neurite-oasis-split/val"
val_image_filename = "aligned_norm.nii.gz"
val_label_filename = "aligned_seg35.nii.gz"

# Artifact output
output_dir = os.path.join("outputs", "train")
best_model_filename = "best_model.pt"
checkpoint_filename = "latest_checkpoint.pt"
checkpoint_every = 1
resume_checkpoint_path = ""
loss_plot_filename = "loss_curve.png"
dice_plot_filename = "dice_curve.png"


# Synthetic generator config used by training dataset
generator_config = {
    "full_image_size": image_size,
    "num_classes": train_num_classes,
    "device": device,
    "integration_steps": integration_steps,
}
