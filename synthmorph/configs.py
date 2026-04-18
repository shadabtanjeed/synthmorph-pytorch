"""Project configuration for SynthMorph training and validation."""

from __future__ import annotations

import os
import torch


is_windows = os.name == "nt"


# Reproducibility
seed = 42

# Runtime and optimization
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8
num_workers = 0
pin_memory = False
prefetch_factor = 2
persistent_workers = num_workers > 0
num_epochs = 150
learning_rate = 1e-4
optimizer_type = "AdamW"
weight_decay = 1e-5
regularization_weight = 1.0
use_amp = device == "cuda"
amp_dtype = "bfloat16"
early_stopping_patience = 15

# Data and label settings
image_size = (160, 192, 224)
train_dataset_size = 100
train_num_classes = 26
val_num_classes = 35
ignore_label = 0

# Integration / deformation settings
integration_steps = 7

# Validation schedule and data
validate_every = 10
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
    "device": "cuda",  # Generate on CPU; H100 trains on GPU data transferred per batch
    "integration_steps": integration_steps,
}
