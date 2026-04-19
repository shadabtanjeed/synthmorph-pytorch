from __future__ import annotations

import math
import os
import random
from datetime import datetime
from pathlib import Path
from contextlib import nullcontext

import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthmorph import configs as config
from synthmorph.dataset import SynthMorphDataset
from synthmorph.loss import diffusion_loss
from synthmorph.network import SynthMorphUNet
from synthmorph.utils import create_integration_layer, PatchProcessor


_GRID_CACHE: dict[
    tuple[int, int, int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]
] = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    min_val = image.min()
    max_val = image.max()
    denom = torch.clamp(max_val - min_val, min=1e-6)
    return (image - min_val) / denom


def _get_grid_components(
    vector_field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, depth, height, width = vector_field.shape
    key = (depth, height, width, str(vector_field.device), vector_field.dtype)

    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    z = torch.linspace(
        -1.0, 1.0, depth, device=vector_field.device, dtype=vector_field.dtype
    )
    y = torch.linspace(
        -1.0, 1.0, height, device=vector_field.device, dtype=vector_field.dtype
    )
    x = torch.linspace(
        -1.0, 1.0, width, device=vector_field.device, dtype=vector_field.dtype
    )
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    base_grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)

    norm = torch.tensor(
        [
            2.0 / max(width - 1, 1),
            2.0 / max(height - 1, 1),
            2.0 / max(depth - 1, 1),
        ],
        device=vector_field.device,
        dtype=vector_field.dtype,
    ).view(1, 3, 1, 1, 1)

    _GRID_CACHE[key] = (base_grid, norm)
    return base_grid, norm


def field_to_sampling_grid(vector_field: torch.Tensor) -> torch.Tensor:
    batch_size = vector_field.shape[0]
    base_grid_single, norm = _get_grid_components(vector_field)
    base_grid = base_grid_single.expand(batch_size, -1, -1, -1, -1)

    normalized_field = vector_field * norm
    return base_grid + normalized_field.permute(0, 2, 3, 4, 1)


def warp_label_map_soft(
    label_map: torch.Tensor,
    integrated_field: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if label_map.dim() == 3:
        label_map = label_map.unsqueeze(0)

    label_map = torch.clamp(label_map.long(), 0, num_classes - 1)
    one_hot = F.one_hot(label_map, num_classes=num_classes)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).to(dtype=integrated_field.dtype)

    sampling_grid = field_to_sampling_grid(integrated_field)
    warped_probs = F.grid_sample(
        one_hot,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped_probs


def soft_dice_loss_from_warped_labels_chunked(
    fixed_label_map: torch.Tensor,
    moving_label_map: torch.Tensor,
    integrated_field: torch.Tensor,
    num_classes: int,
    ignore_label: int,
    chunk_size: int,
    epsilon: float,
) -> torch.Tensor:
    if fixed_label_map.dim() == 3:
        fixed_label_map = fixed_label_map.unsqueeze(0)
    if moving_label_map.dim() == 3:
        moving_label_map = moving_label_map.unsqueeze(0)

    fixed_label_map = torch.clamp(fixed_label_map.long(), 0, num_classes - 1)
    moving_label_map = torch.clamp(moving_label_map.long(), 0, num_classes - 1)

    if fixed_label_map.shape[0] != moving_label_map.shape[0]:
        raise ValueError(
            "Batch size mismatch between fixed and moving labels: "
            f"{fixed_label_map.shape[0]} vs {moving_label_map.shape[0]}"
        )
    if integrated_field.shape[0] != fixed_label_map.shape[0]:
        raise ValueError(
            "Batch size mismatch between labels and deformation field: "
            f"labels batch={fixed_label_map.shape[0]}, field batch={integrated_field.shape[0]}. "
            "If using patch-based forward, ensure it returns one field per input sample."
        )

    valid_classes = [
        class_index for class_index in range(num_classes) if class_index != ignore_label
    ]
    if not valid_classes:
        return integrated_field.new_tensor(0.0)

    sampling_grid = field_to_sampling_grid(integrated_field)
    dice_sum = integrated_field.new_tensor(0.0)
    class_count = 0

    fixed_expanded = fixed_label_map.unsqueeze(1)
    moving_expanded = moving_label_map.unsqueeze(1)

    for start in range(0, len(valid_classes), max(1, chunk_size)):
        class_chunk = valid_classes[start : start + max(1, chunk_size)]
        classes = torch.tensor(
            class_chunk,
            device=integrated_field.device,
            dtype=torch.long,
        ).view(1, -1, 1, 1, 1)

        moving_chunk = (moving_expanded == classes).to(dtype=integrated_field.dtype)
        warped_chunk = F.grid_sample(
            moving_chunk,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        fixed_chunk = (fixed_expanded == classes).to(dtype=integrated_field.dtype)

        intersection = (fixed_chunk * warped_chunk).sum(dim=(0, 2, 3, 4))
        fixed_volume = fixed_chunk.sum(dim=(0, 2, 3, 4))
        moving_volume = warped_chunk.sum(dim=(0, 2, 3, 4))

        dice_chunk = (2.0 * intersection + epsilon) / (
            fixed_volume + moving_volume + epsilon
        )
        dice_sum = dice_sum + dice_chunk.sum()
        class_count += len(class_chunk)

    mean_dice = dice_sum / max(class_count, 1)
    return 1.0 - mean_dice


def warp_intensity_map(
    image: torch.Tensor,
    integrated_field: torch.Tensor,
) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 4:
        image = image.unsqueeze(1)

    image = image.float()
    sampling_grid = field_to_sampling_grid(integrated_field)
    warped = F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.squeeze(1)


def forward_patch_based(
    model: SynthMorphUNet,
    fixed_image: torch.Tensor,
    moving_image: torch.Tensor,
    patch_processor: PatchProcessor,
    device: torch.device,
) -> torch.Tensor:
    """
    Forward pass on overlapping patches.
    Inputs: fixed_image [B, 1, D, H, W], moving_image [B, 1, D, H, W]
    Output: vector_field [B, 3, D, H, W]
    """
    if fixed_image.shape[0] != moving_image.shape[0]:
        raise ValueError(
            "Batch size mismatch between fixed and moving images: "
            f"{fixed_image.shape[0]} vs {moving_image.shape[0]}"
        )

    model_input = torch.cat([fixed_image, moving_image], dim=1)  # [B, 2, D, H, W]
    batch_size = model_input.shape[0]
    full_fields: list[torch.Tensor] = []
    use_activation_checkpointing = torch.is_grad_enabled()

    for batch_index in range(batch_size):
        sample_input = model_input[batch_index]  # [2, D, H, W]
        patches, positions = patch_processor.extract_patches(sample_input)

        vector_field_patches: list[torch.Tensor] = []
        for patch in patches:
            patch_batch = patch.unsqueeze(0)  # [1, 2, P, P, P]
            if use_activation_checkpointing:
                # Checkpoint each patch forward to avoid storing large intermediate activations.
                patch_field = checkpoint(model, patch_batch, use_reentrant=False)
            else:
                patch_field = model(patch_batch)  # [1, 3, P, P, P]
            vector_field_patches.append(patch_field.squeeze(0))  # [3, P, P, P]

        full_vector_field = patch_processor.stitch_patches(
            vector_field_patches,
            positions,
            device=device,
            dtype=model_input.dtype,
        )
        full_fields.append(full_vector_field.squeeze(0))  # [3, D, H, W]

    return torch.stack(full_fields, dim=0)  # [B, 3, D, H, W]


def resolve_run_output_dir(base_output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_parent = os.path.dirname(base_output_dir.rstrip(os.sep)) or "outputs"
    return os.path.join(base_parent, timestamp)


def _normalize_slice_for_plot(slice_2d: torch.Tensor) -> torch.Tensor:
    min_val = slice_2d.min()
    max_val = slice_2d.max()
    denom = torch.clamp(max_val - min_val, min=1e-6)
    return (slice_2d - min_val) / denom


def save_validation_samples(
    samples: list[dict],
    epoch: int,
    output_dir: str,
) -> None:
    if not samples:
        return

    vis_dir = os.path.join(output_dir, "val_samples")
    ensure_dir(vis_dir)
    save_path = os.path.join(vis_dir, f"epoch_{epoch:04d}.png")

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.5 * rows))
    if rows == 1:
        axes = [axes]

    for row_index, sample in enumerate(samples):
        fixed_slice = sample["fixed_slice"]
        moving_slice = sample["moving_slice"]
        warped_slice = sample["warped_slice"]

        axes[row_index][0].imshow(fixed_slice, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index][1].imshow(moving_slice, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index][2].imshow(warped_slice, cmap="gray", vmin=0.0, vmax=1.0)

        axes[row_index][0].set_title(
            f"Fixed\n{sample['fixed_name']}",
            fontsize=9,
        )
        axes[row_index][1].set_title(
            f"Moving\n{sample['moving_name']}",
            fontsize=9,
        )
        axes[row_index][2].set_title("Registered Moving", fontsize=9)

        for axis in axes[row_index]:
            axis.axis("off")

    fig.suptitle(f"Validation Registration Samples - Epoch {epoch}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def get_optimizer(
    parameters, optimizer_name: str, learning_rate: float, weight_decay: float
):
    optimizer_name_lower = optimizer_name.lower()
    if optimizer_name_lower == "adamw":
        return AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name_lower == "adam":
        return Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer_type: {optimizer_name}")


def load_nifti_tensor(
    path: Path, is_label: bool, target_size: tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    volume_np = nib.load(str(path)).get_fdata()
    tensor = torch.from_numpy(volume_np)

    if is_label:
        tensor = tensor.long().unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(tensor, size=target_size, mode="nearest")
        return resized.squeeze(0).squeeze(0).round().long().to(device)

    tensor = tensor.float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor, size=target_size, mode="trilinear", align_corners=True
    )
    resized = normalize_image(resized)
    return resized.squeeze(0).squeeze(0).to(device)


def discover_validation_patients() -> list[Path]:
    val_root = Path(config.val_data_dir)
    if not config.val_data_dir:
        return []
    if not val_root.exists():
        return []

    patients: list[Path] = []
    for entry in val_root.iterdir():
        if not entry.is_dir():
            continue
        image_path = entry / config.val_image_filename
        label_path = entry / config.val_label_filename
        if image_path.exists() and label_path.exists():
            patients.append(entry)
    return patients


def validate_validation_folder_structure() -> list[Path]:
    if config.validate_every <= 0:
        return []

    if not config.val_data_dir:
        raise ValueError(
            "Validation is enabled, but config.val_data_dir is empty. "
            "Set a valid validation folder path in synthmorph/configs.py."
        )

    val_root = Path(config.val_data_dir)
    if not val_root.exists():
        raise FileNotFoundError(
            f"Validation folder not found: {val_root}. "
            "Check config.val_data_dir in synthmorph/configs.py."
        )

    if not val_root.is_dir():
        raise NotADirectoryError(f"Validation path is not a directory: {val_root}.")

    patient_dirs = [entry for entry in val_root.iterdir() if entry.is_dir()]
    if len(patient_dirs) < 2:
        raise ValueError(
            "Validation folder format is invalid. "
            "Expected at least 2 patient subfolders under val_data_dir."
        )

    invalid_patients: list[str] = []
    valid_patients: list[Path] = []

    for patient_dir in sorted(patient_dirs):
        image_path = patient_dir / config.val_image_filename
        label_path = patient_dir / config.val_label_filename

        missing = []
        if not image_path.exists():
            missing.append(config.val_image_filename)
        if not label_path.exists():
            missing.append(config.val_label_filename)

        if missing:
            invalid_patients.append(f"{patient_dir.name}: missing {', '.join(missing)}")
            continue

        valid_patients.append(patient_dir)

    if invalid_patients:
        details = "\n  - " + "\n  - ".join(invalid_patients)
        raise ValueError(
            "Validation folder format is invalid. Each patient subfolder must contain "
            f"{config.val_image_filename} and {config.val_label_filename}."
            f"\nInvalid patient folders:{details}"
        )

    return valid_patients


def build_validation_pairs(patients: list[Path]) -> list[tuple[Path, Path]]:
    if len(patients) < 2:
        return []

    shuffled = patients[:]
    random.shuffle(shuffled)
    num_pairs = math.ceil(len(shuffled) / 2)
    pairs: list[tuple[Path, Path]] = []

    pair_index = 0
    while len(pairs) < num_pairs:
        fixed_patient = shuffled[pair_index % len(shuffled)]
        moving_patient = shuffled[(pair_index + 1) % len(shuffled)]
        if fixed_patient == moving_patient:
            pair_index += 1
            continue
        pairs.append((fixed_patient, moving_patient))
        pair_index += 2

    return pairs


def evaluate_on_validation(
    model: SynthMorphUNet,
    integration_layer,
    device: torch.device,
    epoch: int,
    output_dir: str,
    patch_processor: PatchProcessor | None = None,
) -> tuple[float, float]:
    patients = discover_validation_patients()
    pairs = build_validation_pairs(patients)

    if not pairs:
        return float("nan"), float("nan")

    model.eval()
    total_val_loss = 0.0
    total_val_dice = 0.0
    flow_scale = float(getattr(config, "flow_scale", 1.0))
    dice_chunk_size = int(getattr(config, "dice_chunk_size", 4))
    dice_epsilon = float(getattr(config, "dice_epsilon", 1e-5))
    num_vis_pairs = min(5, len(pairs))
    vis_pair_indices = set(random.sample(range(len(pairs)), num_vis_pairs))
    vis_samples: list[dict] = []

    with torch.no_grad():
        progress = tqdm(pairs, desc="Validation", leave=False)
        for pair_index, (fixed_patient, moving_patient) in enumerate(progress):
            fixed_image = load_nifti_tensor(
                fixed_patient / config.val_image_filename,
                is_label=False,
                target_size=config.image_size,
                device=device,
            )
            moving_image = load_nifti_tensor(
                moving_patient / config.val_image_filename,
                is_label=False,
                target_size=config.image_size,
                device=device,
            )
            fixed_label = load_nifti_tensor(
                fixed_patient / config.val_label_filename,
                is_label=True,
                target_size=config.image_size,
                device=device,
            )
            moving_label = load_nifti_tensor(
                moving_patient / config.val_label_filename,
                is_label=True,
                target_size=config.image_size,
                device=device,
            )

            model_input = torch.stack([fixed_image, moving_image], dim=0).unsqueeze(0)
            if patch_processor is not None:
                fixed_img_batch = fixed_image.unsqueeze(0)
                moving_img_batch = moving_image.unsqueeze(0)
                vector_field = forward_patch_based(
                    model,
                    fixed_img_batch,
                    moving_img_batch,
                    patch_processor,
                    device,
                )
            else:
                vector_field = model(model_input)
            effective_vector_field = vector_field * flow_scale
            integrated_field = integration_layer(effective_vector_field)

            val_similarity_loss = soft_dice_loss_from_warped_labels_chunked(
                fixed_label.unsqueeze(0),
                moving_label.unsqueeze(0),
                integrated_field,
                num_classes=config.val_num_classes,
                ignore_label=config.ignore_label,
                chunk_size=dice_chunk_size,
                epsilon=dice_epsilon,
            )
            val_smooth_loss = diffusion_loss(effective_vector_field)
            val_total_loss = (
                val_similarity_loss
                + float(config.regularization_weight) * val_smooth_loss
            )
            val_dice = 1.0 - val_similarity_loss

            if pair_index in vis_pair_indices:
                warped_moving_image = warp_intensity_map(
                    moving_image.unsqueeze(0),
                    integrated_field,
                ).squeeze(0)

                center_index = fixed_image.shape[0] // 2
                fixed_slice = _normalize_slice_for_plot(
                    fixed_image[center_index].detach().cpu()
                ).numpy()
                moving_slice = _normalize_slice_for_plot(
                    moving_image[center_index].detach().cpu()
                ).numpy()
                warped_slice = _normalize_slice_for_plot(
                    warped_moving_image[center_index].detach().cpu()
                ).numpy()

                vis_samples.append(
                    {
                        "fixed_slice": fixed_slice,
                        "moving_slice": moving_slice,
                        "warped_slice": warped_slice,
                        "fixed_name": fixed_patient.name,
                        "moving_name": moving_patient.name,
                    }
                )

            total_val_loss += float(val_total_loss.item())
            total_val_dice += float(val_dice.item())

            progress.set_postfix(
                loss=f"{val_total_loss.item():.4f}",
                dice=f"{val_dice.item():.4f}",
            )

    mean_val_loss = total_val_loss / len(pairs)
    mean_val_dice = total_val_dice / len(pairs)
    save_validation_samples(vis_samples, epoch=epoch, output_dir=output_dir)
    return mean_val_loss, mean_val_dice


def setup_logger(output_dir: str) -> str:
    """Create and initialize a log file."""
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "training_log.txt")

    # Write header if file is new
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("Epoch,Train_Loss,Train_Similarity_Loss,Train_Smooth_Loss,")
            f.write("Val_Loss,Val_Dice\n")

    return log_path


def log_epoch(
    log_path: str,
    epoch: int,
    train_total_loss: float,
    train_similarity_loss: float,
    train_smooth_loss: float,
    val_loss: float | None = None,
    val_dice: float | None = None,
) -> None:
    """Append epoch results to the log file."""
    with open(log_path, "a") as f:
        f.write(f"{epoch},")
        f.write(f"{train_total_loss:.6f},")
        f.write(f"{train_similarity_loss:.6f},")
        f.write(f"{train_smooth_loss:.8e},")

        if val_loss is not None and val_dice is not None:
            f.write(f"{val_loss:.6f},")
            f.write(f"{val_dice:.6f}\n")
        else:
            f.write("N/A,N/A\n")


def save_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_dices: list[float],
    val_epochs: list[int],
    output_dir: str,
    epoch: int,
) -> None:
    ensure_dir(output_dir)
    loss_path = os.path.join(output_dir, config.loss_plot_filename)
    dice_path = os.path.join(output_dir, config.dice_plot_filename)
    snapshots_dir = os.path.join(output_dir, "curve_snapshots")
    ensure_dir(snapshots_dir)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    if val_epochs:
        plt.plot(val_epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.savefig(
        os.path.join(snapshots_dir, f"loss_curve_epoch_{epoch:04d}.png"),
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    if val_epochs:
        plt.plot(val_epochs, val_dices, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.ylim(0.0, 1.0)
    if val_epochs:
        plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dice_path, dpi=150)
    plt.savefig(
        os.path.join(snapshots_dir, f"dice_curve_epoch_{epoch:04d}.png"),
        dpi=150,
    )
    plt.close()


def main() -> None:
    set_seed(config.seed)

    device = torch.device(config.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    amp_dtype_name = str(getattr(config, "amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    use_amp = bool(getattr(config, "use_amp", False)) and device.type == "cuda"
    debug_training = bool(getattr(config, "debug_training", False))
    debug_every_n_epochs = max(1, int(getattr(config, "debug_every_n_epochs", 1)))
    debug_batches_per_epoch = max(1, int(getattr(config, "debug_batches_per_epoch", 1)))

    early_stopping_patience = int(getattr(config, "early_stopping_patience", 0))
    early_stopping_metric = str(getattr(config, "early_stopping_metric", "val")).lower()
    early_stopping_min_delta = float(getattr(config, "early_stopping_min_delta", 0.0))
    flow_scale = float(getattr(config, "flow_scale", 1.0))
    dice_chunk_size = int(getattr(config, "dice_chunk_size", 4))
    dice_epsilon = float(getattr(config, "dice_epsilon", 1e-5))
    use_patch_based = bool(getattr(config, "use_patch_based", False))

    effective_batch_size = int(config.batch_size)
    if use_patch_based and effective_batch_size != 1:
        print(
            "[PatchMode] Overriding batch_size from "
            f"{effective_batch_size} to 1 to prevent patch-graph OOM."
        )
        effective_batch_size = 1

    run_output_dir = resolve_run_output_dir(config.output_dir)
    ensure_dir(run_output_dir)
    log_path = setup_logger(run_output_dir)

    validated_patients = validate_validation_folder_structure()
    if validated_patients:
        print(
            "Validation folder check passed: "
            f"{len(validated_patients)} patient folders with required files."
        )

    train_dataset = SynthMorphDataset(
        size=config.train_dataset_size,
        config=dict(config.generator_config),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=(
            config.persistent_workers if config.num_workers > 0 else False
        ),
    )

    model = SynthMorphUNet().to(device)
    integration_layer = create_integration_layer(
        image_size=config.image_size,
        integration_steps=config.integration_steps,
    ).to(device)

    patch_processor = None
    if use_patch_based:
        patch_size = int(getattr(config, "patch_size", 64))
        patch_overlap = int(getattr(config, "patch_overlap", 16))
        patch_processor = PatchProcessor(
            full_size=config.image_size,
            patch_size=patch_size,
            overlap=patch_overlap,
        )

    optimizer = get_optimizer(
        model.parameters(),
        optimizer_name=config.optimizer_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_dices: list[float] = []
    val_epochs: list[int] = []

    best_val_loss = float("inf")
    best_early_stop_metric = float("inf")
    epochs_without_improvement = 0
    best_model_path = os.path.join(run_output_dir, config.best_model_filename)

    epoch_progress = tqdm(range(1, config.num_epochs + 1), desc="Epochs")
    for epoch in epoch_progress:
        model.train()
        running_train_loss = 0.0
        running_similarity_loss = 0.0
        running_smooth_loss = 0.0

        batch_progress = tqdm(
            train_loader, desc=f"Train {epoch}/{config.num_epochs}", leave=False
        )
        for batch_index, batch in enumerate(batch_progress, start=1):
            fixed_image = (
                batch["fixed_image"].to(device, non_blocking=True).unsqueeze(1)
            )
            moving_image = (
                batch["moving_image"].to(device, non_blocking=True).unsqueeze(1)
            )
            fixed_label = batch["fixed_label_map"].to(device, non_blocking=True)
            moving_label = batch["moving_label_map"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                if use_patch_based and patch_processor is not None:
                    vector_field = forward_patch_based(
                        model,
                        fixed_image,
                        moving_image,
                        patch_processor,
                        device,
                    )
                else:
                    model_input = torch.cat([fixed_image, moving_image], dim=1)
                    vector_field = model(model_input)
                effective_vector_field = vector_field * flow_scale
                integrated_field = integration_layer(effective_vector_field)
                similarity_loss = soft_dice_loss_from_warped_labels_chunked(
                    fixed_label,
                    moving_label,
                    integrated_field,
                    num_classes=config.train_num_classes,
                    ignore_label=config.ignore_label,
                    chunk_size=dice_chunk_size,
                    epsilon=dice_epsilon,
                )
                smooth_loss = diffusion_loss(effective_vector_field)
                total_loss = (
                    similarity_loss + float(config.regularization_weight) * smooth_loss
                )

            total_loss.backward()

            should_debug_batch = (
                debug_training
                and epoch % debug_every_n_epochs == 0
                and batch_index <= debug_batches_per_epoch
            )
            if should_debug_batch:
                with torch.no_grad():
                    identity_field = torch.zeros_like(integrated_field)
                    identity_similarity = soft_dice_loss_from_warped_labels_chunked(
                        fixed_label,
                        moving_label,
                        identity_field,
                        num_classes=config.train_num_classes,
                        ignore_label=config.ignore_label,
                        chunk_size=dice_chunk_size,
                        epsilon=dice_epsilon,
                    )
                    identity_dice = float((1.0 - identity_similarity).item())
                    warped_dice = float((1.0 - similarity_loss.detach()).item())
                    field_abs_mean = float(vector_field.detach().abs().mean().item())
                    field_abs_max = float(vector_field.detach().abs().max().item())
                    effective_field_abs_mean = float(
                        effective_vector_field.detach().abs().mean().item()
                    )
                    effective_field_abs_max = float(
                        effective_vector_field.detach().abs().max().item()
                    )
                    integrated_abs_mean = float(
                        integrated_field.detach().abs().mean().item()
                    )
                    integrated_abs_max = float(
                        integrated_field.detach().abs().max().item()
                    )

                head_grad = model.vector_field_head.weight.grad
                grad_abs_mean = (
                    float(head_grad.detach().abs().mean().item())
                    if head_grad is not None
                    else float("nan")
                )
                grad_abs_max = (
                    float(head_grad.detach().abs().max().item())
                    if head_grad is not None
                    else float("nan")
                )

                print(
                    "[Debug] "
                    f"epoch={epoch} batch={batch_index} "
                    f"identity_dice={identity_dice:.4f} "
                    f"warped_dice={warped_dice:.4f} "
                    f"flow_scale={flow_scale:.2f} "
                    f"field_abs_mean={field_abs_mean:.6f} "
                    f"field_abs_max={field_abs_max:.6f} "
                    f"effective_field_abs_mean={effective_field_abs_mean:.6f} "
                    f"effective_field_abs_max={effective_field_abs_max:.6f} "
                    f"integrated_abs_mean={integrated_abs_mean:.6f} "
                    f"integrated_abs_max={integrated_abs_max:.6f} "
                    f"head_grad_abs_mean={grad_abs_mean:.6e} "
                    f"head_grad_abs_max={grad_abs_max:.6e}"
                )

            optimizer.step()

            running_train_loss += float(total_loss.item())
            running_similarity_loss += float(similarity_loss.item())
            running_smooth_loss += float(smooth_loss.item())
            batch_progress.set_postfix(loss=f"{total_loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_similarity_loss = running_similarity_loss / len(train_loader)
        epoch_smooth_loss = running_smooth_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        print(
            f"[Train] epoch={epoch} "
            f"train_loss={epoch_train_loss:.6f} "
            f"train_similarity={epoch_similarity_loss:.6f} "
            f"train_smooth={epoch_smooth_loss:.8e}"
        )

        log_suffix = f"train_loss={epoch_train_loss:.4f}"
        val_loss = None
        val_dice = None

        if epoch % config.validate_every == 0:
            val_loss, val_dice = evaluate_on_validation(
                model,
                integration_layer,
                device,
                epoch,
                run_output_dir,
                patch_processor,
            )
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_epochs.append(epoch)

            print(
                f"[Val] epoch={epoch} val_loss={val_loss:.6f} val_dice={val_dice:.6f}"
            )

            log_epoch(
                log_path,
                epoch,
                epoch_train_loss,
                epoch_similarity_loss,
                epoch_smooth_loss,
                val_loss,
                val_dice,
            )

            if not math.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    best_model_path,
                )
                print(
                    f"[Best] epoch={epoch} val_loss={best_val_loss:.6f} "
                    f"saved={best_model_path}"
                )
            log_suffix += f", val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
        else:
            log_epoch(
                log_path,
                epoch,
                epoch_train_loss,
                epoch_similarity_loss,
                epoch_smooth_loss,
            )

        save_curves(
            train_losses,
            val_losses,
            val_dices,
            val_epochs,
            run_output_dir,
            epoch,
        )

        monitor_name = ""
        monitor_value: float | None = None

        if early_stopping_metric == "val":
            if val_loss is not None and not math.isnan(val_loss):
                monitor_name = "validation loss"
                monitor_value = float(val_loss)
        else:
            monitor_name = "training loss"
            monitor_value = float(epoch_train_loss)

        if early_stopping_patience > 0 and monitor_value is not None:
            improved = monitor_value < (
                best_early_stop_metric - early_stopping_min_delta
            )
            if improved:
                best_early_stop_metric = monitor_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(
                    "\nEarly stopping triggered: "
                    f"no improvement in {monitor_name} for "
                    f"{early_stopping_patience} monitored checks."
                )
                epoch_progress.close()
                break

        epoch_progress.set_postfix_str(log_suffix)

    print("Training complete.")
    print(f"Run outputs: {run_output_dir}")
    if best_val_loss < float("inf"):
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best model: {best_model_path}")
    else:
        print("Validation was skipped or unavailable. No best model checkpoint saved.")


if __name__ == "__main__":
    main()
