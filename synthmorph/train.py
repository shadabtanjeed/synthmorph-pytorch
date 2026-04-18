from __future__ import annotations

import math
import os
import random
from pathlib import Path
from contextlib import nullcontext

import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthmorph import configs as config
from synthmorph.dataset import SynthMorphDataset
from synthmorph.loss import SynthMorphLoss
from synthmorph.network import SynthMorphUNet
from synthmorph.utils import create_integration_layer


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
    val_loss_fn: SynthMorphLoss,
    device: torch.device,
) -> tuple[float, float]:
    patients = discover_validation_patients()
    pairs = build_validation_pairs(patients)

    if not pairs:
        return float("nan"), float("nan")

    model.eval()
    total_val_loss = 0.0
    total_val_dice = 0.0

    with torch.no_grad():
        progress = tqdm(pairs, desc="Validation", leave=False)
        for fixed_patient, moving_patient in progress:
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
            vector_field = model(model_input)
            integrated_field = integration_layer(vector_field)
            warped_moving_probs = warp_label_map_soft(
                moving_label.unsqueeze(0),
                integrated_field,
                num_classes=config.val_num_classes,
            )

            # Clamp labels to valid range for loss computation
            fixed_label_clamped = torch.clamp(
                fixed_label.unsqueeze(0),
                0,
                config.val_num_classes - 1,
            )

            val_total_loss, val_similarity_loss, _ = val_loss_fn(
                fixed_label_clamped,
                warped_moving_probs,
                vector_field,
            )
            val_dice = 1.0 - val_similarity_loss

            total_val_loss += float(val_total_loss.item())
            total_val_dice += float(val_dice.item())

            progress.set_postfix(
                loss=f"{val_total_loss.item():.4f}",
                dice=f"{val_dice.item():.4f}",
            )

    mean_val_loss = total_val_loss / len(pairs)
    mean_val_dice = total_val_dice / len(pairs)
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
        f.write(f"{train_smooth_loss:.6f},")
        
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
) -> None:
    ensure_dir(config.output_dir)
    loss_path = os.path.join(config.output_dir, config.loss_plot_filename)
    dice_path = os.path.join(config.output_dir, config.dice_plot_filename)

    if os.path.exists(loss_path):
        os.remove(loss_path)
    if os.path.exists(dice_path):
        os.remove(dice_path)

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
    plt.close()

    plt.figure(figsize=(8, 5))
    if val_epochs:
        plt.plot(val_epochs, val_dices, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dice_path, dpi=150)
    plt.close()


def main() -> None:
    set_seed(config.seed)

    device = torch.device(config.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    amp_dtype_name = str(getattr(config, "amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    use_amp = bool(getattr(config, "use_amp", False)) and device.type == "cuda"

    ensure_dir(config.output_dir)
    log_path = setup_logger(config.output_dir)

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
        batch_size=config.batch_size,
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

    train_loss_fn = SynthMorphLoss(
        num_classes=config.train_num_classes,
        ignore_label=config.ignore_label,
        lambda_smooth=config.regularization_weight,
    )
    val_loss_fn = SynthMorphLoss(
        num_classes=config.val_num_classes,
        ignore_label=config.ignore_label,
        lambda_smooth=config.regularization_weight,
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
    best_train_loss = float("inf")
    epochs_without_improvement = 0
    best_model_path = os.path.join(config.output_dir, config.best_model_filename)

    epoch_progress = tqdm(range(1, config.num_epochs + 1), desc="Epochs")
    for epoch in epoch_progress:
        model.train()
        running_train_loss = 0.0
        running_similarity_loss = 0.0
        running_smooth_loss = 0.0

        batch_progress = tqdm(
            train_loader, desc=f"Train {epoch}/{config.num_epochs}", leave=False
        )
        for batch in batch_progress:
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
                model_input = torch.cat([fixed_image, moving_image], dim=1)
                vector_field = model(model_input)
                integrated_field = integration_layer(vector_field)
                warped_moving_probs = warp_label_map_soft(
                    moving_label,
                    integrated_field,
                    num_classes=config.train_num_classes,
                )

                total_loss, similarity_loss, smooth_loss = train_loss_fn(
                    fixed_label,
                    warped_moving_probs,
                    vector_field,
                )

            total_loss.backward()
            optimizer.step()

            running_train_loss += float(total_loss.item())
            running_similarity_loss += float(similarity_loss.item())
            running_smooth_loss += float(smooth_loss.item())
            batch_progress.set_postfix(loss=f"{total_loss.item():.4f}")

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_similarity_loss = running_similarity_loss / len(train_loader)
        epoch_smooth_loss = running_smooth_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        log_suffix = f"train_loss={epoch_train_loss:.4f}"

        # Early stopping based on training loss
        if epoch_train_loss < best_train_loss:
            best_train_loss = epoch_train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"\nEarly stopping triggered: No improvement in training loss for "
                f"{config.early_stopping_patience} epochs."
            )
            epoch_progress.close()
            break

        if epoch % config.validate_every == 0:
            val_loss, val_dice = evaluate_on_validation(
                model, integration_layer, val_loss_fn, device
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

            save_curves(train_losses, val_losses, val_dices, val_epochs)
            log_suffix += f", val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
        else:
            log_epoch(
                log_path,
                epoch,
                epoch_train_loss,
                epoch_similarity_loss,
                epoch_smooth_loss,
            )

        epoch_progress.set_postfix_str(log_suffix)

    print("Training complete.")
    if best_val_loss < float("inf"):
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best model: {best_model_path}")
    else:
        print("Validation was skipped or unavailable. No best model checkpoint saved.")


if __name__ == "__main__":
    main()
