import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 26,
        ignore_label: int = 0,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)
        self.epsilon = float(epsilon)

    def forward(
        self,
        fixed_label_map: torch.Tensor,
        warped_moving_label_map: torch.Tensor,
    ) -> torch.Tensor:
        if fixed_label_map.dim() == 3:
            fixed_label_map = fixed_label_map.unsqueeze(0)
        if warped_moving_label_map.dim() == 3:
            warped_moving_label_map = warped_moving_label_map.unsqueeze(0)

        fixed_probs = F.one_hot(
            fixed_label_map.long(),
            num_classes=self.num_classes,
        ).permute(0, 4, 1, 2, 3).float()

        moving_probs = F.one_hot(
            warped_moving_label_map.long(),
            num_classes=self.num_classes,
        ).permute(0, 4, 1, 2, 3).float()

        valid_classes = [
            class_index
            for class_index in range(self.num_classes)
            if class_index != self.ignore_label
        ]
        fixed_probs = fixed_probs[:, valid_classes]
        moving_probs = moving_probs[:, valid_classes]

        intersection = (fixed_probs * moving_probs).sum(dim=(0, 2, 3, 4))
        fixed_volume = fixed_probs.sum(dim=(0, 2, 3, 4))
        moving_volume = moving_probs.sum(dim=(0, 2, 3, 4))

        dice = (2.0 * intersection + self.epsilon) / (
            fixed_volume + moving_volume + self.epsilon
        )
        return 1.0 - dice.mean()


def diffusion_loss(vector_field: torch.Tensor) -> torch.Tensor:
    dz = vector_field[:, :, 1:, :, :] - vector_field[:, :, :-1, :, :]
    dy = vector_field[:, :, :, 1:, :] - vector_field[:, :, :, :-1, :]
    dx = vector_field[:, :, :, :, 1:] - vector_field[:, :, :, :, :-1]

    return (dz.square().mean() + dy.square().mean() + dx.square().mean()) / 3.0


class SynthMorphLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 26,
        ignore_label: int = 0,
        lambda_smooth: float = 1.0,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.dice_loss = SoftDiceLoss(
            num_classes=num_classes,
            ignore_label=ignore_label,
            epsilon=epsilon,
        )
        self.lambda_smooth = float(lambda_smooth)

    def forward(
        self,
        fixed_label_map: torch.Tensor,
        warped_moving_label_map: torch.Tensor,
        vector_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        similarity_loss = self.dice_loss(
            fixed_label_map,
            warped_moving_label_map,
        )

        if vector_field is None:
            smoothness_loss = similarity_loss.new_tensor(0.0)
        else:
            smoothness_loss = diffusion_loss(vector_field)

        total_loss = similarity_loss + self.lambda_smooth * smoothness_loss
        return total_loss, similarity_loss, smoothness_loss
