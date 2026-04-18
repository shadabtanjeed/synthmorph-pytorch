import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorFieldIntegration(nn.Module):
    # Scaling-and-squaring integration for dense 3D vector fields in voxel units.
    def __init__(self, image_size: tuple[int, int, int], integration_steps: int = 7):
        super().__init__()
        self.image_size = tuple(image_size)
        self.integration_steps = int(integration_steps)
        self.register_buffer("base_grid", self._create_identity_grid(self.image_size))

    def _create_identity_grid(self, image_size: tuple[int, int, int]) -> torch.Tensor:
        d, h, w = image_size
        z = torch.linspace(-1.0, 1.0, d)
        y = torch.linspace(-1.0, 1.0, h)
        x = torch.linspace(-1.0, 1.0, w)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)

    def _field_to_sampling_grid(self, vector_field: torch.Tensor) -> torch.Tensor:
        d, h, w = self.image_size
        norm = torch.tensor(
            [
                2.0 / max(w - 1, 1),
                2.0 / max(h - 1, 1),
                2.0 / max(d - 1, 1),
            ],
            device=vector_field.device,
            dtype=vector_field.dtype,
        ).view(1, 3, 1, 1, 1)
        normalized_field = vector_field * norm
        return self.base_grid.to(device=vector_field.device, dtype=vector_field.dtype) + normalized_field.permute(0, 2, 3, 4, 1)

    def _warp(self, image: torch.Tensor, vector_field: torch.Tensor) -> torch.Tensor:
        sampling_grid = self._field_to_sampling_grid(vector_field)
        return F.grid_sample(
            image,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def forward(self, vector_field: torch.Tensor) -> torch.Tensor:
        integrated_field = vector_field / (2**self.integration_steps)

        for _ in range(self.integration_steps):
            integrated_field = integrated_field + self._warp(
                integrated_field,
                integrated_field,
            )

        return integrated_field


def create_integration_layer(
    image_size: tuple[int, int, int] = (160, 192, 224),
    integration_steps: int = 7,
) -> VectorFieldIntegration:
    return VectorFieldIntegration(
        image_size=image_size,
        integration_steps=integration_steps,
    )
