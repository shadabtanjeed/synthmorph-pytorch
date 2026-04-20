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
        return self.base_grid.to(
            device=vector_field.device, dtype=vector_field.dtype
        ) + normalized_field.permute(0, 2, 3, 4, 1)

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


class PatchProcessor:
    """Extract overlapping patches from a volume and stitch displacement fields back together."""

    def __init__(
        self,
        full_size: tuple[int, int, int],
        patch_size: int,
        overlap: int,
    ):
        self.full_size = tuple(full_size)
        self.patch_size = int(patch_size)
        self.overlap = int(overlap)
        self.stride = self.patch_size - self.overlap

    def extract_patches(
        self, volume: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[tuple]]:
        """
        Extract overlapping patches from a volume.
        Returns list of patches and list of their (d_start, h_start, w_start) positions.
        """
        if volume.dim() != 4:
            raise ValueError(
                "PatchProcessor.extract_patches expects [C, D, H, W], "
                f"got shape {tuple(volume.shape)}"
            )

        if tuple(volume.shape[1:]) != self.full_size:
            raise ValueError(
                "PatchProcessor.extract_patches got unexpected spatial size: "
                f"{tuple(volume.shape[1:])}, expected {self.full_size}"
            )

        d, h, w = self.full_size
        patches = []
        positions = []

        d_starts = list(range(0, d - self.patch_size + 1, self.stride))
        if d_starts[-1] + self.patch_size < d:
            d_starts.append(d - self.patch_size)

        h_starts = list(range(0, h - self.patch_size + 1, self.stride))
        if h_starts[-1] + self.patch_size < h:
            h_starts.append(h - self.patch_size)

        w_starts = list(range(0, w - self.patch_size + 1, self.stride))
        if w_starts[-1] + self.patch_size < w:
            w_starts.append(w - self.patch_size)

        for d_start in d_starts:
            for h_start in h_starts:
                for w_start in w_starts:
                    d_end = d_start + self.patch_size
                    h_end = h_start + self.patch_size
                    w_end = w_start + self.patch_size

                    patch = volume[:, d_start:d_end, h_start:h_end, w_start:w_end]
                    patches.append(patch)
                    positions.append((d_start, h_start, w_start))

        return patches, positions

    def stitch_patches(
        self,
        patches: list[torch.Tensor],
        positions: list[tuple],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Stitch displacement field patches back into full volume with overlap averaging."""
        output = torch.zeros(
            (1, 3) + self.full_size,
            device=device,
            dtype=dtype,
        )
        counts = torch.zeros(
            (1, 1) + self.full_size,
            device=device,
            dtype=dtype,
        )

        for patch, (d_start, h_start, w_start) in zip(patches, positions):
            d_end = d_start + self.patch_size
            h_end = h_start + self.patch_size
            w_end = w_start + self.patch_size

            output[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += patch
            counts[:, :, d_start:d_end, h_start:h_end, w_start:w_end] += 1.0

        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)
        output = output / counts
        return output
