from __future__ import annotations

from torch.utils.data import Dataset
from .generator import GenerateIntensityPair


class SynthMorphDataset(Dataset):
    def __init__(self, size: int = 2000, config: dict | None = None):
        """
        Initialize the SynthMorphDataset.

        Args:
            size: Number of samples in the dataset
            config: Configuration dictionary for GenerateIntensityPair
        """
        self.size = size
        self.generator_config = dict(config) if config is not None else None
        self.intensity_generator: GenerateIntensityPair | None = None

    def _get_intensity_generator(self) -> GenerateIntensityPair:
        # Lazily initialize in the worker process to avoid serializing heavy tensors.
        if self.intensity_generator is None:
            self.intensity_generator = GenerateIntensityPair(self.generator_config)
        return self.intensity_generator

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single sample from the dataset.

        Process:
        1. Create a label map using GenerateLabelMap
        2. Create a pair of label maps (fixed and moving) using GenerateLabelMapPair
        3. Fill intensities with GenerateIntensityPair
        4. Normalize intensity images to [0, 1]

        Returns:
            Dictionary containing:
            - fixed_image: Fixed intensity image normalized to [0, 1]
            - moving_image: Moving intensity image normalized to [0, 1]
            - fixed_label_map: Fixed label map
            - moving_label_map: Moving label map
        """
        # Generate the label map pair
        intensity_generator = self._get_intensity_generator()
        fixed_label_map, moving_label_map = intensity_generator.createLabelMapPair()

        # Generate the intensity pair from the label maps
        fixed_intensity, moving_intensity = intensity_generator.createIntensityPair(
            label_map_pair=(fixed_label_map, moving_label_map)
        )

        # Normalize intensity images to [0, 1] using the max intensity from config
        max_intensity = max(float(intensity_generator.intensity_clip_range[1]), 1e-6)
        fixed_intensity = fixed_intensity / max_intensity
        moving_intensity = moving_intensity / max_intensity

        return {
            "fixed_image": fixed_intensity,
            "moving_image": moving_intensity,
            "fixed_label_map": fixed_label_map.long(),
            "moving_label_map": moving_label_map.long(),
        }
