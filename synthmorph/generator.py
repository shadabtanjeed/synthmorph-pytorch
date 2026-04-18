# full image size : 160 x 192 x 224
# low res image size : 5 x 6 x 7
# deformation grid size : 10 x 12 x 14
# bias field grid size : 4 x 5 x 6

# Steps:
# Create a label map: s_m
# Create a pair of images: (s_f, s_m) from the label map
# Fill intensity values in the image pair


from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


# Builds one synthetic anatomical label map by combining random low-resolution volumes and smooth deformations.
class GenerateLabelMap:
    DEFAULT_CONFIG = {
        "full_image_size": (160, 192, 224),
        "low_res_image_size": (5, 6, 7),
        "deformation_grid_size": (10, 12, 14),
        "bias_field_grid_size": (4, 5, 6),
        "num_classes": 26,
        "device": "cuda",
        "dtype": torch.float32,
        "integration_steps": 7,
    }

    def __init__(self, config: Mapping[str, Any] | None = None):
        # Merge user-provided config with defaults so config can be optional.
        merged_config = dict(self.DEFAULT_CONFIG)
        if config:
            merged_config.update(config)

        self.config = merged_config
        self.full_image_size = tuple(
            merged_config["full_image_size"]
        )  # (160, 192, 224)
        self.low_res_image_size = tuple(
            merged_config["low_res_image_size"]
        )  # (5, 6, 7)
        self.deformation_grid_size = tuple(
            merged_config["deformation_grid_size"]
        )  # (10, 12, 14)
        self.bias_field_grid_size = tuple(
            merged_config["bias_field_grid_size"]
        )  # (4, 5, 6)
        self.J = int(merged_config["num_classes"])  # number of classes = 26
        self.device = torch.device(merged_config["device"])
        self.dtype = merged_config["dtype"]
        self.integration_steps = int(merged_config["integration_steps"])

        # Precompute the identity sampling grid used during warping.
        self._base_grid = self._create_identity_grid(self.full_image_size)

    def _randn(self, *shape: int) -> torch.Tensor:
        # Create a random tensor directly on the configured device and dtype.
        return torch.randn(*shape, device=self.device, dtype=self.dtype)

    def _create_identity_grid(self, image_size: tuple[int, int, int]) -> torch.Tensor:
        # Build a normalized [-1, 1] coordinate grid for grid_sample.
        d, h, w = image_size
        z = torch.linspace(-1.0, 1.0, d, device=self.device, dtype=self.dtype)
        y = torch.linspace(-1.0, 1.0, h, device=self.device, dtype=self.dtype)
        x = torch.linspace(-1.0, 1.0, w, device=self.device, dtype=self.dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)

    def _resize_scalar_field(
        self,
        image: torch.Tensor,
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        # Upsample a single-channel volumetric image with trilinear interpolation.
        return F.interpolate(
            image,
            size=target_size,
            mode="trilinear",
            align_corners=True,
        )

    def _resize_vector_field(
        self,
        field: torch.Tensor,
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        # Upsample a 3D displacement field and rescale its vector magnitudes.
        resized = F.interpolate(
            field,
            size=target_size,
            mode="trilinear",
            align_corners=True,
        )

        src_size = field.shape[2:]
        scale_factors = [
            (target - 1) / max(source - 1, 1)
            for source, target in zip(src_size, target_size)
        ]
        resized[:, 0] *= scale_factors[2]
        resized[:, 1] *= scale_factors[1]
        resized[:, 2] *= scale_factors[0]
        return resized

    def _field_to_sampling_grid(self, deformation_field: torch.Tensor) -> torch.Tensor:
        # Convert voxel-space displacements into the normalized sampling grid expected by grid_sample.
        d, h, w = self.full_image_size
        norm = torch.tensor(
            [
                2.0 / max(w - 1, 1),
                2.0 / max(h - 1, 1),
                2.0 / max(d - 1, 1),
            ],
            device=self.device,
            dtype=self.dtype,
        ).view(1, 3, 1, 1, 1)
        normalized_field = deformation_field * norm
        return self._base_grid + normalized_field.permute(0, 2, 3, 4, 1)

    def _warp_tensor(
        self,
        image: torch.Tensor,
        integrated_deformation_field: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        # Warp an image/tensor with a displacement field using grid_sample.
        sampling_grid = self._field_to_sampling_grid(integrated_deformation_field)
        return F.grid_sample(
            image,
            sampling_grid,
            mode=mode,
            padding_mode="border",
            align_corners=True,
        )

    def generateLowResImage(self) -> torch.Tensor:
        # genereate a grid of size low_res_image_size
        # each voxel value is random from Normal distribution with mean = 0 and std = 1
        return self._randn(1, 1, *self.low_res_image_size)

    def upscaleLowResImage(self, low_res_image: torch.Tensor) -> torch.Tensor:
        # perform trilinear interpolation to upscale the low res image to the full image size (160, 192, 224)
        # full size image is of the size full_image_size
        # returns the upscaled image of size full_image_size
        return self._resize_scalar_field(low_res_image, self.full_image_size)

    def generateDeformationField(self) -> torch.Tensor:
        # create a grid of size deformation_grid_size
        # each voxel is random vector [dx, dy, dz] with normal distribution with mean = 0 and std = 1
        # returns the deformation field of size deformation_grid_size x 3 (for dx, dy, dz)
        # upscaled to the full image size using trilinear interpolation, resulting in a deformation field of size full_image_size x 3
        low_res_field = self._randn(1, 3, *self.deformation_grid_size)
        return self._resize_vector_field(low_res_field, self.full_image_size)

    def integrateDeformationField(
        self, deformation_field: torch.Tensor
    ) -> torch.Tensor:
        # takes a deformation field of size full_image_size x 3 and integrates it to get the final deformation field
        # integration is done using scaling and squaring method
        integrated_field = deformation_field / (2**self.integration_steps)

        for _ in range(self.integration_steps):
            warped_field = self._warp_tensor(
                integrated_field,
                integrated_field,
                mode="bilinear",
            )
            integrated_field = integrated_field + warped_field

        return integrated_field

    def createWarpedImage(
        self,
        image: torch.Tensor,
        integrated_deformation_field: torch.Tensor,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        # takes the full size image and the integrated deformation field and warps the image using the deformation field
        # returns the warped image of size full_image_size
        return self._warp_tensor(
            image,
            integrated_deformation_field,
            mode=mode,
        )

    def createLabelMap(self) -> torch.Tensor:
        # for J times:
        # first create a low res image
        # then upscale it to the full image size
        # then create a deformation field and integrate it
        # then warp the upscaled image using the integrated deformation field to get a warped image of size full_image_size

        # So now you have J warped images of size full_image_size, each corresponding to a class

        # Next part is to create the label map using all J images
        # at each voxel, it goes through all J classes and assigns the class index with the highest value at that voxel as the label for that voxel
        # So the voxel can have the values ranging from 0 to J-1
        warped_images = []

        for _ in range(self.J):
            low_res_image = self.generateLowResImage()
            full_res_image = self.upscaleLowResImage(low_res_image)
            deformation_field = self.generateDeformationField()
            integrated_deformation_field = self.integrateDeformationField(
                deformation_field
            )
            warped_image = self.createWarpedImage(
                full_res_image, integrated_deformation_field
            )
            warped_images.append(warped_image)

        stacked_images = torch.cat(warped_images, dim=1)
        return torch.argmax(stacked_images, dim=1).squeeze(0)


# Starts from one label map and creates a fixed/moving pair by applying two different random deformations.
class GenerateLabelMapPair(GenerateLabelMap):
    # takes the label map generated by GenerateLabelMap
    # and creates a pair of images (s_f, s_m) from the label map
    PAIR_DEFAULT_CONFIG = {
        "pair_deformation_std_min": 3.0,
        "pair_deformation_std_max": 15.0,
    }

    def __init__(self, config: Mapping[str, Any] | None = None):
        # Extend the base config with pair-specific deformation settings.
        merged_config = dict(self.PAIR_DEFAULT_CONFIG)
        if config:
            merged_config.update(config)

        super().__init__(merged_config)
        self.pair_deformation_std_min = float(self.config["pair_deformation_std_min"])
        self.pair_deformation_std_max = float(self.config["pair_deformation_std_max"])

    def _sample_pair_deformation_std(self) -> float:
        # Randomly choose the deformation strength used to build one field in the pair.
        return float(
            torch.empty(1, device=self.device, dtype=self.dtype)
            .uniform_(
                self.pair_deformation_std_min,
                self.pair_deformation_std_max,
            )
            .item()
        )

    def createLabelMap(self) -> torch.Tensor:
        # Reuse the single label-map pipeline from the parent class without the pair-specific overrides.
        warped_images = []

        for _ in range(self.J):
            low_res_image = GenerateLabelMap.generateLowResImage(self)
            full_res_image = GenerateLabelMap.upscaleLowResImage(self, low_res_image)
            deformation_field = GenerateLabelMap.generateDeformationField(self)
            integrated_deformation_field = GenerateLabelMap.integrateDeformationField(
                self,
                deformation_field,
            )
            warped_image = GenerateLabelMap.createWarpedImage(
                self,
                full_res_image,
                integrated_deformation_field,
            )
            warped_images.append(warped_image)

        stacked_images = torch.cat(warped_images, dim=1)
        return torch.argmax(stacked_images, dim=1).squeeze(0)

    def generateDeformationField(self) -> tuple[torch.Tensor, torch.Tensor]:
        # creates two deformation fields, one for s_f and one for s_m
        # the deformation field is of size deformation_grid_size x 3 (for dx, dy, dz)
        # each voxel is random vector [dx, dy, dz] with normal distribution with mean = 0 and std = 3 to 15 randomly
        # upscaled to the full image size using trilinear interpolation, resulting in a deformation field of size full_image_size x 3
        fixed_std = self._sample_pair_deformation_std()
        moving_std = self._sample_pair_deformation_std()

        fixed_low_res_field = self._randn(1, 3, *self.deformation_grid_size) * fixed_std
        moving_low_res_field = (
            self._randn(1, 3, *self.deformation_grid_size) * moving_std
        )

        fixed_field = self._resize_vector_field(
            fixed_low_res_field, self.full_image_size
        )
        moving_field = self._resize_vector_field(
            moving_low_res_field, self.full_image_size
        )
        return fixed_field, moving_field

    def integrateDeformationField(
        self,
        deformation_fields: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # takes a deformation field of size full_image_size x 3 and integrates it to get the final deformation field
        # integration is done using scaling and squaring method
        fixed_field, moving_field = deformation_fields
        integrated_fixed_field = super().integrateDeformationField(fixed_field)
        integrated_moving_field = super().integrateDeformationField(moving_field)
        return integrated_fixed_field, integrated_moving_field

    def createWarpedImage(
        self,
        label_map: torch.Tensor,
        integrated_deformation_fields: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # takes the full size image and the integrated deformation field and warps the image using the deformation field
        # returns the warped image of size full_image_size
        # basically returns two warped images, one for s_f and one for s_m
        if label_map.dim() == 3:
            label_map = label_map.unsqueeze(0).unsqueeze(0)
        elif label_map.dim() == 4:
            label_map = label_map.unsqueeze(1)

        label_map = label_map.to(device=self.device, dtype=self.dtype)
        integrated_fixed_field, integrated_moving_field = integrated_deformation_fields

        fixed_label_map = self._warp_tensor(
            label_map,
            integrated_fixed_field,
            mode="nearest",
        )
        moving_label_map = self._warp_tensor(
            label_map,
            integrated_moving_field,
            mode="nearest",
        )
        return (
            fixed_label_map.squeeze(0).squeeze(0).round().long(),
            moving_label_map.squeeze(0).squeeze(0).round().long(),
        )

    def createLabelMapPair(
        self,
        label_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # generate a source label map if one is not provided, then create the fixed/moving pair from it
        if label_map is None:
            label_map = self.createLabelMap()

        deformation_fields = self.generateDeformationField()
        integrated_deformation_fields = self.integrateDeformationField(
            deformation_fields
        )
        return self.createWarpedImage(label_map, integrated_deformation_fields)


# Converts the fixed/moving label maps into intensity images by sampling class intensities and applying simple image effects.
class GenerateIntensityPair(GenerateLabelMapPair):
    # takes the pair of label maps generated by GenerateLabelMapPair
    # and fills intensity values in the image pair

    # intensity mean range: 25 to 225
    # intensity std range: 5 to 25
    INTENSITY_DEFAULT_CONFIG = {
        "intensity_mean_range": (25.0, 225.0),
        "intensity_std_range": (5.0, 25.0),
        "blur_sigma_range": (0.0, 2.0),
        "bias_field_std_range": (0.3, 0.5),
        "exp_transform_std_range": (0.3, 0.5),
        "intensity_clip_range": (0.0, 255.0),
    }

    def __init__(self, config: Mapping[str, Any] | None = None):
        # Extend the pair generator with defaults for the intensity synthesis stage.
        merged_config = dict(self.INTENSITY_DEFAULT_CONFIG)
        if config:
            merged_config.update(config)

        super().__init__(merged_config)
        self.intensity_mean_range = tuple(self.config["intensity_mean_range"])
        self.intensity_std_range = tuple(self.config["intensity_std_range"])
        self.blur_sigma_range = tuple(self.config["blur_sigma_range"])
        self.bias_field_std_range = tuple(self.config["bias_field_std_range"])
        self.exp_transform_std_range = tuple(self.config["exp_transform_std_range"])
        self.intensity_clip_range = tuple(self.config["intensity_clip_range"])

    def _sample_uniform(self, value_range: tuple[float, float]) -> float:
        # Sample a scalar uniformly from a configured range.
        low, high = value_range
        return float(
            torch.empty(1, device=self.device, dtype=self.dtype).uniform_(low, high).item()
        )

    def _ensure_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        # Convert an image into [1, 1, D, H, W] format for volumetric ops.
        if image.dim() == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 4:
            image = image.unsqueeze(1)
        return image.to(device=self.device, dtype=self.dtype)

    def _build_gaussian_kernel_1d(self, sigma: float) -> torch.Tensor:
        # Build a normalized 1D Gaussian kernel for separable blurring.
        if sigma <= 0.0:
            return torch.ones(1, device=self.device, dtype=self.dtype)

        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(
            -radius,
            radius + 1,
            device=self.device,
            dtype=self.dtype,
        )
        kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
        return kernel / kernel.sum()

    def _apply_separable_blur(
        self,
        image: torch.Tensor,
        sigmas: tuple[float, float, float],
    ) -> torch.Tensor:
        # Apply 3D Gaussian blur one axis at a time using conv3d.
        blurred = image
        for axis, sigma in enumerate(sigmas):
            kernel_1d = self._build_gaussian_kernel_1d(sigma)
            if kernel_1d.numel() == 1:
                continue

            kernel_shape = [1, 1, 1, 1, 1]
            kernel_shape[2 + axis] = kernel_1d.numel()
            kernel = kernel_1d.view(*kernel_shape)
            padding = [0, 0, 0]
            padding[axis] = kernel_1d.numel() // 2
            blurred = F.conv3d(blurred, kernel, padding=tuple(padding))
        return blurred

    def fillVoxelIntensiies(self, label_map: torch.Tensor) -> torch.Tensor:
        # for each class in label, J randomly pick a mean and std from the specified ranges
        # then for each voxel in that class, fill the voxel value with a random value from the normal distribution with the chosen mean and std
        # it should be run sepately for the fixed and moving label maps, so that the intensity values in the fixed and moving images are different even if the label maps are the same
        label_map = label_map.to(device=self.device, dtype=torch.long)
        
        means = torch.empty(self.J, device=self.device, dtype=self.dtype).uniform_(
            self.intensity_mean_range[0], self.intensity_mean_range[1]
        )
        stds = torch.empty(self.J, device=self.device, dtype=self.dtype).uniform_(
            self.intensity_std_range[0], self.intensity_std_range[1]
        )
        
        pixel_means = means[label_map]
        pixel_stds = stds[label_map]
        
        image = pixel_means + pixel_stds * torch.randn_like(pixel_means)
        return image

    def gaussianBlur(self, image: torch.Tensor) -> torch.Tensor:
        # apply a gaussian blur to the filled image
        # sample three sigma values for the gaussian blur from the range 0 to 2, one for each dimension
        # apply the gaussian blur with the sampled sigma values to the filled image
        image = self._ensure_image_tensor(image)
        sigmas = (
            self._sample_uniform(self.blur_sigma_range),
            self._sample_uniform(self.blur_sigma_range),
            self._sample_uniform(self.blur_sigma_range),
        )
        blurred = self._apply_separable_blur(image, sigmas)
        return blurred.squeeze(0).squeeze(0)

    def biasField(self, image: torch.Tensor) -> torch.Tensor:
        # takes in the blurred image and applies a bias field to it
        # create a bias field of size bias_field_grid_size x 1, where each voxel is a random value from normal distribution with mean = 0 and std = 0.3 to 0.5
        # upsample the bias field to the full image size using trilinear interpolation, resulting in a bias field of size full_image_size x 1
        # add the bias field to the blurred image to get the final image with bias field applied
        # the bias field is applied in log space, so the final image is obtained by multiplying the blurred image with the exponential of the bias field
        image = self._ensure_image_tensor(image)
        bias_std = self._sample_uniform(self.bias_field_std_range)
        low_res_bias = self._randn(1, 1, *self.bias_field_grid_size) * bias_std
        full_res_bias = self._resize_scalar_field(low_res_bias, self.full_image_size)
        biased_image = image * torch.exp(full_res_bias)
        return biased_image.squeeze(0).squeeze(0)

    def exponentialTransform(self, image: torch.Tensor) -> torch.Tensor:
        # apply an exponential transform to the image with bias field applied to get the final image
        # take a low res grid with same size as bias_field_grid_size, and each voxel is a random value from normal distribution with mean = 0 and std = 0.3 to 0.5
        # upsample the low res grid to the full image size using trilinear interpolation,
        # apply the exponential transform to the image
        image = self._ensure_image_tensor(image)
        exp_std = self._sample_uniform(self.exp_transform_std_range)
        low_res_grid = self._randn(1, 1, *self.bias_field_grid_size) * exp_std
        full_res_grid = self._resize_scalar_field(low_res_grid, self.full_image_size)

        safe_image = torch.clamp(image, min=0.0)
        transformed_image = torch.pow(safe_image + 1.0, torch.exp(full_res_grid)) - 1.0
        transformed_image = torch.clamp(
            transformed_image,
            min=self.intensity_clip_range[0],
            max=self.intensity_clip_range[1],
        )
        return transformed_image.squeeze(0).squeeze(0)

    def _createSingleIntensityImage(self, label_map: torch.Tensor) -> torch.Tensor:
        # Run the full intensity synthesis pipeline for one label map.
        image = self.fillVoxelIntensiies(label_map)
        image = self.gaussianBlur(image)
        image = self.biasField(image)
        image = self.exponentialTransform(image)
        return image

    def createIntensityPair(
        self,
        label_map_pair: tuple[torch.Tensor, torch.Tensor] | None = None,
        source_label_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # for each image in the pair (fixed and moving):
        # fill the voxel intensities
        # apply gaussian blur
        # apply bias field
        # apply exponential transform
        if label_map_pair is None:
            label_map_pair = self.createLabelMapPair(source_label_map)

        fixed_label_map, moving_label_map = label_map_pair
        fixed_image = self._createSingleIntensityImage(fixed_label_map)
        moving_image = self._createSingleIntensityImage(moving_label_map)

        # return the two intensity filled image
        return fixed_image, moving_image
