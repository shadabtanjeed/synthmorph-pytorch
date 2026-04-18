import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class SynthMorphUNet(nn.Module):
    # Input:  [B, 2, 160, 192, 224]
    # Output: [B, 3, 160, 192, 224] vector field
    def __init__(self):
        super().__init__()

        encoder_filters = (64, 64, 64, 64)

        self.enc1 = ConvBlock(2, encoder_filters[0])
        self.down1 = nn.Conv3d(
            encoder_filters[0],
            encoder_filters[0],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.enc2 = ConvBlock(encoder_filters[0], encoder_filters[1])
        self.down2 = nn.Conv3d(
            encoder_filters[1],
            encoder_filters[1],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.enc3 = ConvBlock(encoder_filters[1], encoder_filters[2])
        self.down3 = nn.Conv3d(
            encoder_filters[2],
            encoder_filters[2],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.enc4 = ConvBlock(encoder_filters[2], encoder_filters[3])
        self.down4 = nn.Conv3d(
            encoder_filters[3],
            encoder_filters[3],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.bottleneck1 = ConvBlock(encoder_filters[3], encoder_filters[3])
        self.bottleneck2 = ConvBlock(encoder_filters[3], encoder_filters[3])

        self.up4 = UpConvBlock(encoder_filters[3], encoder_filters[3])
        self.dec4_1 = ConvBlock(
            encoder_filters[3] + encoder_filters[3], encoder_filters[3]
        )
        self.dec4_2 = ConvBlock(encoder_filters[3], encoder_filters[3])

        self.up3 = UpConvBlock(encoder_filters[3], encoder_filters[2])
        self.dec3_1 = ConvBlock(
            encoder_filters[2] + encoder_filters[2], encoder_filters[2]
        )
        self.dec3_2 = ConvBlock(encoder_filters[2], encoder_filters[2])

        self.up2 = UpConvBlock(encoder_filters[2], encoder_filters[1])
        self.dec2_1 = ConvBlock(
            encoder_filters[1] + encoder_filters[1], encoder_filters[1]
        )
        self.dec2_2 = ConvBlock(encoder_filters[1], encoder_filters[1])

        self.up1 = UpConvBlock(encoder_filters[1], encoder_filters[0])
        self.dec1_1 = ConvBlock(
            encoder_filters[0] + encoder_filters[0], encoder_filters[0]
        )
        self.dec1_2 = ConvBlock(encoder_filters[0], encoder_filters[0])

        self.vector_field_head = nn.Conv3d(
            encoder_filters[0],
            3,
            kernel_size=3,
            padding=1,
        )

        nn.init.normal_(self.vector_field_head.weight, mean=0.0, std=1e-3)
        if self.vector_field_head.bias is not None:
            nn.init.zeros_(self.vector_field_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(x)
        feat1 = F.leaky_relu(self.down1(skip1), negative_slope=0.2)

        skip2 = self.enc2(feat1)
        feat2 = F.leaky_relu(self.down2(skip2), negative_slope=0.2)

        skip3 = self.enc3(feat2)
        feat3 = F.leaky_relu(self.down3(skip3), negative_slope=0.2)

        skip4 = self.enc4(feat3)
        feat4 = F.leaky_relu(self.down4(skip4), negative_slope=0.2)

        feat_bn = self.bottleneck1(feat4)
        feat_bn = self.bottleneck2(feat_bn)

        up4 = self.up4(feat_bn)
        dec4 = self.dec4_1(torch.cat([up4, skip4], dim=1))
        dec4 = self.dec4_2(dec4)

        up3 = self.up3(dec4)
        dec3 = self.dec3_1(torch.cat([up3, skip3], dim=1))
        dec3 = self.dec3_2(dec3)

        up2 = self.up2(dec3)
        dec2 = self.dec2_1(torch.cat([up2, skip2], dim=1))
        dec2 = self.dec2_2(dec2)

        up1 = self.up1(dec2)
        dec1 = self.dec1_1(torch.cat([up1, skip1], dim=1))
        dec1 = self.dec1_2(dec1)

        vector_field = self.vector_field_head(dec1)
        return vector_field
