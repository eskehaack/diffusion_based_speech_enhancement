from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(eq=False, repr=False)
class UNet1D(nn.Module):
    in_channels: int = 1
    base_channels: int = 32
    num_down: int = 2

    def __post_init__(self):
        super().__init__()
        # Downsampling path
        self.downs = nn.ModuleList()
        ch = self.in_channels + 1
        for _ in range(self.num_down):
            self.downs.append(
                nn.Sequential(
                    nn.Conv1d(ch, self.base_channels, 4, 2, 1), nn.ReLU(inplace=True)
                )
            )
            ch = self.base_channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True)
        )

        # Upsampling path
        self.ups = nn.ModuleList()
        for _ in range(self.num_down):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose1d(ch * 2, self.base_channels, 4, 2, 1),
                    nn.ReLU(inplace=True),
                )
            )
            ch = self.base_channels

        # Final conv: output mean and logvar
        self.final = nn.Conv1d(ch, self.in_channels * 2, 1)

    def forward(self, x: torch.Tensor, t: int):
        # x: (batch, channels, length)
        x = x.reshape((x.shape[0], 1, x.shape[-1]))
        batch_size, channels, length = x.shape
        device = x.device
        t = t.type(x.dtype).to(device)
        t_channel = t.view(batch_size, 1, 1).expand(batch_size, 1, length)
        x = torch.cat([x, t_channel], dim=1)  # (batch, channels+1, length)
        # Start model
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up in self.ups:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        out = self.final(x)
        # Split output into mean and logvar along channel dimension
        if out.dim() == 1:
            # Single sample, shape (2, length)
            mean, logvar = out.chunk(2, dim=0)
        else:
            # Batch, shape (batch, 2, length) or (batch, 2*channels, length)
            mean, logvar = out.chunk(2, dim=1)
        return mean, logvar
