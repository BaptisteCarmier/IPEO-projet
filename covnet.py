import torch
import torch.nn as nn
import torch.optim as optim

class UNetRegressor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetRegressor, self).__init__()

                                                                   # Original size : 12x32x32

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # Output : 64x32x32
            nn.BatchNorm2d(64),                                    
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),           # Output : 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)         # Downsample: 32x32 -> 16x16
        

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # Output : 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Output : 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: 16x16 -> 8x8
        
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output : 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Output : 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample: 8x8 -> 4x4
                                                            # Ouptut : 256x4x4

        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Ouptut : 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Ouptut : 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4) # deeper network tends to overfit the whole result
        )

        # DECODE
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Output : 256x8x8

        self.dec1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),                # Output : 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Second upsampling block
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 8x8 -> 16x16
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128 + 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Third upsampling block
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 + 64
            nn.BatchNorm2d(64),     
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )                                                                 # Output dec3 : 64x32x32

        # Final layer
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)                 # Output final layer : 1x32x32

    def forward(self, x):
        
        x1 = self.enc1(x)   # 12 -> 64 channels
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)  # 64 -> 128 channels
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)  # 128 -> 256 channels
        p3 = self.pool3(x3)

        bottleneck = self.bottleneck(p3)

        x = self.up1(bottleneck)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.dec1(x)
        
        # Decoder2: Upsample + Concatenate + Conv
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.dec2(x)
        
        # Decoder3: Upsample + Concatenate + Conv
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)  # Skip connection
        x = self.dec3(x)
        
        # Final layer
        output = self.final_conv(x)

        return output