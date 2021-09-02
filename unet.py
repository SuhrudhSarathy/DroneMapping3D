# The unet architecture is implemented here

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from torchvision import transforms

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.conv(X)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Store the down conv layers
        self.downs = nn.ModuleList()
        
        # Store the up conv layers
        self.ups_conv_transpose = nn.ModuleList()
        self.ups_double_convs = nn.ModuleList()

        # Down convs
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up conv
        for feature in reversed(features):
            self.ups_conv_transpose.append(
                nn.ConvTranspose2d(feature*2, feature, 2, 2)
            )
            self.ups_double_convs.append(DoubleConv(feature*2, feature))

        
        # MAx Pool Layer
        self.max_pool = nn.MaxPool2d(2, 2)

        self.bottom_layer = DoubleConv(features[-1], features[-1]*2)

        self.last_conv_layer = nn.Conv2d(features[0], self.out_channels, 1, 1)

    def forward(self, X):
        # Move the data forward through the net
        
        # make a list to store the outputs for copy and cropping
        out_tensors = []

        # Go down and copy
        for down in self.downs:
            X = down(X)
            out_tensors.append(X)

            # pool
            X = self.max_pool(X)

        # Bottom Layer
        X = self.bottom_layer(X)

        for up_conv, double_conv, out_tensor in zip(self.ups_conv_transpose, self.ups_double_convs, reversed(out_tensors)):
            # Upconv
            X = up_conv(X)

            # Crop and Concatenate
            # Check for size issues and solve this
            if X.shape != out_tensor.shape:
                X = TF.resize(X, out_tensor.shape[2:])

            X = torch.cat((out_tensor, X), dim=1)

            # Double Conv
            X = double_conv(X)

        X = self.last_conv_layer(X)

        return X

if __name__ == "__main__":
    net = UNet()

    a = torch.randn((1, 3, 512, 512))

    print(net(a).shape)
