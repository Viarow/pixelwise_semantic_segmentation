import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(Decoder, self).__init__()

        self.down_layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            self.down_layers.append(nn.Dropout(0.5))

        self.down_layers.append(nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.down = nn.Sequential(*self.down_layers)

    def forward(self, input):

        return self.down(input)


class Encoder(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(Encoder, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return  self.up(input)


class UNet(nn.Module):

    def __init__(self, classes_num):
        super(UNet, self).__init__()

        self.dec1 = Decoder(3, 64)
        self.dec2 = Decoder(64, 128)
        self.dec3 = Decoder(128, 256)
        self.dec4 = Decoder(256, 512, dropout=True)

        self.bottom = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.enc4 = Encoder(1024, 512, 256)
        self.enc3 = Encoder(512, 256, 128)
        self.enc2 = Encoder(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True)
        )

        self.score = nn.Conv2d(64, classes_num, 1)

    def forward(self, input):

        down1 = self.dec1(input)
        down2 = self.dec2(down1)
        down3 = self.dec3(down2)
        down4 = self.dec4(down3)
        bottom = self.bottom(down4)

        upsample4 = F.interpolate(down4, bottom.size()[2:], mode='bilinear', align_corners=True)
        enc4 = self.enc4(torch.cat([bottom, upsample4], 1))

        upsample3 = F.interpolate(down3, enc4.size()[2:], mode='bilinear', align_corners=True)
        enc3 = self.enc3(torch.cat([enc4, upsample3], 1))

        upsample2 = F.interpolate(down2, enc3.size()[2:], mode='bilinear', align_corners=True)
        enc2 = self.enc2(torch.cat([enc3, upsample2], 1))

        upsample1 = F.interpolate(down1, enc2.size()[2:], mode='bilinear', align_corners=True)
        enc1 = self.enc1(torch.cat([enc2, upsample1], 1))

        output = F.interpolate(self.score(enc1), input.size()[2:], mode='bilinear', align_corners=True)

        return output

