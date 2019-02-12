import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SegNetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, mid_layers=True):
        super(SegNetEncoder, self).__init__()

        layers = [
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
        ]

        if mid_layers:
            layers += [
                nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1),
                nn.BatchNorm2d(in_channels//2),
                nn.ReLU(inplace=True),
            ]

        layers += [
            nn.Conv2d(in_channels//2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        self.encoder = nn.Sequential(*layers)

    def forward(self, input):
        return self.encoder(input)


class SegNet(nn.Module):

    def __init__(self, classes_num):
        super(SegNet, self).__init__()
        features = models.vgg16(pretrained=True).features

        self.dec1 = features[0:4]
        self.dec2 = features[5:9]
        self.dec3 = features[10:16]
        self.dec4 = features[17:23]
        self.dec5 = features[24:-1]

        self.enc5 = SegNetEncoder(512, 512)
        self.enc4 = SegNetEncoder(512, 256)
        self.enc3 = SegNetEncoder(256, 128)
        self.enc2 = SegNetEncoder(128, 64, mid_layers=False)

        self.score = nn.Sequential(
            nn.Conv2d(64, classes_num, 3, padding=1),
            nn.BatchNorm2d(classes_num),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):

        x1 = self.dec1(input)
        p1, indices1 = F.max_pool2d(x1, 2, 2, return_indices=True)
        x2 = self.dec2(p1)
        p2, indices2 = F.max_pool2d(x2, 2, 2, return_indices=True)
        x3 = self.dec3(p2)
        p3, indices3 = F.max_pool2d(x3, 2, 2, return_indices=True)
        x4 = self.dec4(p3)
        p4, indices4 = F.max_pool2d(x4, 2, 2, return_indices=True)
        x5 = self.dec5(p4)
        p5, indices5 = F.max_pool2d(x5, 2, 2, return_indices=True)

        upsample5 = self.enc5(F.max_unpool2d(p5, indices5, 2, 2, output_size=x5.size()))
        upsample4 = self.enc4(F.max_unpool2d(upsample5, indices4, 2, 2, output_size=x4.size()))
        upsample3 = self.enc3(F.max_unpool2d(upsample4, indices3, 2, 2, output_size=x3.size()))
        upsample2 = self.enc2(F.max_unpool2d(upsample3, indices2, 2, 2, output_size=x2.size()))
        upsample1 = F.max_unpool2d(upsample2, indices1, 2, 2, output_size=x1.size())

        output = self.score(upsample1)

        return output
