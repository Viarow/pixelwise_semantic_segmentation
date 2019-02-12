
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class FCN8(nn.Module):

    def __init__(self, classes_num):
        super(FCN8, self).__init__()

        blocks = models.vgg16(pretrained=True).features
        # blocks = list(models.vgg16(pretrained=True).features.children())

        self.block1 = nn.Sequential(*blocks[0:17])
        self.block2 = nn.Sequential(*blocks[17:24])
        self.block3 = nn.Sequential(*blocks[24:31])


        self.block1_score = nn.Conv2d(256, classes_num, 1)
        self.block2_score = nn.Conv2d(512, classes_num, 1)
        self.fc_score = nn.Conv2d(4096, classes_num, 1)

        # maybe try PReLU
        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096,4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )


    def forward(self, input):
        map1 = self.block1(input)
        map2 = self.block2(map1)
        map3 = self.block3(map2)
        original_score = self.fc(map3)

        score1 = self.block1_score(map1)
        score2 = self.block2_score(map2)
        score3 = self.fc_score(original_score)

        score = F.interpolate(score3, score2.size()[2:], mode='bilinear', align_corners=True)
        score += score2
        score = F.interpolate(score, score1.size()[2:], mode='bilinear', align_corners=True)
        score += score1

        output = F.interpolate(score, input.size()[2:], mode='bilinear', align_corners=True)

        return output


class FCN16(nn.Module):

    def __init__(self, classes_num):
        super(FCN16, self).__init__()

        blocks = models.vgg16(pretrained=True).features

        self.block1 = nn.Sequential(*blocks[0:24])
        self.block2 = nn.Sequential(*blocks[24:31])

        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.block1_score = nn.Conv2d(512, classes_num, 1)
        self.fc_score = nn.Conv2d(4096, classes_num, 1)

    def forward(self, input):
        map1 = self.block1(input)
        map2 = self.block2(map1)
        original_score = self.fc(map2)

        score1 = self.block1_score(map1)
        score2 = self.fc_score(original_score)

        score = F.interpolate(score2, score1.size()[2:], mode='bilinear', align_corners=True)
        score += score1

        output = F.interpolate(score, input.size()[2:], mode='bilinear', align_corners=True)

        return output


class FCN32(nn.Module):

    def __init__(self, classes_num):
        super(FCN32, self).__init__()

        self.vgg = models.vgg16(pretrained=True).features

        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.vgg_score = nn.Conv2d(4096, classes_num, 1)

    def forward(self, input):

        vggmap = self.vgg(input)
        original_score = self.fc(vggmap)

        score = self.vgg_score(original_score)

        output = F.interpolate(score, input.size()[2:], mode='bilinear', align_corners=True)

        return output





