import numpy as np
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

def color(n):

    color_image = np.zeros([n,3]).astype(np.uint8)

    for i in range(n):
        r, g, b = np.zeros(3)

        for j in range(8):
            r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
            g = g + (1 << (7 - j)) * ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
            b = b + (1 << (7 - j)) * ((i & (1 << (3 * j + 2))) >> (3 * j + 2))

        color_image[i] = np.array([r, g, b])
        # color[i, :] = np.array([r, g, b])

        return color_image

class Colorize:

    def __init__(self, classes_num):
        self.color_image = color(256)
        self.color_image[classes_num] = self.color_image[-1]
        self.color_image = torch.from_numpy(self.color_image[:classes_num])

    def __call__(self, gray_image):

        size = gray_image.size()
        output = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.color_image)):
            mask = gray_image[0] == label
            output[0][mask] = self.color_image[label][0]
            output[1][mask] = self.color_image[label][1]
            output[2][mask] = self.color_image[label][2]

        return output


class Relabel:

    def __init__(self, old_label, new_label):
        self.olabel = old_label
        self.nlabel = new_label

    def __call__(self, tensor):
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class Tolabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)
