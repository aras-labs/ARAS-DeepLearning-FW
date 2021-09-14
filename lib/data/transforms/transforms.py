
from abc import ABC, abstractmethod

import torch
import torchvision

class abstract_transform(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, img, seg, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Compos(abstract_transform):
    def __init__(self, tf_list=[]):
        super().__init__()
        self.tf_list = tf_list

    def add_transform(self, tf):
        self.tf_list.append(tf)

    def forward(self, image, seg):
        for T in self.tf_list:
            image, seg = T(image, seg)
        return image, seg


class ToTensor(abstract_transform):
    def forward(self, image, seg, dtype=torch.float, *args, **kwargs):
        return torch.flip(torch.from_numpy(image).type(dtype).permute(2, 0, 1), [0]), torch.from_numpy(seg).type(torch.long)


class Normalize(abstract_transform):
    def forward(self, image, seg, *args, **kwargs):
        return image / 255, seg


class RandomHorizontalFlip(abstract_transform):
    def forward(self, image, seg, *args, **kwargs):
        random = torch.rand(1)
        if random[0]>.5:
            return torch.flip(image, [2]) , torch.flip(seg, [1])
        return image, seg


class RandomVerticalFlip(abstract_transform):
    def forward(self, image, seg, *args, **kwargs):
        random = torch.rand(1)
        if random[0]>.5:
            return torch.flip(image, [1]), torch.flip(seg, [0])
        return image, seg


class ColorJitter(abstract_transform):
    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def foward(self, image, target):
        image = self.color_jitter(image)
        return image, target
