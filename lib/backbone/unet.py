import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, init_features=32, pretrained=False):
        super(Unet, self).__init__()
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                    in_channels=in_channels, out_channels=out_channels,
                                    init_features=init_features, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)
