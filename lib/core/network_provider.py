import torch.nn as nn


class MergeBackboneHead(nn.Module):
    def __init__(self, backbone, head):
        super(MergeBackboneHead, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
