import torch
import torch.nn as nn
from resnet import ResNet50Tiny
from lib.non_local_embedded_gaussian import NONLocalBlock1D


class CTCModel(nn.Module):
    def __init__(self, n_classes):
        super(CTCModel, self).__init__()
        self.cnn = ResNet50Tiny()
        self.non_local = NONLocalBlock1D(512, sub_sample=False, bn_layer=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x).squeeze(dim=2)
        x = self.non_local(x).permute(0, 2, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    x = torch.randn(3, 3, 32, 32)
    net = CTCModel(10)
    x = net(x) 
    print(x.shape)
