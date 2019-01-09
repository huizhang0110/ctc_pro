import torch
import torch.nn as nn
from resnet import ResNet50Tiny


class BiLSTM(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        h, c = self.rnn(x)
        out = self.fc(h)
        return out


class CTCModel(nn.Module):
    def __init__(self, n_classes, n_hidden=256):
        super(CTCModel, self).__init__()
        self.cnn = ResNet50Tiny()
        self.rnn = nn.Sequential(
                BiLSTM(512, n_hidden, n_hidden),
                BiLSTM(n_hidden, n_hidden, n_classes))

    def forward(self, x):
        x = self.cnn(x).squeeze(dim=2)
        x = x.permute(0, 2, 1)
        x = self.rnn(x)
        return x


if __name__ == "__main__":
    x = torch.randn(3, 3, 32, 32)
    net = CTCModel(10)
    x = net(x) 
    print(x.shape)
