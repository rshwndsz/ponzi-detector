import torch
import torch.nn as nn
import os

from code.config import config as cfg


class _ConvBNReluUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ConvBNReluUp, self).__init__()
        self.encode = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        ])
        self.dropout = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.dropout(self.encode(x)))


class _ConvBNReluDw(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(_ConvBNReluDw, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.decode(x)


class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.enc1 = _ConvBNReluUp(input_size, 32)
        self.enc2 = _ConvBNReluUp(32, 64)
        self.enc3 = _ConvBNReluUp(64, 128)
        self.dec1 = _ConvBNReluDw(128, 64, 32)
        self.dec2 = _ConvBNReluDw(32, 16, 8)

        self.fc = nn.Linear(8*8, output_size)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.fc(x.view(-1, 1))
        return x


class PonziNet(nn.Module):
    def __init__(self):
        super(PonziNet, self).__init__()
        # Load conv_net
        checkpoint = torch.load(os.path.join(cfg.model_path_root, 'conv_net', 'best_model.pth'))
        self.conv_net.load(checkpoint['conv_net_state_dict'])
        # Freeze conv_net
        for param in self.conv_net.parameters():
            param.requires_grad = False

        # Add classifier
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv_net(x)
        return self.classifier(x)
