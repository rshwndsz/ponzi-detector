import torch
import torch.nn as nn
import os

from code.config import config as cfg


class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()

    def forward(self, x):
        pass


class PonziNet(nn.Module):
    def __init__(self):
        super(PonziNet, self).__init__()
        # Load conv_net
        self.conv_net = torch.load(os.path.join(cfg.model_path_root, 'conv_net', 'best_model'))
        # Freeze conv_net
        for param in self.conv_net.parameters():
            param.requires_grad = False

        # Add classifier
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv_net(x)
        return self.classifier(x)
