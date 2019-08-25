import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from code.config import architecture as arch


class ConvDenoisingAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size, stride):
        super(ConvDenoisingAutoEncoder, self).__init__()

        self.forward = nn.ModuleList([
            nn.Conv2d(input_size,
                      output_size,
                      kernel_size=2,
                      stride=stride,
                      padding=0),
            nn.ReLU()
        ])
        self.backward = nn.ModuleList([
            nn.ConvTranspose2d(output_size,
                               input_size,
                               kernel_size=2,
                               stride=2,
                               padding=0),
            nn.ReLU(),
        ])

        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=arch.lr)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original loss-less input as the target.
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        y = self.forward(x_noisy)

        if self.training:
            x_reconstruct = self.backward(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach()

    def reconstruct(self, x):
        return self.backward(x)


class StackedAutoEncoder(nn.Module):
    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = ConvDenoisingAutoEncoder(3, 128, 2)
        self.ae2 = ConvDenoisingAutoEncoder(128, 256, 2)
        self.ae3 = ConvDenoisingAutoEncoder(256, 512, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
        a2_reconstruct = self.ae3.reconstruct(x)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct
