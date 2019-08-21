import torch.nn as nn
import torch.optim as optim

from code.config import config as cfg
from code.models import SampleNet


model_name = 'SampleNet'
model = SampleNet().to(cfg.device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# Hyper-parameters
batch_size = 1
n_epochs = 2
lr = 0.01
