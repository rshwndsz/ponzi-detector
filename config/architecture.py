import torch.nn.functional as F
import torch.optim as optim

from . import config as cfg
from models import SampleNet


model_name = 'SampleNet'
model = SampleNet().to(cfg.device)
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters())


# Hyper-parameters
batch_size = 1
n_epochs = 2
lr = 0.01
