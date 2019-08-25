import os
import torch


# Torch-specific
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
num_workers = 4


# Train/val-specific
print_freq = 20
val_freq = 1
resume_from_epoch = 0
min_val_loss = 1000

batch_size = 8
n_epochs = 10
initial_lr = 3e-4


# Data-specific
project_root = '/Users/Russel/myProjects/dl-for-blockchain/code/'
dataset_root = os.path.join(project_root, 'dataset')
model_path_root = os.path.join(project_root, 'checkpoints')
results_dir = os.path.join(project_root, 'results')
