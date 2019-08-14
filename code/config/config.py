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


# Data-specific
project_root = os.getcwd()
dataset_root = os.path.join(project_root, 'datasets', 'dataset_name')
model_path = os.path.join(project_root, 'checkpoints', 'model_best.pth')
results_dir = os.path.join(project_root, 'results')
