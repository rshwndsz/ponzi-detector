import torch
import torch.nn as nn
import argparse
import logging
import coloredlogs

from code.config import (config as cfg,
                         data_loaders as dl)
from code.models import Encoders, CNNs

# Setup colorful logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def train_conv_net():
    autoencoder = Encoders.StackedAutoEncoder().to(cfg.device)
    conv_net = CNNs.ConvNet(input_size=1,
                            output_size=16).to(cfg.device)
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    validation_losses = []
    running_train_loss = 0
    train_losses = []
    for epoch in cfg.n_epochs:
        conv_net.train()
        running_train_loss = 0
        for bytecode, tokens in dl.byt_token_trainloader:
            inputs = autoencoder(bytecode).to(cfg.device)
            targets = tokens.to(cfg.device)
            predictions = conv_net(inputs)

            loss = criterion(predictions, targets)
            running_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(running_train_loss/len(dl.byt_token_trainloader))

        if epoch % 5 == 0:
            # Validate every 5 epochs
            conv_net.eval()
            running_val_loss = 0
            for bytecode, tokens in dl.byt_token_valloader:
                running_val_loss = 0
                inputs = autoencoder(bytecode).to(cfg.device)
                targets = tokens.to(cfg.device)
                predictions = conv_net(inputs)

                loss = criterion(predictions, targets)
                running_val_loss += loss.item()
            validation_losses.append(running_val_loss/len(dl.byt_token_valloader))
            if validation_losses[-1] < min(validation_losses):
                torch.save({
                    'conv_net_state_dict': conv_net.state_dict(),
                    'autoencoder_state_dict': autoencoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'min_val_loss': validation_losses[-1]
                }, cfg.model_path_root + 'conv_net' + 'best_model.pth')


def train_ponzi_net():
    ponzi_net = CNNs.PonziNet().to(cfg.device)
    autoencoder = Encoders.StackedAutoEncoder().to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ponzi_net.parameters(), lr=3e-4)

    checkpoint = torch.load(cfg.model_path_root, 'conv_net', 'best_model.pth')
    autoencoder.load(checkpoint['autoencoder_state_dict'])

    for param in autoencoder.parameters():
        param.requires_grad = False

    validation_losses = []
    running_train_loss = 0
    train_losses = []
    for epoch in cfg.n_epochs:
        running_train_loss = 0
        ponzi_net.train()
        for bytecode, is_ponzi in dl.byt_ponzi_trainloader:
            inputs = autoencoder(bytecode).to(cfg.device)
            targets = is_ponzi
            predictions = ponzi_net(inputs)
            loss = criterion(predictions, targets)
            running_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(running_train_loss/len(dl.byt_ponzi_trainloader))

        if epoch % 5 == 0:
            ponzi_net.eval()
            running_val_loss = 0
            for bytecode, is_ponzi in dl.byt_ponzi_trainloader:
                running_val_loss = 0
                inputs = autoencoder(bytecode).to(cfg.device)
                targets = is_ponzi
                predictions = ponzi_net(inputs)
                loss = criterion(predictions, targets)
            validation_losses.append(running_val_loss/len(dl.byt_ponzi_valloader))
            if validation_losses[-1] < min(validation_losses):
                torch.save({
                    'ponzi_net_state_dict': ponzi_net.state_dict(),
                    'autoencoder_state_dict': autoencoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'min_val_loss': validation_losses[-1]
                }, cfg.model_path_root + 'ponzi_net' + 'best_model.pth')


def train():
    train_conv_net()
    train_ponzi_net()


def test():
    model = CNNs.PonziNet().to(cfg.device)
    autoencoder = Encoders.StackedAutoEncoder().to(cfg.device)
    checkpoint = torch.load(cfg.model_path_root + 'ponzi_net' + 'best_model.pth')
    model.load(checkpoint['ponzi_net_state_dict'])

    model.eval()
    for bytecode in dl.byt_ponzi_testloader:
        predictions = model(autoencoder(bytecode))
        print(predictions)


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description=f'CLI')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()

    if args.phase == 'train':
        train()

    elif args.phase == 'test':
        test()

    else:
        raise ValueError('Choose one of train/validate/test')
