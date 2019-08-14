import torch
import argparse
import logging
import coloredlogs

from config import (config as cfg,
                    architecture as arch,
                    data_loaders as dl)

# Setup colorful logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description=f'CLI for {arch.model_name}')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--load', type=bool, default=False)
    args = parser.parse_args()

    if args.phase == 'train':
        pass

    elif args.phase == 'test':
        pass

    else:
        raise ValueError('Choose one of train/validate/test')
