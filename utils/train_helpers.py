import torch
from config import config as cfg
import logging
import coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def save_val_model(model, optimizer, **kwargs):
    """Save model after validation"""
    try:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        save_dict.update(kwargs)
        torch.save(save_dict, cfg.model_path)
    except FileNotFoundError as fnf_error:
        logger.error(f'{fnf_error}')
    else:
        logger.info('Saved model!')
