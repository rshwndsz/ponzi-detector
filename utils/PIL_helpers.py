from PIL import Image


def pil_loader(path, mode='RGB'):
    with open(path, 'rb') as f:
        return Image.open(f).convert(mode=mode)
