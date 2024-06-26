"""addapted from https://github.com/MischaD/chest-distillation"""

import torch
import scipy.ndimage as ndimage
import cv2
from einops import rearrange
import yaml
from ml_collections import ConfigDict
import os.path

def path_to_tensor(path, normalize=True):
    if not os.path.isfile(path): raise FileExistsError
    img = torch.tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=torch.float32)
    if normalize:
        img = ((img / 127.5) - 1)
    img = rearrange(img, "h w c -> 1 c h w")
    return img


def file_to_list(path):
    with open(path) as fp:
        lines = fp.readlines()
    return lines


def resize(img, tosize):
    """resize height and width of dataset that is too large"""
    assert img.ndim == 4
    b, c, h, w = img.size()
    max_size = max(h, w)

    zoom_factor = tosize / max_size

    return torch.tensor(ndimage.zoom(img, (1, 1, zoom_factor,zoom_factor)))

def load_config(filename: str) -> ConfigDict:
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    return ConfigDict(data, type_safe=False)



