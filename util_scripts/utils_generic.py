"""addapted from https://github.com/MischaD/chest-distillation"""

from enum import Enum

import numpy as np
import torch
import torchvision
from einops import rearrange


class DatasetSplit(Enum):
    train="train"
    test="test"
    val="val"
    mscxr="mscxr"
    p19="p19"
    all="all"

def resize_long_edge(img, size_long_edge):
    # torchvision resizes so shorter edge has length - I want longer edge to have spec. length
    assert img.size()[-3] == 3, "Channel dimension expected at third position"
    img_longer_edge = max(img.size()[-2:])
    img_shorter_edge = min(img.size()[-2:])
    resize_factor = size_long_edge / img_longer_edge

    # resized_img = torchvision.transforms.functional.resize(img_longer_edge/img_shorter_edge)
    resize_to = img_shorter_edge * resize_factor
    resizer = torchvision.transforms.Resize(size=round(resize_to))
    return resizer(img)[..., :size_long_edge, :size_long_edge]


SPLIT_TO_DATASETSPLIT = {0:DatasetSplit("test"), 1:DatasetSplit("train"), 2:DatasetSplit("val"), 3:DatasetSplit("p19"), 4:DatasetSplit("mscxr")} #p19 - 3
DATASETSPLIT_TO_SPLIT = {"test":0, "train":1, "val":2, "p19":3, "mscxr":4}#p19 - 3


def collate_batch(batch):
    # make list of dirs to dirs of lists with batchlen
    batched_data = {}
    for data in batch:
        # label could be img, label, path, etc
        for key, value in data.items():
            if batched_data.get(key) is None:
                batched_data[key] = []
            batched_data[key].append(value)

    # cast to torch.tensor
    for key, value in batched_data.items():
        if isinstance(value[0],torch.Tensor):
            if value[0].size()[0] != 1:
                for i in range(len(value)):
                    value[i] = value[i][None,...]
            # check if concatenatable
            if all([value[0].size() == value[i].size() for i in range(len(value))]):
                batched_data[key] = torch.concat(batched_data[key])
    return batched_data


def img_to_viz(img):
    img = rearrange(img, "1 c h w -> h w c")
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = np.array(((img + 1) * 127.5), np.uint8)
    return img

