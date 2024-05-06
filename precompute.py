import torch
import os
import logging
from torchvision.transforms import Compose, Resize, CenterCrop
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from datasets import MimicCXRDataset, get_dataset
from datasets.utils import load_config
from radbert_pipe import FrozenRadBERTPipe

# Assuming necessary imports and class definitions like `MimicCXRDataset` are available here.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define a function to instantiate the dataset and run the precompute process
def precompute_images(config_path, model):


    config = load_config(config_path)
    dataset = get_dataset(config, "train")

    # Run the precompute process using the provided model
    dataset.load_precomputed(model)




if __name__ == "__main__":
    # Example usage of the precompute_images function

    CONFIG_PATH= '/vol/ideadata/ce90tate/cxr_phrase_grounding/config_msxcr.yml'

    vae = FrozenRadBERTPipe().pipe.vae
    vae.requires_grad_(False)
    precompute_images(CONFIG_PATH, vae)
