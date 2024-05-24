
import logging


from datasets import get_dataset
from datasets.utils import load_config
from custom_pipe import FrozenCustomPipe

# Assuming necessary imports and class definitions like `MimicCXRDataset` are available here.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define a function to instantiate the dataset and run the precompute process
def precompute_images(config_path, model):


    config = load_config(config_path)
    dataset = get_dataset(config, "test")

    # Run the precompute process using the provided model
    dataset.load_precomputed(model)




if __name__ == "__main__":
    # Example usage of the precompute_images function

    CONFIG_PATH= '/vol/ideadata/ce90tate/cxr_phrase_grounding/configs/config_msxcr.yml'

    vae = FrozenCustomPipe(path="/vol/ideadata/ce90tate/cxr_phrase_grounding/components").pipe.vae
    vae.requires_grad_(False)
    precompute_images(CONFIG_PATH, vae)
