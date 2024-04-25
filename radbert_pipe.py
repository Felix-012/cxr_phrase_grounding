import os

os.environ['HF_HOME'] = '/vol/ideadata/ce90tate/.cache'

import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, UNet2DConditionModel


def _freeze(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

def load_text_encoder(component_name, path, torch_dtype, variant):
    return AutoModel.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,variant=variant)

def load_unet(component_name, path, torch_dtype, variant):
    return UNet2DConditionModel.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,variant=variant)

def load_tokenizer(component_name, path, torch_dtype, variant):
    return AutoTokenizer.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype, variant=variant)

def load_scheduler(component_name, path, torch_dtype, variant):
    return DDPMScheduler.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype, variant=variant)

def load_vae(component_name, path, torch_dtype, variant):
    return AutoencoderKL.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype, variant=variant)


class FrozenCustomPipe:
    def __init__(self, use_freeze=True, path="/vol/ideadata/ce90tate/chest-distillation/src/lora/components",variant=None,
                 torch_dtype=torch.float32, device="cuda"):

        component_loader = {
            "text_encoder": load_text_encoder,
            "tokenizer": load_tokenizer,
            "unet": load_unet,
            "vae": load_vae,
            "scheduler": load_scheduler,
            }
        component_mapper = {}

        for component_name in component_loader.keys():
            print(f"Loading {component_name}...")
            component = component_loader.get(component_name)(component_name, path, torch_dtype, variant)
            component_mapper[component_name] = component

        if use_freeze:
            _freeze(component_mapper["text_encoder"])

        print("Ensembling custom pipeline...")
        pipe = StableDiffusionPipeline(unet=component_mapper["unet"], text_encoder=component_mapper["text_encoder"],
                                       tokenizer=component_mapper["tokenizer"], vae=component_mapper["vae"],
                                       scheduler=component_mapper["scheduler"],safety_checker=None, feature_extractor=None,
                                       requires_safety_checker=False)
        self.pipe = pipe.to(device)


