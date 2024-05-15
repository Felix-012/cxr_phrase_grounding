import os.path
import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, UNet2DConditionModel, \
    StableDiffusionInpaintPipeline
from utils.attention_maps import (
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
)


def _freeze(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

def _load_text_encoder(component_name, path, torch_dtype, variant):
    return AutoModel.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,variant=variant)

def _load_unet(component_name, path, torch_dtype, variant):
    return UNet2DConditionModel.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype,variant=variant)

def _load_tokenizer(component_name, path, torch_dtype, variant):
    return AutoTokenizer.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype, variant=variant)

def _load_scheduler(component_name, path, torch_dtype, variant):
    return DDPMScheduler.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype, variant=variant)

def _load_vae(component_name, path, torch_dtype, variant):
    return AutoencoderKL.from_pretrained(path, subfolder=component_name, torch_dtype=torch_dtype, variant=variant)


class FrozenCustomPipe:
    def __init__(self, use_freeze=True, path="/vol/ideadata/ce90tate/cxr_phrase_grounding/components",variant=None,
                 llm_name="radbert", torch_dtype=torch.float32, device="cuda", save_attention=False, inpaint=False, accelerator=None):
        self.device = device
        component_loader = {
            "text_encoder": _load_text_encoder,
            "tokenizer": _load_tokenizer,
            "unet": _load_unet,
            "vae": _load_vae,
            "scheduler": _load_scheduler,
            }
        component_mapper = {}

        for component_name in component_loader.keys():
            if accelerator:
                accelerator.print(f"Loading {component_name}...")
            else:
                print(f"Loading {component_name}...")
            if component_name == "tokenizer" or component_name == "text_encoder":
                component = component_loader.get(component_name)(component_name, os.path.join(path, llm_name),
                                                                 torch_dtype, variant)
            else:
                component = component_loader.get(component_name)(component_name, path, torch_dtype, variant)
            component_mapper[component_name] = component

        if use_freeze:
            _freeze(component_mapper["text_encoder"])

        if accelerator:
            accelerator.print("Building custom pipeline...")
        else:
            print("Building custom pipeline...")

        if inpaint:
            pipe = StableDiffusionInpaintPipeline(unet=component_mapper["unet"], text_encoder=component_mapper["text_encoder"],
                                       tokenizer=component_mapper["tokenizer"], vae=component_mapper["vae"],
                                       scheduler=component_mapper["scheduler"],safety_checker=None, feature_extractor=None,                             requires_safety_checker=False)
        else:
            pipe = StableDiffusionPipeline(unet=component_mapper["unet"], text_encoder=component_mapper["text_encoder"],
                                       tokenizer=component_mapper["tokenizer"], vae=component_mapper["vae"],
                                       scheduler=component_mapper["scheduler"],safety_checker=None, feature_extractor=None,
                                       requires_safety_checker=False)

        if save_attention:
            cross_attn_init()
            pipe.unet = set_layer_with_name_and_path(pipe.unet)
            pipe.unet = register_cross_attention_hook(pipe.unet)

        self.pipe = pipe.to(device)
