"""adapted from https://github.com/MischaD/chest-distillation"""

from einops import rearrange, repeat
import torch
from functools import partial


def all_mean(inp):
    return inp.mean(dim=(0, 1, 2), keepdim=True)


def diffusion_steps_mean(x, steps):
    assert x.size()[2] == 1
    return x[-steps:, :, 0].mean(dim=(0, 1), keepdim=True)


def relevant_token_step_mean(x, tok_idx, steps):
    return x[-steps:, :, tok_idx:(tok_idx+1)].mean(dim=(0, 1), keepdim=True)


def all_token_mean(x, steps, max_token=None):
    return x[-steps:,:,:max_token].mean(dim=(0, 1), keepdim=True)


def multi_relevant_token_step_mean(x, tok_idx, steps):
    res = None
    for tok_id in tok_idx:
        if res is None:
            res = x[-steps:, :, tok_id:(tok_id+1)].mean(dim=(0, 1), keepdim=True)
        else:
            res += x[-steps:, :, tok_id:(tok_id+1)].mean(dim=(0, 1), keepdim=True)

    res = res.mean(dim=(0,1), keepdim=True)
    return res


class AttentionExtractor:
    def __init__(self, function=None, *args, **kwargs):
        if isinstance(function, str):
            self.reduction_function = getattr(self, function)
        else:
            self.reduction_function = function

        if args or kwargs:
            self.reduction_function = partial(self.reduction_function, *args, **kwargs)

    def __call__(self, inp, *args, **kwargs):
        """ Called with: Iterations x Layers x Channels X Height x Width

        :param inp: tensor
        :return: attention map
        """
        assert inp.ndim == 5
        out = self.reduction_function(inp, *args, **kwargs)
        assert out.ndim == 5
        return out


def print_attention_info(attention):
    print(f"Num Forward passes: {len(attention)}, Depth:{len(attention[0])}")
    for i in range(len(attention[0])):
        print(f"Layer: {i} - {attention[0][i].size()}")

def normalize_attention_map_size(attention_maps, on_cpu=False):
    if on_cpu:
        for layer_key, layer_list in attention_maps.items():
            for iteration in range(len(layer_list)):
                attention_maps[layer_key][iteration] = attention_maps[layer_key][iteration].to("cpu")
    for key, layer in attention_maps.items():  # trough layers / diffusion steps
        for iteration in range(len(layer)):
            attention_map = attention_maps[key][iteration]  # B x num_resblocks x numrevdiff x H x W
            if attention_map.size()[-1] != 64:
                upsampling_factor = 64 // attention_map.size()[-1]
                attention_map = repeat(attention_map, 'b tok h w -> b tok (h h2) (w w2)', h2=upsampling_factor,
                                       w2=upsampling_factor)
            attention_maps[key][iteration] = attention_map
    attention_maps = torch.cat([torch.stack(lst).unsqueeze(0) for lst in list(attention_maps.values())], dim=0)
    attention_maps = rearrange(attention_maps, "layer depth b tok h w -> b layer depth tok h w")
    return attention_maps


def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)


def preprocess_attention_maps(attention_masks, on_cpu=None):
    return normalize_attention_map_size(attention_masks, on_cpu)