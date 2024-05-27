"""addapted from https://github.com/MischaD/chest-distillation"""

import argparse
import os
import shutil
import time

from datasets.dataset import add_preliminary_to_sample

os.environ["TOKENIZERS_PARALLELISM"]="false"
import json
import numpy as np
import pandas as pd
import torchvision
import torch
from torch import autocast
from datasets import get_dataset
from util_scripts.foreground_masks import GMMMaskSuggestor
from util_scripts.preliminary_masks import preprocess_attention_maps
from visualization.utils import word_to_slice
from visualization.utils import MIMIC_STRING_TO_ATTENTION
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from util_scripts.utils_generic import collate_batch
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from mpl_toolkits.axes_grid1 import ImageGrid
from evaluation.utils import compute_prediction_from_binary_mask
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.utils import load_config
from evaluation.utils import check_mask_exists, samples_to_path, contrast_to_noise_ratio
from torch.utils.data.distributed import DistributedSampler
from custom_pipe import FrozenCustomPipe
from util_scripts.attention_maps import curr_attn_maps, all_attn_maps
from log import logger
from accelerate import Accelerator


def compute_masks(rank, config, world_size, use_lora):
    logger.info(f"Current rank: {rank}")
    if config.phrase_grounding_mode:
        config["phrase_grounding"] = True
    else:
        config["phrase_grounding"] = False

    lora_weights = config.lora_weights
    dataset = get_dataset(config, "test")
    accelerator = Accelerator()


    pipeline = FrozenCustomPipe(path=config.component_dir, save_attention=True, inpaint=True).pipe

    if use_lora:
        pipeline.load_lora_weights(lora_weights)
    else:
        accelerator.load_state(config.checkpoint)

    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    device = torch.device(rank) if torch.cuda.is_available() else torch.device("cpu")
    model = pipeline.to(device)

    if config.filter_bad_impressions:
        if config.phrase_grounding_mode:
            logger.warning("Filtering cannot be combined with phrase grounding")
        dataset.apply_filter_for_disease_in_txt()

    dataset.load_precomputed(model.vae)

    seed_everything(int(time.time()))
    precision_scope = autocast

    cond_key = config.cond_stage_key if hasattr(config, "cond_stage_key") else "label_text"

    data_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    logger.info(f"Relative path to first sample: {dataset[0]['rel_path']}")

    dataloader = DataLoader(dataset,
                            batch_size=config.sample.iou_batch_size,
                            shuffle=False,
                            num_workers=0,  #opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            sampler=data_sampler,
                            )

    if hasattr(config, "mask_dir"):
        mask_dir = config.mask_dir
    else:
        mask_dir = os.path.join(config.log_dir, "preliminary_masks")
    logger.info(f"Mask dir: {mask_dir}")
    for samples in tqdm(dataloader, "generating masks"):
        with torch.no_grad():
            with precision_scope("cuda"):
                if check_mask_exists(mask_dir, samples):
                    logger.info(f"Masks already exists for {samples['rel_path']}")
                    continue
                if len(samples["img"]) < config.sample.iou_batch_size:
                    continue
                samples[cond_key] = [str(x.split("|")[0]) for x in samples[cond_key]]
                samples["impression"] = samples[cond_key]
                mask = torch.ones((config.sample.iou_batch_size, config.sample.latent_C, config.sample.latent_W, config.sample.latent_H)).to(pipeline.device)
                latents = [sample.latent_dist.sample() * pipeline.vae.scaling_factor for sample in samples["img"]]
                curr_attn_maps.clear()
                all_attn_maps.clear()
                pipeline(prompt=samples["impression"], mask_image=mask, image=latents, num_inference_steps=30, cross_attention_kwargs={"scale": 0.9})
                attention_images = preprocess_attention_maps(all_attn_maps, on_cpu=True)

                for j, attention in enumerate(list(attention_images)):
                    tok_attentions = []
                    txt_label = samples[cond_key][j]
                    # determine tokenization
                    txt_label = txt_label.split("|")[0]
                    words = txt_label.split(" ")
                    if not isinstance(words, list):
                        words = list(words)
                    assert isinstance(words[0], str)
                    outs = pipeline.tokenizer(words, padding="max_length",
                                        max_length=pipeline.tokenizer.model_max_length,
                                        truncation=True,
                                         return_tensors="pt")["input_ids"]
                    token_lens = []
                    for out in outs:
                        out = list(filter(lambda x: x != 0, out))
                        token_lens.append(len(out)-2)

                    token_positions = list(np.cumsum(token_lens) + 1)
                    token_positions = [1,] + token_positions
                    label = samples["finding_labels"][j]
                    query_words = MIMIC_STRING_TO_ATTENTION[label]
                    locations = word_to_slice(txt_label.split(" "), query_words)
                    if len(locations) == 0:
                        # use all
                        tok_attention = attention[:,:,token_positions[0]:token_positions[-1]]
                        tok_attentions.append(tok_attention.mean(dim=(0,1,2)))
                        # plt.imsave(f"attn_map_all_{j}.jpg", tok_attention.mean(dim=(0, 1, 2)))
                        # plt.imsave(f"attn_map_all_up_{j}.jpg",
                        #            attention[:8, :, token_positions[0]:token_positions[-1]].mean(
                        #                dim=(0, 1, 2)))
                        # plt.imsave(f"attn_map_all_mid_{j}.jpg",
                        #            attention[8:9, :, token_positions[0]:token_positions[-1]].mean(
                        #                dim=(0, 1, 2)))
                        # plt.imsave(f"attn_map_all_down_{j}.jpg",
                        #            attention[9:, :, token_positions[0]:token_positions[-1]].mean(
                        #                dim=(0, 1, 2)))
                        #vis(tok_attention.mean(dim=(0,1,2)))
                    else:
                        #i = 0
                        for location in locations:
                            tok_attention = attention[:,:,token_positions[location]:token_positions[location+1]]
                            tok_attentions.append(tok_attention.mean(dim=(0,1,2)))
                            # plt.imsave(f"attn_map_{query_words[i]}.jpg", tok_attention.mean(dim=(0,1,2)))
                            # plt.imsave(f"attn_map_up_{query_words[i]}.jpg", attention[:8,:,token_positions[location]:token_positions[location+1]].mean(dim=(0, 1, 2)))
                            # plt.imsave(f"attn_map_mid_{query_words[i]}.jpg", attention[8:9,:,token_positions[location]:token_positions[location+1]].mean(dim=(0, 1, 2)))
                            # plt.imsave(f"attn_map_down_{query_words[i]}.jpg", attention[9:,:,token_positions[location]:token_positions[location+1]].mean(dim=(0, 1, 2)))
                            # plt.show()
                            #vis(attention[7:8,:,token_positions[location]:token_positions[location+1]].mean(dim=(0,1,2)))
                            #vis(samples["bbox_img"][j])
                            #vis(tok_attention.mean(dim=(0,1,2)))
                            #i += 1

                    preliminary_attention_mask = torch.stack(tok_attentions).mean(dim=(0))
                    path = samples_to_path(mask_dir, samples, j)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    logger.info(f"(rank({rank}): Saving attention mask to {path}")
                    torch.save(preliminary_attention_mask.to("cpu"), path)


def compute_iou_score(config):
    if config.phrase_grounding_mode:
        config["phrase_grounding"] = True
    else:
        config["phrase_grounding"] = False

    lora_weights = config.lora_weights
    dataset = get_dataset(config, "test")

    if config.filter_bad_impressions:
        if config.phrase_grounding_mode:
            logger.warning("Filtering cannot be combined with phrase grounding")
        dataset.apply_filter_for_disease_in_txt()
    pipeline = FrozenCustomPipe(path=config.component_dir, save_attention=True, inpaint=True).pipe
    pipeline.load_lora_weights(lora_weights)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    dataset.load_precomputed(pipeline)
    dataloader = DataLoader(dataset,
                            batch_size=config.sample.iou_batch_size,
                            shuffle=False,
                            num_workers=0,  #opt.num_workers,
                            collate_fn=collate_batch,
                            drop_last=False,
                            )

    seed_everything(config.sample.seed)

    if hasattr(config, "mask_dir"):
        mask_dir = config.mask_dir
    else:
        mask_dir = os.path.join(config.log_dir, "preliminary_masks")
    logger.info(f"Mask dir: {mask_dir}")

    dataset.add_preliminary_masks(mask_dir, sanity_check=False)
    mask_suggestor = GMMMaskSuggestor(config)
    log_some = 50
    results = {"rel_path":[], "finding_labels":[], "iou":[], "miou":[], "bboxiou":[], "bboxmiou":[], "distance":[], "top1":[], "aucroc": [], "cnr":[]}

    resize_to_imag_size = torchvision.transforms.Resize(512)
    for samples in tqdm(dataloader, "computing metrics"):
        samples["label_text"] = [str(x.split("|")[0]) for x in samples["label_text"]]
        samples["impression"] = samples["label_text"]

        for i in range(len(samples["img"])):
            sample = {k: v[i] for k, v in samples.items()}
            try:
                add_preliminary_to_sample(sample, samples_to_path(mask_dir, samples, i))
            except FileNotFoundError:
                print(f"{samples_to_path(mask_dir, samples, i)} not found - skipping sample")
                continue

            bboxes = sample["bboxxywh"].split("|")
            for i in range(len(bboxes)):
                bbox = [int(box) for box in bboxes[i].split("-")]
                bboxes[i] = bbox


            ground_truth_img = sample["bbox_img"].float()

            if torch.isnan(sample["preliminary_mask"]).any():
                logger.warning(f"NaN in prediction: {sample['rel_path']} -- {samples_to_path(mask_dir, samples, i)}")
                continue

            binary_mask = repeat(mask_suggestor(sample, key="preliminary_mask"), "h w -> 3 h w")
            binary_mask_large = resize_to_imag_size(binary_mask.float()).round()

            prelim_mask = (sample["preliminary_mask"] - sample["preliminary_mask"].min())/(sample["preliminary_mask"].max() - sample["preliminary_mask"].min())
            prelim_mask_large = resize_to_imag_size(prelim_mask.unsqueeze(dim=0)).squeeze(dim=0)

            results["rel_path"].append(sample["rel_path"])
            results["finding_labels"].append(sample["finding_labels"])
            results["cnr"].append(float(contrast_to_noise_ratio(ground_truth_img, prelim_mask_large)))
            prediction, center_of_mass_prediction, bbox_gmm_pred = compute_prediction_from_binary_mask(binary_mask_large[0])
            iou = torch.tensor(jaccard_score(ground_truth_img.flatten(), binary_mask_large[0].flatten()))
            iou_rev = torch.tensor(jaccard_score(1 - ground_truth_img.flatten(), 1 - binary_mask_large[0].flatten()))
            results["iou"].append(float(iou))
            results["miou"].append(float((iou + iou_rev)/2))

            bboxiou = torch.tensor(jaccard_score(ground_truth_img.flatten(), prediction.flatten()))
            bboxiou_rev = torch.tensor(jaccard_score(1 - ground_truth_img.flatten(), 1 - prediction.flatten()))
            results["bboxiou"].append(float(bboxiou))
            results["bboxmiou"].append(float((bboxiou + bboxiou_rev)/2))

            if len(bboxes) > 1:
                results["distance"].append(np.nan)
            else:
                _, center_of_mass, _ = compute_prediction_from_binary_mask(ground_truth_img)
                distance = np.sqrt((center_of_mass[0] - center_of_mass_prediction[0]) ** 2 +
                                   (center_of_mass[1] - center_of_mass_prediction[1]) ** 2
                                   )
                results["distance"].append(float(distance))


            argmax_idx = np.unravel_index(prelim_mask_large.argmax(), prelim_mask_large.size())
            mode_is_outlier = ground_truth_img[argmax_idx]
            results["top1"].append(float(mode_is_outlier))

            auc = roc_auc_score(ground_truth_img.flatten(), prelim_mask_large.flatten())
            results["aucroc"].append(auc)

            if log_some > 0:
                logger.info(f"Logging example bboxes and attention maps to {config.log_dir}")
                img = (sample["img_raw"] + 1) / 2

                ground_truth_img = repeat(ground_truth_img, "h w -> 3 h w")
                prelim_mask_large = repeat(prelim_mask_large, "h w -> 3 h w")

                fig = plt.figure(figsize=(6, 20))
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(4, 1),
                                 axes_pad=0.1)
                for j, ax, im in zip(np.arange(4), grid, [img, img, binary_mask_large, ground_truth_img]): # 2nd img is below prelim mask
                    ax.imshow(rearrange(im, "c   h w -> h w c"))
                    if j == 1:
                        ax.imshow(prelim_mask_large.mean(axis=0), cmap="jet", alpha=0.25)
                        ax.scatter(argmax_idx[1], argmax_idx[0], s=100, c='red', marker='o')
                    ax.axis('off')

                path = os.path.join(config.log_dir, "localization_examples", os.path.basename(sample["rel_path"]).rstrip(".png") + f"_{sample['finding_labels']}")
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.makedirs(path)
                logger.info(f"Logging to {path}")
                plt.savefig(path + "_raw.png", bbox_inches="tight")
                log_some -= 1

    df = pd.DataFrame(results)
    logger.info(f"Saving file with results to { mask_dir}")
    df.to_csv(os.path.join(mask_dir, f"pgm_{config.phrase_grounding_mode}_bbox_results.csv"))
    mean_results = df.groupby("finding_labels").mean(numeric_only=True)
    mean_results.to_csv(os.path.join(mask_dir,  f"pgm_{config.phrase_grounding_mode}_bbox_results_means.csv"))
    logger.info(df.groupby("finding_labels").mean(numeric_only=True))

    with open(os.path.join(mask_dir, f"pgm_{config.phrase_grounding_mode}_bbox_results.json"), "w") as file:
        json_results = {}
        json_results["all"] = dict(df.mean(numeric_only=True))
        for x in mean_results.index:
            json_results[x] = dict(mean_results.loc[x])

        json.dump(json_results, file, indent=4)

def get_args():
    parser = argparse.ArgumentParser(description="Compute Localization Scores")
    parser.add_argument("--config", type=str, help="Path to the dataset config file")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="dir to save masks in. Default will be inside log dir and should be used!")
    parser.add_argument("--filter_bad_impressions", action="store_true", default=False,
                        help="If set, then we use shortened impressions from mscxr")
    parser.add_argument("--phrase_grounding_mode", action="store_true", default=False,
                        help="If set, then we use shortened impressions from mscxr")
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="If set, then lora weights are sued")
    return parser.parse_args()



def vis(tensor):
    plt.imshow(tensor)
    plt.show()


if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config)
    world_size = torch.cuda.device_count()
    compute_masks(1, config, world_size, args.use_lora)
    compute_iou_score(config)