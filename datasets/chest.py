"""addapted from https://github.com/MischaD/chest-distillation"""

import hashlib
import json
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from datasets.utils import path_to_tensor
from torchvision.transforms import Resize, CenterCrop, Compose
from datasets.dataset import FOBADataset
from log import logger
from util_scripts.utils_generic import DatasetSplit
import torch
import os, pickle
from tqdm import tqdm
from PIL import Image


class MimicCXRDataset(FOBADataset):
    def __init__(self, dataset_args, opt):
        super().__init__(dataset_args, opt)
        self._meta_data = None
        self._csv_file = "mimic_metadata_preprocessed.csv"
        if dataset_args.get("dataset_csv") is not None:
            self._csv_file = dataset_args.get("dataset_csv")
        self.precomputed_base_dir = dataset_args.get("precomputed_base_dir")
        self._build_dataset()
        self.opt = opt
        self._precomputed_path = None
        self._save_original_images = dataset_args.get("save_original_images", False)
        self.text_label_key = dataset_args.get("text_label_key", "impression")
        self.chunk_size = None
        self.num_chunks = dataset_args.get("num_chunks")
        self.current_chunk_index = -1
        self.chunk_path = dataset_args.get("chunk_path")
        self.chunk_load_counter = 0
        if self.num_chunks:
            self.chunk_indices = list(range(self.num_chunks))
        random.seed(4200)

    @property
    def precomputed_path(self):
        if self._precomputed_path is None:
            name = "".join([x["rel_path"] for x in self.data])
            name = hashlib.sha1(name.encode("utf-8")).hexdigest()
            precompute_path = os.path.join(os.path.expandvars(self.precomputed_base_dir), str(name))
            self._precomputed_path = precompute_path
        return self._precomputed_path

    @property
    def is_precomputed(self):
        return os.path.isdir(self.precomputed_path)

    def load_precomputed(self, model):
        logger.info(f"Using precomputed dataset with name: {self.precomputed_path}")
        if not self.is_precomputed:
            logger.info(f"Precomputed dataset not found - precomputing it on my own: {self.precomputed_path}")
            self.precompute(model)

        entries = pickle.load(open(os.path.join(self.precomputed_path, "entries.pkl"), "rb"))
        dir_list = os.listdir(self.precomputed_path)
        for file in dir_list:
            if not file.endswith(".pt"):
                continue
            tensor_key = os.path.basename(file.rstrip(".pt"))
            entries[tensor_key] = torch.load(os.path.join(self.precomputed_path, file))

        self.data = []
        for i in range(len(entries["rel_path"])):
            self.data.append({k: entries[k][i] for k in entries.keys()})

    def load_chunk(self, chunk_index):
        if chunk_index == self.current_chunk_index:
            return  # No need to load if it's already the current chunk
        filename = f"entries_part{chunk_index}.pkl"
        with open(os.path.join(os.path.expandvars(self.chunk_path), filename), "rb") as f:
            entries = pickle.load(f)
        self.data = [{k: entries[k][i] for k in entries.keys()} for i in range(len(entries['rel_path']))]
        self.current_chunk_index = chunk_index
        self.chunk_size = len(self.data)
        logger.info(f"loaded chunk {chunk_index} with size: {self.chunk_size}")

    def compute_latent(self, img, model):
        """
        Preprocoessing. Img is already 512x512 tensor 1xCx512x512 --> compute latent using vqvae - saves Gaussian parameters
        """
        img = img.to(model.device)
        encoder_posterior = model.encode(img)
        return encoder_posterior

    def sample_latent(self, encoder_posterior, scale_factor):
        z = encoder_posterior.sample()
        return z * scale_factor

    def decode_from_latent(self, encoder_posterior, model):
        """
        Helper function to decode latent space of vqvae
        """
        n, c, h, w = encoder_posterior.size()
        assert encoder_posterior.ndim == 4 and n == 1
        old_device = encoder_posterior.device
        encoder_posterior = encoder_posterior.to("cuda")

        if c == 8:
            # params for latent gaussian
            z = self.sample_latent(encoder_posterior, model.scale_factor)
        elif c == 4:
            # sampled values
            z = encoder_posterior
        else:
            raise ValueError(f"Unable to interpret encoder_posterior of shape: {encoder_posterior.size()}")
        img = model.decode_first_stage(z).detach()
        img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
        return img.to(old_device)

    def precompute(self, model):
        # load entries
        entries = {}
        if self._save_original_images:
            entries["img_raw"] = []
        if hasattr(self.opt, "control_cond_path"):
            entries["control"] = []
        j = 0
        for i in tqdm(range(len(self)), "Precomputing Dataset"):
            try:
                entry = self._load_images([j])
            except FileExistsError:
                print(f"skipping {self.data[j]['rel_path']} - file does not exist")
                del self.data[j]
                continue
            for k in entry.keys():
                if entries.get(k) is None:
                    assert i == 0
                    entries[k] = []
                entries[k].append(entry[k])

            # preprocess --> 1 x 8 x 64 x 64 diag gaussian latent
            z = self.compute_latent(entry["img"], model)
            if self._save_original_images:
                entries["img_raw"].append(entry["img"])
            if hasattr(self.opt, "control_cond_path") and self.opt.control_cond_path is None:
                if hasattr(self.opt, "control_preprocessing_type"):
                    entries["control"].append(self.preprocess_control(entries["img"][j], self.opt.control_preprocessing_type))
            entries["img"][j] = z
            j += 1
        if hasattr(self.opt, "control_cond_path") and self.opt.control_cond_path is not None:
            if hasattr(self.opt, "control_preprocessing_type"):
                control_preprocessing_type = self.opt.control_preprocessing_type
            else:
                control_preprocessing_type = None
            if not entries["control"]:
                entries = self.load_control_conditioning(entries, self.opt.control_cond_path, control_preprocessing_type)

        # save entries
        entry_keys = list(entries.keys())
        data_tensors = {}
        for key in entry_keys:
            if isinstance(entries[key][0], torch.Tensor):
                data_tensors[key] = torch.stack(entries.pop(key))

        path = self.precomputed_path
        logger.info(f"Saving precomputed dataset to: {path}")
        os.makedirs(path)
        pickle.dump(entries, open(os.path.join(path, "entries.pkl"), "wb"))
        for key in data_tensors.keys():
            torch.save(data_tensors[key], os.path.join(path, f"{key}.pt"))

    @property
    def meta_data_path(self):
        return os.path.join(os.path.expandvars(self.precomputed_base_dir), self._csv_file)

    @property
    def meta_data(self):
        if self._meta_data is None:
            logger.info(f"Loading image list from {self.meta_data_path}")
            self._meta_data = pd.read_csv(self.meta_data_path, index_col="dicom_id")
            return self._meta_data
        else:
            return self._meta_data

    def _build_dataset(self):
        path = "path"
        if "rel_path" in self.meta_data:
            path = "rel_path"
        try:
            filtered_meta_data = self.meta_data.dropna(subset=['path', 'Finding Labels'])
            paths = filtered_meta_data['path'].to_list()
            labels = filtered_meta_data['Finding Labels'].to_list()
            data = [
                dict(
                    rel_path=os.path.join(img_path.replace(".dcm", ".jpg")),
                    finding_labels=label
                )
                for img_path, label in zip(paths, labels)
            ]
        except KeyError:
            filtered_meta_data = self.meta_data.dropna(subset=[path])
            paths = filtered_meta_data[path].to_list()
            data = [
                dict(
                    rel_path=os.path.join(img_path.replace(".dcm", ".jpg")),
                )
                for img_path in paths]

        try:
            splits = self.meta_data["split"].astype(int)
            self._get_split(data, splits)
        except KeyError:
            self.data = data

        if self.shuffle:
            np.random.shuffle(np.array(self.data))

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

    def load_image(self, img_path):
        img = path_to_tensor(img_path)
        # images too large are resized to self.W^2 using center cropping
        if max(img.size()) > self.W:
            transforms = Compose([Resize(self.W), CenterCrop(self.W)])
            img = transforms(img)
        return img

    def _load_images(self, index):
        assert len(index)
        entry = self.data[index[0]].copy()
        entry["dicom_id"] = os.path.basename(entry["rel_path"]).rstrip(".jpg")
        img_path = os.path.join(self.base_dir, entry["rel_path"].replace(".dcm", ".jpg"))
        entry["img"] = self.load_image(img_path)
        entry["impression"] = self.meta_data.loc[entry["dicom_id"]]["impression"]
        return entry

    def load_control_conditioning(self, entries, control_cond_path, control_preprocessing_type):
        for i in tqdm(range(len(entries)), "Processing control conditioning"):
            control = self.load_image(control_cond_path)
            if control_preprocessing_type:
                control = self.preprocess_control(control, control_preprocessing_type)
            entries[i]["control"] = control
        return entries

    def preprocess_control(self, control, control_preprocessing_type):
        if control_preprocessing_type != "canny":
            raise NotImplementedError("Only canny preprocessing is implemented for control conditioning")
        if torch.is_tensor(control):
            control = cv2.cvtColor(control.numpy().squeeze().transpose(1, 2, 0), cv2.COLOR_BGR2GRAY)
            control = np.round((control + 1) * 255 / 2).astype(np.uint8)
        control = cv2.medianBlur(control, 5)
        control = cv2.Canny(control, np.median(control) * 0.4, np.median(control) * 0.3)
        control = control[:, :, None]
        control = np.concatenate([control, control, control], axis=2)

        return Image.fromarray(control)

    def load_next_chunk(self):
        if self.chunk_load_counter >= len(self.chunk_indices):  # If all chunks have been loaded once
            random.shuffle(self.chunk_indices)  # Reshuffle the list
            self.chunk_load_counter = 0  # Reset the counter
        next_chunk_index = self.chunk_indices[self.chunk_load_counter]
        self.load_chunk(next_chunk_index + 1)
        self.chunk_load_counter += 1

    def __getitem__(self, idx):
        ret = self.data[idx]
        # Apply your custom logic for the text_label_key
        if self.text_label_key in ret:
            if isinstance(ret[self.text_label_key], float):
                ret["impression"] = ""
            else:
                finding_labels = ret[self.text_label_key].split("|")
                ret["impression"] = " ".join(random.sample(finding_labels, len(finding_labels)))
        return ret


class MimicCXRDatasetMSBBOX(MimicCXRDataset):
    def __init__(self, dataset_args, opt):
        self._bbox_meta_data = None
        self._csv_name = "mcxr_with_impressions.csv"
        assert dataset_args["split"] == DatasetSplit("mscxr").value
        if dataset_args.get("phrase_grounding", False):
            logger.info("Phrase grounding mode on in MSBBOX Dataset")
            self._csv_name = "mimi_scxr_phrase_grounding_preprocessed.csv"

        super().__init__(dataset_args, opt)

    @property
    def bbox_meta_data(self):
        return pd.read_csv(os.path.join(self.base_dir, self._csv_name), index_col="dicom_id")

    def _build_dataset(self):
        data = [dict(dicom_id=dicom_id, rel_path=os.path.join(img_path.replace(".dcm", ".jpg")), finding_labels=labels)
                for img_path, labels, dicom_id in
                zip(list(self.bbox_meta_data.paths), list(self.bbox_meta_data["category_name"]),
                    self.bbox_meta_data.index)]
        self.data = data
        if self.shuffle:
            np.random.shuffle(np.array(self.data))

        if self.limit_dataset is not None:
            self.data = self.data[self.limit_dataset[0]:min(self.limit_dataset[1], len(self.data))]

    def apply_filter_for_disease_in_txt(self):
        logger.warning(f"Filtering dataset to only contain impressions where the disease itself is mentioned.")
        data = []
        for entry in self.data:
            meta_data_entry = self.bbox_meta_data.loc[entry["dicom_id"]]
            if isinstance(meta_data_entry, pd.DataFrame):
                meta_data_entry = meta_data_entry[meta_data_entry["category_name"] == entry["finding_labels"]]
                assert len(meta_data_entry) == 1
                meta_data_entry = meta_data_entry.iloc[0]

            if entry["finding_labels"].lower() in meta_data_entry["label_text"].lower():
                data.append(entry)
            else:
                logger.info(
                    f"Dropping the following for {meta_data_entry['category_name']}: {meta_data_entry['label_text']}")
        old_len = len(self.data)
        self.data = data
        logger.info(f"Reduced dataset from {old_len} to {len(self.data)} due to filtering of diseases in txt")

    def _load_images(self, index):
        assert len(index)
        entry = self.data[index[0]].copy()
        entry["img"] = self.load_image(os.path.join(self.base_dir, entry["rel_path"].replace(".dcm", ".jpg")))

        meta_data_entry = self.bbox_meta_data.loc[entry["dicom_id"]]
        if isinstance(meta_data_entry, pd.DataFrame):
            meta_data_entry = meta_data_entry[meta_data_entry["category_name"] == entry["finding_labels"]]
            assert len(meta_data_entry) == 1
            meta_data_entry = meta_data_entry.iloc[0]

        image_width, image_height = meta_data_entry[["image_width", "image_height"]]
        bboxes = meta_data_entry["bboxxywh"].split("|")
        bbox_img = torch.zeros(image_height, image_width, dtype=torch.bool)

        for bbox in bboxes:
            bbox = bbox.split("-")
            bbox = tuple(map(lambda y: int(y), bbox))
            x, y, w, h = bbox
            bbox_img[y: (y + h), x:(x + w)] = True

        if max(bbox_img.size()) > self.W:
            transforms = Compose([Resize(self.W), CenterCrop(self.W)])
            bbox_img = transforms(bbox_img.unsqueeze(dim=0)).squeeze()

        entry["bbox_img"] = bbox_img
        entry["bboxxywh"] = meta_data_entry["bboxxywh"]
        entry["label_text"] = meta_data_entry["label_text"]
        entry["category_name"] = meta_data_entry["category_name"]
        return entry


class MIMIC_Dataset(Dataset):
    def __init__(self, umls_json_path, radgraph_json_path, csv_path, sty_path, img_res, base_path):
        self.p = Path(radgraph_json_path).stem
        self.base_path = base_path
        # spacy.prefer_gpu()
        # nlp = spacy.load("en_core_sci_scibert")
        # self.umls_info = DocBin()
        # self.umls_info = self.umls_info.from_disk(umls_path)
        # self.umls_info = list(self.umls_info.get_docs(vocab=nlp.vocab))

        self.radgraph_json_info = json.load(open(radgraph_json_path, 'r'))
        self.umls_info = json.load(open(umls_json_path, 'r'))

        self.entity_label_dict = {
            'ANAT-DP': 'Anatomy Definitely Present',
            'OBS-DP': 'Observation Definitely Present',
            'OBS-DA': 'Observation Definitely Absent',
            'OBS-U': 'Observation Uncertain',
        }

        data_info = pd.read_csv(csv_path)
        self.dcm_relative_paths = np.asarray(data_info.loc[data_info['p'] == self.p, 'path'])
        self.class_list = np.asarray(list(data_info)[16:-2])
        # sty_info = pd.read_csv(sty_path)
        # self.sty_dict_info = self.csv_to_dict(sty_info)

    def csv_to_dict(self, sty_info):
        tui_list = sty_info.iloc[:, 0]
        sty_list = sty_info.iloc[:, 1]
        sty_dict = defaultdict(list)
        for idx in tqdm(range(len(tui_list))):
            tui_idx = tui_list[idx]
            sty_idx = sty_list[idx]
            sty_dict[tui_idx] = sty_idx
        return sty_dict

    def __len__(self):
        return len(self.dcm_relative_paths)

    def get_entity_list(self, entities):
        entity_dict = defaultdict(list)
        entities_num = len(entities)
        for idx in range(entities_num):
            entity_idx = entities[str(idx + 1)]
            token_idx = entity_idx['tokens']
            label_idx = self.entity_label_dict[entity_idx['label']]
            entity_dict[token_idx] = label_idx
        return entity_dict

    def __getitem__(self, index):
        class_label = self.class_list[index]
        # entities = self.umls_json_info[index]['entities']
        # captions = self.umls_json_info[index]['caption']
        entities = self.umls_info[index]["entity_presence"]
        if len(entities) != 0:
            try:
                radgraph_entities = self.radgraph_json_info[self.umls_info[index]['file_path']]['entities']
                radgraph_entity_dict = self.get_entity_list(radgraph_entities)
                entity_details = ''
                for entity in entities:
                    sub_entities = entity["entities"]
                    sub_entity_details = ''
                    for sub_entity in sub_entities:
                        sub_entity_info = sub_entity["text"]
                        if sub_entity_info in radgraph_entity_dict.keys():
                            sub_entity_details += sub_entity_info + radgraph_entity_dict[sub_entity_info]
                        else:
                            sub_entity_details += sub_entity_info
                    entity_details = entity_details + sub_entity_details + ' [SEP] '
            except:
                entity_details = ''
                for entity in entities:
                    sub_entities = entity["entities"]
                    sub_entity_details = ''
                    for sub_entity in sub_entities:
                        sub_entity_details += sub_entity["text"]
                    entity_details = entity_details + sub_entity_details + ' [SEP] '
        else:
            entity_details = ''
            for sub_entity in entities:
                entity_details = entity_details + sub_entity["sentence_text"] + ' [SEP] '

        return {
            "entity": entity_details
        }
