import hashlib
import random
import numpy as np
import pandas as pd
from datasets.utils import path_to_tensor
from torchvision.transforms import Resize, CenterCrop, Compose
from datasets.dataset import FOBADataset
from log import logger
from utils.utils import DatasetSplit
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os, pickle
from tqdm import tqdm


class MimicCXRDataset(FOBADataset):
    def __init__(self, dataset_args, opt):
        super().__init__(dataset_args, opt)
        self._meta_data = None
        self._csv_file = "mimic_metadata_preprocessed.csv"
        if dataset_args.get("dataset_csv") is not None:
            self._csv_file = dataset_args.get("dataset_csv")
        self._build_dataset()
        self.opt = opt
        self._precomputed_path = None
        self.precomputed_base_dir=dataset_args.get("precomputed_base_dir")
        self._save_original_images = dataset_args.get("save_original_images", False)
        self.text_label_key = dataset_args.get("text_label_key", "impression")

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
            for k in entries.keys():
                if k == "img":
                    for item in entries[k]:
                        item.latent_dist.logvar = item.latent_dist.logvar.cpu()
                        item.latent_dist.mean = item.latent_dist.mean.cpu()
                        item.latent_dist.parameters = item.latent_dist.parameters.cpu()
                        item.latent_dist.std = item.latent_dist.std.cpu()
                        item.latent_dist.var = item.latent_dist.var.cpu()
                self.data.append({k: entries[k][i]})
        del entries

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
        #load entries
        entries = {}
        if self._save_original_images:
            entries["img_raw"] = []
        j = 0


        for i in tqdm(range(len(self)), "Precomputing Dataset"):
            print("start precomputing")
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
                entries["img"][j] = z
            else:
                entries["img"][j] = z
            j +=1

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
        return os.path.join(os.path.expandvars(self.base_dir), self._csv_file)

    @property
    def meta_data(self):
        if self._meta_data is None:
            logger.info(f"Loading image list from {self.meta_data_path}")
            self._meta_data = pd.read_csv(self.meta_data_path, index_col="dicom_id")
            return self._meta_data
        else:
            return self._meta_data

    def _build_dataset(self):
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
            filtered_meta_data = self.meta_data.dropna(subset=['path'])
            paths = filtered_meta_data['path'].to_list()
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

    def _load_image(self, img_path):
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
        entry["img"] = self._load_image(img_path)
        entry["impression"] = self.meta_data.loc[entry["dicom_id"]]["impression"]
        return entry

    def __getitem__(self, item):
        ret = super().__getitem__(item)
        if self.text_label_key == "finding_labels":
            if isinstance(ret["finding_labels"], float):
                ret["impression"] = ""
            else:
                finding_labels = ret["finding_labels"].split("|")
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
        data = [dict(dicom_id=dicom_id, rel_path=os.path.join(img_path.replace(".dcm", ".jpg")), finding_labels=labels) for img_path, labels, dicom_id in zip(list(self.bbox_meta_data.paths), list(self.bbox_meta_data["category_name"]), self.bbox_meta_data.index)]
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
                logger.info(f"Dropping the following for {meta_data_entry['category_name']}: {meta_data_entry['label_text']}")
        old_len = len(self.data)
        self.data = data
        logger.info(f"Reduced dataset from {old_len} to {len(self.data)} due to filtering of diseases in txt")

    def _load_images(self, index):
        assert len(index)
        entry = self.data[index[0]].copy()
        entry["img"] = self._load_image(os.path.join(self.base_dir, entry["rel_path"].replace(".dcm", ".jpg")))

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
