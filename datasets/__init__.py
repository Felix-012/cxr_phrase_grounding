from .chest import MimicCXRDataset, MimicCXRDatasetMSBBOX
from .mscoco import MSCOCODataset, MSCOCOBBoxDataset

def get_dataset(opt, split=None):
    datasets = {"chestxraymimic": MimicCXRDataset, "chestxraymimicbbox": MimicCXRDatasetMSBBOX, "mscoco": MSCOCODataset, "mscocobbox": MSCOCOBBoxDataset}
    assert split is not None
    dataset_args = getattr(opt.datasets, f"{split}")
    getattr(opt, "dataset_args", dataset_args)
    dataset = datasets[dataset_args["dataset"]](dataset_args=dataset_args, opt=opt)
    return dataset