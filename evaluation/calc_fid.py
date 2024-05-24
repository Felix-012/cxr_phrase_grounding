"""addapted from https://github.com/MischaD/chest-distillation"""

import argparse
import json
from evaluation.inception import InceptionV3
from evaluation.xrv_fid import calculate_fid_given_paths
from datasets.utils import load_config
from log import logger
import os
import torch
import torchxrayvision as xrv



IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def main(args):
    device = torch.device('cuda')
    config = load_config(args.config)
    if config.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = config.num_workers

    results = {}
    dims = 0
    model = None
    for fid_model in ["inception", "xrv"]:
        if fid_model == "xrv":
            dims = 1024
            model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
        elif fid_model == "inception":
            dims = 2048
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            model = InceptionV3([block_idx]).to(device)


        fid_value = calculate_fid_given_paths([args.path_src, args.path_tgt],
                                              args.batch_size,
                                              device,
                                              fid_model,
                                              model=model,
                                              dims=dims,
                                              num_workers=num_workers)
        logger.info(f"FID of the following paths: {args.path_src} -- {args.path_tgt}")
        logger.info(f'{fid_model} FID: {fid_value} --> ${fid_value:.1f}$')
        results[fid_model] = fid_value

    if hasattr(args, "result_dir") and args.result_dir is not None:
        with open(os.path.join(args.result_dir, "fid_results.json"), "w") as file:
            results_file = {"dataset_src": args.path_src, "dataset_tgt": args.path_tgt}
            for fid_model, fid_value in results.items():
                results_file[fid_model] = {"FID": fid_value,
                                          "as_string": f"{fid_value:.1f}"
                                          }
            json.dump(results_file, file)


def get_args():
    parser = argparse.ArgumentParser(description="Compute FID of dataset")
    parser.add_argument("--config", type=str, help="Path to the dataset config file")
    parser.add_argument("path_src", type=str, help="Path to first dataset")
    parser.add_argument("path_tgt", type=str, help="Path to second dataset")
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=0,
                        help=('Number of processes to use for data loading.'))
    parser.add_argument("--result_dir", type=str, default=None, help="dir to save results in.")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)