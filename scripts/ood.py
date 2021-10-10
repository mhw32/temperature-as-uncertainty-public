import os
from copy import deepcopy
from pprint import pprint
import random, torch, numpy
from src.ood.selfsup import SelfSupOODEvaluator
from src.ood.odin import OdinOODEvaluator
from src.ood.knn import KnnOODEvaluator
from src.ood.mcdropout import MCDropoutOODEvaluator
from src.ood.ensemble import EnsembleOODEvaluator
from src.utils.utils import load_json
from src.utils.setup import process_config


def run(config_path, dataset=None, gpu_device=None):
    if gpu_device == 'cpu' or not gpu_device:
        gpu_device = None
    
    config = process_config(config_path)

    if dataset is not None:
        config.dataset.in_dataset = dataset

    if gpu_device: 
        config.gpu_device = int(gpu_device)

    SystemClass = globals()[config.system]
    system = SystemClass(config)
    results = system.run()
    pprint(results)
    system.save(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--gpu-device', type=str, default=None)
    args = parser.parse_args()

    gpu_device = str(args.gpu_device) if args.gpu_device else None
    run(args.config, dataset=args.dataset, gpu_device=args.gpu_device)
