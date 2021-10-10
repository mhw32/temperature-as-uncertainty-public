"""
Nearest neighbor accuracy.
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from dotmap import DotMap
from src.utils import utils
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.systems.simclr import SimCLRSystem, TaU_SimCLRSystem
from src.systems.mocov2 import MoCoV2System, TaU_MoCoV2System
from src.systems.hib import HIBSystem
from src.systems.transfer import Pretrained_TaU_SimCLRSystem 


@torch.no_grad()
def get_nearest_neighbor_accuracy(args):
    system, config, device = load_pretrained_system(args)
    train_loader = DataLoader(system.train_dataset, 
                              batch_size=config.optimizer.batch_size,
                              shuffle=False,  # important
                              pin_memory=True,
                              drop_last=False,
                              num_workers=config.dataset.num_workers)
    val_loader = DataLoader(system._val_dataset, 
                            batch_size=config.optimizer.batch_size,
                            shuffle=False,    # important
                            pin_memory=True,
                            drop_last=False,
                            num_workers=config.dataset.num_workers)
    # store training embeddings in here
    train_embeds = []
    train_labels = []

    is_baseline = config.system in ['MoCoV2System', 'SimCLRSystem']

    def get_embeddings(image1, image2, return_vars=False):
        if is_baseline:
            if not args.one_view:
                embed1 = system.forward(image1)
                embed2 = system.forward(image2)
                embed1 = F.normalize(embed1, dim=1)
                embed2 = F.normalize(embed2, dim=1)
                embed = (embed1 + embed2) / 2.
            else:
                embed = system.forward(image1)
        else:
            mean1, _ = system.forward(image1)
            mean2, _ = system.forward(image2)
            mean1 = F.normalize(mean1, dim=1)
            mean2 = F.normalize(mean2, dim=1)

            embed = (mean1 + mean2) / 2.
        
        return embed

    print('Collecting training embeddings...')
    pbar = tqdm(total=len(train_loader))
    for batch in train_loader:
        image1, image2, label = batch[1], batch[2], batch[3]
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)
        embed = get_embeddings(image1, image2)
        train_embeds.append(embed)
        train_labels.append(label)
        pbar.update()
    pbar.close()

    train_embeds = torch.cat(train_embeds, dim=0)   # num_train x out_dim
    train_labels = torch.cat(train_labels, dim=0)   # num_train

    num_correct = 0
    num_total = 0

    print('Evaluating test embeddings...')
    pbar = tqdm(total=len(val_loader))
    for batch in val_loader:
        image1, image2, label = batch[1], batch[2], batch[3]
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)
        batch_size = image1.size(0)

        embed = get_embeddings(image1, image2, return_vars=False)
        dists = embed @ train_embeds.T
        _, idxs = torch.topk(dists, k=1, sorted=False, dim=1)
        preds = torch.index_select(train_labels, 0, idxs.squeeze())

        num_correct += torch.sum(label == preds).item()
        num_total += batch_size

        pbar.update()
    pbar.close()

    return num_correct / float(num_total)


def load_pretrained_system(args):
    config_path = os.path.join(args.exp_dir, 'config.json')
    config_json = utils.load_json(config_path)
    config = DotMap(config_json)
    config.dataset.two_views = True  # NOTE: important for this analysis
    config.dataset.name = args.dataset
    if args.gpu_device >= 0:
        config.gpu_device = args.gpu_device
        config.cuda = True
        device = torch.device(f'cuda:{args.gpu_device}')
    else:
        config.cuda = False
        device = torch.device('cpu')

    SystemClass = globals()[config.system]
    system = SystemClass(config)
    checkpoint_file = os.path.join(args.exp_dir, 'checkpoints', args.checkpoint_name)
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    system.load_state_dict(checkpoint['state_dict'])
    system.config = config
    system = system.eval()
    system = system.to(device)

    return system, config, device


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='experiment directory')
    parser.add_argument('checkpoint_name', type=str, help='checkpoint name')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--one-view', action='store_true', default=False)
    parser.add_argument('--gpu-device', type=int, default=-1)
    args = parser.parse_args()

    test_acc = get_nearest_neighbor_accuracy(args)
    print('--------------------------')
    print(f'Test Accuracy: {test_acc}')
