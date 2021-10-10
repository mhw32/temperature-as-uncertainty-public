import os
import json
import random
import numpy as np
from tqdm import tqdm
from dotmap import DotMap
from pprint import pprint
from collections import OrderedDict
import pytorch_lightning as pl

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.svhn import SVHN
from src.datasets.imagenet import ImageNet
from src.datasets.lsun import LSUN
from src.datasets.mscoco import MSCOCO
from src.datasets.celeba import CelebA
from src.datasets.transforms import get_transforms
from src.models.resnet import ResNet
from src.datasets.utils import DICT_CONV3X3, DICT_ROOT
from src.utils import utils


class BaseOODEvaluator(object):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.set_seed(config)
        self.set_device(config)
        self.encoder = self.get_encoder()  # encoder is a system

        in_dataset, out_datasets = self.load_datasets()
        self.in_loader = self.create_dataloader(in_dataset, config)
        self.out_loaders = [self.create_dataloader(dset, config) for dset in out_datasets]

    def set_seed(self, config):
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    def set_device(self, config):
        if config.cuda:
            self.device = torch.device(f'cuda:{config.gpu_device}')
        else:
            self.device = torch.device('cpu')

    def get_encoder(self):
        raise NotImplementedError

    def load_datasets(self):
        in_name = self.config.dataset.in_dataset
        out_names = self.config.dataset.out_datasets
        num_out = len(out_names)
        use_views = self.config.dataset.use_views

        if use_views:
            transforms, _ = get_transforms(in_name)
        else:
            _, transforms = get_transforms(in_name)

        def load_one_dataset(name):
            root = DICT_ROOT[name]
            if name == 'cifar10':
                dataset = CIFAR10(root, train=False, image_transforms=transforms)
            elif name == 'cifar100':
                dataset = CIFAR100(root, train=False, image_transforms=transforms)
            elif name == 'svhn':
                dataset = SVHN(root, train=False, image_transforms=transforms)
            elif name == 'tinyimagenet' or name == 'imagenet':
                dataset = ImageNet(root, train=False, image_transforms=transforms)
            elif name == 'lsun':
                dataset = LSUN(root, train=False, image_transforms=transforms)
            elif name == 'mscoco':
                dataset = MSCOCO(root, train=False, image_transforms=transforms)
            elif name == 'celeba':
                dataset = CelebA(root, train=False, image_transforms=transforms)
            else:
                raise Exception(f'Dataset {name} not supported.')
            return dataset

        in_dataset = load_one_dataset(in_name)
        out_datasets = []
        for i in range(num_out):
            out_dataset = load_one_dataset(out_names[i])
            out_datasets.append(out_dataset)

        return in_dataset, out_datasets

    def create_dataloader(self, dataset, config):
        dataset_size = len(dataset)
        loader = DataLoader(dataset, batch_size=config.optimizer.batch_size,
                            shuffle=False, pin_memory=True,
                            num_workers=config.dataset.num_workers)
        return loader

    def get_score(self, image):
        raise NotImplementedError

    # @torch.no_grad()
    def evaluate(self, loader):
        use_views = self.config.dataset.use_views
        num_epochs = self.config.dataset.num_views or 1
        epoch_scores = []
        for i in range(num_epochs):
            pbar = tqdm(total=len(loader), 
                        desc=f'epoch {i+1}/{num_epochs}')
            scores = []
            for batch in loader:
                images = batch[1]
                batch_size = images.size(0)
                images = images.to(self.device).float()
                score = self.get_score(images)
                scores.append(score)
                pbar.update()
            pbar.close()
            scores = np.concatenate(scores)
            epoch_scores.append(scores[np.newaxis, ...])
        epoch_scores = np.concatenate(epoch_scores, axis=0)
        epoch_scores = np.mean(epoch_scores, axis=0)
        return epoch_scores

    def run(self):
        print('Evaluating inlier dataset...')
        in_scores = self.evaluate(self.in_loader)
        in_labels = np.ones_like(in_scores)
        out_names = self.config.dataset.out_datasets
        num_out = len(self.out_loaders)

        aurocs = {}
        fprs = {}
        for i in range(num_out):
            out_loader = self.out_loaders[i]
            print(f'Evaluating outlier dataset ({i+1}/{num_out})...')
            out_scores_i = self.evaluate(out_loader)
            out_labels_i = np.zeros_like(out_scores_i)

            all_scores_i = np.concatenate([in_scores, out_scores_i])
            all_labels_i = np.concatenate([in_labels, out_labels_i])
            auroc_i = roc_auc_score(all_labels_i, all_scores_i)
            aurocs[out_names[i]] = auroc_i

        results = {'auroc': aurocs}
        return results

    def save(self, metrics):
        raise NotImplementedError
