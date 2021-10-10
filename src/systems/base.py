import os
import numpy as np
from dotmap import DotMap
from collections import OrderedDict
import matplotlib.pyplot as plt
import pytorch_lightning as pl

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.imagenet import ImageNet
from src.datasets.transforms import get_transforms
from src.datasets.utils import DICT_CONV3X3, DICT_ROOT
from src.models.resnet import ResNet
from src.utils import utils


class PretrainSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        # keep val_dataset hidden because we don't want to validate
        self.train_dataset, self._val_dataset, self.num_classes = self.get_datasets()
        self.model = self.get_model()
        self.save_variances = self.config.loss.save_variances
        self.save_images = self.config.loss.save_images
        self.curr_epoch_vars = np.array([])
        self.t = config.loss.t

    def get_datasets(self):
        name = self.config.dataset.name
        train_transforms, val_transforms = get_transforms(name)
        root = DICT_ROOT[name]

        if name == 'cifar10':
            train_dataset = CIFAR10(root, train=True, image_transforms=train_transforms, 
                                    two_views=self.config.dataset.two_views)
            val_dataset = CIFAR10(root, train=False, image_transforms=val_transforms, 
                                  two_views=self.config.dataset.two_views)
            num_classes = 10
        elif name == 'cifar100':
            train_dataset = CIFAR100(root, train=True, image_transforms=train_transforms, 
                                     two_views=self.config.dataset.two_views)
            val_dataset = CIFAR100(root, train=False, image_transforms=val_transforms, 
                                   two_views=self.config.dataset.two_views)
            num_classes = 100
        elif name == 'tinyimagenet' or name == 'imagenet':
            train_dataset = ImageNet(root, train=True, image_transforms=train_transforms, 
                                     two_views=self.config.dataset.two_views)
            val_dataset = ImageNet(root, train=False, image_transforms=val_transforms, 
                                   two_views=self.config.dataset.two_views)
            num_classes = 200 if name == 'tinyimagenet' else 1000
        else:
            raise Exception(f'Dataset {name} not supported.')

        return train_dataset, val_dataset, num_classes

    def get_model(self):
        if self.config.model.base_model == 'fc':
            model = FullyConnected(
                out_dim=self.config.model.out_dim, 
                in_dim=self.config.model.in_dim,
                posterior_head=self.config.model.posterior_head,
                posterior_family=self.config.loss.posterior,
            )
            return model

        base_model = self.config.model.base_model or 'resnet18'
        conv3x3 = DICT_CONV3X3[self.config.dataset.name]
        model = ResNet(base_model, self.config.model.out_dim, 
                       conv3x3=conv3x3, 
                       final_bn=self.config.model.final_bn,
                       posterior_head=self.config.model.posterior_head,
                       posterior_family=self.config.loss.posterior)
        return model

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.config.optimizer.learning_rate,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay,
        )
        return [optimizer], []

    def encode(self, image):
        return self.model(image)[0]

    def forward(self, image, batch_idx=0):
        if self.config.model.posterior_head:
            _, mean, stdev = self.model(image)
            return mean, stdev
        else:
            return self.model(image)[1]

    def get_loss(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, train=True, batch_idx=batch_idx)
        metrics = {'train_loss': loss}
        self.log_dict(metrics)
        return loss

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.optimizer.batch_size,
            shuffle=True, 
            pin_memory=True,
            drop_last=True,
            num_workers=self.config.dataset.num_workers,
        )
        return train_loader
    
