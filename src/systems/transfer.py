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
import torchvision.models as models
from torch.utils.data import DataLoader

from src.utils import utils
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100
from src.datasets.imagenet import ImageNet
from src.datasets.transforms import get_transforms
from src.datasets.utils import DICT_ROOT
from src.models.byol import ByolWrapper
from src.models.clip import ClipWrapper
from src.objectives.simclr import TaU_SimCLR
from src.systems.simclr import SimCLRSystem, TaU_SimCLRSystem

from pl_bolts.models.self_supervised import SimCLR


class TransferSystem(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder()  # encoder is a system
        self.train_dataset, self.val_dataset, self.num_class = self.get_datasets()
        self._val_dataset = self.val_dataset
        self.model = self.get_model()
        self.val_metrics = []

    def get_datasets(self):
        name = self.config.dataset.name
        train_transforms, val_transforms = get_transforms(name)
        root = DICT_ROOT[name]

        if name == 'cifar10':
            train_dataset = CIFAR10(root, train=True, image_transforms=train_transforms, two_views=self.config.dataset.two_views)
            val_dataset = CIFAR10(root, train=False, image_transforms=val_transforms, two_views=self.config.dataset.two_views)
            num_class = 10
        elif name == 'cifar100':
            train_dataset = CIFAR100(root, train=True, image_transforms=train_transforms, two_views=self.config.dataset.two_views)
            val_dataset = CIFAR100(root, train=False, image_transforms=val_transforms, two_views=self.config.dataset.two_views)
            num_class = 100
        elif name == 'tinyimagenet' or name == 'imagenet':
            train_dataset = ImageNet(root, train=True, image_transforms=train_transforms, two_views=self.config.dataset.two_views)
            val_dataset = ImageNet(root, train=False, image_transforms=val_transforms, two_views=self.config.dataset.two_views)
            num_class = 200 if name == 'tinyimagenet' else 1000
        else:
            raise Exception(f'Dataset {name} not supported.')

        return train_dataset, val_dataset, num_class

    def get_encoder(self):
        base_dir = self.config.model.encoder.exp_dir
        checkpoint_name = self.config.model.encoder.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        config.dataset.name = self.config.dataset.name
        config.gpu_device = self.config.gpu_device

        base_model = config.model.base_model
        # load a deterministic version of the model
        base_model = base_model.replace('dp_', '')
        config.model.base_model = base_model
        config.model.final_dp = False  # for transfer

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        system.load_state_dict(checkpoint['state_dict'])
        system.config = config
        if self.config.model.encoder.has_gradients:
            system = system.train()
        else:
            system = system.eval()

        return system

    def get_model(self):  # linear evaluation
        model = nn.Linear(512, self.num_class)
        return model

    def configure_optimizers(self):
        if self.config.model.encoder.has_gradients:
            parameters = list(self.encoder.model.parameters()) + list(self.model.parameters())
        else:
            parameters = self.model.parameters()
        if self.config.optimizer.name == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=self.config.optimizer.learning_rate,
                momentum=self.config.optimizer.momentum,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif self.config.optimizer.name == 'adam':
            optimizer = optim.Adam(
                parameters,
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise Exception(f'Optimizer {self.config.optimizer.name} not supported.')

        schedulers = []

        if self.config.optimizer.scheduler_type == "step":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, 
                self.config.optimizer.decay_epochs, 
                gamma=self.config.optimizer.gamma,
            )
            schedulers.append(scheduler)
        elif self.config.optimizer.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.config.num_epochs,
                eta_min=self.config.optimizer.final_lr
            )
            schedulers.append(scheduler)

        return [optimizer], schedulers

    def forward(self, image):
        with torch.no_grad():
            h = self.encoder.encode(image)
        if type(h) == tuple:
            h = h[0] # Very hacky!
        return self.model(h)

    def get_loss(self, batch, train=True):
        _, image, label = batch
        logits = self.forward(image)
        return F.cross_entropy(logits, label)

    @torch.no_grad()
    def get_accuracy(self, batch):
        _, image, label = batch
        logits = self.forward(image)
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        preds = preds.long().cpu()
        num_correct = torch.sum(preds == label.long().cpu()).item()
        num_total = image.size(0)
        return num_correct, num_total

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, train=True)
        num_correct, num_total = self.get_accuracy(batch)
        acc = torch.tensor([num_correct / float(num_total)]).float()
        metrics = {'train_loss': loss, 'train_acc': acc}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, train=False)
        num_correct, num_total = self.get_accuracy(batch)
        return {'val_loss': loss, 'val_num_correct': num_correct, 'val_num_total': num_total}

    def validation_epoch_end(self, outputs):
        metrics = dict()
        metrics['val_loss'] = torch.tensor([elem['val_loss'] for elem in outputs]).float().mean()
        total_num_correct = sum([elem['val_num_correct'] for elem in outputs])
        total_num_total = sum([elem['val_num_total'] for elem in outputs])
        acc = torch.tensor([total_num_correct / float(total_num_total)]).float()
        metrics['val_acc'] = acc
        self.log_dict(metrics)
        self.val_metrics.append(acc)

        np.save(os.path.join(self.config.log_dir, 'val_acc.npy'), np.array(self.val_metrics))

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

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.optimizer.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=self.config.dataset.num_workers,
        )
        return val_loader


class Pretrained_TaU_SimCLRSystem(TransferSystem):
    
    def __init__(self, config):
        super().__init__(config)
    
    def get_model(self):  # linear evaluation
        if self.config.model.encoder == 'simclr':
            model = nn.Linear(2048, 1)
        elif self.config.model.encoder == 'byol':
            model = nn.Linear(2048, 1)
        elif self.config.model.encoder == 'clip':
            model = nn.Linear(512, 1)
        elif self.config.model.encoder == 'resnet':
            model = nn.Linear(1000, 1)
        else:
            raise NotImplementedError
        return model
    
    def get_encoder(self):
        if self.config.model.encoder == 'simclr':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            system = SimCLR.load_from_checkpoint(weight_path, strict=False)
            system.freeze()
            system.to(self.config.gpu_device)
        elif self.config.model.encoder == 'byol':
            system = ByolWrapper(self.config.gpu_device)
        elif self.config.model.encoder == 'clip':
            system = ClipWrapper(self.config.gpu_device)
        elif self.config.model.encoder == 'resnet':
            system = models.resnet50(pretrained=True)
            system.to(self.config.gpu_device)
        else:
            raise NotImplementedError
        return system

    def forward(self, image):
        with torch.no_grad():
            h1, h2 = self.encoder.model(image)
        scale = self.model(h1)
        scale = F.softplus(scale) + 1e-6
        return h2, scale
    
    def get_loss(self, batch, train=True):
        _, image1, image2, _ = batch
        loc1, temp1 = self.forward(image1)
        loc2, temp2 = self.forward(image2)
        loss = TaU_SimCLR(loc1, temp1, loc2, temp2,
                          t=self.config.loss.t,
                          eps=self.config.loss.eps,
                          simclr_mask=self.config.loss.simclr_mask).get_loss()
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, train=True)
        metrics = {'train_loss': loss}
        self.log_dict(metrics)
        return loss 

    def forward(self, image):
        with torch.no_grad():
            h = self.encoder.forward(image)
        scale = self.model(h)
        scale = scale
        return h, scale

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        parameters = self.model.parameters()
        if self.config.optimizer.name == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=self.config.optimizer.learning_rate,
                momentum=self.config.optimizer.momentum,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif self.config.optimizer.name == 'adam':
            optimizer = optim.Adam(
                parameters,
                lr=self.config.optimizer.learning_rate,
                weight_decay=self.config.optimizer.weight_decay,
            )
        else:
            raise Exception(f'Optimizer {self.config.optimizer.name} not supported.')

        schedulers = []

        if self.config.optimizer.scheduler_type == "step":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, 
                self.config.optimizer.decay_epochs, 
                gamma=self.config.optimizer.gamma,
            )
            schedulers.append(scheduler)
        elif self.config.optimizer.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.config.num_epochs,
                eta_min=self.config.optimizer.final_lr
            )
            schedulers.append(scheduler)

        return [optimizer], schedulers
