import os
import json
import torch
import torch.nn.functional as F
from dotmap import DotMap
from src.ood.base import BaseOODEvaluator
from src.systems.simclr import SimCLRSystem, TaU_SimCLRSystem
from src.systems.supervised import SupervisedSystem
from src.utils import utils

class EnsembleOODEvaluator(BaseOODEvaluator):

    def __init__(self, config):
        self.config = config
        self.set_seed(config)
        self.set_device(config)
        self.ensemble = []
        for encoder_config in self.config.model.encoder:
            encoder = self.get_encoder(encoder_config.exp_dir, encoder_config.checkpoint_name)  # encoder is a system
            self.ensemble.append(encoder)

        in_dataset, out_datasets = self.load_datasets()
        self.in_loader = self.create_dataloader(in_dataset, config)
        self.out_loaders = [self.create_dataloader(dset, config) for dset in out_datasets]

    def get_encoder(self, base_dir, checkpoint_name):
        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        config.dataset.name = self.config.dataset.in_dataset
        config.gpu_device = self.config.gpu_device

        base_model = config.model.base_model
        config.model.base_model = base_model

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        system.load_state_dict(checkpoint['state_dict'])
        system.config = config
        system = system.eval()
        system = system.to(self.device)

        for param in system.parameters():
            param.requires_grad = False

        return system

    def get_score(self, image):
        outputs = []
        for encoder in self.ensemble:
            outputs.append(encoder.forward(image).detach().cpu())
        
        outputs = torch.stack(outputs)
        outputs = F.normalize(outputs, dim=2)
        score = torch.std(outputs, dim=0).mean(dim=1)
        
        score = score.numpy()  # batch_size
        return score
