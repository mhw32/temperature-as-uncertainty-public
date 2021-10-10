import os
import json
import torch
from dotmap import DotMap
from src.ood.base import BaseOODEvaluator
from src.utils import utils
from src.systems.simclr import SimCLRSystem, TaU_SimCLRSystem
from src.systems.mocov2 import MoCoV2System, TaU_MoCoV2System
from src.systems.hib import HIBSystem
from src.systems.transfer import Pretrained_TaU_SimCLRSystem

class SelfSupOODEvaluator(BaseOODEvaluator):

    def get_encoder(self):
        base_dir = self.config.model.encoder.exp_dir
        checkpoint_name = self.config.model.encoder.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        config.dataset.name = self.config.dataset.in_dataset
        config.gpu_device = self.config.gpu_device

        base_model = config.model.base_model
        # load a deterministic version of the model
        base_model = base_model.replace('dp_', '')
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

    def save(self, metrics):
        eval_dir = os.path.join(self.config.model.encoder.exp_dir, 'eval')
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)
        with open(os.path.join(eval_dir, 'metrics.json'), 'w') as fp:
            json.dump(metrics, fp)

    def get_score(self, image):
        outputs = self.encoder.forward(image)
        if self.encoder.config.loss.posterior in ['gaussian', 'gaussian_hib']:
            stdev = outputs[1]
            variance = torch.pow(stdev, 2)
            score = torch.norm(variance, p=2, dim=1)
        elif self.encoder.config.loss.posterior == 'vmf':
            concentration = outputs[1]
            score = concentration.squeeze(1)
        else:
            raise Exception('Posterior not supported.')
        score = score.detach().cpu().numpy()  # batch_size
        return score
