import os
import json
import torch
import numpy as np
from dotmap import DotMap
from sklearn.metrics import roc_auc_score

import torch.nn.functional as F
from src.utils import utils
from src.ood.base import BaseOODEvaluator
from src.systems.supervised import SupervisedSystem


class OdinOODEvaluator(BaseOODEvaluator):
    # https://arxiv.org/pdf/1706.02690.pdf
    # https://github.com/pokaxpoka/deep_Mahalanobis_detector
    # MAGNITUDES = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]
    # TEMPERATURES = [1, 10, 100, 1000]

    def get_encoder(self):
        base_dir = self.config.model.encoder.exp_dir
        checkpoint_name = self.config.model.encoder.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        config.dataset.name = self.config.dataset.in_dataset
        config.gpu_device = self.config.gpu_device

        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(base_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        system.load_state_dict(checkpoint['state_dict'])
        system.config = config
        system = system.to(self.device)
        system = system.eval()

        return system

    def get_score(self, image):
        magnitude = self.config.model.magnitude or 0.0014
        temperature = self.config.model.temperature or 1000

        image.requires_grad = True  # so backward works
        outputs = self.encoder.forward(image)
        labels = outputs.data.max(1)[1]
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(image.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(
            1, 
            torch.LongTensor([0]).to(self.device), 
            gradient.index_select(
                1, 
                torch.LongTensor([0]).to(self.device),
            ) / (0.2023),
        )
        gradient.index_copy_(
            1, 
            torch.LongTensor([1]).to(self.device), 
            gradient.index_select(
                1, 
                torch.LongTensor([1]).to(self.device),
            ) / (0.1994),
        )
        gradient.index_copy_(
            1, 
            torch.LongTensor([2]).to(self.device), 
            gradient.index_select(
                1, 
                torch.LongTensor([2]).to(self.device),
            ) / (0.2010),
        )

        # the perturbation can have stronger effect on the in- distribution image 
        # than that on out-of-distribution image, making them more separable
        #
        #   \tilde{x} = x - \epsilon * sign(-gradient)
        tempInputs = image.detach() - magnitude * gradient
        
        outputs = self.encoder.forward(tempInputs)
        outputs = outputs / temperature
        
        soft_out = F.softmax(outputs, dim=1)
        soft_out = torch.max(soft_out.data, dim=1)[0]
        score = soft_out.detach().cpu().numpy()
        return score

    def save(self, metrics):
        eval_dir = os.path.join(self.config.model.encoder.exp_dir, 'eval')
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)
        with open(os.path.join(eval_dir, 'metrics.json'), 'w') as fp:
            json.dump(metrics, fp)


