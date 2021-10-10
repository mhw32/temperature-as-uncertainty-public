import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from dotmap import DotMap
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.ood.base import BaseOODEvaluator
from src.utils import utils
from src.systems.simclr import SimCLRSystem
from src.systems.mocov2 import MoCoV2System


class KnnOODEvaluator(BaseOODEvaluator):

    def get_encoder(self):
        base_dir = self.config.model.encoder.exp_dir
        checkpoint_name = self.config.model.encoder.checkpoint_name

        config_path = os.path.join(base_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
        config.dataset.name = self.config.dataset.in_dataset
        config.gpu_device = self.config.gpu_device

        base_model = 'resnet18' #config.model.base_model
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

    def get_encodings(self, loader):
        pbar = tqdm(total=len(loader),
                    desc=f'generating vector representations')
        encodings = []
        for batch in loader:
            images = batch[1]
            batch_size = images.size(0)
            images = images.to(self.device).float()
            vec = self.encoder.forward(images).detach()
            vec = F.normalize(vec, dim=1).cpu().numpy()
            encodings.append(vec)
            pbar.update()
        pbar.close()
        encodings = np.concatenate(encodings)
        return encodings

    def save(self, metrics):
        eval_dir = os.path.join(self.config.model.encoder.exp_dir, 'eval')
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)
        with open(os.path.join(eval_dir, 'metrics.json'), 'w') as fp:
            json.dump(metrics, fp)

    def evaluate(self, loader, in_encodings, remove_first=False):
        offset = 0
        if remove_first:
            offset = 1
        encodings = self.get_encodings(loader)
        y = np.zeros((in_encodings.shape[0]))
        knn = KNeighborsClassifier(n_neighbors=self.config.ood_scores.nearest_neighbors + offset)
        knn.fit(in_encodings, y)
        dists, _ = knn.kneighbors(encodings, n_neighbors=self.config.ood_scores.nearest_neighbors + offset, return_distance=True)
        scores = []
        for dist in dists:
            score = self.get_score(dist[offset:])
            scores.append(score)
        return np.array(scores)

    def get_score(self, dist):
        if self.config.ood_scores.aggregate in ["average", "mean"]:
            return np.mean(dist)
        elif self.config.ood_scores.aggregate == "max":
            return np.max(dist)
        raise Exception(f"The aggregate option {self.config.ood_scores.aggregate} has not been implemented!")

    def run(self):
        print('Evaluating inlier dataset...')
        in_encodings = self.get_encodings(self.in_loader)
        in_scores = self.evaluate(self.in_loader, in_encodings, remove_first=True)
        in_labels = np.zeros_like(in_scores)
        out_names = self.config.dataset.out_datasets
        num_out = len(self.out_loaders)

        aurocs = {}
        fprs = {}
        for i in range(num_out):
            out_loader = self.out_loaders[i]
            print(f'Evaluating outlier dataset ({i+1}/{num_out})...')
            datasets_equivalent = self.config.dataset.in_dataset == self.config.dataset.out_datasets[i]
            out_scores_i = self.evaluate(out_loader, in_encodings, remove_first=datasets_equivalent)
            out_labels_i = np.ones_like(out_scores_i)

            all_scores_i = np.concatenate([in_scores, out_scores_i])
            all_labels_i = np.concatenate([in_labels, out_labels_i])
            auroc_i = roc_auc_score(all_labels_i, all_scores_i)
            auroc_i = max(auroc_i, 1-auroc_i)
            aurocs[out_names[i]] = auroc_i

        results = {'auroc': aurocs}
        return results
