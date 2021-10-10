import numpy as np
import math
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision
from typing import Callable, Optional
from pytorch_lightning.core.optimizer import LightningOptimizer


from src.objectives.mocov2 import MoCoV2, TaU_MoCoV2
from src.systems.base import PretrainSystem
from src.utils import utils
from torch.optim.optimizer import Optimizer
from src.scheduler.lars import LARSWrapper


class MoCoV2System(PretrainSystem):

    def __init__(self, config):
        super().__init__(config)
        self.model_k = self.get_model()

        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        queue = torch.randn(self.config.loss.k, self.config.model.out_dim)
        self.register_buffer("queue", queue)
        self.queue = F.normalize(queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self.config.loss.m
        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        if self.use_ddp or self.use_ddp2:
            keys = utils.concat_all_gather(keys)

        batch_size = keys.size(0)
        k = self.config.loss.k
        ptr = int(self.queue_ptr)
        assert k % batch_size == 0 

        self.queue[ptr:ptr+batch_size] = keys
        ptr = (ptr + batch_size) % k  # move pointer
        self.queue_ptr[0] = ptr

    def get_loss(self, batch, train=True):
        _, image_q, image_k, _ = batch

        # compute query features
        outputs_q = self.forward(image_q)

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update key encoder

            # shuffle batch
            idx = torch.randperm(image_k.size(0))
            image_k = image_k[idx]

            outputs_k = self.model_k(image_k)[1]
            
            # unshuffle batch
            outputs_k_tmp = torch.zeros_like(outputs_k)
            for i, j in enumerate(idx):
                outputs_k_tmp[j] = outputs_k[i]
            outputs_k = outputs_k_tmp

        loss_fn = MoCoV2(outputs_q, outputs_k, self.queue, t=self.config.loss.t)
        loss = loss_fn.get_loss()

        if train:
            outputs_k = F.normalize(outputs_k, dim=1)
            self._dequeue_and_enqueue(outputs_k)

        return loss

    # Copied directly from the SimCLR System!
    def get_lr_schedule(self):
        batch_size = self.config.optimizer.batch_size
        iters_per_epoch = len(self.train_dataset) // batch_size
        start_lr = self.config.optimizer.start_lr
        final_lr = self.config.optimizer.final_lr
        learning_rate = self.config.optimizer.learning_rate
        warmup_epochs = self.config.optimizer.warmup_epochs
        max_epochs = self.config.num_epochs

        warmup_lr_schedule = np.linspace(start_lr, learning_rate, iters_per_epoch * warmup_epochs)
        iters = np.arange(iters_per_epoch * (max_epochs - warmup_epochs))
        cosine_lr_schedule = np.array([
            final_lr + 0.5 * (learning_rate - final_lr) *
            (1 + math.cos(math.pi * t / (iters_per_epoch * (max_epochs - warmup_epochs))))
            for t in iters
        ])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        return lr_schedule

    def configure_optimizers(self):
        self.lr_schedule = self.get_lr_schedule()  # make lr schedule
        weight_decay = self.config.optimizer.weight_decay
        exclude_bn_bias = self.config.optimizer.exclude_bn_bias
        learning_rate = self.config.optimizer.learning_rate

        if exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=weight_decay)
        else:
            params = self.parameters()

        if self.config.optimizer.name == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif self.config.optimizer.name == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception(f'Optimizer {self.config.optimizer.name} not supported.')

        optimizer = LARSWrapper(optimizer, eta=0.001, clip=False)
        return [optimizer], []

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        if not isinstance(optimizer, LightningOptimizer):
            optimizer = LightningOptimizer.to_lightning_optimizer(optimizer, self.trainer)
        optimizer.step(closure=optimizer_closure)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]
    
    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, train=True)
        metrics = {'train_loss': loss, 'learning_rate': self.lr_schedule[self.trainer.global_step]}
        self.log_dict(metrics)
        return loss


class TaU_MoCoV2System(MoCoV2System):

    def __init__(self, config):
        super().__init__(config)

        temp_queue = torch.randn(self.config.loss.k, 1)
        self.register_buffer("temp_queue", temp_queue)
        self.temp_queue = 1 + F.softplus(temp_queue)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, temps):
        batch_size = keys.size(0)
        k = self.config.loss.k
        ptr = int(self.queue_ptr)
        assert k % batch_size == 0 

        self.queue[ptr:ptr+batch_size] = keys
        self.temp_queue[ptr:ptr+batch_size] = temps
        ptr = (ptr + batch_size) % k  # move pointer
        self.queue_ptr[0] = ptr

    def get_loss(self, batch, train=True):
        _, image_q, image_k, _ = batch

        # compute query features
        outputs_q, temp_q = self.forward(image_q)

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update key encoder

            # shuffle batch
            idx = torch.randperm(image_k.size(0))
            image_k = image_k[idx]

            _, outputs_k, temp_k = self.model_k(image_k)
            
            # unshuffle batch
            outputs_k_tmp = torch.zeros_like(outputs_k)
            for i, j in enumerate(idx):
                outputs_k_tmp[j] = outputs_k[i]
            outputs_k = outputs_k_tmp

        loss_fn = TaU_MoCoV2(outputs_q, temp_q, outputs_k, temp_k, self.queue, self.temp_queue, t=self.config.loss.t)
        loss = loss_fn.get_loss()

        if train:
            outputs_k = F.normalize(outputs_k, dim=1)
            self._dequeue_and_enqueue(outputs_k, temp_k)

        t = temp_q.clone().detach().squeeze(-1).cpu().numpy()
        self.log("temperature_mean", np.mean(t), on_step=True)
        self.log("temperature_std", np.std(t), on_step=True)
        return loss
