import os
from copy import deepcopy
import random, torch, numpy
from src.systems.simclr import SimCLRSystem, TaU_SimCLRSystem
from src.systems.mocov2 import MoCoV2System, TaU_MoCoV2System
from src.systems.supervised import SupervisedSystem
from src.systems.transfer import (
    TransferSystem, 
    Pretrained_TaU_SimCLRSystem)
from src.systems.hib import HIBSystem
from src.utils.utils import load_json
from src.utils.setup import process_config
from src.scheduler.moco import MoCoLRScheduler
import pytorch_lightning as pl
import getpass


torch.backends.cudnn.benchmark = True


def run(config_path, dataset=None, gpu_device=None):
    if gpu_device == 'cpu' or not gpu_device:
        gpu_device = None
    
    config = process_config(config_path)

    if dataset is not None:
        config.dataset.name = dataset

    if gpu_device: 
        config.gpu_device = int(gpu_device)

    seed_everything(config.seed, use_cuda=config.cuda)
    SystemClass = globals()[config.system]
    system = SystemClass(config)

    if config.optimizer.moco_scheduler:
        lr_callback = MoCoLRScheduler(
            initial_lr=config.optimizer.learning_rate,
            use_cosine_scheduler=False,
            max_epochs=config.num_epochs,
            schedule=(
                int(0.68*config.num_epochs),
                int(0.83*config.num_epochs),
            ),
        )
        callbacks = [lr_callback]
    else:
        callbacks = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(config.exp_dir, 'checkpoints'),
        save_top_k=-1,
        period=20,
    )
    trainer = pl.Trainer(
        default_root_dir=config.exp_dir,
        gpus=gpu_device,
        distributed_backend=config.distributed_backend or 'dp',
        max_epochs=config.num_epochs,
        min_epochs=config.num_epochs,
        checkpoint_callback=ckpt_callback,
        callbacks=callbacks,
        resume_from_checkpoint=config.continue_from_checkpoint,
    )
    trainer.fit(system)


def seed_everything(seed, use_cuda=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda: torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--gpu-device', type=str, default=None)
    args = parser.parse_args()
    
    gpu_device = str(args.gpu_device) if args.gpu_device else None
    run(args.config, dataset=args.dataset, gpu_device=args.gpu_device)
