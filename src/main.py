import logging
import os
import sys
from datetime import datetime

import lightning.pytorch.loggers as pl_loggers
import torch
import torch.distributed as dist
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from omegaconf import OmegaConf

from dataset import get_dataloader
from losses import get_loss_func
from models import MODEL_LIST
from utils.trainer import HybridTrainer
from utils.tools import check_dir, get_activation


logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_CONFIG = {
    'mode': 'train',
    'strategy': 'ddp',
    'experiment': 'baseline',
    'model_name': 'fengyuan',
    'gpus': [-1],
    'lossfx': 'combineloss',
    'tensor_model_parallel_size': 1,
    'logger': 'wandb',
    'noedge': False,
    'init_method': 'env://',
    'world_size': 1,
    'local_rank': 0,
    'activation': None,
    'save_all': False,
    'backbone': None,
    'backbone_configs': None,
    'aux_backbone': None,
    'aux_configs': None,
    'forward_type': 'alone',
    'trainer_mode': 'normal',
    'init_ckpt': None,
    'checkpoint_path': None,
    'wandb': {'project': 'meteoai', 'mode': 'offline'},
    'inference': {'plot': False},
    'log_level': 0,
    'submit_suffix': None,
    'finetune': False,
}

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed_everything(666, workers=True)


def load_model(configs):
    """loads a model from a checkpoint or from scratch if checkpoint_path is None"""

    loss_func = get_loss_func(configs)

    logger.info(f"load model: {configs.model_name}...")

    activation = get_activation(configs.activation)

    if configs.model_name in MODEL_LIST:
        trainer_mode = configs.get('trainer_mode', 'normal')
        if trainer_mode == 'normal':
            model = HybridTrainer(configs.model_name, configs, loss_func,
                                  activation=activation,
                                  backbone=configs.backbone,
                                  backbone_configs=configs.backbone_configs,
                                  save_all=configs.save_all,
                                  init_ckpt=configs.init_ckpt)
        else:
            raise ValueError(f'No support mode {trainer_mode}!')
    else:
        raise NotImplementedError(f'No support {configs.model_name} model!')

    return model


def get_trainer(configs, version, gpus):

    logger.info(f"process started on the following GPUs: {gpus}...")
    logger.info(f"get trainer and set callback and checkpoint on {configs.mode} mode...")

    if configs.mode not in ['inference', 'predict']:
        if configs['train'].get('logger') is None or configs['train'].get('logger') == 'tensorboard':
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=configs['project']['log_path'],
                                                     name='', version='')
        else:
            raise NotImplementedError(f"No support {configs['train'].get('logger')}!")
    else:
        tb_logger = False

    dirpath = configs['project']['weight_path']

    if not configs.get('finetune', False):
        filename = f'{configs.model_name}' + '_loss={epoch_valid_loss:.8f}_{epoch}'
    else:
        filename = f'{configs.model_name}' + '_loss={epoch_valid_loss:.8f}_{epoch}_{step}'

    save_callback = ModelCheckpoint(dirpath=dirpath, filename=filename,
                                    **configs['train']['modelcheckpoint'])

    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger.info(f"{configs.mode} mode, checkpoint save path: {dirpath}, and filename: {filename}...")

    callbacks = [save_callback, lr_monitor]
    progress_bar = TQDMProgressBar(refresh_rate=50)
    callbacks.append(progress_bar)

    use_early_stopping = configs['train'].get('use_early_stopping')
    if use_early_stopping is not None and use_early_stopping:
        logger.info(f"using EarlyStopping: {use_early_stopping}")
        logger.info(f"early stopping params: {configs['train']['early_stopping']}")
        early_stop = EarlyStopping(**configs['train']['early_stopping'])
        callbacks.append(early_stop)

    if gpus[0] == -1:
        devices = 1
        accelerator = 'cpu'
        parallel_training = None
    else:
        devices = list(gpus)
        accelerator = 'gpu'
        parallel_training = configs.strategy if len(gpus) > 1 else 'auto'

    inference_mode = True if configs.mode in ['predict', 'inference', 'test'] else False

    trainer = Trainer(devices=devices,
                      logger=tb_logger,
                      accelerator=accelerator,
                      strategy=parallel_training,
                      callbacks=callbacks,
                      inference_mode=inference_mode,
                      num_nodes=configs.get('num_nodes', 1),
                      **configs['train']['trainer'])

    return trainer


def get_version(configs):
    if configs['project'].get('name') is not None:
        version = os.path.join(configs['project'].get('result_path'),
                               configs['project']['name'],
                               configs.experiment + f'_{configs.mode}'
                               )
    else:
        version = configs.experiment + f'{configs.mode}' + f'_{datetime.now():%Y%m%d_%H%M}'

    return version


def set_logger(configs, version):

    log_format = configs['log']['log_format']
    date_format = configs['log']['date_format']
    log_level = {0: logging.INFO, 1: logging.WARNING, 2: logging.DEBUG,
                 3: logging.ERROR, 4: logging.CRITICAL}

    filename = os.path.join(configs['project']['result_path'],
                            configs['project']['name'],
                            f"{os.path.basename(os.path.normpath(version))}.log")

    logging.basicConfig(filename=filename, filemode='w',
                        level=log_level.get(configs.log_level),
                        format=log_format, datefmt=date_format)


def get_kwargs(kwargs):
    args = OmegaConf.create({})
    if kwargs is not None:
        for key, value in kwargs.items():
            OmegaConf.update(args, key, value, merge=True)

    return args


def run(configs, **kwargs):
    configs = OmegaConf.load(configs)
    configs = OmegaConf.merge(
        OmegaConf.create(DEFAULT_RUNTIME_CONFIG),
        configs,
        get_kwargs(kwargs),
    )

    version = get_version(configs)

    log_path = os.path.join(version, 'log')
    save_path = os.path.join(version, 'save')
    weight_path = os.path.join(version, 'ckpt')

    configs['command'] = ' '.join(sys.argv)
    configs['project']['log_path'] = log_path
    configs['project']['save_path'] = save_path
    configs['project']['weight_path'] = weight_path

    check_dir(log_path)
    check_dir(save_path)
    check_dir(weight_path)

    if configs['inference'].get('plot') is not None and configs['inference'].get('plot'):
        plot_path = os.path.join(version, 'save', 'project')
        check_dir(plot_path)
        configs['project']['plot_path'] = plot_path

    set_logger(configs, version)

    logger.info(f"configuration params: {configs}")

    if configs.mode in ['train', 'test', 'predict', 'inference']:
        if configs.mode in ['predict', 'inference']:
            configs['train']['batch_size'] = 1

        datamodule = get_dataloader(configs)
    else:
        raise ValueError(f'No support {configs.mode} mode!')

    model = load_model(configs)

    trainer = get_trainer(configs, version, configs.gpus)

    if configs.finetune:
        ckpt_path = None
    else:
        ckpt_path = configs.checkpoint_path

    if dist.is_available() and dist.is_initialized():
        logger.info("Distributed initialized: %s / %s", dist.get_rank(), dist.get_world_size())
    else:
        logger.info("Distributed not initialized")

    if configs.mode == 'train':
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif configs.mode == 'test':
        trainer.test(model, datamodule=datamodule, ckpt_path=configs.checkpoint_path)
    elif configs.mode in ['predict', 'inference']:
        trainer.predict(model, datamodule=datamodule, ckpt_path=configs.checkpoint_path)
    else:
        raise ValueError(f'No support mode {configs.mode}!')

    logger.info(f"Finished successfully.")


if __name__ == '__main__':
    from fire import Fire

    Fire(run)
