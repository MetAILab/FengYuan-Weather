import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler


def warmup_lambda(warmup_steps, min_lr_ratio=0.1):
    def ret_lambda(epoch):
        if epoch <= warmup_steps:
            return min_lr_ratio + (1.0 - min_lr_ratio) * epoch / warmup_steps
        else:
            return 1.0
    return ret_lambda


class LinearHalfCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step <= self.warmup_steps:
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            decay_steps = self.total_steps - self.warmup_steps
            cosine_decay = 0.5 * (1 + torch.cos(torch.pi * torch.tensor((step - self.warmup_steps) / decay_steps)))
            return [base_lr * cosine_decay for base_lr in self.base_lrs]


def configure_optimizer(configs, model_parameters):
    """set optimizer for Trainer"""

    if configs['train'].get('optim') is not None:
        optim_name = configs['train'].get('optim')
        params = configs['train'][optim_name]
    else:
        raise ValueError('Must be setting optimizer!')

    if optim_name.lower() == 'adam':
        optimizer = optim.Adam(model_parameters, lr=configs['train']['lr'], **params)
    elif optim_name.lower() == 'adamw':
        optimizer = optim.AdamW(model_parameters, lr=configs['train']['lr'], **params)
    else:
        raise ValueError(f'No support {optim_name} optimizer!')

    if configs['train'].get('scheduler') is not None:
        scheduler_name = configs['train'].get('scheduler')
        params = configs['train'][scheduler_name]
    else:
        raise ValueError('Must be setting scheduler!')

    if scheduler_name.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, **params)
        return optimizer, scheduler
    elif scheduler_name.lower() == 'reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=configs['train']['reducelr']['mode'],
                                                   patience=configs['train']['reducelr']['patience'],
                                                   factor=configs['train']['reducelr']['factor'])
        return optimizer, scheduler
    elif scheduler_name.lower() == 'linearhalfcosine':
        scheduler = LinearHalfCosineLR(optimizer, **params)
        return optimizer, scheduler
    else:
        raise ValueError(f'No support {scheduler_name} scheduler!')
