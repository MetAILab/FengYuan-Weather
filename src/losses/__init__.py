import logging
import torch.nn as nn


logger = logging.getLogger(__name__)

LOSS_LIST = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
}


class CombineLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.loss_params = configs['train']['loss']['combineloss']

    def forward(self, inputs, targets):
        loss = 0.
        for key, value in self.loss_params.items():
            loss_type = value.get('loss')
            idx = value.get('idx')
            weight = value.get('weight')

            if loss_type in LOSS_LIST:
                loss_config = self.configs['train']['loss']
                loss_params = loss_config.get(loss_type) if loss_config.get(loss_type) is not None else {}
                loss_func = LOSS_LIST.get(loss_type)(**loss_params)
            else:
                raise ValueError(f'loss function {loss_type} in {key} is not supported.')

            sub_loss = weight * loss_func(inputs[:, :, idx], targets[:, :, idx])
            loss += sub_loss

        return loss


class LossFunc(nn.Module):
    def __init__(self, loss_func, aux_params=None, use_uncertainty=False):
        super().__init__()
        self.loss_func = loss_func
        self.aux_params = aux_params
        self.use_uncertainty = use_uncertainty

    def forward(self, preds, targets, max_logvar=None, min_logvar=None, current_step=None):
        if self.use_uncertainty:
            if current_step is not None:
                return self.loss_func(
                    preds, targets, max_logvar, min_logvar, current_step=current_step
                )
            return self.loss_func(preds, targets, max_logvar, min_logvar)

        if self.aux_params is None:
            loss = self.loss_func(preds, targets)
        else:
            var_idx = self.aux_params.get('var_idx')
            var_weight = self.aux_params.get('var_weight')
            var_names = self.aux_params.get('var_names')
            assert len(var_idx) == preds.shape[2]

            loss = 0
            for idx, name, weight in zip(var_idx, var_names, var_weight):
                sub_loss = weight * self.loss_func(preds[:, :, idx], targets[:, :, idx])
                loss += sub_loss

        return loss


def get_loss_func(configs):
    if configs.mode not in ['predict', 'inference']:
        if configs.lossfx in LOSS_LIST:
            loss_config = configs['train']['loss']
            loss_params = loss_config.get(configs.lossfx) if loss_config.get(configs.lossfx) is not None else {}
            loss_func = LOSS_LIST.get(configs.lossfx)(**loss_params)
        elif configs.lossfx == 'combineloss':
            loss_func = CombineLoss(configs)
        else:
            raise ValueError(f'loss function {configs.lossfx} not in {LOSS_LIST.keys()}.')

        logger.info(f"loss function and params in configuration file: {configs['train']['loss']}")
        logger.info(f"{configs.lossfx} loss was used to train model {configs.model_name}")
    else:
        loss_func = None
        logger.info(f"{configs.mode} mode, no loss function")

    if configs['train']['loss'].get('aux_params') is not None:
        params = configs['train']['loss'].get('aux_params')
    else:
        params = None

    return LossFunc(loss_func, params)
