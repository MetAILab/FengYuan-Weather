import logging
import os
import pickle

import numpy as np
import torch
import lightning as L

from models import get_model
from utils.metrics import cal_metrics
from utils.optim import configure_optimizer
from utils.tools import change_keys


logger = logging.getLogger(__name__)


def get_index(test_names, variables_names):
    return [variables_names.index(var) for var in test_names]


class HybridTrainer(L.LightningModule):
    def __init__(self, model_name, configs, loss_fx, save_all=False, init_ckpt=None,
                 save_hparams=True, ignore=None, **kwargs):
        super().__init__()

        if ignore is None:
            ignore = ('loss_fx', 'metric')
        if save_hparams:
            self.save_hyperparameters(ignore=ignore)

        self.model_name = model_name
        self.configs = configs
        self.loss_fx = loss_fx
        self.model = get_model(model_name, configs)
        self.save_all = save_all
        self.input_length = self.configs['model']['input_length']
        self.finetune = self.configs.get('finetune', False)
        self.finetune_step = self.configs.get('finetune_step', 1000)
        self._pre_normalized = configs['data'].get('pre_normalized', False)

        self.norm_flag = configs['train'].get('norm_flag')
        var_names = configs['train']['var_names']
        input_names = configs['train']['var_names']
        target_names = configs['train']['target_names']
        self.test_names = configs['train']['target_names']

        self.mean = np.load(configs['data']['mean_file'])
        self.std = np.load(configs['data']['std_file'])
        self.variables_names = configs['data'].get('variables_names')
        self.names = list(np.load(self.variables_names))

        self.variables_index = get_index(self.test_names, self.names)
        self.in_var_idx = get_index(input_names, var_names)
        self.out_var_idx = get_index(target_names, var_names)
        mean_var = self.mean[self.variables_index]
        std_var = self.std[self.variables_index]
        if mean_var.ndim == 1:
            mean_var = mean_var[:, np.newaxis, np.newaxis]
            std_var = std_var[:, np.newaxis, np.newaxis]
        self.mean_var = torch.from_numpy(mean_var)
        self.std_var = torch.from_numpy(std_var)

        self._init_metrics()
        self._init_weights(init_ckpt)

    def _init_metrics(self):
        self.train_step_metrics = {'train_loss': []}
        self.val_step_metrics = {'valid_loss': []}
        self.test_step_metrics = {'test_loss': []}

    def _init_weights(self, init_ckpt):
        if init_ckpt is not None:
            logger.info(f'init weights from {init_ckpt}...')
            ckpt = torch.load(init_ckpt, map_location=torch.device('cpu'))['state_dict']
            ckpt = change_keys(ckpt, old_name='model.', new_name='')
            self.model.load_state_dict(ckpt)
            self.model = self.model.to(self.device)

    def time_condition(self, x, targets=None):
        bs, frames, chs, height, width = x.shape

        t = torch.tensor(range(1, self.configs['train']['num_step'] + 1, 1))
        t = t[None, :].repeat(x.shape[0], 1).contiguous()
        t = t.view(-1).contiguous().to(x.device)

        x = x.view(bs, frames*chs, height, width).contiguous()
        x = x.unsqueeze(1).repeat(1, self.configs['train']['num_step'], 1, 1, 1).contiguous()
        x = x.view(bs*self.configs['train']['num_step'], frames, chs, height, width).contiguous()

        return x, t

    def _squeeze(self, x, params):
        if x.ndim == 5:
            bs, _, _, height, width = x.shape
        elif x.ndim == 4:
            bs, _, height, width = x.shape
            x = x.unsqueeze(1)

        if params['train'].get('reshape'):
            return x.view(bs, -1, height, width)
        else:
            return x

    def _get_ar_step(self, params):
        max_steps = int(params['train']['num_step'])
        if self.finetune:
            return min(max_steps, 2 + int(self.global_step / self.finetune_step))
        return max_steps

    def _forward_model(self, x, backbone, params, height, width, dim_out):
        bs = x.shape[0]
        backbone = backbone.to(self.device)

        ti = params['model'].get('input_length', 1)
        x = x[:, -ti:]
        out = []
        num_step = self._get_ar_step(params)
        iterative_step = params['train']['iterative_step']
        for i in range(iterative_step):
            for t in range(1, num_step + 1):
                pred = backbone(self._squeeze(x, params))
                out.append(pred)
                x = torch.cat([x, pred.unsqueeze(dim=1)], dim=1)[:, -ti:]
        outputs = torch.stack(out, dim=1)
        timestep = len(out)

        outputs = outputs.reshape(bs, timestep, dim_out, height, width)

        return outputs

    def _forward_backbone(self, x, t=None):
        bs, frames, chs, height, width = x.shape

        if self.save_all:
            alldays = []
        else:
            alldays = None

        backbone_x = x.clone()

        timestep = self.configs['model'].get('target_length')
        dim_out = self.configs['model'].get('cube_out_channels')
        iterative_type = self.configs['train'].get('iterative_type', 'all2one')
        bti = -1 * self.configs['model'].get('input_length')
        x_in = backbone_x[:, bti:]
        x = self._forward_model(x_in, self.model, self.configs, height, width, dim_out)

        if self.save_all:
            alldays.append(x.clone())

        backbone_x = torch.cat([backbone_x, x], dim=1)

        return backbone_x, alldays

    def forward(self, x, t=None):
        bs, frames, chs, height, width = x.shape

        dim_out = self.configs['model'].get('cube_out_channels')

        if self.configs['data']['swap_ch_time']:
            x = x.permute(0, 2, 1, 3, 4)

        out = self._forward_model(x, self.model, self.configs, height, width, dim_out)

        _, height, width = out.shape[0], out.shape[-2], out.shape[-1]

        return out

    def _denormalize(self, *tensors, device):
        """Inverse of normalization: x_physical = x_norm * std + mean.
        Applied for evaluation metrics when data is in normalized space.
        """
        if not (self._pre_normalized or self.configs.train.get('norm_flag')):
            return tensors
        dtype = tensors[0].dtype
        mean = self.mean_var.to(device=device, dtype=dtype)
        std = self.std_var.to(device=device, dtype=dtype)
        return tuple(t * std + mean for t in tensors)

    def _shared_eval_step(self, batch, split):
        inputs, targets_all, init_time = batch

        iterative_step = self.configs['train'].get('iterative_step', 1)
        interval_idx = self._get_ar_step(self.configs)
        init_idx = 0
        fnl_idx = init_idx + iterative_step * interval_idx
        targets = targets_all[:, init_idx:fnl_idx]

        if self.configs['train'].get('time_condition'):
            inputs, t = self.time_condition(inputs, targets)
        else:
            t = None

        inputs = inputs.float()
        targets = targets.float()
        preds = self(inputs, t)

        if self.configs['train'].get('clip') is not None:
            cmin, cmax = self.configs['train']['clip']
            preds = torch.clip(preds, cmin, cmax)

        loss = self.loss_fx(preds, targets).to(self.device)
        targets_eval, preds_eval = self._denormalize(targets, preds, device=inputs.device)

        scores = cal_metrics(preds_eval, targets_eval, self.configs['train']['target_names'])
        metrics = {f'{split}_loss': loss.detach()}
        metrics.update({f'{split}_{key}': value for key, value in scores.items()})

        return metrics, loss

    def training_step(self, batch, batch_idx):

        inputs, targets_all, init_time = batch

        iterative_step = self.configs['train'].get('iterative_step', 1)
        interval_idx = self._get_ar_step(self.configs)
        init_idx = 0
        fnl_idx = init_idx + iterative_step * interval_idx
        targets = targets_all[:, init_idx:fnl_idx]
        
        if self.configs['train'].get('time_condition'):
            inputs, t = self.time_condition(inputs, targets)
        else:
            t = None

        inputs = inputs.float()
        targets = targets.float()
        preds = self(inputs, t)

        if self.configs['train'].get('clip') is not None and self.configs['train'].get('clip'):
            cmin, cmax = self.configs['train']['clip']
            preds = torch.clip(preds, cmin, cmax)

        train_loss = self.loss_fx(preds, targets).to(self.device)

        self.train_step_metrics['train_loss'].append(train_loss.detach())

        metrics_train = {'train_loss': train_loss.item()}

        self.log_dict(
            {**metrics_train, 'train_loss': train_loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=inputs.shape[0],
        )

        return {'loss': train_loss, **metrics_train}

    def on_training_epoch_end(self):
        train_loss_epoch = torch.stack(self.train_step_metrics['train_loss']).mean()
        self.train_step_metrics['train_loss'].clear()

        logger.info(f"loss on train epoch {self.current_epoch} end: {train_loss_epoch}")

        self.log_dict({'epoch_train_loss': train_loss_epoch}, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        metrics_valid, losses = self._shared_eval_step(batch, 'valid')

        self.val_step_metrics['valid_loss'].append(losses)

        self.log_dict(metrics_valid, on_step=False, on_epoch=True, prog_bar=True,
                      sync_dist=True, batch_size=batch[0].shape[0])

        return metrics_valid

    def test_step(self, batch, batch_idx):
        metrics_test, loss = self._shared_eval_step(batch, 'test')
        self.test_step_metrics['test_loss'].append(loss.detach())
        self.log_dict(metrics_test, on_step=False, on_epoch=True, prog_bar=True,
                      sync_dist=True, batch_size=batch[0].shape[0])
        return metrics_test

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, init_time = batch
        inputs = inputs.float()
        preds = self(inputs)
        if self.configs['train'].get('clip') is not None:
            cmin, cmax = self.configs['train']['clip']
            preds = torch.clip(preds, cmin, cmax)
        preds, = self._denormalize(preds, device=inputs.device)
        return {'init_time': init_time, 'prediction': preds.detach().cpu()}

    @torch.no_grad()
    def on_validation_epoch_end(self):

        scores = dict()
        for key, value in self.val_step_metrics.items():
            scores[key] = torch.stack(self.val_step_metrics[key]).mean()
            self.val_step_metrics[key].clear()

        valid_loss_epoch = scores['valid_loss']

        logger.info(f'validation loss on epoch {self.current_epoch}: {valid_loss_epoch:.6f}')

        for key, value in scores.items():
            logger.info(f"validation score on epoch {self.current_epoch}: {key}, {value:.6f}")

        self.log('valid_loss_epoch', valid_loss_epoch, sync_dist=True)

        self.save_scores(scores, 'valid')

        self.log_dict({'epoch_valid_loss': valid_loss_epoch}, sync_dist=True)

    @torch.no_grad()
    def on_test_epoch_end(self):
        if not self.test_step_metrics['test_loss']:
            return
        test_loss_epoch = torch.stack(self.test_step_metrics['test_loss']).mean()
        self.test_step_metrics['test_loss'].clear()
        logger.info(f'test loss: {test_loss_epoch:.6f}')
        self.log_dict({'epoch_test_loss': test_loss_epoch}, sync_dist=True)

    def save_scores(self, scores, split):
        save_path = self.configs.get('project', {}).get('save_path')
        if save_path is None:
            return
        os.makedirs(save_path, exist_ok=True)
        serializable_scores = {}
        for key, value in scores.items():
            if torch.is_tensor(value):
                serializable_scores[key] = float(value.detach().cpu())
            else:
                serializable_scores[key] = value
        score_path = os.path.join(save_path, f'{split}_scores.pkl')
        with open(score_path, 'wb') as f:
            pickle.dump(serializable_scores, f)

    def configure_optimizers(self):
        model_parameters = self.parameters()

        if self.configs['train'].get('scheduler_params', None) is not None:
            sch_params = self.configs['train']['scheduler_params']
        else:
            sch_params = {'interval': 'epoch'}

        if self.configs['train'].get('scheduler') is not None:
            optimizer, scheduler = configure_optimizer(self.configs, model_parameters)
            if self.configs['train'].get('scheduler').lower() not in ['reducelr']:
                return {'optimizer': optimizer,
                        'lr_scheduler': {'scheduler': scheduler,
                                         **sch_params
                                         }
                        }
            else:
                return {'optimizer': optimizer,
                        'lr_scheduler': {'scheduler': scheduler,
                                         'monitor': self.configs['train']['reducelr']['monitor'],
                                         **sch_params
                                         }
                        }
        else:
            raise ValueError('Must be setting optimizer!')
