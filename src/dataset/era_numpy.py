import os
from glob import glob

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def get_index(test_names, variables_names):
    return [variables_names.index(var) for var in test_names]


def load_npy(file_path):
    return np.load(file_path, mmap_mode='r')


def _valid_init_times(index_file, sample_len):
    """Keep starts that have enough subsequent timestamps in a continuous index."""
    init_times = pd.to_datetime(pd.read_csv(index_file)['init_times'].values)
    valid_count = len(init_times) - max(sample_len - 1, 0)
    if valid_count <= 0:
        return init_times[:0]
    return init_times[:valid_count]


def get_flist(data_dir, configs=None, sample_len=6, split_ratio=0.9):
    if configs is None:
        flist = glob(os.path.join(data_dir, '*.npy'))
        df = pd.DataFrame(flist, columns=['file'])
        df['time'] = df['file'].apply(lambda x: os.path.basename(x).split('.')[0])
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')
        df = df.reset_index(drop=True)
        t = list(df['time'])
        t[1:] = t[:-1]
        df['time2'] = t
        df['interval'] = (df['time'] - df['time2']) / pd.Timedelta(1, 'hour')
        inter = df['interval'].values
        ind = [0] + list(np.where(inter > 6)[0])
        flist = list(df['time'])
        lists = []
        for i in range(len(ind)):
            if i < len(ind) - 1:
                dls = flist[ind[i]:ind[i + 1]]
            else:
                dls = flist[ind[i]:]
            if len(dls) < sample_len:
                continue
            lists += dls[:-(sample_len - 1)]
        df = pd.to_datetime(lists)
        train_list = df[:int(len(df) * split_ratio)]
        valid_list = df[int(len(df) * split_ratio):]
        return train_list, valid_list

    train_list = _valid_init_times(configs['data']['train_file'], sample_len)
    valid_list = _valid_init_times(configs['data']['valid_file'], sample_len)
    test_list = _valid_init_times(configs['data']['test_file'], sample_len)
    return train_list, valid_list, test_list


class ECDataModule(L.LightningDataModule):
    def __init__(self, configs, mode, **kwargs):
        super().__init__()
        self.mode = mode
        self.ratio = configs.get('ratio', None)
        self.deg1 = configs.get('deg1', False)

        self.batch_size = configs['train']['batch_size']
        self.val_batch_size = configs['train'].get('val_batch_size', 1)
        self.num_workers = configs['train']['num_workers']
        self.val_num_workers = configs['train'].get('val_num_workers', 1)
        self.pin_memory = configs['train']['pin_memory']
        self.swap_ch_time = configs['data']['swap_ch_time']
        self.data_dir = configs['data'].get('data_dir')
        self.valid_dir = configs['data'].get('valid_dir', self.data_dir)
        self.test_dir = configs['data'].get('test_dir', self.data_dir)
        self.iterative_step = configs['train'].get('iterative_step')
        self.initial_length = configs['train']['initial_length']
        self.forecast_length = configs['train']['forecast_length']
        self.time_interval = configs['train']['time_interval']

        self.mean = np.load(configs['data']['mean_file'])
        self.std = np.load(configs['data']['std_file'])
        self.norm_flag = configs['train'].get('norm_flag')

        self.test_names = configs['train']['target_names']
        self.variables_names = configs['data'].get('variables_names')
        self.names = list(np.load(self.variables_names))
        self.variable_index = get_index(self.test_names, self.names)

        var_names = configs['train']['var_names']
        input_names = configs['train']['var_names']
        target_names = configs['train']['target_names']
        assert len(input_names) == configs['model']['cube_in_channels']
        assert len(target_names) == configs['model']['cube_out_channels']
        self.in_var_idx = get_index(input_names, var_names)
        self.out_var_idx = get_index(target_names, var_names)

        self.start_idx = configs['train']['start_idx']
        self.end_idx = configs['train']['end_idx']

        sample_len = self.initial_length + self.forecast_length * self.iterative_step
        self.train_list, self.valid_list, self.test_list = get_flist(
            self.data_dir,
            configs=configs,
            sample_len=sample_len,
        )
        self.setup()

    def setup(self, stage=None):
        self.train = EC2Sets(
            data_dir=self.data_dir,
            init_times=self.train_list,
            mode='train',
            ratio=self.ratio,
            swap_ch_time=self.swap_ch_time,
            input_step=self.initial_length,
            output_step=self.forecast_length,
            interval=self.time_interval,
            iterative_step=self.iterative_step,
            transform=None,
            in_var_index=self.in_var_idx,
            out_var_index=self.out_var_idx,
            mean=self.mean,
            std=self.std,
            norm_flag=self.norm_flag,
            variable_index=self.variable_index,
            deg1=self.deg1,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
        )
        self.valid = EC2Sets(
            data_dir=self.valid_dir,
            init_times=self.valid_list,
            mode='valid',
            ratio=self.ratio,
            swap_ch_time=self.swap_ch_time,
            input_step=self.initial_length,
            output_step=self.forecast_length,
            interval=self.time_interval,
            iterative_step=self.iterative_step,
            transform=None,
            in_var_index=self.in_var_idx,
            out_var_index=self.out_var_idx,
            mean=self.mean,
            std=self.std,
            norm_flag=self.norm_flag,
            variable_index=self.variable_index,
            deg1=self.deg1,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
        )

        self.test = EC2Sets(
            data_dir=self.test_dir,
            init_times=self.test_list,
            mode='test',
            ratio=self.ratio,
            swap_ch_time=self.swap_ch_time,
            input_step=self.initial_length,
            output_step=self.forecast_length,
            interval=self.time_interval,
            iterative_step=self.iterative_step,
            transform=None,
            in_var_index=self.in_var_idx,
            out_var_index=self.out_var_idx,
            mean=self.mean,
            std=self.std,
            norm_flag=self.norm_flag,
            variable_index=self.variable_index,
            deg1=self.deg1,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
        )

    def _dataloader(self, dataset, batch_size, num_workers, shuffle=False):
        persistent_workers = num_workers > 0
        kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'shuffle': shuffle,
            'pin_memory': self.pin_memory,
            'persistent_workers': persistent_workers,
        }
        if num_workers > 0:
            kwargs['prefetch_factor'] = 1
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return self._dataloader(self.train, self.batch_size, self.num_workers)

    def val_dataloader(self):
        return self._dataloader(self.valid, self.val_batch_size, self.val_num_workers)

    def test_dataloader(self):
        return self._dataloader(self.test, self.batch_size, self.num_workers)

    def predict_dataloader(self):
        predict = EC2Sets(
            data_dir=self.test_dir,
            init_times=self.test_list,
            mode='predict',
            ratio=self.ratio,
            swap_ch_time=self.swap_ch_time,
            input_step=self.initial_length,
            output_step=self.forecast_length,
            interval=self.time_interval,
            iterative_step=self.iterative_step,
            transform=None,
            in_var_index=self.in_var_idx,
            out_var_index=self.out_var_idx,
            mean=self.mean,
            std=self.std,
            norm_flag=self.norm_flag,
            variable_index=self.variable_index,
            deg1=self.deg1,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
        )
        return self._dataloader(predict, 1, self.val_num_workers)


class EC2Sets(Dataset):
    def __init__(self, data_dir=None, init_times=None, mode='train', ratio=None, swap_ch_time=False,
                 input_step=2, output_step=4, interval=6, iterative_step=1, transform=None,
                 in_var_index=None, out_var_index=None, mean=None, std=None, norm_flag=None,
                 variable_index=None, deg1=False, start_idx=None, end_idx=None):
        super().__init__()
        self.init_times = init_times
        self.data_dir = data_dir
        self.in_var_index = in_var_index
        self.out_var_index = out_var_index
        self.iterative_step = iterative_step
        self.mode = mode
        self.swap_ch_time = swap_ch_time
        self.input_step = input_step
        self.output_step = output_step
        self.interval = interval
        self.transform = transform
        self.load_file = self._load_file
        self.deg1 = deg1

        self.mean = mean
        self.std = std
        self.norm_flag = norm_flag
        self.variables_index = variable_index

        self.start_idx = start_idx
        self.end_idx = end_idx

        if ratio is not None:
            total = len(init_times)
            nums = int(total * ratio)
            self.init_times = init_times[:nums]

    def _load_file(self, file_path):
        return np.load(file_path, mmap_mode='r')

    def __len__(self):
        return len(self.init_times)

    def _normalize(self, datas):
        if not self.norm_flag:
            return datas
        mean_var = torch.from_numpy(self.mean)
        std_var = torch.from_numpy(self.std)
        return (datas - mean_var) / std_var

    def _select_inputs(self, datas):
        inputs = datas[:self.input_step]
        if self.in_var_index is not None:
            inputs = inputs[:, self.in_var_index]
        if self.swap_ch_time:
            inputs = inputs.permute(1, 0, 2, 3)
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs

    def _select_targets(self, datas):
        targets = datas[self.input_step:]
        if self.out_var_index is not None:
            targets = targets[:, self.out_var_index]
        return targets[self.start_idx:self.end_idx]

    def __getitem__(self, index):
        t = self.init_times[index]
        init_time = t + pd.Timedelta(hours=self.interval * (self.input_step - 1))
        t2 = init_time + pd.Timedelta(
            hours=self.interval * self.output_step * self.iterative_step
        )
        tid = pd.date_range(t, t2, freq=f'{self.interval}h')

        if self.mode in ['train', 'valid', 'test']:
            datas = [
                torch.from_numpy(np.copy(self.load_file(os.path.join(self.data_dir, f'{dt:%Y%m%d%H%M%S}.npy'))))
                for dt in tid
            ]
            datas = torch.stack(datas, axis=0)
            datas = self._normalize(datas)
            inputs = self._select_inputs(datas)
            targets = self._select_targets(datas)

            if self.deg1:
                return inputs[..., ::4, ::4], targets[..., ::4, ::4], f'{init_time:%Y%m%d%H%M%S}'
            return inputs, targets, f'{init_time:%Y%m%d%H%M%S}'

        inputs = [
            torch.from_numpy(np.copy(self.load_file(os.path.join(self.data_dir, f'{dt:%Y%m%d%H%M%S}.npy'))))
            for dt in tid[:self.input_step]
        ]
        inputs = torch.stack(inputs, axis=0)
        inputs = self._normalize(inputs)
        inputs = self._select_inputs(inputs)

        if self.deg1:
            return inputs[..., ::4, ::4], f'{init_time:%Y%m%d%H%M%S}'
        return inputs, f'{init_time:%Y%m%d%H%M%S}'

    def get_target_by_time(self, target_time_str):
        path = os.path.join(self.data_dir, f'{target_time_str}.npy')
        data = torch.from_numpy(np.copy(self.load_file(path)))
        if self.out_var_index is not None:
            data = data[self.out_var_index]
        if self.deg1:
            data = data[..., ::4, ::4]
        return data


if __name__ == '__main__':
    from tqdm import tqdm

    data_dir = '/workspace/datas/Datasets/ERA_Global_Normalized/'
    flist = glob(os.path.join(data_dir, '*.npy'))
    for i in tqdm(range(len(flist) - 6)):
        x = [torch.from_numpy(load_npy(f)) for f in flist[i:i + 6]]
        x = torch.stack(x, dim=0).float()
