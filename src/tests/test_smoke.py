import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from dataset.era_numpy import EC2Sets
from main import DEFAULT_RUNTIME_CONFIG, get_kwargs
from models.utransformer import SphericalPad2d


def test_cli_overrides_do_not_replace_yaml_values():
    yaml_config = OmegaConf.create({'lossfx': 'mae', 'model_name': 'fengyuan'})
    configs = OmegaConf.merge(
        OmegaConf.create(DEFAULT_RUNTIME_CONFIG),
        yaml_config,
        get_kwargs({}),
    )

    assert configs.lossfx == 'mae'
    assert configs.model_name == 'fengyuan'


def test_spherical_pad2d_wraps_longitude_and_zero_pads_latitude():
    inputs = torch.tensor([[[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0]]]])

    outputs = SphericalPad2d((1, 1, 1, 1))(inputs)

    assert outputs.shape == (1, 1, 4, 5)
    assert torch.all(outputs[:, :, 0] == 0)
    assert torch.all(outputs[:, :, -1] == 0)
    assert outputs[0, 0, 1, 0] == 3.0
    assert outputs[0, 0, 1, -1] == 1.0


def test_ec2sets_test_and_predict_shapes(tmp_path):
    init_time = pd.Timestamp('2020-01-01 00:00:00')
    for offset, value in enumerate([1.0, 2.0, 3.0]):
        timestamp = init_time + pd.Timedelta(hours=6 * offset)
        np.save(tmp_path / f'{timestamp:%Y%m%d%H%M%S}.npy',
                np.full((2, 4, 4), value, dtype=np.float32))

    supervised = EC2Sets(
        data_dir=tmp_path,
        init_times=pd.to_datetime([init_time]),
        mode='test',
        input_step=2,
        output_step=1,
        interval=6,
        iterative_step=1,
        in_var_index=[0],
        out_var_index=[1],
        start_idx=0,
        end_idx=1,
    )
    inputs, targets, name = supervised[0]

    assert inputs.shape == (2, 1, 4, 4)
    assert targets.shape == (1, 1, 4, 4)
    assert name == '20200101060000'

    predict = EC2Sets(
        data_dir=tmp_path,
        init_times=pd.to_datetime([init_time]),
        mode='predict',
        input_step=2,
        output_step=1,
        interval=6,
        iterative_step=1,
        in_var_index=[0],
    )
    pred_inputs, pred_name = predict[0]

    assert pred_inputs.shape == (2, 1, 4, 4)
    assert pred_name == '20200101060000'
