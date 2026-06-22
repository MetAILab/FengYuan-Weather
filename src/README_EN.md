# FengYuan Training Code

[中文 README](README.md)

This repository provides the training code for FengYuan, a meteorological foundation model for medium-range weather forecasting. It is implemented with PyTorch and Lightning. The current open-source version includes a NumPy-based ERA5 data pipeline, the FengYuan backbone, training/validation/test/prediction entry points, combined loss configuration, optimizer setup, and minimal smoke tests.

The FengYuan inference code is available in a separate open-source repository: [MetAILab/FengYuan-Weather](https://github.com/MetAILab/FengYuan-Weather). The FengYuan paper has been published on the [Journal of Meteorological Research (JMR)](http://jmr.cmsjournal.net/) website: [FengYuan: An End-to-End Global Weather Forecasting Model Driven by Multi-Source Observational Data](http://jmr.cmsjournal.net/article/doi/10.1007/s13351-026-6042-4).

## Scope

This repository focuses on the training workflow. It does not include large ERA5 datasets or the dedicated inference service scripts. The default configuration uses 70 ERA5 variables, 2 historical input time steps, and 1 forecast target step. Multi-step autoregressive forecasting and fine-tuning can be configured through `num_step`, `iterative_step`, and `forecast_length`.

Key features:

- Lightning-based training and data modules.
- Time-indexed ERA5 `.npy` loading.
- Variable-wise combined MAE loss and per-variable RMSE metrics.
- Default attention path without mandatory `flash_attn` or `natten`.
- CPU-friendly smoke tests for imports, dataset contracts, and spherical padding.

## Related Resources

- Training code: this repository.
- Inference code: [https://github.com/MetAILab/FengYuan-Weather](https://github.com/MetAILab/FengYuan-Weather)
- Paper page: [FengYuan: An End-to-End Global Weather Forecasting Model Driven by Multi-Source Observational Data](http://jmr.cmsjournal.net/article/doi/10.1007/s13351-026-6042-4)

## Code Overview

```text
.
├── main.py                         # Train, test, and predict entry point
├── configs/
│   ├── fengyuan_train.yaml          # Default training config
│   └── fengyuan_fientune.yaml       # Fine-tuning config, current file name kept
├── dataset/
│   └── era_numpy.py                 # ERA5 NumPy dataset and LightningDataModule
├── models/
│   ├── fengyuan.py                  # FengYuan model wrapper
│   ├── utransformer.py              # U-Transformer backbone
│   ├── attention.py                 # Attention modules
│   ├── lora.py                      # LoRA modules
│   └── utils.py                     # Model utilities
├── losses/
│   └── __init__.py                  # MSE, MAE, and combined loss
├── utils/
│   ├── trainer.py                   # LightningModule training logic
│   ├── optim.py                     # Optimizer and LR scheduler
│   ├── metrics.py                   # Validation/test metrics
│   └── tools.py                     # Config and file helpers
├── index/                           # Example index CSV files
├── cons/                            # Variable names, mean, and std files
└── tests/
    └── test_smoke.py                # Minimal smoke tests
```

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

The default `attn_type: basic` configuration does not require `flash_attn` or `natten`. Install those optional dependencies only if you enable the corresponding attention implementations.

## Data Preparation

Prepare ERA5 data as one `.npy` file per timestamp:

- File name format: `YYYYmmddHHMMSS.npy`, for example `20200101000000.npy`.
- Default tensor shape: `(70, H, W)`.
- Default full-resolution grid: `H=721`, `W=1440`.

Update these fields before training:

```yaml
data:
  data_dir: "/path/to/era5_npy"
  valid_dir: "/path/to/era5_npy"
  test_dir: "/path/to/era5_npy"
  train_file: "index/train_index_41yr.csv"
  valid_file: "index/valid_index.csv"
  test_file: "index/test_index.csv"
  variables_names: "cons/variables_names.npy"
  mean_file: "cons/mean-1979-2019.npy"
  std_file: "cons/std-1979-2019.npy"
```

The index CSV files must contain an `init_times` column. This repository includes example `index/*.csv` and `cons/*.npy` files, but does not include large ERA5 data files.

## Training

Run default training from the repository root:

```bash
python main.py run --configs configs/fengyuan_train.yaml
```

The default `gpus: [-1]` setting uses CPU and is suitable for checking the code path. For GPU training:

```bash
python main.py run --configs configs/fengyuan_train.yaml --gpus='[0,1]' --strategy=ddp
```

Resume from a checkpoint:

```bash
python main.py run \
  --configs configs/fengyuan_train.yaml \
  --checkpoint_path /path/to/checkpoint.ckpt
```

Fine-tuning:

```bash
python main.py run --configs configs/fengyuan_fientune.yaml
```

Outputs are written to:

```text
results/<project.name>/<experiment>_<mode>/
```

The output directory contains logs, checkpoints, and validation score files.

## Test And Predict

Test a checkpoint:

```bash
python main.py run \
  --configs configs/fengyuan_train.yaml \
  --mode test \
  --checkpoint_path /path/to/checkpoint.ckpt
```

Run prediction:

```bash
python main.py run \
  --configs configs/fengyuan_train.yaml \
  --mode predict \
  --checkpoint_path /path/to/checkpoint.ckpt
```

If you only need to run inference with trained weights, please refer to the dedicated inference repository: [MetAILab/FengYuan-Weather](https://github.com/MetAILab/FengYuan-Weather).

## Quick Checks

Run smoke tests:

```bash
pytest -q
```

Check model imports:

```bash
python -c "from models.fengyuan import FengYuan; print('ok')"
```

Check Python syntax:

```bash
python -m compileall -q main.py dataset models utils losses tests
```

## Configuration Notes

Common configuration fields:

- `model.img_size`: input time length, latitude size, and longitude size. Default: `[2, 721, 1440]`.
- `model.patch_size`: 3D patch size. Default: `[2, 4, 4]`.
- `model.in_channels` / `model.out_channels`: number of input and output variables. Default: 70.
- `train.initial_length`: number of historical input time steps.
- `train.forecast_length`: target length read for each sample.
- `train.num_step`: autoregressive model forward steps.
- `train.iterative_step`: outer iteration count.
- `train.norm_flag`: whether to normalize data with `mean_file` and `std_file`.
- `train.loss.combineloss`: variable-wise loss type and weight settings.

## Notes

- The default model is large and requires adequate GPU memory for full training.
- ERA5 raw data must be prepared separately.
- The fine-tuning config keeps the current repository file name: `configs/fengyuan_fientune.yaml`.
- If you change the variable set, update `train.var_names`, `train.target_names`, `model.in_channels`, `model.out_channels`, `model.cube_in_channels`, and `model.cube_out_channels` together.
- Please refer to the official JMR page for paper citation details.

## References And Acknowledgements

This repository refers to the design and implementation of several excellent open-source projects while preparing the training code and related modules. We thank the corresponding teams and open-source communities:

- [Microsoft Aurora](https://github.com/microsoft/aurora): a foundation model project for Earth system forecasting.
- [WeatherLearn](https://github.com/lizhuoq/WeatherLearn): a deep learning framework and toolkit for weather forecasting.
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer): Swin Transformer and window attention implementations.
