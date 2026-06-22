# 风源 FengYuan 训练代码

[English README](README_EN.md)

本仓库提供风源（FengYuan）气象大模型的训练代码，面向中期天气预报任务，基于 PyTorch 和 Lightning 实现。当前开源版本包含 NumPy 格式 ERA5 数据读取、FengYuan 主干模型、训练/验证/测试/预测入口、组合损失函数、优化器配置和最小 smoke tests，便于研究人员在本地数据上复现、训练和扩展模型。

风源论文已发布在 [Journal of Meteorological Research (JMR)](http://jmr.cmsjournal.net/) 网站，论文链接[FengYuan: An End-to-End Global Weather Forecasting Model Driven by Multi-Source Observational Data](http://jmr.cmsjournal.net/article/doi/10.1007/s13351-026-6042-4)。

## 项目定位

本仓库聚焦训练流程，不包含大体量 ERA5 数据和推断服务。默认配置面向 70 个 ERA5 变量，输入 2 个历史时次，输出后续 1 个预报时次。通过调整配置中的 `num_step`、`iterative_step`、`forecast_length` 等参数，可以扩展为自回归多步预报或微调训练流程。

主要功能：

- 基于 `LightningModule` 和 `LightningDataModule` 的训练框架。
- 支持按时间索引读取 ERA5 `.npy` 文件。
- 支持变量级组合 MAE 损失和按变量 RMSE 验证指标。
- 默认使用普通注意力路径，不强制依赖 `flash_attn` 或 `natten`。

## 相关资源

- 训练代码：当前仓库。
- 推断代码：[https://github.com/MetAILab/FengYuan-Weather](https://github.com/MetAILab/FengYuan-Weather)
- 论文页面：[FengYuan: An End-to-End Global Weather Forecasting Model Driven by Multi-Source Observational Data](http://jmr.cmsjournal.net/article/doi/10.1007/s13351-026-6042-4)

## 代码结构

```text
.
├── main.py                         # 训练、测试、预测入口
├── configs/
│   ├── fengyuan_train.yaml          # 默认训练配置
│   └── fengyuan_fientune.yaml       # 微调配置，保留当前文件名
├── dataset/
│   └── era_numpy.py                 # ERA5 NumPy 数据集和 LightningDataModule
├── models/
│   ├── fengyuan.py                  # FengYuan 模型封装
│   ├── utransformer.py              # U-Transformer 主体结构
│   ├── attention.py                 # 注意力模块
│   ├── lora.py                      # LoRA 相关模块
│   └── utils.py                     # 模型辅助函数
├── losses/
│   └── __init__.py                  # MSE、MAE 和组合损失
├── utils/
│   ├── trainer.py                   # LightningModule 训练逻辑
│   ├── optim.py                     # 优化器和学习率调度器
│   ├── metrics.py                   # 验证/测试指标
│   └── tools.py                     # 配置和文件工具
├── index/                           # 默认索引 CSV 示例
├── cons/                            # 变量名、均值和标准差文件
└── tests/
    └── test_smoke.py                # 最小测试
```

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

默认配置使用 `attn_type: basic`，不需要安装 `flash_attn` 和 `natten`。如果需要启用 Flash Attention 或 Neighborhood Attention，请根据本机 CUDA、PyTorch 版本单独安装对应依赖。

## 数据准备

训练前需要准备 ERA5 NumPy 文件。默认读取格式如下：

- 每个时次一个 `.npy` 文件。
- 文件名格式为 `YYYYmmddHHMMSS.npy`，例如 `20200101000000.npy`。
- 单个文件默认为 `(70, H, W)`，与配置中的变量列表一致。
- 默认全分辨率配置为 `H=721`、`W=1440`。

配置文件中需要重点修改以下字段：

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

索引 CSV 必须包含 `init_times` 列，时间格式可以被 `pandas.to_datetime` 解析。仓库提供了 `index/*.csv` 和 `cons/*.npy` 示例文件，但不包含大体量 ERA5 数据。

## 训练

从仓库根目录启动默认训练：

```bash
python main.py run --configs configs/fengyuan_train.yaml
```

默认配置中的 `gpus: [-1]` 表示 CPU 运行，适合检查配置和代码链路。真实训练建议显式指定 GPU：

```bash
python main.py run --configs configs/fengyuan_train.yaml --gpus='[0,1]' --strategy=ddp
```

继续训练或从 checkpoint 启动：

```bash
python main.py run \
  --configs configs/fengyuan_train.yaml \
  --checkpoint_path /path/to/checkpoint.ckpt
```

微调配置：

```bash
python main.py run --configs configs/fengyuan_fientune.yaml
```

训练结果默认写入：

```text
results/<project.name>/<experiment>_<mode>/
```

其中包含日志、checkpoint 和验证分数文件。

## 测试与预测

测试 checkpoint：

```bash
python main.py run \
  --configs configs/fengyuan_train.yaml \
  --mode test \
  --checkpoint_path /path/to/checkpoint.ckpt
```

预测：

```bash
python main.py run \
  --configs configs/fengyuan_train.yaml \
  --mode predict \
  --checkpoint_path /path/to/checkpoint.ckpt
```

如果只需要运行已训练模型的推断流程，请优先参考推断代码仓库：[MetAILab/FengYuan-Weather](https://github.com/MetAILab/FengYuan-Weather)。

## 快速检查

不依赖真实 ERA5 数据的 smoke tests：

```bash
pytest -q
```

检查模型导入：

```bash
python -c "from models.fengyuan import FengYuan; print('ok')"
```

检查 Python 语法：

```bash
python -m compileall -q main.py dataset models utils losses tests
```

## 配置说明

常用配置项：

- `model.img_size`：输入时间长度、纬向大小和经向大小，默认 `[2, 721, 1440]`。
- `model.patch_size`：三维 patch 大小，默认 `[2, 4, 4]`。
- `model.in_channels` / `model.out_channels`：输入和输出变量数，默认 70。
- `train.initial_length`：输入历史时次数。
- `train.forecast_length`：每次样本读取的目标长度。
- `train.num_step`：模型前向自回归步数。
- `train.iterative_step`：外层迭代轮数。
- `train.norm_flag`：是否使用 `mean_file` 和 `std_file` 做标准化。
- `train.loss.combineloss`：按变量配置损失类型和权重。

## 注意事项

- 默认模型规模较大，完整训练需要足够 GPU 显存。
- 仓库不包含 ERA5 原始数据，需要用户自行准备。
- `configs/fengyuan_fientune.yaml` 的文件名保留了当前仓库拼写，使用时请按实际文件名调用。
- 如果修改变量数量，需要同步更新 `train.var_names`、`train.target_names`、`model.in_channels`、`model.out_channels`、`model.cube_in_channels` 和 `model.cube_out_channels`。
- 论文引用信息请以 JMR 网站正式页面为准。

## 参考与致谢

本仓库在整理开源训练代码和相关模块时，参考了以下优秀项目的设计和实现。感谢相关团队和社区的开源贡献：

- [Microsoft Aurora](https://github.com/microsoft/aurora)：面向地球系统预测的基础模型开源项目。
- [WeatherLearn](https://github.com/lizhuoq/WeatherLearn)：天气预报深度学习训练框架与工具。
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)：Swin Transformer 模型及窗口注意力相关实现。
