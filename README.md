[English](README.md) [中文](README_zh.md)
    
# FengYuan-Weather
FengYuan-Weather model utilizes data from past 12-hour as input to forecast surface and upper-air meteorological variables for the next 10 days at 6-hour intervals and a spatial resolution 0.25 degree, as shown in the table show. This repository presents the inference code and pre-trained model of FengYuan-Weather, a deep learning-based weather forecasting model.

    
## Data Format

The model takes two consecutive six-hour data frames as input. input1.npy represents the atmospheric data at the first time moment, while input2.npy represents the atmospheric data six hours later. For example, if input1.npy represents the atmospheric state at 00:00 on January 1, 2022, then input2.npy represents the atmospheric state at 06:00 on the same day. The first predicted data corresponds to the atmospheric state at 12:00 on January 1, 2022, and the second predicted data corresponds to the atmospheric state at 18:00 on January 1, 2022.

The data is organized in the following order: Each individual data has a shape of 70x721x1440, where 70 represents 70 atmospheric features. The  latitude range is the [90N, 90S], and the longitude range is [0, 360]. 

The first 65 variables are arranged in the order ['t', 'u', 'v', 'z', 'q']. Each variable contains 13 levels, which are ordered as [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]. The following five variables are surface variables are ['t2m', 'u10', 'v10', 'msl', 'tp']. Therefore, the order of the 70 variables is ['t50','u50','v50','z50','q50','t100','u100','v100','z100','q100','t150','u150','v150','z150','q150', 't200','u200','v200','z200','q200','t250','u250','v250','z250','q250','t300','u300','v300','z300','q300', 't400','u400','v400','z400','q400','t500','u500','v500','z500','q500','t600','u600','v600','z600','q600', 't700','u700','v700','z700','q700','t850','u850','v850','z850','q850','t925','u925','v925','z925','q925', 't1000','u1000','v1000','z1000','q1000','t2m','u10','v10','msl','tp']


## Inference

Before performing inference for forecasting, you need to convert the input data into the required format according to the **Data Format** specification. Once the data format conversion is complete, you can directly run inference to generate a forecast for the next 10 days.
    
If you want a 6-hourly forecast for the next 10 days, you can use the following command:
    
```bash
python inference.py \
    --in1 data/20220101000000.npy \
    --in2 data/20220101060000.npy \
    --mean data/mean-1979-2019.npy \
    --std data/std-1979-2019.npy \
    --model-short data/ckpts/fengyuan_short.onnx \
    --model-medium data/ckpts/fengyuan_medium.onnx \
    --n-short 20 \
    --n-medium 20 \
    --step-hours 6 \
    --output-dir ./output \
    --engine netcdf4 \
    --verbose
```

You can also customize the forecast horizon. If you only need a 5-day forecast, simply run inference with the short model. In addition, you can also customize the output naming format using the following command.

```bash
python inference.py \
    --in1 data/20220101000000.npy \
    --in2 data/20220101060000.npy \
    --mean data/mean-1979-2019.npy \
    --std data/std-1979-2019.npy \
    --n-short 20 \
    --step-hours 6 \
    --output-dir ./output \
    --model-short data/ckpts/fengyuan_short.onnx \
    --filename-format "%Y-%m-%d_%H:%M:%S.nc" \
    --filename-prefix "weather_" \
    --filename-suffix "_forecast"
```
    
Additionally, you can access the [run_fengyuan_v1.0.ipynb](https://github.com/MetAILab/FengYuan-Weather/blob/main/run_fengyuan_v1.0.ipynb) to learn how to download realtime ECMWF open data and load the FengYuan-Weather model weights to generate weather forecasts for the next 10 days.
    
For example, downloading the ECMWF open data from 00:00 on December 21, 2025, to forecast the weather for the next 10 days. The following figure shows the 2m temperature prediction for the upcoming 10 days.

![2m temperature](https://github.com/MetAILab/FengYuan-Weather/blob/main/imgs/20251221.gif)


## Requriments

- numpy
- onnxruntime
- xarray
- netcdf4 or h5netcdf

