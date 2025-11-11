import os
import re
import argparse
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import onnxruntime as ort


def parse_arguments():
    parser = argparse.ArgumentParser(description='the autoregressive prediction tool for the FengYuan-Weather model')
    
    # input files 
    parser.add_argument('--in1', '--input1', type=str, 
                       default="data/20220101120000.npy",
                       help='the first time file')
    parser.add_argument('--in2', '--input2', type=str,
                       default="data/20220101180000.npy", 
                       help='the second time file')
    parser.add_argument('--mean', type=str, default="data/mean-1979-2019.npy",
                       help='the file of variables mean')
    parser.add_argument('--std', type=str, default="data/std-1979-2019.npy",
                       help='the file of variables standard deviation')
    
    parser.add_argument('--model-short', type=str, default="data/ckpts/fengyuan_short.onnx",
                       help='the short model file')
    parser.add_argument('--model-medium', type=str, default="data/ckpts/fengyuan_medium.onnx", 
                       help='the medium model file')
    parser.add_argument('--n-short', type=int, default=20,
                       help='the number of prediction steps for the short model')
    parser.add_argument('--n-medium', type=int, default=20,
                       help='the number of prediction steps for the medium model')
    parser.add_argument('--step-hours', type=int, default=6,
                       help='the time step for the prediction')
    parser.add_argument('--forecast-hours', type=int, default=240,
                       help='the total forecast time')
    
    # output parameters
    parser.add_argument('--output-dir', type=str, default="./",
                       help='the output directory')
    parser.add_argument('--engine', type=str, default="netcdf4", 
                       choices=["netcdf4", "h5netcdf"],
                       help='the engine for the output file')
    parser.add_argument('--filename-format', type=str, default="%Y%m%d%H%M%S.nc",
                       help='the format of the output file name, default: %%Y%%m%%d%%H%%M%%S.nc')
    parser.add_argument('--filename-prefix', type=str, default="",
                       help='the prefix of the output file name')
    parser.add_argument('--filename-suffix', type=str, default="",
                       help='the suffix of the output file name')
    
    # other parameters
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='the verbose mode')
    
    return parser.parse_args()


class Config:
    """the configuration manager"""
    
    def __init__(self, args):
        self.in1_path = args.in1
        self.in2_path = args.in2
        self.mean_path = args.mean
        self.std_path = args.std
        
        self.model_short = args.model_short
        self.model_medium = args.model_medium
        self.n_short = args.n_short
        self.n_medium = args.n_medium
        self.step_hours = args.step_hours
        self.forecast_hours = args.forecast_hours
        
        self.save_dir = args.output_dir
        self.engine = args.engine
        self.verbose = args.verbose
        
        # file name configuration
        self.filename_format = args.filename_format
        self.filename_prefix = args.filename_prefix
        self.filename_suffix = args.filename_suffix
        
        # variable names list
        self.var_names = [
            't50','u50','v50','z50','q50','t100','u100','v100','z100','q100','t150','u150','v150','z150','q150',
            't200','u200','v200','z200','q200','t250','u250','v250','z250','q250','t300','u300','v300','z300','q300',
            't400','u400','v400','z400','q400','t500','u500','v500','z500','q500','t600','u600','v600','z600','q600',
            't700','u700','v700','z700','q700','t850','u850','v850','z850','q850','t925','u925','v925','z925','q925',
            't1000','u1000','v1000','z1000','q1000','t2m','u10','v10','msl','tp'
        ]
        
        # NetCDF encoding configuration
        self.encoding = {
            "forecast": {
                "zlib": True if self.engine == "netcdf4" else False,
                "complevel": 9,
                "dtype": "float32",
                "chunksizes": (1, 10, 180, 180)
            }
        }
        
        self._calculate_model_steps()
    
    def _calculate_model_steps(self):
        """calculate the number of steps for the model based on the forecast time"""
        if self.forecast_hours is not None:
            # calculate the total number of steps
            total_steps = self.forecast_hours // self.step_hours
            
            # calculate the maximum number of steps for the short model
            short_max_steps = self.n_short
            
            if total_steps <= short_max_steps:
                # only use the short model
                self.actual_n_short = total_steps
                self.actual_n_medium = 0
                self.use_medium_model = False
                if self.verbose:
                    print(f"the forecast time {self.forecast_hours} hours, using the short model ({self.actual_n_short} steps)")
            else:
                # use the short + medium model
                self.actual_n_short = self.n_short
                self.actual_n_medium = total_steps - self.n_short
                if self.actual_n_medium > self.n_medium:
                    print(f"the number of steps for the medium model {self.actual_n_medium} is greater than the maximum value {self.n_medium}, using the maximum value {self.n_medium} steps")
                    self.actual_n_medium = self.n_medium
                self.use_medium_model = True
                if self.verbose:
                    print(f"the forecast time {self.forecast_hours} hours, using the short model ({self.actual_n_short} steps) + the medium model ({self.actual_n_medium} steps)")
        else:
            # use the user specified steps
            self.actual_n_short = self.n_short
            self.actual_n_medium = self.n_medium
            self.use_medium_model = self.n_medium > 0
            if self.verbose:
                print(f"use the user specified steps: short {self.actual_n_short} steps, medium {self.actual_n_medium} steps")


class ModelManager:
    """ONNX model manager"""
    
    def __init__(self, config: Config):
        self.config = config
        self.providers = self._get_available_providers()
        self.sess_short = None
        self.sess_medium = None
        self._load_models()
    
    def _get_available_providers(self):
        """get the available execution providers"""
        avail = ort.get_available_providers()
        pick = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]
        return pick if pick else avail
    
    def _load_models(self):
        """load the ONNX models"""
        try:
            self.sess_short = ort.InferenceSession(self.config.model_short, providers=self.providers)
            self.sess_medium = ort.InferenceSession(self.config.model_medium, providers=self.providers)
            
            if self.config.verbose:
                print(f"Providers (short): {self.sess_short.get_providers()}")
                print(f"Providers (medium): {self.sess_medium.get_providers()}")
                
        except Exception as e:
            raise RuntimeError(f"model loading failed: {e}")
    
    def get_io_info(self, session):
        """get the input and output information of the model"""
        typemap = {
            "tensor(float16)": np.float16,
            "tensor(float)":   np.float32,
            "tensor(double)":  np.float64,
            "tensor(int64)":   np.int64,
            "tensor(int32)":   np.int32,
            "tensor(int16)":   np.int16,
            "tensor(int8)":    np.int8,
            "tensor(uint8)":   np.uint8,
            "tensor(bool)":    np.bool_,
        }
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        return inp.name, typemap.get(inp.type, np.float32), typemap.get(out.type, np.float32)
    
    def run_inference(self, session, in_name: str, in_dtype: np.dtype,
                     prev_frame: np.ndarray, curr_frame: np.ndarray,
                     mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """run the model inference"""
        x = self._normalize_pair(prev_frame, curr_frame, mean, std, in_dtype)
        y_list = session.run(None, {in_name: x})
        y = y_list[0]
        y = self._ensure_var_hw(y)
        y = self._denorm(y, mean, std)
        return y
    
    def _normalize_pair(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                       mean: np.ndarray, std: np.ndarray,
                       dtype: np.dtype) -> np.ndarray:
        """normalize the input data pair"""
        stacked = np.stack([prev_frame, curr_frame], axis=0)
        normed = (stacked - mean) / std
        x = normed[None, ...]
        return x.astype(dtype, copy=False)
    
    def _denorm(self, y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """denormalize the output data"""
        return (y * std + mean).astype(np.float32, copy=False)
    
    def _ensure_var_hw(self, y: np.ndarray) -> np.ndarray:
        """ensure the output shape is (70, 721, 1440)"""
        yy = np.asarray(y)
        while yy.ndim > 3:
            if yy.shape[0] == 1:
                yy = yy[0]
            else:
                break
        assert yy.ndim == 3, f"Unexpected output shape after squeeze: {yy.shape}"
        assert yy.shape[0] == 70, f"Channel/variable dimension != 70: {yy.shape}"
        return yy


class DataProcessor:
    """data processor"""
    
    def __init__(self, config: Config):
        self.config = config
        self.mean = None
        self.std = None
        self._load_normalization_data()
    
    def _load_normalization_data(self):
        """load the normalization data"""
        try:
            self.mean = np.load(self.config.mean_path).astype(np.float32)
            self.std = np.load(self.config.std_path).astype(np.float32)
            if self.config.verbose:
                print(f"normalization data loaded: mean shape={self.mean.shape}, std shape={self.std.shape}")
        except Exception as e:
            raise RuntimeError(f"normalization data loading failed: {e}")
    
    def load_initial_data(self):
        """load the initial data"""
        try:
            f0 = np.load(self.config.in1_path).astype(np.float32)
            f1 = np.load(self.config.in2_path).astype(np.float32)
            if self.config.verbose:
                print(f"initial data loaded: f0 shape={f0.shape}, f1 shape={f1.shape}")
            return f0, f1
        except Exception as e:
            raise RuntimeError(f"initial data loading failed: {e}")
    
    def parse_timestamp(self, path: str) -> Optional[datetime]:
        """parse the timestamp from the file path"""
        m = re.search(r"(\d{14})", os.path.basename(path))
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except Exception:
            return None


class OutputManager:
    """output manager"""
    
    def __init__(self, config: Config):
        self.config = config
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """ensure the output directory exists"""
        os.makedirs(self.config.save_dir, exist_ok=True)
        if self.config.verbose:
            print(f"output directory: {os.path.abspath(self.config.save_dir)}")
    
    def save_step_to_netcdf(self, y_var_hw: np.ndarray, out_path: str, 
                           when: Optional[datetime]) -> None:
        """save the single step prediction result to the NetCDF file"""
        variable = np.array(self.config.var_names, dtype=object)
        y_coords = np.arange(y_var_hw.shape[1], dtype=np.int32)
        x_coords = np.arange(y_var_hw.shape[2], dtype=np.int32)

        da = xr.DataArray(
            y_var_hw[None, ...],  # add the time dimension
            dims=("time", "variable", "y", "x"),
            coords={
                "time": [np.datetime64(when) if when is not None else np.datetime64("NaT")],
                "variable": variable,
                "y": y_coords,
                "x": x_coords
            },
            name="forecast",
            attrs={"description": "Autoregressive forecast (one step)", "units": "SI/mixed"}
        )
        ds = xr.Dataset({"forecast": da})
        ds.to_netcdf(out_path, format="NETCDF4" if self.config.engine == "netcdf4" else None,
                     engine=self.config.engine, encoding=self.config.encoding)
        ds.close()
    
    def generate_output_filename(self, step: int, when: Optional[datetime], 
                               model_type: str) -> str:
        """generate the output file name"""
        if when is not None:
            # use the configured time format
            time_str = when.strftime(self.config.filename_format)
            # if the format does not end with .nc, add it
            if not time_str.endswith('.nc'):
                time_str += '.nc'
            out_name = f"{self.config.filename_prefix}{time_str}{self.config.filename_suffix}"
        else:
            # if there is no timestamp, use the step number
            out_name = f"{self.config.filename_prefix}{model_type}_step{step:02d}.nc{self.config.filename_suffix}"
        
        return out_name


class ForecastRunner:
    """forecast runner"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_processor = DataProcessor(config)
        self.output_manager = OutputManager(config)
        
        self.in_name_s, self.in_dtype_s, _ = self.model_manager.get_io_info(self.model_manager.sess_short)
        self.in_name_m, self.in_dtype_m, _ = self.model_manager.get_io_info(self.model_manager.sess_medium)
        
        if self.config.verbose:
            print(f"Input dtype (short/medium): {self.in_dtype_s}, {self.in_dtype_m}")
    
    def run_forecast(self):
        # load the initial data 
        f0, f1 = self.data_processor.load_initial_data()
        
        # parse the timestamp 
        t = self.data_processor.parse_timestamp(self.config.in2_path)
        has_ts = t is not None
        
        prev, curr = f0, f1
        
        if self.config.actual_n_short > 0:
            prev, curr, t = self._run_short_term_forecast(prev, curr, t, has_ts)
        
        if self.config.use_medium_model and self.config.actual_n_medium > 0:
            self._run_medium_term_forecast(prev, curr, t, has_ts)
        
        print(f"[Done] save to {os.path.abspath(self.config.save_dir)}")
    
    def _run_short_term_forecast(self, prev: np.ndarray, curr: np.ndarray, 
                                t: Optional[datetime], has_ts: bool):
        for i in range(1, self.config.actual_n_short + 1):
            y = self.model_manager.run_inference(
                self.model_manager.sess_short, self.in_name_s, self.in_dtype_s,
                prev, curr, self.data_processor.mean, self.data_processor.std
            )
            
            t = (t + timedelta(hours=self.config.step_hours)) if has_ts else None
            out_name = self.output_manager.generate_output_filename(i, t, "short")
            out_path = os.path.join(self.config.save_dir, out_name)
            
            self.output_manager.save_step_to_netcdf(y, out_path, t)
            prev, curr = curr, y 
            
            print(f"[Short] {i*self.config.step_hours:03d} hours prediction -> {out_path}")
        
        return prev, curr, t
    
    def _run_medium_term_forecast(self, prev: np.ndarray, curr: np.ndarray,
                                 t: Optional[datetime], has_ts: bool):
        for i in range(1, self.config.actual_n_medium + 1):
            y = self.model_manager.run_inference(
                self.model_manager.sess_medium, self.in_name_m, self.in_dtype_m,
                prev, curr, self.data_processor.mean, self.data_processor.std
            )
            
            t = (t + timedelta(hours=self.config.step_hours)) if has_ts else None
            out_name = self.output_manager.generate_output_filename(i, t, "medium")
            out_path = os.path.join(self.config.save_dir, out_name)
            
            self.output_manager.save_step_to_netcdf(y, out_path, t)
            prev, curr = curr, y
            
            print(f"[medium] {self.config.actual_n_short*self.config.step_hours + i*self.config.step_hours:03d} hours prediction -> {out_path}")


def main():
    try:
        args = parse_arguments()
        
        config = Config(args)
        
        runner = ForecastRunner(config)
        runner.run_forecast()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

