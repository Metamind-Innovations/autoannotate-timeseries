from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TimeSeriesLoader:

    def __init__(self, input_file: Path, timestamp_column: Optional[str] = None):
        self.input_file = Path(input_file)
        self.timestamp_column = timestamp_column

        if not self.input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")

        if not self.input_file.is_file():
            raise ValueError(f"Path must be a file, not a directory: {input_file}")
        
        if self.input_file.suffix.lower() not in [".csv", ".tsv", ".parquet"]:
            raise ValueError(f"Unsupported file format: {self.input_file.suffix}")

    def load_timeseries(self) -> Tuple[List[np.ndarray], List[str], pd.DataFrame]:
        suffix = self.input_file.suffix.lower()
        
        if suffix in [".csv", ".tsv"]:
            sep = '\t' if suffix == ".tsv" else ','
            df = pd.read_csv(self.input_file, sep=sep)
        elif suffix == ".parquet":
            df = pd.read_parquet(self.input_file)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        if self.timestamp_column:
            if self.timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")
            timeseries_columns = [col for col in df.columns if col != self.timestamp_column]
        else:
            first_col = df.columns[0]
            if pd.api.types.is_datetime64_any_dtype(df[first_col]) or \
               pd.api.types.is_string_dtype(df[first_col]):
                logger.info(f"Auto-detected timestamp column: {first_col}")
                self.timestamp_column = first_col
                timeseries_columns = df.columns[1:].tolist()
            else:
                timeseries_columns = df.columns.tolist()
        
        if len(timeseries_columns) == 0:
            raise ValueError("No time series columns found in the file")
        
        logger.info(f"Found {len(timeseries_columns)} time series columns")
        
        series_list = []
        series_names = []
        
        for col in timeseries_columns:
            series = df[col].values.astype(np.float32)
            
            if np.any(np.isnan(series)):
                series = np.nan_to_num(series, nan=np.nanmean(series))
            
            if len(series) < 10:
                logger.warning(f"Series '{col}' too short ({len(series)} points), skipping")
                continue
            
            series_list.append(series)
            series_names.append(col)
        
        if not series_list:
            raise ValueError("No valid time series could be loaded")
        
        logger.info(f"Successfully loaded {len(series_list)} time series")
        
        return series_list, series_names, df

    @staticmethod
    def validate_timeseries_file(file_path: Path) -> bool:
        try:
            suffix = file_path.suffix.lower()
            if suffix in [".csv", ".tsv"]:
                df = pd.read_csv(file_path, nrows=5)
                return len(df) > 0 and len(df.columns) >= 1
            elif suffix == ".parquet":
                df = pd.read_parquet(file_path)
                return len(df) > 0 and len(df.columns) >= 1
            return False
        except Exception:
            return False