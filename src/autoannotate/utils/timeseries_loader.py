from pathlib import Path
from typing import List, Tuple, Optional
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

    def _load_dataframe(self) -> pd.DataFrame:
        suffix = self.input_file.suffix.lower()
        if suffix in [".csv", ".tsv"]:
            sep = "\t" if suffix == ".tsv" else ","
            return pd.read_csv(self.input_file, sep=sep)
        elif suffix == ".parquet":
            return pd.read_parquet(self.input_file)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _detect_timeseries_columns(self, df: pd.DataFrame) -> List[str]:
        if self.timestamp_column:
            if self.timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")
            return list([col for col in df.columns if col != self.timestamp_column])

        first_col = df.columns[0]
        is_timestamp = pd.api.types.is_datetime64_any_dtype(
            df[first_col]
        ) or pd.api.types.is_string_dtype(df[first_col])
        if is_timestamp:
            logger.info(f"Auto-detected timestamp column: {first_col}")
            self.timestamp_column = first_col
            return list(df.columns[1:])
        return list(df.columns)

    def _process_series_column(self, df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
        series: np.ndarray = df[col].values.astype(np.float32)
        if np.any(np.isnan(series)):
            series = np.nan_to_num(series, nan=np.nanmean(series))
        if len(series) < 10:
            logger.warning(f"Series '{col}' too short ({len(series)} points), skipping")
            return None
        return series

    def load_timeseries(self) -> Tuple[List[np.ndarray], List[str], pd.DataFrame]:
        df = self._load_dataframe()
        timeseries_columns = self._detect_timeseries_columns(df)

        if len(timeseries_columns) == 0:
            raise ValueError("No time series columns found in the file")

        logger.info(f"Found {len(timeseries_columns)} time series columns")

        series_list = []
        series_names = []

        for col in timeseries_columns:
            series = self._process_series_column(df, col)
            if series is not None:
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
