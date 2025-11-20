import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from autoannotate.utils.timeseries_loader import TimeSeriesLoader


@pytest.fixture
def temp_csv_file(tmp_path):
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "series_1": np.random.randn(100),
        "series_2": np.random.randn(100) + 1,
        "series_3": np.random.randn(100) - 1,
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_csv_no_timestamp(tmp_path):
    data = {"series_1": np.random.randn(100), "series_2": np.random.randn(100) + 1}
    df = pd.DataFrame(data)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_tsv_file(tmp_path):
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="d"),
        "temp": np.random.randn(50) + 20,
        "humidity": np.random.randn(50) + 60,
    }
    df = pd.DataFrame(data)
    tsv_path = tmp_path / "data.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    return tsv_path


class TestTimeSeriesLoader:
    def test_initialization_valid_file(self, temp_csv_file):
        loader = TimeSeriesLoader(temp_csv_file)
        assert loader.input_file == temp_csv_file

    def test_initialization_file_not_exists(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            TimeSeriesLoader(Path("/nonexistent/file.csv"))

    def test_initialization_path_is_directory(self, tmp_path):
        with pytest.raises(ValueError, match="Path must be a file"):
            TimeSeriesLoader(tmp_path)

    def test_load_timeseries_with_timestamp(self, temp_csv_file):
        loader = TimeSeriesLoader(temp_csv_file)
        series_list, series_names, _df = loader.load_timeseries()

        assert len(series_list) == 3
        assert len(series_names) == 3
        assert series_names == ["series_1", "series_2", "series_3"]
        assert all(isinstance(s, np.ndarray) for s in series_list)
        assert all(s.dtype == np.float32 for s in series_list)

    def test_load_timeseries_without_timestamp(self, temp_csv_no_timestamp):
        loader = TimeSeriesLoader(temp_csv_no_timestamp)
        series_list, series_names, _df = loader.load_timeseries()

        assert len(series_list) == 2
        assert len(series_names) == 2
        assert series_names == ["series_1", "series_2"]

    def test_load_timeseries_explicit_timestamp_column(self, temp_csv_file):
        loader = TimeSeriesLoader(temp_csv_file, timestamp_column="timestamp")
        series_list, series_names, _df = loader.load_timeseries()

        assert len(series_list) == 3
        assert "timestamp" not in series_names

    def test_load_tsv_file(self, temp_tsv_file):
        loader = TimeSeriesLoader(temp_tsv_file)
        series_list, series_names, _df = loader.load_timeseries()

        assert len(series_list) == 2
        assert "temp" in series_names
        assert "humidity" in series_names

    def test_auto_detect_timestamp_column(self, temp_csv_file):
        loader = TimeSeriesLoader(temp_csv_file)
        series_list, series_names, _df = loader.load_timeseries()

        assert loader.timestamp_column == "timestamp"
        assert "timestamp" not in series_names

    def test_handle_nan_values(self, tmp_path):
        data = {
            "series_1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            "series_2": [5.0, np.nan, 3.0, 2.0, 1.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "data_with_nan.csv"
        df.to_csv(csv_path, index=False)

        loader = TimeSeriesLoader(csv_path)
        series_list, series_names, _df = loader.load_timeseries()

        assert len(series_list) == 2
        assert all(not np.isnan(s).any() for s in series_list)

    def test_skip_short_series(self, tmp_path):
        csv_content = "series_1,series_2\n"
        for i in range(100):
            if i < 8:
                csv_content += f"{np.random.randn()},{i+1}\n"
            else:
                csv_content += f"{np.random.randn()},\n"

        csv_path = tmp_path / "data_short.csv"
        csv_path.write_text(csv_content)

        loader = TimeSeriesLoader(csv_path)
        series_list, series_names, _df = loader.load_timeseries()

        assert len(series_list) >= 1
        assert "series_1" in series_names

    def test_validate_timeseries_file_valid(self, temp_csv_file):
        assert TimeSeriesLoader.validate_timeseries_file(temp_csv_file) is True

    def test_validate_timeseries_file_invalid(self, tmp_path):
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("not a valid file")

        assert TimeSeriesLoader.validate_timeseries_file(invalid_file) is False

    def test_validate_timeseries_file_nonexistent(self):
        assert TimeSeriesLoader.validate_timeseries_file(Path("/nonexistent.csv")) is False

    def test_empty_file(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")

        loader = TimeSeriesLoader(empty_csv)
        with pytest.raises(Exception):
            loader.load_timeseries()

    def test_single_column_file(self, tmp_path):
        data = {"values": np.random.randn(100)}
        df = pd.DataFrame(data)
        csv_path = tmp_path / "single_column.csv"
        df.to_csv(csv_path, index=False)

        loader = TimeSeriesLoader(csv_path)
        series_list, series_names, _df = loader.load_timeseries()

        assert len(series_list) == 1
        assert series_names == ["values"]

    def test_parquet_file(self, tmp_path):
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "sensor_1": np.random.randn(100),
            "sensor_2": np.random.randn(100),
        }
        df = pd.DataFrame(data)
        parquet_path = tmp_path / "data.parquet"
        df.to_parquet(parquet_path, index=False)

        loader = TimeSeriesLoader(parquet_path)
        series_list, series_names, _df_loaded = loader.load_timeseries()

        assert len(series_list) == 2
        assert "sensor_1" in series_names
        assert "sensor_2" in series_names

    def test_timestamp_column_not_found(self, temp_csv_file):
        with pytest.raises(ValueError, match="Timestamp column"):
            loader = TimeSeriesLoader(temp_csv_file, timestamp_column="nonexistent")
            loader.load_timeseries()

    def test_unsupported_file_format(self, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            TimeSeriesLoader(txt_file)

    def test_series_dtype_conversion(self, temp_csv_file):
        loader = TimeSeriesLoader(temp_csv_file)
        series_list, _series_names, _df = loader.load_timeseries()

        for series in series_list:
            assert series.dtype == np.float32

    def test_multiple_timestamp_formats(self, tmp_path):
        data = {
            "time": [f"2024-01-{i:02d}" for i in range(1, 16)],
            "value": np.random.randn(15),
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "time_data.csv"
        df.to_csv(csv_path, index=False)

        loader = TimeSeriesLoader(csv_path)
        series_list, series_names, _df_loaded = loader.load_timeseries()

        assert len(series_list) == 1
        assert "value" in series_names
