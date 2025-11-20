import pytest
import json
import numpy as np
import pandas as pd

from autoannotate.core.organizer import DatasetOrganizer


@pytest.fixture
def temp_df(tmp_path):
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "series_1": np.random.randn(100),
        "series_2": np.random.randn(100) + 1,
        "series_3": np.random.randn(100) - 1,
        "series_4": np.sin(np.linspace(0, 4 * np.pi, 100)),
        "series_5": np.cos(np.linspace(0, 4 * np.pi, 100)),
    }
    return pd.DataFrame(data)


@pytest.fixture
def output_dir(tmp_path):
    out_dir = tmp_path / "output"
    return out_dir


@pytest.fixture
def sample_labels():
    return np.array([0, 0, 1, 1, 2])


@pytest.fixture
def sample_series_names():
    return ["series_1", "series_2", "series_3", "series_4", "series_5"]


@pytest.fixture
def class_names():
    return {0: "trend_up", 1: "trend_down", 2: "seasonal"}


class TestDatasetOrganizer:
    def test_initialization(self, output_dir):
        organizer = DatasetOrganizer(output_dir)
        assert organizer.output_dir == output_dir
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_organize_by_clusters_basic(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        assert "metadata" in metadata
        assert "classes" in metadata
        assert metadata["metadata"]["total_timeseries"] == 5
        assert metadata["metadata"]["n_classes"] == 3

        assert (output_dir / "trend_up").exists()
        assert (output_dir / "trend_down").exists()
        assert (output_dir / "seasonal").exists()

        trend_up_csv = output_dir / "trend_up" / "timeseries.csv"
        assert trend_up_csv.exists()
        df_trend_up = pd.read_csv(trend_up_csv)
        assert "timestamp" in df_trend_up.columns
        assert "series_1" in df_trend_up.columns
        assert "series_2" in df_trend_up.columns

    def test_organize_with_noise(self, temp_df, output_dir, class_names):
        labels = np.array([0, 0, 1, 1, -1])
        series_names = ["series_1", "series_2", "series_3", "series_4", "series_5"]

        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, series_names, labels, class_names, timestamp_column="timestamp"
        )

        assert (output_dir / "unclustered").exists()
        unclustered_csv = output_dir / "unclustered" / "timeseries.csv"
        assert unclustered_csv.exists()

        df_unclustered = pd.read_csv(unclustered_csv)
        assert "series_5" in df_unclustered.columns

    def test_organize_without_timestamp(self, output_dir, class_names):
        df = pd.DataFrame({"series_1": np.random.randn(50), "series_2": np.random.randn(50) + 1})
        labels = np.array([0, 1])
        series_names = ["series_1", "series_2"]

        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(df, series_names, labels, class_names, timestamp_column=None)

        trend_up_csv = output_dir / "trend_up" / "timeseries.csv"
        df_trend_up = pd.read_csv(trend_up_csv)
        assert "timestamp" not in df_trend_up.columns
        assert "series_1" in df_trend_up.columns

    def test_metadata_json_created(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "metadata" in metadata
        assert "classes" in metadata
        assert "created_at" in metadata["metadata"]

    def test_class_metadata_content(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        assert "trend_up" in metadata["classes"]
        assert "trend_down" in metadata["classes"]
        assert "seasonal" in metadata["classes"]

        assert metadata["classes"]["trend_up"]["count"] == 2
        assert metadata["classes"]["trend_down"]["count"] == 2
        assert metadata["classes"]["seasonal"]["count"] == 1

        assert "timeseries" in metadata["classes"]["trend_up"]
        assert len(metadata["classes"]["trend_up"]["timeseries"]) == 2

    def test_create_split_invalid_ratios(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            organizer.create_split(train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)

    def test_create_split_valid(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        organizer.create_split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        assert (output_dir / "splits" / "train").exists()
        assert (output_dir / "splits" / "val").exists()
        assert (output_dir / "splits" / "test").exists()
        assert (output_dir / "splits" / "split_info.json").exists()

        with open(output_dir / "splits" / "split_info.json") as f:
            split_metadata = json.load(f)

        assert "train_count" in split_metadata
        assert "val_count" in split_metadata
        assert "test_count" in split_metadata

    def test_export_labels_csv(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        output_path = organizer.export_labels_file(format="csv")

        assert output_path.exists()
        assert output_path.name == "labels.csv"

        with open(output_path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "series_name,class_name"
        assert len(lines) > 1

    def test_export_labels_json(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        output_path = organizer.export_labels_file(format="json")

        assert output_path.exists()
        assert output_path.name == "labels.json"

        with open(output_path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) > 0
        assert "series_name" in data[0]
        assert "class_name" in data[0]

    def test_export_labels_invalid_format(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            organizer.export_labels_file(format="xml")

    def test_export_labels_without_organization(self, output_dir):
        organizer = DatasetOrganizer(output_dir)

        with pytest.raises(FileNotFoundError, match="metadata.json not found"):
            organizer.export_labels_file(format="csv")

    def test_empty_class_handling(self, temp_df, output_dir):
        labels = np.array([0, 0, 0, 0, 0])
        series_names = ["series_1", "series_2", "series_3", "series_4", "series_5"]
        class_names = {0: "all_series"}

        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(
            temp_df, series_names, labels, class_names, timestamp_column="timestamp"
        )

        assert len(metadata["classes"]) == 1
        assert metadata["classes"]["all_series"]["count"] == 5

    def test_split_preserves_timestamp(
        self, temp_df, output_dir, sample_labels, sample_series_names, class_names
    ):
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(
            temp_df, sample_series_names, sample_labels, class_names, timestamp_column="timestamp"
        )

        organizer.create_split()

        train_trend_up = output_dir / "splits" / "train" / "trend_up" / "timeseries.csv"
        if train_trend_up.exists():
            df_train = pd.read_csv(train_trend_up)
            assert "timestamp" in df_train.columns
