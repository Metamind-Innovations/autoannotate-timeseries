import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from autoannotate import AutoAnnotator
from autoannotate.core.organizer import DatasetOrganizer
from autoannotate.utils.timeseries_loader import TimeSeriesLoader


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_timeseries():
    np.random.seed(42)
    series_list = [
        np.random.randn(100) + np.linspace(0, 1, 100),
        np.random.randn(100) + np.linspace(1, 0, 100),
        np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1,
        np.cos(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.1,
        np.random.randn(100),
    ]
    return [s.astype(np.float32) for s in series_list]


class TestTimeSeriesLoader:

    def test_load_csv_file_single_column(self, temp_dir):
        series = np.random.randn(100)
        df = pd.DataFrame({"value": series})
        csv_file = temp_dir / "data.csv"
        df.to_csv(csv_file, index=False)

        loader = TimeSeriesLoader(csv_file)
        series_list, series_names, _original_df = loader.load_timeseries()

        assert len(series_list) == 1
        assert len(series_names) == 1
        assert series_names[0] == "value"
        assert isinstance(series_list[0], np.ndarray)

    def test_load_csv_file_multiple_columns(self, temp_dir):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
                "temperature": np.random.randn(100),
                "humidity": np.random.randn(100),
            }
        )
        csv_file = temp_dir / "data.csv"
        df.to_csv(csv_file, index=False)

        loader = TimeSeriesLoader(csv_file)
        series_list, series_names, _original_df = loader.load_timeseries()

        assert len(series_list) == 2
        assert "temperature" in series_names
        assert "humidity" in series_names
        assert len(series_list[0]) == 100

    def test_validate_timeseries_file(self, temp_dir):
        series = np.random.randn(100)
        df = pd.DataFrame({"value": series})
        csv_path = temp_dir / "valid.csv"
        df.to_csv(csv_path, index=False)

        assert TimeSeriesLoader.validate_timeseries_file(csv_path) is True

    def test_invalid_file(self):
        with pytest.raises(FileNotFoundError):
            TimeSeriesLoader(Path("/nonexistent/path/file.csv"))

    def test_directory_instead_of_file(self, temp_dir):
        with pytest.raises(ValueError, match="Path must be a file, not a directory"):
            TimeSeriesLoader(temp_dir)


class TestDatasetOrganizer:

    def test_organize_by_clusters(self, temp_dir):
        output_dir = temp_dir / "output"

        df = pd.DataFrame({f"series_{i}": np.random.randn(50) for i in range(10)})

        series_names = [f"series_{i}" for i in range(10)]
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, -1])
        class_names = {0: "class_a", 1: "class_b", 2: "class_c"}

        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(df, series_names, labels, class_names)

        assert (output_dir / "class_a").exists()
        assert (output_dir / "class_b").exists()
        assert (output_dir / "class_c").exists()
        assert (output_dir / "unclustered").exists()
        assert (output_dir / "metadata.json").exists()

        assert (output_dir / "class_a" / "timeseries.csv").exists()
        assert (output_dir / "class_b" / "timeseries.csv").exists()
        assert (output_dir / "class_c" / "timeseries.csv").exists()

    def test_create_splits(self, temp_dir):
        output_dir = temp_dir / "output"

        df = pd.DataFrame({f"series_{i}": np.random.randn(50) for i in range(20)})

        series_names = [f"series_{i}" for i in range(20)]
        labels = np.array([0] * 10 + [1] * 10)
        class_names = {0: "class_a", 1: "class_b"}

        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(df, series_names, labels, class_names)

        split_info = organizer.create_split(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        assert (output_dir / "splits" / "train").exists()
        assert (output_dir / "splits" / "val").exists()
        assert (output_dir / "splits" / "test").exists()
        assert (output_dir / "splits" / "split_info.json").exists()

        all_classes = set()
        for split_dict in [split_info["train"], split_info["val"], split_info["test"]]:
            all_classes.update(split_dict.keys())
        assert len(all_classes) <= 2

    def test_export_labels_csv(self, temp_dir):
        output_dir = temp_dir / "output"

        df = pd.DataFrame({f"series_{i}": np.random.randn(50) for i in range(5)})

        series_names = [f"series_{i}" for i in range(5)]
        labels = np.array([0, 0, 1, 1, 2])
        class_names = {0: "class_a", 1: "class_b", 2: "class_c"}

        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(df, series_names, labels, class_names)

        labels_file = organizer.export_labels_file(format="csv")
        assert labels_file.exists()

        labels_df = pd.read_csv(labels_file)
        assert "class_name" in labels_df.columns
        assert "series_name" in labels_df.columns
        assert len(labels_df) == 5

    def test_export_labels_json(self, temp_dir):
        output_dir = temp_dir / "output"

        df = pd.DataFrame({f"series_{i}": np.random.randn(50) for i in range(5)})

        series_names = [f"series_{i}" for i in range(5)]
        labels = np.array([0, 0, 1, 1, 2])
        class_names = {0: "class_a", 1: "class_b", 2: "class_c"}

        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(df, series_names, labels, class_names)

        labels_file = organizer.export_labels_file(format="json")
        assert labels_file.exists()

        import json

        with open(labels_file, "r") as f:
            labels_data = json.load(f)

        assert len(labels_data) == 5
        assert "class_name" in labels_data[0]
        assert "series_name" in labels_data[0]


class TestAutoAnnotator:

    def test_full_pipeline_manual(self, temp_dir, sample_timeseries):
        output_dir = temp_dir / "output"

        df = pd.DataFrame({f"series_{i}": series for i, series in enumerate(sample_timeseries)})
        input_file = temp_dir / "data.csv"
        df.to_csv(input_file, index=False)

        annotator = AutoAnnotator(
            input_file=input_file,
            output_dir=output_dir,
            model="chronos-t5-tiny",
            clustering_method="kmeans",
            n_clusters=3,
            batch_size=2,
            context_length=64,
            reduce_dims=False,
        )

        series_list, _series_names = annotator.load_timeseries()
        assert len(series_list) == 5

        embeddings = annotator.extract_embeddings()
        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] > 0

        labels = annotator.cluster()
        assert len(labels) == 5
        assert len(np.unique(labels)) <= 3

        stats = annotator.get_cluster_stats()
        assert stats["n_clusters"] <= 3
        assert stats["total_samples"] == 5

        representatives = annotator.get_representative_indices(n_samples=2)
        assert len(representatives) <= 3

    def test_initialization_validation(self):
        with pytest.raises(FileNotFoundError):
            AutoAnnotator(
                input_file=Path("/nonexistent/file.csv"),
                output_dir=Path("/tmp/output"),
                n_clusters=3,
            )
