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
        np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1,
        np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1,
        np.random.randn(100),
    ]
    return [s.astype(np.float32) for s in series_list]


class TestTimeSeriesLoader:
    
    def test_load_csv_files(self, temp_dir):
        for i in range(5):
            series = np.random.randn(100)
            df = pd.DataFrame({"value": series})
            df.to_csv(temp_dir / f"series_{i}.csv", index=False)
        
        loader = TimeSeriesLoader(temp_dir, recursive=False)
        series_list, paths = loader.load_timeseries()
        
        assert len(series_list) == 5
        assert len(paths) == 5
        assert all(isinstance(s, np.ndarray) for s in series_list)
    
    def test_load_npy_files(self, temp_dir):
        for i in range(3):
            series = np.random.randn(100).astype(np.float32)
            np.save(temp_dir / f"series_{i}.npy", series)
        
        loader = TimeSeriesLoader(temp_dir)
        series_list, paths = loader.load_timeseries()
        
        assert len(series_list) == 3
        assert all(s.dtype == np.float32 for s in series_list)
    
    def test_column_selection(self, temp_dir):
        df = pd.DataFrame({
            "timestamp": range(100),
            "temperature": np.random.randn(100),
            "humidity": np.random.randn(100)
        })
        df.to_csv(temp_dir / "data.csv", index=False)
        
        loader = TimeSeriesLoader(temp_dir, column_name="temperature")
        series_list, paths = loader.load_timeseries()
        
        assert len(series_list) == 1
        assert len(series_list[0]) == 100
    
    def test_validate_timeseries(self, temp_dir):
        series = np.random.randn(100)
        df = pd.DataFrame({"value": series})
        csv_path = temp_dir / "valid.csv"
        df.to_csv(csv_path, index=False)
        
        assert TimeSeriesLoader.validate_timeseries(csv_path) is True
    
    def test_invalid_directory(self):
        with pytest.raises(FileNotFoundError):
            loader = TimeSeriesLoader(Path("/nonexistent/path"))
    
    def test_empty_directory(self, temp_dir):
        with pytest.raises(ValueError, match="No valid time series files found"):
            loader = TimeSeriesLoader(temp_dir)
            loader.load_timeseries()


class TestDatasetOrganizer:
    
    def test_organize_by_clusters(self, temp_dir):
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        
        ts_paths = []
        for i in range(10):
            series = np.random.randn(50)
            df = pd.DataFrame({"value": series})
            path = input_dir / f"series_{i}.csv"
            df.to_csv(path, index=False)
            ts_paths.append(path)
        
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, -1])
        class_names = {0: "class_a", 1: "class_b", 2: "class_c"}
        
        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(
            ts_paths, labels, class_names, copy_files=True
        )
        
        assert (output_dir / "class_a").exists()
        assert (output_dir / "class_b").exists()
        assert (output_dir / "class_c").exists()
        assert (output_dir / "unclustered").exists()
        assert (output_dir / "metadata.json").exists()
        
        assert len(list((output_dir / "class_a").glob("*"))) == 3
        assert len(list((output_dir / "class_b").glob("*"))) == 3
        assert len(list((output_dir / "class_c").glob("*"))) == 3
        assert len(list((output_dir / "unclustered").glob("*"))) == 1
    
    def test_create_splits(self, temp_dir):
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        
        ts_paths = []
        for i in range(20):
            series = np.random.randn(50)
            df = pd.DataFrame({"value": series})
            path = input_dir / f"series_{i}.csv"
            df.to_csv(path, index=False)
            ts_paths.append(path)
        
        labels = np.array([0]*10 + [1]*10)
        class_names = {0: "class_a", 1: "class_b"}
        
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(ts_paths, labels, class_names, copy_files=True)
        
        split_info = organizer.create_split(
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        assert (output_dir / "splits" / "train").exists()
        assert (output_dir / "splits" / "val").exists()
        assert (output_dir / "splits" / "test").exists()
        assert (output_dir / "splits" / "split_info.json").exists()
        
        total_files = len(split_info["train"]) + len(split_info["val"]) + len(split_info["test"])
        assert total_files == 20
    
    def test_export_labels_csv(self, temp_dir):
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        
        ts_paths = []
        for i in range(5):
            series = np.random.randn(50)
            df = pd.DataFrame({"value": series})
            path = input_dir / f"series_{i}.csv"
            df.to_csv(path, index=False)
            ts_paths.append(path)
        
        labels = np.array([0, 0, 1, 1, 2])
        class_names = {0: "class_a", 1: "class_b", 2: "class_c"}
        
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(ts_paths, labels, class_names, copy_files=True)
        
        labels_file = organizer.export_labels_file(format="csv")
        assert labels_file.exists()
        
        df = pd.read_csv(labels_file)
        assert "class_name" in df.columns
        assert "cluster_id" in df.columns
        assert len(df) == 5
    
    def test_export_labels_json(self, temp_dir):
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        
        ts_paths = []
        for i in range(5):
            series = np.random.randn(50)
            df = pd.DataFrame({"value": series})
            path = input_dir / f"series_{i}.csv"
            df.to_csv(path, index=False)
            ts_paths.append(path)
        
        labels = np.array([0, 0, 1, 1, 2])
        class_names = {0: "class_a", 1: "class_b", 2: "class_c"}
        
        organizer = DatasetOrganizer(output_dir)
        organizer.organize_by_clusters(ts_paths, labels, class_names, copy_files=True)
        
        labels_file = organizer.export_labels_file(format="json")
        assert labels_file.exists()
        
        import json
        with open(labels_file, "r") as f:
            labels_data = json.load(f)
        
        assert len(labels_data) == 5
        assert "class_name" in labels_data[0]
        assert "cluster_id" in labels_data[0]


class TestAutoAnnotator:
    
    def test_full_pipeline_manual(self, temp_dir, sample_timeseries):
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        output_dir = temp_dir / "output"
        
        for i, series in enumerate(sample_timeseries):
            df = pd.DataFrame({"value": series})
            df.to_csv(input_dir / f"series_{i}.csv", index=False)
        
        annotator = AutoAnnotator(
            input_dir=input_dir,
            output_dir=output_dir,
            model="chronos-t5-tiny",
            clustering_method="kmeans",
            n_clusters=3,
            batch_size=2,
            context_length=64
        )
        
        series_list, paths = annotator.load_timeseries()
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
            annotator = AutoAnnotator(
                input_dir=Path("/nonexistent"),
                output_dir=Path("/tmp/output"),
                n_clusters=3
            )