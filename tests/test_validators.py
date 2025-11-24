import pytest
import numpy as np
import pandas as pd

from autoannotate.utils.validators import (
    ValidationError,
    validate_directory,
    validate_file,
    validate_timeseries_file,
    validate_model_name,
    validate_clustering_method,
    validate_n_clusters,
    validate_batch_size,
    validate_embeddings,
    validate_labels,
    validate_class_names,
    validate_split_ratios,
    validate_representative_samples,
    validate_export_format,
    validate_timeseries_list,
    validate_device,
    validate_context_length,
)


class TestValidateDirectory:
    def test_valid_existing_directory(self, tmp_path):
        result = validate_directory(tmp_path, must_exist=True)
        assert result == tmp_path
        assert result.is_dir()

    def test_nonexistent_directory_must_exist(self):
        with pytest.raises(ValidationError, match="Directory does not exist"):
            validate_directory("/nonexistent/path", must_exist=True)

    def test_file_not_directory(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValidationError, match="Path is not a directory"):
            validate_directory(file_path, must_exist=True)

    def test_nonempty_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("test")

        with pytest.raises(ValidationError, match="Directory is not empty"):
            validate_directory(tmp_path, must_exist=True, must_be_empty=True)

    def test_empty_directory(self, tmp_path):
        result = validate_directory(tmp_path, must_exist=True, must_be_empty=True)
        assert result == tmp_path


class TestValidateFile:
    def test_valid_csv_file(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"col1": [1, 2, 3]})
        df.to_csv(csv_path, index=False)

        result = validate_file(csv_path, must_exist=True)
        assert result == csv_path

    def test_nonexistent_file(self):
        with pytest.raises(ValidationError, match="File does not exist"):
            validate_file("/nonexistent/file.csv", must_exist=True)

    def test_unsupported_format(self, tmp_path):
        file_path = tmp_path / "data.xml"
        file_path.write_text("data")

        with pytest.raises(ValidationError, match="Unsupported file format"):
            validate_file(file_path)


class TestValidateTimeSeriesFile:
    def test_valid_csv_file(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({"value": np.random.randn(100)})
        df.to_csv(csv_path, index=False)

        assert validate_timeseries_file(csv_path) is True

    def test_nonexistent_file(self):
        assert validate_timeseries_file("/nonexistent/file.csv") is False

    def test_directory_not_file(self, tmp_path):
        assert validate_timeseries_file(tmp_path) is False

    def test_unsupported_format(self, tmp_path):
        file_path = tmp_path / "data.txt"
        file_path.write_text("data")

        assert validate_timeseries_file(file_path) is False


class TestValidateModelName:
    def test_valid_chronos_tiny(self):
        assert validate_model_name("chronos-t5-tiny") == "chronos-t5-tiny"

    def test_valid_chronos_small(self):
        assert validate_model_name("chronos-t5-small") == "chronos-t5-small"

    def test_invalid_model(self):
        with pytest.raises(ValidationError, match="Invalid model name"):
            validate_model_name("invalid_model")


class TestValidateClusteringMethod:
    def test_valid_kmeans(self):
        assert validate_clustering_method("kmeans") == "kmeans"

    def test_valid_hdbscan(self):
        assert validate_clustering_method("hdbscan") == "hdbscan"

    def test_valid_spectral(self):
        assert validate_clustering_method("spectral") == "spectral"

    def test_valid_dbscan(self):
        assert validate_clustering_method("dbscan") == "dbscan"

    def test_invalid_method(self):
        with pytest.raises(ValidationError, match="Invalid clustering method"):
            validate_clustering_method("invalid_method")


class TestValidateNClusters:
    def test_valid_n_clusters(self):
        result = validate_n_clusters(5, "kmeans", 100)
        assert result == 5

    def test_none_for_hdbscan(self):
        result = validate_n_clusters(None, "hdbscan", 100)
        assert result is None

    def test_none_for_kmeans_raises(self):
        with pytest.raises(ValidationError, match="requires n_clusters"):
            validate_n_clusters(None, "kmeans", 100)

    def test_n_clusters_less_than_2(self):
        with pytest.raises(ValidationError, match="n_clusters must be at least 2"):
            validate_n_clusters(1, "kmeans", 100)

    def test_n_clusters_exceeds_samples(self):
        with pytest.raises(ValidationError, match="cannot exceed number of samples"):
            validate_n_clusters(150, "kmeans", 100)

    def test_n_clusters_more_than_half_samples_warns(self):
        with pytest.warns(UserWarning, match="more than half the number of samples"):
            validate_n_clusters(60, "kmeans", 100)


class TestValidateBatchSize:
    def test_valid_batch_size(self):
        result = validate_batch_size(32, 100)
        assert result == 32

    def test_batch_size_less_than_1(self):
        with pytest.raises(ValidationError, match="batch_size must be at least 1"):
            validate_batch_size(0, 100)

    def test_batch_size_larger_than_samples_warns(self):
        with pytest.warns(UserWarning, match="larger than number of samples"):
            result = validate_batch_size(150, 100)
            assert result == 100


class TestValidateEmbeddings:
    def test_valid_embeddings(self):
        embeddings = np.random.randn(10, 128)
        result = validate_embeddings(embeddings)
        assert np.array_equal(result, embeddings)

    def test_not_numpy_array(self):
        with pytest.raises(ValidationError, match="must be numpy array"):
            validate_embeddings([[1, 2, 3]])

    def test_wrong_dimensions(self):
        embeddings = np.random.randn(10)
        with pytest.raises(ValidationError, match="must be 2D array"):
            validate_embeddings(embeddings)

    def test_too_few_samples(self):
        embeddings = np.random.randn(1, 128)
        with pytest.raises(ValidationError, match="Need at least 2 samples"):
            validate_embeddings(embeddings)

    def test_non_finite_values(self):
        embeddings = np.random.randn(10, 128)
        embeddings[0, 0] = np.nan
        with pytest.raises(ValidationError, match="non-finite values"):
            validate_embeddings(embeddings)


class TestValidateLabels:
    def test_valid_labels(self):
        labels = np.array([0, 1, 0, 1, 2])
        result = validate_labels(labels, 5)
        assert np.array_equal(result, labels)

    def test_not_numpy_array(self):
        with pytest.raises(ValidationError, match="must be numpy array"):
            validate_labels([0, 1, 2], 3)

    def test_wrong_dimensions(self):
        labels = np.array([[0, 1], [2, 3]])
        with pytest.raises(ValidationError, match="must be 1D array"):
            validate_labels(labels, 4)

    def test_length_mismatch(self):
        labels = np.array([0, 1, 2])
        with pytest.raises(ValidationError, match="must match number of samples"):
            validate_labels(labels, 5)


class TestValidateClassNames:
    def test_valid_class_names(self):
        class_names = {0: "trend_up", 1: "trend_down", 2: "seasonal"}
        labels = np.array([0, 1, 0, 2, 1])
        result = validate_class_names(class_names, labels)
        assert result == class_names

    def test_not_dict(self):
        with pytest.raises(ValidationError, match="must be a dictionary"):
            validate_class_names(["trend_up", "trend_down"], np.array([0, 1]))

    def test_empty_dict(self):
        with pytest.raises(ValidationError, match="dictionary is empty"):
            validate_class_names({}, np.array([0, 1]))

    def test_non_integer_label(self):
        class_names = {"0": "trend_up"}
        with pytest.raises(ValidationError, match="must be integer"):
            validate_class_names(class_names, np.array([0]))

    def test_non_string_name(self):
        class_names = {0: 123}
        with pytest.raises(ValidationError, match="must be string"):
            validate_class_names(class_names, np.array([0]))

    def test_empty_class_name(self):
        class_names = {0: "  "}
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_class_names(class_names, np.array([0]))

    def test_label_not_in_results_warns(self):
        class_names = {0: "trend_up", 5: "unknown"}
        labels = np.array([0, 1, 2])
        with pytest.warns(UserWarning, match="not found in clustering results"):
            validate_class_names(class_names, labels)

    def test_invalid_characters_in_name(self):
        class_names = {0: "trend/up"}
        with pytest.raises(ValidationError, match="invalid characters"):
            validate_class_names(class_names, np.array([0]))


class TestValidateSplitRatios:
    def test_valid_ratios(self):
        result = validate_split_ratios(0.7, 0.15, 0.15)
        assert result == (0.7, 0.15, 0.15)

    def test_ratios_sum_not_1(self):
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            validate_split_ratios(0.6, 0.2, 0.1)

    def test_negative_ratio(self):
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_split_ratios(-0.1, 0.6, 0.5)


class TestValidateRepresentativeSamples:
    def test_valid_n_samples(self):
        result = validate_representative_samples(5, 10)
        assert result == 5

    def test_n_samples_less_than_1(self):
        with pytest.raises(ValidationError, match="must be at least 1"):
            validate_representative_samples(0, 10)

    def test_n_samples_exceeds_cluster_size_warns(self):
        with pytest.warns(UserWarning, match="exceeds cluster size"):
            result = validate_representative_samples(15, 10)
            assert result == 10


class TestValidateExportFormat:
    def test_valid_csv(self):
        assert validate_export_format("csv") == "csv"

    def test_valid_json(self):
        assert validate_export_format("json") == "json"

    def test_case_insensitive(self):
        assert validate_export_format("CSV") == "csv"
        assert validate_export_format("JSON") == "json"

    def test_invalid_format(self):
        with pytest.raises(ValidationError, match="Invalid export format"):
            validate_export_format("xml")


class TestValidateTimeSeriesList:
    def test_valid_timeseries_list(self):
        series_list = [np.random.randn(100) for _ in range(5)]
        result = validate_timeseries_list(series_list)
        assert result == series_list

    def test_empty_list(self):
        with pytest.raises(ValidationError, match="Time series list is empty"):
            validate_timeseries_list([])

    def test_too_few_series(self):
        series_list = [np.random.randn(100)]
        with pytest.raises(ValidationError, match="Need at least 2 time series"):
            validate_timeseries_list(series_list, min_series=2)

    def test_non_array_object(self):
        series_list = [np.random.randn(100), [1, 2, 3]]
        with pytest.raises(ValidationError, match="not a numpy array"):
            validate_timeseries_list(series_list)

    def test_wrong_dimensions(self):
        series_list = [np.random.randn(100), np.random.randn(10, 10)]
        with pytest.raises(ValidationError, match="must be 1D array"):
            validate_timeseries_list(series_list)


class TestValidateDevice:
    def test_valid_cpu(self):
        assert validate_device("cpu") == "cpu"

    def test_valid_auto(self):
        assert validate_device("auto") == "auto"

    def test_invalid_device(self):
        with pytest.raises(ValidationError, match="Invalid device"):
            validate_device("invalid")

    @pytest.mark.skipif(
        not hasattr(__import__("torch"), "cuda") or not __import__("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    def test_valid_cuda(self):
        assert validate_device("cuda") == "cuda"

    def test_cuda_not_available(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(ValidationError, match="CUDA is not available"):
            validate_device("cuda")


class TestValidateContextLength:
    def test_valid_context_length(self):
        result = validate_context_length(512, 1000)
        assert result == 512

    def test_context_length_too_small(self):
        with pytest.raises(ValidationError, match="context_length must be at least 10"):
            validate_context_length(5, 100)

    def test_context_length_very_large_warns(self):
        with pytest.warns(UserWarning, match="very large"):
            validate_context_length(5000, 10000)
