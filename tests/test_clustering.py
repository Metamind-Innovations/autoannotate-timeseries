import pytest
import numpy as np
from autoannotate.core.clustering import ClusteringEngine


class TestClusteringEngine:

    def test_kmeans_initialization(self):
        clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)

        assert clusterer.method == "kmeans"
        assert clusterer.n_clusters == 3
        assert clusterer.reduce_dims is False

    def test_kmeans_clustering(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)

        clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)

        labels = clusterer.fit_predict(embeddings)

        assert len(labels) == 50
        assert len(np.unique(labels)) == 3
        assert labels.min() >= 0
        assert labels.max() < 3

    def test_kmeans_requires_n_clusters(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)

        clusterer = ClusteringEngine(method="kmeans", n_clusters=None)

        with pytest.raises(ValueError, match="n_clusters must be specified"):
            clusterer.fit_predict(embeddings)

    def test_cluster_stats(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)

        clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)
        labels = clusterer.fit_predict(embeddings)
        stats = clusterer.get_cluster_stats(labels)

        assert stats["n_clusters"] == 3
        assert stats["total_samples"] == 50
        assert stats["n_noise"] == 0
        assert len(stats["cluster_sizes"]) == 3
        assert sum(stats["cluster_sizes"].values()) == 50

    def test_representative_indices(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)

        clusterer = ClusteringEngine(method="kmeans", n_clusters=3, reduce_dims=False)
        labels = clusterer.fit_predict(embeddings)
        representatives = clusterer.get_representative_indices(embeddings, labels, n_samples=5)

        assert len(representatives) == 3
        for cluster_id, indices in representatives.items():
            assert len(indices) <= 5
            assert all(labels[idx] == cluster_id for idx in indices)

    def test_dimensionality_reduction_pca(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 400)

        clusterer = ClusteringEngine(
            method="kmeans", n_clusters=3, reduce_dims=True, target_dims=50
        )

        labels = clusterer.fit_predict(embeddings)

        assert clusterer.dim_reducer is not None
        assert len(labels) == 50

    def test_dimensionality_reduction_umap(self):
        np.random.seed(42)
        embeddings = np.random.randn(100, 128)

        clusterer = ClusteringEngine(
            method="kmeans", n_clusters=3, reduce_dims=True, target_dims=50
        )

        labels = clusterer.fit_predict(embeddings)

        assert clusterer.dim_reducer is not None
        assert len(labels) == 100

    def test_dbscan_clustering(self):
        np.random.seed(42)
        embeddings = np.random.randn(50, 128)

        clusterer = ClusteringEngine(method="dbscan", reduce_dims=False)
        labels = clusterer.fit_predict(embeddings)

        assert len(labels) == 50
        assert -1 in labels or len(np.unique(labels)) > 0

    def test_spectral_clustering(self):
        np.random.seed(42)
        embeddings = np.random.randn(100, 128)

        clusterer = ClusteringEngine(method="spectral", n_clusters=3, reduce_dims=False)
        labels = clusterer.fit_predict(embeddings)

        assert len(labels) == 100
        assert len(np.unique(labels)) == 3

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            clusterer = ClusteringEngine(method="invalid_method", n_clusters=3, reduce_dims=False)
            embeddings = np.random.randn(50, 128)
            clusterer.fit_predict(embeddings)
