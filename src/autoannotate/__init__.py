from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, cast
import numpy as np
import pandas as pd

from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.core.organizer import DatasetOrganizer
from autoannotate.ui.interactive import InteractiveLabelingSession
from autoannotate.utils.timeseries_loader import TimeSeriesLoader


class AutoAnnotator:

    def __init__(
        self,
        input_file: Path,
        output_dir: Path,
        model: str = "chronos-2",
        clustering_method: str = "kmeans",
        n_clusters: Optional[int] = None,
        batch_size: int = 32,
        reduce_dims: bool = True,
        context_length: int = 512,
        timestamp_column: Optional[str] = None,
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.model_name = model
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.reduce_dims = reduce_dims
        self.context_length = context_length
        self.timestamp_column = timestamp_column

        self.timeseries_list: Optional[List[np.ndarray]] = None
        self.series_names: Optional[List[str]] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.cluster_stats: Optional[Dict] = None

        self.loader = TimeSeriesLoader(self.input_file, timestamp_column=self.timestamp_column)
        self.extractor = EmbeddingExtractor(
            model_name=cast(
                Literal["chronos-t5-tiny", "chronos-t5-small", "chronos-2"], self.model_name
            ),
            batch_size=self.batch_size,
            context_length=self.context_length,
        )
        self.clusterer = ClusteringEngine(
            method=cast(Literal["kmeans", "hdbscan", "spectral", "dbscan"], self.clustering_method),
            n_clusters=self.n_clusters,
            reduce_dims=self.reduce_dims,
        )
        self.organizer = DatasetOrganizer(self.output_dir)

    def load_timeseries(self) -> Tuple[List[np.ndarray], List[str]]:
        self.timeseries_list, self.series_names, self.original_df = self.loader.load_timeseries()
        return self.timeseries_list, self.series_names

    def extract_embeddings(self) -> np.ndarray:
        if self.timeseries_list is None:
            raise ValueError("Load time series first using load_timeseries()")

        self.embeddings = self.extractor(self.timeseries_list)
        return self.embeddings

    def cluster(self) -> np.ndarray:
        if self.embeddings is None:
            raise ValueError("Extract embeddings first using extract_embeddings()")

        self.labels = self.clusterer.fit_predict(self.embeddings)
        self.cluster_stats = self.clusterer.get_cluster_stats(self.labels)
        return self.labels

    def get_cluster_stats(self) -> Dict:
        if self.cluster_stats is None:
            raise ValueError("Run clustering first using cluster()")
        return self.cluster_stats

    def get_representative_indices(self, n_samples: int = 5) -> Dict[int, np.ndarray]:
        if self.embeddings is None or self.labels is None:
            raise ValueError("Run clustering first")

        return self.clusterer.get_representative_indices(self.embeddings, self.labels, n_samples)

    def interactive_labeling(self, n_samples: int = 5) -> Dict[int, str]:
        if self.series_names is None or self.labels is None:
            raise ValueError("Load time series and run clustering first")

        assert self.timeseries_list is not None
        assert self.cluster_stats is not None

        session = InteractiveLabelingSession()
        session.display_cluster_stats(self.cluster_stats)

        representatives = self.get_representative_indices(n_samples)
        class_names = session.label_all_clusters_by_names(
            self.timeseries_list,
            self.series_names,
            self.labels,
            representatives,
            self.cluster_stats,
        )

        session.display_labeling_summary(class_names, self.labels)
        return class_names

    def organize_dataset(self, class_names: Dict[int, str]) -> Dict:
        if self.series_names is None or self.labels is None:
            raise ValueError("Load time series and run clustering first")

        return self.organizer.organize_by_clusters(
            self.original_df,
            self.series_names,
            self.labels,
            class_names,
            timestamp_column=self.loader.timestamp_column,
        )

    def create_splits(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
    ):
        return self.organizer.create_split(train_ratio, val_ratio, test_ratio)

    def export_labels(self, format: str = "csv"):
        return self.organizer.export_labels_file(format=format)

    def run_full_pipeline(
        self,
        n_samples: int = 5,
        create_splits: bool = True,
        export_format: str = "csv",
    ) -> Dict:
        self.load_timeseries()
        self.extract_embeddings()
        self.cluster()

        class_names = self.interactive_labeling(n_samples=n_samples)

        if not class_names:
            raise ValueError("No clusters were labeled")

        self.organize_dataset(class_names)
        self.export_labels(format=export_format)

        if create_splits:
            self.create_splits()

        assert self.timeseries_list is not None
        return {
            "n_timeseries": len(self.timeseries_list),
            "n_clusters": len(class_names),
            "output_dir": str(self.output_dir),
            "class_names": class_names,
        }


__all__ = [
    "AutoAnnotator",
    "EmbeddingExtractor",
    "ClusteringEngine",
    "DatasetOrganizer",
    "TimeSeriesLoader",
    "InteractiveLabelingSession",
]
