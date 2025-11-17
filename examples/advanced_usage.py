from pathlib import Path
from autoannotate import AutoAnnotator


def main():
    
    annotator = AutoAnnotator(
        input_file=Path("./data/sensor_data.csv"),
        output_dir=Path("./data/organized"),
        model="chronos-t5-small",
        clustering_method="hdbscan",
        batch_size=16,
        context_length=1024,
        timestamp_column="datetime"
    )
    
    series_list, series_names = annotator.load_timeseries()
    print(f"Loaded {len(series_list)} time series: {series_names}")
    
    embeddings = annotator.extract_embeddings()
    print(f"Embedding shape: {embeddings.shape}")
    
    labels = annotator.cluster()
    stats = annotator.get_cluster_stats()
    print(f"Found {stats['n_clusters']} clusters")
    
    representatives = annotator.get_representative_indices(n_samples=10)
    
    class_names = {
        0: "high_temperature",
        1: "low_temperature",
        2: "fluctuating",
        3: "stable"
    }
    
    annotator.organize_dataset(class_names)
    annotator.export_labels(format="json")
    annotator.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print(f"✓ Dataset organized in {annotator.output_dir}")
    print(f"\nEach class folder contains one CSV with all time series of that class")


if __name__ == "__main__":
    main()