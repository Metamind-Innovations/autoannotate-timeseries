from pathlib import Path
from autoannotate import AutoAnnotator


def main():
    
    input_csv_file = Path("./data/timeseries_data.csv")
    output_directory = Path("./data/annotated_timeseries")
    
    annotator = AutoAnnotator(
        input_file=input_csv_file,
        output_dir=output_directory,
        model="chronos-t5-tiny",
        clustering_method="kmeans",
        n_clusters=5,
        batch_size=32,
        reduce_dims=True,
        context_length=512,
        timestamp_column="timestamp"
    )
    
    result = annotator.run_full_pipeline(
        n_samples=5,
        create_splits=True,
        export_format="csv"
    )
    
    print(f"\n{'='*60}")
    print(f"Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total time series processed: {result['n_timeseries']}")
    print(f"Number of classes: {result['n_clusters']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()