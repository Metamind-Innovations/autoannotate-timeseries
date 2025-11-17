import click
from pathlib import Path
import logging
from rich.console import Console
from rich.logging import RichHandler

from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.core.organizer import DatasetOrganizer
from autoannotate.ui.interactive import InteractiveLabelingSession
from autoannotate.utils.timeseries_loader import TimeSeriesLoader

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--n-clusters", "-n", type=int, help="Number of clusters (required for kmeans/spectral)"
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["kmeans", "hdbscan", "spectral", "dbscan"]),
    default="kmeans",
    help="Clustering method",
)
@click.option(
    "--model",
    type=click.Choice(["chronos-t5-tiny", "chronos-t5-small"]),
    default="chronos-t5-tiny",
    help="Embedding model",
)
@click.option(
    "--batch-size", "-b", type=int, default=32, help="Batch size for embedding extraction"
)
@click.option("--reduce-dims/--no-reduce-dims", default=True, help="Apply dimensionality reduction")
@click.option(
    "--n-samples", type=int, default=5, help="Number of representative samples per cluster"
)
@click.option("--create-splits", is_flag=True, help="Create train/val/test splits")
@click.option(
    "--export-format",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Export labels format",
)
@click.option(
    "--timestamp-column", type=str, default=None, help="Name of timestamp column (auto-detected if not specified)"
)
@click.option(
    "--context-length", type=int, default=512, help="Context length for time series models"
)
def annotate(
    input_file: Path,
    output_dir: Path,
    n_clusters: int,
    method: str,
    model: str,
    batch_size: int,
    reduce_dims: bool,
    n_samples: int,
    create_splits: bool,
    export_format: str,
    timestamp_column: str,
    context_length: int,
):
    try:
        console.print("[bold blue]AutoAnnotate-TimeSeries[/bold blue] - SOTA Time Series Auto-Annotation\n")

        if method in ["kmeans", "spectral"] and n_clusters is None:
            raise click.UsageError(f"--n-clusters is required for {method} clustering")

        console.print(f"[cyan]Loading time series from:[/cyan] {input_file}")
        loader = TimeSeriesLoader(input_file, timestamp_column=timestamp_column)
        series_list, series_names, original_df = loader.load_timeseries()
        console.print(f"[green]✓[/green] Loaded {len(series_list)} time series columns\n")

        console.print(f"[cyan]Extracting embeddings using {model}...[/cyan]")
        extractor = EmbeddingExtractor(
            model_name=model, 
            batch_size=batch_size,
            context_length=context_length
        )
        embeddings = extractor(series_list)
        console.print(f"[green]✓[/green] Extracted embeddings: {embeddings.shape}\n")

        console.print(f"[cyan]Clustering with {method}...[/cyan]")
        clusterer = ClusteringEngine(method=method, n_clusters=n_clusters, reduce_dims=reduce_dims)
        labels = clusterer.fit_predict(embeddings)
        stats = clusterer.get_cluster_stats(labels)
        console.print(f"[green]✓[/green] Clustering complete\n")

        session = InteractiveLabelingSession()
        session.display_cluster_stats(stats)

        console.print("\n[cyan]Getting representative samples...[/cyan]")
        representatives = clusterer.get_representative_indices(
            embeddings, labels, n_samples=n_samples
        )
        console.print(
            f"[green]✓[/green] Found representatives for {len(representatives)} clusters\n"
        )

        class_names = session.label_all_clusters_by_names(
            series_list, series_names, labels, representatives, stats
        )

        session.display_labeling_summary(class_names, labels)

        if not class_names:
            console.print("[yellow]No clusters were labeled. Exiting.[/yellow]")
            return

        console.print("\n[cyan]Organizing dataset...[/cyan]")
        organizer = DatasetOrganizer(output_dir)
        metadata = organizer.organize_by_clusters(
            original_df, series_names, labels, class_names, timestamp_column=loader.timestamp_column
        )
        console.print(f"[green]✓[/green] Dataset organized in {output_dir}\n")

        console.print(f"[cyan]Exporting labels to {export_format}...[/cyan]")
        labels_file = organizer.export_labels_file(format=export_format)
        console.print(f"[green]✓[/green] Labels exported to {labels_file}\n")

        if create_splits:
            console.print("[cyan]Creating train/val/test splits...[/cyan]")
            split_info = organizer.create_split()
            console.print(f"[green]✓[/green] Created splits in {output_dir / 'splits'}\n")

        session.show_completion_message(output_dir)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Full traceback:")
        raise click.Abort()


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--timestamp-column", type=str, default=None, help="Name of timestamp column")
def validate(input_file: Path, timestamp_column: str):
    console.print("[bold blue]Validating time series file...[/bold blue]\n")

    if TimeSeriesLoader.validate_timeseries_file(input_file):
        console.print(f"[green]✓ Valid file:[/green] {input_file}")
        
        try:
            loader = TimeSeriesLoader(input_file, timestamp_column=timestamp_column)
            series_list, series_names, df = loader.load_timeseries()
            console.print(f"[green]✓ Found {len(series_names)} time series columns[/green]")
            console.print(f"[cyan]Column names:[/cyan] {', '.join(series_names)}")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] {e}")
    else:
        console.print(f"[red]✗ Invalid file:[/red] {input_file}")


def main():
    cli()


if __name__ == "__main__":
    main()