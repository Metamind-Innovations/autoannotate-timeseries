# API Reference

## AutoAnnotator

Main class for time series auto-annotation.

### Constructor
```python
AutoAnnotator(
    input_file: Path,              # Path to input CSV file
    output_dir: Path,              # Path to output directory
    model: str = "chronos-t5-tiny",
    clustering_method: str = "kmeans",
    n_clusters: int = None,
    batch_size: int = 32,
    context_length: int = 512,
    timestamp_column: str = None   # Optional: name of timestamp column
)
```

### Methods

#### `load_timeseries()`
Load time series from input CSV file.

**Returns**: `Tuple[List[np.ndarray], List[str]]`
- List of time series arrays
- List of column names (series identifiers)

#### `extract_embeddings()`
Extract embeddings using foundation model.

**Returns**: `np.ndarray` - Embedding matrix

#### `cluster()`
Perform clustering on embeddings.

**Returns**: `np.ndarray` - Cluster labels

#### `organize_dataset(class_names: Dict[int, str])`
Organize clustered time series into folders.

**Parameters**:
- `class_names`: Mapping from cluster ID to class name

**Output**: One CSV per class containing all time series

#### `run_full_pipeline()`
Execute complete annotation workflow.

**Returns**: `Dict` with results summary

## TimeSeriesLoader

### Constructor
```python
TimeSeriesLoader(
    input_file: Path,
    timestamp_column: Optional[str] = None
)
```

**Parameters**:
- `input_file`: Path to CSV/TSV file
- `timestamp_column`: Name of timestamp column (auto-detected if None)

### Methods

#### `load_timeseries()`
Load all time series from file.

**Returns**: `Tuple[List[np.ndarray], List[str], pd.DataFrame]`
- List of time series arrays
- List of series names
- Original DataFrame

## EmbeddingExtractor

### Constructor
```python
EmbeddingExtractor(
    model_name: str = "chronos-t5-tiny",
    device: Optional[str] = None,
    batch_size: int = 32,
    context_length: int = 512
)
```

**Available Models**:
- `chronos-t5-tiny`
- `chronos-t5-small`

## DatasetOrganizer

### Constructor
```python
DatasetOrganizer(output_dir: Path)
```

### Methods

#### `organize_by_clusters()`
Organize time series into class folders.

**Parameters**:
- `original_df`: Original DataFrame with all data
- `series_names`: List of series column names
- `labels`: Cluster assignments
- `class_names`: Class name mapping
- `timestamp_column`: Optional timestamp column to include

**Output Structure**:
```
output/
├── class_1/
│   └── timeseries.csv  # timestamp + all series from this class
├── class_2/
│   └── timeseries.csv
...
```