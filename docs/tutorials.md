# Tutorials

## Tutorial 1: Basic Usage

### Step 1: Prepare Your Data

Create a CSV file with time series as columns:
```csv
timestamp,sensor_1,sensor_2,sensor_3,sensor_4
2024-01-01 00:00:00,10.5,20.1,15.3,18.2
2024-01-01 01:00:00,11.2,19.8,14.9,17.8
2024-01-01 02:00:00,9.8,21.2,15.7,18.5
...
```

### Step 2: Run Auto-Annotation
```python
from autoannotate import AutoAnnotator
from pathlib import Path

annotator = AutoAnnotator(
    input_file=Path("./data/sensors.csv"),
    output_dir=Path("./output"),
    model="chronos-t5-tiny",
    n_clusters=3,
    timestamp_column="timestamp"
)

result = annotator.run_full_pipeline()
```

### Step 3: Check Output
```
output/
├── stable_pattern/
│   └── timeseries.csv      # Contains sensor_1, sensor_3
├── increasing_pattern/
│   └── timeseries.csv      # Contains sensor_2
├── volatile_pattern/
│   └── timeseries.csv      # Contains sensor_4
└── metadata.json
```

## Tutorial 2: Advanced Customization

### Manual Pipeline Control
```python
from autoannotate import AutoAnnotator
from pathlib import Path

annotator = AutoAnnotator(
    input_file=Path("./data.csv"),
    output_dir=Path("./output"),
    model="chronos-t5-small",
    clustering_method="hdbscan"
)

series_list, series_names = annotator.load_timeseries()
print(f"Loaded columns: {series_names}")

embeddings = annotator.extract_embeddings()
labels = annotator.cluster()

class_names = {
    0: "trend_up",
    1: "trend_down",
    2: "seasonal"
}

annotator.organize_dataset(class_names)
annotator.export_labels(format="json")
```

## Tutorial 3: Working Without Timestamps

If your CSV doesn't have timestamps:
```csv
series_1,series_2,series_3
10.5,20.1,15.3
11.2,19.8,14.9
9.8,21.2,15.7
...
```

Simply don't specify `timestamp_column`:
```python
annotator = AutoAnnotator(
    input_file=Path("./data.csv"),
    output_dir=Path("./output"),
    n_clusters=3
)
```

## Tutorial 4: CLI Workflow
```bash
autoannotate-ts validate ./data/sensors.csv

autoannotate-ts annotate ./data/sensors.csv ./output \
    --n-clusters 5 \
    --model chronos-t5-tiny \
    --timestamp-column "datetime" \
    --create-splits \
    --export-format csv

cd output
cat metadata.json
```