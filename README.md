# AutoAnnotate-TimeSeries 📊

**State-of-the-art unsupervised auto-annotation SDK for time series classification with GUI**

[![Tests](https://github.com/Metamind-Innovations/autoannotate-timeseries/actions/workflows/tests.yml/badge.svg)](https://github.com/Metamind-Innovations/autoannotate-timeseries/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AutoAnnotate-TimeSeries automatically clusters and organizes unlabeled time series datasets using cutting-edge
foundation models (Chronos, Moirai, Lag-Llama). Features a **graphical user interface** for easy use and **interactive
HTML preview** with Plotly charts for visual cluster inspection.

## ✨ Features

- 🎨 **Graphical User Interface**: Easy file browser and visual controls via `autoannotate-ts`
- 📈 **Interactive Plotly Charts**: View cluster samples in browser before labeling
- 🤖 **SOTA Foundation Models**: Chronos-T5, Moirai, Lag-Llama
- 🔬 **Multiple Clustering**: K-means, HDBSCAN, Spectral, DBSCAN
- 📁 **Smart Organization**: CSV files named after cluster names for easy identification
- 🕐 **Flexible Timestamp Handling**: Auto-detect or specify timestamp column (GUI uses indices, CLI uses names)
- 📂 **Clean Output**: HTML preview files saved in output folder alongside results
- ✂️ **Auto Splits**: Train/val/test dataset splitting
- 💾 **Export**: CSV, JSON formats
- 📊 **Single CSV Input**: All time series in one file
- 🔌 **Python API**: Full programmatic control

## 🚀 Installation

```bash
pip install autoannotate-timeseries
```

### Optional Dependencies

**HDBSCAN Clustering (Optional):**

If you want to use the HDBSCAN clustering method:

```bash
# Option 1: Install with the package
pip install autoannotate-timeseries[hdbscan]

# Option 2: Install separately before running autoannotate
pip install hdbscan
```

**Note:** HDBSCAN is not required for the default K-means, Spectral, or DBSCAN methods. Only install it if you
specifically need HDBSCAN clustering.

**Development Tools:**

```bash
pip install -e .[dev]
```

### After Installation

Two commands are available:

- `autoannotate-ts` - Launch the graphical user interface
- `autoannotate-ts-cli` - Command-line interface for automation

Check installation:

```bash
autoannotate-ts-cli --version
autoannotate-ts-cli --help
```

## 📝 Input Data Format

### Your CSV Structure

**INPUT: One CSV file with multiple time series as columns**

```csv
timestamp,series_1,series_2,series_3,series_4,series_5
2024-01-01 00:00:00,10.5,20.1,15.3,18.2,22.5
2024-01-01 01:00:00,11.2,19.8,14.9,17.8,23.1
2024-01-01 02:00:00,9.8,21.2,15.7,18.5,21.8
2024-01-01 03:00:00,10.1,19.5,16.1,18.0,22.2
...
```

**Key Points:**

- First column can be timestamp (auto-detected or specify explicitly)
- Each column = one time series to be clustered
- Column names are preserved as series identifiers
- Variable length series supported
- Missing values automatically handled

**Timestamp Column Handling:**

- **Auto-detect** (recommended): Leave empty in GUI or omit `--timestamp-column` in CLI
- **GUI**: Use column index (0 = first column, 1 = second column, etc.)
- **CLI**: Use column name (e.g., `--timestamp-column "timestamp"`)

Specify timestamp column:

```bash
autoannotate-ts-cli annotate data.csv output --timestamp-column "datetime" --n-clusters 5
```

### Output Structure

```
output/
├── increasing_trend/
│   └── increasing_trend.csv    # Contains series_1, series_4 (all rows)
├── decreasing_trend/
│   └── decreasing_trend.csv    # Contains series_2 (all rows)
├── seasonal/
│   └── seasonal.csv            # Contains series_3, series_5 (all rows)
├── unclustered/
│   └── unclustered.csv         # Outliers/noise
├── splits/                     # Available with a CLI parameter
│   ├── train/
│   │   ├── increasing_trend/
│   │   │   └── increasing_trend.csv
│   │   └── ...
│   ├── val/
│   └── test/
├── cluster_0_preview.html      # HTML preview files (saved in output folder)
├── cluster_1_preview.html
├── cluster_2_preview.html
├── metadata.json
└── labels.csv
```

**Key Points:**

- Each class folder contains ONE CSV file **named after the class**
- CSV file includes timestamp column and all time series belonging to that class
- HTML preview files are saved in the output folder for reference

## 🎨 Quick Start - GUI

The easiest way to use AutoAnnotate-TimeSeries:

```bash
autoannotate-ts
```

**Workflow:**

1. 📁 Select input CSV file (with multiple time series as columns)
2. 📂 Select output folder
3. 🔢 Set number of classes
4. 🤖 Choose model
5. 📏 Configure context length (512 for typical series, 1024+ for long series)
6. 📊 **[Optional]** Specify timestamp column index (e.g., 0 for first column, leave empty for auto-detect)
7. ▶️ Click "Start Auto-Annotation"

The app will:

- Cluster your time series automatically
- Open **interactive HTML previews** in your browser with Plotly charts for each cluster
- Save all preview files in the output folder (not project root)
- Prompt you to label each cluster interactively

## 💻 CLI Usage

### Basic Command

```bash
autoannotate-ts-cli annotate /path/to/data.csv /path/to/output \
    --n-clusters 5 \
    --model chronos-t5-tiny \
    --create-splits
```

### Advanced CLI Options

```bash
autoannotate-ts-cli annotate ./data/sensors.csv ./output \
    --n-clusters 8 \
    --method hdbscan \
    --model chronos-2 \
    --batch-size 16 \
    --context-length 512 \
    --timestamp-column "datetime" \
    --create-splits \
    --export-format json
```

**Available models:** `chronos-t5-tiny`, `chronos-t5-small`, `chronos-2`

**Note:** CLI uses column **names** for timestamp (e.g., `--timestamp-column "timestamp"`), while GUI uses column *
*indices** (e.g., 0 for first column).

## 🐍 Python API

```python
from autoannotate import AutoAnnotator
from pathlib import Path

annotator = AutoAnnotator(
    input_file=Path("./data/timeseries.csv"),
    output_dir=Path("./output"),
    model="chronos-t5-tiny",
    clustering_method="kmeans",
    n_clusters=5,
    context_length=512,
    timestamp_column="timestamp"  # Optional
)

result = annotator.run_full_pipeline(
    n_samples=7,
    create_splits=True,
    export_format="csv"
)

print(f"Processed {result['n_timeseries']} time series")
print(f"Created {result['n_clusters']} classes")
```

### Manual Pipeline Control

```python
annotator.load_timeseries()
annotator.extract_embeddings()
annotator.cluster()

stats = annotator.get_cluster_stats()
print(f"Found {stats['n_clusters']} clusters")

class_names = {
    0: "increasing_trend",
    1: "decreasing_trend",
    2: "seasonal_pattern",
    3: "stationary"
}

annotator.organize_dataset(class_names)
annotator.export_labels(format="json")
```

## 📊 Example: Real-World Sensor Data

**Input CSV** (`sensors.csv`):

```csv
timestamp,temp_A,temp_B,temp_C,humidity_A,humidity_B
2024-01-01 00:00,22.5,23.1,21.8,65.2,64.8
2024-01-01 01:00,22.8,23.0,21.9,65.5,64.9
2024-01-01 02:00,23.1,22.9,22.1,65.8,65.1
...
```

**Command:**

```bash
autoannotate-ts-cli annotate sensors.csv ./organized \
    --n-clusters 3 \
    --timestamp-column "timestamp"
```

**Output:**

```
organized/
├── stable_temperature/
│   └── stable_temperature.csv        # Contains: timestamp, temp_A, temp_C
├── variable_temperature/
│   └── variable_temperature.csv      # Contains: timestamp, temp_B
├── high_humidity/
│   └── high_humidity.csv             # Contains: timestamp, humidity_A, humidity_B
├── cluster_0_preview.html
├── cluster_1_preview.html
├── cluster_2_preview.html
├── metadata.json
└── labels.csv
```

## 🧠 Model Comparison

| Model            | Context   | Speed | Quality | Best For                              |
|------------------|-----------|-------|---------|---------------------------------------|
| chronos-t5-tiny  | 512       | ⚡⚡⚡   | ⭐⭐⭐     | Fast inference, small datasets        |
| chronos-t5-small | 512       | ⚡⚡    | ⭐⭐⭐⭐    | Balanced (recommended)                |
| chronos-2        | up to 8192| ⚡      | ⭐⭐⭐⭐⭐   | Best quality, long series (v2 model)  |

**Important Notes:**
- **chronos-2** is a completely new architecture (uses `Chronos2Pipeline`) with support for much longer time series (up to 8192 tokens vs 512)
- **chronos-2** requires `chronos-forecasting>=2.0.0`
- For most use cases, `chronos-t5-small` offers the best balance of speed and quality

## 🔬 Clustering Methods

| Method   | Auto K | Handles Noise | Best For                 | Installation                            |
|----------|--------|---------------|--------------------------|-----------------------------------------|
| kmeans   | ❌      | ❌             | Fast, spherical clusters | ✅ Included                              |
| hdbscan  | ✅      | ✅             | Complex shapes, outliers | ⚠️ Optional: `pip install ...[hdbscan]` |
| spectral | ❌      | ❌             | Non-convex shapes        | ✅ Included                              |
| dbscan   | ✅      | ✅             | Density-based            | ✅ Included                              |

**Note:** HDBSCAN requires separate installation. See [Optional Dependencies](#optional-dependencies) section.

## ✅ Quick Validation

Test if your CSV file is valid:

```bash
autoannotate-ts-cli validate ./your_data.csv
```

This shows:

- Number of time series columns found
- Column names
- Auto-detected timestamp column (if present)

With explicit timestamp column:

```bash
autoannotate-ts-cli validate ./your_data.csv --timestamp-column "timestamp"
```

## 🔍 Pre-Push Checklist

Before pushing code:

```bash
# Format code with Black
black src/autoannotate tests

# Run tests
pytest tests/ -v
```

## 🐛 Troubleshooting

### Out of Memory?

```python
annotator = AutoAnnotator(
    input_file=Path("./data.csv"),
    output_dir=Path("./output"),
    batch_size=8,
    context_length=256,
    model="chronos-t5-tiny"
)
```

### Too Many/Few Clusters?

Try HDBSCAN for automatic cluster detection:

```bash
autoannotate-ts-cli annotate data.csv output --method hdbscan
```

**Note:** HDBSCAN must be installed first:

```bash
pip install autoannotate-timeseries[hdbscan]
```

If you try to use HDBSCAN without installing it, you'll get an error:
`ImportError: HDBSCAN is not installed. Install it with: pip install autoannotate-timeseries[hdbscan]`

### Need to specify timestamp column?

**CLI (uses column name):**

```bash
autoannotate-ts-cli annotate data.csv output --timestamp-column "datetime" --n-clusters 5
```

**GUI (uses column index):**

- Enter `0` for first column, `1` for second column, etc.
- Leave empty to auto-detect

## 🔄 Data Preparation Tips

### If you have separate CSV files per time series:

**Merge them first:**

```python
import pandas as pd
from pathlib import Path

dfs = []
for csv_file in Path("./separate_files").glob("*.csv"):
    df = pd.read_csv(csv_file)
    series_name = csv_file.stem
    df_renamed = df.rename(columns={"value": series_name})
    dfs.append(df_renamed)

merged_df = pd.concat(dfs, axis=1)
merged_df.to_csv("combined_timeseries.csv", index=False)
```

### If you have wide format with row-based time series:

**Transpose it:**

```python
import pandas as pd

df = pd.read_csv("wide_format.csv")
df_transposed = df.T
df_transposed.to_csv("column_format.csv")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. **Format with Black**: `black src/autoannotate tests`
4. **Run tests**: `pytest tests/ -v`
5. Push and create PR

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

Built with PyTorch, Transformers, scikit-learn, Plotly. Foundation models: Chronos-T5 (Amazon), Moirai (Salesforce),
Lag-Llama.

**Made for the [RAIDO Project](https://raido-project.eu/), from [MetaMind Innovations](https://metamind.gr/)**

---

**Sister Project**: [AutoAnnotate-Vision](https://github.com/Metamind-Innovations/autoannotate-vision) - For image
classification