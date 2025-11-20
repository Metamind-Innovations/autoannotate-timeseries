# AutoAnnotate-TimeSeries 📊

**State-of-the-art unsupervised auto-annotation SDK for time series classification with GUI**

[![Tests](https://github.com/Metamind-Innovations/autoannotate-timeseries/actions/workflows/tests.yml/badge.svg)](https://github.com/Metamind-Innovations/autoannotate-timeseries/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AutoAnnotate-TimeSeries automatically clusters and organizes unlabeled time series datasets using cutting-edge foundation models (Chronos, Moirai, Lag-Llama). Features a **graphical user interface** for easy use and **interactive HTML preview** with Plotly charts for visual cluster inspection.

## ✨ Features

- 🎨 **Graphical User Interface**: Easy file browser and visual controls
- 📈 **Interactive Plotly Charts**: View cluster samples in browser before labeling
- 🤖 **SOTA Foundation Models**: Chronos-T5, Moirai, Lag-Llama
- 🔬 **Multiple Clustering**: K-means, HDBSCAN, Spectral, DBSCAN
- 📁 **Smart Organization**: One CSV per class containing all time series
- ✂️ **Auto Splits**: Train/val/test dataset splitting
- 💾 **Export**: CSV, JSON formats
- 📊 **Single CSV Input**: All time series in one file
- 🔌 **Python API**: Full programmatic control

## 🚀 Installation

### Quick Install (Recommended)
```bash
# Clone repository
git clone https://github.com/Metamind-Innovations/autoannotate-timeseries.git
cd autoannotate-timeseries

# Install core package
pip install -e .

# Install Chronos model (required)
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

### Alternative: Using requirements.txt
```bash
git clone https://github.com/Metamind-Innovations/autoannotate-timeseries.git
cd autoannotate-timeseries
pip install -r requirements.txt
pip install -e .
```

### Optional Dependencies
```bash
# For HDBSCAN clustering
pip install hdbscan

# For development tools
pip install -e .[dev]
```

## ⚠️ Important Notes

- **Chronos model** must be installed separately from GitHub
- Requires Git installed on your system
- First-time installation may take 3-5 minutes
- Requires ~500MB disk space for model weights

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
- ✅ First column can be timestamp (auto-detected or specify with `--timestamp-column`)
- ✅ Each column = one time series to be clustered
- ✅ Column names are preserved as series identifiers
- ✅ Variable length series supported
- ✅ Missing values automatically handled

### Alternative Formats

**Without timestamp:**
```csv
series_1,series_2,series_3
10.5,20.1,15.3
11.2,19.8,14.9
9.8,21.2,15.7
...
```

**With explicit timestamp column:**
```csv
datetime,temperature_sensor_1,temperature_sensor_2,humidity
2024-01-01,22.5,23.1,65.2
2024-01-02,21.8,22.9,64.8
...
```

Specify timestamp column:
```bash
autoannotate-ts annotate data.csv output --timestamp-column "datetime" --n-clusters 5
```

### Output Structure
```
output/
├── increasing_trend/
│   └── timeseries.csv          # Contains series_1, series_4 (all rows)
├── decreasing_trend/
│   └── timeseries.csv          # Contains series_2 (all rows)
├── seasonal/
│   └── timeseries.csv          # Contains series_3, series_5 (all rows)
├── unclustered/
│   └── timeseries.csv          # Outliers/noise
├── splits/
│   ├── train/
│   │   ├── increasing_trend/
│   │   │   └── timeseries.csv
│   │   └── ...
│   ├── val/
│   └── test/
├── metadata.json
└── labels.csv
```

**Each class folder contains ONE CSV file with:**
- Timestamp column (if present in input)
- All time series belonging to that class

## 🎨 Quick Start - GUI

The easiest way to use AutoAnnotate-TimeSeries:
```bash
python run_autoannotate_gui.py
```

**Workflow:**
1. 📁 Select input CSV file (with multiple time series as columns)
2. 📂 Select output folder  
3. 🔢 Set number of classes
4. 🤖 Choose model (chronos-t5-tiny recommended for speed)
5. ⚡ Configure batch size and context length
6. 📊 Optionally specify timestamp column name
7. ▶️ Click "Start Auto-Annotation"

The app will cluster time series and open **interactive HTML previews** in your browser showing Plotly charts from each cluster for easy labeling!

## 💻 CLI Usage
```bash
autoannotate-ts annotate /path/to/data.csv /path/to/output \
    --n-clusters 5 \
    --model chronos-t5-tiny \
    --create-splits
```

### Advanced CLI Options
```bash
autoannotate-ts annotate ./data/sensors.csv ./output \
    --n-clusters 8 \
    --method hdbscan \
    --model chronos-t5-small \
    --batch-size 16 \
    --context-length 512 \
    --timestamp-column "datetime" \
    --create-splits \
    --export-format json
```

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
autoannotate-ts annotate sensors.csv ./organized \
    --n-clusters 3 \
    --timestamp-column "timestamp"
```

**Output:**
```
organized/
├── stable_temperature/
│   └── timeseries.csv        # Contains: timestamp, temp_A, temp_C
├── variable_temperature/
│   └── timeseries.csv        # Contains: timestamp, temp_B
├── high_humidity/
│   └── timeseries.csv        # Contains: timestamp, humidity_A, humidity_B
└── metadata.json
```

## 🧠 Model Comparison

| Model | Context | Speed | Quality | Best For |
|-------|---------|-------|---------|----------|
| chronos-t5-tiny | 512 | ⚡⚡⚡ | ⭐⭐⭐ | Fast inference, small datasets |
| chronos-t5-small | 512 | ⚡⚡ | ⭐⭐⭐⭐ | Balanced (recommended) |

## 🔬 Clustering Methods

| Method | Auto K | Handles Noise | Best For |
|--------|--------|---------------|----------|
| kmeans | ❌ | ❌ | Fast, spherical clusters |
| hdbscan | ✅ | ✅ | Complex shapes, outliers |
| spectral | ❌ | ❌ | Non-convex shapes |
| dbscan | ✅ | ✅ | Density-based |

## 📋 Supported Input Formats

- **CSV** (`.csv`): Comma-separated values
- **TSV** (`.tsv`): Tab-separated values  
- **Parquet** (`.parquet`): Apache Parquet format

All formats follow the same structure: one file with multiple time series as columns.

## ✅ Quick Validation

Test if your CSV file is valid:
```bash
autoannotate-ts validate ./your_data.csv
```

This shows:
- ✓ Number of time series columns found
- ✓ Column names
- ✓ Auto-detected timestamp column

## 🔍 Pre-Push Checklist

Before pushing code:
```bash
black src/autoannotate tests

pytest tests/ -v

black --check src/autoannotate tests && pytest tests/ -v
```

## 🧪 Testing
```bash
pytest tests/ -v --cov=autoannotate
```

## 📚 Documentation

Full documentation available at: [autoannotate-timeseries.readthedocs.io](https://autoannotate-timeseries.readthedocs.io)

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
```bash
autoannotate-ts annotate data.csv output --method hdbscan
```

### Need to specify timestamp column?
```bash
autoannotate-ts annotate data.csv output --timestamp-column "datetime" --n-clusters 5
```

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

Built with PyTorch, Transformers, scikit-learn, Plotly. Foundation models: Chronos-T5 (Amazon), Moirai (Salesforce), Lag-Llama.

**Made for the [RAIDO Project](https://raido-project.eu/)**

---

**Sister Project**: [AutoAnnotate-Vision](https://github.com/Metamind-Innovations/autoannotate-vision) - For image classification