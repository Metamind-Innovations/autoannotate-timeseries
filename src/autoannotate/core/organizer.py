import json
from pathlib import Path
from typing import Dict, List, Optional, cast
from datetime import datetime
import numpy as np
import pandas as pd


class DatasetOrganizer:

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def organize_by_clusters(
        self,
        original_df: pd.DataFrame,
        series_names: List[str],
        labels: np.ndarray,
        class_names: Dict[int, str],
        timestamp_column: Optional[str] = None,
    ) -> Dict:
        organized_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_timeseries": len(series_names),
                "n_classes": len(class_names),
            },
            "classes": {},
        }

        for cluster_id, class_name in class_names.items():
            cluster_mask = labels == cluster_id
            cluster_series_names = [name for name, mask in zip(series_names, cluster_mask) if mask]

            if not cluster_series_names:
                continue

            class_dir = self.output_dir / class_name
            class_dir.mkdir(exist_ok=True)

            columns_to_save = cluster_series_names.copy()
            if timestamp_column and timestamp_column in original_df.columns:
                columns_to_save.insert(0, timestamp_column)

            cluster_df = original_df[columns_to_save]

            output_file = class_dir / f"{class_name}.csv"
            cluster_df.to_csv(output_file, index=False)

            organized_data["classes"][class_name] = {
                "count": len(cluster_series_names),
                "timeseries": cluster_series_names,
                "file": str(output_file),
            }

        unclustered_mask = labels == -1
        unlabeled_mask = np.array([label not in class_names and label != -1 for label in labels])
        combined_unclustered_mask = unclustered_mask | unlabeled_mask
        unclustered_series = [
            name for name, mask in zip(series_names, combined_unclustered_mask) if mask
        ]

        if unclustered_series:
            noise_dir = self.output_dir / "unclustered"
            noise_dir.mkdir(exist_ok=True)

            columns_to_save = unclustered_series.copy()
            if timestamp_column and timestamp_column in original_df.columns:
                columns_to_save.insert(0, timestamp_column)

            unclustered_df = original_df[columns_to_save]
            output_file = noise_dir / "unclustered.csv"
            unclustered_df.to_csv(output_file, index=False)

            organized_data["unclustered"] = {
                "count": len(unclustered_series),
                "timeseries": unclustered_series,
                "file": str(output_file),
            }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(organized_data, f, indent=2)

        return organized_data

    def create_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Split ratios must sum to 1.0")

        np.random.seed(seed)

        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)

        for split_name in ["train", "val", "test"]:
            (splits_dir / split_name).mkdir(exist_ok=True)

        class_dirs = [
            d
            for d in self.output_dir.iterdir()
            if d.is_dir() and d.name not in ["splits", "unclustered"]
        ]

        split_info: Dict[str, Dict[str, Dict[str, object]]] = {
            "train": {},
            "val": {},
            "test": {},
        }

        for class_dir in class_dirs:
            class_name = class_dir.name
            csv_file = class_dir / f"{class_name}.csv"

            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)

            timeseries_cols = [
                col for col in df.columns if col not in ["timestamp", "time", "date"]
            ]

            if len(timeseries_cols) == 0:
                continue

            n_series = len(timeseries_cols)
            indices = np.arange(n_series)
            np.random.shuffle(indices)

            n_train = int(n_series * train_ratio)
            n_val = int(n_series * val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            timestamp_col = [col for col in df.columns if col in ["timestamp", "time", "date"]]

            for split_name, split_indices in [
                ("train", train_idx),
                ("val", val_idx),
                ("test", test_idx),
            ]:
                if len(split_indices) == 0:
                    continue

                split_class_dir = splits_dir / split_name / class_name
                split_class_dir.mkdir(exist_ok=True, parents=True)

                selected_cols = [timeseries_cols[i] for i in split_indices]
                cols_to_save = timestamp_col + selected_cols if timestamp_col else selected_cols

                split_df = df[cols_to_save]
                output_file = split_class_dir / f"{class_name}.csv"
                split_df.to_csv(output_file, index=False)

                split_info[split_name][class_name] = {
                    "n_series": len(selected_cols),
                    "series_names": selected_cols,
                    "file": str(output_file),
                }

        split_metadata = {
            "train_count": sum(
                cast(int, info["n_series"]) for info in split_info["train"].values()
            ),
            "val_count": sum(cast(int, info["n_series"]) for info in split_info["val"].values()),
            "test_count": sum(cast(int, info["n_series"]) for info in split_info["test"].values()),
            "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "details": split_info,
        }

        split_metadata_path = splits_dir / "split_info.json"
        with open(split_metadata_path, "w") as f:
            json.dump(split_metadata, f, indent=2)

        return split_info

    def export_labels_file(self, format: str = "csv"):
        metadata_path = self.output_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError("metadata.json not found. Run organize_by_clusters first.")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        if format == "csv":
            output_path = self.output_dir / "labels.csv"
            with open(output_path, "w") as f:
                f.write("series_name,class_name\n")
                for class_name, class_data in metadata["classes"].items():
                    for series_name in class_data["timeseries"]:
                        f.write(f"{series_name},{class_name}\n")

        elif format == "json":
            output_path = self.output_dir / "labels.json"
            labels_data = []
            for class_name, class_data in metadata["classes"].items():
                for series_name in class_data["timeseries"]:
                    labels_data.append({"series_name": series_name, "class_name": class_name})

            with open(output_path, "w") as f:
                json.dump(labels_data, f, indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path
