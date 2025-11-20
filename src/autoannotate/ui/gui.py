import sys
from pathlib import Path
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    tk = None  # type: ignore[assignment]

from autoannotate.core.embeddings import EmbeddingExtractor
from autoannotate.core.clustering import ClusteringEngine
from autoannotate.core.organizer import DatasetOrganizer
from autoannotate.ui.interactive import InteractiveLabelingSession
from autoannotate.utils.timeseries_loader import TimeSeriesLoader


class AutoAnnotateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AutoAnnotate-TimeSeries - Time Series Clustering & Labeling")
        self.root.geometry("750x550")
        self.root.resizable(False, False)

        self.input_file = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.n_clusters = tk.IntVar(value=5)
        self.model_choice = tk.StringVar(value="chronos-t5-tiny")
        self.batch_size = tk.IntVar(value=32)
        self.context_length = tk.IntVar(value=512)
        self.timestamp_column = tk.StringVar(value="")

        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="AutoAnnotate-TimeSeries",
            font=("Arial", 20, "bold"),
            fg="#667eea",
        )
        title.pack(pady=15)

        frame = tk.Frame(self.root, padx=20, pady=5)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Input CSV File:", font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=8
        )
        tk.Entry(frame, textvariable=self.input_file, width=45, font=("Arial", 10)).grid(
            row=0, column=1, padx=10
        )
        tk.Button(
            frame,
            text="Browse...",
            command=self.browse_input,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 9),
        ).grid(row=0, column=2)

        tk.Label(frame, text="Output Folder:", font=("Arial", 11, "bold")).grid(
            row=1, column=0, sticky="w", pady=8
        )
        tk.Entry(frame, textvariable=self.output_folder, width=45, font=("Arial", 10)).grid(
            row=1, column=1, padx=10
        )
        tk.Button(
            frame,
            text="Browse...",
            command=self.browse_output,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 9),
        ).grid(row=1, column=2)

        tk.Label(frame, text="Number of Classes:", font=("Arial", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=8
        )
        tk.Spinbox(
            frame, from_=2, to=50, textvariable=self.n_clusters, width=10, font=("Arial", 10)
        ).grid(row=2, column=1, sticky="w", padx=10)

        tk.Label(frame, text="Model:", font=("Arial", 11, "bold")).grid(
            row=3, column=0, sticky="w", pady=8
        )
        model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_choice,
            values=["chronos-t5-tiny", "chronos-t5-small"],
            state="readonly",
            width=25,
            font=("Arial", 10),
        )
        model_combo.grid(row=3, column=1, sticky="w", padx=10)

        tk.Label(frame, text="Batch Size:", font=("Arial", 11, "bold")).grid(
            row=4, column=0, sticky="w", pady=8
        )
        tk.Spinbox(
            frame, from_=1, to=64, textvariable=self.batch_size, width=10, font=("Arial", 10)
        ).grid(row=4, column=1, sticky="w", padx=10)

        tk.Label(frame, text="Context Length:", font=("Arial", 11, "bold")).grid(
            row=5, column=0, sticky="w", pady=8
        )
        tk.Spinbox(
            frame,
            from_=128,
            to=2048,
            textvariable=self.context_length,
            width=10,
            font=("Arial", 10),
        ).grid(row=5, column=1, sticky="w", padx=10)

        tk.Label(frame, text="Timestamp Column (optional):", font=("Arial", 11, "bold")).grid(
            row=6, column=0, sticky="w", pady=8
        )
        tk.Entry(frame, textvariable=self.timestamp_column, width=25, font=("Arial", 10)).grid(
            row=6, column=1, sticky="w", padx=10
        )

        self.status_label = tk.Label(
            self.root, text="Ready to start", font=("Arial", 10), fg="blue"
        )
        self.status_label.pack(pady=8)

        self.progress = ttk.Progressbar(self.root, length=600, mode="indeterminate")
        self.progress.pack(pady=8)

        self.run_button = tk.Button(
            self.root,
            text="▶ Start Auto-Annotation",
            command=self.run_annotation,
            bg="#667eea",
            fg="white",
            font=("Arial", 16, "bold"),
            width=25,
            height=2,
            relief="raised",
            bd=3,
            cursor="hand2",
        )
        self.run_button.pack(pady=15, ipady=15)

    def browse_input(self):
        file = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("TSV files", "*.tsv"),
                ("Parquet files", "*.parquet"),
                ("All files", "*.*"),
            ],
        )
        if file:
            self.input_file.set(file)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)

    def update_status(self, message, color="blue"):
        self.status_label.config(text=message, fg=color)
        self.root.update()

    def run_annotation(self):
        input_file = self.input_file.get()
        output_dir = self.output_folder.get()

        if not input_file or not output_dir:
            messagebox.showerror("Error", "Please select both input file and output folder!")
            return

        thread = threading.Thread(target=self.annotation_process, args=(input_file, output_dir))
        thread.daemon = True
        thread.start()

    def annotation_process(self, input_file, output_dir):
        try:
            self.run_button.config(state="disabled")
            self.progress.start()

            self.update_status("Initializing...", "blue")

            ts_col = self.timestamp_column.get() if self.timestamp_column.get() else None

            self.update_status("Loading time series from CSV...", "blue")
            loader = TimeSeriesLoader(Path(input_file), timestamp_column=ts_col)
            series_list, series_names, original_df = loader.load_timeseries()
            self.update_status(f"✓ Loaded {len(series_list)} time series columns", "green")

            if len(series_list) < self.n_clusters.get() * 3:
                n_series = len(series_list)
                n_clusters = self.n_clusters.get()
                recommended = n_clusters * 3
                response = messagebox.askyesno(
                    "Small Dataset Warning",
                    f"You have {n_series} time series but requested {n_clusters} clusters.\n\n"
                    f"Recommended: At least {recommended} time series for good clustering.\n\n"
                    f"Continue anyway?",
                )
                if not response:
                    self.update_status("Cancelled by user", "orange")
                    return

            self.update_status("Extracting embeddings (this may take a while)...", "blue")
            extractor = EmbeddingExtractor(
                model_name=self.model_choice.get(),
                batch_size=self.batch_size.get(),
                context_length=self.context_length.get(),
            )
            embeddings = extractor(series_list)
            self.update_status("✓ Embeddings extracted", "green")

            self.update_status("Clustering time series...", "blue")
            reduce_dims = len(series_list) > 50
            clusterer = ClusteringEngine(
                method="kmeans",
                n_clusters=self.n_clusters.get(),
                reduce_dims=reduce_dims,
            )
            labels = clusterer.fit_predict(embeddings)
            stats = clusterer.get_cluster_stats(labels)
            self.update_status(f"✓ Found {stats['n_clusters']} clusters", "green")

            self.progress.stop()
            self.update_status("Opening HTML preview for labeling...", "orange")

            session = InteractiveLabelingSession()
            session.display_cluster_stats(stats)

            representatives = clusterer.get_representative_indices(embeddings, labels, n_samples=7)

            class_names = session.label_all_clusters_by_names(
                series_list, series_names, labels, representatives, stats
            )

            if class_names:
                self.update_status("Organizing dataset...", "blue")
                self.progress.start()

                organizer = DatasetOrganizer(Path(output_dir))
                organizer.organize_by_clusters(
                    original_df,
                    series_names,
                    labels,
                    class_names,
                    timestamp_column=loader.timestamp_column,
                )
                organizer.export_labels_file(format="csv")

                self.progress.stop()
                self.update_status("✓ Complete!", "green")

                messagebox.showinfo(
                    "Success",
                    f"✓ Annotation Complete!\n\n"
                    f"Processed: {len(series_list)} time series\n"
                    f"Classes: {len(class_names)}\n"
                    f"Output: {output_dir}\n\n"
                    f"Each class folder contains one CSV file with all time series.",
                )
            else:
                self.update_status("No classes labeled", "orange")
                messagebox.showwarning("Warning", "No clusters were labeled.")

        except Exception as e:
            self.progress.stop()
            self.update_status(f"Error: {str(e)}", "red")
            messagebox.showerror("Error", f"An error occurred:\n\n{str(e)}")
            import traceback

            traceback.print_exc()

        finally:
            self.run_button.config(state="normal")
            self.progress.stop()


def main():
    if tk is None:
        print("ERROR: tkinter is not installed!", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "The GUI requires tkinter, which is part of Python's standard library",
            file=sys.stderr,
        )
        print("but must be installed separately on some systems:", file=sys.stderr)
        print(file=sys.stderr)
        print("  Ubuntu/Debian:    sudo apt-get install python3-tk", file=sys.stderr)
        print("  RHEL/CentOS:      sudo yum install python3-tkinter", file=sys.stderr)
        print("  Fedora:           sudo dnf install python3-tkinter", file=sys.stderr)
        print("  Arch Linux:       sudo pacman -S tk", file=sys.stderr)
        print(file=sys.stderr)
        print("On Windows and macOS, tkinter is usually included with Python.", file=sys.stderr)
        print(
            "If it's missing, you may need to reinstall Python with tkinter support.",
            file=sys.stderr,
        )
        sys.exit(1)

    root = tk.Tk()
    _ = AutoAnnotateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
