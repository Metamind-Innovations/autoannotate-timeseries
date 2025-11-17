from pathlib import Path
from typing import List
import webbrowser
import numpy as np
import json


def generate_cluster_preview_html(
    cluster_id: int,
    timeseries_list: List[np.ndarray],
    series_names: List[str],
    indices: np.ndarray,
    cluster_size: int,
    output_path: Path = None,
) -> Path:

    if output_path is None:
        output_path = Path(f"cluster_{cluster_id}_preview.html")

    series_data = []
    for idx in indices:
        series = timeseries_list[idx]
        name = series_names[idx]
        series_data.append({
            "name": name,
            "values": series.tolist()
        })

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Cluster {cluster_id} - Time Series Preview</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 20px;
                margin: 0;
            }}
            .header {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }}
            h1 {{
                margin: 0;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
            }}
            .stats {{
                font-size: 1.2em;
                margin-top: 10px;
                opacity: 0.9;
            }}
            .gallery {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .ts-card {{
                background: rgba(255, 255, 255, 0.15);
                backdrop-filter: blur(10px);
                border-radius: 12px;
                overflow: hidden;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                padding: 15px;
            }}
            .ts-card:hover {{
                transform: translateY(-10px) scale(1.02);
                box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
            }}
            .ts-name {{
                text-align: center;
                font-size: 1em;
                font-weight: bold;
                margin-bottom: 10px;
                background: rgba(0, 0, 0, 0.3);
                padding: 10px;
                border-radius: 8px;
                word-break: break-all;
            }}
            .plot-container {{
                width: 100%;
                height: 300px;
                background: white;
                border-radius: 8px;
            }}
            .instruction {{
                background: rgba(255, 255, 255, 0.1);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #4CAF50;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Cluster {cluster_id}</h1>
            <div class="stats">
                📈 Total Time Series in Cluster: <strong>{cluster_size}</strong><br>
                👁️ Showing Representative Samples: <strong>{len(indices)}</strong>
            </div>
        </div>
        
        <div class="instruction">
            <strong>📝 Instructions:</strong> Review these sample time series and return to the terminal to enter a class name for this cluster.
        </div>
        
        <div class="gallery">
    """

    for idx, series_info in enumerate(series_data, 1):
        html_content += f"""
            <div class="ts-card">
                <div class="ts-name">
                    <strong>Sample {idx}</strong><br>
                    {series_info['name']}
                </div>
                <div class="plot-container" id="plot{idx}"></div>
            </div>
        """

    html_content += """
        </div>
        <script>
    """

    for idx, series_info in enumerate(series_data, 1):
        values_json = json.dumps(series_info['values'])
        html_content += f"""
            {{
                const data = {values_json};
                const trace = {{
                    y: data,
                    type: 'scatter',
                    mode: 'lines',
                    line: {{color: '#667eea', width: 2}},
                    name: '{series_info['name']}'
                }};
                
                const layout = {{
                    margin: {{l: 50, r: 30, t: 30, b: 40}},
                    paper_bgcolor: 'rgba(255,255,255,0.9)',
                    plot_bgcolor: 'rgba(255,255,255,0.9)',
                    xaxis: {{title: 'Time', gridcolor: '#e0e0e0'}},
                    yaxis: {{title: 'Value', gridcolor: '#e0e0e0'}},
                    showlegend: false
                }};
                
                Plotly.newPlot('plot{idx}', [trace], layout, {{responsive: true}});
            }}
        """

    html_content += """
        </script>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def open_html_in_browser(html_path: Path):
    webbrowser.open(html_path.absolute().as_uri())