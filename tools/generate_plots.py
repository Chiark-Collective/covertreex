#!/usr/bin/env python3
import sys
import csv
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try importing pandas, if not, fall back to manual or fail
try:
    import pandas as pd
except ImportError:
    print("Pandas not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

OUTPUT_DIR = Path("docs/plots")
DATA_DIR = Path("artifacts/scaling_data")

def get_latest_csv() -> Path:
    files = sorted(DATA_DIR.glob("scaling_results_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No scaling results found in artifacts/scaling_data")
    return files[0]

def generate_plots(csv_path: Path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter for K=10 for main scaling plots
    df_k10 = df[df["k"] == 10].copy()
    
    # Implementations map to nice names
    impl_map = {
        "PCCT": "PCCT (Ours)",
        "scipy_ckdtree": "Scipy cKDTree",
        "sklearn_balltree": "Sklearn BallTree"
    }
    df_k10["Implementation"] = df_k10["impl"].map(impl_map).fillna(df_k10["impl"])
    
    # Colors
    color_map = {
        "PCCT (Ours)": "#636EFA", # Blue
        "Scipy cKDTree": "#EF553B", # Red
        "Sklearn BallTree": "#00CC96" # Green
    }

    # 1. Throughput vs N (Faceted by Dimension)
    fig_qps = px.line(
        df_k10, 
        x="n", 
        y="qps", 
        color="Implementation", 
        facet_col="d", 
        log_y=True,
        markers=True,
        color_discrete_map=color_map,
        title="Query Throughput vs Dataset Size (k=10)",
        labels={"n": "Dataset Size (N)", "qps": "Queries Per Second (Log Scale)", "d": "Dimension"}
    )
    fig_qps.update_layout(template="plotly_white")
    fig_qps.write_html(OUTPUT_DIR / "scaling_throughput.html")
    print("Generated scaling_throughput.html")

    # 2. Build Time vs N (Faceted by Dimension)
    fig_build = px.line(
        df_k10, 
        x="n", 
        y="build_time_s", 
        color="Implementation", 
        facet_col="d", 
        log_y=True,
        markers=True,
        color_discrete_map=color_map,
        title="Build Time vs Dataset Size",
        labels={"n": "Dataset Size (N)", "build_time_s": "Build Time (s) - Log Scale", "d": "Dimension"}
    )
    fig_build.update_layout(template="plotly_white")
    fig_build.write_html(OUTPUT_DIR / "scaling_build_time.html")
    print("Generated scaling_build_time.html")

    # 3. RSS Delta vs N (Faceted by Dimension)
    fig_rss = px.line(
        df_k10, 
        x="n", 
        y="rss_mb", 
        color="Implementation", 
        facet_col="d", 
        markers=True,
        color_discrete_map=color_map,
        title="Memory Usage (RSS Delta) vs Dataset Size (Query Phase)",
        labels={"n": "Dataset Size (N)", "rss_mb": "RSS Delta (MB)", "d": "Dimension"}
    )
    fig_rss.update_layout(template="plotly_white")
    fig_rss.write_html(OUTPUT_DIR / "scaling_memory.html")
    print("Generated scaling_memory.html")
    
    # 4. CPU Time (Latency) vs N
    # Latency is per-query latency in ms.
    fig_lat = px.line(
        df_k10, 
        x="n", 
        y="latency_ms", 
        color="Implementation", 
        facet_col="d", 
        log_y=True,
        markers=True,
        color_discrete_map=color_map,
        title="Query Latency vs Dataset Size (k=10)",
        labels={"n": "Dataset Size (N)", "latency_ms": "Latency (ms) - Log Scale", "d": "Dimension"}
    )
    fig_lat.update_layout(template="plotly_white")
    fig_lat.write_html(OUTPUT_DIR / "scaling_latency.html")
    print("Generated scaling_latency.html")

    # 5. Combined Summary Plot (D=64 only)
    # Subplots: QPS, Latency, Build Time
    df_high_d = df_k10[df_k10["d"] == 64]
    
    # We can't easily use px with make_subplots for different types, but we can try.
    # Or just stick to individual plots. 
    # Let's stick to the HTML files generated above.

if __name__ == "__main__":
    csv_path = get_latest_csv()
    generate_plots(csv_path)
