# plot_snapshots.py
# Usage: python plot_snapshots.py results/sim_<timestamp>/snapshots
# Produces PNGs like petri_step_0000.png etc.

import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_snapshot_csv(csv_path, out_png, dish_size=5.0, show=False):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        return
    segments = []
    intensities = []
    vals = (df['intensity'].values)
    max_val = vals.max() if vals.max() > 1e-12 else 1.0
    for _, row in df.iterrows():
        segments.append([[row['x1'], row['y1']], [row['x2'], row['y2']]])
        intensities.append(row['intensity'] / max_val)
    lc = LineCollection(segments, cmap='viridis', array=np.array(intensities), norm=Normalize(vmin=0, vmax=1), linewidths=1.2)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.add_collection(lc)
    # inoculum points (optional; hard-coded 5x5 grid like default)
    # We'll draw nothing here unless user provides points.
    ax.set_aspect('equal')
    ax.set_xlim(-dish_size/2, dish_size/2)
    ax.set_ylim(-dish_size/2, dish_size/2)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    plt.title(os.path.basename(out_png).replace('.png',''))
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_snapshots.py <snapshot_dir>")
        sys.exit(1)
    snap_dir = sys.argv[1]
    out_dir = os.path.join(os.path.dirname(snap_dir), "pngs")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(snap_dir, "step_*.csv")))
    for f in files:
        base = os.path.basename(f).replace('.csv','.png')
        out = os.path.join(out_dir, base)
        print("Plotting", f, "->", out)
        plot_snapshot_csv(f, out)
    print("Done. PNGs in", out_dir)
