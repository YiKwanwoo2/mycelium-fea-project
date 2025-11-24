# plot_fea_results.py
# Simple plotting wrapper to produce the per-step network images and the force-displacement curve.
# Usage: python plot_fea_results.py results/sim_xxx fea_results

import sys, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_network(nodes_csv, elems_csv, stress_row, active_row, out_png, max_stress=1.0):
    nodes = pd.read_csv(nodes_csv)
    elems = pd.read_csv(elems_csv)
    coords = nodes[['x','y','z']].values
    xy = coords[:, :2]

    segments = []
    colors = []
    for i,row in elems.iterrows():
        if not active_row[i]:
            continue
        n1 = int(row.n1); n2 = int(row.n2)
        segments.append([xy[n1], xy[n2]])
        colors.append(stress_row[i] / max_stress)

    if len(segments) == 0:
        print("No active segments to plot.")
        return

    lc = LineCollection(segments, cmap='plasma', array=np.array(colors), norm=Normalize(vmin=0, vmax=1), linewidths=1.2)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.add_collection(lc)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main(results_dir):
    fea_dir = os.path.join(results_dir, 'fea_results')
    nodes_csv = os.path.join(results_dir, 'nodes.csv')
    elems_csv = os.path.join(results_dir, 'elements.csv')
    stress_df = pd.read_csv(os.path.join(fea_dir, 'stress_record.csv'))
    active_df = pd.read_csv(os.path.join(fea_dir, 'active_elements.csv'))
    fd = pd.read_csv(os.path.join(fea_dir, 'force_displacement.csv'))

    os.makedirs(os.path.join(fea_dir,'pngs'), exist_ok=True)
    nsteps = len(stress_df)
    for i in range(nsteps):
        stress_row = stress_df.iloc[i].values[:-1] if 'step' in stress_df.columns else stress_df.iloc[i].values
        active_row = active_df.iloc[i].values[:-1] if 'step' in active_df.columns else active_df.iloc[i].values
        out_png = os.path.join(fea_dir, 'pngs', f'fea_step_{i:03d}.png')
        plot_network(nodes_csv, elems_csv, stress_row, active_row, out_png, max_stress=1.0)

    # Force-displacement plot
    plt.figure(figsize=(6,4))
    plt.plot(fd['total_displacement'], fd['total_force'], marker='o')
    plt.xlabel('Total displacement (mm)')
    plt.ylabel('Reaction force')
    plt.title('Force-Displacement')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fea_dir, 'force_displacement.png'))
    plt.close()
    print("Plots saved to", os.path.join(fea_dir,'pngs'))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_fea_results.py <results_dir>")
        sys.exit(1)
    main(sys.argv[1])
