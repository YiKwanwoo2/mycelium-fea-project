# fea_solver.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------
# Parameters
# ---------------------
MAX_STRAIN = 0.05  # element fails at this strain
N_STEPS = 50
DISPLACEMENT_MAX = 0.2  # mm
E_mod = 1e6             # Young’s modulus (example)

# Nonlinear stress-strain tables (example placeholders)
axial_strain = np.array([0.0, 0.01, 0.02, 0.04, 0.05])
axial_stress = np.array([0.0, 2e4, 3.5e4, 4.2e4, 4.25e4])

bending_strain = np.array([0.0, 0.01, 0.02, 0.04, 0.05])
bending_stress = np.array([0.0, 1.5e4, 2.5e4, 3.0e4, 3.05e4])

# ---------------------
# Helper functions
# ---------------------
def interp_stress(strain, strain_table, stress_table):
    if strain <= strain_table[0]:
        return stress_table[0]
    if strain >= strain_table[-1]:
        return stress_table[-1]
    return np.interp(strain, strain_table, stress_table)

# ---------------------
# FEA Solver
# ---------------------
def fea_solver(results_dir):
    nodes = pd.read_csv(os.path.join(results_dir, "nodes.csv"))
    elems = pd.read_csv(os.path.join(results_dir, "elements.csv"))

    coords = nodes[["x", "y", "z"]].values
    active = np.ones(len(elems), dtype=bool)
    stress_record = []

    # Find top and bottom boundary nodes
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    tol = 0.1
    top_nodes = nodes.loc[np.abs(nodes["y"] - y_max) < tol, "node_id"].values
    bot_nodes = nodes.loc[np.abs(nodes["y"] - y_min) < tol, "node_id"].values

    print(f"Top boundary nodes: {len(top_nodes)}, Bottom: {len(bot_nodes)}")

    for step in range(N_STEPS):
        disp_factor = step / (N_STEPS - 1)
        dy_top = +DISPLACEMENT_MAX * disp_factor
        dy_bot = -DISPLACEMENT_MAX * disp_factor

        new_coords = coords.copy()
        new_coords[top_nodes, 1] += dy_top
        new_coords[bot_nodes, 1] += dy_bot

        stress = np.zeros(len(elems))
        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            p1, p2 = new_coords[n1], new_coords[n2]
            L0 = np.linalg.norm(coords[n2] - coords[n1])
            L = np.linalg.norm(p2 - p1)
            strain = (L - L0) / L0

            stress_ax = interp_stress(abs(strain), axial_strain, axial_stress)
            stress_bend = interp_stress(abs(strain), bending_strain, bending_stress)
            stress[i] = stress_ax + 0.2 * stress_bend

            if abs(strain) > MAX_STRAIN:
                active[i] = False  # element breaks

        stress_record.append(stress)

        # Visualization
        plt.figure(figsize=(6, 6))
        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            p1, p2 = new_coords[n1], new_coords[n2]
            color_val = stress[i] / max(1e-6, np.max(stress))
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=plt.cm.plasma(color_val))
        plt.title(f"Step {step} - Active elements: {active.sum()}")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.axis("equal")
        plt.savefig(os.path.join(results_dir, f"fea_step_{step:03d}.png"))
        plt.close()

        if active.sum() == 0:
            print(f"Simulation stopped early at step {step}")
            break

    np.save(os.path.join(results_dir, "stress_record.npy"), stress_record)
    print(f"Saved FEA results to {results_dir}")

# ---------------------
# Run
# ---------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fea_solver.py <results_dir>")
        exit()
    fea_solver(sys.argv[1])
