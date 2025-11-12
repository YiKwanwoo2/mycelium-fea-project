import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------
# Material & Simulation Parameters
# ----------------------------
E_mod = 1e6
A = 1.0
I = 0.05
N_STEPS = 40
DISPLACEMENT_MAX = 0.2  # mm

# Nonlinear stress–strain curves
axial_strain = np.array([0.0, 0.01, 0.02, 0.04, 0.05, 5.00])
axial_stress = np.array([0.0, 2e4, 3.5e4, 4.2e4, 4.25e4, 4.25e4])

bending_strain = np.array([0.0, 0.01, 0.02, 0.04, 0.05, 5.00])
bending_stress = np.array([0.0, 1.5e4, 2.5e4, 3.0e4, 3.05e4, 3.05e4])

# Automatically set failure strain
MAX_STRAIN = max(axial_strain.max(), bending_strain.max())


# ----------------------------
# Helpers
# ----------------------------
def interp_stress(strain, strain_table, stress_table):
    return np.interp(abs(strain), strain_table, stress_table)


def bar_stiffness(p1, p2, E=E_mod, A=A, I=I):
    L_vec = p2 - p1
    L = np.linalg.norm(L_vec)
    if L < 1e-12:
        return np.zeros((6, 6)), L
    n = L_vec / L
    nx, ny, nz = n
    T = np.array([[nx, ny, nz]])
    k_axial = E * A / L
    k_ax = k_axial * np.block([
        [T.T @ T, -(T.T @ T)],
        [-(T.T @ T), T.T @ T]
    ])
    k_bend_val = 12 * E * I / (L ** 3)
    perp = np.eye(3) - np.outer(n, n)
    k_bend = k_bend_val * np.block([
        [perp, -perp],
        [-perp, perp]
    ])
    return k_ax + k_bend, L


# ----------------------------
# FEA Solver
# ----------------------------
def fea_solver(results_dir):
    print(f"🔧 Running FEA on geometry from {results_dir}")

    fea_dir = os.path.join(results_dir, "fea_results")
    os.makedirs(fea_dir, exist_ok=True)

    nodes = pd.read_csv(os.path.join(results_dir, "nodes.csv"))
    elems = pd.read_csv(os.path.join(results_dir, "elements.csv"))

    coords = nodes[["x", "y", "z"]].values
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes
    n_elems = len(elems)

    active = np.ones(n_elems, dtype=bool)
    stress_record = []
    active_record = []
    disp_record = []      # node displacement per step
    force_disp_curve = [] # total force vs displacement

    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    tol = 0.1
    top_nodes = nodes.loc[np.abs(nodes["y"] - y_max) < tol, "node_id"].values.astype(int)
    bot_nodes = nodes.loc[np.abs(nodes["y"] - y_min) < tol, "node_id"].values.astype(int)
    print(f"Top nodes: {len(top_nodes)}, Bottom nodes: {len(bot_nodes)}")

    for step in range(N_STEPS):
        disp_factor = step / (N_STEPS - 1)
        dy_top = +DISPLACEMENT_MAX * disp_factor
        dy_bot = -DISPLACEMENT_MAX * disp_factor
        print(f"➡️  Step {step+1}/{N_STEPS} | dy_top={dy_top:.3f}, dy_bot={dy_bot:.3f}")

        # Assemble global stiffness
        K = np.zeros((n_dof, n_dof))
        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            k_e, _ = bar_stiffness(coords[n1], coords[n2])
            dof = np.r_[3*n1:3*n1+3, 3*n2:3*n2+3]
            K[np.ix_(dof, dof)] += k_e

        # Boundary conditions
        U = np.zeros(n_dof)
        disp_dofs = {3*n+1: dy_top for n in top_nodes}
        disp_dofs.update({3*n+1: dy_bot for n in bot_nodes})
        known_dofs = np.array(list(disp_dofs.keys()))
        known_vals = np.array([disp_dofs[d] for d in known_dofs])
        free_dofs = np.setdiff1d(np.arange(n_dof), known_dofs)

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        K_fk = K[np.ix_(free_dofs, known_dofs)]
        F = np.zeros(n_dof)
        F_f = F[free_dofs] - K_fk @ known_vals
        U_f = np.linalg.solve(K_ff + np.eye(len(K_ff)) * 1e-8, F_f)
        U[free_dofs] = U_f
        U[known_dofs] = known_vals

        # Compute reaction forces on constrained DOFs
        F_react = K @ U
        F_top = F_react[np.array([3*n+1 for n in top_nodes])]
        total_force = F_top.sum()
        total_disp = dy_top - dy_bot
        force_disp_curve.append([total_disp, total_force])

        # Stress computation
        stress = np.zeros(n_elems)
        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            p1, p2 = coords[n1], coords[n2]
            L_vec = p2 - p1
            L = np.linalg.norm(L_vec)
            n = L_vec / L
            u1 = U[3*n1:3*n1+3]
            u2 = U[3*n2:3*n2+3]
            strain = np.dot(n, (u2 - u1)) / L
            stress_ax = interp_stress(strain, axial_strain, axial_stress)
            stress_bend = interp_stress(strain, bending_strain, bending_stress)
            stress[i] = stress_ax + 0.2 * stress_bend
            if abs(strain) > MAX_STRAIN:
                active[i] = False

        stress_record.append(stress)
        active_record.append(active.copy())
        disp_record.append(U.copy())

        # Visualization
        new_coords = coords + U.reshape((-1, 3))
        plt.figure(figsize=(6, 6))
        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            p1, p2 = new_coords[n1], new_coords[n2]
            color_val = stress[i] / max(1e-6, np.max(stress))
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=plt.cm.plasma(color_val))
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.title(f"Step {step+1}/{N_STEPS} - Active: {active.sum()}")
        plt.xlabel("x [mm]")
        plt.ylabel("y [mm]")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(fea_dir, f"fea_step_{step:03d}.png"))
        plt.close()

        if active.sum() == 0:
            print(f"⚠️  Simulation stopped early at step {step+1}.")
            break

    # Save per-element stress and activity
    stress_df = pd.DataFrame(stress_record, columns=[f"elem_{i}" for i in range(n_elems)])
    stress_df["step"] = np.arange(1, len(stress_record)+1)
    stress_df.to_csv(os.path.join(fea_dir, "stress_record.csv"), index=False)

    active_df = pd.DataFrame(active_record, columns=[f"elem_{i}" for i in range(n_elems)])
    active_df["step"] = np.arange(1, len(active_record)+1)
    active_df.to_csv(os.path.join(fea_dir, "active_elements.csv"), index=False)

    # Save node displacements
    disp_cols = [f"node_{i}_x" for i in range(n_nodes)] + \
                [f"node_{i}_y" for i in range(n_nodes)] + \
                [f"node_{i}_z" for i in range(n_nodes)]
    disp_df = pd.DataFrame(disp_record, columns=np.arange(len(disp_record[0])))
    disp_df["step"] = np.arange(1, len(disp_record)+1)
    disp_df.to_csv(os.path.join(fea_dir, "node_displacements.csv"), index=False)

    # Save Force–Displacement curve
    fd_df = pd.DataFrame(force_disp_curve, columns=["total_displacement", "total_force"])
    fd_df.to_csv(os.path.join(fea_dir, "force_displacement.csv"), index=False)

    # Plot Force–Displacement
    plt.figure(figsize=(6, 4))
    plt.plot(fd_df["total_displacement"], fd_df["total_force"], marker="o")
    plt.xlabel("Total Displacement (mm)")
    plt.ylabel("Reaction Force (N)")
    plt.title("Force–Displacement Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(fea_dir, "force_displacement.png"))
    plt.close()

    print(f"✅ FEA completed. Results saved to {fea_dir}")


# ----------------------------
# Run from command line
# ----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fea_solver.py <results_dir>")
        exit()
    fea_solver(sys.argv[1])
