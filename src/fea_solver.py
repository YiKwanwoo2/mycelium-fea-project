import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from scipy.sparse import lil_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve

# ----------------------------
# Material & Simulation Parameters
# ----------------------------
E_mod = 2500  # Young's Modulus in MPa
d = 0.0002  # mm
t = 0.000001  # mm
A = 3.14 * ((d / 2) ** 2 - (d / 2 - t) ** 2) # mm²
# I = 3.14 * ( (d)**4 - (d - 2*t)**4 ) / 64  # mm^4
# A = 3.14 * ((d / 2) ** 2) # mm²
I = A*0.001  # mm^4
N_STEPS = 40
DISPLACEMENT_MAX = 2.00  # mm

# Automatically set failure strain
MAX_STRAIN = 0.018
MAX_STRESS = E_mod * MAX_STRAIN

GRIP_LENGTH = 0.5  # mm

# ----------------------------
# Helpers
# ----------------------------
def interp_stress(strain, strain_table, stress_table):
    return np.interp(abs(strain), strain_table, stress_table)


def bar_stiffness_bulk(p1s, p2s, E=E_mod, A=A, I=I):
    """
    Vectorized stiffness calculator.
    p1s, p2s: (N,3) arrays
    returns K_bulk: (N, 6, 6)
    """
    # vectorized geometry
    L_vec = p2s - p1s                 # (N,3)
    L = np.linalg.norm(L_vec, axis=1) # (N,)

    # avoid divide by zero
    L_safe = np.where(L < 1e-12, 1e-12, L)
    n = L_vec / L_safe[:,None]        # (N,3)

    # axial part
    k_axial = (E * A) / L_safe        # (N,)
    T = n[:, :, None]                 # (N,3,1)
    TT = T @ T.transpose(0,2,1)       # (N,3,3)

    K_ax = np.zeros((len(L), 6, 6))
    K_ax[:,0:3,0:3] =  TT
    K_ax[:,0:3,3:6] = -TT
    K_ax[:,3:6,0:3] = -TT
    K_ax[:,3:6,3:6] =  TT
    K_ax *= k_axial[:,None,None]

    # bending part (vectorized)
    perp = np.eye(3) - n[:,None,:] * n[:,:,None]   # (N,3,3)
    k_bend_val = 12 * E * I / (L_safe**3)

    # same block pattern as axial
    K_b = np.zeros((len(L), 6, 6))
    K_b[:,0:3,0:3] =  perp
    K_b[:,3:6,0:3] = -perp
    K_b[:,0:3,3:6] = -perp
    K_b[:,3:6,3:6] =  perp
    K_b *= k_bend_val[:,None,None]

    return K_ax + K_b, L


# ----------------------------
# Matrix Assembly Function
# ----------------------------
def assemble_global_stiffness(coords, elems, active):
    n_nodes = coords.shape[0]
    n_dof = 3 * n_nodes

    # extract only active elements
    eidx = np.where(active)[0]

    # batch endpoints
    p1s = coords[elems.loc[eidx,"n1"].values]
    p2s = coords[elems.loc[eidx,"n2"].values]

    # vectorized stiffness computation
    K_e_all, _ = bar_stiffness_bulk(p1s, p2s)

    # build COO lists
    rows = []
    cols = []
    vals = []

    for k, e in enumerate(eidx):
        n1 = int(elems.loc[e,"n1"])
        n2 = int(elems.loc[e,"n2"])
        dof = np.r_[3*n1:3*n1+3, 3*n2:3*n2+3]

        Ke = K_e_all[k]
        for i_local in range(6):
            for j_local in range(6):
                rows.append(dof[i_local])
                cols.append(dof[j_local])
                vals.append(Ke[i_local, j_local])

    K = csr_matrix((vals, (rows, cols)), shape=(n_dof, n_dof))
    return K


# ----------------------------
# Solver Function
# ----------------------------
def solve_system(K, known_dofs, known_vals):
    n_dof = K.shape[0]

    free_dofs = np.setdiff1d(np.arange(n_dof), known_dofs)

    # Extract submatrices as sparse
    K_ff = K[free_dofs][:, free_dofs].tocsr()
    K_fk = K[free_dofs][:, known_dofs]

    F = np.zeros(n_dof)
    F_f = F[free_dofs] - K_fk @ known_vals

    # Add tiny regularization for numerical stability
    K_ff = K_ff + 1e-12 * identity(K_ff.shape[0], format="csr")

    # --- HERE IS THE SCIPY SOLVER ---
    U_f = spsolve(K_ff, F_f)   # ← SPARSE, FAST

    # Reconstruct full displacement vector
    U = np.zeros(n_dof)
    U[free_dofs] = U_f
    U[known_dofs] = known_vals

    return U

# ----------------------------
# FEA Solver
# ----------------------------
def fea_solver(results_dir, tol=GRIP_LENGTH):
    start_time = time.time()
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

    top_nodes = nodes.loc[np.abs(nodes["y"] - y_max) < tol, "node_id"].values.astype(int)
    bot_nodes = nodes.loc[np.abs(nodes["y"] - y_min) < tol, "node_id"].values.astype(int)
    print(f"Top nodes: {len(top_nodes)}, Bottom nodes: {len(bot_nodes)}")

    for step in range(N_STEPS):
        disp_factor = step / (N_STEPS - 1)
        dy_top = +DISPLACEMENT_MAX * disp_factor
        dy_bot = -DISPLACEMENT_MAX * disp_factor
        print(f"➡️  Step {step+1}/{N_STEPS} | dy_top={dy_top:.3f}, dy_bot={dy_bot:.3f}")

        # --- ASSEMBLE STIFFNESS ---
        K = assemble_global_stiffness(coords, elems, active)

        # --- BOUNDARY CONDITIONS ---
        disp_dofs = {}

        # Top nodes
        for n in top_nodes:
            disp_dofs.update({
                3*n+0: 0.0,        # x DOF fixed
                3*n+1: dy_top,     # y DOF prescribed
                3*n+2: 0.0         # z DOF fixed
            })

        # Bottom nodes
        for n in bot_nodes:
            disp_dofs.update({
                3*n+0: 0.0,        # x DOF fixed
                3*n+1: dy_bot,     # y DOF prescribed
                3*n+2: 0.0         # z DOF fixed
            })

        known_dofs = np.array(list(disp_dofs.keys()))
        known_vals = np.array([disp_dofs[d] for d in known_dofs])

        # --- SOLVE ---
        try:
            U = solve_system(K, known_dofs, known_vals)
        except np.linalg.LinAlgError:
            print(f"❌ Singular matrix at step {step+1}. Saving partial results and stopping.")
            break

        # --- Now safe to compute reactions ---
        F_react = K @ U
        F_top = F_react[[3*n+1 for n in top_nodes]]
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
            stress_ax = E_mod * strain
            stress[i] = stress_ax
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
            color_val = stress[i] / MAX_STRESS
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

    total_time = time.time() - start_time
    with open(os.path.join(fea_dir, "runtime.txt"), "w") as f:
        f.write(f"Total FEA runtime: {total_time:.6f} seconds\n")

    print(f"⏱️ Total runtime: {total_time:.3f} seconds")

# ----------------------------
# Run from command line
# ----------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fea_solver.py <results_dir>")
        exit()
    fea_solver(sys.argv[1], tol=GRIP_LENGTH)