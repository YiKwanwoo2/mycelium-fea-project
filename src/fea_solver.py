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

# Nonlinear stress–strain curves (axial used; bending stress/strain kept but will not be used for stress)
axial_strain = np.array([0.0, 0.01, 0.02, 0.04, 0.05, 5.00])
axial_stress = np.array([0.0, 2e4, 3.5e4, 4.2e4, 4.25e4, 4.30e4])

# Automatically set failure strain
MAX_STRAIN = axial_strain.max()
MAX_STRESS = axial_stress.max()

# Iteration params
MAX_ITERS = 25
TOL_U = 1e-6

# ----------------------------
# Helpers
# ----------------------------
def interp_stress_and_tangent(strain, strain_table, stress_table):
    """
    Returns (stress, tangent_modulus) for given scalar strain (can be negative).
    Uses absolute strain for lookup (material symmetric in given tables).
    Tangent modulus is the slope of the piecewise linear stress-strain curve at that point.
    """
    s = abs(strain)
    # stress by interpolation
    stress = np.interp(s, strain_table, stress_table)

    # find tangent (piecewise slope)
    # handle edge cases
    if s <= strain_table[0]:
        tangent = (stress_table[1] - stress_table[0]) / (strain_table[1] - strain_table[0])
    elif s >= strain_table[-1]:
        # beyond last point, tangent -> 0
        tangent = 0.0
    else:
        idx = np.searchsorted(strain_table, s) - 1
        x1, x2 = strain_table[idx], strain_table[idx + 1]
        y1, y2 = stress_table[idx], stress_table[idx + 1]
        # avoid division by zero
        if x2 == x1:
            tangent = 0.0
        else:
            tangent = (y2 - y1) / (x2 - x1)
    return np.sign(strain) * stress, tangent  # preserve sign of stress for compression/tension


def bar_stiffness(p1, p2, E=E_mod, A=A, I=I):
    """
    Returns (k_local, length, direction_n, perp) where k_local is the 6x6 stiffness
    containing axial and perpendicular (bending) stiffness terms. The axial stiffness
    value k_axial is returned implicitly inside the matrix; for nonlinear solver we
    replace axial stiffness with tangent E_t*A/L as needed by reconstructing the axial block.
    """
    L_vec = p2 - p1
    L = np.linalg.norm(L_vec)
    if L < 1e-12:
        return np.zeros((6, 6)), L, np.zeros(3), np.eye(3)
    n = L_vec / L
    nx, ny, nz = n
    T = np.array([[nx, ny, nz]])
    # linear axial stiffness (used as baseline; for nonlinear we'll swap tangent)
    k_axial = E * A / L
    k_ax = k_axial * np.block([
        [T.T @ T, -(T.T @ T)],
        [-(T.T @ T), T.T @ T]
    ])
    # perpendicular stiffness retained (depends on E*I and geometry)
    k_bend_val = 12 * E * I / (L ** 3)
    perp = np.eye(3) - np.outer(n, n)
    k_bend = k_bend_val * np.block([
        [perp, -perp],
        [-perp, perp]
    ])
    return k_ax + k_bend, L, n, perp


def assemble_global_tangent(nodes_coords, elems_df, active, axial_tangents):
    """
    Assemble global tangent stiffness K using axial_tangents (array per element)
    for the axial contribution, while keeping the perpendicular stiffness term from geometry.
    axial_tangents should be E_t * A / L for each element (scalar).
    """
    n_nodes = len(nodes_coords)
    n_dof = 3 * n_nodes
    K = np.zeros((n_dof, n_dof))
    for i, row in elems_df.iterrows():
        if not active[i]:
            continue
        n1, n2 = int(row.n1), int(row.n2)
        p1, p2 = nodes_coords[n1], nodes_coords[n2]
        k_lin, L, n_vec, perp = bar_stiffness(p1, p2)  # returns k_ax + k_bend but k_ax uses E_mod
        # reconstruct axial block with provided tangent
        if L < 1e-12:
            continue
        T = np.array([[n_vec[0], n_vec[1], n_vec[2]]])
        k_ax_t = axial_tangents[i] * np.block([
            [T.T @ T, -(T.T @ T)],
            [-(T.T @ T), T.T @ T]
        ])
        # extract bending block from k_lin by subtracting linear axial (E_mod*A/L) contribution
        k_ax_lin = (E_mod * A / L) * np.block([
            [T.T @ T, -(T.T @ T)],
            [-(T.T @ T), T.T @ T]
        ])
        k_bend = k_lin - k_ax_lin
        k_e = k_ax_t + k_bend
        dof = np.r_[3*n1:3*n1+3, 3*n2:3*n2+3]
        K[np.ix_(dof, dof)] += k_e
    return K


# ----------------------------
# FEA Solver (with iterative tangent updates)
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

    # initial active flags
    active = np.ones(n_elems, dtype=bool)

    # initial axial strains & stresses (zero)
    elem_strain = np.zeros(n_elems)
    elem_stress = np.zeros(n_elems)

    stress_record = []
    active_record = []
    disp_record = []      # node displacement per step
    force_disp_curve = [] # total force vs displacement

    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    tol = 0.1
    top_nodes = nodes.loc[np.abs(nodes["y"] - y_max) < tol, "node_id"].values.astype(int)
    bot_nodes = nodes.loc[np.abs(nodes["y"] - y_min) < tol, "node_id"].values.astype(int)
    print(f"Top nodes: {len(top_nodes)}, Bottom nodes: {len(bot_nodes)}")

    # initial tangent moduli: use initial slope of axial table at zero
    initial_tangent = (axial_stress[1] - axial_stress[0]) / (axial_strain[1] - axial_strain[0])
    axial_tangents = np.full(n_elems, initial_tangent * A)  # note multiplied by A; we'll divide by L later

    for step in range(N_STEPS):
        disp_factor = step / (N_STEPS - 1)
        dy_top = +DISPLACEMENT_MAX * disp_factor
        dy_bot = -DISPLACEMENT_MAX * disp_factor
        print(f"➡️  Step {step+1}/{N_STEPS} | dy_top={dy_top:.6f}, dy_bot={dy_bot:.6f}")

        # prescribed displacements
        disp_dofs = {3*n+1: dy_top for n in top_nodes}
        disp_dofs.update({3*n+1: dy_bot for n in bot_nodes})
        known_dofs = np.array(list(disp_dofs.keys()), dtype=int)
        known_vals = np.array([disp_dofs[d] for d in known_dofs], dtype=float)
        free_dofs = np.setdiff1d(np.arange(n_dof), known_dofs)

        # initialize iteration: use previous converged displacements if available
        U = np.zeros(n_dof)
        if len(disp_record) > 0:
            U = disp_record[-1].copy()  # start from previous step solution

        # iterative loop to account for nonlinear axial material (load-controlled would differ)
        converged = False
        for it in range(1, MAX_ITERS + 1):
            # Build axial_tangents per element as E_t * A / L
            axial_tangent_per_length = np.zeros(n_elems)
            for i, row in elems.iterrows():
                if not active[i]:
                    axial_tangent_per_length[i] = 0.0
                    continue
                n1, n2 = int(row.n1), int(row.n2)
                p1, p2 = coords[n1], coords[n2]
                L_vec = p2 - p1
                L = np.linalg.norm(L_vec)
                if L < 1e-12:
                    axial_tangent_per_length[i] = 0.0
                    continue
                # compute current element axial strain based on current U
                u1 = U[3*n1:3*n1+3]
                u2 = U[3*n2:3*n2+3]
                n_vec = L_vec / L
                strain_i = np.dot(n_vec, (u2 - u1)) / L
                elem_strain[i] = strain_i
                # get stress & tangent modulus from axial table
                stress_i, tangent_i = interp_stress_and_tangent(strain_i, axial_strain, axial_stress)
                elem_stress[i] = stress_i
                # tangent contribution to stiffness: E_t * A / L  (tangent_i already in stress units per strain)
                axial_tangent_per_length[i] = (tangent_i * A) / L

            # assemble global tangent stiffness
            # note: assemble_global_tangent expects axial_tangents = E_t * A / L for each element (scalar)
            K = assemble_global_tangent(coords, elems, active, axial_tangent_per_length)

            # partition and solve for free DOFs with the known displacements
            K_ff = K[np.ix_(free_dofs, free_dofs)]
            K_fk = K[np.ix_(free_dofs, known_dofs)]

            # external load vector zero (we're displacement prescribing known_dofs)
            F_ext = np.zeros(n_dof)
            F_f = F_ext[free_dofs] - K_fk @ known_vals

            # small regularization to avoid singular K_ff in early steps
            reg = 1e-10 * np.eye(len(K_ff))
            try:
                U_f_new = np.linalg.solve(K_ff + reg, F_f)
            except np.linalg.LinAlgError:
                # fallback: pseudo-inverse
                U_f_new = np.linalg.pinv(K_ff + reg) @ F_f

            U_new = U.copy()
            U_new[free_dofs] = U_f_new
            U_new[known_dofs] = known_vals

            # check convergence on free DOF displacement increment
            du = U_new - U
            max_du = np.max(np.abs(du))
            print(f"   iter {it:02d} | max dU = {max_du:.3e}")

            U = U_new
            if max_du < TOL_U:
                converged = True
                print(f"   ✅ Converged in {it} iterations (max dU {max_du:.3e}).")
                break

        if not converged:
            print(f"   ⚠️  Did not converge in {MAX_ITERS} iterations (last max dU {max_du:.3e}). Proceeding with last iterate.")

        # --- Assemble global internal force from element axial stresses (true internal resisting force) ---
        F_internal = np.zeros(n_dof)

        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            p1, p2 = coords[n1], coords[n2]
            L_vec = p2 - p1
            L = np.linalg.norm(L_vec)
            if L < 1e-12:
                continue
            n_vec = L_vec / L
            # axial internal force (positive in tension)
            N_axial = elem_stress[i] * A
            # nodal contributions: node1 gets -N*n, node2 gets +N*n
            f1 = -N_axial * n_vec
            f2 = +N_axial * n_vec
            dof = np.r_[3*n1:3*n1+3, 3*n2:3*n2+3]
            F_internal[dof[:3]] += f1
            F_internal[dof[3:]] += f2

        # Reaction at top nodes (extract y-components)
        F_top = F_internal[np.array([3*n+1 for n in top_nodes])]
        total_force = F_top.sum()
        total_disp = dy_top - dy_bot
        force_disp_curve.append([total_disp, total_force])

        # Update element strains & stresses again (final)
        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            p1, p2 = coords[n1], coords[n2]
            L_vec = p2 - p1
            L = np.linalg.norm(L_vec)
            if L < 1e-12:
                elem_strain[i] = 0.0
                elem_stress[i] = 0.0
                continue
            n_vec = L_vec / L
            u1 = U[3*n1:3*n1+3]
            u2 = U[3*n2:3*n2+3]
            strain = np.dot(n_vec, (u2 - u1)) / L
            stress_ax, _ = interp_stress_and_tangent(strain, axial_strain, axial_stress)
            # Bending stress is discarded in constitutive stress result as requested.
            elem_strain[i] = strain
            elem_stress[i] = stress_ax
            # check failure
            if abs(strain) > MAX_STRAIN:
                active[i] = False

        stress_record.append(elem_stress.copy())
        active_record.append(active.copy())
        disp_record.append(U.copy())

        # Visualization (only plot active elements)
        new_coords = coords + U.reshape((-1, 3))
        plt.figure(figsize=(6, 6))
        # for color scaling use absolute stress (avoid division by zero)
        max_stress_val = max(1e-6, np.max(np.abs(elem_stress)))
        for i, row in elems.iterrows():
            if not active[i]:
                continue
            n1, n2 = int(row.n1), int(row.n2)
            p1, p2 = new_coords[n1], new_coords[n2]
            color_val = abs(elem_stress[i]) / MAX_STRESS
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
    # flatten U vectors into columns: node_0_x, node_1_x, ... node_0_y, ...
    disp_mat = np.array(disp_record)  # each row is global U vector
    # Build column names for flattened U vector
    disp_cols = []
    for comp in ["x", "y", "z"]:
        for i in range(n_nodes):
            disp_cols.append(f"node_{i}_{comp}")
    disp_df = pd.DataFrame(disp_mat, columns=disp_cols)
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
