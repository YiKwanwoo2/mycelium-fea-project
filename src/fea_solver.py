# src/fea_solver_beam.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------
# Parameters
# ----------------------------
A = 1.0           # cross-sectional area (mm^2)
Iy = 1e-3         # moment of inertia about local y (mm^4)
Iz = 1e-3         # moment of inertia about local z (mm^4)
Jt = 1e-3         # torsional constant (mm^4)
G_mod = 3.846e5   # shear modulus placeholder (N/mm^2)
N_STEPS = 40
DISPLACEMENT_MAX = 0.2  # mm
TOL_NR = 1e-6
MAX_ITER = 25

# axial nonlinear table (example) — replace with your actual table
axial_strain = np.array([0.0, 0.01, 0.02, 0.04, 0.05, 5.00])
axial_stress = np.array([0.0, 2e4, 3.5e4, 4.2e4, 4.25e4, 4.25e4])  # N/mm^2

MAX_STRAIN = float(axial_strain.max())

# ----------------------------
# Helpers: material interpolation (value + tangent)
# ----------------------------
def interp_stress_and_tangent(eps, eps_table, sig_table):
    s = abs(eps)
    if s <= eps_table[0]:
        sig = sig_table[0]
        Et = ((sig_table[1]-sig_table[0])/(eps_table[1]-eps_table[0])) if len(eps_table)>1 else 0.0
        return np.sign(eps)*sig, Et
    if s >= eps_table[-1]:
        return np.sign(eps)*sig_table[-1], 0.0
    idx = np.searchsorted(eps_table, s) - 1
    idx = max(0, min(idx, len(eps_table)-2))
    e0, e1 = eps_table[idx], eps_table[idx+1]
    s0, s1 = sig_table[idx], sig_table[idx+1]
    t = (s - e0)/(e1-e0) if (e1-e0)!=0 else 0.0
    sig_pos = s0 + t*(s1-s0)
    Et = (s1 - s0)/(e1 - e0) if (e1-e0)!=0 else 0.0
    return np.sign(eps)*sig_pos, Et

# ----------------------------
# Geometry & transformation
# ----------------------------
def make_rotation_matrix(nvec):
    # nvec: element axis direction (3,)
    nx = nvec / np.linalg.norm(nvec)
    # pick arbitrary vector not parallel to nx
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, nx)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    vy = arbitrary - np.dot(arbitrary, nx) * nx
    vy /= np.linalg.norm(vy)
    vz = np.cross(nx, vy)
    R = np.vstack([nx, vy, vz]).T  # columns are local x,y,z in global coords
    return R

def transform_T(R):
    # build 12x12 transformation matrix for element (u1,phi1,u2,phi2) where each block 3x3
    T = np.zeros((12,12))
    for i in range(4):
        T[3*i:3*i+3, 3*i:3*i+3] = R
    return T

# ----------------------------
# Local element stiffness (12x12) in local coords (linear beam)
# axial uses EA/L, bending uses 12 EI / L^3 etc., torsion GJ/L
def ke_local_linear(E_axial_over_L, EIy_over_L3, EIz_over_L3, GJ_over_L, L):
    # Build in local coordinates (see standard 3D beam stiffness)
    # We'll construct axial, torsion, and two bending blocks.
    k = np.zeros((12,12))
    # axial
    k[0,0] =  E_axial_over_L
    k[0,6] = -E_axial_over_L
    k[6,0] = -E_axial_over_L
    k[6,6] =  E_axial_over_L

    # Torsion (rot x)
    k[3,3] =  GJ_over_L
    k[3,9] = -GJ_over_L
    k[9,3] = -GJ_over_L
    k[9,9] =  GJ_over_L

    # bending about local z (plane y)
    # using standard 2-node beam 4x4 block for transverse DOFs
    # For local w (z) DOF indices: (1,5,7,11)?? careful mapping:
    # local DOF ordering chosen: [u_x1, u_y1, u_z1, rx1, ry1, rz1, u_x2, u_y2, u_z2, rx2, ry2, rz2]
    # For bending about local z, transverse DOF is u_y and rotations rz (around local z produce warping; actually bending about z relates to u_y and rz)
    # We implement bending using classic coefficients for u_trans and rot around out-of-plane axis.
    # For bending about local z (deflection in local y):
    k_y = np.zeros((12,12))
    # DOF indices for deflection in local y: u_y1(1), rz1(5), u_y2(7), rz2(11)
    i1, i2, i3, i4 = 1, 5, 7, 11
    k_y[i1,i1] = 12*EIy_over_L3
    k_y[i1,i2] = 6*EIy_over_L3 * L
    k_y[i1,i3] = -12*EIy_over_L3
    k_y[i1,i4] = 6*EIy_over_L3 * L

    k_y[i2,i1] = 6*EIy_over_L3 * L
    k_y[i2,i2] = 4*EIy_over_L3 * (L**2)
    k_y[i2,i3] = -6*EIy_over_L3 * L
    k_y[i2,i4] = 2*EIy_over_L3 * (L**2)

    k_y[i3,i1] = -12*EIy_over_L3
    k_y[i3,i2] = -6*EIy_over_L3 * L
    k_y[i3,i3] = 12*EIy_over_L3
    k_y[i3,i4] = -6*EIy_over_L3 * L

    k_y[i4,i1] = 6*EIy_over_L3 * L
    k_y[i4,i2] = 2*EIy_over_L3 * (L**2)
    k_y[i4,i3] = -6*EIy_over_L3 * L
    k_y[i4,i4] = 4*EIy_over_L3 * (L**2)

    # bending about local y (deflection in local z)
    k_z = np.zeros((12,12))
    # DOF indices for deflection in local z: u_z1(2), ry1(4), u_z2(8), ry2(10)
    j1, j2, j3, j4 = 2, 4, 8, 10
    k_z[j1,j1] = 12*EIz_over_L3
    k_z[j1,j2] = -6*EIz_over_L3 * L
    k_z[j1,j3] = -12*EIz_over_L3
    k_z[j1,j4] = -6*EIz_over_L3 * L

    k_z[j2,j1] = -6*EIz_over_L3 * L
    k_z[j2,j2] = 4*EIz_over_L3 * (L**2)
    k_z[j2,j3] = 6*EIz_over_L3 * L
    k_z[j2,j4] = 2*EIz_over_L3 * (L**2)

    k_z[j3,j1] = -12*EIz_over_L3
    k_z[j3,j2] = 6*EIz_over_L3 * L
    k_z[j3,j3] = 12*EIz_over_L3
    k_z[j3,j4] = 6*EIz_over_L3 * L

    k_z[j4,j1] = -6*EIz_over_L3 * L
    k_z[j4,j2] = 2*EIz_over_L3 * (L**2)
    k_z[j4,j3] = 6*EIz_over_L3 * L
    k_z[j4,j4] = 4*EIz_over_L3 * (L**2)

    k += k_y + k_z
    k += k_y.T * 0  # keep symmetry (k_y and k_z were constructed symmetric)
    # ensure symmetry
    return 0.5*(k + k.T)

# ----------------------------
# Element internal and tangent (local) using axial nonlinear tangent Et
# ----------------------------
def element_local_response(p1, p2, ulocal, A, Iy, Iz, Jt, axial_eps_table, axial_sig_table):
    # ulocal: 12-vector local element dofs [u1x,u1y,u1z,rx1,ry1,rz1,u2x,u2y,u2z,rx2,ry2,rz2]
    L_vec = p2 - p1
    L = np.linalg.norm(L_vec)
    if L < 1e-12:
        return 0.0, 0.0, np.zeros(12), np.zeros((12,12)), L

    # axial strain approx: (u2x - u1x)/L in local coords
    eps = (ulocal[6] - ulocal[0]) / L
    sig, Et = interp_stress_and_tangent(eps, axial_eps_table, axial_sig_table)
    N = A * sig
    # Build local tangent stiffness replacing axial EA/L with A*Et/L
    EAx_over_L = (A * Et) / L
    EIy_over_L3 = (Et if Et>0 else axial_sig_table[1]/(axial_eps_table[1]+1e-12)) * Iy / (L**3)
    # NOTE: for bending tangent we use linear EI with material E approximated from small-strain tangent (or a chosen E)
    # For simplicity, use Et for bending as well (this is conservative); alternatively use fixed elastic E_bend.
    # We'll use a fixed bending E approximated from initial slope
    E_bend = (axial_sig_table[1] / (axial_eps_table[1]+1e-12))
    EIy_over_L3 = E_bend * Iy / (L**3)
    EIz_over_L3 = E_bend * Iz / (L**3)
    GJ_over_L = G_mod * Jt / L

    k_local = ke_local_linear(EAx_over_L, EIy_over_L3, EIz_over_L3, GJ_over_L, L)

    # Internal force (local)
    f_int = k_local.dot(ulocal)

    # Override the *axial* components to reflect the actual nonlinear stress result
    # Local DOFs: [u1x,u1y,u1z,rx1,ry1,rz1,u2x,u2y,u2z,rx2,ry2,rz2]
    # Axial force acts along local x, positive tension = +N at node 2, -N at node 1
    # f_int[0] = -N
    # f_int[6] = +N

    # axial stress returned, and tangent Et
    return eps, sig, f_int, k_local, L

# ----------------------------
# Global solver
# ----------------------------
def fea_solver_beam(results_dir):
    print("🔧 Running beam FEA (linear EI, nonlinear axial)...")
    fea_dir = os.path.join(results_dir, "fea_results")
    os.makedirs(fea_dir, exist_ok=True)

    nodes = pd.read_csv(os.path.join(results_dir, "nodes.csv"))
    elems = pd.read_csv(os.path.join(results_dir, "elements.csv"))
    coords = nodes[["x","y","z"]].values.copy()
    n_nodes = len(nodes)
    n_dof = 6 * n_nodes   # (ux,uy,uz,rx,ry,rz) per node
    n_elems = len(elems)

    active = np.ones(n_elems, dtype=bool)
    # precompute element reference geometry and rotations
    p1_ref = coords[elems["n1"].astype(int).values]
    p2_ref = coords[elems["n2"].astype(int).values]
    Rs = []
    Ts = []
    Ls = []
    for e in range(n_elems):
        v = p2_ref[e] - p1_ref[e]
        L = np.linalg.norm(v)
        if L < 1e-12:
            R = np.eye(3)
        else:
            R = make_rotation_matrix(v)
        Rs.append(R)
        Ts.append(transform_T(R))
        Ls.append(L)

    # initial global displacement vector (zeros)
    U = np.zeros(n_dof)

    # outputs
    stress_rec = []
    active_rec = []
    disp_rec = []
    fd_curve = []

    y_min, y_max = coords[:,1].min(), coords[:,1].max()
    tol = 0.1
    top_nodes = nodes.loc[np.abs(nodes["y"] - y_max) < tol, "node_id"].values.astype(int)
    bot_nodes = nodes.loc[np.abs(nodes["y"] - y_min) < tol, "node_id"].values.astype(int)
    print(f"Top nodes: {len(top_nodes)}, Bottom nodes: {len(bot_nodes)}")

    for step in range(N_STEPS):
        disp_factor = step / (N_STEPS - 1)
        dy_top = +DISPLACEMENT_MAX * disp_factor
        dy_bot = -DISPLACEMENT_MAX * disp_factor
        print(f"\n➡️ Step {step+1}/{N_STEPS} dy_top={dy_top:.4f} dy_bot={dy_bot:.4f}")

        # prescribed DOFs (only vertical translations here)
        disp_dofs = {6*n+1: dy_top for n in top_nodes}
        disp_dofs.update({6*n+1: dy_bot for n in bot_nodes})
        known_dofs = np.array(sorted(disp_dofs.keys()), dtype=int)
        known_vals = np.array([disp_dofs[d] for d in known_dofs], dtype=float)
        free_dofs = np.setdiff1d(np.arange(n_dof), known_dofs)

        # initial guess: keep previous solution but overwrite known dofs
        U[known_dofs] = known_vals

        # Newton iterations
        converged = False
        for it in range(MAX_ITER):
            K = np.zeros((n_dof, n_dof))
            Fint = np.zeros(n_dof)
            elem_eps = np.zeros(n_elems)
            elem_sig = np.zeros(n_elems)

            for e_idx, row in elems.iterrows():
                if not active[e_idx]:
                    continue
                n1 = int(row.n1); n2 = int(row.n2)
                dof_e = np.r_[6*n1:6*n1+6, 6*n2:6*n2+6]
                # extract global element DOFs and transform to local
                Ue = U[dof_e]
                T = Ts[e_idx]
                ulocal = T.dot(Ue)  # local 12-vector

                # element response
                p1 = p1_ref[e_idx]; p2 = p2_ref[e_idx]
                eps, sig, f_local, k_local, L = element_local_response(p1, p2, ulocal, A, Iy, Iz, Jt, axial_strain, axial_stress)
                elem_eps[e_idx] = eps
                elem_sig[e_idx] = sig

                # transform local internal force/tangent to global
                f_global = T.T.dot(f_local)
                K_global = T.T.dot(k_local).dot(T)

                Fint[dof_e] += f_global
                K[np.ix_(dof_e, dof_e)] += K_global

                # check failure (axial)
                if abs(eps) > MAX_STRAIN:
                    active[e_idx] = False
                    # remove last contributions
                    Fint[dof_e] -= f_global
                    K[np.ix_(dof_e, dof_e)] -= K_global
                    elem_eps[e_idx] = 0.0
                    elem_sig[e_idx] = 0.0
                    print(f"  ⚠ Element {e_idx} failed (|eps|={abs(eps):.6f}).")

            # residual for equilibrium (free DOFs): r = K_ff * U_f - (Fint_f - K_fk*U_k)
            if len(free_dofs) == 0:
                print("No free DOFs left; stopping.")
                converged = True
                break
            K_ff = K[np.ix_(free_dofs, free_dofs)]
            K_fk = K[np.ix_(free_dofs, known_dofs)]
            Fint_f = Fint[free_dofs]
            U_f = U[free_dofs]
            U_k = U[known_dofs]
            r = K_ff.dot(U_f) - (Fint_f - K_fk.dot(U_k))
            rnorm = np.linalg.norm(r)
            print(f"  iter {it+1}: residual = {rnorm:.3e}")

            if rnorm < TOL_NR:
                converged = True
                break

            # solve for correction
            try:
                delta = np.linalg.solve(K_ff + np.eye(len(K_ff))*1e-12, -r)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(K_ff + np.eye(len(K_ff))*1e-12, -r, rcond=None)[0]
            U[free_dofs] += delta

        if not converged:
            print(f"  ❗ Newton not converged (resid {rnorm:.3e})")

        # Final recompute to store stresses and reactions
        # recompute final element stresses and Fint
        final_eps = np.zeros(n_elems)
        final_sig = np.zeros(n_elems)
        Fint = np.zeros(n_dof)
        for e_idx, row in elems.iterrows():
            if not active[e_idx]:
                continue
            n1 = int(row.n1); n2 = int(row.n2)
            dof_e = np.r_[6*n1:6*n1+6, 6*n2:6*n2+6]
            Ue = U[dof_e]
            T = Ts[e_idx]
            ulocal = T.dot(Ue)
            eps, sig, f_local, k_local, L = element_local_response(p1_ref[e_idx], p2_ref[e_idx], ulocal, A, Iy, Iz, Jt, axial_strain, axial_stress)
            final_eps[e_idx] = eps
            final_sig[e_idx] = sig

            # f_global = T.T.dot(f_local)
            # Fint[dof_e] += f_global

            # Replace the internal force with physically correct nonlinear axial result
            N = A * sig  # actual axial force

            # Build local force vector for pure axial contribution
            f_local_true = np.zeros(12)
            f_local_true[0]  = -N
            f_local_true[6]  = +N

            # Keep tangent-based bending/shear parts (from f_local)
            # if your element includes bending
            f_local_true += (f_local - np.array([f_local[0],0,0,0,0,0,f_local[6],0,0,0,0,0]))

            # Transform to global coordinates
            f_global = T.T.dot(f_local_true)
            Fint[dof_e] += f_global

            if abs(eps) > MAX_STRAIN and active[e_idx]:
                active[e_idx] = False
                print(f"  ⚠ Element {e_idx} failed post-step (|eps|={abs(eps):.6f}).")
        reactions = -Fint
        total_force = reactions[[6*n+1 for n in top_nodes]].sum()
        total_disp = dy_top - dy_bot
        fd_curve.append([total_disp, total_force])

        stress_rec.append(final_sig.copy())
        active_rec.append(active.copy().astype(int))
        disp_rec.append(U.copy())

        # Visualization (top view)
        new_coords = coords.copy()
        # apply translations only (ignore rotation in node coords for plotting)
        for n in range(n_nodes):
            new_coords[n] += U[6*n:6*n+3]
        plt.figure(figsize=(6,6))
        vmax = np.max(np.abs(final_sig)) if np.any(final_sig) else 1e-6
        for e_idx, row in elems.iterrows():
            if not active[e_idx]:
                continue
            n1 = int(row.n1); n2 = int(row.n2)
            p1, p2 = new_coords[n1], new_coords[n2]
            cval = final_sig[e_idx] / max(1e-12, np.max(np.abs(axial_stress)))
            cmapv = np.clip(cval, 0.0, 1.0)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=plt.cm.plasma(cmapv))
        plt.xlim(-3,3); plt.ylim(-3,3)
        plt.title(f"Step {step+1}/{N_STEPS} Active: {active.sum()}")
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(fea_dir, f"fea_step_{step:03d}.png"))
        plt.close()

        print(f"  ✔ Step {step+1} done. Active elems: {active.sum()}, total vertical reaction: {total_force:.6e}")

        if active.sum() == 0:
            print("All elements broken. Stop.")
            break

    # Save CSVs
    stress_df = pd.DataFrame(stress_rec, columns=[f"elem_{i}" for i in range(n_elems)])
    stress_df["step"] = np.arange(1, len(stress_rec)+1)
    stress_df.to_csv(os.path.join(fea_dir, "stress_record.csv"), index=False)

    active_df = pd.DataFrame(active_rec, columns=[f"elem_{i}" for i in range(n_elems)])
    active_df["step"] = np.arange(1, len(active_rec)+1)
    active_df.to_csv(os.path.join(fea_dir, "active_elements.csv"), index=False)

    disp_df = pd.DataFrame(np.vstack(disp_rec) if disp_rec else np.zeros((0,n_dof)), columns=[f"dof_{i}" for i in range(n_dof)])
    disp_df["step"] = np.arange(1, len(disp_rec)+1)
    disp_df.to_csv(os.path.join(fea_dir, "node_displacements.csv"), index=False)

    fd_df = pd.DataFrame(fd_curve, columns=["total_displacement", "total_force"])
    fd_df.to_csv(os.path.join(fea_dir, "force_displacement.csv"), index=False)
    if len(fd_df):
        plt.figure(figsize=(6,4))
        plt.plot(fd_df["total_displacement"], fd_df["total_force"], marker="o")
        plt.xlabel("Total displacement (mm)")
        plt.ylabel("Total reaction (N)")
        plt.tight_layout()
        plt.savefig(os.path.join(fea_dir, "force_displacement.png"))
        plt.close()

    print(f"✅ FEA complete. Results saved to {fea_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/fea_solver_beam.py <results_dir>")
        exit()
    fea_solver_beam(sys.argv[1])
