# mycelium_sim.py
# Minimal Python prototype of the lattice-free fungal growth model (Ulzurrun et al. 2017)
# Requires: numpy, matplotlib

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math
import os
import pandas as pd
import datetime

SEED = 42   # choose any integer
random.seed(SEED)
np.random.seed(SEED)

# -------------------------
# PARAMETERS (defaults come from Ulzurrun et al. 2017 Table 5/3)
# -------------------------
h0 = 0.05               # mm, segment length
dt = 0.01               # days, time step
lambda_angle = math.pi/6  # max angular variation for new segment
P_branch = 0.5          # branching probability
c_g = 1e-7              # mol/mm, cost-of-growth per mm
D = 3.456               # mm/day, internal substrate diffusion coeff
M_cap = 2e-6            # mol/mm, max conc per mm
initial_tips = 25
Omega0 = 5e-6 #5e-6           # total initial internal substrate (mol)
T_steps = 25           # number of steps for demo

WALL_THICKNESS = 0.05
DISH_SIZE = 5.0
HEIGHT = 0.1
H0_PER_POINT = 10
SUBSTRATE_WIDTH = 2.0  # mm, width of substrate in Y direction

dist_inoculum = 0.5  # mm, distance between multiple inoculum points

INOCULUM_POINTS = [
    [-dist_inoculum, dist_inoculum/2, 0.0],   # UP
    [0.0,            dist_inoculum/2, 0.0],
    [dist_inoculum,  dist_inoculum/2, 0.0],
    [-dist_inoculum,-dist_inoculum/2, 0.0],
    [0.0,           -dist_inoculum/2, 0.0],
    [dist_inoculum, -dist_inoculum/2, 0.0]   # DOWN
]

# geometric & numerical tolerances
ANASTOMOSIS_TOL = 5e-3 #1e-3  # mm, tolerance to detect intersection (very small)

# Save outputs under results/
RESULTS_DIR = "results"
SNAPSHOT_DIR = os.path.join(RESULTS_DIR, "snapshots")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# -------------------------
# Basic geometry helpers
# -------------------------
def sph_to_cart(theta, phi):
    # theta: polar angle (0..pi), phi: azimuth (0..2pi)
    st = math.sin(theta)
    return np.array([st*math.cos(phi), st*math.sin(phi), math.cos(theta)])

def rand_direction_from(theta, phi, lam=lambda_angle):
    # add uniform noise in both angles ∈ [-lam/2, lam/2]
    dth = (random.random()-0.5)*lam
    dph = (random.random()-0.5)*lam
    return theta + dth, phi + dph

def segment_length(p1, p2):
    return np.linalg.norm(p2-p1)

def point_segment_distance(p, a, b):
    """
    Compute shortest distance between point p and segment a-b.
    Returns (distance, projection_point)
    """
    ap = p - a
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 < 1e-12:  # degenerate segment
        return np.linalg.norm(ap), a
    t = np.dot(ap, ab) / ab2
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    dist = np.linalg.norm(p - proj)
    return dist, proj

# -------------------------
# Data structures
# -------------------------
class Segment:
    # store: start point, end point, direction (theta,phi), internal substrate I, state, age
    # state: 'A' active tip, 'P' passive, 'B' branched, 'S' anastomosed
    def __init__(self, start, end, theta, phi, I=0.0, state='A', age=0):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.theta = theta
        self.phi = phi
        self.I = float(I)
        self.state = state
        self.age = age
    def length(self):
        return segment_length(self.start, self.end)
    def endpoint(self):
        return self.end

class Hypha:
    # list of connected segments (in order)
    def __init__(self, segments=None):
        self.segments = [] if segments is None else list(segments)

class Cuboid:
    # axis-aligned cuboid for environment: center (x,y,z), size (Lx,Ly,Lz)
    # type: 'substrate', 'impenetrable', 'inhibitor', 'tropism' (only substrate implemented here)
    def __init__(self, center, size, ctype='substrate', attrs=None):
        self.center = np.array(center, dtype=float)
        self.size = np.array(size, dtype=float)
        self.ctype = ctype
        self.attrs = {} if attrs is None else dict(attrs)
        # for substrate cuboid, attrs may include 'E' amount (mol) and 'mu' uptake coeff
    def contains_point(self, p):
        minp = self.center - 0.5*self.size
        maxp = self.center + 0.5*self.size
        return np.all(p >= minp - 1e-12) and np.all(p <= maxp + 1e-12)

# -------------------------
# Mycelium container (F_t)
# -------------------------
class Mycelium:
    def __init__(self):
        self.hyphae = []   # list of Hypha objects
        # indices for operations
        self.segments_index = []  # flat list of (hypha_idx, seg_idx) for quick traversal

    def rebuild_index(self):
        self.segments_index = []
        for i,h in enumerate(self.hyphae):
            for j,s in enumerate(h.segments):
                self.segments_index.append((i,j))

    def all_segments(self):
        for i,h in enumerate(self.hyphae):
            for j,s in enumerate(h.segments):
                yield (i,j,s)

    def tip_segments(self):
        # yield tip segments (the last segment of each hypha) that are active
        for i,h in enumerate(self.hyphae):
            if len(h.segments)==0: continue
            s = h.segments[-1]
            yield (i, len(h.segments)-1, s)

    def total_internal_substrate(self):
        return sum(s.I*s.length() for _,_,s in self.all_segments())

    def total_hyphal_length(self):
        return sum(s.length() for _,_,s in self.all_segments())

def summarize_mycelium(M):
    """
    Compute and return key network statistics:
      - Total hyphae
      - Total segments
      - Active / passive / anastomosed counts
      - Number of branches
      - Number of merges (anastomoses)
      - Total hyphal length
    """
    n_hypha = len(M.hyphae)
    n_segments = 0
    n_active = 0
    n_passive = 0
    n_anasto = 0
    n_branches = 0

    for _, _, s in M.all_segments():
        n_segments += 1
        if s.state == 'A':
            n_active += 1
        elif s.state == 'P':
            n_passive += 1
        elif s.state == 'S':
            n_anasto += 1

    # Approximate number of branches = number of hyphae - number of inoculum sites
    # since each new hypha starts from a branching event
    n_branches = max(0, n_hypha - len(INOCULUM_POINTS))

    total_len = M.total_hyphal_length()

    stats = {
        "hyphae": n_hypha,
        "segments": n_segments,
        "active_tips": n_active,
        "passive_tips": n_passive,
        "anastomosed": n_anasto,
        "branches": n_branches,
        "total_length_mm": total_len
    }
    return stats

# -------------------------
# Initialization
# -------------------------
def initialize_inoculum(points=INOCULUM_POINTS, H0_per_point=10, Omega=Omega0):
    """
    Initialize multiple inoculum sites.
    Each point spawns several hyphae with random directions.
    """
    M = Mycelium()
    total_points = len(points)
    per_site_substrate = Omega / max(1, total_points)

    for site_idx, pos in enumerate(points):
        per_seg = per_site_substrate / H0_per_point
        for i in range(H0_per_point):
            theta = random.random() * math.pi
            phi = random.random() * 2 * math.pi
            start = np.array(pos)
            dir_v = sph_to_cart(theta, phi)
            end = start + dir_v * h0
            seg = Segment(
                start, end, theta, phi,
                I=per_seg / h0, state='A', age=0
            )
            M.hyphae.append(Hypha([seg]))
    M.rebuild_index()
    return M

# -------------------------
# Substrate translocation (simplified following Eqs (2)-(5))
# -------------------------
def translocate_internal_substrate(M, D=D, dt=dt):
    # We'll compute pairwise exchanges only between connected neighbors (predecessor-successor)
    # plus anastomosis-linked neighbors (we approximate by checking connectivity via shared endpoints)
    # Implementation simplified: iterate segments and exchange with their immediate predecessor only
    # For each hypha: for each segment s (except first) exchange with predecessor
    updates = []
    for i,h in enumerate(M.hyphae):
        for j, s in enumerate(h.segments):
            # predecessor within same hypha
            if j>0:
                pred = h.segments[j-1]
                # maximal transferable amount (eq.2 simplified)
                denom = (s.length() + pred.length())/2.0
                if denom <= 0: continue
                delta = dt * D * (pred.I - s.I) / denom  # note: I stored as mol/mm
                # clamp to ranges so resulting I within [0, M_cap]
                new_s = s.I + delta
                new_pred = pred.I - delta
                # Clamp so not negative or >M_cap
                # We adjust delta if clamp violated
                if new_s < 0:
                    delta_adj = -s.I
                elif new_s > M_cap:
                    delta_adj = M_cap - s.I
                elif new_pred < 0:
                    delta_adj = pred.I
                elif new_pred > M_cap:
                    delta_adj = M_cap - pred.I
                else:
                    delta_adj = delta
                updates.append((s, delta_adj))
                updates.append((pred, -delta_adj))
    # apply updates (additive)
    for seg, dI in updates:
        seg.I += dI
        seg.I = max(0.0, min(M_cap, seg.I))

# -------------------------
# Uptake from cuboids
# -------------------------
def uptake_from_cuboids(M, cuboids, dt=dt):
    # For every substrate cuboid and any segment whose endpoint lies inside it,
    # transfer using eq (7): theta = dt * mu_k * E_k * I_segment  (simplified linearized form)
    for c in cuboids:
        if c.ctype != 'substrate': continue
        mu = c.attrs.get('mu', 1e8)  # default from paper
        E = c.attrs.get('E', 0.0)
        if E <= 0: continue
        # find intersecting segments (endpoint inside)
        intersecting = []
        for _,_,s in M.all_segments():
            p = s.endpoint()
            if c.contains_point(p):
                intersecting.append(s)
        # iterate in order (paper uses array order; we just iterate)
        for s in intersecting:
            theta = dt * mu * E * s.I  # simplistic linear uptake; note units are illustrative
            # clamp so segment I <= M_cap and E >= 0
            theta = max(0.0, min(theta, min(M_cap - s.I, E)))
            s.I += theta
            E -= theta
            if E <= 0:
                break
        c.attrs['E'] = E

def enforce_impenetrable_boundaries(M, cuboids, max_iter=3):
    """
    Prevent hyphae from growing through impenetrable cuboids.
    If a tip penetrates a wall, project it tangentially along the wall surface.
    Re-check until it is outside all walls (handles corner intersections).
    """
    for _, _, s in M.tip_segments():
        for _ in range(max_iter):  # allow multiple adjustments
            penetrated = False
            for c in cuboids:
                if c.ctype != 'impenetrable':
                    continue
                if c.contains_point(s.endpoint()):
                    penetrated = True
                    # compute which face was hit
                    delta = s.end - c.center
                    half_size = c.size / 2.0
                    normal = np.zeros(3)
                    overlap = np.abs(delta) - half_size
                    idx = np.argmax(overlap)
                    normal[idx] = np.sign(delta[idx])

                    # project growth direction to slide along surface
                    dir_vec = s.end - s.start
                    if np.linalg.norm(dir_vec) < 1e-12:
                        dir_vec = np.random.randn(3)
                    dir_vec /= np.linalg.norm(dir_vec)

                    # slide along surface (remove normal component)
                    dir_vec = dir_vec - np.dot(dir_vec, normal) * normal
                    if np.linalg.norm(dir_vec) < 1e-12:
                        # fallback small random tangent
                        dir_vec = np.random.randn(3)
                        dir_vec[idx] = 0.0
                    dir_vec /= np.linalg.norm(dir_vec)

                    # move endpoint slightly inside valid space
                    s.end = s.start + dir_vec * s.length()
                    s.theta = math.acos(dir_vec[2])
                    s.phi = math.atan2(dir_vec[1], dir_vec[0])
                    s.state = 'A'
                    break  # restart checking from first cuboid
            if not penetrated:
                break  # endpoint is valid, stop early

# -------------------------
# Growth: apical and branching
# -------------------------
def attempt_growth(M, P_branch=P_branch, c_g=c_g, h0=h0):
    """
    For each active tip segment, extend the mycelium if sufficient internal substrate is available.
    Distribute internal substrate to new segments to keep them alive.
    """
    new_hyphae = []

    for i, h in enumerate(M.hyphae):
        if not h.segments:
            continue

        tip = h.segments[-1]
        if tip.state != 'A':
            continue

        available_mol = tip.I * tip.length()
        cost_one = c_g * h0

        if available_mol < cost_one:
            continue  # not enough to grow

        do_branch = (random.random() < P_branch) and (available_mol >= 2 * cost_one)

        if do_branch:
            # Consume substrate
            total_cost = 2 * cost_one
            tip.I = max(0.0, (available_mol - total_cost) / tip.length())
            tip.state = 'P'

            # Continue parent direction (slightly varied)
            th0, ph0 = rand_direction_from(tip.theta, tip.phi)
            dir_v0 = sph_to_cart(th0, ph0)
            end0 = tip.endpoint() + dir_v0 * h0
            new_seg_parent = Segment(
                tip.endpoint(), end0, th0, ph0,
                I=0.5 * tip.I,  # give it half of remaining substrate
                state='A'
            )

            # Create child branch with a different random direction
            th1, ph1 = rand_direction_from(tip.theta, tip.phi)
            dir_v1 = sph_to_cart(th1, ph1)
            end1 = tip.endpoint() + dir_v1 * h0
            new_seg_child = Segment(
                tip.endpoint(), end1, th1, ph1,
                I=0.5 * tip.I,  # give half to the branch
                state='A'
            )

            # Add both to the system
            h.segments.append(new_seg_parent)
            new_hyphae.append(Hypha([new_seg_child]))

        else:
            # Simple apical growth
            tip.state = 'P'
            tip.I = max(0.0, (available_mol - cost_one) / tip.length())

            th, ph = rand_direction_from(tip.theta, tip.phi)
            dir_v = sph_to_cart(th, ph)
            end = tip.endpoint() + dir_v * h0
            new_seg = Segment(
                tip.endpoint(), end, th, ph,
                I=0.5 * tip.I,  # give it some substrate to continue growing
                state='A'
            )
            h.segments.append(new_seg)

    # Add new hyphae from branching
    for nh in new_hyphae:
        M.hyphae.append(nh)

    M.rebuild_index()

# ============================================================
# Spatial Hash (voxel grid) for accelerating segment queries
# ============================================================
class SpatialHash:
    def __init__(self, voxel_size=0.2):
        self.voxel_size = float(voxel_size)
        self.hash = {}  # key: (ix,iy,iz), value: list of (hypha_idx, seg_idx, segment)

    def _voxel_coords(self, p):
        return (
            int(p[0] // self.voxel_size),
            int(p[1] // self.voxel_size),
            int(p[2] // self.voxel_size)
        )

    def insert_segment(self, hi, si, seg):
        # Insert both endpoints (or center) so segment is discoverable
        p = (seg.start + seg.end) * 0.5
        key = self._voxel_coords(p)
        self.hash.setdefault(key, []).append((hi, si, seg))

    def rebuild(self, M):
        """Recompute all voxel entries from the entire mycelium."""
        self.hash.clear()
        for hi, h in enumerate(M.hyphae):
            for si, seg in enumerate(h.segments):
                self.insert_segment(hi, si, seg)

    def nearby_segments(self, p):
        """Return all segments in voxels around point p."""
        ix, iy, iz = self._voxel_coords(p)

        # 3×3×3 neighboring voxel search
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    key = (ix+dx, iy+dy, iz+dz)
                    if key in self.hash:
                        for entry in self.hash[key]:
                            yield entry  # (hypha_idx, seg_idx, seg)

# -------------------------
# Simple anastomosis detection: if new endpoint within tol of any existing segment (not its parent),
# snap to nearest point and mark it as 'S' (anastomosed)
# -------------------------
def detect_anastomosis(M, spatial, tol=ANASTOMOSIS_TOL):
    """
    Voxel-accelerated anastomosis detection.
    Only checks nearby segments instead of *all* segments.
    """
    for hi, h in enumerate(M.hyphae):
        if not h.segments:
            continue

        tip_idx = len(h.segments) - 1
        tip = h.segments[tip_idx]
        if tip.state != 'A':
            continue

        p = tip.endpoint()
        found = False

        # Query nearby segments (≈O(1) average)
        for ii, jj, seg in spatial.nearby_segments(p):

            # Skip self & parent segment
            if ii == hi and jj == tip_idx:
                continue

            dist, proj = point_segment_distance(p, seg.start, seg.end)
            if dist <= tol:
                # Snap endpoint to segment
                tip.end = proj.copy()
                tip.state = 'S'
                found = True
                break

        # If anastomosis occurred, update spatial hash entry for this tip
        if found:
            # Update hash for modified segment
            spatial.insert_segment(hi, tip_idx, tip)

# -------------------------
# Visualization
# -------------------------
def plot_mycelium(M, step, cuboids=None, title=None, show=True, snapshot_dir=None):
    if snapshot_dir is None:
        snapshot_dir = SNAPSHOT_DIR
    os.makedirs(snapshot_dir, exist_ok=True)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # --- hide z-axis completely ---
    ax.set_zticks([])
    ax.set_zticklabels([])
    ax.zaxis.line.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    # --- plot mycelium segments ---
    max_val = max((s.I * s.length()) for _,_,s in M.all_segments()) if any(True for _ in M.all_segments()) else 1.0
    for _,_,s in M.all_segments():
        x = [s.start[0], s.end[0]]
        y = [s.start[1], s.end[1]]
        z = [s.start[2], s.end[2]]
        cmap_val = (s.I * s.length()) / max_val
        ax.plot(x, y, z, linewidth=1.3, color=plt.cm.viridis(cmap_val))

    # --- plot cuboids (substrate = green, wall = gray transparent) ---
    if cuboids:
        for c in cuboids:
            draw_cuboid(ax, c)

    ax.set_title(title or f"Step {step}")
    ax.view_init(elev=90, azim=-90)  # top-down XY view
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    # ax.set_zlabel("Z [mm]")
    # HIDE Z AXIS
    ax.zaxis.set_visible(False)
    ax.set_box_aspect([1,1,0.1])
    ax.grid(False)

    # draw inoculum origins as red dots
    for pos in INOCULUM_POINTS:
        ax.scatter(pos[0], pos[1], pos[2], color='red', s=30, label='inoculum')

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

    plt.tight_layout()
    plt.savefig(os.path.join(snapshot_dir, f"petri_step_{step:04d}.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def draw_cuboid(ax, cuboid):
    """Draw a translucent box for cuboid boundaries."""
    from itertools import product, combinations
    cx, cy, cz = cuboid.center
    sx, sy, sz = cuboid.size / 2.0
    # 8 corners
    r = np.array([[cx - sx, cx + sx],
                  [cy - sy, cy + sy],
                  [cz - sz, cz + sz]])
    corners = np.array(list(product(r[0], r[1], r[2])))
    # draw edges
    color = 'limegreen' if cuboid.ctype == 'substrate' else 'gray'
    alpha = 0.3 if cuboid.ctype == 'impenetrable' else 0.4
    for s, e in combinations(corners, 2):
        if np.sum(np.abs(s-e) < 1e-8) == 2:
            ax.plot3D(*zip(s, e), color=color, alpha=alpha, linewidth=1)

# -------------------------
# Demo simulation loop
# -------------------------
def run_demo():
    M = initialize_inoculum(points=INOCULUM_POINTS, H0_per_point=H0_PER_POINT)
    cuboids = []

    cuboids.append(Cuboid(
        center=[0.0, 0.0, 0.0],
        size=[DISH_SIZE, SUBSTRATE_WIDTH, HEIGHT],
        ctype='substrate',
        attrs={'E': 2e-6, 'mu': 1e8}
    ))

    cuboids += [
        Cuboid(center=[0, 0, -HEIGHT/2 - WALL_THICKNESS/2],
               size=[DISH_SIZE, DISH_SIZE, WALL_THICKNESS],
               ctype='impenetrable'),
        Cuboid(center=[0, 0, HEIGHT/2 + WALL_THICKNESS/2],
               size=[DISH_SIZE, DISH_SIZE, WALL_THICKNESS],
               ctype='impenetrable')
    ]

    out_dir = os.path.join(RESULTS_DIR, f"sim_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    spatial = SpatialHash(voxel_size=0.1)  # tune voxel size for speed/accuracy
    spatial.rebuild(M)

    # --- NEW: statistics collection ---
    history = []

    for t in range(T_steps):
        translocate_internal_substrate(M, D=D, dt=dt)
        attempt_growth(M, P_branch=P_branch, c_g=c_g, h0=h0)
        spatial.rebuild(M)
        detect_anastomosis(M, spatial, tol=ANASTOMOSIS_TOL)
        uptake_from_cuboids(M, cuboids, dt=dt)
        enforce_impenetrable_boundaries(M, cuboids)

        # record stats
        stats = summarize_mycelium(M)
        stats["step"] = t
        history.append(stats)

        if t % 50 == 0 or t == T_steps - 1:
            print(f"Step {t}: {stats}")
            plot_mycelium(M, t, cuboids=cuboids, title=f"Step {t} (Petri Dish)", show=False, snapshot_dir=snapshot_dir)

    # --- After simulation ---
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(out_dir, "mycelium_growth_stats.csv"), index=False)
    print(f"\nSaved growth statistics to {out_dir}/mycelium_growth_stats.csv")

    plot_growth_summary(df, out_dir)
    export_geometry(M, out_dir)
    print(f"✅ All results saved under {out_dir}")

def export_geometry(M, out_dir):
    """
    Export node and element data for FEA solver.
    Each segment's start/end are stored as nodes with unique IDs.
    Saves into the provided output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    nodes = []
    node_map = {}
    elements = []

    node_counter = 0
    elem_counter = 0

    def add_node(p):
        nonlocal node_counter
        key = tuple(np.round(p, 6))
        if key not in node_map:
            node_map[key] = node_counter
            nodes.append([node_counter, *p])
            node_counter += 1
        return node_map[key]

    for _, _, s in M.all_segments():
        n1 = add_node(s.start)
        n2 = add_node(s.end)
        elements.append([elem_counter, n1, n2])
        elem_counter += 1

    nodes_df = pd.DataFrame(nodes, columns=["node_id", "x", "y", "z"])
    elems_df = pd.DataFrame(elements, columns=["elem_id", "n1", "n2"])

    nodes_df.to_csv(os.path.join(out_dir, "nodes.csv"), index=False)
    elems_df.to_csv(os.path.join(out_dir, "elements.csv"), index=False)
    print(f"✅ Exported geometry to {out_dir}")

def plot_growth_summary(df, out_dir):
    """Plot key growth indicators over time."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # Remove the right/top spines ("z-axis"-looking lines)
    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axes[0].plot(df["step"], df["total_length_mm"], label="Total Hyphal Length")
    axes[0].set_ylabel("Length [mm]")
    axes[0].legend()

    axes[1].plot(df["step"], df["branches"], label="Branches")
    axes[1].plot(df["step"], df["anastomosed"], label="Merges (Anastomoses)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    axes[2].plot(df["step"], df["active_tips"], label="Active Tips")
    axes[2].plot(df["step"], df["passive_tips"], label="Passive Tips")
    axes[2].set_ylabel("Tips")
    axes[2].set_xlabel("Simulation Step")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mycelium_growth_summary.png"), dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    run_demo()
