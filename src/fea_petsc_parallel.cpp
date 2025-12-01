// fea_petsc.cpp
// PETSc-based translation of the provided Python FEA solver.
// Compile with mpicxx and link against PETSc. Example compile shown after the code.

#include <petscksp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <array>
#include <chrono>

// ----------------------------
// Material & Simulation Parameters (kept same as Python)
// ----------------------------
static const double E_mod = 2500.0;          // MPa
static const double d_param = 0.0002;        // mm
static const double t_param = 0.000001;      // mm
static const double A_param = 3.14 * ( std::pow(d_param/2.0,2) - std::pow(d_param/2.0 - t_param,2) ); // mm^2
static const double I_param = A_param * 0.001; // mm^4 (as in Python)
static const int N_STEPS = 40;
static const double DISPLACEMENT_MAX = 0.02; // mm
static const double MAX_STRAIN = 0.018;
static const double MAX_STRESS = E_mod * MAX_STRAIN;
static const double GRIP_LENGTH = 1.5;       // mm

// ----------------------------
// Utility CSV readers
// nodes.csv expected columns: node_id,x,y,z  (node_id numeric but we will index by row order)
// elements.csv expected columns: elem_id,n1,n2
// ----------------------------
struct Node { int id; double x,y,z; };
struct Elem { int id; int n1, n2; };

static std::vector<Node> read_nodes_csv(const std::string &path) {
    std::vector<Node> nodes;
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Failed to open nodes file: " + path);
    std::string line;
    // read header
    std::getline(ifs, line);
    while (std::getline(ifs,line)) {
        if (line.size()==0) continue;
        std::stringstream ss(line);
        std::string token;
        Node node;
        // assume order: node_id,x,y,z (or at least first 4 columns)
        std::getline(ss, token, ','); node.id = std::stoi(token);
        std::getline(ss, token, ','); node.x = std::stod(token);
        std::getline(ss, token, ','); node.y = std::stod(token);
        std::getline(ss, token, ','); node.z = std::stod(token);
        nodes.push_back(node);
    }
    return nodes;
}

static std::vector<Elem> read_elems_csv(const std::string &path) {
    std::vector<Elem> elems;
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Failed to open elements file: " + path);
    std::string line;
    std::getline(ifs,line);
    while (std::getline(ifs,line)) {
        if (line.size()==0) continue;
        std::stringstream ss(line);
        std::string token;
        Elem e;
        // assume order: elem_id,n1,n2
        std::getline(ss, token, ','); e.id = std::stoi(token);
        std::getline(ss, token, ','); e.n1 = std::stoi(token);
        std::getline(ss, token, ','); e.n2 = std::stoi(token);
        elems.push_back(e);
    }
    return elems;
}

// ----------------------------
// Compute element stiffness matrix (6x6) for a bar-like element with axial + a simple 'perp' bending term.
// Mirrors Python's bar_stiffness_bulk element-by-element.
// ----------------------------
static void element_stiffness_6x6(const double p1[3], const double p2[3], double Ke[6][6], double &Lout) {
    double vx = p2[0] - p1[0];
    double vy = p2[1] - p1[1];
    double vz = p2[2] - p1[2];
    double L = std::sqrt(vx*vx + vy*vy + vz*vz);
    if (L < 1e-12) L = 1e-12;
    Lout = L;
    double nx = vx / L;
    double ny = vy / L;
    double nz = vz / L;

    // Build outer product TT = n * n^T (3x3)
    double TT[3][3];
    TT[0][0] = nx*nx; TT[0][1] = nx*ny; TT[0][2] = nx*nz;
    TT[1][0] = ny*nx; TT[1][1] = ny*ny; TT[1][2] = ny*nz;
    TT[2][0] = nz*nx; TT[2][1] = nz*ny; TT[2][2] = nz*nz;

    // perp = I - TT
    double perp[3][3];
    for (int i=0;i<3;i++){
        for (int j=0;j<3;j++){
            perp[i][j] = (i==j?1.0:0.0) - TT[i][j];
        }
    }

    double k_axial = (E_mod * A_param) / L;           // scalar
    double k_bend  = (12.0 * E_mod * I_param) / (L*L*L);

    // Initialize Ke to zero
    for (int i=0;i<6;i++) for (int j=0;j<6;j++) Ke[i][j] = 0.0;

    // Fill axial contribution: blocks (0:3,0:3)=TT, (0:3,3:6)=-TT, etc.
    for (int i=0;i<3;i++){
        for (int j=0;j<3;j++){
            double val = TT[i][j] * k_axial;
            Ke[i][j] += val;
            Ke[i][j+3] += -val;
            Ke[i+3][j] += -val;
            Ke[i+3][j+3] += val;
        }
    }

    // Fill bending-like contribution using perp with same block pattern
    for (int i=0;i<3;i++){
        for (int j=0;j<3;j++){
            double val = perp[i][j] * k_bend;
            Ke[i][j] += val;
            Ke[i][j+3] += -val;
            Ke[i+3][j] += -val;
            Ke[i+3][j+3] += val;
        }
    }
}

// ----------------------------
// Make directory if not exists (POSIX)
// ----------------------------
static void make_dir_if_needed(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        // try to create
        mkdir(path.c_str(), 0755);
    }
}

// ----------------------------
// Main FEA solver
// ----------------------------
int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, (char*)NULL, (char*)NULL); CHKERRQ(ierr);

    if (argc < 2) {
        PetscPrintf(PETSC_COMM_WORLD, "Usage: %s <results_dir>\n", argv[0]);
        PetscFinalize();
        return 1;
    }
    std::string results_dir = argv[1];
    std::string fea_dir = results_dir + "/fea_results";
    make_dir_if_needed(fea_dir);

    int rank = 0, size = 1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);


    PetscPrintf(PETSC_COMM_WORLD, "🔧 Running FEA on geometry from %s\n", results_dir.c_str());

    auto start = std::chrono::high_resolution_clock::now();

    // Read nodes/elements
    std::string nodes_file = results_dir + "/nodes.csv";
    std::string elems_file = results_dir + "/elements.csv";
    std::vector<Node> nodes;
    std::vector<Elem> elems;
    try {
        nodes = read_nodes_csv(nodes_file);
        elems = read_elems_csv(elems_file);
    } catch (const std::exception &ex) {
        PetscPrintf(PETSC_COMM_WORLD, "Error reading input CSVs: %s\n", ex.what());
        PetscFinalize();
        return 1;
    }

    const int n_nodes = (int)nodes.size();
    const int n_elems = (int)elems.size();
    const int n_dof = 3 * n_nodes;

    // Build coordinate array
    std::vector<std::array<double,3>> coords(n_nodes);
    for (int i=0;i<n_nodes;i++){
        coords[i][0] = nodes[i].x;
        coords[i][1] = nodes[i].y;
        coords[i][2] = nodes[i].z;
    }

    // Active flags
    std::vector<char> active(n_elems, 1);

    // find y min/max and top/bot nodes within GRIP_LENGTH tolerance
    double y_min = coords[0][1], y_max = coords[0][1];
    for (int i=1;i<n_nodes;i++){
        y_min = std::min(y_min, coords[i][1]);
        y_max = std::max(y_max, coords[i][1]);
    }
    std::vector<int> top_nodes, bot_nodes;
    for (int i=0;i<n_nodes;i++){
        if (std::fabs(coords[i][1] - y_max) < GRIP_LENGTH) top_nodes.push_back(i);
        if (std::fabs(coords[i][1] - y_min) < GRIP_LENGTH) bot_nodes.push_back(i);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Top nodes: %zu, Bottom nodes: %zu\n", top_nodes.size(), bot_nodes.size());

    // Records for outputs
    std::vector<std::vector<double>> stress_record; // each step: n_elems values
    std::vector<std::vector<int>> active_record;    // each step: n_elems ints
    std::vector<std::vector<double>> disp_record;   // each step: n_dof values
    std::vector<std::array<double,2>> fd_curve;     // total_disp, total_force

    // Main step loop
    for (int step=0; step < N_STEPS; ++step) {
        double disp_factor = (double)step / (double)(N_STEPS - 1);
        double dy_top = +DISPLACEMENT_MAX * disp_factor;
        double dy_bot = -DISPLACEMENT_MAX * disp_factor;
        PetscPrintf(PETSC_COMM_WORLD, "➡️  Step %d/%d | dy_top=%.6f, dy_bot=%.6f\n", step+1, N_STEPS, dy_top, dy_bot);

        // Assemble global stiffness matrix K (PETSc Mat)
        Mat K;
        ierr = MatCreate(PETSC_COMM_WORLD, &K); CHKERRQ(ierr);
        ierr = MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, n_dof, n_dof); CHKERRQ(ierr);
        ierr = MatSetFromOptions(K); CHKERRQ(ierr);
        ierr = MatSetUp(K); CHKERRQ(ierr);

        // Pre-estimate nonzero per row: each DOF connects to small number - leave default or set small
        // Insert element contributions
        for (int e=0;e<n_elems;e++){
            if (!active[e]) continue;
            int n1 = elems[e].n1;
            int n2 = elems[e].n2;
            if (n1 < 0 || n1 >= n_nodes || n2 < 0 || n2 >= n_nodes) continue;

            double p1[3] = { coords[n1][0], coords[n1][1], coords[n1][2] };
            double p2[3] = { coords[n2][0], coords[n2][1], coords[n2][2] };
            double Ke[6][6];
            double L;
            element_stiffness_6x6(p1, p2, Ke, L);

            // Map local to global DOF indices
            PetscInt dof_map[6];
            for (int k=0;k<3;k++) { dof_map[k] = 3*n1 + k; dof_map[k+3] = 3*n2 + k; }

            // Insert into K
            for (int i=0;i<6;i++){
                for (int j=0;j<6;j++){
                    double val = Ke[i][j];
                    ierr = MatSetValue(K, dof_map[i], dof_map[j], val, ADD_VALUES); CHKERRQ(ierr);
                }
            }
        }

        ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        // Keep a copy of original K for reaction computation later
        Mat K_orig;
        ierr = MatDuplicate(K, MAT_COPY_VALUES, &K_orig); CHKERRQ(ierr);

        // Build known DOFs map
        std::vector<PetscInt> known_dofs;
        std::vector<double> known_vals;
        // Top nodes
        for (int n : top_nodes) {
            known_dofs.push_back(3*n + 0); known_vals.push_back(0.0);
            known_dofs.push_back(3*n + 1); known_vals.push_back(dy_top);
            known_dofs.push_back(3*n + 2); known_vals.push_back(0.0);
        }
        // Bottom nodes
        for (int n : bot_nodes) {
            known_dofs.push_back(3*n + 0); known_vals.push_back(0.0);
            known_dofs.push_back(3*n + 1); known_vals.push_back(dy_bot);
            known_dofs.push_back(3*n + 2); known_vals.push_back(0.0);
        }

        // Make vector x with prescribed values at known DOFs for MatZeroRowsColumns
        Vec x_prescribed, bvec;
        ierr = VecCreate(PETSC_COMM_WORLD, &x_prescribed); CHKERRQ(ierr);
        ierr = VecSetSizes(x_prescribed, PETSC_DECIDE, n_dof); CHKERRQ(ierr);
        ierr = VecSetFromOptions(x_prescribed); CHKERRQ(ierr);
        ierr = VecSet(x_prescribed, 0.0); CHKERRQ(ierr);

        for (size_t i=0;i<known_dofs.size();++i) {
            ierr = VecSetValue(x_prescribed, known_dofs[i], known_vals[i], INSERT_VALUES); CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(x_prescribed); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(x_prescribed); CHKERRQ(ierr);

        // Prepare RHS vector b (initially zero)
        ierr = VecCreate(PETSC_COMM_WORLD, &bvec); CHKERRQ(ierr);
        ierr = VecSetSizes(bvec, PETSC_DECIDE, n_dof); CHKERRQ(ierr);
        ierr = VecSetFromOptions(bvec); CHKERRQ(ierr);
        ierr = VecSet(bvec, 0.0); CHKERRQ(ierr);

        // Create IS for known dofs if any; else solve full system
        IS is_known = NULL;
        if (!known_dofs.empty()) {
            ierr = ISCreateGeneral(PETSC_COMM_WORLD, (PetscInt)known_dofs.size(), known_dofs.data(), PETSC_COPY_VALUES, &is_known); CHKERRQ(ierr);
            // Zero rows/cols of K and set diag=1.0 and adjust RHS with prescribed values via x_prescribed
            ierr = MatZeroRowsColumnsIS(K, is_known, 1.0, x_prescribed, bvec); CHKERRQ(ierr);
            // Note: MatZeroRowsColumns sets b[row] = diag * x[row] when x given.
        }

        // Regularize tiny diagonal if needed (not necessary usually, but mirror python's tiny identity)
        // Add very small to diagonal
        for (int i=0;i<n_dof;i++) {
            double tiny = 1e-12;
            ierr = MatSetValue(K, i, i, tiny, ADD_VALUES); CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        // -----------------------------
        // Solve K * U = b (MPI-safe)
        // -----------------------------
        KSP ksp;
        Vec U;
        ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp, K, K); CHKERRQ(ierr);

        // Solver + PC types
        ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
        PC pc;
        ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
        ierr = PCSetType(pc, PCBJACOBI); CHKERRQ(ierr);  // Parallel-friendly

        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
        ierr = KSPSetUp(ksp); CHKERRQ(ierr);

        // Create solution vector
        ierr = VecCreate(PETSC_COMM_WORLD, &U); CHKERRQ(ierr);
        ierr = VecSetSizes(U, PETSC_DECIDE, n_dof); CHKERRQ(ierr);
        ierr = VecSetFromOptions(U); CHKERRQ(ierr);
        ierr = VecSet(U, 0.0); CHKERRQ(ierr);

        // Solve
        ierr = KSPSolve(ksp, bvec, U); CHKERRQ(ierr);

        // Check convergence
        KSPConvergedReason reason;
        ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
        if (reason < 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                "❌ Solver failed to converge at step %d. Reason %d\n", step+1, reason);
            if (is_known) ISDestroy(&is_known);
            KSPDestroy(&ksp);
            MatDestroy(&K); MatDestroy(&K_orig);
            VecDestroy(&x_prescribed); VecDestroy(&bvec); VecDestroy(&U);
            break;
        }
        PetscPrintf(PETSC_COMM_WORLD, "KSP converged reason: %d\n", reason);
        KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);

        // --------------------------------------------------------------------
        // Gather solution vector U to rank 0, then broadcast Uvals to all ranks
        // so the subsequent stress computation can use Uvals on every rank.
        // --------------------------------------------------------------------

        // Create gather: full vector on rank 0 (U_local)
        Vec U_local;
        VecScatter scatter;
        ierr = VecScatterCreateToZero(U, &scatter, &U_local); CHKERRQ(ierr);
        ierr = VecScatterBegin(scatter, U, U_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
        ierr = VecScatterEnd(scatter, U, U_local, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

        // Prepare Uvals on all ranks (will be filled on rank 0 then broadcast)
        std::vector<double> Uvals(n_dof, 0.0);

        if (rank == 0) {
            const PetscScalar *Uarr;
            ierr = VecGetArrayRead(U_local, &Uarr); CHKERRQ(ierr);
            for (int i = 0; i < n_dof; ++i) Uvals[i] = (double)Uarr[i];
            ierr = VecRestoreArrayRead(U_local, &Uarr); CHKERRQ(ierr);
        }

        // Broadcast Uvals from rank 0 to all ranks
        MPI_Bcast(Uvals.data(), n_dof, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

        // We no longer need U_local or scatter after Uvals broadcast
        ierr = VecScatterDestroy(&scatter); CHKERRQ(ierr);
        ierr = VecDestroy(&U_local); CHKERRQ(ierr);

        // --------------------------------------------------------------------
        // Compute reactions: local multiplication MatMult(K_orig, U) -> gather to rank 0
        // --------------------------------------------------------------------
        Vec F_react_local;
        ierr = VecDuplicate(U, &F_react_local); CHKERRQ(ierr);
        ierr = MatMult(K_orig, U, F_react_local); CHKERRQ(ierr);

        // Gather F_react_local to rank 0
        Vec F_react_gather;
        VecScatter scatter2;
        ierr = VecScatterCreateToZero(F_react_local, &scatter2, &F_react_gather); CHKERRQ(ierr);
        ierr = VecScatterBegin(scatter2, F_react_local, F_react_gather, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
        ierr = VecScatterEnd(scatter2, F_react_local, F_react_gather, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

        // On rank 0 compute total reaction on top nodes and save fd point + store displacements
        if (rank == 0) {
            const PetscScalar *Farr;
            ierr = VecGetArrayRead(F_react_gather, &Farr); CHKERRQ(ierr);

            double total_force = 0.0;
            for (int n : top_nodes) {
                int idx = 3*n + 1; // y-DOF
                total_force += (double)Farr[idx];
            }
            double total_disp = dy_top - dy_bot;
            fd_curve.push_back({ total_disp, total_force });

            ierr = VecRestoreArrayRead(F_react_gather, &Farr); CHKERRQ(ierr);

            // Store displacement history on rank 0 (we already have Uvals)
            disp_record.push_back(Uvals);
        }

        // Cleanup reaction gather objects
        ierr = VecDestroy(&F_react_local); CHKERRQ(ierr);
        ierr = VecScatterDestroy(&scatter2); CHKERRQ(ierr);
        ierr = VecDestroy(&F_react_gather); CHKERRQ(ierr);

        // Destroy KSP & local U (we still have Uvals for postprocessing)
        ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
        ierr = VecDestroy(&U); CHKERRQ(ierr);

        // Compute stresses and deactivate elements that exceed MAX_STRAIN
        std::vector<double> stress(n_elems, 0.0);
        for (int e=0;e<n_elems;e++){
            if (!active[e]) continue;
            int n1 = elems[e].n1;
            int n2 = elems[e].n2;
            double p1[3] = { coords[n1][0], coords[n1][1], coords[n1][2] };
            double p2[3] = { coords[n2][0], coords[n2][1], coords[n2][2] };
            double Lvec[3] = { p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2] };
            double L = std::sqrt(Lvec[0]*Lvec[0] + Lvec[1]*Lvec[1] + Lvec[2]*Lvec[2]);
            if (L < 1e-12) L = 1e-12;
            double nvec[3] = { Lvec[0]/L, Lvec[1]/L, Lvec[2]/L };

            double u1[3], u2[3];
            for (int k=0;k<3;k++) { u1[k] = Uvals[3*n1 + k]; u2[k] = Uvals[3*n2 + k]; }
            double du[3] = { u2[0]-u1[0], u2[1]-u1[1], u2[2]-u1[2] };
            double strain = nvec[0]*du[0] + nvec[1]*du[1] + nvec[2]*du[2];
            strain /= L;
            double stress_ax = E_mod * strain;
            stress[e] = stress_ax;
            if (std::fabs(strain) > MAX_STRAIN) active[e] = 0;
        }

        // Save records for this step
        stress_record.emplace_back(stress);
        std::vector<int> active_int(n_elems);
        for (int i=0;i<n_elems;i++) active_int[i] = active[i] ? 1 : 0;
        active_record.push_back(active_int);

        // Clean up PETSc objects for this step
        if (is_known) { ISDestroy(&is_known); is_known = NULL; }
        MatDestroy(&K_orig);
        MatDestroy(&K);
        VecDestroy(&x_prescribed);
        VecDestroy(&bvec);
        // VecDestroy(&F_react);
        KSPDestroy(&ksp);
        VecDestroy(&U);

        // stop early if no active elements remain
        bool any_active = false;
        for (int i=0;i<n_elems;i++) if (active[i]) { any_active = true; break; }
        if (!any_active) {
            PetscPrintf(PETSC_COMM_WORLD, "⚠️  Simulation stopped early at step %d.\n", step+1);
            break;
        }
    } // end steps

    // ---- Save CSV outputs ----
    // stress_record: rows = steps, cols = n_elems
    if (!stress_record.empty()) {
        std::string stress_csv = fea_dir + "/stress_record.csv";
        std::ofstream sf(stress_csv);
        // header
        for (int e=0;e<n_elems;e++){
            if (e) sf << ",";
            sf << "elem_" << e;
        }
        sf << ",step\n";

        for (size_t s=0;s<stress_record.size();++s){
            const auto &row = stress_record[s];
            for (int e=0;e<n_elems;e++){
                if (e) sf << ",";
                sf << std::setprecision(12) << row[e];
            }
            sf << "," << (s+1) << "\n";
        }
        sf.close();
    }

    // active elements
    if (!active_record.empty()) {
        std::string active_csv = fea_dir + "/active_elements.csv";
        std::ofstream af(active_csv);
        for (int e=0;e<n_elems;e++){
            if (e) af << ",";
            af << "elem_" << e;
        }
        af << ",step\n";
        for (size_t s=0;s<active_record.size();++s){
            const auto &row = active_record[s];
            for (int e=0;e<n_elems;e++){
                if (e) af << ",";
                af << row[e];
            }
            af << "," << (s+1) << "\n";
        }
        af.close();
    }

    // displacements
    if (!disp_record.empty()) {
        std::string disp_csv = fea_dir + "/node_displacements.csv";
        std::ofstream df(disp_csv);
        // header: node_0_x,... node_n_z, step
        for (int i=0;i<n_nodes;i++){
            if (i) df << ",";
            df << "node_" << i << "_x";
        }
        for (int i=0;i<n_nodes;i++){
            df << "," << "node_" << i << "_y";
        }
        for (int i=0;i<n_nodes;i++){
            df << "," << "node_" << i << "_z";
        }
        df << ",step\n";

        for (size_t s=0;s<disp_record.size();++s){
            // disp_record[s] is length n_dof arranged as [node0_x,node0_y,node0_z,node1_x,...]
            // But your Python output arranged differently: they used a concatenation then wrote columns differently.
            // Here we'll output the same ordering we stored: 0..n_dof-1
            const auto &row = disp_record[s];
            for (int i=0;i<n_dof;i++){
                if (i) df << ",";
                df << std::setprecision(12) << row[i];
            }
            df << "," << (s+1) << "\n";
        }
        df.close();
    }

    // force-displacement curve
    if (!fd_curve.empty()) {
        std::string fd_csv = fea_dir + "/force_displacement.csv";
        std::ofstream ff(fd_csv);
        ff << "total_displacement,total_force\n";
        for (const auto &p : fd_curve) {
            ff << std::setprecision(12) << p[0] << "," << p[1] << "\n";
        }
        ff.close();
    }

    // runtime output: not measuring exact time here to keep code simple; user can measure externally
    std::string runtime_txt = fea_dir + "/runtime.txt";
    std::ofstream rt(runtime_txt);
    rt << "FEA run finished (no timing collected inside C++ version).\n";
    rt.close();

    PetscPrintf(PETSC_COMM_WORLD, "✅ FEA completed. Results saved to %s\n", fea_dir.c_str());

        // Get the ending time point
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print the duration
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

    ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}
