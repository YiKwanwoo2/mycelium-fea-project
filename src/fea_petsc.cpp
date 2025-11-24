// fea_petsc.cpp
// PETSc-based FEA (3 DOF per node) translated from user's Python solver
//
// Build (example):
// mpicxx -O3 -std=c++17 -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include fea_petsc.cpp -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -o fea_petsc
//
// Run (example):
// mpirun -n 4 ./fea_petsc
//
// Input (expected):
// ../results/nodes.csv    (columns: node_id,x,y,z ...)
// ../results/elements.csv (columns: elem_id,n1,n2 ...)
//
// Outputs written to ../results/fea_results/

#include <petscksp.h>
#include <mpi.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>

struct Node {
    int id;
    double x,y,z;
};

struct Elem {
    int id;
    int n1, n2;
};

static constexpr double E_mod = 2500.0; // MPa
static constexpr double d = 0.0002;     // mm
static constexpr double t = 0.000001;   // mm
static const double A = 3.141592653589793 * ( (d/2.0)*(d/2.0) - (d/2.0 - t)*(d/2.0 - t) );
static const double I = A * 0.001;
static const int N_STEPS = 100;
static const double DISPLACEMENT_MAX = 0.06; // mm
static const double MAX_STRAIN = 0.018;
static const double MAX_STRESS = E_mod * MAX_STRAIN;
static const double GRIP_LENGTH = 0.1; // mm

// utility: trim
static inline std::string trim(const std::string &s) {
    auto a = s.find_first_not_of(" \t\r\n");
    if (a==std::string::npos) return "";
    auto b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<Node> read_nodes(const std::string &path) {
    std::vector<Node> nodes;
    std::ifstream f(path);
    if (!f.is_open()) {
        PetscPrintf(PETSC_COMM_WORLD, "Error: cannot open %s\n", path.c_str());
        return nodes;
    }

    std::string line;
    // read header
    if (!std::getline(f, line)) return nodes;
    std::vector<std::string> heads;
    {
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) heads.push_back(trim(tok));
    }
    // find indices for node_id,x,y,z
    int idx_id=-1, idx_x=-1, idx_y=-1, idx_z=-1;
    for (size_t i=0;i<heads.size();++i){
        std::string h=heads[i];
        if (h == "node_id") idx_id = i;
        if (h == "x") idx_x = i;
        if (h == "y") idx_y = i;
        if (h == "z") idx_z = i;
    }

    while (std::getline(f, line)) {
        if (trim(line).empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ',')) cols.push_back(trim(tok));
        Node n{};
        if (idx_id>=0) n.id = std::stoi(cols[idx_id]);
        else n.id = (int)nodes.size();
        n.x = (idx_x>=0) ? std::stod(cols[idx_x]) : 0.0;
        n.y = (idx_y>=0) ? std::stod(cols[idx_y]) : 0.0;
        n.z = (idx_z>=0) ? std::stod(cols[idx_z]) : 0.0;
        nodes.push_back(n);
    }
    return nodes;
}

std::vector<Elem> read_elems(const std::string &path) {
    std::vector<Elem> elems;
    std::ifstream f(path);
    if (!f.is_open()) {
        PetscPrintf(PETSC_COMM_WORLD, "Error: cannot open %s\n", path.c_str());
        return elems;
    }
    std::string line;
    if (!std::getline(f,line)) return elems;
    std::vector<std::string> heads;
    {
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) heads.push_back(trim(tok));
    }
    int idx_id=-1, idx_n1=-1, idx_n2=-1;
    for (size_t i=0;i<heads.size();++i){
        std::string h=heads[i];
        if (h=="elem_id") idx_id=i;
        if (h=="n1") idx_n1=i;
        if (h=="n2") idx_n2=i;
    }

    while (std::getline(f,line)) {
        if (trim(line).empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss,tok,',')) cols.push_back(trim(tok));
        Elem e{};
        if (idx_id>=0) e.id = std::stoi(cols[idx_id]);
        else e.id = (int)elems.size();
        e.n1 = (idx_n1>=0) ? std::stoi(cols[idx_n1]) : 0;
        e.n2 = (idx_n2>=0) ? std::stoi(cols[idx_n2]) : 0;
        elems.push_back(e);
    }
    return elems;
}

// Create 6x6 element stiffness matrix (axial + "perp" bending) like Python version
void element_stiffness(const double p1[3], const double p2[3], double Ke[6][6], double &Lout) {
    double Lvec[3];
    for (int i=0;i<3;++i) Lvec[i] = p2[i] - p1[i];
    double L = std::sqrt(Lvec[0]*Lvec[0] + Lvec[1]*Lvec[1] + Lvec[2]*Lvec[2]);
    if (L < 1e-12) L = 1e-12;
    Lout = L;
    double n[3];
    for (int i=0;i<3;++i) n[i] = Lvec[i]/L;

    double k_axial = (E_mod * A) / L;

    // Build TT = n * n^T
    double TT[3][3];
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) TT[i][j] = n[i]*n[j];

    // perp = I - TT
    double perp[3][3];
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) perp[i][j] = (i==j?1.0:0.0) - TT[i][j];

    double k_bend_val = 12.0 * E_mod * I / (L*L*L);

    // initialize Ke to zeros
    for (int i=0;i<6;++i) for (int j=0;j<6;++j) Ke[i][j] = 0.0;

    // axial contribution into blocks
    for (int ia=0; ia<3; ++ia) {
        for (int ja=0; ja<3; ++ja) {
            double v = TT[ia][ja] * k_axial;
            Ke[ia][ja] += v;
            Ke[ia][ja+3] += -v;
            Ke[ia+3][ja] += -v;
            Ke[ia+3][ja+3] += v;
        }
    }

    // bending contribution similarly with perp
    for (int ia=0; ia<3; ++ia) {
        for (int ja=0; ja<3; ++ja) {
            double v = perp[ia][ja] * k_bend_val;
            Ke[ia][ja] += v;
            Ke[ia][ja+3] += -v;
            Ke[ia+3][ja] += -v;
            Ke[ia+3][ja+3] += v;
        }
    }
}

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    MPI_Comm comm = PETSC_COMM_WORLD;

    PetscPrintf(comm, "🔧 Running C++ PETSc FEA (nodes/elements expected in ../results)\n");

    std::string nodes_path = "../results/nodes.csv";
    std::string elems_path = "../results/elements.csv";
    std::string out_dir = "../results/fea_results";

    // read nodes/elements
    auto nodes = read_nodes(nodes_path);
    auto elems = read_elems(elems_path);

    if (nodes.empty() || elems.empty()) {
        PetscPrintf(comm, "Error: nodes or elements empty. Exiting.\n");
        PetscFinalize();
        return 1;
    }

    int n_nodes = (int)nodes.size();
    int n_elems = (int)elems.size();
    int n_dof = 3 * n_nodes;

    // coords array for convenience
    std::vector<std::array<double,3>> coords(n_nodes);
    for (int i=0;i<n_nodes;++i) {
        coords[i] = { nodes[i].x, nodes[i].y, nodes[i].z };
    }

    // active flags
    std::vector<char> active(n_elems, 1);

    // find y min/max for grips
    double y_min = coords[0][1], y_max = coords[0][1];
    for (int i=1;i<n_nodes;++i) { y_min = std::min(y_min, coords[i][1]); y_max = std::max(y_max, coords[i][1]); }

    // collect node ids (assumes node_id in CSV correspond to array index; if not, user should ensure)
    std::vector<int> top_nodes, bot_nodes;
    for (int i=0;i<n_nodes;++i) {
        if (std::abs(coords[i][1] - y_max) < GRIP_LENGTH) top_nodes.push_back(i);
        if (std::abs(coords[i][1] - y_min) < GRIP_LENGTH) bot_nodes.push_back(i);
    }
    PetscPrintf(comm, "Top nodes: %d, Bottom nodes: %d\n", (int)top_nodes.size(), (int)bot_nodes.size());

    // storage for time history
    std::vector< std::vector<double> > stress_record; stress_record.reserve(N_STEPS);
    std::vector< std::vector<int> > active_record; active_record.reserve(N_STEPS);
    std::vector< std::vector<double> > disp_record; disp_record.reserve(N_STEPS);
    std::vector<std::pair<double,double>> force_disp_curve; force_disp_curve.reserve(N_STEPS);

    // Prepare PETSc matrix (we'll recreate per step)
    for (int step=0; step<N_STEPS; ++step) {
        double disp_factor = (double)step / (double)(N_STEPS - 1);
        double dy_top = +DISPLACEMENT_MAX * disp_factor;
        double dy_bot = -DISPLACEMENT_MAX * disp_factor;
        PetscPrintf(comm, "➡️  Step %d/%d | dy_top=%.6f, dy_bot=%.6f\n", step+1, N_STEPS, dy_top, dy_bot);

        // Create MAT (AIJ)
        Mat K;
        MatCreate(comm, &K);
        MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, n_dof, n_dof);
        MatSetFromOptions(K);
        MatSetUp(K);

        // Assemble element contributions
        // For each active element compute Ke and insert into global K
        for (int e=0;e<n_elems;++e) {
            if (!active[e]) continue;
            int n1 = elems[e].n1;
            int n2 = elems[e].n2;
            double p1[3] = { coords[n1][0], coords[n1][1], coords[n1][2] };
            double p2[3] = { coords[n2][0], coords[n2][1], coords[n2][2] };
            double Ke[6][6];
            double L;
            element_stiffness(p1,p2,Ke,L);

            // global dof indices
            PetscInt gdof[6];
            for (int i=0;i<3;++i) gdof[i]   = 3*n1 + i;
            for (int i=0;i<3;++i) gdof[i+3] = 3*n2 + i;

            // Insert values
            for (int i=0;i<6;++i) {
                for (int j=0;j<6;++j) {
                    MatSetValue(K, gdof[i], gdof[j], Ke[i][j], ADD_VALUES);
                }
            }
        }

        MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

        // Save a copy of original stiffness for reactions (K_full)
        Mat K_full;
        MatDuplicate(K, MAT_COPY_VALUES, &K_full);

        // Build RHS (initially zeros)
        Vec b;
        VecCreate(comm, &b);
        VecSetSizes(b, PETSC_DECIDE, n_dof);
        VecSetFromOptions(b);
        VecSet(b, 0.0);

        // Build known DOFs mapping
        std::vector<PetscInt> known_dofs;
        std::vector<double> known_vals;
        known_dofs.reserve((top_nodes.size()+bot_nodes.size())*3);
        known_vals.reserve((top_nodes.size()+bot_nodes.size())*3);

        for (int n : top_nodes) {
            known_dofs.push_back(3*n + 0); known_vals.push_back(0.0);
            known_dofs.push_back(3*n + 1); known_vals.push_back(dy_top);
            known_dofs.push_back(3*n + 2); known_vals.push_back(0.0);
        }
        for (int n : bot_nodes) {
            known_dofs.push_back(3*n + 0); known_vals.push_back(0.0);
            known_dofs.push_back(3*n + 1); known_vals.push_back(dy_bot);
            known_dofs.push_back(3*n + 2); known_vals.push_back(0.0);
        }

        // Create PETSc Vec x_known containing known displacement values (for MatZeroRowsColumns)
        Vec x_known;
        VecCreate(comm, &x_known);
        VecSetSizes(x_known, PETSC_DECIDE, n_dof);
        VecSetFromOptions(x_known);
        VecSet(x_known, 0.0);
        for (size_t i=0;i<known_dofs.size();++i) {
            VecSetValue(x_known, known_dofs[i], known_vals[i], INSERT_VALUES);
        }
        VecAssemblyBegin(x_known); VecAssemblyEnd(x_known);

        // Set b entries to zero initially then MatZeroRowsColumns will adjust b if x_known provided
        // But we also want the solver to produce the prescribed displacements; call MatZeroRowsColumns on a duplicate K_mod
        Mat K_mod;
        MatDuplicate(K, MAT_COPY_VALUES, &K_mod);

        // Zero rows/cols and set diag to 1.0. Provide x_known and b so PETSc adjusts RHS properly.
        PetscInt nknown = (PetscInt)known_dofs.size();
        std::vector<PetscInt> known_arr = known_dofs;
        MatZeroRowsColumns(K_mod, nknown, known_arr.data(), 1.0, x_known, b);

        // Build linear solver (KSP)
        KSP ksp;
        KSPCreate(comm, &ksp);
        KSPSetOperators(ksp, K_mod, K_mod);
        KSPSetFromOptions(ksp);
        KSPSetUp(ksp);

        // Solve K_mod * U = b
        Vec U;
        VecDuplicate(b, &U);
        KSPSolve(ksp, b, U);

        // Reaction forces: F_react = K_full * U
        Vec F_react;
        VecDuplicate(b, &F_react);
        MatMult(K_full, U, F_react);

        // Extract top reactions and sum
        double total_force = 0.0;
        for (int n : top_nodes) {
            PetscScalar val;
            VecGetValues(F_react, 1, (PetscInt[]){3*n + 1}, &val);
            total_force += val;
        }
        double total_disp = dy_top - dy_bot;
        force_disp_curve.emplace_back(total_disp, total_force);

        // Extract U to host array
        std::vector<double> Uhost(n_dof, 0.0);
        for (int i=0;i<n_dof;++i) {
            PetscScalar val;
            VecGetValues(U, 1, (PetscInt[]){i}, &val);
            Uhost[i] = val;
        }

        // compute per-element stress and deactivate if needed
        std::vector<double> stress(n_elems, 0.0);
        for (int e=0;e<n_elems;++e) {
            if (!active[e]) continue;
            int n1 = elems[e].n1;
            int n2 = elems[e].n2;
            double p1[3] = { coords[n1][0], coords[n1][1], coords[n1][2] };
            double p2[3] = { coords[n2][0], coords[n2][1], coords[n2][2] };
            double Lvec[3] = { p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2] };
            double L = std::sqrt(Lvec[0]*Lvec[0] + Lvec[1]*Lvec[1] + Lvec[2]*Lvec[2]);
            if (L < 1e-12) L = 1e-12;
            double nvec[3] = { Lvec[0]/L, Lvec[1]/L, Lvec[2]/L };
            double u1[3] = { Uhost[3*n1+0], Uhost[3*n1+1], Uhost[3*n1+2] };
            double u2[3] = { Uhost[3*n2+0], Uhost[3*n2+1], Uhost[3*n2+2] };
            double du[3] = { u2[0]-u1[0], u2[1]-u1[1], u2[2]-u1[2] };
            double strain = nvec[0]*du[0] + nvec[1]*du[1] + nvec[2]*du[2];
            strain /= L;
            double s_ax = E_mod * strain;
            stress[e] = s_ax;
            if (std::abs(strain) > MAX_STRAIN) {
                active[e] = 0; // deactivate
            }
        }

        // Save step history in memory
        stress_record.emplace_back(stress.begin(), stress.end());
        active_record.emplace_back();
        active_record.back().reserve(n_elems);
        for (int e=0;e<n_elems;++e) active_record.back().push_back(active[e] ? 1 : 0);

        disp_record.emplace_back(Uhost.begin(), Uhost.end());

        // Clean up PETSc for this step
        VecDestroy(&U);
        VecDestroy(&F_react);
        VecDestroy(&b);
        VecDestroy(&x_known);
        MatDestroy(&K_full);
        MatDestroy(&K_mod);
        MatDestroy(&K);
        KSPDestroy(&ksp);

        // print early stop if nothing active
        int n_active = 0;
        for (int e=0;e<n_elems;++e) if (active[e]) ++n_active;
        if (n_active == 0) {
            PetscPrintf(comm, "⚠️  Simulation stopped early at step %d (no active elements)\n", step+1);
            break;
        }
    } // end step loop

    // Write outputs (CSV) to ../results/fea_results
    // Create directory via system call (avoid filesystem). This is a portability hack:
    system(("mkdir -p " + out_dir).c_str());

    // stress_record.csv
    {
        std::ofstream fs(out_dir + "/stress_record.csv");
        // header
        for (int e=0;e<n_elems;++e) {
            fs << "elem_" << e << ",";
        }
        fs << "step\n";
        for (size_t s=0; s<stress_record.size(); ++s) {
            for (int e=0;e<n_elems;++e) {
                fs << stress_record[s][e] << ",";
            }
            fs << (s+1) << "\n";
        }
        fs.close();
    }

    // active_elements.csv
    {
        std::ofstream fa(out_dir + "/active_elements.csv");
        for (int e=0;e<n_elems;++e) fa << "elem_" << e << ",";
        fa << "step\n";
        for (size_t s=0; s<active_record.size(); ++s) {
            for (int e=0;e<n_elems;++e) fa << active_record[s][e] << ",";
            fa << (s+1) << "\n";
        }
        fa.close();
    }

    // node_displacements.csv
    {
        std::ofstream fd(out_dir + "/node_displacements.csv");
        // write header: node_0_x,... node_{n-1}_x, node_0_y,..., node_{n-1}_y, node_0_z...
        for (int i=0;i<n_nodes;++i) fd << "node_"<<i<<"_x,";
        for (int i=0;i<n_nodes;++i) fd << "node_"<<i<<"_y,";
        for (int i=0;i<n_nodes;++i) fd << "node_"<<i<<"_z,";
        fd << "step\n";

        for (size_t s=0; s<disp_record.size(); ++s) {
            // disp_record[s] is Uhost for this step of length n_dof (all dofs)
            // write x for nodes 0..n-1
            for (int i=0;i<n_nodes;++i) fd << disp_record[s][3*i + 0] << ",";
            for (int i=0;i<n_nodes;++i) fd << disp_record[s][3*i + 1] << ",";
            for (int i=0;i<n_nodes;++i) fd << disp_record[s][3*i + 2] << ",";
            fd << (s+1) << "\n";
        }
        fd.close();
    }

    // force_displacement.csv
    {
        std::ofstream ff(out_dir + "/force_displacement.csv");
        ff << "total_displacement,total_force\n";
        for (auto &p : force_disp_curve) {
            ff << p.first << "," << p.second << "\n";
        }
        ff.close();
    }

    // Write runtime (simple)
    {
        std::ofstream fr(out_dir + "/runtime.txt");
        fr << "FEA run completed. Steps written: " << stress_record.size() << "\n";
        fr.close();
    }

    PetscPrintf(comm, "✅ FEA completed. Results saved to %s\n", out_dir.c_str());

    PetscFinalize();
    return 0;
}
