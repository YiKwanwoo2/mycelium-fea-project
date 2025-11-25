// fea_petsc_fixed.cpp
// PETSc-based FEA of rod/beam network (3 DOF per node).
// Updated to use CHKERRQ for PETSc calls and keep C++ try/catch for non-PETSc errors.

#include <petscksp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <sys/stat.h>   // for mkdir
#include <sys/types.h>
#include <iomanip>
#include <unordered_map>

using std::string;
using std::vector;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -----------------------
// Parameters (match Python behavior)
static const double E_mod = 2500.0;        // MPa
static const double d = 0.0002;            // mm
static const double t_shell = 0.000001;    // mm
static const double A_area = 3.14 * ((d/2.0)*(d/2.0) - (d/2.0 - t_shell)*(d/2.0 - t_shell)); // mm^2
static const double I_moment = A_area * 0.001; // mm^4 (approx)
static const int N_STEPS = 40;                     // match Python default
static const double DISPLACEMENT_MAX = 0.02; // mm
static const double MAX_STRAIN = 0.018;
static const double MAX_STRESS = E_mod * MAX_STRAIN;
static const double GRIP_LENGTH = 1.5; // mm (match Python)

// -----------------------
// Data containers
struct Node {
    int id;      // original id from CSV
    double x,y,z;
};

struct Elem {
    int id;
    int n1, n2;   // stored as node INDICES (after conversion)
    bool active;
};

// -----------------------
// CSV helpers
static void trim_inplace(std::string &s){
    const char *ws = " \t\n\r";
    auto p0 = s.find_first_not_of(ws);
    if(p0==string::npos){ s.clear(); return; }
    auto p1 = s.find_last_not_of(ws);
    s = s.substr(p0, p1-p0+1);
}

vector<Node> read_nodes_csv(const string &path){
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("Cannot open nodes file: "+path);
    string line;
    vector<Node> nodes;
    // header
    std::getline(ifs, line);
    while(std::getline(ifs,line)){
        trim_inplace(line);
        if(line.empty()) continue;
        std::stringstream ss(line);
        string tok;
        vector<string> toks;
        while(std::getline(ss, tok, ',')) { trim_inplace(tok); toks.push_back(tok); }
        if(toks.size() < 4) continue;
        Node n;
        // assume CSV columns: node_id, x, y, z
        n.id = std::stoi(toks[0]);
        n.x  = std::stod(toks[1]);
        n.y  = std::stod(toks[2]);
        n.z  = std::stod(toks[3]);
        nodes.push_back(n);
    }
    return nodes;
}

vector<Elem> read_elems_csv(const string &path){
    std::ifstream ifs(path);
    if(!ifs) throw std::runtime_error("Cannot open elements file: "+path);
    string line;
    vector<Elem> el;
    std::getline(ifs, line);
    while(std::getline(ifs,line)){
        trim_inplace(line);
        if(line.empty()) continue;
        std::stringstream ss(line);
        string tok;
        vector<string> toks;
        while(std::getline(ss, tok, ',')) { trim_inplace(tok); toks.push_back(tok); }
        if(toks.size() < 3) continue;
        Elem e;
        e.id = std::stoi(toks[0]);
        e.n1 = std::stoi(toks[1]); // temporarily store raw node id
        e.n2 = std::stoi(toks[2]);
        e.active = true;
        el.push_back(e);
    }
    return el;
}

// -----------------------
// Element stiffness (6x6) for axial + simple bending term
// Ke returned as flat 36-array in row-major.
static void element_stiffness(const double p1[3], const double p2[3], double Ke_out[36], double &Lout){
    double vx = p2[0]-p1[0], vy = p2[1]-p1[1], vz = p2[2]-p1[2];
    double L = std::sqrt(vx*vx + vy*vy + vz*vz);
    if(L < 1e-12) L = 1e-12;
    Lout = L;
    double nx = vx / L, ny = vy / L, nz = vz / L;

    // T = n (3x1), TT = T*T^T (3x3)
    double TT[9];
    TT[0] = nx*nx; TT[1] = nx*ny; TT[2] = nx*nz;
    TT[3] = ny*nx; TT[4] = ny*ny; TT[5] = ny*nz;
    TT[6] = nz*nx; TT[7] = nz*ny; TT[8] = nz*nz;

    double k_axial = (E_mod * A_area) / L;
    // perp = I - TT
    double perp[9];
    perp[0] = 1.0 - TT[0]; perp[1] = -TT[1];     perp[2] = -TT[2];
    perp[3] = -TT[3];     perp[4] = 1.0 - TT[4]; perp[5] = -TT[5];
    perp[6] = -TT[6];     perp[7] = -TT[7];     perp[8] = 1.0 - TT[8];

    double k_bend = 12.0 * E_mod * I_moment / (L*L*L);

    // Initialize Ke to zero
    for(int i=0;i<36;i++) Ke_out[i]=0.0;

    // Fill blocks. BLOCK index mapping: local dof 0..5 -> node1(0..2), node2(3..5)
    for(int r=0;r<3;r++){
        for(int c=0;c<3;c++){
            double val = k_axial * TT[r*3 + c] + k_bend * perp[r*3 + c];
            Ke_out[r*6 + c] = val;             // K(0..2,0..2)
            Ke_out[r*6 + (c+3)] = -val;        // K(0..2,3..5)
            Ke_out[(r+3)*6 + c] = -val;        // K(3..5,0..2)
            Ke_out[(r+3)*6 + (c+3)] = val;     // K(3..5,3..5)
        }
    }
}

// -----------------------
// Main FEA routine
int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
    CHKERRQ(ierr);

    if(argc < 2){
        PetscPrintf(PETSC_COMM_WORLD, "Usage: %s <results_dir>\n", argv[0]);
        PetscFinalize();
        return 1;
    }

    string results_dir = argv[1];
    auto make_dir_if_not_exist = [](const std::string &path){
        struct stat info;
        if(stat(path.c_str(), &info) != 0){
#ifdef _WIN32
            _mkdir(path.c_str());
#else
            mkdir(path.c_str(), 0777);
#endif
        }
    };
    std::string fea_dir = results_dir + "/fea_results";
    make_dir_if_not_exist(fea_dir);

    vector<Node> nodes;
    vector<Elem> elems;
    int n_nodes = 0, n_elems = 0, n_dof = 0;

    // Prepare output containers
    vector< vector<double> > stress_record; stress_record.reserve(N_STEPS);
    vector< vector<int> > active_record; active_record.reserve(N_STEPS);
    vector< vector<double> > disp_record; disp_record.reserve(N_STEPS);
    vector< std::pair<double,double> > fd_curve; fd_curve.reserve(N_STEPS);

    try {
        // --- Read nodes & elements
        nodes = read_nodes_csv(results_dir + "/nodes.csv");
        elems = read_elems_csv(results_dir + "/elements.csv");
        n_nodes = (int)nodes.size();
        n_elems = (int)elems.size();
        n_dof = 3 * n_nodes;
        PetscPrintf(PETSC_COMM_WORLD, "Read %d nodes, %d elements\n", n_nodes, n_elems);

        // Build id->index map (CSV node ids may not be 0..N-1)
        std::unordered_map<int,int> id2idx;
        for(int i=0;i<n_nodes;i++){
            id2idx[nodes[i].id] = i;
        }

        // Convert element node ids to indices (in-place)
        for(auto &e : elems){
            auto it1 = id2idx.find(e.n1);
            auto it2 = id2idx.find(e.n2);
            if(it1==id2idx.end() || it2==id2idx.end()){
                throw std::runtime_error("Element references unknown node id");
            }
            e.n1 = it1->second; // now store index 0..n_nodes-1
            e.n2 = it2->second;
        }

        // locate top & bottom node INDICES (use node indices)
        double y_min = 1e300, y_max = -1e300;
        for(const auto &n : nodes){
            if(n.y < y_min) y_min = n.y;
            if(n.y > y_max) y_max = n.y;
        }
        vector<int> top_nodes_idx, bot_nodes_idx;
        for(int i=0;i<n_nodes;i++){
            const auto &n = nodes[i];
            if(std::abs(n.y - y_max) < GRIP_LENGTH) top_nodes_idx.push_back(i); // index
            if(std::abs(n.y - y_min) < GRIP_LENGTH) bot_nodes_idx.push_back(i); // index
        }
        PetscPrintf(PETSC_COMM_WORLD, "Top nodes: %d, Bottom nodes: %d\n", (int)top_nodes_idx.size(), (int)bot_nodes_idx.size());

        // --- Time-stepping loop ---
        for(int step=0; step < N_STEPS; ++step){
            double disp_factor = double(step) / double(N_STEPS - 1);
            double dy_top = +DISPLACEMENT_MAX * disp_factor;
            double dy_bot = -DISPLACEMENT_MAX * disp_factor;
            PetscPrintf(PETSC_COMM_WORLD, "Step %d/%d dy_top=%.6f dy_bot=%.6f\n", step+1, N_STEPS, dy_top, dy_bot);

            // PETSc objects per step
            Mat K_phys = nullptr;   // physical stiffness assembled from active elements
            Mat K_mod = nullptr;    // modified stiffness with BCs applied (for solver)
            Vec F = nullptr, U = nullptr, b = nullptr, F_react = nullptr;
            KSP ksp = nullptr;

            try {
                // --- Assemble physical stiffness K_phys (only active elements) ---
                ierr = MatCreate(PETSC_COMM_WORLD, &K_phys); CHKERRQ(ierr);
                ierr = MatSetSizes(K_phys, PETSC_DECIDE, PETSC_DECIDE, n_dof, n_dof); CHKERRQ(ierr);
                ierr = MatSetFromOptions(K_phys); CHKERRQ(ierr);
                ierr = MatSetUp(K_phys); CHKERRQ(ierr);

                // RHS vector F (all zeros)
                ierr = VecCreate(PETSC_COMM_WORLD, &F); CHKERRQ(ierr);
                ierr = VecSetSizes(F, PETSC_DECIDE, n_dof); CHKERRQ(ierr);
                ierr = VecSetFromOptions(F); CHKERRQ(ierr);
                ierr = VecSet(F, 0.0); CHKERRQ(ierr);

                // Assemble element contributions into K_phys
                for(const auto &e : elems){
                    if(!e.active) continue; // skip inactive
                    int n1 = e.n1; // index
                    int n2 = e.n2;
                    double p1[3] = { nodes[n1].x, nodes[n1].y, nodes[n1].z };
                    double p2[3] = { nodes[n2].x, nodes[n2].y, nodes[n2].z };
                    double Ke[36]; double L;
                    element_stiffness(p1,p2,Ke,L);

                    PetscInt dof[6];
                    for(int k=0;k<3;k++){ dof[k] = 3*n1 + k; dof[k+3] = 3*n2 + k; }

                    for(int i=0;i<6;i++){
                        for(int j=0;j<6;j++){
                            double val = Ke[i*6 + j];
                            if(std::abs(val)<1e-18) continue;
                            ierr = MatSetValue(K_phys, dof[i], dof[j], (PetscScalar)val, ADD_VALUES); CHKERRQ(ierr);
                        }
                    }
                }

                ierr = MatAssemblyBegin(K_phys, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
                ierr = MatAssemblyEnd(K_phys, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

                // Duplicate physical K to K_mod (we'll apply BCs to K_mod and use it for the solver)
                ierr = MatDuplicate(K_phys, MAT_COPY_VALUES, &K_mod); CHKERRQ(ierr);

                // Dirichlet BCs (collect indices as PetscInt)
                std::vector<PetscInt> known_dofs_idx;
                std::vector<PetscScalar> known_vals;

                known_dofs_idx.reserve((top_nodes_idx.size()+bot_nodes_idx.size())*3);
                known_vals.reserve((top_nodes_idx.size()+bot_nodes_idx.size())*3);

                for(int idx : top_nodes_idx){
                    known_dofs_idx.push_back((PetscInt)(3*idx + 0)); known_vals.push_back((PetscScalar)0.0);
                    known_dofs_idx.push_back((PetscInt)(3*idx + 1)); known_vals.push_back((PetscScalar)dy_top);
                    known_dofs_idx.push_back((PetscInt)(3*idx + 2)); known_vals.push_back((PetscScalar)0.0);
                }
                for(int idx : bot_nodes_idx){
                    known_dofs_idx.push_back((PetscInt)(3*idx + 0)); known_vals.push_back((PetscScalar)0.0);
                    known_dofs_idx.push_back((PetscInt)(3*idx + 1)); known_vals.push_back((PetscScalar)dy_bot);
                    known_dofs_idx.push_back((PetscInt)(3*idx + 2)); known_vals.push_back((PetscScalar)0.0);
                }

                // Create b (copy of F) and set prescribed entries
                ierr = VecDuplicate(F, &b); CHKERRQ(ierr);
                ierr = VecCopy(F, b); CHKERRQ(ierr);

                if(!known_dofs_idx.empty()){
                    ierr = VecSetValues(b, (PetscInt)known_dofs_idx.size(), known_dofs_idx.data(), known_vals.data(), INSERT_VALUES); CHKERRQ(ierr);
                }
                ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
                ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

                // Apply BCs to K_mod (not K_phys)
                // 1) build x_known vector with prescribed displacements
                Vec x_known = nullptr, tmp = nullptr;
                ierr = VecDuplicate(F, &x_known); CHKERRQ(ierr);
                ierr = VecSet(x_known, 0.0); CHKERRQ(ierr);
                if(!known_dofs_idx.empty()){
                    ierr = VecSetValues(x_known, (PetscInt)known_dofs_idx.size(), known_dofs_idx.data(), known_vals.data(), INSERT_VALUES); CHKERRQ(ierr);
                }
                ierr = VecAssemblyBegin(x_known); CHKERRQ(ierr);
                ierr = VecAssemblyEnd(x_known); CHKERRQ(ierr);

                // 2) tmp = K_phys * x_known  (tmp_i = sum_j K_ij * x_known_j)
                ierr = VecDuplicate(F, &tmp); CHKERRQ(ierr);
                ierr = MatMult(K_phys, x_known, tmp); CHKERRQ(ierr);

                // 3) b = -tmp  (make RHS for free DOFs equal to -K_fk * known_vals)
                ierr = VecCopy(tmp, b); CHKERRQ(ierr);
                ierr = VecScale(b, -1.0); CHKERRQ(ierr);

                // 4) overwrite known rows of b with the actual prescribed values
                if(!known_dofs_idx.empty()){
                    ierr = VecSetValues(b, (PetscInt)known_dofs_idx.size(), known_dofs_idx.data(), known_vals.data(), INSERT_VALUES); CHKERRQ(ierr);
                }
                ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
                ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

                // 5) Now zero rows/cols in K_mod and solve
                if(!known_dofs_idx.empty()){
                    ierr = MatZeroRowsColumns(K_mod, (PetscInt)known_dofs_idx.size(), known_dofs_idx.data(), 1.0, NULL, b); CHKERRQ(ierr);
                }

                ierr = MatAssemblyBegin(K_mod, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
                ierr = MatAssemblyEnd(K_mod, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
                
                // --- BEGIN: ensure non-zero diagonals on K_mod to avoid ILU failure ---
                // (Place this AFTER MatAssemblyEnd(K_mod, MAT_FINAL_ASSEMBLY) and BEFORE KSPCreate)
                {
                    // get diagonal into a vector
                    Vec diag;
                    ierr = VecCreate(PETSC_COMM_WORLD, &diag); CHKERRQ(ierr);
                    ierr = VecSetSizes(diag, PETSC_DECIDE, n_dof); CHKERRQ(ierr);
                    ierr = VecSetFromOptions(diag); CHKERRQ(ierr);
                    ierr = MatGetDiagonal(K_mod, diag); CHKERRQ(ierr);

                    // fetch values
                    std::vector<PetscInt> idx(n_dof);
                    std::vector<PetscScalar> dvals(n_dof);
                    for(int ii=0; ii<n_dof; ++ii) idx[ii] = ii;
                    ierr = VecGetValues(diag, n_dof, idx.data(), dvals.data()); CHKERRQ(ierr);

                    const double diag_eps = 1e-14;   // treat anything smaller as zero
                    const PetscScalar fill_diag = 1e-8; // small diagonal to insert (adjust if needed)

                    bool any_zero = false;
                    for(int ii=0; ii<n_dof; ++ii){
                        if(std::abs((double)dvals[ii]) < diag_eps){
                            // Insert a small diagonal value. Use INSERT_VALUES to overwrite (row might not exist)
                            ierr = MatSetValue(K_mod, ii, ii, fill_diag, INSERT_VALUES); CHKERRQ(ierr);
                            any_zero = true;
                        }
                    }

                    if(any_zero){
                        ierr = MatAssemblyBegin(K_mod, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
                        ierr = MatAssemblyEnd(K_mod, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
                    }

                    if(diag){ VecDestroy(&diag); diag = nullptr; }
                }
                // --- END: ensure non-zero diagonals ---

                ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
                ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

                // Solution vector
                ierr = VecCreate(PETSC_COMM_WORLD, &U); CHKERRQ(ierr);
                ierr = VecSetSizes(U, PETSC_DECIDE, n_dof); CHKERRQ(ierr);
                ierr = VecSetFromOptions(U); CHKERRQ(ierr);
                ierr = VecSet(U, 0.0); CHKERRQ(ierr);

                // Solve K_mod U = b
                ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
                ierr = KSPSetOperators(ksp, K_mod, K_mod); CHKERRQ(ierr);
                ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
                ierr = KSPSetUp(ksp); CHKERRQ(ierr);

                ierr = KSPSolve(ksp, b, U); CHKERRQ(ierr);

                // Compute reactions using physical stiffness: F_react = K_phys * U
                ierr = VecDuplicate(b, &F_react); CHKERRQ(ierr);
                ierr = MatMult(K_phys, U, F_react); CHKERRQ(ierr);

                // Gather U locally in a batch
                std::vector<PetscInt> all_idx(n_dof);
                for(int ii=0; ii<n_dof; ++ii) all_idx[ii] = ii;
                std::vector<PetscScalar> all_vals(n_dof);
                ierr = VecGetValues(U, n_dof, all_idx.data(), all_vals.data()); CHKERRQ(ierr);
                std::vector<double> Uvals(n_dof);
                for(int ii=0; ii<n_dof; ++ii) Uvals[ii] = (double)all_vals[ii];

                // Gather reactions for top nodes (batch)
                std::vector<PetscInt> top_reac_idx;
                top_reac_idx.reserve(top_nodes_idx.size());
                for(int idx : top_nodes_idx) top_reac_idx.push_back((PetscInt)(3*idx + 1));
                std::vector<PetscScalar> top_reac_vals(top_reac_idx.size());
                if(!top_reac_idx.empty()){
                    ierr = VecGetValues(F_react, (PetscInt)top_reac_idx.size(), top_reac_idx.data(), top_reac_vals.data()); CHKERRQ(ierr);
                }
                double total_force = 0.0;
                for(size_t ii=0; ii<top_reac_vals.size(); ++ii) total_force += (double)top_reac_vals[ii];

                double total_disp = dy_top - dy_bot;
                fd_curve.emplace_back(total_disp, total_force);

                // Compute stress & deactivate elements (we will modify elems[].active directly)
                vector<double> stress(n_elems, 0.0);
                vector<int> active_flags(n_elems, 0);
                for(int i=0;i<n_elems;i++){
                    if(!elems[i].active){ stress[i] = 0.0; active_flags[i] = 0; continue; }
                    int n1 = elems[i].n1, n2 = elems[i].n2; // indices
                    double p1[3] = { nodes[n1].x, nodes[n1].y, nodes[n1].z };
                    double p2[3] = { nodes[n2].x, nodes[n2].y, nodes[n2].z };
                    double vx = p2[0]-p1[0], vy = p2[1]-p1[1], vz = p2[2]-p1[2];
                    double L = std::sqrt(vx*vx + vy*vy + vz*vz);
                    if(L<1e-12) L=1e-12;
                    double nx=vx/L, ny=vy/L, nz=vz/L;
                    double u1[3]={Uvals[3*n1+0],Uvals[3*n1+1],Uvals[3*n1+2]};
                    double u2[3]={Uvals[3*n2+0],Uvals[3*n2+1],Uvals[3*n2+2]};
                    double du[3]={u2[0]-u1[0], u2[1]-u1[1], u2[2]-u1[2]};
                    double axial_disp = nx*du[0]+ny*du[1]+nz*du[2];
                    double strain = axial_disp/L;
                    double stress_ax = E_mod*strain;
                    stress[i] = stress_ax;
                    if(std::abs(strain) > MAX_STRAIN){
                        // deactivate element for subsequent steps
                        elems[i].active = false;
                        active_flags[i] = 0;
                    } else active_flags[i] = 1;
                }

                stress_record.emplace_back(stress);
                active_record.emplace_back(active_flags);
                disp_record.emplace_back(Uvals);

                // Early stop if all elements failed
                int n_active=0; for(const auto &e: elems) if(e.active) ++n_active;
                if(n_active==0){
                    PetscPrintf(PETSC_COMM_WORLD, "All elements failed at step %d. Stopping early.\n", step+1);
                    // cleanup of PETSc objects will happen below; break after cleanup stage
                }

                // cleanup temporary vectors
                if(x_known){ VecDestroy(&x_known); x_known = nullptr; }
                if(tmp){ VecDestroy(&tmp); tmp = nullptr; }

            } catch(const std::exception &e){
                PetscPrintf(PETSC_COMM_WORLD, "Step %d error: %s\n", step+1, e.what());
                // continue to cleanup
            }

            // --- Cleanup PETSc objects for this step ---
            if(ksp) { KSPDestroy(&ksp); ksp = nullptr; }
            if(F_react) { VecDestroy(&F_react); F_react = nullptr; }
            if(U) { VecDestroy(&U); U = nullptr; }
            if(b) { VecDestroy(&b); b = nullptr; }
            if(F) { VecDestroy(&F); F = nullptr; }
            if(K_mod) { MatDestroy(&K_mod); K_mod = nullptr; }
            if(K_phys) { MatDestroy(&K_phys); K_phys = nullptr; }

            // Check for early stop (if no active elements remain)
            int n_active_now = 0; for(const auto &e: elems) if(e.active) ++n_active_now;
            if(n_active_now == 0){
                // we've already recorded final state; break the time loop
                break;
            }
        }

    } catch(const std::exception &e){
        PetscPrintf(PETSC_COMM_WORLD, "Fatal error: %s\n", e.what());
        // proceed to save partial results
    }

    // -----------------------
    // Save CSV results (same as before)
    {
        // stress_record
        std::string filename = fea_dir + "/stress_record.csv";
        std::ofstream ofs(filename);
        if(ofs){
            // header
            for(int j=0;j<n_elems;j++){
                ofs << "elem_" << j;
                if(j < n_elems-1) ofs << ",";
            }
            ofs << ",step\n";
            for(size_t i=0;i<stress_record.size();i++){
                for(int j=0;j<n_elems;j++){
                    ofs << std::setprecision(12) << stress_record[i][j];
                    if(j < n_elems-1) ofs << ",";
                }
                ofs << "," << (i+1) << "\n";
            }
        }

        // active_record
        filename = fea_dir + "/active_elements.csv";
        std::ofstream ofs2(filename);
        if(ofs2){
            for(int j=0;j<n_elems;j++){
                ofs2 << "elem_" << j;
                if(j < n_elems-1) ofs2 << ",";
            }
            ofs2 << ",step\n";
            for(size_t i=0;i<active_record.size();i++){
                for(int j=0;j<n_elems;j++){
                    ofs2 << active_record[i][j];
                    if(j < n_elems-1) ofs2 << ",";
                }
                ofs2 << "," << (i+1) << "\n";
            }
        }

        // disp_record
        filename = fea_dir + "/node_displacements.csv";
        std::ofstream ofs3(filename);
        if(ofs3){
            // headers for node_x, node_y, node_z
            for(int n=0;n<n_nodes;n++){
                ofs3 << "node_" << n << "_x";
                if(n < n_nodes-1) ofs3 << ",";
            }
            ofs3 << ",";
            for(int n=0;n<n_nodes;n++){
                ofs3 << "node_" << n << "_y";
                if(n < n_nodes-1) ofs3 << ",";
            }
            ofs3 << ",";
            for(int n=0;n<n_nodes;n++){
                ofs3 << "node_" << n << "_z";
                if(n < n_nodes-1) ofs3 << ",";
            }
            ofs3 << "step\n";

            for(size_t i=0;i<disp_record.size();i++){
                const auto &Uvals = disp_record[i];
                // x components
                for(int n=0;n<n_nodes;n++){
                    ofs3 << std::setprecision(12) << Uvals[3*n+0];
                    if(n < n_nodes-1) ofs3 << ",";
                }
                ofs3 << ",";
                // y components
                for(int n=0;n<n_nodes;n++){
                    ofs3 << std::setprecision(12) << Uvals[3*n+1];
                    if(n < n_nodes-1) ofs3 << ",";
                }
                ofs3 << ",";
                // z components
                for(int n=0;n<n_nodes;n++){
                    ofs3 << std::setprecision(12) << Uvals[3*n+2];
                    if(n < n_nodes-1) ofs3 << ",";
                }
                ofs3 << "," << (i+1) << "\n";
            }
        }

        // fd_curve
        filename = fea_dir + "/force_displacement.csv";
        std::ofstream ofs4(filename);
        if(ofs4){
            ofs4 << "total_displacement,total_force\n";
            for(auto &p: fd_curve){
                ofs4 << std::setprecision(12) << p.first << "," << p.second << "\n";
            }
        }
    }

    PetscPrintf(PETSC_COMM_WORLD, "FEA complete. %zu steps saved.\n", stress_record.size());

    ierr = PetscFinalize();
    if(ierr) PetscPrintf(PETSC_COMM_WORLD,"Error in PetscFinalize: %d\n",ierr);

    return 0;
}
