// fea_petsc.cpp
// PETSc-based FEA of rod/beam network (3 DOF per node).
// Reads nodes.csv and elements.csv (same CSV format produced by your exporter).
// Compile on Great Lakes after loading petsc + mpi: see instructions below.

#include <petscksp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <sys/stat.h>   // for mkdir
#include <iomanip>

using std::string;
using std::vector;

// -----------------------
// Parameters (matching Python)
static const double E_mod = 2500.0;        // MPa
static const double d = 0.0002;            // mm
static const double t_shell = 0.000001;    // mm
static const double A_area = 3.14159265358979323846 * ((d/2.0)*(d/2.0) - (d/2.0 - t_shell)*(d/2.0 - t_shell));
static const double I_moment = A_area * 0.001; // mm^4 (approx)
static const int N_STEPS = 100;
static const double DISPLACEMENT_MAX = 0.06; // mm
static const double MAX_STRAIN = 0.018;
static const double MAX_STRESS = E_mod * MAX_STRAIN;
static const double GRIP_LENGTH = 0.1; // mm

// -----------------------
// Data containers
struct Node {
    int id;
    double x,y,z;
};

struct Elem {
    int id;
    int n1, n2;
    bool active;
};

// -----------------------
// CSV helpers
static void trim_inplace(std::string &s){
    // trim whitespace
    const char *ws = " \t\n\r";
    auto p0 = s.find_first_not_of(ws);
    if (p0==string::npos){ s.clear(); return; }
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
        e.n1 = std::stoi(toks[1]);
        e.n2 = std::stoi(toks[2]);
        e.active = true;
        el.push_back(e);
    }
    return el;
}

// -----------------------
// Element stiffness (6x6) for axial + simple bending term
// This mirrors the Python logic: axial TT scaled by EA/L, plus a perpendicular projector scaled by 12 E I / L^3
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
    // upper-left (0:3,0:3) == + (k_axial * TT + k_bend * perp)
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
    if(ierr){
        PetscPrintf(PETSC_COMM_WORLD, "Error in PetscInitialize: %d\n", ierr);
        return 1;
    }

    if(argc < 2){
        PetscPrintf(PETSC_COMM_WORLD, "Usage: %s <results_dir>\n", argv[0]);
        PetscFinalize();
        return 1;
    }

    string results_dir = argv[1];
    auto make_dir_if_not_exist = [](const std::string &path){
        struct stat info;
        if(stat(path.c_str(), &info) != 0){
            mkdir(path.c_str(), 0777);
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

        // locate top & bottom nodes
        double y_min = 1e300, y_max = -1e300;
        for(const auto &n : nodes){
            if(n.y < y_min) y_min = n.y;
            if(n.y > y_max) y_max = n.y;
        }
        vector<int> top_nodes, bot_nodes;
        for(const auto &n : nodes){
            if(std::abs(n.y - y_max) < GRIP_LENGTH) top_nodes.push_back(n.id);
            if(std::abs(n.y - y_min) < GRIP_LENGTH) bot_nodes.push_back(n.id);
        }
        PetscPrintf(PETSC_COMM_WORLD, "Top nodes: %d, Bottom nodes: %d\n", (int)top_nodes.size(), (int)bot_nodes.size());

        // --- Time-stepping loop ---
        for(int step=0; step < N_STEPS; ++step){
            double disp_factor = double(step) / double(N_STEPS - 1);
            double dy_top = +DISPLACEMENT_MAX * disp_factor;
            double dy_bot = -DISPLACEMENT_MAX * disp_factor;
            PetscPrintf(PETSC_COMM_WORLD, "Step %d/%d dy_top=%.6f dy_bot=%.6f\n", step+1, N_STEPS, dy_top, dy_bot);

            // PETSc objects per step
            Mat K = nullptr;
            Vec F = nullptr, U = nullptr, b = nullptr, F_react = nullptr;
            KSP ksp = nullptr;

            // Wrap the step in try/catch for PETSc errors
            try {
                // Create matrix K
                ierr = MatCreate(PETSC_COMM_WORLD, &K); if(ierr) throw std::runtime_error("MatCreate failed");
                ierr = MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, n_dof, n_dof); if(ierr) throw std::runtime_error("MatSetSizes failed");
                ierr = MatSetFromOptions(K); if(ierr) throw std::runtime_error("MatSetFromOptions failed");
                ierr = MatSetUp(K); if(ierr) throw std::runtime_error("MatSetUp failed");

                // RHS vector F
                ierr = VecCreate(PETSC_COMM_WORLD, &F); if(ierr) throw std::runtime_error("VecCreate failed");
                ierr = VecSetSizes(F, PETSC_DECIDE, n_dof); if(ierr) throw std::runtime_error("VecSetSizes failed");
                ierr = VecSetFromOptions(F); if(ierr) throw std::runtime_error("VecSetFromOptions failed");
                ierr = VecSet(F, 0.0); if(ierr) throw std::runtime_error("VecSet failed");

                // Assemble element contributions
                for(const auto &e : elems){
                    if(!e.active) continue;
                    int n1 = e.n1, n2 = e.n2;
                    double p1[3] = { nodes[n1].x, nodes[n1].y, nodes[n1].z };
                    double p2[3] = { nodes[n2].x, nodes[n2].y, nodes[n2].z };
                    double Ke[36]; double L;
                    element_stiffness(p1,p2,Ke,L);

                    int dof[6];
                    for(int k=0;k<3;k++){ dof[k] = 3*n1 + k; dof[k+3] = 3*n2 + k; }

                    for(int i=0;i<6;i++){
                        for(int j=0;j<6;j++){
                            double val = Ke[i*6 + j];
                            if(std::abs(val)<1e-18) continue;
                            ierr = MatSetValue(K, dof[i], dof[j], val, ADD_VALUES);
                            if(ierr) throw std::runtime_error("MatSetValue failed");
                        }
                    }
                }

                ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY); if(ierr) throw std::runtime_error("MatAssemblyBegin failed");
                ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY); if(ierr) throw std::runtime_error("MatAssemblyEnd failed");

                // Dirichlet BCs
                std::vector<int> known_dofs_idx;
                std::vector<double> known_vals;
                for(int n : top_nodes){
                    known_dofs_idx.push_back(3*n + 0); known_vals.push_back(0.0);
                    known_dofs_idx.push_back(3*n + 1); known_vals.push_back(dy_top);
                    known_dofs_idx.push_back(3*n + 2); known_vals.push_back(0.0);
                }
                for(int n : bot_nodes){
                    known_dofs_idx.push_back(3*n + 0); known_vals.push_back(0.0);
                    known_dofs_idx.push_back(3*n + 1); known_vals.push_back(dy_bot);
                    known_dofs_idx.push_back(3*n + 2); known_vals.push_back(0.0);
                }

                ierr = VecDuplicate(F, &b); if(ierr) throw std::runtime_error("VecDuplicate failed");
                ierr = VecCopy(F, b); if(ierr) throw std::runtime_error("VecCopy failed");

                std::vector<PetscInt> rows(known_dofs_idx.begin(), known_dofs_idx.end());
                std::vector<PetscScalar> vals(known_vals.begin(), known_vals.end());
                for(size_t i=0;i<rows.size();++i){
                    ierr = VecSetValue(b, rows[i], vals[i], INSERT_VALUES);
                    if(ierr) throw std::runtime_error("VecSetValue failed");
                }
                ierr = VecAssemblyBegin(b); if(ierr) throw std::runtime_error("VecAssemblyBegin b failed");
                ierr = VecAssemblyEnd(b); if(ierr) throw std::runtime_error("VecAssemblyEnd b failed");

                ierr = MatZeroRowsColumns(K, rows.size(), rows.data(), 1.0, nullptr, b);
                if(ierr) throw std::runtime_error("MatZeroRowsColumns failed");

                ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY); if(ierr) throw std::runtime_error("MatAssemblyBegin after BC failed");
                ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY); if(ierr) throw std::runtime_error("MatAssemblyEnd after BC failed");

                ierr = VecAssemblyBegin(b); if(ierr) throw std::runtime_error("VecAssemblyBegin b after BC failed");
                ierr = VecAssemblyEnd(b); if(ierr) throw std::runtime_error("VecAssemblyEnd b after BC failed");

                // Solution vector
                ierr = VecCreate(PETSC_COMM_WORLD, &U); if(ierr) throw std::runtime_error("VecCreate U failed");
                ierr = VecSetSizes(U, PETSC_DECIDE, n_dof); if(ierr) throw std::runtime_error("VecSetSizes U failed");
                ierr = VecSetFromOptions(U); if(ierr) throw std::runtime_error("VecSetFromOptions U failed");
                ierr = VecSet(U, 0.0); if(ierr) throw std::runtime_error("VecSet U failed");

                // Solve K U = b
                ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); if(ierr) throw std::runtime_error("KSPCreate failed");
                ierr = KSPSetOperators(ksp, K, K); if(ierr) throw std::runtime_error("KSPSetOperators failed");
                ierr = KSPSetFromOptions(ksp); if(ierr) throw std::runtime_error("KSPSetFromOptions failed");
                ierr = KSPSetUp(ksp); if(ierr) throw std::runtime_error("KSPSetUp failed");

                ierr = KSPSolve(ksp, b, U); if(ierr) throw std::runtime_error("KSPSolve failed");

                // Reactions
                ierr = VecDuplicate(b, &F_react); if(ierr) throw std::runtime_error("VecDuplicate F_react failed");
                ierr = MatMult(K, U, F_react); if(ierr) throw std::runtime_error("MatMult F_react failed");

                // Gather U locally
                std::vector<double> Uvals(n_dof);
                for(int i=0;i<n_dof;i++){
                    double v;
                    ierr = VecGetValues(U, 1, &i, &v);
                    if(ierr) throw std::runtime_error("VecGetValues failed");
                    Uvals[i] = v;
                }

                // Total force
                double total_force = 0.0;
                for(int n : top_nodes){
                    int dof_y = 3*n + 1;
                    double fy;
                    ierr = VecGetValues(F_react, 1, &dof_y, &fy);
                    if(ierr) throw std::runtime_error("VecGetValues F_react failed");
                    total_force += fy;
                }
                double total_disp = dy_top - dy_bot;
                fd_curve.emplace_back(total_disp, total_force);

                // Compute stress & deactivate elements
                vector<double> stress(n_elems, 0.0);
                vector<int> active_flags(n_elems, 0);
                for(int i=0;i<n_elems;i++){
                    if(!elems[i].active){ stress[i] = 0.0; active_flags[i] = 0; continue; }
                    int n1 = elems[i].n1, n2 = elems[i].n2;
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
                    break;
                }

            } catch(const std::exception &e){
                PetscPrintf(PETSC_COMM_WORLD, "Step %d error: %s\n", step+1, e.what());
                // still continue to cleanup step
            }

            // --- Cleanup PETSc objects for this step ---
            if(ksp) KSPDestroy(&ksp);
            if(F_react) VecDestroy(&F_react);
            if(U) VecDestroy(&U);
            if(b) VecDestroy(&b);
            if(F) VecDestroy(&F);
            if(K) MatDestroy(&K);
        }

    } catch(const std::exception &e){
        PetscPrintf(PETSC_COMM_WORLD, "Fatal error: %s\n", e.what());
        // proceed to save
    }

    // -----------------------
    // Save CSV results
    {
        // stress_record
        std::string filename = fea_dir + "/stress_record.csv";
        std::ofstream ofs(filename);
        if(ofs){
            for(int j=0;j<n_elems;j++) ofs<<"elem_"<<j<<","; ofs<<"step\n";
            for(size_t i=0;i<stress_record.size();i++){
                for(int j=0;j<n_elems;j++) ofs<<std::setprecision(12)<<stress_record[i][j]<<",";
                ofs<<(i+1)<<"\n";
            }
        }

        // active_record
        filename = fea_dir + "/active_elements.csv";
        std::ofstream ofs2(filename);
        if(ofs2){
            for(int j=0;j<n_elems;j++) ofs2<<"elem_"<<j<<","; ofs2<<"step\n";
            for(size_t i=0;i<active_record.size();i++){
                for(int j=0;j<n_elems;j++) ofs2<<active_record[i][j]<<",";
                ofs2<<(i+1)<<"\n";
            }
        }

        // disp_record
        filename = fea_dir + "/node_displacements.csv";
        std::ofstream ofs3(filename);
        if(ofs3){
            for(int n=0;n<n_nodes;n++) ofs3<<"node_"<<n<<"_x,";
            for(int n=0;n<n_nodes;n++) ofs3<<"node_"<<n<<"_y,";
            for(int n=0;n<n_nodes;n++) ofs3<<"node_"<<n<<"_z,";
            ofs3<<"step\n";
            for(size_t i=0;i<disp_record.size();i++){
                const auto &Uvals = disp_record[i];
                for(int n=0;n<n_nodes;n++) ofs3<<std::setprecision(12)<<Uvals[3*n+0]<<",";
                for(int n=0;n<n_nodes;n++) ofs3<<std::setprecision(12)<<Uvals[3*n+1]<<",";
                for(int n=0;n<n_nodes;n++) ofs3<<std::setprecision(12)<<Uvals[3*n+2]<<",";
                ofs3<<(i+1)<<"\n";
            }
        }

        // fd_curve
        filename = fea_dir + "/force_displacement.csv";
        std::ofstream ofs4(filename);
        if(ofs4){
            ofs4<<"total_displacement,total_force\n";
            for(auto &p: fd_curve){
                ofs4<<std::setprecision(12)<<p.first<<","<<p.second<<"\n";
            }
        }
    }

    PetscPrintf(PETSC_COMM_WORLD, "FEA complete. %zu steps saved.\n", stress_record.size());

    ierr = PetscFinalize();
    if(ierr) PetscPrintf(PETSC_COMM_WORLD,"Error in PetscFinalize: %d\n",ierr);

    return 0;
}
