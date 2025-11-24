// main.cpp
// C++ version of mycelium_sim.py (core simulation).
// Produces snapshot CSVs per step and summary CSV at the end.
// Compile: g++ -O2 -std=c++17 main.cpp -o mycelium_sim
// Run: ./mycelium_sim
//
// Note: plotting is provided separately in plot_snapshots.py

#include <bits/stdc++.h>
#include <cstdlib> 
using namespace std;
using uint = unsigned int;
static constexpr double PI = 3.14159265358979323846;

// -------------------------
// Parameters (same defaults as Python)
static unsigned SEED = 42;
static double h0 = 0.05;                 // mm
static double dt = 0.01;                 // days
static double lambda_angle = PI/6.0;
static double P_branch = 0.5;
static double c_g = 1e-7;
static double Dcoef = 3.456;
static double M_cap = 2e-6;
static int initial_tips = 25;
static double Omega0 = 5e-6;
static int T_steps = 150;
static double ANASTOMOSIS_TOL = 1e-3;
static double WALL_THICKNESS = 0.05;
static double DISH_SIZE = 5.0;
static int H0_PER_POINT = 10;
static double SUBSTRATE_WIDTH = 5.0;
static double dist_inoculum = 0.5;

struct Vec3 {
    double x,y,z;
    Vec3(double xx=0,double yy=0,double zz=0):x(xx),y(yy),z(zz){}
    Vec3 operator+(const Vec3& o)const{return Vec3(x+o.x,y+o.y,z+o.z);}
    Vec3 operator-(const Vec3& o)const{return Vec3(x-o.x,y-o.y,z-o.z);}
    Vec3 operator*(double s)const{return Vec3(x*s,y*s,z*s);}
    Vec3 operator/(double s)const{return Vec3(x/s,y/s,z/s);}
    double norm()const{return sqrt(x*x+y*y+z*z);}
    void normalize(){double n=norm(); if(n>1e-15){x/=n;y/=n;z/=n;}}
};

double dot(const Vec3 &a,const Vec3 &b){ return a.x*b.x + a.y*b.y + a.z*b.z; }

// -------------------------
// Utility RNG
static std::mt19937_64 rng;
double uniform01(){ return std::uniform_real_distribution<double>(0.0,1.0)(rng); }
double uniformRange(double a,double b){ return a + (b-a)*uniform01(); }

// -------------------------
// geometry helpers
Vec3 sph_to_cart(double theta, double phi){
    // we keep z = 0 (theta = pi/2)
    return Vec3(cos(phi), sin(phi), 0.0);
}

pair<double,double> rand_direction_from(double theta, double phi, double lam = lambda_angle){
    double dph = (uniform01() - 0.5) * lam;
    double phi_new = phi + dph;
    double theta_new = PI/2.0;
    return {theta_new, phi_new};
}

double segment_length(const Vec3 &a, const Vec3 &b){
    return (b-a).norm();
}

pair<double,Vec3> point_segment_distance(const Vec3 &p, const Vec3 &a, const Vec3 &b){
    Vec3 ap = p - a;
    Vec3 ab = b - a;
    double ab2 = dot(ab,ab);
    if (ab2 < 1e-12) return {ap.norm(), a};
    double t = dot(ap,ab) / ab2;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    Vec3 proj = a + ab * t;
    return { (p - proj).norm(), proj };
}

// -------------------------
// Data structures
struct Segment {
    Vec3 start, end;
    double theta, phi;
    double I; // mol/mm
    char state; // 'A','P','S'
    int age;
    Segment(const Vec3 &s, const Vec3 &e, double th, double ph, double Ii=0.0, char st='A', int ag=0)
        : start(s), end(e), theta(th), phi(ph), I(Ii), state(st), age(ag) {}
    double length() const { return segment_length(start,end); }
    Vec3 endpoint() const { return end; }
};

struct Hypha {
    vector<Segment> segments;
    Hypha() {}
    Hypha(const vector<Segment>& segs):segments(segs){}
};

struct Cuboid {
    Vec3 center;
    Vec3 size; // Lx,Ly,Lz
    string ctype; // "substrate","impenetrable"
    double E; // amount for substrate
    double mu;
    Cuboid(const Vec3 &c,const Vec3 &s, const string &t="substrate", double E_=0.0, double mu_=1e8)
        : center(c), size(s), ctype(t), E(E_), mu(mu_) {}
    bool contains_point(const Vec3 &p) const {
        Vec3 half = size * 0.5;
        return (p.x >= (center.x - half.x - 1e-12) && p.x <= (center.x + half.x + 1e-12)
             && p.y >= (center.y - half.y - 1e-12) && p.y <= (center.y + half.y + 1e-12)
             && p.z >= (center.z - half.z - 1e-12) && p.z <= (center.z + half.z + 1e-12));
    }
};

// -------------------------
// Mycelium container
struct Mycelium {
    vector<Hypha> hyphae;
    void rebuild_index(){} // unused in C++ version (we iterate directly)
    double total_internal_substrate() const {
        double sum=0;
        for(const auto &h: hyphae)
            for(const auto &s: h.segments)
                sum += s.I * s.length();
        return sum;
    }
    double total_hyphal_length() const {
        double sum=0;
        for(const auto &h: hyphae)
            for(const auto &s: h.segments)
                sum += s.length();
        return sum;
    }
};

// -------------------------
// Initialization
vector<Vec3> generate_inoculum_grid(int nx,int ny,double dist,double z=0.0,bool centered=true){
    vector<Vec3> pts;
    double x0=0.0, y0=0.0;
    if (centered){
        x0 = -(nx - 1) * dist / 2.0;
        y0 = -(ny - 1) * dist / 2.0;
    }
    for(int i=0;i<nx;i++) for(int j=0;j<ny;j++){
        double x = x0 + i*dist;
        double y = y0 + j*dist;
        pts.emplace_back(x,y,z);
    }
    return pts;
}

// default inoculum grid 5x5
vector<Vec3> INOCULUM_POINTS = generate_inoculum_grid(5,5,dist_inoculum);

Mycelium initialize_inoculum(const vector<Vec3> &points, int H0_per_point, double Omega=Omega0){
    Mycelium M;
    int total_points = (int)points.size();
    double per_site_substrate = Omega / max(1, total_points);
    for(size_t site_idx=0; site_idx<points.size(); ++site_idx){
        double per_seg = per_site_substrate / double(H0_per_point);
        for(int i=0;i<H0_per_point;i++){
            double theta = uniform01() * PI;
            double phi = uniform01() * 2.0 * PI;
            Vec3 start = points[site_idx];
            Vec3 dir_v = sph_to_cart(theta, phi);
            Vec3 end = start + dir_v * h0;
            double Ival = per_seg / h0;
            Segment seg(start,end,theta,phi,Ival,'A',0);
            Hypha H;
            H.segments.push_back(seg);
            M.hyphae.push_back(std::move(H));
        }
    }
    return M;
}

// -------------------------
// Spatial hash for nearby queries
struct SpatialHash {
    double voxel_size;
    unordered_map<long long, vector<pair<int,int>>> hash; // key -> list of (hypha_idx, seg_idx)
    vector<pair<int,int>> keys_storage; // not needed
    SpatialHash(double v=0.2):voxel_size(v){}
    inline tuple<int,int,int> voxel_coords(const Vec3 &p) const {
        int ix = int(floor(p.x / voxel_size));
        int iy = int(floor(p.y / voxel_size));
        int iz = int(floor(p.z / voxel_size));
        return {ix,iy,iz};
    }
    static inline long long key_from_ijk(int ix,int iy,int iz){
        // pack into 64-bit key
        long long key = ((long long)(ix) & 0x1FFFFFLL) << 42 |
                        ((long long)(iy) & 0x1FFFFFLL) << 21 |
                        ((long long)(iz) & 0x1FFFFFLL);
        return key;
    }
    void insert_segment(int hi,int si,const Segment &seg){
        Vec3 p = (seg.start + seg.end) * 0.5;
        auto [ix,iy,iz] = voxel_coords(p);
        long long key = key_from_ijk(ix,iy,iz);
        hash[key].push_back({hi,si});
    }
    void rebuild(const Mycelium &M){
        hash.clear();
        for(int hi=0; hi<(int)M.hyphae.size(); ++hi){
            const Hypha &h = M.hyphae[hi];
            for(int si=0; si<(int)h.segments.size(); ++si){
                insert_segment(hi,si,h.segments[si]);
            }
        }
    }
    vector<pair<int,int>> nearby(const Vec3 &p) const {
        vector<pair<int,int>> out;
        auto [ix0,iy0,iz0] = voxel_coords(p);
        for(int dx=-1; dx<=1; ++dx) for(int dy=-1; dy<=1; ++dy) for(int dz=-1; dz<=1; ++dz){
            int ix = ix0+dx, iy = iy0+dy, iz = iz0+dz;
            long long key = key_from_ijk(ix,iy,iz);
            auto it = hash.find(key);
            if(it != hash.end()){
                for(const auto &pr : it->second) out.push_back(pr);
            }
        }
        return out;
    }
};

// -------------------------
// Processes: translocation, uptake, growth, anastomosis, boundaries

void translocate_internal_substrate(Mycelium &M, double D=Dcoef, double dt_local=dt){
    // iterate hyphae segments and exchange with predecessor
    struct Update { Segment *seg; double dI; };
    vector<Update> updates;
    for(auto &h : M.hyphae){
        for(size_t j=0; j<h.segments.size(); ++j){
            if (j==0) continue;
            Segment &s = h.segments[j];
            Segment &pred = h.segments[j-1];
            double denom = (s.length() + pred.length()) / 2.0;
            if (denom <= 0.0) continue;
            double delta = dt_local * D * (pred.I - s.I) / denom;
            // attempt clamping like Python
            double new_s = s.I + delta;
            double new_pred = pred.I - delta;
            double delta_adj = delta;
            if (new_s < 0) delta_adj = -s.I;
            else if (new_s > M_cap) delta_adj = M_cap - s.I;
            else if (new_pred < 0) delta_adj = pred.I;
            else if (new_pred > M_cap) delta_adj = M_cap - pred.I;
            updates.push_back({&s, delta_adj});
            updates.push_back({&pred, -delta_adj});
        }
    }
    for(const auto &u : updates){
        u.seg->I += u.dI;
        if (u.seg->I < 0.0) u.seg->I = 0.0;
        if (u.seg->I > M_cap) u.seg->I = M_cap;
    }
}

void uptake_from_cuboids(Mycelium &M, vector<Cuboid> &cuboids, double dt_local=dt){
    for(auto &c : cuboids){
        if (c.ctype != "substrate") continue;
        double mu = c.mu;
        double E = c.E;
        if (E <= 0.0) continue;
        // collect segments whose endpoint inside
        vector<Segment*> intersecting;
        for(auto &h : M.hyphae){
            for(auto &s : h.segments){
                Vec3 p = s.endpoint();
                if (c.contains_point(p)) intersecting.push_back(&s);
            }
        }
        for(auto *s : intersecting){
            double theta = dt_local * mu * E * s->I;
            double clamp = min(M_cap - s->I, E);
            if (theta < 0.0) theta = 0.0;
            if (theta > clamp) theta = clamp;
            s->I += theta;
            E -= theta;
            if (E <= 0.0) break;
        }
        c.E = E;
    }
}

void enforce_impenetrable_boundaries(Mycelium &M, vector<Cuboid> &cuboids, int max_iter=3){
    for(auto &h : M.hyphae){
        if (h.segments.empty()) continue;
        Segment &tip = h.segments.back();
        for(int iter=0; iter<max_iter; ++iter){
            bool penetrated = false;
            for(const auto &c : cuboids){
                if (c.ctype != "impenetrable") continue;
                if (c.contains_point(tip.endpoint())){
                    penetrated = true;
                    // compute normal by largest overlap
                    Vec3 delta = tip.end - c.center;
                    Vec3 half = c.size * 0.5;
                    double ox = fabs(delta.x) - half.x;
                    double oy = fabs(delta.y) - half.y;
                    double oz = fabs(delta.z) - half.z;
                    int idx = 0;
                    double om = ox;
                    if (oy > om) { om = oy; idx = 1; }
                    if (oz > om) { om = oz; idx = 2; }
                    Vec3 normal(0,0,0);
                    if (idx==0) normal.x = (delta.x >= 0 ? 1.0 : -1.0);
                    if (idx==1) normal.y = (delta.y >= 0 ? 1.0 : -1.0);
                    if (idx==2) normal.z = (delta.z >= 0 ? 1.0 : -1.0);
                    Vec3 dir = tip.end - tip.start;
                    if (dir.norm() < 1e-12){
                        dir = Vec3(uniformRange(-1,1), uniformRange(-1,1), uniformRange(-1,1));
                    }
                    dir.normalize();
                    // remove normal component
                    double comp = dot(dir, normal);
                    Vec3 dir2 = dir - normal * comp;
                    if (dir2.norm() < 1e-12){
                        dir2 = dir;
                        if (idx==0) dir2.x = 0.0;
                        else if (idx==1) dir2.y = 0.0;
                        else dir2.z = 0.0;
                        dir2.normalize();
                    } else dir2.normalize();
                    tip.end = tip.start + dir2 * tip.length();
                    tip.theta = acos(max(-1.0,min(1.0,dir2.z)));
                    tip.phi = atan2(dir2.y, dir2.x);
                    tip.state = 'A';
                    break;
                }
            }
            if (!penetrated) break;
        }
    }
}

void attempt_growth(Mycelium &M, double P_branch_local=P_branch, double c_g_local=c_g, double h0_local=h0){
    vector<Hypha> new_hyphae;
    for(auto &h : M.hyphae){
        if (h.segments.empty()) continue;
        Segment &tip = h.segments.back();
        if (tip.state != 'A') continue;
        double available_mol = tip.I * tip.length();
        double cost_one = c_g_local * h0_local;
        if (available_mol < cost_one) continue;
        bool do_branch = (uniform01() < P_branch_local) && (available_mol >= 2.0 * cost_one);
        if (do_branch){
            double total_cost = 2.0 * cost_one;
            tip.I = max(0.0, (available_mol - total_cost) / tip.length());
            tip.state = 'P';
            auto thph0 = rand_direction_from(tip.theta, tip.phi);
            double th0 = thph0.first, ph0 = thph0.second;
            Vec3 dir0 = sph_to_cart(th0, ph0);
            Vec3 end0 = tip.endpoint() + dir0 * h0_local;
            Segment new_parent(tip.endpoint(), end0, th0, ph0, 0.5*tip.I, 'A', 0);
            auto thph1 = rand_direction_from(tip.theta, tip.phi);
            double th1 = thph1.first, ph1 = thph1.second;
            Vec3 dir1 = sph_to_cart(th1, ph1);
            Vec3 end1 = tip.endpoint() + dir1 * h0_local;
            Segment new_child(tip.endpoint(), end1, th1, ph1, 0.5*tip.I, 'A', 0);
            h.segments.push_back(new_parent);
            Hypha nh;
            nh.segments.push_back(new_child);
            new_hyphae.push_back(std::move(nh));
        } else {
            // simple apical growth
            tip.state = 'P';
            tip.I = max(0.0, (available_mol - cost_one) / tip.length());
            auto thph = rand_direction_from(tip.theta, tip.phi);
            double th = thph.first, ph = thph.second;
            Vec3 dir = sph_to_cart(th, ph);
            Vec3 end = tip.endpoint() + dir * h0_local;
            Segment new_seg(tip.endpoint(), end, th, ph, 0.5*tip.I, 'A', 0);
            h.segments.push_back(new_seg);
        }
    }
    for(auto &nh : new_hyphae) M.hyphae.push_back(std::move(nh));
}

void detect_anastomosis(Mycelium &M, SpatialHash &spatial, double tol=ANASTOMOSIS_TOL){
    for(int hi=0; hi<(int)M.hyphae.size(); ++hi){
        Hypha &h = M.hyphae[hi];
        if (h.segments.empty()) continue;
        int tip_idx = (int)h.segments.size() - 1;
        Segment &tip = h.segments[tip_idx];
        if (tip.state != 'A') continue;
        Vec3 p = tip.endpoint();
        auto nearby = spatial.nearby(p);
        bool found = false;
        for(auto pr : nearby){
            int ii = pr.first, jj = pr.second;
            if (ii == hi && jj == tip_idx) continue;
            Segment &seg = M.hyphae[ii].segments[jj];
            auto [dist, proj] = point_segment_distance(p, seg.start, seg.end);
            if (dist <= tol){
                tip.end = proj;
                tip.state = 'S';
                found = true;
                break;
            }
        }
        if (found){
            spatial.insert_segment(hi, tip_idx, tip);
        }
    }
}

// -------------------------
// Statistics
struct Stats {
    int hyphae;
    int segments;
    int active_tips;
    int passive_tips;
    int anastomosed;
    int branches;
    double total_length_mm;
    int step;
};

Stats summarize_mycelium(const Mycelium &M, int n_inoculum_sites){
    Stats s{};
    s.hyphae = (int)M.hyphae.size();
    s.segments = 0;
    s.active_tips = s.passive_tips = s.anastomosed = 0;
    for(const auto &h : M.hyphae){
        for(const auto &seg : h.segments){
            s.segments++;
            if (seg.state == 'A') s.active_tips++;
            else if (seg.state == 'P') s.passive_tips++;
            else if (seg.state == 'S') s.anastomosed++;
        }
    }
    s.branches = max(0, s.hyphae - n_inoculum_sites);
    s.total_length_mm = M.total_hyphal_length();
    return s;
}

// -------------------------
// Export geometry & snapshots
string now_timestamp(){
    auto t = chrono::system_clock::now();
    auto tt = chrono::system_clock::to_time_t(t);
    tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &tt);
#else
    localtime_r(&tt, &tmv);
#endif
    char buf[64];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
    return string(buf);
}

void write_snapshot_csv(const Mycelium &M, const string &path){
    // write per-segment rows: x1,y1,x2,y2,intensity
    // intensity = s.I * s.length() normalized? We'll keep raw mol*mm value; Python script will normalize
    ofstream ofs(path);
    ofs << "x1,y1,x2,y2,intensity\n";
    for(const auto &h : M.hyphae){
        for(const auto &s : h.segments){
            double val = s.I * s.length();
            ofs << s.start.x << ',' << s.start.y << ',' << s.end.x << ',' << s.end.y << ',' << val << '\n';
        }
    }
    ofs.close();
}

void export_geometry(const Mycelium &M, const string &out_dir){
    // nodes.csv and elements.csv
    unordered_map<string,int> node_map;
    vector<array<double,4>> nodes; // id,x,y,z
    vector<array<int,3>> elems; // elem_id, n1,n2
    int node_counter = 0;
    auto add_node = [&](const Vec3 &p)->int{
        char buf[128]; snprintf(buf,sizeof(buf),"%.6f_%.6f_%.6f", p.x,p.y,p.z);
        string key(buf);
        auto it = node_map.find(key);
        if (it!=node_map.end()) return it->second;
        node_map[key] = node_counter;
        nodes.push_back({(double)node_counter, p.x, p.y, p.z});
        return node_counter++;
    };
    int elem_counter = 0;
    for(const auto &h : M.hyphae){
        for(const auto &s : h.segments){
            int n1 = add_node(s.start);
            int n2 = add_node(s.end);
            elems.push_back({elem_counter, n1, n2});
            elem_counter++;
        }
    }
    // write nodes.csv
    {
        string fname = out_dir + "/nodes.csv";
        ofstream ofs(fname);
        ofs << "node_id,x,y,z\n";
        for(auto &n : nodes) ofs << (int)n[0] << ',' << n[1] << ',' << n[2] << ',' << n[3] << '\n';
    }
    {
        string fname = out_dir + "/elements.csv";
        ofstream ofs(fname);
        ofs << "elem_id,n1,n2\n";
        for(auto &e : elems) ofs << e[0] << ',' << e[1] << ',' << e[2] << '\n';
    }
    cerr << "✅ Exported geometry to " << out_dir << "\n";
}

void create_directories(const std::string &path) {
#ifdef _WIN32
    std::string cmd = "mkdir \"" + path + "\"";
#else
    std::string cmd = "mkdir -p \"" + path + "\""; // -p creates intermediate dirs
#endif
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "⚠️ Failed to create directory: " << path << "\n";
    }
}

int main(int argc, char **argv){
    // optional seed override by env var or arg
    if (argc>=2) SEED = (unsigned)atoi(argv[1]);
    rng.seed(SEED);
    cerr << "Seed: " << SEED << "\n";

    // create results directories
    string out_dir_root = "../results";
    string timestamp = now_timestamp();
    string out_dir = out_dir_root + "/sim_" + timestamp;
    string snapshot_dir = out_dir + "/snapshots";
    create_directories(snapshot_dir);

    // initialize
    Mycelium M = initialize_inoculum(INOCULUM_POINTS, H0_PER_POINT, Omega0);

    vector<Cuboid> cuboids;
    cuboids.emplace_back(Vec3(0,0,0), Vec3(DISH_SIZE, SUBSTRATE_WIDTH, 0.1), "substrate", 2e-6, 1e8);
    // walls
    cuboids.emplace_back(Vec3(0, DISH_SIZE/2 + WALL_THICKNESS/2, 0), Vec3(DISH_SIZE, WALL_THICKNESS, WALL_THICKNESS), "impenetrable");
    cuboids.emplace_back(Vec3(0, -DISH_SIZE/2 - WALL_THICKNESS/2, 0), Vec3(DISH_SIZE, WALL_THICKNESS, WALL_THICKNESS), "impenetrable");
    cuboids.emplace_back(Vec3(DISH_SIZE/2 + WALL_THICKNESS/2, 0, 0), Vec3(WALL_THICKNESS, DISH_SIZE, WALL_THICKNESS), "impenetrable");
    cuboids.emplace_back(Vec3(-DISH_SIZE/2 - WALL_THICKNESS/2, 0, 0), Vec3(WALL_THICKNESS, DISH_SIZE, WALL_THICKNESS), "impenetrable");

    SpatialHash spatial(0.1);
    spatial.rebuild(M);

    // history CSV
    string history_csv = out_dir + "/mycelium_growth_stats.csv";
    ofstream hist_ofs(history_csv);
    hist_ofs << "step,hyphae,segments,active_tips,passive_tips,anastomosed,branches,total_length_mm\n";

    for(int t=0; t<T_steps; ++t){
        translocate_internal_substrate(M, Dcoef, dt);
        attempt_growth(M, P_branch, c_g, h0);
        spatial.rebuild(M);
        detect_anastomosis(M, spatial, ANASTOMOSIS_TOL);
        uptake_from_cuboids(M, cuboids, dt);
        enforce_impenetrable_boundaries(M, cuboids);

        Stats s = summarize_mycelium(M, (int)INOCULUM_POINTS.size());
        s.step = t;
        hist_ofs << s.step << ',' << s.hyphae << ',' << s.segments << ',' << s.active_tips << ',' << s.passive_tips << ',' << s.anastomosed << ',' << s.branches << ',' << s.total_length_mm << '\n';

        // write snapshot per step
        char fname[512];
        snprintf(fname,sizeof(fname), "%s/step_%04d.csv", snapshot_dir.c_str(), t);
        write_snapshot_csv(M, string(fname));

        if (t % 1 == 0 || t == T_steps - 1){
            cerr << "Step " << t << ": hyphae=" << s.hyphae << " segments=" << s.segments << " total_length=" << s.total_length_mm << "\n";
        }
    }
    hist_ofs.close();

    export_geometry(M, out_dir);

    cerr << "✅ All results saved under " << out_dir << "\n";
    return 0;
}
